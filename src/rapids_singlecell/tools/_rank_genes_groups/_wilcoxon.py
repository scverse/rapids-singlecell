from __future__ import annotations

import warnings
from collections import Counter
from contextlib import suppress
from copy import copy
from itertools import batched, chain
from operator import methodcaller
from typing import TYPE_CHECKING

import cupy as cp
import cupyx.scipy.sparse as cpsp
import cupyx.scipy.special as cupyx_special
import numpy as np
import scipy.sparse as sp

from rapids_singlecell._compat import DaskArray
from rapids_singlecell._cuda import _wilcoxon_cuda as _wc

from ._utils import EPS, MIN_GROUP_SIZE_WARNING, _canonicalize_sparse, _data_device
from ._wilcoxon_host import _build_ovo_host_context, _kernel_dtype, _OvoHostContext, _run_sharded_wilcoxon

if TYPE_CHECKING:
    from numpy.typing import NDArray

    from ._core import _RankGenes

# Default genes per column batch by (layout, OVR); rows x width sizes the sort
# and slab scratch, and sparse batches are additionally capped by nnz.
_SUB_BATCH = {
    ("dense", True): 64,
    ("dense", False): 128,
    ("csc_host", True): 512,
    ("csr_host", True): 2048,
    ("sparse_device", True): 2048,
    ("csc_host", False): 256,
    ("csr_host", False): 2048,
    ("sparse_device", False): 128,
}

_WilcoxonResult = tuple[np.ndarray, cp.ndarray, cp.ndarray, cp.ndarray | None]
_DaskWilcoxonResult = tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray | None]
_HALF_INTEGER_LIMIT = 2**52  # Above this magnitude float64 can lose half-ranks.
_MAX_RECORD_GROUPS = 2**16 - 1  # Reserve a uint16 code for unselected cells.


def _log1p_scale(rg: _RankGenes) -> float:
    """Natural-log factor of the log1p base: expm1(scale * x) inverts log1p."""
    return float(np.log(rg._log1p_base)) if rg._log1p_base is not None else 1.0


def _logfoldchanges_from_means(
    rg: _RankGenes,
    mean_group: cp.ndarray,
    mean_reference: cp.ndarray,
) -> cp.ndarray:
    if rg.mean_in_log_space:
        scale = cp.float64(_log1p_scale(rg))
        group_expr = cp.expm1(mean_group * scale)
        reference_expr = cp.expm1(mean_reference * scale)
    else:
        group_expr = mean_group
        reference_expr = mean_reference
    return cp.log2((group_expr + EPS) / (reference_expr + EPS))


def _wilcoxon_z_p(diff, variance, *, use_continuity):
    if use_continuity:
        diff = cp.sign(diff) * cp.maximum(cp.abs(diff) - 0.5, 0.0)
    z = diff / cp.sqrt(variance)
    cp.nan_to_num(z, copy=False)
    return z, cupyx_special.erfc(cp.abs(z) * cp.float64(cp.sqrt(0.5)))


def _finish_wilcoxon(
    rank_sums: cp.ndarray,
    expected: cp.ndarray,
    variance: cp.ndarray,
    sizes: cp.ndarray,
    group_indices: np.ndarray,
    *,
    use_continuity: bool,
    return_u_values: bool,
    logfoldchanges_gpu: cp.ndarray | None,
) -> _WilcoxonResult:
    """Compute the shared normal approximation and return its GPU result."""
    z, p_values = _wilcoxon_z_p(rank_sums - expected, variance, use_continuity=use_continuity)
    scores = rank_sums - sizes[:, None] * (sizes[:, None] + 1.0) / 2.0 if return_u_values else z
    return (
        group_indices,
        scores,
        p_values,
        logfoldchanges_gpu,
    )


def _validate_wilcoxon_sparse_dtype(X) -> None:
    if not (sp.issparse(X) or cpsp.issparse(X)):
        return
    if X.format not in {"csr", "csc"}:
        raise TypeError(
            f"Wilcoxon sparse input must be CSR or CSC; refusing hidden full-matrix conversion from {X.format!r}."
        )
    data_dtype = np.dtype(X.data.dtype)
    if data_dtype.kind == "c":
        msg = "Wilcoxon sparse input data dtype must be real; complex sparse data is not supported."
        raise TypeError(msg)
    if cpsp.issparse(X) and data_dtype not in {
        np.dtype(np.float32),
        np.dtype(np.float64),
    }:
        msg = f"Wilcoxon device sparse input data dtype must be float32 or float64; got {data_dtype}."
        raise TypeError(msg)
    if (
        sp.issparse(X)
        and data_dtype
        not in {
            np.dtype(np.float32),
            np.dtype(np.float64),
        }
        and data_dtype.kind not in {"b", "i", "u"}
    ):
        msg = f"Wilcoxon sparse input data dtype must be float32, float64, bool, or integer; got {data_dtype}."
        raise TypeError(msg)
    if getattr(X, "format", None) in {"csr", "csc"}:
        indices_dtype = np.dtype(X.indices.dtype)
        indptr_dtype = np.dtype(X.indptr.dtype)
        if indices_dtype != indptr_dtype:
            msg = (
                "Wilcoxon sparse indices and indptr must have the same dtype; "
                f"got indices={indices_dtype} and indptr={indptr_dtype}."
            )
            raise TypeError(msg)
        if indices_dtype not in {np.dtype(np.int32), np.dtype(np.int64)}:
            msg = f"Wilcoxon sparse indices and indptr must be int32 or int64; got {indices_dtype}."
            raise TypeError(msg)


def _device_dense(X: cp.ndarray, *, exact: bool = False) -> cp.ndarray:
    """Float32/64 C- or F-contiguous array the dense kernels read in place."""
    if X.dtype not in (cp.float32, cp.float64):
        X = X.astype(_kernel_dtype(X.dtype, exact=exact))
    return X if X.flags.c_contiguous or X.flags.f_contiguous else cp.ascontiguousarray(X)


def wilcoxon(
    rg: _RankGenes,
    *,
    tie_correct: bool,
    use_continuity: bool = False,
    multi_gpu: bool | list[int] | str | None = None,
    return_u_values: bool = False,
) -> _WilcoxonResult | _DaskWilcoxonResult | None:
    """Compute Wilcoxon rank-sum test statistics."""
    X = rg.X
    _validate_wilcoxon_sparse_dtype(X._meta if isinstance(X, DaskArray) else X)
    rg.means = rg.vars = rg.pts = None
    rg.means_rest = rg.vars_rest = rg.pts_rest = None
    if rg.ireference is None:
        _warn_small_ovr_groups(rg, rg.group_sizes, X.shape[0])
    else:
        _warn_small_ovo_groups(rg, rg.group_sizes)

    run = _run_dask_wilcoxon if isinstance(X, DaskArray) else _run_sharded_wilcoxon
    return run(
        rg,
        tie_correct=tie_correct,
        use_continuity=use_continuity,
        multi_gpu=multi_gpu,
        return_u_values=return_u_values,
        shard_runner=_run_wilcoxon_shard,
    )


def _compact_rank_groups(rg, targets):
    """Give large targets, the reference and the remainder distinct uint16 codes."""
    ref = rg.ireference
    selected = targets if ref is None else np.union1d(targets, [ref])
    codes = np.full(len(rg.group_sizes) + 1, len(selected), dtype=np.uint16)
    codes[selected] = np.arange(len(selected), dtype=np.uint16)
    subset = copy(rg)
    subset.group_codes = codes[rg.group_codes]
    subset.group_sizes = rg.group_sizes[selected]
    subset.ireference = None if ref is None else int(codes[ref])
    subset.comp_pts = False
    return subset, targets if ref is None else targets - (targets > ref)


def _rank_dask_batch(rows, template, fmt, options):
    from dask.array.dispatch import concatenate_lookup

    rg = copy(template)
    with _data_device(rows[0]):
        first = rows[0]
        exact = None
        positions = slice(None)
        if first.shape[1] and len(rg.group_codes) * int(rg.group_sizes.max()) > _HALF_INTEGER_LIMIT:
            from ._wilcoxon_dask import rank_large_groups

            if len(rg.group_sizes) <= _MAX_RECORD_GROUPS:
                exact = rank_large_groups(rows, rg, options)
                if exact is not None and first.dtype == np.float32:
                    return exact
            else:
                # Under the int32 cell limit at most 1,023 targets need integer rank sums.
                targets = np.flatnonzero(len(rg.group_codes) * rg.group_sizes > _HALF_INTEGER_LIMIT)
                if rg.ireference is not None:
                    targets = targets[targets != rg.ireference]
                if targets.size:
                    subset, positions = _compact_rank_groups(rg, targets)
                    exact = rank_large_groups(rows, subset, options)
        # Dask registers CuPy's spmatrix but not its newer sparse array classes.
        join = concatenate_lookup.dispatch(cpsp.spmatrix if cpsp.issparse(first) else type(first))
        X = first if len(rows) == 1 else join(rows, axis=0)
        if fmt:
            X = X.asformat(fmt)  # stacked CuPy CSC chunks can become COO
            _validate_wilcoxon_sparse_dtype(X)
            if not X.has_canonical_format:
                X = _canonicalize_sparse(X.copy())
        rg.X = X
        _, scores, pvals, lfc = _run_sharded_wilcoxon(rg, **options)
        # Keep expression summaries in the input's native precision.
        stats = {"scores": cp.asnumpy(scores), "pvals": cp.asnumpy(pvals)}
        if exact is not None:
            for field in stats:
                stats[field][positions] = exact[field]
        if lfc is not None:
            stats["logfoldchanges"] = cp.asnumpy(lfc)
        return stats | {k: v for k in ("means", "means_rest", "pts", "pts_rest") if (v := getattr(rg, k)) is not None}


def _rank_dask_window(window, width, template, options, *, client, workers, compute_kw):
    import dask

    # Detach projections: a narrow dense slice can keep its full-width parent alive.
    parts = window.rechunk({1: width}, method="tasks").map_blocks(methodcaller("copy"), meta=window._meta)
    fmt = getattr(window._meta, "format", None)
    panels = [[] for _ in range(parts.numblocks[1])]
    row_limit = max(1, min(len(workers), (parts.numblocks[0] + 1) // 2))
    for row_wave in batched(parts.to_delayed(), row_limit, strict=False):
        pending = list(chain.from_iterable(row_wave))
        if workers:
            from distributed import wait

            projected = client.compute(pending)
            wait(projected)  # Release this wave's full-width input before reading more rows.
        else:
            projected = dask.compute(*pending, **compute_kw)
        for i, panel in enumerate(panels):
            panel.extend(projected[i :: len(panels)])
    if workers:
        ranks = [
            client.submit(_rank_dask_batch, panel, template, fmt, options, workers=[worker], pure=False)
            for panel, worker in zip(panels, workers, strict=False)
        ]
        return client.gather(ranks)
    return [_rank_dask_batch(panels[0], template, fmt, options)]


def _worker_rank_memory(dask_worker):
    import psutil
    from distributed.system import MEMORY_LIMIT

    _, device = cp.cuda.runtime.memGetInfo()
    pool = next(
        (
            getattr(p, "maximum_pool_size", None)
            for p in dask_worker.plugins.values()
            if type(p).__name__ == "RMMSetup"
            and (getattr(p, "initial_pool_size", None) is not None or getattr(p, "async_alloc", False))
        ),
        None,
    )
    limit = dask_worker.memory_manager.memory_limit or MEMORY_LIMIT
    remaining = max(0, limit // 2 - dask_worker.monitor.get_process_memory())
    available = min(MEMORY_LIMIT, psutil.virtual_memory().available)
    return device, pool, dask_worker.host, min(limit // 4, remaining), available


def _rank_budgets(memory, input_bytes):
    import dask
    from dask.utils import parse_bytes

    hosts = Counter(host for _, _, host, _, _ in memory)
    limit = min(device // 4 for device, *_ in memory)
    # Leave room in a capped pool for reading, normalizing and sorting a CSR chunk.
    pool = min(pool or device for device, pool, *_ in memory)
    default = min(limit, 3 * pool // 10)
    target = min(limit, pool, parse_bytes(dask.config.get("rapids-singlecell.wilcoxon.batch-size", default=default)))
    host = min(min(worker, available // (4 * hosts[name])) for _, _, name, worker, available in memory)
    # Host transfers pay off when they avoid multiple full CSR scans.
    if host < (2 if default < limit else 4) * target or input_bytes <= target * len(memory):
        host = 0
    return target, host


def _run_dask_wilcoxon(rg, **options):
    import dask
    from dask import threaded
    from dask.base import get_scheduler
    from dask.utils import parse_bytes

    X = rg.X
    groups = np.arange(len(rg.groups_order), dtype=np.intp)
    if rg.ireference is not None:
        groups = np.delete(groups, rg.ireference)
    if not groups.size:
        return None
    scheduler = get_scheduler(collections=[X])
    client = getattr(scheduler, "__self__", None)
    workers = client.nthreads() if hasattr(client, "nthreads") else {}
    fmt = getattr(X._meta, "format", None)
    entry_bytes = X.itemsize + (X._meta.indices.itemsize if fmt else 0)
    target_bytes = parse_bytes(dask.config.get("array.chunk-size"))
    host_bytes = 0
    if workers and (isinstance(X._meta, cp.ndarray) or cpsp.issparse(X._meta)):
        memory = client.run(_worker_rank_memory, workers=list(workers))
        target_bytes, host_bytes = _rank_budgets(memory.values(), X.nbytes)
    row_blocks = (
        (isinstance(X._meta, cp.ndarray) or cpsp.isspmatrix_csr(X._meta)) and X.numblocks[0] > 1 and X.numblocks[1] == 1
    )
    fits_records = X.dtype == np.float32 and X.shape[0] < 2**31 and len(rg.group_sizes) <= _MAX_RECORD_GROUPS
    if workers and row_blocks and X.shape[1] > 0 and fits_records and options["multi_gpu"] in (None, False):
        from ._wilcoxon_dask import rank_record_windows

        host_config = dask.config.get("rapids-singlecell.wilcoxon.host-batch-size", default=None)
        if host_config is not None:
            host_bytes = parse_bytes(host_config)
        if host_bytes < 0:
            raise ValueError("Wilcoxon host-batch-size must be nonnegative.")
        # Concurrent full-width reads also draw from a small device pool.
        rank_workers = dict.fromkeys(workers, 1) if host_bytes and host_config is None else workers
        with suppress(MemoryError):
            result = rank_record_windows(rg, options, client, rank_workers, target_bytes, host_budget=host_bytes)
            if result is not None:
                return groups, *result
    width = max(1, target_bytes // (max(1, X.shape[0]) * entry_bytes))
    template = copy(rg)
    del template.X, template.labels, template.var_names
    if workers:
        template = client.scatter(template, broadcast=True, hash=False)
    if options["multi_gpu"] is None:
        options["multi_gpu"] = False
    compute_kw = {"num_workers": 1} if scheduler is threaded.get else {}
    stats = {}
    n_workers = max(1, len(workers))
    # Even a large Dask byte target should leave the next genes on disk.
    window_width = max(1, min(width * n_workers, (X.shape[1] + 1) // 2))
    width = (window_width + n_workers - 1) // n_workers
    for start in range(0, max(1, X.shape[1]), window_width):
        window = X[:, start : start + window_width]
        batches = _rank_dask_window(
            window, width, template, options, client=client, workers=workers, compute_kw=compute_kw
        )
        for i, batch in enumerate(batches):
            offset = start + i * width
            for name, values in batch.items():
                if name not in stats:
                    stats[name] = np.empty((values.shape[0], X.shape[1]), dtype=values.dtype)
                stats[name][:, offset : offset + values.shape[1]] = values
    rg.means, rg.means_rest = stats.get("means"), stats.get("means_rest")
    rg.pts, rg.pts_rest = stats.get("pts"), stats.get("pts_rest")
    return groups, stats["scores"], stats["pvals"], stats.get("logfoldchanges")


def _run_wilcoxon_shard(
    rg: _RankGenes,
    *,
    tie_correct: bool,
    use_continuity: bool,
    return_u_values: bool,
    column_range: tuple[int, int] | None = None,
    sparse_row_spans: tuple[np.ndarray, np.ndarray] | None = None,
    ovo_host_context: _OvoHostContext | None = None,
    dense_rows: np.ndarray | None = None,
) -> _WilcoxonResult | None:
    """Rank one gene shard; the kernels also return per-group sums / nnz."""
    X = rg.X
    n_cells = X.shape[0]
    col_start, col_stop = (0, X.shape[1]) if column_range is None else column_range
    n_cols = col_stop - col_start
    n_groups = len(rg.groups_order)
    ovr = rg.ireference is None
    sizes = cp.asarray(rg.group_sizes, dtype=cp.float64)
    # Kernels accumulate the stats atomically into zeroed outputs.
    group_sums = cp.zeros((n_groups, n_cols), dtype=cp.float64)
    group_nnz = cp.zeros((n_groups, n_cols) if rg.comp_pts else (1, 1), dtype=cp.float64)
    is_dense = isinstance(X, np.ndarray | cp.ndarray)
    if is_dense:
        layout = "dense"
        X = _device_dense(X, exact=not rg.mean_in_log_space) if isinstance(X, cp.ndarray) else X
    else:
        layout = f"{X.format}_host" if sp.issparse(X) else "sparse_device"
    kw = {
        "compute_tie_corr": tie_correct,
        "compute_nnz": rg.comp_pts,
        "col_start": col_start,
        "sub_batch_cols": _SUB_BATCH[layout, ovr],
        # Expression-space means sum expm1 of the log1p values in the same pass.
        "expm1_scale": 0.0 if rg.mean_in_log_space else _log1p_scale(rg),
    }
    if is_dense:
        kw["stream"] = cp.cuda.get_current_stream().ptr
        args = (X,)
    else:
        kw["csc"] = X.format == "csc"
        if sparse_row_spans is None:  # whole rows; only host CSR reads them
            sparse_row_spans = (X.indptr[:-1], X.indptr[1:])
        args = (X.data, X.indices, X.indptr, *sparse_row_spans)
    codes = cp.asarray(rg.group_codes, dtype=cp.int32)

    if ovr:
        # Cells outside the selected groups only enter the rest through totals.
        has_rest = bool(np.any(rg.group_codes == n_groups))
        total_sums = cp.zeros((1, n_cols) if has_rest else (1, 1), dtype=cp.float64)
        total_nnz = cp.zeros((1, n_cols) if has_rest and rg.comp_pts else (1, 1), dtype=cp.float64)
        rank_sums = cp.empty((n_groups, n_cols), dtype=cp.float64)
        tie_corr = cp.ones(n_cols, dtype=cp.float64)
        rank = _wc.ovr_dense if is_dense else _wc.ovr_sparse
        rank(
            *args,
            codes,
            sizes,
            rank_sums,
            tie_corr,
            group_sums,
            group_nnz,
            total_sums,
            total_nnz,
            compute_totals=has_rest,
            **kw,
        )
        if not has_rest:
            total_sums = group_sums.sum(axis=0, keepdims=True)
            total_nnz = group_nnz.sum(axis=0, keepdims=True)
        rest = cp.float64(n_cells) - sizes[:, None]
        means = group_sums / sizes[:, None]
        rest_means = (total_sums - group_sums) / rest
        logfoldchanges = None
        if rg.comp_pts:  # pts keep the host log-fold-change epilogue
            rg.means, rg.means_rest = cp.asnumpy(means), cp.asnumpy(rest_means)
            rg.pts = cp.asnumpy(group_nnz / sizes[:, None])
            rg.pts_rest = cp.asnumpy((total_nnz - group_nnz) / rest)
        else:
            logfoldchanges = _logfoldchanges_from_means(rg, means, rest_means)
        return _finish_ovr(
            rank_sums,
            sizes,
            n_cells,
            tie_corr,
            use_continuity=use_continuity,
            return_u_values=return_u_values,
            logfoldchanges_gpu=logfoldchanges,
        )

    ctx = _build_ovo_host_context(rg) if ovo_host_context is None else ovo_host_context
    if ctx.n_test == 0:
        return None
    test = np.asarray(ctx.test_group_indices, dtype=np.intp)
    rank_sums = cp.empty((ctx.n_test, n_cols), dtype=cp.float64)
    tie_corr = cp.ones((ctx.n_test, n_cols) if tie_correct else (1,), dtype=cp.float64)
    offsets = cp.asarray(ctx.offsets_np)
    if is_dense:
        rows = (cp.asarray(ctx.ref_row_ids), cp.asarray(ctx.all_grp_row_ids), offsets)
        _wc.ovo_dense(*args, *rows, codes, rank_sums, tie_corr, group_sums, group_nnz, rows=dense_rows, **kw)
    else:
        # CSC batches map rows to slab positions; CSR batches read the row lists.
        empty = cp.empty(0, dtype=cp.int32)
        if kw["csc"]:
            ref_map = np.full(n_cells, -1, dtype=np.int32)
            ref_map[ctx.ref_row_ids] = np.arange(ctx.n_ref, dtype=np.int32)
            grp_map = np.full(n_cells, -1, dtype=np.int32)
            grp_map[ctx.all_grp_row_ids] = np.arange(ctx.n_all_grp, dtype=np.int32)
            maps = (cp.asarray(ref_map), cp.asarray(grp_map), empty, empty, offsets)
        else:
            maps = (empty, empty, cp.asarray(ctx.ref_row_ids), cp.asarray(ctx.all_grp_row_ids), offsets)
        _wc.ovo_sparse(*args, codes, *maps, rank_sums, tie_corr, group_sums, group_nnz, n_ref=ctx.n_ref, **kw)
    means = group_sums / sizes[:, None]
    logfoldchanges = None
    if rg.comp_pts:
        rg.means, rg.pts = cp.asnumpy(means), cp.asnumpy(group_nnz / sizes[:, None])
    else:
        logfoldchanges = _logfoldchanges_from_means(rg, means[test], means[int(rg.ireference)][None, :])
    test_sizes = sizes[test]
    n_combined = test_sizes[:, None] + ctx.n_ref
    return _finish_wilcoxon(
        rank_sums,
        test_sizes[:, None] * (n_combined + 1) / 2.0,
        test_sizes[:, None] * ctx.n_ref * (n_combined + 1) * tie_corr / 12.0,
        test_sizes,
        test,
        use_continuity=use_continuity,
        return_u_values=return_u_values,
        logfoldchanges_gpu=logfoldchanges,
    )


def _finish_ovr(
    rank_sums,
    group_sizes_dev,
    n_cells,
    tie_corr,
    *,
    use_continuity,
    return_u_values,
    logfoldchanges_gpu=None,
):
    sizes = group_sizes_dev[:, None]
    expected = sizes * (n_cells + 1) / 2.0
    variance = tie_corr[None, :] * sizes * (n_cells - group_sizes_dev)[:, None]
    variance *= (n_cells + 1) / 12.0
    return _finish_wilcoxon(
        rank_sums,
        expected,
        variance,
        group_sizes_dev,
        np.arange(group_sizes_dev.size, dtype=np.intp),
        use_continuity=use_continuity,
        return_u_values=return_u_values,
        logfoldchanges_gpu=logfoldchanges_gpu,
    )


def _warn_small_ovr_groups(rg: _RankGenes, group_sizes: NDArray, n_cells: int) -> None:
    for name, size in zip(rg.groups_order, group_sizes, strict=True):
        rest = n_cells - size
        if size <= MIN_GROUP_SIZE_WARNING or rest <= MIN_GROUP_SIZE_WARNING:
            warnings.warn(
                f"Group {name} has size {size} (rest {rest}); normal approximation "
                "of the Wilcoxon statistic may be inaccurate.",
                RuntimeWarning,
                stacklevel=4,
            )


def _warn_small_ovo_groups(rg: _RankGenes, group_sizes: NDArray) -> None:
    ireference = int(rg.ireference)
    test_group_indices = [i for i in range(len(rg.groups_order)) if i != ireference]
    if not test_group_indices:
        return
    n_ref = int(group_sizes[ireference])
    small_groups = [
        str(rg.groups_order[group_index])
        for group_index in test_group_indices
        if int(group_sizes[group_index]) <= MIN_GROUP_SIZE_WARNING
    ]
    if n_ref > MIN_GROUP_SIZE_WARNING and not small_groups:
        return
    parts = []
    if small_groups:
        parts.append(
            f"{len(small_groups)} test group(s) have size "
            f"<= {MIN_GROUP_SIZE_WARNING} (first few: "
            f"{', '.join(small_groups[:5])}"
            f"{'...' if len(small_groups) > 5 else ''})"
        )
    if n_ref <= MIN_GROUP_SIZE_WARNING:
        parts.append(f"reference has size {n_ref}")
    warnings.warn(
        f"Small groups detected: {'; '.join(parts)}. normal approximation of the Wilcoxon statistic may be inaccurate.",
        RuntimeWarning,
        stacklevel=4,
    )
