from __future__ import annotations

import warnings
from dataclasses import dataclass
from typing import TYPE_CHECKING

import cupy as cp
import cupyx.scipy.sparse as cpsp
import cupyx.scipy.special as cupyx_special
import numpy as np
import scipy.sparse as sp

from rapids_singlecell._cuda import _wilcoxon_cuda as _wc
from rapids_singlecell._cuda import _wilcoxon_sparse_cuda as _wcs
from rapids_singlecell.get._aggregated import Aggregate

from ._utils import (
    EPS,
    MIN_GROUP_SIZE_WARNING,
    _choose_chunk_size,
)
from ._wilcoxon_host import (
    _build_ovo_host_context,
    _OvoHostContext,
    _run_sharded_wilcoxon,
)

if TYPE_CHECKING:
    from numpy.typing import NDArray

    from ._core import _RankGenes

DEFAULT_WILCOXON_CHUNK_SIZE = 512
OVR_HOST_CSC_SUB_BATCH = 512
OVR_HOST_CSR_SUB_BATCH = 2048
OVR_DEVICE_SPARSE_SUB_BATCH = 2048
OVO_HOST_SPARSE_SUB_BATCH = 256
OVO_HOST_SPARSE_RANGE_SUB_BATCH = 1024
OVO_DEVICE_SPARSE_SUB_BATCH = 128
OVR_HOST_DENSE_SUB_BATCH = 64
OVR_DEVICE_DENSE_SMALL_SUB_BATCH = 64
OVR_DEVICE_DENSE_LARGE_SUB_BATCH = 32
OVR_DEVICE_DENSE_LARGE_N_ROWS = 65_536
OVO_HOST_DENSE_SUB_BATCH = 256
OVO_DEVICE_DENSE_SUB_BATCH = 512


@dataclass(frozen=True)
class _OvoContext(_OvoHostContext):
    test_sizes: cp.ndarray


_WilcoxonResult = tuple[np.ndarray, cp.ndarray, cp.ndarray, cp.ndarray | None]
_RankBuffers = tuple[cp.ndarray, cp.ndarray]
_HostRankBuffers = tuple[cp.ndarray, cp.ndarray, cp.ndarray | None]


def _choose_wilcoxon_chunk_size(requested: int | None, n_genes: int) -> int:
    if requested is not None:
        return _choose_chunk_size(requested)
    return min(DEFAULT_WILCOXON_CHUNK_SIZE, max(1, n_genes))


def _device_wilcoxon_sums(rg: _RankGenes) -> tuple[cp.ndarray, cp.ndarray | None]:
    """Compute only the device sums needed for Wilcoxon log-fold changes."""
    agg = Aggregate(groupby=rg._aggregation_groupby, data=rg.X)
    sums_all = agg.count_mean_var({"sum"}, dof=1)["sum"]

    group_sums = sums_all[: len(rg.groups_order)]
    total_sums = sums_all.sum(axis=0, keepdims=True) if rg.ireference is None else None
    return group_sums, total_sums


def _fill_basic_stats_from_accumulators(
    rg: _RankGenes,
    group_sums: cp.ndarray,
    group_nnz: cp.ndarray,
    group_sizes: np.ndarray,
    *,
    n_cells: int,
    total_sums: cp.ndarray | None = None,
    total_nnz: cp.ndarray | None = None,
) -> None:
    # Wilcoxon output does not consume per-group variance.
    n = cp.asarray(group_sizes, dtype=cp.float64)[:, None]
    means = group_sums / n
    rg.means = cp.asnumpy(means)
    rg.vars = None
    rg.pts = cp.asnumpy(group_nnz / n)

    n_rest = cp.float64(n_cells) - n
    if total_sums is None:
        total_sums = group_sums.sum(axis=0, keepdims=True)
    rest_sums = total_sums - group_sums
    rest_means = rest_sums / n_rest
    rg.means_rest = cp.asnumpy(rest_means)
    rg.vars_rest = None
    if total_nnz is None:
        total_nnz = group_nnz.sum(axis=0, keepdims=True)
    rg.pts_rest = cp.asnumpy((total_nnz - group_nnz) / n_rest)


def _logfoldchanges_from_means(
    rg: _RankGenes,
    mean_group: cp.ndarray,
    mean_reference: cp.ndarray,
) -> cp.ndarray:
    if rg.mean_in_log_space:
        scale = (
            cp.float64(np.log(rg._log1p_base))
            if rg._log1p_base is not None
            else cp.float64(1.0)
        )
        group_expr = cp.expm1(mean_group * scale)
        reference_expr = cp.expm1(mean_reference * scale)
    else:
        group_expr = mean_group
        reference_expr = mean_reference
    return cp.log2((group_expr + EPS) / (reference_expr + EPS))


def _ovo_logfoldchanges_from_sums(
    rg: _RankGenes,
    group_sums_slots: cp.ndarray,
    test_sizes: cp.ndarray,
    n_ref: int,
) -> cp.ndarray:
    n_test = int(test_sizes.shape[0])
    mean_group = group_sums_slots[:n_test] / test_sizes[:, None]
    mean_ref = group_sums_slots[n_test][None, :] / cp.float64(n_ref)
    return _logfoldchanges_from_means(rg, mean_group, mean_ref)


def _ovr_logfoldchanges_from_sums(
    rg: _RankGenes,
    group_sums: cp.ndarray,
    group_sizes: cp.ndarray,
    n_cells: int,
    total_sums: cp.ndarray | None,
) -> cp.ndarray:
    sizes = group_sizes[:, None]
    mean_group = group_sums / sizes
    if total_sums is None:
        total_sums = group_sums.sum(axis=0, keepdims=True)
    mean_rest = (total_sums - group_sums) / (cp.float64(n_cells) - sizes)
    return _logfoldchanges_from_means(rg, mean_group, mean_rest)


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
    diff = rank_sums - expected
    if use_continuity:
        diff = cp.sign(diff) * cp.maximum(cp.abs(diff) - 0.5, 0.0)
    z = diff / cp.sqrt(variance)
    cp.nan_to_num(z, copy=False)
    p_values = cupyx_special.erfc(cp.abs(z) * cp.float64(cp.sqrt(0.5)))
    scores = (
        rank_sums - sizes[:, None] * (sizes[:, None] + 1.0) / 2.0
        if return_u_values
        else z
    )
    return (
        group_indices,
        scores,
        p_values,
        logfoldchanges_gpu,
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


def _finish_ovo(
    ctx,
    rank_sums,
    tie_corr_arr,
    *,
    use_continuity,
    return_u_values,
    logfoldchanges_gpu,
):
    n_combined = ctx.test_sizes + ctx.n_ref
    expected = ctx.test_sizes[:, None] * (n_combined[:, None] + 1) / 2.0
    variance = (
        ctx.test_sizes[:, None]
        * ctx.n_ref
        * (n_combined[:, None] + 1)
        * tie_corr_arr
        / 12.0
    )
    return _finish_wilcoxon(
        rank_sums,
        expected,
        variance,
        ctx.test_sizes,
        np.asarray(ctx.test_group_indices, dtype=np.intp),
        use_continuity=use_continuity,
        return_u_values=return_u_values,
        logfoldchanges_gpu=logfoldchanges_gpu,
    )


def _validate_wilcoxon_sparse_dtype(X) -> None:
    if not (sp.issparse(X) or cpsp.issparse(X)):
        return
    if X.format not in {"csr", "csc"}:
        raise TypeError(
            "Wilcoxon sparse input must be CSR or CSC; refusing hidden "
            f"full-matrix conversion from {X.format!r}."
        )
    data_dtype = np.dtype(X.data.dtype)
    if data_dtype.kind == "c":
        msg = (
            "Wilcoxon sparse input data dtype must be real; complex sparse "
            "data is not supported."
        )
        raise TypeError(msg)
    if cpsp.issparse(X) and data_dtype not in {
        np.dtype(np.float32),
        np.dtype(np.float64),
    }:
        msg = (
            "Wilcoxon device sparse input data dtype must be float32 or "
            f"float64; got {data_dtype}."
        )
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
        msg = (
            "Wilcoxon sparse input data dtype must be float32, float64, bool, "
            f"or integer; got {data_dtype}."
        )
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
            msg = (
                "Wilcoxon sparse indices and indptr must be int32 or int64; "
                f"got {indices_dtype}."
            )
            raise TypeError(msg)


def _device_sparse_arrays(X):
    """Cast validated sparse values for float32-key ranking kernels."""
    return X.data.astype(cp.float32, copy=False), X.indices, X.indptr


def wilcoxon(
    rg: _RankGenes,
    *,
    tie_correct: bool,
    use_continuity: bool = False,
    chunk_size: int | None = None,
    multi_gpu: bool | list[int] | str | None = None,
    return_u_values: bool = False,
) -> _WilcoxonResult | None:
    """Compute Wilcoxon rank-sum test statistics."""
    # Host dense streams column windows; device dense stays device-resident.
    # Aggregate stats on GPU, otherwise compute them inside streaming paths.
    X = rg.X
    _validate_wilcoxon_sparse_dtype(X)
    rg.means = rg.vars = rg.pts = None
    rg.means_rest = rg.vars_rest = rg.pts_rest = None
    n_cells = X.shape[0]
    group_sizes = rg.group_sizes
    if rg.ireference is None:
        _warn_small_ovr_groups(rg, group_sizes, n_cells)
    else:
        _warn_small_ovo_groups(rg, group_sizes)

    return _run_sharded_wilcoxon(
        rg,
        tie_correct=tie_correct,
        use_continuity=use_continuity,
        chunk_size=chunk_size,
        multi_gpu=multi_gpu,
        return_u_values=return_u_values,
        shard_runner=_run_wilcoxon_device_shard,
    )


def _run_wilcoxon_device_shard(
    rg: _RankGenes,
    *,
    tie_correct: bool,
    use_continuity: bool,
    chunk_size: int | None,
    return_u_values: bool,
    column_range: tuple[int, int] | None = None,
    sparse_row_spans: tuple[np.ndarray, np.ndarray] | None = None,
    ovo_host_context: _OvoHostContext | None = None,
) -> _WilcoxonResult | None:
    X = rg.X
    n_cells = X.shape[0]
    group_sizes = rg.group_sizes
    device_group_sums = None
    device_total_sums = None
    is_device_input = isinstance(X, cp.ndarray) or cpsp.issparse(X)
    if is_device_input:
        if rg.comp_pts:
            rg._basic_stats()
        else:
            device_group_sums, device_total_sums = _device_wilcoxon_sums(rg)

    if rg.ireference is None:
        group_sizes_dev = cp.asarray(group_sizes, dtype=cp.float64)
        logfoldchanges_gpu = (
            _ovr_logfoldchanges_from_sums(
                rg,
                device_group_sums,
                group_sizes_dev,
                n_cells,
                device_total_sums,
            )
            if device_group_sums is not None
            else None
        )
        if is_device_input:
            rank_sums, tie_corr = _run_ovr_device(
                rg,
                group_sizes_dev,
                tie_correct=tie_correct,
                chunk_size=chunk_size,
            )
        else:
            rank_sums, tie_corr, logfoldchanges_gpu = _run_ovr_host(
                rg,
                group_sizes_dev,
                tie_correct=tie_correct,
                column_range=column_range,
                sparse_row_spans=sparse_row_spans,
            )
        return _finish_ovr(
            rank_sums,
            group_sizes_dev,
            n_cells,
            tie_corr,
            use_continuity=use_continuity,
            return_u_values=return_u_values,
            logfoldchanges_gpu=logfoldchanges_gpu,
        )

    host_ctx = (
        _build_ovo_host_context(rg) if ovo_host_context is None else ovo_host_context
    )
    ctx = _OvoContext(
        **vars(host_ctx),
        test_sizes=cp.asarray(
            group_sizes[np.asarray(host_ctx.test_group_indices, dtype=np.intp)],
            dtype=cp.float64,
        ),
    )
    if ctx.n_test == 0:
        return None
    logfoldchanges_gpu = None
    if device_group_sums is not None:
        slot_indices = [*ctx.test_group_indices, int(rg.ireference)]
        logfoldchanges_gpu = _ovo_logfoldchanges_from_sums(
            rg,
            device_group_sums[slot_indices],
            ctx.test_sizes,
            ctx.n_ref,
        )
    if is_device_input:
        rank_sums, tie_corr_arr = _run_ovo_device(
            X,
            ctx,
            tie_correct=tie_correct,
            chunk_size=chunk_size,
        )
    else:
        rank_sums, tie_corr_arr, logfoldchanges_gpu = _run_ovo_host(
            rg,
            ctx,
            tie_correct=tie_correct,
            chunk_size=chunk_size,
            column_range=column_range,
            sparse_row_spans=sparse_row_spans,
        )
    return _finish_ovo(
        ctx,
        rank_sums,
        tie_corr_arr,
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
        f"Small groups detected: {'; '.join(parts)}. normal approximation "
        "of the Wilcoxon statistic may be inaccurate.",
        RuntimeWarning,
        stacklevel=4,
    )


def _finish_ovo_stats(
    rg: _RankGenes,
    ctx: _OvoContext,
    group_sums: cp.ndarray,
    group_nnz: cp.ndarray,
) -> cp.ndarray | None:
    if not rg.comp_pts:
        return _ovo_logfoldchanges_from_sums(
            rg,
            group_sums,
            ctx.test_sizes,
            ctx.n_ref,
        )
    indices = np.asarray([*ctx.test_group_indices, int(rg.ireference)], dtype=np.intp)
    sizes = cp.asarray(rg.group_sizes[indices], dtype=cp.float64)[:, None]
    n_genes = int(group_sums.shape[1])
    shape = (len(rg.groups_order), n_genes)
    rg.means = np.zeros(shape, dtype=np.float64)
    rg.pts = np.zeros(shape, dtype=np.float64)
    rg.means[indices] = cp.asnumpy(group_sums / sizes)
    rg.pts[indices] = cp.asnumpy(group_nnz / sizes)
    rg.vars = rg.means_rest = rg.vars_rest = rg.pts_rest = None
    return None


def _run_ovr_host(
    rg: _RankGenes,
    group_sizes_dev: cp.ndarray,
    *,
    tie_correct: bool,
    column_range: tuple[int, int] | None,
    sparse_row_spans: tuple[np.ndarray, np.ndarray] | None,
) -> _HostRankBuffers:
    X = rg.X
    is_sparse = sp.issparse(X)
    n_cells = X.shape[0]
    col_start, col_stop = column_range
    n_total_genes = col_stop - col_start
    n_groups = len(rg.groups_order)
    group_sizes_np = rg.group_sizes.astype(np.float64, copy=False)
    compute_nnz = rg.comp_pts
    compute_totals = bool(np.any(rg.group_codes == n_groups))
    rank_sums = cp.empty((n_groups, n_total_genes), dtype=cp.float64)
    tie_corr = (
        cp.ones(n_total_genes, dtype=cp.float64)
        if is_sparse or not tie_correct
        else cp.empty(n_total_genes, dtype=cp.float64)
    )
    group_sums = cp.empty((n_groups, n_total_genes), dtype=cp.float64)
    group_nnz = cp.empty(
        (n_groups, n_total_genes) if compute_nnz else (1, 1),
        dtype=cp.float64,
    )
    total_sums = cp.empty(
        (1, n_total_genes) if compute_totals else (1, 1),
        dtype=cp.float64,
    )
    total_nnz = cp.empty(
        (1, n_total_genes) if (compute_totals and compute_nnz) else (1, 1),
        dtype=cp.float64,
    )

    if is_sparse:
        group_codes = rg.group_codes.astype(np.int32, copy=False)
        if X.format == "csc":
            _wcs.ovr_sparse_csc_host(
                X.data,
                X.indices,
                X.indptr,
                group_codes,
                group_sizes_np,
                rank_sums,
                tie_corr,
                group_sums,
                group_nnz,
                total_sums,
                total_nnz,
                compute_tie_corr=tie_correct,
                compute_nnz=compute_nnz,
                compute_totals=compute_totals,
                sub_batch_cols=OVR_HOST_CSC_SUB_BATCH,
            )
        else:
            row_starts, row_stops = sparse_row_spans
            _wcs.ovr_sparse_csr_host(
                X.data,
                X.indices,
                X.indptr,
                row_starts,
                row_stops,
                group_codes,
                group_sizes_np,
                rank_sums,
                tie_corr,
                group_sums,
                group_nnz,
                total_sums,
                total_nnz,
                n_cols=X.shape[1],
                compute_tie_corr=tie_correct,
                compute_nnz=compute_nnz,
                compute_totals=compute_totals,
                sub_batch_cols=OVR_HOST_CSR_SUB_BATCH,
                col_start=col_start,
                col_stop=col_stop,
            )
    else:
        group_codes_gpu = cp.asarray(rg.group_codes, dtype=cp.int32)
        _wc.ovr_rank_dense_host_streaming(
            X,
            group_codes_gpu,
            rank_sums,
            tie_corr,
            group_sums,
            group_nnz,
            total_sums,
            total_nnz,
            compute_tie_corr=tie_correct,
            compute_nnz=compute_nnz,
            compute_totals=compute_totals,
            col_start=col_start,
            col_stop=col_stop,
            sub_batch_cols=OVR_HOST_DENSE_SUB_BATCH,
        )

    logfoldchanges_gpu = None
    if rg.comp_pts:
        _fill_basic_stats_from_accumulators(
            rg,
            group_sums,
            group_nnz,
            group_sizes_np,
            n_cells=n_cells,
            total_sums=total_sums if compute_totals else None,
            total_nnz=total_nnz if compute_totals else None,
        )
    else:
        logfoldchanges_gpu = _ovr_logfoldchanges_from_sums(
            rg,
            group_sums,
            group_sizes_dev,
            n_cells,
            total_sums if compute_totals else None,
        )
    return rank_sums, tie_corr, logfoldchanges_gpu


def _run_ovr_device(
    rg: _RankGenes,
    group_sizes_dev: cp.ndarray,
    *,
    tie_correct: bool,
    chunk_size: int | None,
) -> _RankBuffers:
    X = rg.X
    n_cells, n_total_genes = X.shape
    sparse_format = X.format if cpsp.issparse(X) else None
    n_groups = len(rg.groups_order)
    group_codes_gpu = cp.asarray(rg.group_codes, dtype=cp.int32)
    rank_sums = cp.empty((n_groups, n_total_genes), dtype=cp.float64)
    tie_corr = (
        cp.ones(n_total_genes, dtype=cp.float64)
        if sparse_format is not None or not tie_correct
        else cp.empty(n_total_genes, dtype=cp.float64)
    )

    if sparse_format is not None:
        data, indices, indptr = _device_sparse_arrays(X)
        kernel = (
            _wcs.ovr_sparse_csc_device
            if sparse_format == "csc"
            else _wcs.ovr_sparse_csr_device
        )
        kernel(
            data,
            indices,
            indptr,
            group_codes_gpu,
            group_sizes_dev,
            rank_sums,
            tie_corr,
            compute_tie_corr=tie_correct,
            sub_batch_cols=OVR_DEVICE_SPARSE_SUB_BATCH,
        )
    else:
        sub_batch_cols = (
            OVR_DEVICE_DENSE_LARGE_SUB_BATCH
            if n_cells >= OVR_DEVICE_DENSE_LARGE_N_ROWS
            else OVR_DEVICE_DENSE_SMALL_SUB_BATCH
        )
        if chunk_size is None and X.dtype == cp.float32 and X.flags.f_contiguous:
            _wc.ovr_rank_dense_streaming(
                X,
                group_codes_gpu,
                rank_sums,
                tie_corr,
                compute_tie_corr=tie_correct,
                sub_batch_cols=sub_batch_cols,
                stream=cp.cuda.get_current_stream().ptr,
            )
        else:
            chunk_width = _choose_wilcoxon_chunk_size(chunk_size, n_total_genes)
            for start in range(0, n_total_genes, chunk_width):
                stop = min(start + chunk_width, n_total_genes)
                block_f32 = cp.asarray(X[:, start:stop], dtype=cp.float32, order="F")
                n_cols = stop - start
                sub_rank_sums = cp.empty((n_groups, n_cols), dtype=cp.float64)
                sub_tie_corr = (
                    cp.empty(n_cols, dtype=cp.float64)
                    if tie_correct
                    else cp.ones(n_cols, dtype=cp.float64)
                )
                _wc.ovr_rank_dense_streaming(
                    block_f32,
                    group_codes_gpu,
                    sub_rank_sums,
                    sub_tie_corr,
                    compute_tie_corr=tie_correct,
                    sub_batch_cols=sub_batch_cols,
                    stream=cp.cuda.get_current_stream().ptr,
                )
                rank_sums[:, start:stop] = sub_rank_sums
                if tie_correct:
                    tie_corr[start:stop] = sub_tie_corr
    return rank_sums, tie_corr


def _run_ovo_host(
    rg: _RankGenes,
    ctx: _OvoContext,
    *,
    tie_correct: bool,
    chunk_size: int | None,
    column_range: tuple[int, int] | None,
    sparse_row_spans: tuple[np.ndarray, np.ndarray] | None,
) -> _HostRankBuffers:
    X = rg.X
    is_sparse = sp.issparse(X)
    col_start, col_stop = column_range
    n_total_genes = col_stop - col_start
    rank_sums = cp.zeros((ctx.n_test, n_total_genes), dtype=cp.float64)
    tie_corr_arr = cp.ones(
        (ctx.n_test, n_total_genes) if tie_correct else (1,), dtype=cp.float64
    )
    n_groups_stats = ctx.n_test + 1
    compute_nnz = rg.comp_pts
    stats_shape = (n_groups_stats, n_total_genes)
    group_sums = cp.empty(stats_shape, dtype=cp.float64)
    group_nnz = cp.empty(
        stats_shape if compute_nnz else ((1,) if is_sparse else (1, 1)),
        dtype=cp.float64,
    )

    if is_sparse:
        if X.format == "csc":
            n_groups = len(rg.groups_order)
            ireference = int(rg.ireference)
            stats_code_lookup = np.full(n_groups + 1, n_groups_stats, dtype=np.int32)
            test_group_indices_np = np.asarray(ctx.test_group_indices, dtype=np.intp)
            stats_code_lookup[test_group_indices_np] = np.arange(
                ctx.n_test, dtype=np.int32
            )
            stats_code_lookup[ireference] = ctx.n_test
            stats_codes = stats_code_lookup[rg.group_codes]
            ref_row_map = np.full(X.shape[0], -1, dtype=np.int32)
            ref_row_map[ctx.ref_row_ids] = np.arange(ctx.n_ref, dtype=np.int32)
            grp_row_map = np.full(X.shape[0], -1, dtype=np.int32)
            grp_row_map[ctx.all_grp_row_ids] = np.arange(ctx.n_all_grp, dtype=np.int32)
            _wcs.ovo_streaming_csc_host(
                X.data,
                X.indices,
                X.indptr,
                ref_row_map,
                grp_row_map,
                ctx.offsets_np,
                stats_codes,
                rank_sums,
                tie_corr_arr,
                group_sums,
                group_nnz,
                n_ref=ctx.n_ref,
                n_all_grp=ctx.n_all_grp,
                compute_tie_corr=tie_correct,
                compute_nnz=compute_nnz,
                sub_batch_cols=OVO_HOST_SPARSE_SUB_BATCH,
                analytic_zeros=not rg._sparse_negative_fallback,
            )
        else:
            row_starts, row_stops = sparse_row_spans
            _wcs.ovo_streaming_csr_host(
                X.data,
                X.indices,
                row_starts,
                row_stops,
                ctx.ref_row_ids,
                ctx.all_grp_row_ids,
                ctx.offsets_np,
                rank_sums,
                tie_corr_arr,
                group_sums,
                group_nnz,
                n_cols=X.shape[1],
                compute_tie_corr=tie_correct,
                compute_nnz=compute_nnz,
                sub_batch_cols=OVO_HOST_SPARSE_RANGE_SUB_BATCH,
                analytic_zeros=not rg._sparse_negative_fallback,
                col_start=col_start,
                col_stop=col_stop,
            )
    else:
        dense_sub_batch_cols = (
            _choose_wilcoxon_chunk_size(chunk_size, n_total_genes)
            if chunk_size is not None
            else OVO_HOST_DENSE_SUB_BATCH
        )
        _wc.ovo_rank_dense_host_streaming(
            X,
            ctx.ref_row_ids,
            ctx.all_grp_row_ids,
            ctx.offsets_np,
            rank_sums,
            tie_corr_arr,
            group_sums,
            group_nnz,
            compute_tie_corr=tie_correct,
            compute_nnz=compute_nnz,
            col_start=col_start,
            col_stop=col_stop,
            sub_batch_cols=dense_sub_batch_cols,
        )

    logfoldchanges_gpu = _finish_ovo_stats(rg, ctx, group_sums, group_nnz)
    return rank_sums, tie_corr_arr, logfoldchanges_gpu


def _run_ovo_device(
    X,
    ctx: _OvoContext,
    *,
    tie_correct: bool,
    chunk_size: int | None,
) -> _RankBuffers:
    n_total_genes = X.shape[1]
    sparse_format = X.format if cpsp.issparse(X) else None
    offsets_gpu = cp.asarray(ctx.offsets_np)
    rank_sums = (
        cp.zeros((ctx.n_test, n_total_genes), dtype=cp.float64)
        if sparse_format is not None
        else cp.empty((ctx.n_test, n_total_genes), dtype=cp.float64)
    )
    tie_corr_arr = cp.ones(
        (ctx.n_test, n_total_genes) if tie_correct else (1,), dtype=cp.float64
    )

    if sparse_format is not None:
        data, indices, indptr = _device_sparse_arrays(X)
        if sparse_format == "csc":
            ref_row_map = np.full(X.shape[0], -1, dtype=np.int32)
            ref_row_map[ctx.ref_row_ids] = np.arange(ctx.n_ref, dtype=np.int32)
            grp_row_map = np.full(X.shape[0], -1, dtype=np.int32)
            grp_row_map[ctx.all_grp_row_ids] = np.arange(ctx.n_all_grp, dtype=np.int32)
            kernel = _wcs.ovo_streaming_csc_device
            ref_rows = cp.asarray(ref_row_map)
            group_rows = cp.asarray(grp_row_map)
        else:
            kernel = _wcs.ovo_streaming_csr_device
            ref_rows = cp.asarray(ctx.ref_row_ids, dtype=cp.int32)
            group_rows = cp.asarray(ctx.all_grp_row_ids, dtype=cp.int32)
        kernel(
            data,
            indices,
            indptr,
            ref_rows,
            group_rows,
            offsets_gpu,
            rank_sums,
            tie_corr_arr,
            n_ref=ctx.n_ref,
            n_all_grp=ctx.n_all_grp,
            compute_tie_corr=tie_correct,
            sub_batch_cols=OVO_DEVICE_SPARSE_SUB_BATCH,
        )
    else:
        chunk_width = _choose_wilcoxon_chunk_size(chunk_size, n_total_genes)
        ref_rows = cp.asarray(ctx.ref_row_ids, dtype=cp.int32)
        grp_rows = cp.asarray(ctx.all_grp_row_ids, dtype=cp.int32)
        for start in range(0, n_total_genes, chunk_width):
            stop = min(start + chunk_width, n_total_genes)
            n_cols = stop - start
            ref_f32 = cp.asarray(X[ref_rows, start:stop], dtype=cp.float32, order="F")
            grp_f32 = cp.asarray(X[grp_rows, start:stop], dtype=cp.float32, order="F")
            sub_rank_sums = cp.empty((ctx.n_test, n_cols), dtype=cp.float64)
            sub_tie_corr = (
                cp.ones((ctx.n_test, n_cols), dtype=cp.float64)
                if tie_correct
                else tie_corr_arr
            )
            _wc.ovo_rank_dense_tiered_unsorted_ref(
                ref_f32,
                grp_f32,
                offsets_gpu,
                sub_rank_sums,
                sub_tie_corr,
                compute_tie_corr=tie_correct,
                sub_batch_cols=OVO_DEVICE_DENSE_SUB_BATCH,
                stream=cp.cuda.get_current_stream().ptr,
            )
            rank_sums[:, start:stop] = sub_rank_sums
            if tie_correct:
                tie_corr_arr[:, start:stop] = sub_tie_corr
    return rank_sums, tie_corr_arr
