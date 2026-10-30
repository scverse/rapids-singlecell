from __future__ import annotations

import warnings
from typing import TYPE_CHECKING

import cupy as cp
import cupyx.scipy.sparse as cpsp
import cupyx.scipy.special as cupyx_special
import numpy as np
import scipy.sparse as sp

from rapids_singlecell._cuda import _wilcoxon_cuda as _wc

from ._utils import EPS, MIN_GROUP_SIZE_WARNING
from ._wilcoxon_host import _build_ovo_host_context, _OvoHostContext, _run_sharded_wilcoxon

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


def _device_dense(X: cp.ndarray) -> cp.ndarray:
    """Float32/64 C- or F-contiguous array the dense kernels read in place."""
    if X.dtype not in (cp.float32, cp.float64):
        X = X.astype(cp.float32)
    return X if X.flags.c_contiguous or X.flags.f_contiguous else cp.ascontiguousarray(X)


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
    X = rg.X
    _validate_wilcoxon_sparse_dtype(X)
    rg.means = rg.vars = rg.pts = None
    rg.means_rest = rg.vars_rest = rg.pts_rest = None
    if rg.ireference is None:
        _warn_small_ovr_groups(rg, rg.group_sizes, X.shape[0])
    else:
        _warn_small_ovo_groups(rg, rg.group_sizes)

    return _run_sharded_wilcoxon(
        rg,
        tie_correct=tie_correct,
        use_continuity=use_continuity,
        chunk_size=chunk_size,
        multi_gpu=multi_gpu,
        return_u_values=return_u_values,
        shard_runner=_run_wilcoxon_shard,
    )


def _run_wilcoxon_shard(
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
        X = _device_dense(X) if isinstance(X, cp.ndarray) else X
    else:
        layout = f"{X.format}_host" if sp.issparse(X) else "sparse_device"
    kw = {
        "compute_tie_corr": tie_correct,
        "compute_nnz": rg.comp_pts,
        "col_start": col_start,
        "sub_batch_cols": _SUB_BATCH[layout, ovr] if chunk_size is None else chunk_size,
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
        _wc.ovo_dense(*args, *rows, codes, rank_sums, tie_corr, group_sums, group_nnz, **kw)
    else:
        ref_map = np.full(n_cells, -1, dtype=np.int32)
        ref_map[ctx.ref_row_ids] = np.arange(ctx.n_ref, dtype=np.int32)
        grp_map = np.full(n_cells, -1, dtype=np.int32)
        grp_map[ctx.all_grp_row_ids] = np.arange(ctx.n_all_grp, dtype=np.int32)
        rows = (cp.asarray(ctx.ref_row_ids), cp.asarray(ctx.all_grp_row_ids))
        maps = (cp.asarray(ref_map), cp.asarray(grp_map), *rows, offsets)
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
