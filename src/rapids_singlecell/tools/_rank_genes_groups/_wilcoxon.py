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

from ._utils import (
    EPS,
    MIN_GROUP_SIZE_WARNING,
    _choose_chunk_size,
    _get_column_block,
    _ovo_dense_block,
    _ovr_dense_block_f32,
)

if TYPE_CHECKING:
    from numpy.typing import NDArray

    from ._core import _RankGenes

DEFAULT_WILCOXON_CHUNK_SIZE = 512
OVR_HOST_CSC_SUB_BATCH = 512
OVR_HOST_CSR_SUB_BATCH = 2048
OVR_DEVICE_CSC_SUB_BATCH = 2048
OVR_DEVICE_CSR_SUB_BATCH = 2048
OVO_HOST_SPARSE_SUB_BATCH = 256
OVO_DEVICE_SPARSE_SUB_BATCH = 128
OVR_DENSE_SUB_BATCH = 64
OVO_DENSE_TIERED_SUB_BATCH = 256


@dataclass(frozen=True)
class _OvoContext:
    codes: np.ndarray
    n_groups: int
    ireference: int
    n_ref: int
    ref_row_ids: np.ndarray
    test_group_indices: list[int]
    all_grp_row_ids: np.ndarray
    offsets_np: np.ndarray
    offsets_gpu: cp.ndarray
    n_all_grp: int
    n_test: int
    test_sizes: cp.ndarray


def _choose_wilcoxon_chunk_size(requested: int | None, n_genes: int) -> int:
    if requested is not None:
        return _choose_chunk_size(requested)
    return min(DEFAULT_WILCOXON_CHUNK_SIZE, max(1, n_genes))


def _fill_ovo_chunk_stats(
    rg: _RankGenes,
    ref_block: cp.ndarray,
    grp_block: cp.ndarray,
    *,
    offsets: np.ndarray,
    test_group_indices: list[int],
    start: int,
    stop: int,
    group_sizes: NDArray,
) -> None:
    if not rg._compute_stats_in_chunks:
        return

    ireference = rg.ireference
    n_ref = int(group_sizes[ireference])
    ref_mean = ref_block.mean(axis=0)
    rg.means[ireference, start:stop] = cp.asnumpy(ref_mean)
    if n_ref > 1:
        rg.vars[ireference, start:stop] = cp.asnumpy(ref_block.var(axis=0, ddof=1))
    if rg.comp_pts:
        ref_nnz = (ref_block != 0).sum(axis=0)
        rg.pts[ireference, start:stop] = cp.asnumpy(ref_nnz / n_ref)

    for slot, group_index in enumerate(test_group_indices):
        begin = int(offsets[slot])
        end = int(offsets[slot + 1])
        n_group = int(group_sizes[group_index])
        group_block = grp_block[begin:end]
        group_mean = group_block.mean(axis=0)
        rg.means[group_index, start:stop] = cp.asnumpy(group_mean)
        if n_group > 1:
            rg.vars[group_index, start:stop] = cp.asnumpy(
                group_block.var(axis=0, ddof=1)
            )
        if rg.comp_pts:
            group_nnz = (group_block != 0).sum(axis=0)
            rg.pts[group_index, start:stop] = cp.asnumpy(group_nnz / n_group)


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
    # vars left zero: wilcoxon does not output per-group variance.
    n = cp.asarray(group_sizes, dtype=cp.float64)[:, None]
    means = group_sums / n
    rg.means = cp.asnumpy(means)
    rg.vars = np.zeros_like(rg.means)
    rg.pts = cp.asnumpy(group_nnz / n) if rg.comp_pts else None

    n_rest = cp.float64(n_cells) - n
    if total_sums is None:
        total_sums = group_sums.sum(axis=0, keepdims=True)
    rest_sums = total_sums - group_sums
    rest_means = rest_sums / n_rest
    rg.means_rest = cp.asnumpy(rest_means)
    rg.vars_rest = np.zeros_like(rg.means_rest)
    if rg.comp_pts:
        if total_nnz is None:
            total_nnz = group_nnz.sum(axis=0, keepdims=True)
        rg.pts_rest = cp.asnumpy((total_nnz - group_nnz) / n_rest)
    else:
        rg.pts_rest = None
    rg._compute_stats_in_chunks = False


def _fill_ovo_stats_from_accumulators(
    rg: _RankGenes,
    group_sums_slots: cp.ndarray,
    group_nnz_slots: cp.ndarray,
    *,
    group_sizes: NDArray,
    test_group_indices: list[int],
    n_ref: int,
) -> None:
    n_test = len(test_group_indices)
    n_genes = int(group_sums_slots.shape[1])
    n_groups = len(rg.groups_order)
    slot_group_indices = np.empty(n_test + 1, dtype=np.intp)
    slot_group_indices[:n_test] = np.asarray(test_group_indices, dtype=np.intp)
    slot_group_indices[n_test] = rg.ireference
    slot_sizes = np.empty(n_test + 1, dtype=np.float64)
    slot_sizes[:n_test] = group_sizes[slot_group_indices[:n_test]]
    slot_sizes[n_test] = n_ref
    slot_sizes_dev = cp.asarray(slot_sizes, dtype=cp.float64)[:, None]

    rg.means = np.zeros((n_groups, n_genes), dtype=np.float64)
    rg.vars = np.zeros((n_groups, n_genes), dtype=np.float64)
    rg.pts = np.zeros((n_groups, n_genes), dtype=np.float64) if rg.comp_pts else None

    means_slots = group_sums_slots / slot_sizes_dev
    rg.means[slot_group_indices] = cp.asnumpy(means_slots)
    # vars left zero: wilcoxon does not output per-group variance.
    if rg.comp_pts:
        rg.pts[slot_group_indices] = cp.asnumpy(group_nnz_slots / slot_sizes_dev)

    rg.means_rest = None
    rg.vars_rest = None
    rg.pts_rest = None
    rg._compute_stats_in_chunks = False


def _fill_ovo_dense_stats_from_accumulators(
    rg: _RankGenes,
    group_sums_slots: cp.ndarray,
    group_sum_sq_slots: cp.ndarray,
    group_nnz_slots: cp.ndarray,
    *,
    group_sizes: NDArray,
    test_group_indices: list[int],
    n_ref: int,
) -> None:
    n_test = len(test_group_indices)
    n_genes = int(group_sums_slots.shape[1])
    n_groups = len(rg.groups_order)
    slot_group_indices = np.empty(n_test + 1, dtype=np.intp)
    slot_group_indices[:n_test] = np.asarray(test_group_indices, dtype=np.intp)
    slot_group_indices[n_test] = rg.ireference
    slot_sizes = np.empty(n_test + 1, dtype=np.float64)
    slot_sizes[:n_test] = group_sizes[slot_group_indices[:n_test]]
    slot_sizes[n_test] = n_ref
    slot_sizes_dev = cp.asarray(slot_sizes, dtype=cp.float64)[:, None]

    rg.means = np.zeros((n_groups, n_genes), dtype=np.float64)
    rg.vars = np.zeros((n_groups, n_genes), dtype=np.float64)
    rg.pts = np.zeros((n_groups, n_genes), dtype=np.float64) if rg.comp_pts else None

    means_slots = group_sums_slots / slot_sizes_dev
    vars_slots = group_sum_sq_slots / slot_sizes_dev - means_slots**2
    vars_slots = cp.where(
        slot_sizes_dev > 1.0,
        vars_slots * slot_sizes_dev / (slot_sizes_dev - 1.0),
        0.0,
    )
    rg.means[slot_group_indices] = cp.asnumpy(means_slots)
    rg.vars[slot_group_indices] = cp.asnumpy(vars_slots)
    if rg.comp_pts:
        rg.pts[slot_group_indices] = cp.asnumpy(group_nnz_slots / slot_sizes_dev)

    rg.means_rest = None
    rg.vars_rest = None
    rg.pts_rest = None
    rg._compute_stats_in_chunks = False


def _ovo_logfoldchanges_from_sums(
    rg: _RankGenes,
    group_sums_slots: cp.ndarray,
    test_sizes: cp.ndarray,
    n_ref: int,
) -> cp.ndarray:
    n_test = int(test_sizes.shape[0])
    mean_group = group_sums_slots[:n_test] / test_sizes[:, None]
    mean_ref = group_sums_slots[n_test][None, :] / cp.float64(n_ref)
    if rg._log1p_base is not None:
        scale = cp.float64(np.log(rg._log1p_base))
        group_expr = cp.expm1(mean_group * scale)
        ref_expr = cp.expm1(mean_ref * scale)
    else:
        group_expr = cp.expm1(mean_group)
        ref_expr = cp.expm1(mean_ref)
    return cp.log2((group_expr + EPS) / (ref_expr + EPS))


def _wilcoxon_scores(
    rank_sums: cp.ndarray,
    group_sizes: cp.ndarray,
    z_scores: cp.ndarray,
    *,
    return_u_values: bool,
) -> cp.ndarray:
    if not return_u_values:
        return z_scores
    n_group = group_sizes[:, None]
    return rank_sums - n_group * (n_group + 1.0) / 2.0


def _z_scores_pvals(
    rank_sums: cp.ndarray,
    expected: cp.ndarray,
    variance: cp.ndarray,
    sizes: cp.ndarray,
    *,
    use_continuity: bool,
    return_u_values: bool,
) -> tuple[cp.ndarray, cp.ndarray]:
    """Shared Wilcoxon normal-approximation epilogue -> (scores, p_values)."""
    diff = rank_sums - expected
    if use_continuity:
        diff = cp.sign(diff) * cp.maximum(cp.abs(diff) - 0.5, 0.0)
    z = diff / cp.sqrt(variance)
    cp.nan_to_num(z, copy=False)
    p_values = cupyx_special.erfc(cp.abs(z) * cp.float64(cp.sqrt(0.5)))
    scores = _wilcoxon_scores(rank_sums, sizes, z, return_u_values=return_u_values)
    return scores, p_values


def _ovr_z_pvals(
    rank_sums: cp.ndarray,
    group_sizes_dev: cp.ndarray,
    rest_sizes: cp.ndarray,
    n_cells: int,
    tie_corr: cp.ndarray,
    *,
    use_continuity: bool,
    return_u_values: bool,
) -> tuple[cp.ndarray, cp.ndarray]:
    """Group-vs-rest scores/p-values (tie_corr is ones when not correcting)."""
    expected = group_sizes_dev[:, None] * (n_cells + 1) / 2.0
    variance = tie_corr[None, :] * group_sizes_dev[:, None] * rest_sizes[:, None]
    variance *= (n_cells + 1) / 12.0
    return _z_scores_pvals(
        rank_sums,
        expected,
        variance,
        group_sizes_dev,
        use_continuity=use_continuity,
        return_u_values=return_u_values,
    )


def _finish_ovr(
    rank_sums,
    group_sizes_dev,
    rest_sizes,
    n_cells,
    tie_corr,
    *,
    use_continuity,
    return_u_values,
    n_groups,
):
    """OVR epilogue: z/p-values -> host -> per-group (idx, scores, pvals)."""
    scores, p_values = _ovr_z_pvals(
        rank_sums,
        group_sizes_dev,
        rest_sizes,
        n_cells,
        tie_corr,
        use_continuity=use_continuity,
        return_u_values=return_u_values,
    )
    scores_host = scores.get()
    p_host = p_values.get()
    return [(gi, scores_host[gi], p_host[gi]) for gi in range(n_groups)]


def _ovo_z_pvals(
    rank_sums: cp.ndarray,
    test_sizes: cp.ndarray,
    n_ref: int,
    tie_corr_arr: cp.ndarray,
    *,
    tie_correct: bool,
    use_continuity: bool,
    return_u_values: bool,
) -> tuple[cp.ndarray, cp.ndarray]:
    """Group-vs-reference scores/p-values from rank sums and tie correction."""
    n_combined = test_sizes + n_ref
    expected = test_sizes[:, None] * (n_combined[:, None] + 1) / 2.0
    variance = test_sizes[:, None] * n_ref * (n_combined[:, None] + 1) / 12.0
    if tie_correct:
        variance = variance * tie_corr_arr
    return _z_scores_pvals(
        rank_sums,
        expected,
        variance,
        test_sizes,
        use_continuity=use_continuity,
        return_u_values=return_u_values,
    )


def _finish_ovo(
    rank_sums,
    test_sizes,
    n_ref,
    tie_corr_arr,
    *,
    tie_correct,
    use_continuity,
    return_u_values,
    rg,
    test_group_indices,
    logfoldchanges_gpu,
):
    """OVO epilogue: z/p-values; stash GPU result if requested, else host tuples."""
    scores, p_values = _ovo_z_pvals(
        rank_sums,
        test_sizes,
        n_ref,
        tie_corr_arr,
        tie_correct=tie_correct,
        use_continuity=use_continuity,
        return_u_values=return_u_values,
    )
    if rg._store_wilcoxon_gpu_result:
        rg._wilcoxon_gpu_result = (
            np.asarray(test_group_indices, dtype=np.intp),
            scores,
            p_values,
            logfoldchanges_gpu,
        )
        return []
    scores_host = scores.get()
    p_host = p_values.get()
    return [
        (group_index, scores_host[slot], p_host[slot])
        for slot, group_index in enumerate(test_group_indices)
    ]


def _host_sparse_data_array(X):
    data_dtype = np.dtype(X.data.dtype)
    if data_dtype == np.float64:
        return X.data
    if data_dtype == np.float32 or data_dtype.kind in {"b", "i", "u"}:
        return X.data.astype(np.float32, copy=False)
    if data_dtype.kind == "c":
        msg = (
            "Wilcoxon sparse input data dtype must be real; complex sparse "
            "data is not supported."
        )
        raise TypeError(msg)
    msg = (
        "Wilcoxon sparse input data dtype must be float32, float64, bool, "
        f"or integer; got {data_dtype}."
    )
    raise TypeError(msg)


def _validate_wilcoxon_sparse_dtype(X) -> None:
    if not (sp.issparse(X) or cpsp.issparse(X)):
        return
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
    """Prepare device-sparse arrays for the Wilcoxon kernels.

    Wilcoxon ranking sorts float32 keys on every path -- the sparse fast paths
    AND the dense fallback (``_ovr_dense_block_f32``); the CUB segmented
    sort is float-keyed throughout. Casting ``X.data`` to float32 here therefore
    does not diverge from any float64 ranking path, because there is none. This
    only loses precision when preprocessing ran in float64; float32-preprocessed
    values (even if later stored as float64) are float32-exact, so ranking
    matches scanpy bit-for-bit (~1e-13). For a fully float64 pipeline the
    rank-derived scores/p-values match scanpy-on-float64 to ~1e-4 on
    log-normalized data (below any significance threshold, no DE calls change),
    while means and log fold changes are still computed in float64. See the
    ``rank_genes_groups`` note on ranking precision. float64 input is accepted
    to spare the caller a pre-cast.
    """
    data_dtype = np.dtype(X.data.dtype)
    if data_dtype == np.float32:
        data = X.data
    elif data_dtype == np.float64:
        data = X.data.astype(cp.float32, copy=False)
    elif data_dtype.kind == "c":
        msg = (
            "Wilcoxon device sparse input data dtype must be real; complex "
            "sparse data is not supported."
        )
        raise TypeError(msg)
    else:
        msg = (
            "Wilcoxon device sparse input data dtype must be float32 or "
            f"float64; got {data_dtype}."
        )
        raise TypeError(msg)

    # Keep int64 index buffers native and let the nanobind overloads dispatch by
    # dtype. Normal CuPy sparse matrices keep indices and indptr in lockstep.
    if X.indices.dtype == cp.int64:
        indices = X.indices
        indptr = X.indptr
    else:
        indices = X.indices.astype(cp.int32, copy=False)
        indptr = X.indptr.astype(cp.int32, copy=False)
    return data, indices, indptr


def wilcoxon(
    rg: _RankGenes,
    *,
    tie_correct: bool,
    use_continuity: bool = False,
    chunk_size: int | None = None,
    return_u_values: bool = False,
) -> list[tuple[int, NDArray, NDArray]]:
    """Compute Wilcoxon rank-sum test statistics."""
    # Host dense OVR and OVO stream column windows from host. Already-device
    # dense OVO still uses the device-resident tiered planner.
    # Aggregate if on GPU, else defer to chunks.
    X = rg.X
    _validate_wilcoxon_sparse_dtype(X)
    rg._basic_stats()
    n_cells, n_total_genes = rg.X.shape
    group_sizes = rg.group_sizes

    if rg.ireference is not None:
        return _wilcoxon_with_reference(
            rg,
            X,
            n_total_genes,
            group_sizes,
            tie_correct=tie_correct,
            use_continuity=use_continuity,
            chunk_size=chunk_size,
            return_u_values=return_u_values,
        )
    return _wilcoxon_vs_rest(
        rg,
        X,
        n_cells,
        n_total_genes,
        group_sizes,
        tie_correct=tie_correct,
        use_continuity=use_continuity,
        chunk_size=chunk_size,
        return_u_values=return_u_values,
    )


def _host_sparse_format(X, *, sparse_negative_fallback: bool) -> str | None:
    if sparse_negative_fallback or not isinstance(X, sp.spmatrix | sp.sparray):
        return None
    if X.format not in {"csr", "csc"}:
        raise TypeError(
            "Wilcoxon sparse input must be CSR or CSC; refusing hidden "
            f"full-matrix conversion from {X.format!r}."
        )
    return X.format


def _device_sparse_format(X, *, sparse_negative_fallback: bool) -> str | None:
    if sparse_negative_fallback:
        return None
    if cpsp.isspmatrix_csc(X):
        return "csc"
    if cpsp.isspmatrix_csr(X):
        return "csr"
    return None


def _host_dense_matrix(X) -> np.ndarray | None:
    if not isinstance(X, np.ndarray):
        return None
    matrix = X
    if matrix.dtype.kind != "f" or matrix.dtype.itemsize < 4:
        return np.asarray(matrix, dtype=np.float32, order="F")
    if matrix.flags.c_contiguous or matrix.flags.f_contiguous:
        return matrix
    return np.asfortranarray(matrix)


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


def _warn_small_ovo_groups(
    rg: _RankGenes, ctx: _OvoContext, group_sizes: NDArray
) -> None:
    small_groups = [
        str(rg.groups_order[group_index])
        for group_index in ctx.test_group_indices
        if int(group_sizes[group_index]) <= MIN_GROUP_SIZE_WARNING
    ]
    if ctx.n_ref > MIN_GROUP_SIZE_WARNING and not small_groups:
        return
    parts = []
    if small_groups:
        parts.append(
            f"{len(small_groups)} test group(s) have size "
            f"<= {MIN_GROUP_SIZE_WARNING} (first few: "
            f"{', '.join(small_groups[:5])}"
            f"{'...' if len(small_groups) > 5 else ''})"
        )
    if ctx.n_ref <= MIN_GROUP_SIZE_WARNING:
        parts.append(f"reference has size {ctx.n_ref}")
    warnings.warn(
        f"Small groups detected: {'; '.join(parts)}. normal approximation "
        "of the Wilcoxon statistic may be inaccurate.",
        RuntimeWarning,
        stacklevel=4,
    )


def _build_ovo_context(rg: _RankGenes, group_sizes: NDArray) -> _OvoContext:
    codes = rg.group_codes
    n_groups = len(rg.groups_order)
    ireference = int(rg.ireference)
    n_ref = int(group_sizes[ireference])
    ref_row_ids = np.flatnonzero(codes == ireference).astype(np.int32, copy=False)
    test_group_indices = [i for i in range(n_groups) if i != ireference]

    offsets = [0]
    row_id_parts = []
    for group_index in test_group_indices:
        group_rows = np.flatnonzero(codes == group_index).astype(np.int32, copy=False)
        row_id_parts.append(group_rows)
        offsets.append(offsets[-1] + int(group_rows.size))

    all_grp_row_ids = (
        np.concatenate(row_id_parts).astype(np.int32, copy=False)
        if row_id_parts
        else np.empty(0, dtype=np.int32)
    )
    offsets_np = np.asarray(offsets, dtype=np.int32)
    test_sizes = cp.asarray(
        group_sizes[np.asarray(test_group_indices, dtype=np.intp)].astype(
            np.float64, copy=False
        )
    )
    return _OvoContext(
        codes=codes,
        n_groups=n_groups,
        ireference=ireference,
        n_ref=n_ref,
        ref_row_ids=ref_row_ids,
        test_group_indices=test_group_indices,
        all_grp_row_ids=all_grp_row_ids,
        offsets_np=offsets_np,
        offsets_gpu=cp.asarray(offsets_np),
        n_all_grp=int(all_grp_row_ids.size),
        n_test=len(test_group_indices),
        test_sizes=test_sizes,
    )


def _finish_ovo_sparse_stats(
    rg: _RankGenes,
    ctx: _OvoContext,
    group_sums: cp.ndarray,
    group_nnz: cp.ndarray,
    group_sizes: NDArray,
) -> cp.ndarray | None:
    if not rg._compute_stats_in_chunks:
        return None
    if rg._store_wilcoxon_gpu_result and not rg.comp_pts:
        rg._compute_stats_in_chunks = False
        return _ovo_logfoldchanges_from_sums(
            rg,
            group_sums,
            ctx.test_sizes,
            ctx.n_ref,
        )
    _fill_ovo_stats_from_accumulators(
        rg,
        group_sums,
        group_nnz,
        group_sizes=group_sizes,
        test_group_indices=ctx.test_group_indices,
        n_ref=ctx.n_ref,
    )
    return None


def _finish_ovo_dense_stats(
    rg: _RankGenes,
    ctx: _OvoContext,
    group_sums: cp.ndarray,
    group_sum_sq: cp.ndarray,
    group_nnz: cp.ndarray,
    *,
    group_sizes: NDArray,
) -> cp.ndarray | None:
    if not rg._compute_stats_in_chunks:
        return None
    if rg._store_wilcoxon_gpu_result and not rg.comp_pts:
        rg._compute_stats_in_chunks = False
        return _ovo_logfoldchanges_from_sums(
            rg,
            group_sums,
            ctx.test_sizes,
            ctx.n_ref,
        )
    _fill_ovo_dense_stats_from_accumulators(
        rg,
        group_sums,
        group_sum_sq,
        group_nnz,
        group_sizes=group_sizes,
        test_group_indices=ctx.test_group_indices,
        n_ref=ctx.n_ref,
    )
    return None


def _run_ovr_host_sparse(
    rg: _RankGenes,
    X,
    n_cells: int,
    n_total_genes: int,
    group_sizes: NDArray,
    *,
    tie_correct: bool,
    use_continuity: bool,
    return_u_values: bool,
) -> list[tuple[int, NDArray, NDArray]] | None:
    sparse_format = _host_sparse_format(
        X, sparse_negative_fallback=rg._sparse_negative_fallback
    )
    if sparse_format is None:
        return None

    n_groups = len(rg.groups_order)
    group_codes = rg.group_codes.astype(np.int32, copy=False)
    group_sizes_np = group_sizes.astype(np.float64, copy=False)
    group_sizes_dev = cp.asarray(group_sizes_np, dtype=cp.float64)
    rest_sizes = n_cells - group_sizes_dev
    compute_nnz = rg.comp_pts
    rank_sums = cp.empty((n_groups, n_total_genes), dtype=cp.float64)
    tie_corr = cp.ones(n_total_genes, dtype=cp.float64)
    group_sums = cp.empty((n_groups, n_total_genes), dtype=cp.float64)
    group_nnz = cp.empty(
        (n_groups, n_total_genes) if compute_nnz else (1, 1),
        dtype=cp.float64,
    )
    compute_totals = bool(
        rg._compute_stats_in_chunks and np.any(group_codes == n_groups)
    )
    total_sums = cp.empty(
        (1, n_total_genes) if compute_totals else (1, 1),
        dtype=cp.float64,
    )
    total_nnz = cp.empty(
        (1, n_total_genes) if (compute_totals and compute_nnz) else (1, 1),
        dtype=cp.float64,
    )

    if isinstance(X, sp.spmatrix | sp.sparray) and X.format == "csc":
        X.sort_indices()
        _wcs.ovr_sparse_csc_host(
            _host_sparse_data_array(X),
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
            n_rows=n_cells,
            n_cols=n_total_genes,
            n_groups=n_groups,
            compute_tie_corr=tie_correct,
            compute_nnz=compute_nnz,
            compute_totals=compute_totals,
            sub_batch_cols=OVR_HOST_CSC_SUB_BATCH,
        )
    else:
        X.sort_indices()
        _wcs.ovr_sparse_csr_host(
            _host_sparse_data_array(X),
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
            n_rows=n_cells,
            n_cols=n_total_genes,
            n_groups=n_groups,
            compute_tie_corr=tie_correct,
            compute_nnz=compute_nnz,
            compute_totals=compute_totals,
            sub_batch_cols=OVR_HOST_CSR_SUB_BATCH,
        )

    if rg._compute_stats_in_chunks:
        _fill_basic_stats_from_accumulators(
            rg,
            group_sums,
            group_nnz,
            group_sizes_np,
            n_cells=n_cells,
            total_sums=total_sums if compute_totals else None,
            total_nnz=total_nnz if compute_totals and compute_nnz else None,
        )

    return _finish_ovr(
        rank_sums,
        group_sizes_dev,
        rest_sizes,
        n_cells,
        tie_corr,
        use_continuity=use_continuity,
        return_u_values=return_u_values,
        n_groups=n_groups,
    )


def _run_ovr_device_sparse(
    rg: _RankGenes,
    X,
    n_cells: int,
    n_total_genes: int,
    group_sizes: NDArray,
    *,
    tie_correct: bool,
    use_continuity: bool,
    return_u_values: bool,
) -> list[tuple[int, NDArray, NDArray]] | None:
    sparse_format = _device_sparse_format(
        X, sparse_negative_fallback=rg._sparse_negative_fallback
    )
    if sparse_format is None:
        return None

    X.sort_indices()
    data, indices, indptr = _device_sparse_arrays(X)
    n_groups = len(rg.groups_order)
    group_codes_gpu = cp.asarray(rg.group_codes, dtype=cp.int32)
    group_sizes_dev = cp.asarray(group_sizes, dtype=cp.float64)
    rest_sizes = n_cells - group_sizes_dev
    rank_sums = cp.empty((n_groups, n_total_genes), dtype=cp.float64)
    tie_corr = cp.ones(n_total_genes, dtype=cp.float64)

    if sparse_format == "csc":
        _wcs.ovr_sparse_csc_device(
            data,
            indices,
            indptr,
            group_codes_gpu,
            group_sizes_dev,
            rank_sums,
            tie_corr,
            n_rows=n_cells,
            n_cols=n_total_genes,
            n_groups=n_groups,
            compute_tie_corr=tie_correct,
            sub_batch_cols=OVR_DEVICE_CSC_SUB_BATCH,
        )
    else:
        _wcs.ovr_sparse_csr_device(
            data,
            indices,
            indptr,
            group_codes_gpu,
            group_sizes_dev,
            rank_sums,
            tie_corr,
            n_rows=n_cells,
            n_cols=n_total_genes,
            n_groups=n_groups,
            compute_tie_corr=tie_correct,
            sub_batch_cols=OVR_DEVICE_CSR_SUB_BATCH,
        )

    return _finish_ovr(
        rank_sums,
        group_sizes_dev,
        rest_sizes,
        n_cells,
        tie_corr,
        use_continuity=use_continuity,
        return_u_values=return_u_values,
        n_groups=n_groups,
    )


def _run_ovr_host_dense(
    rg: _RankGenes,
    X,
    n_cells: int,
    n_total_genes: int,
    group_sizes: NDArray,
    *,
    tie_correct: bool,
    use_continuity: bool,
    return_u_values: bool,
) -> list[tuple[int, NDArray, NDArray]] | None:
    matrix = _host_dense_matrix(X)
    if matrix is None:
        return None
    n_groups = len(rg.groups_order)
    group_codes_gpu = cp.asarray(rg.group_codes, dtype=cp.int32)
    group_sizes_dev = cp.asarray(group_sizes, dtype=cp.float64)
    rest_sizes = n_cells - group_sizes_dev
    compute_nnz = rg.comp_pts
    compute_stats = rg._compute_stats_in_chunks
    compute_totals = bool(compute_stats and np.any(rg.group_codes == n_groups))
    rank_sums = cp.empty((n_groups, n_total_genes), dtype=cp.float64)
    tie_corr = (
        cp.empty(n_total_genes, dtype=cp.float64)
        if tie_correct
        else cp.ones(n_total_genes, dtype=cp.float64)
    )
    stats_shape = (n_groups, n_total_genes) if compute_stats else (1, 1)
    group_sums = cp.empty(stats_shape, dtype=cp.float64)
    group_nnz = cp.empty(
        (n_groups, n_total_genes) if (compute_stats and compute_nnz) else (1, 1),
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
    _wc.ovr_rank_dense_host_streaming(
        matrix,
        group_codes_gpu,
        rank_sums,
        tie_corr,
        group_sums,
        group_nnz,
        total_sums,
        total_nnz,
        n_groups=n_groups,
        compute_tie_corr=tie_correct,
        compute_nnz=compute_stats and compute_nnz,
        compute_stats=compute_stats,
        compute_totals=compute_totals,
        sub_batch_cols=OVR_DENSE_SUB_BATCH,
    )
    if compute_stats:
        _fill_basic_stats_from_accumulators(
            rg,
            group_sums,
            group_nnz,
            group_sizes,
            n_cells=n_cells,
            total_sums=total_sums if compute_totals else None,
            total_nnz=total_nnz if compute_totals and compute_nnz else None,
        )
    return _finish_ovr(
        rank_sums,
        group_sizes_dev,
        rest_sizes,
        n_cells,
        tie_corr,
        use_continuity=use_continuity,
        return_u_values=return_u_values,
        n_groups=n_groups,
    )


def _run_ovr_dense_chunks(
    rg: _RankGenes,
    X,
    n_cells: int,
    n_total_genes: int,
    group_sizes: NDArray,
    *,
    tie_correct: bool,
    use_continuity: bool,
    chunk_size: int | None,
    return_u_values: bool,
) -> list[tuple[int, NDArray, NDArray]]:
    n_groups = len(rg.groups_order)
    chunk_width = _choose_wilcoxon_chunk_size(chunk_size, n_total_genes)
    group_codes_gpu = cp.asarray(rg.group_codes, dtype=cp.int32)
    group_sizes_dev = cp.asarray(group_sizes, dtype=cp.float64)
    rest_sizes = n_cells - group_sizes_dev
    all_scores: dict[int, list] = {i: [] for i in range(n_groups)}
    all_pvals: dict[int, list] = {i: [] for i in range(n_groups)}

    for start in range(0, n_total_genes, chunk_width):
        stop = min(start + chunk_width, n_total_genes)
        if rg._compute_stats_in_chunks:
            block = _get_column_block(X, start, stop)
            rg._accumulate_chunk_stats_vs_rest(
                block,
                start,
                stop,
                group_codes_dev=group_codes_gpu,
                group_sizes_dev=group_sizes_dev,
                n_cells=n_cells,
            )
            block_f32 = cp.asfortranarray(block.astype(cp.float32, copy=False))
        else:
            block_f32 = _ovr_dense_block_f32(X, start, stop)

        n_cols = stop - start
        rank_sums = cp.empty((n_groups, n_cols), dtype=cp.float64)
        tie_corr = (
            cp.empty(n_cols, dtype=cp.float64)
            if tie_correct
            else cp.ones(n_cols, dtype=cp.float64)
        )
        _wc.ovr_rank_dense_streaming(
            block_f32,
            group_codes_gpu,
            rank_sums,
            tie_corr,
            n_rows=n_cells,
            n_cols=n_cols,
            n_groups=n_groups,
            compute_tie_corr=tie_correct,
            sub_batch_cols=OVR_DENSE_SUB_BATCH,
            stream=cp.cuda.get_current_stream().ptr,
        )
        scores, p_values = _ovr_z_pvals(
            rank_sums,
            group_sizes_dev,
            rest_sizes,
            n_cells,
            tie_corr,
            use_continuity=use_continuity,
            return_u_values=return_u_values,
        )
        scores_host = scores.get()
        p_host = p_values.get()

        for idx in range(n_groups):
            all_scores[idx].append(scores_host[idx])
            all_pvals[idx].append(p_host[idx])

    return [
        (gi, np.concatenate(all_scores[gi]), np.concatenate(all_pvals[gi]))
        for gi in range(n_groups)
    ]


def _wilcoxon_vs_rest(
    rg: _RankGenes,
    X,
    n_cells: int,
    n_total_genes: int,
    group_sizes: NDArray,
    *,
    tie_correct: bool,
    use_continuity: bool,
    chunk_size: int | None,
    return_u_values: bool,
) -> list[tuple[int, NDArray, NDArray]]:
    """Wilcoxon test: each group vs rest of cells."""
    _warn_small_ovr_groups(rg, group_sizes, n_cells)
    for runner in (
        _run_ovr_host_sparse,
        _run_ovr_device_sparse,
        _run_ovr_host_dense,
    ):
        result = runner(
            rg,
            X,
            n_cells,
            n_total_genes,
            group_sizes,
            tie_correct=tie_correct,
            use_continuity=use_continuity,
            return_u_values=return_u_values,
        )
        if result is not None:
            return result
    return _run_ovr_dense_chunks(
        rg,
        X,
        n_cells,
        n_total_genes,
        group_sizes,
        tie_correct=tie_correct,
        use_continuity=use_continuity,
        chunk_size=chunk_size,
        return_u_values=return_u_values,
    )


def _run_ovo_host_sparse(
    rg: _RankGenes,
    X,
    ctx: _OvoContext,
    n_total_genes: int,
    group_sizes: NDArray,
    *,
    tie_correct: bool,
    use_continuity: bool,
    return_u_values: bool,
) -> list[tuple[int, NDArray, NDArray]] | None:
    sparse_format = _host_sparse_format(
        X, sparse_negative_fallback=rg._sparse_negative_fallback
    )
    if sparse_format is None:
        return None

    rank_sums = cp.zeros((ctx.n_test, n_total_genes), dtype=cp.float64)
    tie_corr_arr = cp.ones((ctx.n_test, n_total_genes), dtype=cp.float64)
    n_groups_stats = ctx.n_test + 1
    compute_sums = rg._compute_stats_in_chunks
    compute_nnz = rg.comp_pts
    group_sums = cp.empty(
        (n_groups_stats, n_total_genes)
        if (compute_sums or sparse_format == "csc")
        else (1,),
        dtype=cp.float64,
    )
    group_nnz = cp.empty(
        (n_groups_stats, n_total_genes) if compute_nnz else (1,),
        dtype=cp.float64,
    )
    stats_code_lookup = np.full(ctx.n_groups + 1, n_groups_stats, dtype=np.int32)
    test_group_indices_np = np.asarray(ctx.test_group_indices, dtype=np.intp)
    stats_code_lookup[test_group_indices_np] = np.arange(ctx.n_test, dtype=np.int32)
    stats_code_lookup[ctx.ireference] = ctx.n_test
    stats_codes = stats_code_lookup[ctx.codes]

    if sparse_format == "csc":
        X.sort_indices()
        ref_row_map = np.full(X.shape[0], -1, dtype=np.int32)
        ref_row_map[ctx.ref_row_ids] = np.arange(ctx.n_ref, dtype=np.int32)
        grp_row_map = np.full(X.shape[0], -1, dtype=np.int32)
        grp_row_map[ctx.all_grp_row_ids] = np.arange(ctx.n_all_grp, dtype=np.int32)
        _wcs.ovo_streaming_csc_host(
            _host_sparse_data_array(X),
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
            n_rows=X.shape[0],
            n_cols=n_total_genes,
            n_groups=ctx.n_test,
            n_groups_stats=n_groups_stats,
            compute_tie_corr=tie_correct,
            compute_nnz=compute_nnz,
            sub_batch_cols=OVO_HOST_SPARSE_SUB_BATCH,
        )
    else:
        X.sort_indices()
        _wcs.ovo_streaming_csr_host(
            _host_sparse_data_array(X),
            X.indices,
            X.indptr,
            ctx.ref_row_ids,
            ctx.all_grp_row_ids,
            ctx.offsets_np,
            rank_sums,
            tie_corr_arr,
            group_sums,
            group_nnz,
            n_full_rows=X.shape[0],
            n_ref=ctx.n_ref,
            n_all_grp=ctx.n_all_grp,
            n_cols=n_total_genes,
            n_test=ctx.n_test,
            n_groups_stats=n_groups_stats,
            compute_tie_corr=tie_correct,
            compute_nnz=compute_nnz,
            compute_sums=compute_sums,
            sub_batch_cols=OVO_HOST_SPARSE_SUB_BATCH,
        )

    logfoldchanges_gpu = _finish_ovo_sparse_stats(
        rg, ctx, group_sums, group_nnz, group_sizes
    )
    return _finish_ovo(
        rank_sums,
        ctx.test_sizes,
        ctx.n_ref,
        tie_corr_arr,
        tie_correct=tie_correct,
        use_continuity=use_continuity,
        return_u_values=return_u_values,
        rg=rg,
        test_group_indices=ctx.test_group_indices,
        logfoldchanges_gpu=logfoldchanges_gpu,
    )


def _run_ovo_device_sparse(
    rg: _RankGenes,
    X,
    ctx: _OvoContext,
    n_total_genes: int,
    _group_sizes: NDArray,
    *,
    tie_correct: bool,
    use_continuity: bool,
    return_u_values: bool,
) -> list[tuple[int, NDArray, NDArray]] | None:
    sparse_format = _device_sparse_format(
        X, sparse_negative_fallback=rg._sparse_negative_fallback
    )
    if sparse_format is None:
        return None

    if isinstance(X, cpsp.spmatrix) and X.format == "csr":
        X.sort_indices()
    data, indices, indptr = _device_sparse_arrays(X)
    rank_sums = cp.zeros((ctx.n_test, n_total_genes), dtype=cp.float64)
    tie_corr_arr = cp.ones((ctx.n_test, n_total_genes), dtype=cp.float64)

    if sparse_format == "csc":
        ref_row_map = np.full(X.shape[0], -1, dtype=np.int32)
        ref_row_map[ctx.ref_row_ids] = np.arange(ctx.n_ref, dtype=np.int32)
        grp_row_map = np.full(X.shape[0], -1, dtype=np.int32)
        grp_row_map[ctx.all_grp_row_ids] = np.arange(ctx.n_all_grp, dtype=np.int32)
        _wcs.ovo_streaming_csc_device(
            data,
            indices,
            indptr,
            cp.asarray(ref_row_map),
            cp.asarray(grp_row_map),
            ctx.offsets_gpu,
            rank_sums,
            tie_corr_arr,
            n_ref=ctx.n_ref,
            n_all_grp=ctx.n_all_grp,
            n_cols=n_total_genes,
            n_groups=ctx.n_test,
            compute_tie_corr=tie_correct,
            sub_batch_cols=OVO_DEVICE_SPARSE_SUB_BATCH,
        )
    else:
        _wcs.ovo_streaming_csr_device(
            data,
            indices,
            indptr,
            cp.asarray(ctx.ref_row_ids, dtype=cp.int32),
            cp.asarray(ctx.all_grp_row_ids, dtype=cp.int32),
            ctx.offsets_gpu,
            rank_sums,
            tie_corr_arr,
            n_ref=ctx.n_ref,
            n_all_grp=ctx.n_all_grp,
            n_cols=n_total_genes,
            n_groups=ctx.n_test,
            compute_tie_corr=tie_correct,
            sub_batch_cols=OVO_DEVICE_SPARSE_SUB_BATCH,
        )

    return _finish_ovo(
        rank_sums,
        ctx.test_sizes,
        ctx.n_ref,
        tie_corr_arr,
        tie_correct=tie_correct,
        use_continuity=use_continuity,
        return_u_values=return_u_values,
        rg=rg,
        test_group_indices=ctx.test_group_indices,
        logfoldchanges_gpu=None,
    )


def _run_ovo_host_dense(
    rg: _RankGenes,
    X,
    ctx: _OvoContext,
    n_total_genes: int,
    group_sizes: NDArray,
    *,
    tie_correct: bool,
    use_continuity: bool,
    chunk_size: int | None,
    return_u_values: bool,
) -> list[tuple[int, NDArray, NDArray]] | None:
    matrix = _host_dense_matrix(X)
    if matrix is None:
        return None
    dense_sub_batch_cols = (
        _choose_wilcoxon_chunk_size(chunk_size, n_total_genes)
        if chunk_size is not None
        else OVO_DENSE_TIERED_SUB_BATCH
    )
    rank_sums = cp.zeros((ctx.n_test, n_total_genes), dtype=cp.float64)
    tie_corr_arr = cp.ones((ctx.n_test, n_total_genes), dtype=cp.float64)
    compute_stats = rg._compute_stats_in_chunks
    compute_nnz = compute_stats and rg.comp_pts
    n_groups_stats = ctx.n_test + 1
    stats_shape = (n_groups_stats, n_total_genes) if compute_stats else (1, 1)
    group_sums = cp.empty(stats_shape, dtype=cp.float64)
    group_sum_sq = cp.empty(stats_shape, dtype=cp.float64)
    group_nnz = cp.empty(
        stats_shape if compute_nnz else (1, 1),
        dtype=cp.float64,
    )
    _wc.ovo_rank_dense_host_streaming(
        matrix,
        ctx.ref_row_ids,
        ctx.all_grp_row_ids,
        ctx.offsets_np,
        rank_sums,
        tie_corr_arr,
        group_sums,
        group_sum_sq,
        group_nnz,
        n_groups=ctx.n_test,
        compute_tie_corr=tie_correct,
        compute_nnz=compute_nnz,
        compute_stats=compute_stats,
        sub_batch_cols=dense_sub_batch_cols,
    )
    logfoldchanges_gpu = _finish_ovo_dense_stats(
        rg,
        ctx,
        group_sums,
        group_sum_sq,
        group_nnz,
        group_sizes=group_sizes,
    )
    return _finish_ovo(
        rank_sums,
        ctx.test_sizes,
        ctx.n_ref,
        tie_corr_arr,
        tie_correct=tie_correct,
        use_continuity=use_continuity,
        return_u_values=return_u_values,
        rg=rg,
        test_group_indices=ctx.test_group_indices,
        logfoldchanges_gpu=logfoldchanges_gpu,
    )


def _run_ovo_dense_chunks(
    rg: _RankGenes,
    X,
    ctx: _OvoContext,
    n_total_genes: int,
    group_sizes: NDArray,
    *,
    tie_correct: bool,
    use_continuity: bool,
    chunk_size: int | None,
    return_u_values: bool,
) -> list[tuple[int, NDArray, NDArray]]:
    chunk_width = _choose_wilcoxon_chunk_size(chunk_size, n_total_genes)
    scores_host = np.empty((ctx.n_test, n_total_genes), dtype=np.float64)
    pvals_host = np.empty((ctx.n_test, n_total_genes), dtype=np.float64)

    for start in range(0, n_total_genes, chunk_width):
        stop = min(start + chunk_width, n_total_genes)
        n_cols = stop - start
        ref_block = _ovo_dense_block(X, ctx.ref_row_ids, start, stop)
        grp_block = _ovo_dense_block(X, ctx.all_grp_row_ids, start, stop)

        _fill_ovo_chunk_stats(
            rg,
            ref_block,
            grp_block,
            offsets=ctx.offsets_np,
            test_group_indices=ctx.test_group_indices,
            start=start,
            stop=stop,
            group_sizes=group_sizes,
        )

        ref_f32 = cp.asarray(ref_block, dtype=cp.float32, order="F")
        grp_f32 = cp.asarray(grp_block, dtype=cp.float32, order="F")
        rank_sums = cp.zeros((ctx.n_test, n_cols), dtype=cp.float64)
        tie_corr = cp.ones((ctx.n_test, n_cols), dtype=cp.float64)
        _wc.ovo_rank_dense_tiered_unsorted_ref(
            ref_f32,
            grp_f32,
            ctx.offsets_gpu,
            rank_sums,
            tie_corr,
            n_ref=ctx.n_ref,
            n_all_grp=ctx.n_all_grp,
            n_cols=n_cols,
            n_groups=ctx.n_test,
            compute_tie_corr=tie_correct,
            sub_batch_cols=OVO_DENSE_TIERED_SUB_BATCH,
            stream=cp.cuda.get_current_stream().ptr,
        )
        scores, p_values = _ovo_z_pvals(
            rank_sums,
            ctx.test_sizes,
            ctx.n_ref,
            tie_corr,
            tie_correct=tie_correct,
            use_continuity=use_continuity,
            return_u_values=return_u_values,
        )
        scores_host[:, start:stop] = scores.get()
        pvals_host[:, start:stop] = p_values.get()

    return [
        (group_index, scores_host[slot], pvals_host[slot])
        for slot, group_index in enumerate(ctx.test_group_indices)
    ]


def _wilcoxon_with_reference(
    rg: _RankGenes,
    X,
    n_total_genes: int,
    group_sizes: NDArray,
    *,
    tie_correct: bool,
    use_continuity: bool,
    chunk_size: int | None,
    return_u_values: bool,
) -> list[tuple[int, NDArray, NDArray]]:
    """Wilcoxon test: all selected groups vs a specific reference group."""
    ctx = _build_ovo_context(rg, group_sizes)
    if ctx.n_test == 0:
        return []
    _warn_small_ovo_groups(rg, ctx, group_sizes)
    for runner, extra in (
        (_run_ovo_host_sparse, {}),
        (_run_ovo_device_sparse, {}),
        (_run_ovo_host_dense, {"chunk_size": chunk_size}),
    ):
        result = runner(
            rg,
            X,
            ctx,
            n_total_genes,
            group_sizes,
            tie_correct=tie_correct,
            use_continuity=use_continuity,
            return_u_values=return_u_values,
            **extra,
        )
        if result is not None:
            return result
    return _run_ovo_dense_chunks(
        rg,
        X,
        ctx,
        n_total_genes,
        group_sizes,
        tie_correct=tie_correct,
        use_continuity=use_continuity,
        chunk_size=chunk_size,
        return_u_values=return_u_values,
    )
