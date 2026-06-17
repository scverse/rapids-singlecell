from __future__ import annotations

import warnings
from typing import TYPE_CHECKING

import cupy as cp
import cupyx.scipy.sparse as cpsp
import cupyx.scipy.special as cupyx_special
import numpy as np
import scipy.sparse as sp

from rapids_singlecell._cuda import _wilcoxon_cuda as _wc
from rapids_singlecell._cuda import _wilcoxon_sparse_cuda as _wcs

from ._utils import EPS, MIN_GROUP_SIZE_WARNING, _choose_chunk_size, _get_column_block

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
DENSE_HOST_PRELOAD_MAX_GPU_FRACTION = 0.55  # leave headroom for rank buffers


def _maybe_preload_host_dense(rg: _RankGenes) -> None:
    """Preload moderate host-dense matrices to avoid repeated chunk transfers."""
    X = rg.X
    if not isinstance(X, np.ndarray) or X.size == 0:
        return

    try:
        _, total = cp.cuda.runtime.memGetInfo()
    except cp.cuda.runtime.CUDARuntimeError:
        return

    if X.nbytes > total * DENSE_HOST_PRELOAD_MAX_GPU_FRACTION:
        return

    registered = False
    if X.flags.c_contiguous or X.flags.f_contiguous:
        try:
            cp.cuda.runtime.hostRegister(X.ctypes.data, X.nbytes, 0)
            registered = True
        except cp.cuda.runtime.CUDARuntimeError:
            registered = False

    try:
        X_gpu = cp.asarray(X)
        cp.cuda.get_current_stream().synchronize()
    except cp.cuda.memory.OutOfMemoryError:
        cp.get_default_memory_pool().free_all_blocks()
        return
    except cp.cuda.runtime.CUDARuntimeError:
        return
    finally:
        if registered:
            try:
                cp.cuda.runtime.hostUnregister(X.ctypes.data)
            except cp.cuda.runtime.CUDARuntimeError:
                pass
    rg.X = X_gpu


def _get_dense_column_block_f32(X, start: int, stop: int) -> cp.ndarray:
    """Extract a dense column block as F-order float32 CuPy memory.

    For sparse X (the negative-values dense fallback) the column window is
    densified on the fly via the shared CSR/CSC densify path, so no full-matrix
    dense materialization happens.
    """
    if isinstance(X, np.ndarray | cp.ndarray):
        return cp.asarray(X[:, start:stop], dtype=cp.float32, order="F")
    if sp.issparse(X) or cpsp.issparse(X):
        block = _get_column_block(X, start, stop)  # float64 F-order chunk
        return cp.asfortranarray(block.astype(cp.float32, copy=False))
    raise TypeError(f"Expected dense matrix, got {type(X)}")


def _extract_dense_rows_cols(
    X, row_ids: np.ndarray, start: int, stop: int
) -> cp.ndarray:
    """Extract a bounded row/column block as F-order CuPy dense memory."""
    if isinstance(X, np.ndarray):
        return cp.asarray(X[row_ids, start:stop], order="F")
    if isinstance(X, cp.ndarray):
        rows = cp.asarray(row_ids, dtype=cp.int32)
        return cp.asfortranarray(X[rows, start:stop])
    if isinstance(X, sp.spmatrix | sp.sparray):
        return cp.asarray(X[row_ids][:, start:stop].toarray(), order="F")
    if cpsp.issparse(X):
        rows = cp.asarray(row_ids, dtype=cp.int32)
        return cp.asfortranarray(X[rows][:, start:stop].toarray())
    raise TypeError(f"Unsupported matrix type: {type(X)}")


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
    # Wilcoxon does not output per-group variance; vars are left zero (real
    # group means/pts come from group_sums/group_nnz, which drive the lfc/pts
    # output and the rank test).
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


def _host_sparse_fn_and_arrays(module, base_name: str, X):
    data_dtype = np.dtype(X.data.dtype)
    if data_dtype == np.float64:
        is_f64 = True
        data_arr = X.data
    elif data_dtype == np.float32 or data_dtype.kind in {"b", "i", "u"}:
        is_f64 = False
        data_arr = X.data.astype(np.float32, copy=False)
    else:
        msg = (
            "Wilcoxon sparse input data dtype must be float32, float64, bool, "
            f"or integer; got {data_dtype}."
        )
        raise TypeError(msg)

    # Row/column indices always fit int32 (cells and genes are < 2^31); only the
    # indptr (cumulative nnz) can need int64. Mirrors the rest of the sparse code.
    is_i64 = X.indptr.dtype == np.int64
    # The *_f64 binding only changes the host pointer dtype so float64 data can
    # be passed without a host-side copy; it still ranks in float32 on-device
    # (the kernels cast InT -> float before the segmented sort). See
    # _device_sparse_arrays_f32 for why float32 ranking is the uniform design.
    suffix = ""
    if is_f64:
        suffix += "_f64"
    if is_i64:
        suffix += "_i64"
    # int64 column indices: if a native-int64 binding exists for this path, use
    # it and pass the indices as-is. astype(int32) on int64 indices materializes
    # a full copy of every nonzero (~nnz * 4 bytes, e.g. tens of GB on large
    # matrices), so avoid it when the kernel can read int64 directly.
    if X.indices.dtype == np.int64:
        idx_fn = getattr(module, base_name + suffix + "_idx64", None)
        if idx_fn is not None:
            return idx_fn, data_arr, X.indices
    fn = getattr(module, base_name + suffix)
    indices_arr = X.indices.astype(np.int32, copy=False)
    return fn, data_arr, indices_arr


def _device_sparse_arrays_f32(X):
    """Cast device-sparse arrays for the Wilcoxon kernels.

    Wilcoxon ranking sorts float32 keys on every path -- the sparse fast paths
    AND the dense fallback (``_get_dense_column_block_f32``); the CUB segmented
    sort is float-keyed throughout. Casting ``X.data`` to float32 here therefore
    does not diverge from any float64 ranking path, because there is none. For
    count data float32 is exact (integer values < 2**24) and scanpy parity holds
    at 1e-13. float64 input is accepted only to spare the caller a pre-cast.
    """
    data_dtype = np.dtype(X.data.dtype)
    if data_dtype == np.float32 or data_dtype == np.float64:
        pass
    elif data_dtype.kind in {"b", "i", "u"}:
        pass
    else:
        msg = (
            "Wilcoxon device sparse input data dtype must be float32, float64, "
            f"bool, or integer; got {data_dtype}."
        )
        raise TypeError(msg)

    data = X.data.astype(cp.float32, copy=False)
    # Row/column indices fit int32 (cells and genes are < 2^31); indptr
    # (cumulative nnz) may need int64, which the *_i64 device kernels handle.
    indices = X.indices.astype(cp.int32, copy=False)
    if X.indptr.dtype == cp.int64:
        indptr = X.indptr
    else:
        indptr = X.indptr.astype(cp.int32, copy=False)
    return data, indices, indptr


def _device_sparse_fn(module, base_name: str, indptr: cp.ndarray):
    """Select the device kernel binding, using the int64-indptr variant if needed."""
    suffix = "_i64" if indptr.dtype == cp.int64 else ""
    return getattr(module, base_name + suffix)


def _column_totals_for_host_matrix(
    X, *, compute_nnz: bool
) -> tuple[cp.ndarray, cp.ndarray | None]:
    n_cols = X.shape[1]
    if isinstance(X, sp.spmatrix | sp.sparray):
        data = np.asarray(X.data)
        values = data.astype(np.float64, copy=False)
        if X.format == "csc":
            indptr = np.asarray(X.indptr)
            counts = np.diff(indptr)
            nonempty = counts > 0
            starts = indptr[:-1][nonempty]
            sums = np.zeros(n_cols, dtype=np.float64)
            if starts.size:
                sums[nonempty] = np.add.reduceat(values, starts)
            nnz = None
            if compute_nnz:
                nnz = np.zeros(n_cols, dtype=np.float64)
                if starts.size:
                    nnz[nonempty] = np.add.reduceat(
                        (data != 0).astype(np.float64, copy=False), starts
                    )
        elif X.format == "csr":
            indices = np.asarray(X.indices, dtype=np.intp)
            sums = np.bincount(indices, weights=values, minlength=n_cols).astype(
                np.float64, copy=False
            )
            nnz = (
                np.bincount(
                    indices,
                    weights=(data != 0).astype(np.float64, copy=False),
                    minlength=n_cols,
                ).astype(np.float64, copy=False)
                if compute_nnz
                else None
            )
        else:
            raise TypeError(
                "Wilcoxon sparse input must be CSR or CSC; refusing hidden "
                f"full-matrix conversion from {X.format!r}."
            )
    else:
        raise TypeError(f"Unsupported host matrix type: {type(X)}")

    total_sums = cp.asarray(sums.reshape(1, n_cols), dtype=cp.float64)
    total_nnz = (
        cp.asarray(nnz.reshape(1, n_cols), dtype=cp.float64)
        if nnz is not None
        else None
    )
    return total_sums, total_nnz


def _host_ovr_totals_if_needed(
    X,
    group_codes: np.ndarray,
    n_groups: int,
    *,
    compute_nnz: bool,
) -> tuple[cp.ndarray | None, cp.ndarray | None]:
    if not np.any(group_codes == n_groups):
        return None, None
    return _column_totals_for_host_matrix(X, compute_nnz=compute_nnz)


def wilcoxon(
    rg: _RankGenes,
    *,
    tie_correct: bool,
    use_continuity: bool = False,
    chunk_size: int | None = None,
    return_u_values: bool = False,
) -> list[tuple[int, NDArray, NDArray]]:
    """Compute Wilcoxon rank-sum test statistics."""
    _maybe_preload_host_dense(rg)
    # Compute basic stats - uses Aggregate if on GPU, else defers to chunks
    rg._basic_stats()
    X = rg.X
    n_cells, n_total_genes = rg.X.shape
    group_sizes = rg.group_sizes

    if rg.ireference is not None:
        # Compare each group against a specific reference group
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
    # Compare each group against "rest" (all other cells)
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
    n_groups = len(rg.groups_order)

    # Warn for small groups
    for name, size in zip(rg.groups_order, group_sizes, strict=True):
        rest = n_cells - size
        if size <= MIN_GROUP_SIZE_WARNING or rest <= MIN_GROUP_SIZE_WARNING:
            warnings.warn(
                f"Group {name} has size {size} (rest {rest}); normal approximation "
                "of the Wilcoxon statistic may be inaccurate.",
                RuntimeWarning,
                stacklevel=4,
            )

    host_sparse = (
        isinstance(X, sp.spmatrix | sp.sparray) and not rg._sparse_negative_fallback
    )
    if host_sparse:
        if X.format not in {"csr", "csc"}:
            raise TypeError(
                "Wilcoxon sparse input must be CSR or CSC; refusing hidden "
                f"full-matrix conversion from {X.format!r}."
            )

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

        if X.format == "csc":
            csc = X
            if not csc.has_sorted_indices:
                csc = csc.copy()
                csc.sort_indices()
            csc_host_fn, data_arr, indices_arr = _host_sparse_fn_and_arrays(
                _wcs, "ovr_sparse_csc_host", csc
            )
            csc_host_fn(
                data_arr,
                indices_arr,
                csc.indptr,
                group_codes,
                group_sizes_np,
                rank_sums,
                tie_corr,
                group_sums,
                group_nnz,
                n_rows=n_cells,
                n_cols=n_total_genes,
                n_groups=n_groups,
                compute_tie_corr=tie_correct,
                compute_nnz=compute_nnz,
                sub_batch_cols=OVR_HOST_CSC_SUB_BATCH,
            )
        else:
            csr = X
            if not csr.has_sorted_indices:
                csr = csr.copy()
                csr.sort_indices()
            csr_host_fn, data_arr, indices_arr = _host_sparse_fn_and_arrays(
                _wcs, "ovr_sparse_csr_host", csr
            )
            csr_host_fn(
                data_arr,
                indices_arr,
                csr.indptr,
                group_codes,
                group_sizes_np,
                rank_sums,
                tie_corr,
                group_sums,
                group_nnz,
                n_rows=n_cells,
                n_cols=n_total_genes,
                n_groups=n_groups,
                compute_tie_corr=tie_correct,
                compute_nnz=compute_nnz,
                sub_batch_cols=OVR_HOST_CSR_SUB_BATCH,
            )

        if rg._compute_stats_in_chunks:
            total_sums, total_nnz = _host_ovr_totals_if_needed(
                X,
                group_codes,
                n_groups,
                compute_nnz=compute_nnz,
            )
            _fill_basic_stats_from_accumulators(
                rg,
                group_sums,
                group_nnz,
                group_sizes_np,
                n_cells=n_cells,
                total_sums=total_sums,
                total_nnz=total_nnz,
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
        return [(gi, scores_host[gi], p_host[gi]) for gi in range(n_groups)]

    if (
        cpsp.isspmatrix_csc(X) or cpsp.isspmatrix_csr(X)
    ) and not rg._sparse_negative_fallback:
        data, indices, indptr = _device_sparse_arrays_f32(X)
        group_codes_gpu = cp.asarray(rg.group_codes, dtype=cp.int32)
        group_sizes_dev = cp.asarray(group_sizes, dtype=cp.float64)
        rest_sizes = n_cells - group_sizes_dev
        rank_sums = cp.empty((n_groups, n_total_genes), dtype=cp.float64)
        tie_corr = cp.ones(n_total_genes, dtype=cp.float64)
        if cpsp.isspmatrix_csc(X):
            _device_sparse_fn(_wcs, "ovr_sparse_csc_device", indptr)(
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
            sparse_X = X
            if not sparse_X.has_sorted_indices:
                sparse_X = sparse_X.copy()
                sparse_X.sort_indices()
                data, indices, indptr = _device_sparse_arrays_f32(sparse_X)
            _device_sparse_fn(_wcs, "ovr_sparse_csr_device", indptr)(
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

    group_codes_gpu = cp.asarray(rg.group_codes, dtype=cp.int32)

    group_sizes_dev = cp.asarray(group_sizes, dtype=cp.float64)
    rest_sizes = n_cells - group_sizes_dev

    chunk_width = _choose_wilcoxon_chunk_size(chunk_size, n_total_genes)

    # Accumulate results per group
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
            block_f32 = _get_dense_column_block_f32(X, start, stop)

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

    # Collect results per group
    return [
        (gi, np.concatenate(all_scores[gi]), np.concatenate(all_pvals[gi]))
        for gi in range(n_groups)
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
    codes = rg.group_codes
    n_groups = len(rg.groups_order)
    ireference = rg.ireference
    n_ref = int(group_sizes[ireference])
    ref_row_ids = np.flatnonzero(codes == ireference).astype(np.int32, copy=False)

    test_group_indices = [i for i in range(n_groups) if i != ireference]
    if not test_group_indices:
        return []

    offsets = [0]
    row_id_parts = []
    small_groups = []
    for group_index in test_group_indices:
        group_rows = np.flatnonzero(codes == group_index).astype(np.int32, copy=False)
        row_id_parts.append(group_rows)
        offsets.append(offsets[-1] + int(group_rows.size))
        if int(group_sizes[group_index]) <= MIN_GROUP_SIZE_WARNING:
            small_groups.append(str(rg.groups_order[group_index]))

    if n_ref <= MIN_GROUP_SIZE_WARNING or small_groups:
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

    all_grp_row_ids = (
        np.concatenate(row_id_parts).astype(np.int32, copy=False)
        if row_id_parts
        else np.empty(0, dtype=np.int32)
    )
    offsets_np = np.asarray(offsets, dtype=np.int32)
    offsets_gpu = cp.asarray(offsets_np)
    n_all_grp = int(all_grp_row_ids.size)
    n_test = len(test_group_indices)
    test_sizes = cp.asarray(
        group_sizes[np.asarray(test_group_indices, dtype=np.intp)].astype(
            np.float64, copy=False
        )
    )

    host_sparse = (
        isinstance(X, sp.spmatrix | sp.sparray) and not rg._sparse_negative_fallback
    )
    if host_sparse:
        if X.format not in {"csr", "csc"}:
            raise TypeError(
                "Wilcoxon sparse input must be CSR or CSC; refusing hidden "
                f"full-matrix conversion from {X.format!r}."
            )

        # zeros, not empty: an all-empty test batch (n_all_grp == 0)
        # short-circuits the kernel without writing rank_sums.
        rank_sums = cp.zeros((n_test, n_total_genes), dtype=cp.float64)
        tie_corr_arr = cp.ones((n_test, n_total_genes), dtype=cp.float64)
        n_groups_stats = n_test + 1
        compute_sums = rg._compute_stats_in_chunks
        compute_nnz = rg.comp_pts
        group_sums = cp.empty(
            (n_groups_stats, n_total_genes)
            if (compute_sums or X.format == "csc")
            else (1,),
            dtype=cp.float64,
        )
        group_nnz = cp.empty(
            (n_groups_stats, n_total_genes) if compute_nnz else (1,),
            dtype=cp.float64,
        )

        stats_code_lookup = np.full(n_groups + 1, n_groups_stats, dtype=np.int32)
        test_group_indices_np = np.asarray(test_group_indices, dtype=np.intp)
        stats_code_lookup[test_group_indices_np] = np.arange(n_test, dtype=np.int32)
        stats_code_lookup[ireference] = n_test
        stats_codes = stats_code_lookup[codes]

        if X.format == "csc":
            csc = X
            if not csc.has_sorted_indices:
                csc = csc.copy()
                csc.sort_indices()
            ref_row_map = np.full(X.shape[0], -1, dtype=np.int32)
            ref_row_map[ref_row_ids] = np.arange(n_ref, dtype=np.int32)
            grp_row_map = np.full(X.shape[0], -1, dtype=np.int32)
            grp_row_map[all_grp_row_ids] = np.arange(n_all_grp, dtype=np.int32)
            csc_host_fn, data_arr, indices_arr = _host_sparse_fn_and_arrays(
                _wcs, "ovo_streaming_csc_host", csc
            )
            csc_host_fn(
                data_arr,
                indices_arr,
                csc.indptr,
                ref_row_map,
                grp_row_map,
                offsets_np,
                stats_codes,
                rank_sums,
                tie_corr_arr,
                group_sums,
                group_nnz,
                n_ref=n_ref,
                n_all_grp=n_all_grp,
                n_rows=X.shape[0],
                n_cols=n_total_genes,
                n_groups=n_test,
                n_groups_stats=n_groups_stats,
                compute_tie_corr=tie_correct,
                compute_nnz=compute_nnz,
                sub_batch_cols=OVO_HOST_SPARSE_SUB_BATCH,
            )
        else:
            csr = X
            # Host CSR gather scans each row's native index list and tolerates
            # unsorted row indices; avoid a full CSR copy just to sort.
            csr_host_fn, data_arr, indices_arr = _host_sparse_fn_and_arrays(
                _wcs, "ovo_streaming_csr_host", csr
            )
            csr_host_fn(
                data_arr,
                indices_arr,
                csr.indptr,
                ref_row_ids.astype(np.int32, copy=False),
                all_grp_row_ids.astype(np.int32, copy=False),
                offsets_np,
                rank_sums,
                tie_corr_arr,
                group_sums,
                group_nnz,
                n_full_rows=X.shape[0],
                n_ref=n_ref,
                n_all_grp=n_all_grp,
                n_cols=n_total_genes,
                n_test=n_test,
                n_groups_stats=n_groups_stats,
                compute_tie_corr=tie_correct,
                compute_nnz=compute_nnz,
                compute_sums=compute_sums,
                sub_batch_cols=OVO_HOST_SPARSE_SUB_BATCH,
            )

        logfoldchanges_gpu = None
        if rg._compute_stats_in_chunks:
            if rg._store_wilcoxon_gpu_result and not rg.comp_pts:
                logfoldchanges_gpu = _ovo_logfoldchanges_from_sums(
                    rg,
                    group_sums,
                    test_sizes,
                    n_ref,
                )
                rg._compute_stats_in_chunks = False
            else:
                _fill_ovo_stats_from_accumulators(
                    rg,
                    group_sums,
                    group_nnz,
                    group_sizes=group_sizes,
                    test_group_indices=test_group_indices,
                    n_ref=n_ref,
                )

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

    if (
        cpsp.isspmatrix_csc(X) or cpsp.isspmatrix_csr(X)
    ) and not rg._sparse_negative_fallback:
        sparse_X = X
        if cpsp.isspmatrix_csr(sparse_X) and not sparse_X.has_sorted_indices:
            sparse_X = sparse_X.copy()
            sparse_X.sort_indices()
        data, indices, indptr = _device_sparse_arrays_f32(sparse_X)
        # offsets_gpu (built once above as int32) is reused here.
        # zeros, not empty: an all-empty test batch (n_all_grp == 0)
        # short-circuits the kernel without writing rank_sums.
        rank_sums = cp.zeros((n_test, n_total_genes), dtype=cp.float64)
        tie_corr_arr = cp.ones((n_test, n_total_genes), dtype=cp.float64)

        if cpsp.isspmatrix_csc(sparse_X):
            ref_row_map = np.full(X.shape[0], -1, dtype=np.int32)
            ref_row_map[ref_row_ids] = np.arange(n_ref, dtype=np.int32)
            grp_row_map = np.full(X.shape[0], -1, dtype=np.int32)
            grp_row_map[all_grp_row_ids] = np.arange(n_all_grp, dtype=np.int32)
            _device_sparse_fn(_wcs, "ovo_streaming_csc_device", indptr)(
                data,
                indices,
                indptr,
                cp.asarray(ref_row_map),
                cp.asarray(grp_row_map),
                offsets_gpu,
                rank_sums,
                tie_corr_arr,
                n_ref=n_ref,
                n_all_grp=n_all_grp,
                n_cols=n_total_genes,
                n_groups=n_test,
                compute_tie_corr=tie_correct,
                sub_batch_cols=OVO_DEVICE_SPARSE_SUB_BATCH,
            )
        else:
            _device_sparse_fn(_wcs, "ovo_streaming_csr_device", indptr)(
                data,
                indices,
                indptr,
                cp.asarray(ref_row_ids, dtype=cp.int32),
                cp.asarray(all_grp_row_ids, dtype=cp.int32),
                offsets_gpu,
                rank_sums,
                tie_corr_arr,
                n_ref=n_ref,
                n_all_grp=n_all_grp,
                n_cols=n_total_genes,
                n_groups=n_test,
                compute_tie_corr=tie_correct,
                sub_batch_cols=OVO_DEVICE_SPARSE_SUB_BATCH,
            )

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
                None,
            )
            return []
        scores_host = scores.get()
        p_host = p_values.get()
        return [
            (group_index, scores_host[slot], p_host[slot])
            for slot, group_index in enumerate(test_group_indices)
        ]

    chunk_width = _choose_wilcoxon_chunk_size(chunk_size, n_total_genes)

    scores_host = np.empty((n_test, n_total_genes), dtype=np.float64)
    pvals_host = np.empty((n_test, n_total_genes), dtype=np.float64)

    for start in range(0, n_total_genes, chunk_width):
        stop = min(start + chunk_width, n_total_genes)
        n_cols = stop - start

        ref_block = _extract_dense_rows_cols(X, ref_row_ids, start, stop)
        grp_block = _extract_dense_rows_cols(X, all_grp_row_ids, start, stop)

        _fill_ovo_chunk_stats(
            rg,
            ref_block,
            grp_block,
            offsets=offsets_np,
            test_group_indices=test_group_indices,
            start=start,
            stop=stop,
            group_sizes=group_sizes,
        )

        ref_f32 = cp.asarray(ref_block, dtype=cp.float32, order="F")
        grp_f32 = cp.asarray(grp_block, dtype=cp.float32, order="F")
        # zeros/ones, not empty: an all-empty test batch (n_all_grp == 0)
        # short-circuits the kernel, leaving these outputs unwritten.
        rank_sums = cp.zeros((n_test, n_cols), dtype=cp.float64)
        tie_corr = cp.ones((n_test, n_cols), dtype=cp.float64)

        _wc.ovo_rank_dense_tiered_unsorted_ref(
            ref_f32,
            grp_f32,
            offsets_gpu,
            rank_sums,
            tie_corr,
            n_ref=n_ref,
            n_all_grp=n_all_grp,
            n_cols=n_cols,
            n_groups=n_test,
            compute_tie_corr=tie_correct,
            sub_batch_cols=OVO_DENSE_TIERED_SUB_BATCH,
            stream=cp.cuda.get_current_stream().ptr,
        )

        scores, p_values = _ovo_z_pvals(
            rank_sums,
            test_sizes,
            n_ref,
            tie_corr,
            tie_correct=tie_correct,
            use_continuity=use_continuity,
            return_u_values=return_u_values,
        )

        scores_host[:, start:stop] = scores.get()
        pvals_host[:, start:stop] = p_values.get()

    return [
        (group_index, scores_host[slot], pvals_host[slot])
        for slot, group_index in enumerate(test_group_indices)
    ]
