from __future__ import annotations

from typing import TYPE_CHECKING

import cupy as cp
import cupyx.scipy.sparse as cpsp
import numpy as np
import scipy.sparse as sp

if TYPE_CHECKING:
    import pandas as pd
    from numpy.typing import NDArray

EPS = 1e-9
MIN_GROUP_SIZE_WARNING = 25


def _sparse_has_negative(X) -> bool:
    """Whether an in-memory sparse ``X`` stores an explicit negative value.

    The fast sparse Wilcoxon paths add implicit (structural) zeros as a tie at
    the column minimum, which is correct only for nonnegative stored values. A
    negative breaks that, so the in-memory Wilcoxon paths fall back to the dense
    full-sort path (valid for any sign). Dask arrays are not inspected here
    (they are neither ``scipy`` nor ``cupy`` sparse); ``wilcoxon_binned`` guards
    Dask sparse separately. Dense and t-test/logreg never need this.
    """
    if sp.issparse(X) or cpsp.issparse(X):
        return X.nnz > 0 and float(X.data.min()) < 0
    return False


def _canonicalize_sparse(X):
    """Sum duplicate entries and sort indices of sparse ``X`` in place.

    The fast Wilcoxon paths rank each stored nonzero once, so non-canonical
    input with duplicate ``(row, col)`` entries would diverge from scanpy,
    which sums duplicates when it densifies. Canonicalizing keeps them in
    agreement. A no-op for already-canonical or dense input.
    """
    if (
        (sp.issparse(X) or cpsp.issparse(X))
        and getattr(X, "format", None) in {"csr", "csc"}
        and not X.has_canonical_format
    ):
        X.sum_duplicates()  # also sorts indices and sets the canonical flag
    return X


def _select_groups(
    labels: pd.Series,
    selected: list | None,
    *,
    reference: str = "rest",
    skip_empty_groups: bool = False,
) -> tuple[NDArray, NDArray[np.int32], NDArray[np.int64]]:
    """Build integer group codes from a categorical Series.

    Parameters
    ----------
    labels
        Categorical Series (from ``adata.obs[groupby]``).
    selected
        Group names to keep, or ``None`` for all groups.
        Must already include the reference group if applicable.

    Returns
    -------
    groups_order
        Selected group names as a numpy array.
    group_codes
        Per-cell int32 codes: ``0..n_groups-1`` for selected cells,
        ``n_groups`` (sentinel) for unselected cells.
    group_sizes
        Number of cells per selected group (int64).
    """
    all_categories = labels.cat.categories

    if selected is None:
        selected = list(all_categories)
    # else: preserve the user-provided order. scanpy's select_groups does NOT
    # re-sort to category order, so the output column order echoes `groups=`.

    if skip_empty_groups:
        counts = {
            str(name): int(count) for name, count in labels.value_counts().items()
        }
        valid_selected = [group for group in selected if counts.get(str(group), 0) >= 2]
        if reference != "rest":
            ref_matches = [group for group in selected if str(group) == str(reference)]
            if ref_matches:
                ref_group = ref_matches[0]
                if ref_group not in valid_selected:
                    msg = (
                        f"reference = {reference} has fewer than two samples after "
                        "filtering and cannot be used for rank_genes_groups."
                    )
                    raise ValueError(msg)
        selected = valid_selected
        if len(selected) == 0:
            msg = (
                "No groups with at least two samples remain after applying "
                "skip_empty_groups=True."
            )
            raise ValueError(msg)

    n_groups = len(selected)
    groups_order = np.array(selected)

    # Map original category index → selected group index
    str_to_sel = {str(name): idx for idx, name in enumerate(selected)}
    orig_to_sel: dict[int, int] = {}
    for cat_idx, cat_name in enumerate(all_categories):
        sel_idx = str_to_sel.get(str(cat_name))
        if sel_idx is not None:
            orig_to_sel[cat_idx] = sel_idx

    orig_codes = labels.cat.codes.to_numpy()
    group_codes = np.full(len(orig_codes), n_groups, dtype=np.int32)
    for orig_idx, sel_idx in orig_to_sel.items():
        group_codes[orig_codes == orig_idx] = sel_idx

    group_sizes = np.bincount(group_codes, minlength=n_groups + 1)[:n_groups].astype(
        np.int64
    )

    invalid_groups = {str(selected[i]) for i in range(n_groups) if group_sizes[i] < 2}
    if invalid_groups:
        msg = (
            f"Could not calculate statistics for groups {', '.join(invalid_groups)} "
            "since they contain fewer than two samples."
        )
        raise ValueError(msg)

    return groups_order, group_codes, group_sizes


def _choose_chunk_size(requested: int | None) -> int:
    """Choose chunk size for gene processing."""
    if requested is not None:
        return int(requested)
    return 128


def _csc_columns_to_gpu(X_csc, start: int, stop: int, n_rows: int) -> cp.ndarray:
    """
    Densify a CSC column window [start, stop) into an F-order float64 block via
    the fused ``csc_tile_to_dense`` kernel (column-major, coalesced, no atomics).

    Slices the window by indptr pointers so only that window's nonzeros are
    touched (and, for host CSC, transferred). Works for scipy and CuPy CSC.
    """
    from rapids_singlecell._cuda import _rank_stats_cuda as _rs

    s_ptr = int(X_csc.indptr[start])
    e_ptr = int(X_csc.indptr[stop])
    out = cp.zeros((n_rows, stop - start), dtype=cp.float64, order="F")
    if e_ptr > s_ptr:
        chunk_data = cp.asarray(X_csc.data[s_ptr:e_ptr])
        chunk_indices = cp.asarray(X_csc.indices[s_ptr:e_ptr])
        chunk_indptr = cp.asarray(X_csc.indptr[start : stop + 1] - s_ptr)
        _rs.csc_tile_to_dense(
            chunk_indptr,
            chunk_indices,
            chunk_data,
            out,
            col_lb=0,
            col_ub=stop - start,
            stream=cp.cuda.get_current_stream().ptr,
        )
    return out


def _csr_tile_to_dense_block(X, start: int, stop: int) -> cp.ndarray:
    """Densify a CSR column window [start, stop) straight into an F-order
    float64 block via a single fused CSR->dense kernel, skipping the CSR->CSC
    tile rebuild that ``X[:, start:stop].tocsc()`` (host) / ``X[:, start:stop]``
    (device) would do. For device CSR the index arrays are already on the GPU,
    so there is no transfer.
    """
    from rapids_singlecell._cuda import _rank_stats_cuda as _rs

    n_rows = X.shape[0]
    out = cp.zeros((n_rows, stop - start), dtype=cp.float64, order="F")
    if X.nnz == 0:
        return out
    _rs.csr_tile_to_dense(
        cp.asarray(X.indptr),
        cp.asarray(X.indices),
        cp.asarray(X.data),
        out,
        col_lb=int(start),
        col_ub=int(stop),
        stream=cp.cuda.get_current_stream().ptr,
    )
    return out


def _get_column_block(X, start: int, stop: int) -> cp.ndarray:
    """Extract a column block as a dense F-order float64 CuPy array."""
    match X:
        # Device CSR: the fused csr_tile_to_dense kernel densifies the window in
        # one pass with no transfer (index arrays are already on the GPU) -- the
        # big win. Host CSR is intentionally NOT routed here: doing so would
        # re-transfer the whole CSR every chunk (only ~1.15x and worse with more
        # chunks); host data should be moved to the device once upstream
        # (`X_to_GPU`) so it lands in this fast device branch, otherwise it falls
        # through to the `.tocsc()` path below.
        case cpsp.csr_matrix():
            return _csr_tile_to_dense_block(X, start, stop)
        case sp.csc_matrix() | sp.csc_array():
            return _csc_columns_to_gpu(X, start, stop, X.shape[0])
        case sp.spmatrix() | sp.sparray():
            chunk = cpsp.csc_matrix(X[:, start:stop].tocsc())
            return _csc_columns_to_gpu(chunk, 0, chunk.shape[1], X.shape[0])
        case cpsp.csc_matrix():
            return _csc_columns_to_gpu(X, start, stop, X.shape[0])
        case cpsp.spmatrix():
            chunk = cpsp.csc_matrix(X[:, start:stop].tocsc())
            return _csc_columns_to_gpu(chunk, 0, chunk.shape[1], X.shape[0])
        case np.ndarray() | cp.ndarray():
            return cp.asarray(X[:, start:stop], dtype=cp.float64, order="F")
        case _:
            raise ValueError(f"Unsupported matrix type: {type(X)}")


def _ovr_dense_block_f32(X, start: int, stop: int) -> cp.ndarray:
    """OVR (vs-rest): ALL cells x gene-window, F-order float32.

    For sparse X (the negative-values dense fallback) the window is densified on
    the fly via the shared CSR/CSC densify path (`_get_column_block`), so no
    full-matrix dense materialization happens.
    """
    if isinstance(X, np.ndarray | cp.ndarray):
        return cp.asarray(X[:, start:stop], dtype=cp.float32, order="F")
    if sp.issparse(X) or cpsp.issparse(X):
        block = _get_column_block(X, start, stop)  # float64 F-order chunk
        return cp.asfortranarray(block.astype(cp.float32, copy=False))
    raise TypeError(f"Expected dense matrix, got {type(X)}")


def _ovo_dense_block(X, row_ids: np.ndarray, start: int, stop: int) -> cp.ndarray:
    """OVO (with-reference): a ROW SUBSET (`row_ids`) x gene-window, F-order.

    OVO ranks the reference group against each other group, so it materializes
    only the selected rows -- unlike `_ovr_dense_block_f32`, which takes all
    cells.
    """
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
