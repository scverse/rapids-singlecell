from __future__ import annotations

import os
from concurrent.futures import ThreadPoolExecutor
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
SPARSE_NEGATIVE_SCAN_MIN_ITEMS = 64_000_000
SPARSE_NEGATIVE_SCAN_MAX_WORKERS = 64


def _sparse_has_negative(X) -> bool:
    """Return whether an in-memory sparse matrix stores a negative value.
    Signed sparse Wilcoxon needs the sign-safe sparse-dense ranker."""
    if sp.issparse(X):
        data = X.data
        dtype_kind = np.dtype(data.dtype).kind
        if dtype_kind in {"b", "c", "u"}:
            return False
        if data.size == 0:
            return False
        if data.size < SPARSE_NEGATIVE_SCAN_MIN_ITEMS:
            return float(data.min()) < 0

        n_workers = min(
            SPARSE_NEGATIVE_SCAN_MAX_WORKERS,
            os.cpu_count() or 1,
            max(1, data.size // SPARSE_NEGATIVE_SCAN_MIN_ITEMS),
        )
        bounds = np.linspace(0, data.size, n_workers + 1, dtype=np.intp)

        def chunk_min(chunk_index: int):
            start = int(bounds[chunk_index])
            stop = int(bounds[chunk_index + 1])
            return data[start:stop].min()

        with ThreadPoolExecutor(max_workers=n_workers) as executor:
            minima = list(executor.map(chunk_min, range(n_workers)))
        return float(np.min(minima)) < 0
    if cpsp.issparse(X):
        if np.dtype(X.data.dtype).kind in {"b", "c", "u"}:
            return False
        return X.nnz > 0 and float(X.data.min()) < 0
    return False


def _canonicalize_sparse(X):
    """Sum duplicates and sort sparse indices in place when needed.
    Fast Wilcoxon ranks stored nnz once, so it expects scanpy's summed view."""
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
    """Build selected group names, per-cell int32 codes, and group sizes.
    Unselected cells receive the sentinel code ``n_groups``."""
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
    # The extra final slot maps pandas' missing-category code (-1) to the
    # unselected sentinel through normal negative indexing.
    code_lookup = np.full(len(all_categories) + 1, n_groups, dtype=np.int32)
    for orig_idx, sel_idx in orig_to_sel.items():
        code_lookup[orig_idx] = sel_idx
    group_codes = code_lookup[orig_codes]

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
    """Densify a CSC column window into an F-order float64 GPU block.
    Slices by indptr so only window nonzeros are touched/transferred."""
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
    """Densify a CSR column window into an F-order float64 GPU block.
    Device CSR avoids rebuilding a CSR/CSC slice before densifying."""
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
        # Device CSR can densify in one pass without transfer.
        # Host CSR intentionally falls through to avoid per-chunk full transfers.
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
