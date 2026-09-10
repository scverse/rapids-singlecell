"""Construct spatial CSR pairs from row-grouped, unique off-diagonal edges."""

from __future__ import annotations

import cupy as cp
import numpy as np
from cupyx.scipy import sparse as cp_sparse

from rapids_singlecell._cuda import _spatial_cuda


def build_graphs(
    rows: cp.ndarray,
    cols: cp.ndarray,
    values: cp.ndarray,
    n_obs: int,
    *,
    set_diag: bool,
) -> tuple[cp_sparse.csr_matrix, cp_sparse.csr_matrix]:
    """Build independent CSR graphs from row-grouped unique off-diagonal edges.

    Columns may be unsorted; callers validate distances. Stored zero diagonals
    must survive until percentile pruning, including when ``set_diag=False``.
    """
    adj, dst = _assemble_graphs(rows, cols, values, n_obs, set_diag=set_diag)
    assert dst is not None
    return adj, dst


def build_adjacency(
    rows: cp.ndarray, cols: cp.ndarray, n_obs: int, *, set_diag: bool
) -> cp_sparse.csr_matrix:
    """Build adjacency only, with the same edge contract as :func:`build_graphs`."""
    return _assemble_graphs(rows, cols, None, n_obs, set_diag=set_diag)[0]


def _assemble_graphs(
    rows: cp.ndarray,
    cols: cp.ndarray,
    values: cp.ndarray | None,
    n_obs: int,
    *,
    set_diag: bool,
) -> tuple[cp_sparse.csr_matrix, cp_sparse.csr_matrix | None]:
    rows, cols = (cp.ascontiguousarray(array) for array in (rows, cols))
    with_distances = values is not None
    if with_distances:
        values = cp.ascontiguousarray(values)
    dtype = values.dtype if with_distances else np.dtype("float32")
    index_dtype = np.dtype(
        "int64" if max(n_obs, len(rows) + n_obs) > np.iinfo(np.int32).max else "int32"
    )
    n_graphs = 2 if with_distances else 1
    indptr = [cp.empty(n_obs + 1, dtype=index_dtype) for _ in range(n_graphs)]
    columns = [cp.empty(len(rows) + n_obs, dtype=index_dtype) for _ in range(n_graphs)]
    adj_data = cp.empty(len(rows) + n_obs, dtype=cp.float32)
    if with_distances:
        dst_data = cp.empty(len(rows) + n_obs, dtype=dtype)
    else:
        # These dummy pointers are never dereferenced in the adjacency-only
        # specialization, which avoids all distance allocations and writes.
        indptr.append(indptr[0])
        columns.append(columns[0])
        values = dst_data = adj_data
    _spatial_cuda.assemble_graphs(
        rows,
        cols,
        values,
        set_diag=set_diag,
        with_distances=with_distances,
        adj_indptr=indptr[0],
        dst_indptr=indptr[1],
        adj_columns=columns[0],
        dst_columns=columns[1],
        adj_data=adj_data,
        dst_data=dst_data,
        stream=cp.cuda.get_current_stream().ptr,
    )
    shape = (n_obs, n_obs)
    adj = cp_sparse.csr_matrix((adj_data, columns[0], indptr[0]), shape=shape)
    dst = (
        cp_sparse.csr_matrix((dst_data, columns[1], indptr[1]), shape=shape)
        if with_distances
        else None
    )
    return adj, dst
