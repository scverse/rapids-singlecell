"""Construct spatial CSR graphs from row-grouped, unique off-diagonal edges."""

from __future__ import annotations

import cupy as cp
import numpy as np
from cupyx.scipy import sparse as cp_sparse

from rapids_singlecell._cuda import _spatial_cuda


def build_graphs(
    rows: cp.ndarray,
    cols: cp.ndarray,
    values: cp.ndarray | None,
    n_obs: int,
    *,
    set_diag: bool,
) -> tuple[cp_sparse.csr_matrix, cp_sparse.csr_matrix | None]:
    """Build separate adjacency and distance CSRs (``values=None``: adjacency only)."""
    nnz, with_dst = len(rows) + n_obs, values is not None
    index = np.int64 if nnz > np.iinfo(np.int32).max else np.int32
    n_dst = nnz if with_dst else 0
    values = values if with_dst else cp.empty(0, dtype=cp.float32)
    rows, cols, values = map(cp.ascontiguousarray, (rows, cols, values))
    indptr, indices, dst_indices = (cp.empty(n, index) for n in (n_obs + 1, nnz, n_dst))
    adj, dst = cp.empty(nnz, cp.float32), cp.empty(n_dst, values.dtype)
    args = (indptr, indices, adj, dst_indices, dst)
    stream = cp.cuda.get_current_stream().ptr
    _spatial_cuda.assemble_graphs(rows, cols, values, set_diag, *args, stream)
    adj = cp_sparse.csr_matrix((adj, indices, indptr), shape=(n_obs, n_obs))
    if not with_dst:
        return adj, None
    return adj, cp_sparse.csr_matrix((dst, dst_indices, indptr.copy()), shape=adj.shape)


def build_adjacency(
    rows: cp.ndarray, cols: cp.ndarray, n_obs: int, *, set_diag: bool
) -> cp_sparse.csr_matrix:
    """Build the adjacency CSR, with the same edge contract as :func:`build_graphs`."""
    return build_graphs(rows, cols, None, n_obs, set_diag=set_diag)[0]
