"""Fast parallel CSR to CSC conversion using Numba."""

from __future__ import annotations

import numpy as np
import scipy.sparse as sp
from numba import get_num_threads, njit, prange


@njit(parallel=True, boundscheck=False)
def _csr_to_csc_kernel(csr_data, csr_indices, csr_indptr, n_cols):
    """
    Numba kernel for parallel CSR to CSC conversion.

    Uses a tiled approach with parallel histogram + scatter phases,
    targeting ~256 MB per block buffer for L3 cache efficiency.
    """
    num_threads = get_num_threads()
    nnz = len(csr_data)
    n_rows = len(csr_indptr) - 1

    # Allocate output arrays (int64 for matrices > 2GB)
    csc_data = np.empty(nnz, dtype=csr_data.dtype)
    csc_indices = np.empty(nnz, dtype=np.int64)
    csc_indptr = np.empty(n_cols + 1, dtype=np.int64)
    csc_indptr[0] = 0

    # Block size targeting 256 MB per block buffer (L3 cache target)
    TARGET_MEM_BYTES = 256 * 1024 * 1024
    MIN_BLOCK_SIZE = 1000
    block_size = TARGET_MEM_BYTES // (num_threads * 8)
    if block_size < MIN_BLOCK_SIZE:
        block_size = MIN_BLOCK_SIZE

    # Workspace: threads x block_width
    counts = np.zeros((num_threads, block_size), dtype=np.int64)
    row_chunk_size = (n_rows + num_threads - 1) // num_threads

    current_global_offset = 0

    # Tiled execution over column blocks
    for col_start in range(0, n_cols, block_size):
        col_end = min(col_start + block_size, n_cols)
        current_block_width = col_end - col_start

        # 1. Zero counters for this block
        counts[:, :current_block_width] = 0

        # 2. Parallel histogram (count items per column)
        for t in prange(num_threads):
            r_start = t * row_chunk_size
            r_end = min((t + 1) * row_chunk_size, n_rows)
            for r in range(r_start, r_end):
                for i in range(csr_indptr[r], csr_indptr[r + 1]):
                    c = csr_indices[i]
                    if c >= col_start and c < col_end:
                        counts[t, c - col_start] += 1

        # 3. Compute write offsets (sequential, fast)
        for c in range(current_block_width):
            total_in_col = 0
            for t in range(num_threads):
                count = counts[t, c]
                counts[t, c] = current_global_offset + total_in_col
                total_in_col += count

            current_global_offset += total_in_col
            csc_indptr[col_start + c + 1] = current_global_offset

        # 4. Parallel scatter (write data)
        for t in prange(num_threads):
            r_start = t * row_chunk_size
            r_end = min((t + 1) * row_chunk_size, n_rows)
            for r in range(r_start, r_end):
                for i in range(csr_indptr[r], csr_indptr[r + 1]):
                    c = csr_indices[i]
                    if c >= col_start and c < col_end:
                        local_c = c - col_start
                        dest = counts[t, local_c]

                        csc_data[dest] = csr_data[i]
                        csc_indices[dest] = r

                        counts[t, local_c] += 1

    return csc_data, csc_indices, csc_indptr


def _fast_csr_to_csc(mat_csr: sp.csr_matrix) -> sp.csc_matrix:
    """
    Convert a SciPy CSR matrix to CSC using parallel Numba kernel.

    Uses a tiled multi-threaded approach for better cache utilization
    compared to scipy's default conversion.

    Parameters
    ----------
    mat_csr
        Input CSR matrix.

    Returns
    -------
    CSC matrix with the same data.
    """
    if not sp.issparse(mat_csr) or mat_csr.format != "csr":
        raise TypeError("Input must be a SciPy CSR matrix")

    rows, cols = mat_csr.shape

    data, indices, indptr = _csr_to_csc_kernel(
        mat_csr.data,
        mat_csr.indices,
        mat_csr.indptr,
        cols,
    )

    return sp.csc_matrix((data, indices, indptr), shape=mat_csr.shape)
