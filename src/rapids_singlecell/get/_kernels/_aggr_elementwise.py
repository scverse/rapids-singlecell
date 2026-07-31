from __future__ import annotations

import cupy as cp

_sum_duplicates_diff = cp.ElementwiseKernel(
    "raw T row, raw T col",
    "T diff",
    """
    T diff_out = 1;
    if (i == 0 || (row[i - 1] == row[i] && col[i - 1] == col[i])) {
    diff_out = 0;
    }
    diff = diff_out;
    """,
    "cupyx_scipy_sparse_coo_sum_duplicates_diff",
)

_sum_duplicates_assign = cp.ElementwiseKernel(
    "T src_row, T src_col, I index",
    "raw T rows, raw T indices",
    """
    rows[index] = src_row;
    indices[index] = src_col;
    """,
    "cupyx_scipy_sparse_coo_sum_duplicates_assign",
)

_scatter_sum = cp.ElementwiseKernel(
    "float64 src, I index",
    "raw float64 sums",
    """
    atomicAdd(&sums[index], src);
    """,
    "create_sum_sparse_matrix",
)

_scatter_mean_var = cp.ElementwiseKernel(
    "float64 src, I index",
    "raw float64 means, raw float64 var",
    """
    atomicAdd(&means[index], src);
    atomicAdd(&var[index], src * src);
    """,
    "create_mean_var_sparse_matrix",
)

_scatter_count_nonzero = cp.ElementwiseKernel(
    "float64 src, I index",
    "raw float32 counts",
    """
    if (src != 0){
        atomicAdd(&counts[index], 1.0f);
    }
    """,
    "create_count_nonzero_sparse_matrix",
)
