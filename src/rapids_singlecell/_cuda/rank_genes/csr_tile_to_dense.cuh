#pragma once

#include <cuda_runtime.h>

// CSR-slice + densify in a single pass: scatter the nonzeros of column window
// [col_lb, col_ub) straight into a dense (n_cells, col_ub-col_lb) F-order
// (column-major) double buffer. This skips the CSR -> CSC tile rebuild that a
// `X[:, lb:ub].tocsc()` densify would do.
//
// `out` must be pre-zeroed; the atomicAdd also sums duplicate column indices
// (like scipy's sum_duplicates) -- bit-identical to a dense materialization for
// canonical CSR. Output is always double; input dtype is templated.

template <typename TData, typename IndptrT, typename IndexT>
__global__ void csr_tile_to_dense_kernel(const IndptrT* __restrict__ indptr,
                                         const IndexT* __restrict__ indices,
                                         const TData* __restrict__ data,
                                         double* __restrict__ out, int col_lb,
                                         int col_ub, int n_cells) {
    const long long row =
        static_cast<long long>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (row >= n_cells) {
        return;
    }
    const long long row_start = static_cast<long long>(indptr[row]);
    const long long row_end = static_cast<long long>(indptr[row + 1]);
    // Keep column ids in IndexT: narrowing a 64-bit IndexT to int would
    // truncate large column ids and misplace writes.
    const IndexT lb = static_cast<IndexT>(col_lb);
    const IndexT ub = static_cast<IndexT>(col_ub);
    for (long long k = row_start; k < row_end; ++k) {
        const IndexT col = indices[k];
        if (col >= lb && col < ub) {
            atomicAdd(&out[static_cast<long long>(col - lb) * n_cells + row],
                      static_cast<double>(data[k]));
        }
    }
}
