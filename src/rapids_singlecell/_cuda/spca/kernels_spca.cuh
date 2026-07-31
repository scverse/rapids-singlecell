#pragma once

#include <cuda_runtime.h>

template <typename T, typename IdxT>
__global__ void gram_csr_upper_kernel(const IdxT* indptr, const IdxT* index,
                                      const T* data, size_t nrows, size_t ncols,
                                      T* out) {
    int col_offset = threadIdx.x;
    for (size_t row = static_cast<size_t>(blockIdx.x); row < nrows;
         row += gridDim.x) {
        IdxT start = indptr[row];
        IdxT end = indptr[row + 1];

        for (IdxT idx1 = start; idx1 < end; ++idx1) {
            IdxT index1 = index[idx1];
            if (index1 < 0 || index1 >= ncols) continue;
            T data1 = data[idx1];
            for (IdxT idx2 = idx1 + col_offset; idx2 < end;
                 idx2 += blockDim.x) {
                IdxT index2 = index[idx2];
                if (index2 < 0 || index2 >= ncols) continue;
                T data2 = data[idx2];
                size_t lo = min(index1, index2);
                size_t hi = max(index1, index2);
                atomicAdd(&out[lo * ncols + hi], data1 * data2);
            }
        }
    }
}

template <typename T>
__global__ void copy_upper_to_lower_kernel(T* output, size_t ncols) {
    const size_t row_stride = static_cast<size_t>(blockDim.y) * gridDim.y;
    const size_t col_stride = static_cast<size_t>(blockDim.x) * gridDim.x;
    for (size_t row =
             static_cast<size_t>(blockIdx.y) * blockDim.y + threadIdx.y;
         row < ncols; row += row_stride) {
        for (size_t col =
                 static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
             col < ncols; col += col_stride) {
            if (row > col) {
                output[row * ncols + col] = output[col * ncols + row];
            }
        }
    }
}

template <typename T>
__global__ void cov_from_gram_kernel(T* cov_values, const T* gram_matrix,
                                     const T* mean_x, const T* mean_y,
                                     size_t ncols) {
    const size_t row_stride = static_cast<size_t>(blockDim.x) * gridDim.x;
    const size_t col_stride = static_cast<size_t>(blockDim.y) * gridDim.y;
    for (size_t rid =
             static_cast<size_t>(blockDim.x) * blockIdx.x + threadIdx.x;
         rid < ncols; rid += row_stride) {
        for (size_t cid =
                 static_cast<size_t>(blockDim.y) * blockIdx.y + threadIdx.y;
             cid < ncols; cid += col_stride) {
            const size_t idx = rid * ncols + cid;
            cov_values[idx] = gram_matrix[idx] - mean_x[rid] * mean_y[cid];
        }
    }
}

template <typename IdxT>
__global__ void check_zero_genes_kernel(const IdxT* indices, int* genes,
                                        long long nnz, int num_genes) {
    const long long stride = (long long)blockDim.x * gridDim.x;
    for (long long value = (long long)blockIdx.x * blockDim.x + threadIdx.x;
         value < nnz; value += stride) {
        long long gene_index = static_cast<long long>(indices[value]);
        if (gene_index < 0 || gene_index >= num_genes) continue;
        atomicAdd(&genes[gene_index], 1);
    }
}
