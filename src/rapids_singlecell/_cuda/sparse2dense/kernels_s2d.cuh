#pragma once

#include <cuda_runtime.h>

template <typename T, typename IdxT, bool C_ORDER>
__global__ void sparse2dense_kernel(const IdxT* __restrict__ indptr,
                                    const IdxT* __restrict__ index,
                                    const T* __restrict__ data,
                                    T* __restrict__ out, long long major,
                                    long long minor) {
    long long stride_x = (long long)blockDim.x * gridDim.x;
    long long stride_y = (long long)blockDim.y * gridDim.y;

    for (long long row = (long long)blockIdx.x * blockDim.x + threadIdx.x;
         row < major; row += stride_x) {
        long long start = (long long)indptr[row];
        long long stop = (long long)indptr[row + 1];
        long long nnz_row = stop - start;

        for (long long col = (long long)blockIdx.y * blockDim.y + threadIdx.y;
             col < nnz_row; col += stride_y) {
            long long idx = (long long)index[start + col];
            if (idx < 0 || idx >= minor) {
                continue;
            }
            long long res_index;
            if constexpr (C_ORDER) {
                // row-major: [row, idx] -> row*minor + idx
                res_index = row * minor + idx;
            } else {
                // col-major (Fortran): [row, idx] -> row + idx*major
                res_index = row + idx * major;
            }
            atomicAdd(&out[res_index], data[start + col]);
        }
    }
}
