#pragma once

#include <cuda_runtime.h>

// All scale kernels assume std[col] > 0, guaranteed by the Python caller
// (_scale.py clips std to a minimum value before invoking these kernels).

template <typename T, typename IdxT>
__global__ void csc_scale_diff_kernel(const IdxT* __restrict__ indptr,
                                      T* __restrict__ data,
                                      const T* __restrict__ std, int ncols) {
    int col = blockIdx.x;
    if (col >= ncols) return;
    long long start_idx = (long long)indptr[col];
    long long stop_idx = (long long)indptr[col + 1];
    T diver = T(1) / std[col];
    for (long long i = start_idx + threadIdx.x; i < stop_idx; i += blockDim.x) {
        data[i] *= diver;
    }
}

template <typename T, typename IdxT>
__global__ void csr_scale_diff_kernel(const IdxT* __restrict__ indptr,
                                      const IdxT* __restrict__ indices,
                                      T* __restrict__ data,
                                      const T* __restrict__ std,
                                      const int* __restrict__ mask, T clipper,
                                      int nrows) {
    int row = blockIdx.x;
    if (row >= nrows) return;
    if (mask[row]) {
        long long start_idx = (long long)indptr[row];
        long long stop_idx = (long long)indptr[row + 1];
        for (long long i = start_idx + threadIdx.x; i < stop_idx;
             i += blockDim.x) {
            long long idx = (long long)indices[i];
            T res = data[i] / std[idx];
            data[i] = res < clipper ? res : clipper;
        }
    }
}

template <typename T>
__global__ void dense_scale_center_diff_kernel(
    T* data, const T* __restrict__ mean, const T* __restrict__ std,
    const int* __restrict__ mask, T clipper, long long nrows, long long ncols) {
    const long long row_stride = (long long)blockDim.x * gridDim.x;
    const long long col_stride = (long long)blockDim.y * gridDim.y;
    for (long long row = (long long)blockIdx.x * blockDim.x + threadIdx.x;
         row < nrows; row += row_stride) {
        if (!mask[row]) continue;
        for (long long col = (long long)blockIdx.y * blockDim.y + threadIdx.y;
             col < ncols; col += col_stride) {
            const long long idx = row * ncols + col;
            T res = data[idx] - mean[col];
            res = res / std[col];
            if (res > clipper) res = clipper;
            if (res < -clipper) res = -clipper;
            data[idx] = res;
        }
    }
}

template <typename T>
__global__ void dense_scale_diff_kernel(T* __restrict__ data,
                                        const T* __restrict__ std,
                                        const int* __restrict__ mask, T clipper,
                                        long long nrows, long long ncols) {
    const long long row_stride = (long long)blockDim.x * gridDim.x;
    const long long col_stride = (long long)blockDim.y * gridDim.y;
    for (long long row = (long long)blockIdx.x * blockDim.x + threadIdx.x;
         row < nrows; row += row_stride) {
        if (!mask[row]) continue;
        for (long long col = (long long)blockIdx.y * blockDim.y + threadIdx.y;
             col < ncols; col += col_stride) {
            const long long idx = row * ncols + col;
            T res = data[idx] / std[col];
            data[idx] = res < clipper ? res : clipper;
        }
    }
}
