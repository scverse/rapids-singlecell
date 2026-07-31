#pragma once

#include <cuda_runtime.h>

constexpr int BLOCK_SIZE_MAJOR = 64;

template <typename T, typename IdxT>
__global__ void mean_var_major_kernel(const IdxT* __restrict__ indptr,
                                      const IdxT* __restrict__ indices,
                                      const T* __restrict__ data,
                                      double* __restrict__ means,
                                      double* __restrict__ vars, int major,
                                      int /*minor*/) {
    int major_idx = blockIdx.x;
    if (major_idx >= major) return;

    IdxT start_idx = indptr[major_idx];
    IdxT stop_idx = indptr[major_idx + 1];

    __shared__ double mean_place[BLOCK_SIZE_MAJOR];
    __shared__ double var_place[BLOCK_SIZE_MAJOR];

    mean_place[threadIdx.x] = 0.0;
    var_place[threadIdx.x] = 0.0;
    __syncthreads();

    for (IdxT minor_idx = start_idx + threadIdx.x; minor_idx < stop_idx;
         minor_idx += blockDim.x) {
        double value = static_cast<double>(data[minor_idx]);
        mean_place[threadIdx.x] += value;
        var_place[threadIdx.x] += value * value;
    }
    __syncthreads();

    for (unsigned int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (threadIdx.x < s) {
            mean_place[threadIdx.x] += mean_place[threadIdx.x + s];
            var_place[threadIdx.x] += var_place[threadIdx.x + s];
        }
        __syncthreads();
    }
    if (threadIdx.x == 0) {
        means[major_idx] = mean_place[0];
        vars[major_idx] = var_place[0];
    }
}

template <typename T, typename IdxT>
__global__ void mean_var_minor_kernel(const IdxT* __restrict__ indices,
                                      const T* __restrict__ data,
                                      double* __restrict__ means,
                                      double* __restrict__ vars,
                                      long long nnz) {
    const long long stride = (long long)blockDim.x * gridDim.x;
    for (long long idx = (long long)blockDim.x * blockIdx.x + threadIdx.x;
         idx < nnz; idx += stride) {
        double value = static_cast<double>(data[idx]);
        IdxT minor_pos = indices[idx];
        atomicAdd(&means[minor_pos], value);
        atomicAdd(&vars[minor_pos], value * value);
    }
}
