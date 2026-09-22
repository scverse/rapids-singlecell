#pragma once

#include <cuda_runtime.h>

template <typename T>
__global__ void colsum_kernel(const T* __restrict__ A, T* __restrict__ out,
                              size_t rows, size_t cols) {
    size_t tid = threadIdx.x;
    for (size_t col = blockIdx.x; col < cols; col += gridDim.x) {
        T acc = (T)0;
        for (size_t i = tid; i < rows; i += blockDim.x) {
            acc += A[i * cols + col];
        }
#pragma unroll
        for (int offset = 16; offset > 0; offset >>= 1)
            acc += __shfl_down_sync(0xffffffff, acc, offset);
        __shared__ T s[32];
        if ((threadIdx.x & 31) == 0) s[threadIdx.x >> 5] = acc;
        __syncthreads();
        if (threadIdx.x < 32) {
            T val = (threadIdx.x < (blockDim.x >> 5)) ? s[threadIdx.x] : (T)0;
#pragma unroll
            for (int off = 16; off > 0; off >>= 1)
                val += __shfl_down_sync(0xffffffff, val, off);
            if (threadIdx.x == 0) out[col] = val;
        }
        __syncthreads();
    }
}
