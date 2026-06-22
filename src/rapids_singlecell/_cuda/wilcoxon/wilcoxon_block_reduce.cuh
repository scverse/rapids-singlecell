#pragma once

#include <cuda_runtime.h>

// Block-wide sum of `val` across all threads. `warp_buf` is shared scratch
// holding one double per warp (>= ceil(blockDim.x / 32) <= 32). Result is
// returned on thread 0 (lane 0 of warp 0); other threads get 0.0.
__device__ __forceinline__ double wilcoxon_block_sum(double val,
                                                     double* warp_buf) {
#pragma unroll
    for (int off = 16; off > 0; off >>= 1)
        val += __shfl_down_sync(0xffffffff, val, off);
    int lane = threadIdx.x & 31;
    int wid = threadIdx.x >> 5;
    if (lane == 0) warp_buf[wid] = val;
    __syncthreads();
    if (threadIdx.x < 32) {
        double v = (threadIdx.x < ((blockDim.x + 31) >> 5))
                       ? warp_buf[threadIdx.x]
                       : 0.0;
#pragma unroll
        for (int off = 16; off > 0; off >>= 1)
            v += __shfl_down_sync(0xffffffff, v, off);
        return v;
    }
    return 0.0;
}
