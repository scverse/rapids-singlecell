#pragma once

#include <cuda_runtime.h>

// Sum `v` across the 32 lanes of a warp via shuffle-down; result on lane 0.
__device__ __forceinline__ double warp_reduce_sum(double v) {
#pragma unroll
    for (int off = 16; off > 0; off >>= 1)
        v += __shfl_down_sync(0xffffffff, v, off);
    return v;
}

// Block-wide sum of `val` using one shared double per warp.
// Result is returned on thread 0; other threads get 0.0.
__device__ __forceinline__ double wilcoxon_block_sum(double val,
                                                     double* warp_buf) {
    val = warp_reduce_sum(val);
    int lane = threadIdx.x & 31;
    int wid = threadIdx.x >> 5;
    if (lane == 0) warp_buf[wid] = val;
    __syncthreads();
    if (threadIdx.x < 32) {
        double v = (threadIdx.x < ((blockDim.x + 31) >> 5))
                       ? warp_buf[threadIdx.x]
                       : 0.0;
        return warp_reduce_sum(v);
    }
    return 0.0;
}

// Final tie-correction factor: 1 - sum(t^3 - t) / (n^3 - n), or 1.0 when the
// ranking population n_total is too small for a correction.
__device__ __forceinline__ double finalize_tie_corr(int n_total,
                                                    double tie_sum) {
    double dn = (double)n_total;
    double denom = dn * dn * dn - dn;
    return (denom > 0.0) ? (1.0 - tie_sum / denom) : 1.0;
}
