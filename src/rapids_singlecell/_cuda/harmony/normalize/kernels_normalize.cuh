#pragma once

#include <cuda_runtime.h>

// ---- L2 row normalize ----
// One block per row. Supports src == dst; all reads finish before writes begin.
template <typename T>
__global__ void l2_row_normalize_kernel(const T* src, T* dst, int n_rows,
                                        int n_cols) {
    int row = blockIdx.x;
    if (row >= n_rows) return;

    const T* src_row = src + (size_t)row * n_cols;
    T* dst_row = dst + (size_t)row * n_cols;

    // Phase 1: sum of squares
    T acc = T(0);
    for (int col = threadIdx.x; col < n_cols; col += blockDim.x) {
        T v = src_row[col];
        acc += v * v;
    }

// Phase 2: warp reduction
#pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1)
        acc += __shfl_down_sync(0xffffffff, acc, offset);

    // Phase 3: block reduction
    __shared__ T warp_sums[32];
    int warp_id = threadIdx.x >> 5;
    int lane = threadIdx.x & 31;
    int num_warps = (blockDim.x + 31) >> 5;

    if (lane == 0) warp_sums[warp_id] = acc;
    __syncthreads();

    if (threadIdx.x < 32) {
        T val = (threadIdx.x < num_warps) ? warp_sums[threadIdx.x] : T(0);
#pragma unroll
        for (int offset = 16; offset > 0; offset >>= 1)
            val += __shfl_down_sync(0xffffffff, val, offset);
        if (threadIdx.x == 0) {
            T inv_norm = rsqrt(val);
            if (inv_norm > T(1e12)) inv_norm = T(1e12);
            warp_sums[0] = inv_norm;
        }
    }
    __syncthreads();

    T scale = warp_sums[0];
    for (int col = threadIdx.x; col < n_cols; col += blockDim.x)
        dst_row[col] = src_row[col] * scale;
}
