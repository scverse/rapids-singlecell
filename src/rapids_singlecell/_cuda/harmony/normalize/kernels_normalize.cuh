#pragma once

#include <cuda_runtime.h>

// ---- L2 row normalize ----
// One block per row. Writes to separate output buffer.
template <typename T>
__global__ void l2_row_normalize_kernel(const T* __restrict__ src,
                                        T* __restrict__ dst, int n_rows,
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

// ---- L1 row normalize (in-place) ----
// One block per row. Warp-shuffle + cross-warp reduction for L1 normalization.
template <typename T>
__global__ void normalize_kernel(T* X, long long rows, long long cols) {
    __shared__ T warp_sums[32];

    long long row = blockIdx.x;
    int tid = threadIdx.x;
    int lane = tid & 31;
    int warp_id = tid >> 5;
    int num_warps = blockDim.x >> 5;

    if (row >= rows) return;

    T* row_ptr = X + row * cols;

    // Step 1: Each thread accumulates its portion
    T acc = (T)0;
    for (long long col = tid; col < cols; col += blockDim.x) {
        T v = row_ptr[col];
        acc += (v < (T)0) ? -v : v;
    }

// Step 2: Warp-level reduction using shuffle
#pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1) {
        acc += __shfl_down_sync(0xffffffff, acc, offset);
    }

    // Lane 0 of each warp writes to shared memory
    if (lane == 0) warp_sums[warp_id] = acc;
    __syncthreads();

    // Step 3: First warp reduces all warp results
    if (tid < 32) {
        T val = (tid < num_warps) ? warp_sums[tid] : (T)0;
#pragma unroll
        for (int offset = 16; offset > 0; offset >>= 1)
            val += __shfl_down_sync(0xffffffff, val, offset);
        if (tid == 0) {
            T final_norm = val;
            if (final_norm < (T)1e-12) final_norm = (T)1e-12;
            warp_sums[0] = (T)1 / final_norm;
        }
    }
    __syncthreads();

    // Step 4: Normalize the row
    T scale = warp_sums[0];
    for (long long col = tid; col < cols; col += blockDim.x) {
        row_ptr[col] *= scale;
    }
}
