#pragma once

#include <cuda_runtime.h>

// ---- Penalty kernel ----
// Stabilized=false: penalty = pow((E+1) / (O+1), theta)       [Harmony1]
// Stabilized=true:  penalty = pow((E+1) / (O+E+1), theta)     [Harmony2]
template <typename T, bool Stabilized>
__global__ void penalty_kernel(const T* __restrict__ E, const T* __restrict__ O,
                               const T* __restrict__ theta,
                               T* __restrict__ penalty, int n_batches,
                               int n_clusters) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int total = n_batches * n_clusters;
    if (i >= total) return;
    int batch = i / n_clusters;
    T denom = Stabilized ? (O[i] + E[i] + T(1)) : (O[i] + T(1));
    T ratio = (E[i] + T(1)) / denom;
    T th = theta[batch];
    penalty[i] = pow(ratio, th);
}

// ---- Fused penalty + normalize ----
// One block per row (cell). Computes exp(term*(1-sim)) * penalty, then
// row-normalizes. IdxT is the index type for idx_in: size_t (Python path) or
// int (C++ clustering loop).
template <typename T, typename IdxT>
__global__ void fused_pen_norm_kernel(const T* __restrict__ similarities,
                                      const T* __restrict__ penalty,
                                      const int* __restrict__ cats,
                                      const IdxT* __restrict__ idx_in,
                                      T* __restrict__ R_out, T term, int n_rows,
                                      int n_cols) {
    int row = blockIdx.x;
    if (row >= n_rows) return;

    int cat = cats[row];
    size_t sim_row = static_cast<size_t>(idx_in[row]);

    // Phase 1: compute exp(term * (1 - sim)) * penalty and accumulate sum
    T local_sum = T(0);
    for (int col = threadIdx.x; col < n_cols; col += blockDim.x) {
        T sim = similarities[sim_row * n_cols + col];
        T val = exp(term * (T(1) - sim));
        val *= penalty[(size_t)cat * n_cols + col];
        R_out[(size_t)row * n_cols + col] = val;
        local_sum += val;
    }

// Phase 2: warp-level reduction
#pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1)
        local_sum += __shfl_down_sync(0xffffffff, local_sum, offset);

    // Phase 3: block-level reduction
    __shared__ T shared_sum[32];
    int warp_id = threadIdx.x >> 5;
    int lane = threadIdx.x & 31;

    if (lane == 0) shared_sum[warp_id] = local_sum;
    __syncthreads();

    T row_sum = T(0);
    if (threadIdx.x < 32) {
        int num_warps = (blockDim.x + 31) >> 5;
        if (static_cast<int>(threadIdx.x) < num_warps)
            row_sum = shared_sum[threadIdx.x];
#pragma unroll
        for (int offset = 16; offset > 0; offset >>= 1)
            row_sum += __shfl_down_sync(0xffffffff, row_sum, offset);
    }

    // Broadcast sum
    if (threadIdx.x == 0) shared_sum[0] = row_sum;
    __syncthreads();
    row_sum = shared_sum[0];

    // Phase 4: normalize
    T inv_sum = T(1) / row_sum;
    for (int col = threadIdx.x; col < n_cols; col += blockDim.x)
        R_out[(size_t)row * n_cols + col] *= inv_sum;
}
