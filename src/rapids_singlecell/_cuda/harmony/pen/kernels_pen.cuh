#pragma once

#include <cuda_runtime.h>
#include <math_constants.h>

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

// Multi-covariate counterpart. `cats` is cell-major with one global marginal
// category index per covariate. The selected marginal penalty factors are
// combined in the log domain and normalized with log-sum-exp so their product
// cannot overflow before normalization. N_COVARIATES=2 and 3 are specialized;
// zero is the runtime fallback.
template <typename T, typename IdxT, int N_COVARIATES>
__global__ void fused_pen_norm_multi_kernel(
    const T* __restrict__ similarities, const T* __restrict__ penalty,
    const int* __restrict__ cats, const IdxT* __restrict__ idx_in,
    T* __restrict__ R_out, T term, int n_rows, int n_cols, int n_covariates) {
    int row = blockIdx.x;
    if (row >= n_rows) return;

    const int* row_cats =
        cats + (size_t)row * (N_COVARIATES > 0 ? N_COVARIATES : n_covariates);
    int cat0 = row_cats[0];
    int cat1 = N_COVARIATES > 0 ? row_cats[1] : 0;
    int cat2 = N_COVARIATES == 3 ? row_cats[2] : 0;
    size_t sim_row = static_cast<size_t>(idx_in[row]);

    T local_max = static_cast<T>(-CUDART_INF);
    for (int col = threadIdx.x; col < n_cols; col += blockDim.x) {
        T sim = similarities[sim_row * n_cols + col];
        T log_value = term * (T(1) - sim);

        if constexpr (N_COVARIATES == 2) {
            log_value += log(penalty[(size_t)cat0 * n_cols + col]);
            log_value += log(penalty[(size_t)cat1 * n_cols + col]);
        } else if constexpr (N_COVARIATES == 3) {
            log_value += log(penalty[(size_t)cat0 * n_cols + col]);
            log_value += log(penalty[(size_t)cat1 * n_cols + col]);
            log_value += log(penalty[(size_t)cat2 * n_cols + col]);
        } else {
            for (int covariate = 0; covariate < n_covariates; ++covariate) {
                int cat = row_cats[covariate];
                log_value += log(penalty[(size_t)cat * n_cols + col]);
            }
        }

        R_out[(size_t)row * n_cols + col] = log_value;
        local_max = fmax(local_max, log_value);
    }

#pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1)
        local_max =
            fmax(local_max, __shfl_down_sync(0xffffffff, local_max, offset));

    __shared__ T shared_reduce[32];
    __shared__ T shared_row_max;
    int warp_id = threadIdx.x >> 5;
    int lane = threadIdx.x & 31;

    if (lane == 0) shared_reduce[warp_id] = local_max;
    __syncthreads();

    T row_max = static_cast<T>(-CUDART_INF);
    if (threadIdx.x < 32) {
        int num_warps = (blockDim.x + 31) >> 5;
        if (static_cast<int>(threadIdx.x) < num_warps)
            row_max = shared_reduce[threadIdx.x];
#pragma unroll
        for (int offset = 16; offset > 0; offset >>= 1)
            row_max =
                fmax(row_max, __shfl_down_sync(0xffffffff, row_max, offset));
    }

    if (threadIdx.x == 0) shared_row_max = row_max;
    __syncthreads();
    row_max = shared_row_max;

    T local_sum = T(0);
    for (int col = threadIdx.x; col < n_cols; col += blockDim.x) {
        T value = exp(R_out[(size_t)row * n_cols + col] - row_max);
        R_out[(size_t)row * n_cols + col] = value;
        local_sum += value;
    }

#pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1)
        local_sum += __shfl_down_sync(0xffffffff, local_sum, offset);

    if (lane == 0) shared_reduce[warp_id] = local_sum;
    __syncthreads();

    T row_sum = T(0);
    if (threadIdx.x < 32) {
        int num_warps = (blockDim.x + 31) >> 5;
        if (static_cast<int>(threadIdx.x) < num_warps)
            row_sum = shared_reduce[threadIdx.x];
#pragma unroll
        for (int offset = 16; offset > 0; offset >>= 1)
            row_sum += __shfl_down_sync(0xffffffff, row_sum, offset);
    }

    if (threadIdx.x == 0) shared_reduce[0] = row_sum;
    __syncthreads();
    row_sum = shared_reduce[0];

    T inv_sum = T(1) / row_sum;
    for (int col = threadIdx.x; col < n_cols; col += blockDim.x)
        R_out[(size_t)row * n_cols + col] *= inv_sum;
}
