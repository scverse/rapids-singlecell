#pragma once

#include <cuda_runtime.h>

#include <cmath>

constexpr int BLOCK_SIZE = 256;
constexpr int NUM_WARPS = BLOCK_SIZE / 32;
constexpr int HIST_BINS = 4096;

template <typename T>
__device__ inline T warp_reduce_sum(T val) {
    for (int offset = warpSize / 2; offset > 0; offset >>= 1) {
        val += __shfl_down_sync(0xffffffff, val, offset);
    }
    return val;
}

template <typename T>
__device__ inline T warp_reduce_max(T val) {
    for (int offset = warpSize / 2; offset > 0; offset >>= 1) {
        val = max(val, __shfl_down_sync(0xffffffff, val, offset));
    }
    return val;
}

template <typename T>
__device__ inline T block_reduce_sum_thread0(T val, T* warp_sums) {
    int lane = threadIdx.x & (warpSize - 1);
    int warp = threadIdx.x >> 5;
    int n_warps = (blockDim.x + warpSize - 1) / warpSize;

    val = warp_reduce_sum(val);
    if (lane == 0) {
        warp_sums[warp] = val;
    }
    __syncthreads();

    val = threadIdx.x < n_warps ? warp_sums[threadIdx.x] : static_cast<T>(0);
    if (warp == 0) {
        val = warp_reduce_sum(val);
    }
    return val;
}

template <typename T>
__device__ inline T block_reduce_max_thread0(T val, T* warp_maxes) {
    int lane = threadIdx.x & (warpSize - 1);
    int warp = threadIdx.x >> 5;
    int n_warps = (blockDim.x + warpSize - 1) / warpSize;

    val = warp_reduce_max(val);
    if (lane == 0) {
        warp_maxes[warp] = val;
    }
    __syncthreads();

    val = threadIdx.x < n_warps ? warp_maxes[threadIdx.x] : static_cast<T>(0);
    if (warp == 0) {
        val = warp_reduce_max(val);
    }
    return val;
}

__device__ inline float poisson_log_prob(float value, float lam) {
    lam = fmaxf(lam, 1.0e-10f);
    return value * logf(lam) - lam - lgammaf(value + 1.0f);
}

__device__ inline float normal_log_prob(float value, float mu, float sigma) {
    constexpr float log_2pi = 1.8378770664093453f;
    sigma = fmaxf(sigma, 1.0e-10f);
    float z = (value - mu) / sigma;
    return -0.5f * z * z - logf(sigma) - 0.5f * log_2pi;
}

template <bool F_ORDER>
__device__ inline float load_guide_count(const float* __restrict__ X, int cell,
                                         int guide, int n_cells, int n_guides) {
    long long idx = F_ORDER ? static_cast<long long>(guide) * n_cells + cell
                            : static_cast<long long>(cell) * n_guides + guide;
    return X[idx];
}

template <bool F_ORDER>
__global__ void assign_threshold_dense_kernel(
    const float* __restrict__ X, const int* __restrict__ valid_guides,
    const float* __restrict__ lam, const float* __restrict__ mu,
    const float* __restrict__ sigma, const float* __restrict__ pi0,
    bool* __restrict__ assignments, float* __restrict__ thresholds, int n_cells,
    int n_guides, float posterior_threshold) {
    int valid_idx = blockIdx.x;
    int guide = valid_guides[valid_idx];
    int tid = threadIdx.x;

    __shared__ int max_counts[NUM_WARPS];
    __shared__ float guide_threshold;

    int local_max = 0;
    for (int cell = tid; cell < n_cells; cell += blockDim.x) {
        float count =
            load_guide_count<F_ORDER>(X, cell, guide, n_cells, n_guides);
        int count_int = static_cast<int>(ceilf(count));
        local_max = max(local_max, count_int);
    }
    int guide_max_count = block_reduce_max_thread0(local_max, max_counts);

    if (tid == 0) {
        float threshold = NAN;
        float guide_lam = lam[valid_idx];
        float guide_mu = mu[valid_idx];
        float guide_sigma = sigma[valid_idx];
        float guide_pi0 =
            fminf(fmaxf(pi0[valid_idx], 1.0e-10f), 1.0f - 1.0e-10f);
        float guide_pi1 = 1.0f - guide_pi0;

        for (int raw_count = 1; raw_count <= guide_max_count; ++raw_count) {
            float log_count = log2f(static_cast<float>(raw_count));
            float log_p0 =
                poisson_log_prob(log_count, guide_lam) + logf(guide_pi0);
            float log_p1 = normal_log_prob(log_count, guide_mu, guide_sigma) +
                           logf(guide_pi1);
            float posterior = 1.0f / (1.0f + expf(log_p0 - log_p1));
            if (posterior > posterior_threshold) {
                threshold = static_cast<float>(raw_count);
                break;
            }
        }

        thresholds[valid_idx] = threshold;
        guide_threshold = threshold;
    }
    __syncthreads();

    bool has_threshold = !isnan(guide_threshold);
    for (int cell = tid; cell < n_cells; cell += blockDim.x) {
        float count =
            load_guide_count<F_ORDER>(X, cell, guide, n_cells, n_guides);
        assignments[valid_idx * n_cells + cell] =
            has_threshold && count >= guide_threshold;
    }
}

template <bool F_ORDER>
__global__ void fit_assign_dense_kernel(
    const float* __restrict__ X, bool* __restrict__ assignments,
    float* __restrict__ thresholds, float* __restrict__ lam_out,
    float* __restrict__ mu_out, float* __restrict__ sigma_out,
    float* __restrict__ pi0_out, bool* __restrict__ valid_mask,
    int* __restrict__ nonzero_counts, int* __restrict__ max_counts, int n_cells,
    int n_guides, int max_iter, float tol, float posterior_threshold) {
    int guide = blockIdx.x;
    int tid = threadIdx.x;

    __shared__ float sum_log_warp[NUM_WARPS];
    __shared__ int nz_counts_warp[NUM_WARPS];
    __shared__ int raw_max_counts_warp[NUM_WARPS];
    __shared__ float red_n0_warp[NUM_WARPS];
    __shared__ float red_n1_warp[NUM_WARPS];
    __shared__ float red_sum_r0_y_warp[NUM_WARPS];
    __shared__ float red_sum_r1_y_warp[NUM_WARPS];
    __shared__ float red_sum_r1_y2_warp[NUM_WARPS];
    __shared__ int count_hist[HIST_BINS + 1];
    __shared__ float guide_threshold;
    __shared__ float guide_sum_log;
    __shared__ int guide_nz;
    __shared__ int guide_max_raw;

    float local_sum = 0.0f;
    int local_nz = 0;
    int local_max_raw = 0;

    for (int cell = tid; cell < n_cells; cell += blockDim.x) {
        float count =
            load_guide_count<F_ORDER>(X, cell, guide, n_cells, n_guides);
        if (count > 0.0f) {
            float log_count = log2f(count);
            local_sum += log_count;
            ++local_nz;
            local_max_raw = max(local_max_raw, static_cast<int>(ceilf(count)));
        }
    }

    float reduced_sum_log = block_reduce_sum_thread0(local_sum, sum_log_warp);
    int reduced_nz = block_reduce_sum_thread0(local_nz, nz_counts_warp);
    int reduced_max_raw =
        block_reduce_max_thread0(local_max_raw, raw_max_counts_warp);

    if (tid == 0) {
        guide_sum_log = reduced_sum_log;
        guide_nz = reduced_nz;
        guide_max_raw = reduced_max_raw;
        nonzero_counts[guide] = guide_nz;
        max_counts[guide] = guide_max_raw;
        valid_mask[guide] = guide_nz >= 2 && guide_max_raw >= 2;
    }
    __syncthreads();

    if (guide_nz < 2 || guide_max_raw < 2) {
        if (tid == 0) {
            thresholds[guide] = NAN;
            lam_out[guide] = NAN;
            mu_out[guide] = NAN;
            sigma_out[guide] = NAN;
            pi0_out[guide] = NAN;
            guide_threshold = NAN;
        }
        for (int cell = tid; cell < n_cells; cell += blockDim.x) {
            assignments[guide * n_cells + cell] = false;
        }
        return;
    }

    float n_valid = static_cast<float>(guide_nz);
    float mean_log = guide_sum_log / fmaxf(n_valid, 1.0f);
    float lam = fminf(fmaxf(mean_log * 0.5f, 0.01f), 0.5f);
    for (int bin = tid; bin <= HIST_BINS; bin += blockDim.x) {
        count_hist[bin] = 0;
    }
    __syncthreads();

    for (int cell = tid; cell < n_cells; cell += blockDim.x) {
        float count =
            load_guide_count<F_ORDER>(X, cell, guide, n_cells, n_guides);
        if (count > 0.0f) {
            int bin = min(max(static_cast<int>(ceilf(count)), 1), HIST_BINS);
            atomicAdd(&count_hist[bin], 1);
        }
    }
    __syncthreads();

    __shared__ float init_mu;
    if (tid == 0) {
        int target = static_cast<int>(n_valid * 0.75f);
        int cumulative = 0;
        int percentile_bin = 1;
        for (int bin = 1; bin <= HIST_BINS; ++bin) {
            cumulative += count_hist[bin];
            if (cumulative > target) {
                percentile_bin = bin;
                break;
            }
        }
        float p75 = log2f(static_cast<float>(percentile_bin));
        init_mu = p75 > 0.5f ? p75 : 3.0f;
    }
    __syncthreads();

    float mu = init_mu;
    float sigma = 1.0f;
    float pi0 = 0.85f;

    for (int iter = 0; iter < max_iter; ++iter) {
        float local_n0 = 0.0f;
        float local_n1 = 0.0f;
        float local_sum_r0_y = 0.0f;
        float local_sum_r1_y = 0.0f;
        float local_sum_r1_y2 = 0.0f;

        float safe_pi0 = fminf(fmaxf(pi0, 1.0e-10f), 1.0f - 1.0e-10f);
        float safe_pi1 = 1.0f - safe_pi0;

        for (int cell = tid; cell < n_cells; cell += blockDim.x) {
            float count =
                load_guide_count<F_ORDER>(X, cell, guide, n_cells, n_guides);
            if (count <= 0.0f) continue;

            float y = log2f(count);
            float log_p0 = poisson_log_prob(y, lam) + logf(safe_pi0);
            float log_p1 = normal_log_prob(y, mu, sigma) + logf(safe_pi1);
            float r1 = 1.0f / (1.0f + expf(log_p0 - log_p1));
            float r0 = 1.0f - r1;

            local_n0 += r0;
            local_n1 += r1;
            local_sum_r0_y += r0 * y;
            local_sum_r1_y += r1 * y;
            local_sum_r1_y2 += r1 * y * y;
        }

        float n0 = block_reduce_sum_thread0(local_n0, red_n0_warp);
        float n1 = block_reduce_sum_thread0(local_n1, red_n1_warp);
        float sum_r0_y =
            block_reduce_sum_thread0(local_sum_r0_y, red_sum_r0_y_warp);
        float sum_r1_y =
            block_reduce_sum_thread0(local_sum_r1_y, red_sum_r1_y_warp);
        float sum_r1_y2 =
            block_reduce_sum_thread0(local_sum_r1_y2, red_sum_r1_y2_warp);

        __shared__ float next_lam;
        __shared__ float next_mu;
        __shared__ float next_sigma;
        __shared__ float next_pi0;
        __shared__ bool converged;

        if (tid == 0) {
            float lam_mle = sum_r0_y / fmaxf(n0, 1.0e-10f);
            float log_lam_mle = logf(fmaxf(lam_mle, 1.0e-10f));
            next_lam = expf((n0 * log_lam_mle) / (n0 + 1.0f));

            float sigma_sq = sigma * sigma;
            next_mu = (sum_r1_y / fmaxf(sigma_sq, 1.0e-10f) + 3.0f / 4.0f) /
                      (n1 / fmaxf(sigma_sq, 1.0e-10f) + 1.0f / 4.0f);

            float sigma_sq_mle = (sum_r1_y2 - 2.0f * next_mu * sum_r1_y +
                                  next_mu * next_mu * n1) /
                                 fmaxf(n1, 1.0e-10f);
            sigma_sq_mle = fmaxf(sigma_sq_mle, 0.0f);
            float sigma_mle = fmaxf(sqrtf(sigma_sq_mle), 1.0e-2f);
            next_sigma = fmaxf(
                expf((n1 * logf(sigma_mle) + 2.0f) / (n1 + 1.0f)), 1.0e-2f);

            float denom = fmaxf(n_valid - 1.0f, 1.0e-10f);
            next_pi0 = (n0 - 0.1f) / denom;
            next_pi0 = fminf(fmaxf(next_pi0, 0.01f), 0.99f);

            float max_change =
                fmaxf(fmaxf(fabsf(next_lam - lam), fabsf(next_mu - mu)),
                      fmaxf(fabsf(next_sigma - sigma), fabsf(next_pi0 - pi0)));
            converged = max_change < tol;
        }
        __syncthreads();

        lam = next_lam;
        mu = next_mu;
        sigma = next_sigma;
        pi0 = next_pi0;
        bool done = converged;
        __syncthreads();
        if (done) break;
    }

    if (tid == 0) {
        float threshold = NAN;
        float safe_pi0 = fminf(fmaxf(pi0, 1.0e-10f), 1.0f - 1.0e-10f);
        float safe_pi1 = 1.0f - safe_pi0;

        for (int raw_count = 1; raw_count <= guide_max_raw; ++raw_count) {
            float log_count = log2f(static_cast<float>(raw_count));
            float log_p0 = poisson_log_prob(log_count, lam) + logf(safe_pi0);
            float log_p1 =
                normal_log_prob(log_count, mu, sigma) + logf(safe_pi1);
            float posterior = 1.0f / (1.0f + expf(log_p0 - log_p1));
            if (posterior > posterior_threshold) {
                threshold = static_cast<float>(raw_count);
                break;
            }
        }

        lam_out[guide] = lam;
        mu_out[guide] = mu;
        sigma_out[guide] = sigma;
        pi0_out[guide] = pi0;
        thresholds[guide] = threshold;
        guide_threshold = threshold;
    }
    __syncthreads();

    bool has_threshold = !isnan(guide_threshold);
    for (int cell = tid; cell < n_cells; cell += blockDim.x) {
        float count =
            load_guide_count<F_ORDER>(X, cell, guide, n_cells, n_guides);
        assignments[guide * n_cells + cell] =
            has_threshold && count >= guide_threshold;
    }
}
