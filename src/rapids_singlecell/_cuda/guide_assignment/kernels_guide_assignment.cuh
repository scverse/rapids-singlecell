#pragma once

#include <cuda_runtime.h>

#include <cmath>

namespace guide_assignment {

constexpr int BLOCK_SIZE = 128;
constexpr int NUM_WARPS = BLOCK_SIZE / 32;
constexpr int NEWTON_STEPS = 12;
constexpr int PRIOR_SAMPLES = 15;
constexpr int N_QUANTILES = 5;
constexpr int N_PARAMS = 4;
constexpr int N_OUTPUTS = 6;
// Relative change of the loss below which the EM stops.
constexpr float TOL = 1e-7f;
constexpr float LOG_SQRT_2PI = 0.91893853320467274f;
constexpr double LOG_SQRT_2PI_D = 0.91893853320467274;
constexpr float SCALE_FLOOR = 1e-3f;
constexpr float LOG_SCALE_FLOOR = -6.9077552789821368f;
constexpr float WEIGHT_FLOOR = 1e-6f;

struct Params {
    float rate;
    float mu;
    float scale;
    float w_norm;
};

__device__ inline float quantile(int i) {
    constexpr float q[N_QUANTILES] = {0.5f, 0.75f, 0.9f, 0.97f, 0.99f};
    return q[i];
}

__device__ inline float warp_sum(float v) {
    for (int offset = 16; offset > 0; offset >>= 1) {
        v += __shfl_down_sync(0xffffffff, v, offset);
    }
    return v;
}

// Block-wide sums of N values, identical on every thread afterwards.
template <int N>
__device__ inline void block_sum(float (&s)[N], float (*partials)[4]) {
    const int lane = threadIdx.x & 31;
    const int warp = threadIdx.x >> 5;
#pragma unroll
    for (int k = 0; k < N; ++k) {
        s[k] = warp_sum(s[k]);
    }
    __syncthreads();
    if (lane == 0) {
#pragma unroll
        for (int k = 0; k < N; ++k) {
            partials[warp][k] = s[k];
        }
    }
    __syncthreads();
#pragma unroll
    for (int k = 0; k < N; ++k) {
        float total = 0.0f;
#pragma unroll
        for (int w = 0; w < NUM_WARPS; ++w) {
            total += partials[w][k];
        }
        s[k] = total;
    }
}

__device__ inline float block_max(float v, float (*partials)[4]) {
    const int lane = threadIdx.x & 31;
    const int warp = threadIdx.x >> 5;
    for (int offset = 16; offset > 0; offset >>= 1) {
        v = fmaxf(v, __shfl_down_sync(0xffffffff, v, offset));
    }
    __syncthreads();
    if (lane == 0) {
        partials[warp][0] = v;
    }
    __syncthreads();
    float result = partials[0][0];
#pragma unroll
    for (int w = 1; w < NUM_WARPS; ++w) {
        result = fmaxf(result, partials[w][0]);
    }
    return result;
}

// Inclusive prefix sum over the block; `total` receives the block-wide sum.
__device__ inline float block_inclusive_scan(float v, float& total,
                                             float (*partials)[4]) {
    const int lane = threadIdx.x & 31;
    const int warp = threadIdx.x >> 5;
    for (int offset = 1; offset < 32; offset <<= 1) {
        const float n = __shfl_up_sync(0xffffffff, v, offset);
        if (lane >= offset) {
            v += n;
        }
    }
    __syncthreads();
    if (lane == 31) {
        partials[warp][0] = v;
    }
    __syncthreads();
    total = 0.0f;
#pragma unroll
    for (int w = 0; w < NUM_WARPS; ++w) {
        if (w < warp) {
            v += partials[w][0];
        }
        total += partials[w][0];
    }
    return v;
}

// Median of 15 samples stored `stride` apart, as the 8th smallest value.
__device__ inline float median15(const float* x, long long stride) {
    float a[PRIOR_SAMPLES];
    for (int i = 0; i < PRIOR_SAMPLES; ++i) {
        const float value = x[i * stride];
        int j = i;
        while (j > 0 && a[j - 1] > value) {
            a[j] = a[j - 1];
            --j;
        }
        a[j] = value;
    }
    return a[PRIOR_SAMPLES / 2];
}

// Start from prior medians, as crispat: 15 standard normal draws per parameter.
// draws is [3, n_init_seeds, 15, guides] for the rate, mean and scale.
__device__ inline Params prior_init(const float* draws, int n_init_seeds,
                                    int start, int guide, int n_guides) {
    const long long plane =
        static_cast<long long>(n_init_seeds) * PRIOR_SAMPLES * n_guides;
    const float* d = draws +
                     static_cast<long long>(start) * PRIOR_SAMPLES * n_guides +
                     guide;
    Params p;
    p.rate = expf(median15(d, n_guides));
    p.mu = 3.0f + 2.0f * median15(d + plane, n_guides);
    p.scale = expf(2.0f + median15(d + 2 * plane, n_guides));
    p.w_norm = 0.5f;
    return p;
}

// Start that splits the values at a weight quantile: Poisson below, Gaussian
// above.
__device__ inline Params data_init(const float* __restrict__ v,
                                   const float* __restrict__ w, int nv,
                                   float sum_w, float vmax, float q,
                                   float (*partials)[4]) {
    const float total = fmaxf(sum_w, 1.0f);
    float s[4] = {0.0f, 0.0f, 0.0f, 0.0f};
    float carry = 0.0f;
    for (int base = 0; base < nv; base += BLOCK_SIZE) {
        const int j = base + threadIdx.x;
        const float wj = j < nv ? w[j] : 0.0f;
        float chunk_total;
        const float cumulative =
            carry + block_inclusive_scan(wj, chunk_total, partials);
        if (j < nv) {
            const bool upper = (cumulative / total > q) && (wj > 0.0f);
            if (upper) {
                s[0] += wj;
                s[1] += wj * v[j];
            } else {
                s[2] += wj;
                s[3] += wj * v[j];
            }
        }
        carry += chunk_total;
    }
    block_sum<4>(s, partials);
    const float n_up = s[0];
    const float n_low = s[2];
    Params p;
    p.rate = fmaxf(s[3] / fmaxf(n_low, 1.0f), 0.05f);
    p.mu = n_up > 0.0f ? s[1] / fmaxf(n_up, 1.0f) : vmax;
    float ss[1] = {0.0f};
    carry = 0.0f;
    for (int base = 0; base < nv; base += BLOCK_SIZE) {
        const int j = base + threadIdx.x;
        const float wj = j < nv ? w[j] : 0.0f;
        float chunk_total;
        const float cumulative =
            carry + block_inclusive_scan(wj, chunk_total, partials);
        if (j < nv && (cumulative / total > q) && (wj > 0.0f)) {
            const float d = v[j] - p.mu;
            ss[0] += wj * d * d;
        }
        carry += chunk_total;
    }
    block_sum<1>(ss, partials);
    p.scale = fmaxf(sqrtf(ss[0] / fmaxf(n_up, 1.0f)), 0.5f);
    p.w_norm = fminf(fmaxf(n_up / total, 0.01f), 0.9f);
    return p;
}

// Log mixture density of one log2 count and the Gaussian responsibility.
__device__ inline void mixture_terms(float value, const Params& p,
                                     float log_rate, float log_scale,
                                     float log_w, float log1m_w, float& log_mix,
                                     float& resp) {
    const float log_pois = value * log_rate - p.rate - lgammaf(value + 1.0f);
    const float z = (value - p.mu) / p.scale;
    const float log_norm = -LOG_SQRT_2PI - log_scale - 0.5f * z * z;
    const float log_norm_weighted = log_w + log_norm;
    const float log_pois_weighted = log1m_w + log_pois;
    const float upper = fmaxf(log_norm_weighted, log_pois_weighted);
    log_mix =
        upper + log1pf(expf(-fabsf(log_norm_weighted - log_pois_weighted)));
    resp = expf(log_norm_weighted - log_mix);
}

// Priors of crispat's model: rate ~ LogNormal(0, 1), mu ~ Normal(3, 2),
// scale ~ LogNormal(2, 1) and Poisson and Gaussian weights ~ Dirichlet(0.9,
// 0.1), which give the exponents 0.9 - 1 and 0.1 - 1 here and 3 / 2^2 = 0.75
// and 1 / 2^2 = 0.25 in the update of mu. Additive constants are left out.
__device__ inline float log_prior(const Params& p) {
    const float log_scale = logf(p.scale);
    const float log_rate = logf(p.rate);
    const float d = (p.mu - 3.0f) / 2.0f;
    return -0.1f * log1pf(-p.w_norm) - 0.9f * logf(p.w_norm) - 0.5f * d * d -
           log_scale - 0.5f * (log_scale - 2.0f) * (log_scale - 2.0f) -
           log_rate - 0.5f * log_rate * log_rate;
}

// MAP log scale of the Gaussian: the root of a convex decreasing function.
__device__ inline float solve_log_scale(float sum_squares, float n_norm) {
    float v = 0.5f * logf(fmaxf(sum_squares, 1e-12f) / fmaxf(n_norm, 1e-12f));
    v = fminf(fmaxf(v, LOG_SCALE_FLOOR), 10.0f);
    for (int i = 0; i < NEWTON_STEPS; ++i) {
        const float e = sum_squares * expf(-2.0f * v);
        const float f = e - (v - 2.0f) - n_norm - 1.0f;
        const float next =
            fminf(fmaxf(v - f / (-2.0f * e - 1.0f), LOG_SCALE_FLOOR), 10.0f);
        if (next == v) {
            break;
        }
        v = next;
    }
    return v;
}

// MAP log rate of the Poisson: the root of a concave decreasing function.
__device__ inline float solve_log_rate(float sum_values, float n_pois) {
    float log_rate = logf(fmaxf(sum_values, 1e-12f) / fmaxf(n_pois, 1e-12f));
    log_rate = fminf(fmaxf(log_rate, -20.0f), 10.0f);
    for (int i = 0; i < NEWTON_STEPS; ++i) {
        const float e = n_pois * expf(log_rate);
        const float g = sum_values - 1.0f - log_rate - e;
        const float next =
            fminf(fmaxf(log_rate - g / (-1.0f - e), -20.0f), 10.0f);
        if (next == log_rate) {
            break;
        }
        log_rate = next;
    }
    return log_rate;
}

// Sums of w*log_mix, w*resp and w*resp*value, or of w*resp*(value - center)^2,
// w_rest = w - w*resp and w_rest*value, the Poisson sums summed per value as in
// pertpy rather than as differences of large totals.
template <bool SECOND_PASS>
__device__ inline void e_step_sums(const float* __restrict__ values,
                                   const float* __restrict__ weights, int nv,
                                   const Params& p, float center, float (&s)[3],
                                   float (*partials)[4]) {
    const float log_rate = logf(p.rate);
    const float log_scale = logf(p.scale);
    const float log_w = logf(p.w_norm);
    const float log1m_w = log1pf(-p.w_norm);
    s[0] = 0.0f;
    s[1] = 0.0f;
    s[2] = 0.0f;
    for (int j = threadIdx.x; j < nv; j += BLOCK_SIZE) {
        float log_mix;
        float resp;
        mixture_terms(values[j], p, log_rate, log_scale, log_w, log1m_w,
                      log_mix, resp);
        const float w_resp = weights[j] * resp;
        if (SECOND_PASS) {
            const float d = values[j] - center;
            const float w_rest = weights[j] - w_resp;
            s[0] += w_resp * d * d;
            s[1] += w_rest;
            s[2] += w_rest * values[j];
        } else {
            s[0] += weights[j] * log_mix;
            s[1] += w_resp;
            s[2] += w_resp * values[j];
        }
    }
    block_sum<3>(s, partials);
}

// One block per (guide, start): MAP-EM until the relative loss change is below
// TOL. Values are [guides, width] with n_values used entries per guide, outputs
// [starts, guides, 4] and [starts, guides].
__global__ void fit_poisson_gauss_kernel(
    const float* __restrict__ values, const float* __restrict__ weights,
    const int* __restrict__ n_values, const float* __restrict__ draws,
    float* __restrict__ params_out, float* __restrict__ loss_out, int n_guides,
    int width, int n_init_seeds, int n_iter) {
    const int guide = blockIdx.x;
    const int start = blockIdx.y;
    const int nv = n_values[guide];
    const float* v = values + static_cast<long long>(guide) * width;
    const float* w = weights + static_cast<long long>(guide) * width;
    const long long slot = static_cast<long long>(start) * n_guides + guide;

    __shared__ float partials[NUM_WARPS][4];

    float s[3] = {0.0f, 0.0f, 0.0f};
    float vmax = 0.0f;
    for (int j = threadIdx.x; j < nv; j += BLOCK_SIZE) {
        s[0] += w[j];
        vmax = fmaxf(vmax, v[j]);
    }
    block_sum<3>(s, partials);
    const float sum_w = s[0];
    vmax = block_max(vmax, partials);

    Params p = start < n_init_seeds
                   ? prior_init(draws, n_init_seeds, start, guide, n_guides)
                   : data_init(v, w, nv, sum_w, vmax,
                               quantile(start - n_init_seeds), partials);

    float previous = INFINITY;
    float current = 1e30f;
    for (int iteration = 0; iteration < n_iter; ++iteration) {
        e_step_sums<false>(v, w, nv, p, 0.0f, s, partials);
        const float loss = -(s[0] + log_prior(p));
        const float n_norm = s[1];
        const float sum_resp_values = s[2];
        const float precision = 1.0f / (p.scale * p.scale);
        const float mu_new = (sum_resp_values * precision + 0.75f) /
                             (n_norm * precision + 0.25f);
        e_step_sums<true>(v, w, nv, p, mu_new, s, partials);
        const float n_pois = s[1];
        const float w_new = fminf(
            fmaxf((n_norm - 0.9f) / (n_norm + n_pois - 1.0f), WEIGHT_FLOOR),
            1.0f - WEIGHT_FLOOR);
        const float scale_new =
            fmaxf(expf(solve_log_scale(s[0], n_norm)), SCALE_FLOOR);
        const float rate_new = expf(solve_log_rate(s[2], n_pois));
        p = Params{rate_new, mu_new, scale_new, w_new};
        previous = current;
        current = loss;
        if (!(fabsf(previous - current) > TOL * fabsf(current))) {
            break;
        }
    }

    e_step_sums<false>(v, w, nv, p, 0.0f, s, partials);
    if (threadIdx.x == 0) {
        params_out[slot * N_PARAMS] = p.rate;
        params_out[slot * N_PARAMS + 1] = p.mu;
        params_out[slot * N_PARAMS + 2] = p.scale;
        params_out[slot * N_PARAMS + 3] = p.w_norm;
        loss_out[slot] = -(s[0] + log_prior(p));
    }
}

__device__ inline bool posterior_above_half(double log_count, double rate,
                                            double mu, double scale,
                                            double w_norm) {
    const double log_norm =
        -LOG_SQRT_2PI_D - log(scale) -
        0.5 * ((log_count - mu) / scale) * ((log_count - mu) / scale);
    const double log_pois =
        log_count * log(rate) - rate - lgamma(log_count + 1.0);
    const double log_num = log(w_norm) + log_norm;
    const double other = log(1.0 - w_norm) + log_pois;
    const double upper = fmax(log_num, other);
    const double log_den = upper + log1p(exp(-fabs(log_num - other)));
    return exp(log_num - log_den) > 0.5;
}

// One block per guide: keep the admissible start with the lowest loss and scan
// counts 1..max_count for the first with a Gaussian posterior above 0.5.
// out is [guides, 6]: rate, mu, scale, w_norm, loss, threshold, and stays nan
// without an admissible start or threshold.
__global__ void select_threshold_kernel(
    const float* __restrict__ params, const float* __restrict__ loss,
    const long long* __restrict__ max_counts, float* __restrict__ out,
    int n_guides, int n_starts) {
    const int guide = blockIdx.x;
    __shared__ long long first;

    int best = -1;
    float best_loss = INFINITY;
    for (int start = 0; start < n_starts; ++start) {
        const long long slot = static_cast<long long>(start) * n_guides + guide;
        const float* q = params + slot * N_PARAMS;
        const bool admissible = isfinite(loss[slot]) && q[1] > q[0] &&
                                q[3] > 2.0f * WEIGHT_FLOOR &&
                                q[2] > 2.0f * SCALE_FLOOR;
        if (admissible && loss[slot] < best_loss) {
            best_loss = loss[slot];
            best = start;
        }
    }
    if (best < 0) {
        return;
    }
    const float* q =
        params + (static_cast<long long>(best) * n_guides + guide) * N_PARAMS;
    const long long max_count = max_counts[guide];
    if (threadIdx.x == 0) {
        first = max_count + 1;
    }
    __syncthreads();
    for (long long t = threadIdx.x + 1; t <= max_count; t += BLOCK_SIZE) {
        if (posterior_above_half(log2(static_cast<double>(t)), q[0], q[1], q[2],
                                 q[3])) {
            atomicMin(&first, t);
            break;
        }
    }
    __syncthreads();
    if (threadIdx.x == 0) {
        float* o = out + static_cast<long long>(guide) * N_OUTPUTS;
        for (int k = 0; k < N_PARAMS; ++k) {
            o[k] = q[k];
        }
        o[N_PARAMS] = best_loss;
        o[N_PARAMS + 1] = first <= max_count ? static_cast<float>(first) : NAN;
    }
}

}  // namespace guide_assignment
