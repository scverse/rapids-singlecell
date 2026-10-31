#pragma once

// Batched univariate two-component Gaussian mixtures reproducing
// mclust::Mclust(x, G = 2) for dsb. Clean-room: EM (Dempster et al. 1977),
// models E/V and BIC (Fraley & Raftery 2002; Scrucca et al. 2016); start
// values, stopping rule, singularity threshold and final update were found by
// black-box runs of Mclust (see _mixture_fits in preprocessing/_dsb.py).

#include <cuda_runtime.h>
#include <math_constants.h>

#include <cfloat>
#include <type_traits>

namespace dsb {

constexpr int kWarp = 32;
constexpr double kTol = 1e-5;  // relative log-likelihood change
// One more M-step when sum((pro - colMeans(z))^2) exceeds sqrt(DBL_EPSILON).
constexpr double kExtraMStepTol = 1.4901161193847656e-08;

// float32 computes in double-single (hi, lo) where float rounding would move
// Mclust's decisions (see _mixture_fits): the parameters, the deviations from
// the means and the component log-densities, plus compensated sums. exp and
// log1p stay in float. float64 computes natively.
struct dsf {
    float hi, lo;
    dsf() = default;
    __host__ __device__ constexpr dsf(float h, float l = 0.f) : hi(h), lo(l) {
    }
};

// Error-free transforms (Knuth two-sum, FMA two-product).
__device__ __forceinline__ dsf two_sum(float a, float b) {
    const float s = a + b, bb = s - a;
    return {s, (a - (s - bb)) + (b - bb)};
}
__device__ __forceinline__ dsf quick_two_sum(float a, float b) {
    const float s = a + b;
    return {s, b - (s - a)};
}
__device__ __forceinline__ dsf operator+(dsf a, dsf b) {
    dsf s = two_sum(a.hi, b.hi);
    const dsf t = two_sum(a.lo, b.lo);
    s = quick_two_sum(s.hi, s.lo + t.hi);
    return quick_two_sum(s.hi, s.lo + t.lo);
}
__device__ __forceinline__ dsf operator+(dsf a, float b) {
    const dsf s = two_sum(a.hi, b);
    return quick_two_sum(s.hi, s.lo + a.lo);
}
__device__ __forceinline__ dsf operator-(dsf a) {
    return {-a.hi, -a.lo};
}
__device__ __forceinline__ dsf operator-(dsf a, dsf b) {
    return a + (-b);
}
__device__ __forceinline__ dsf operator*(dsf a, dsf b) {
    const float p = a.hi * b.hi;
    const float e = fmaf(a.hi, b.lo, fmaf(a.lo, b.hi, fmaf(a.hi, b.hi, -p)));
    return quick_two_sum(p, e);
}
__device__ __forceinline__ dsf operator*(dsf a, float b) {
    const float p = a.hi * b;
    return quick_two_sum(p, fmaf(a.lo, b, fmaf(a.hi, b, -p)));
}
__device__ __forceinline__ dsf operator/(dsf a, dsf b) {
    const float q1 = a.hi / b.hi;
    const dsf r = a - b * q1;
    return quick_two_sum(q1, r.hi / b.hi);
}
__device__ __forceinline__ dsf shfl_xor(dsf v, int o) {
    return {__shfl_xor_sync(~0u, v.hi, o), __shfl_xor_sync(~0u, v.lo, o)};
}
__device__ __forceinline__ dsf shfl_from(dsf v, int src) {
    return {__shfl_sync(~0u, v.hi, src), __shfl_sync(~0u, v.lo, src)};
}
__device__ __forceinline__ double shfl_xor(double v, int o) {
    return __shfl_xor_sync(~0u, v, o);
}
__device__ __forceinline__ float shfl_xor(float v, int o) {
    return __shfl_xor_sync(~0u, v, o);
}

// Working type R of storage type T and conversions.
template <typename T>
using Work = std::conditional_t<std::is_same_v<T, float>, dsf, double>;
__device__ __forceinline__ double to_d(double v) {
    return v;
}
__device__ __forceinline__ double to_d(dsf v) {
    return double(v.hi) + v.lo;
}
__device__ __forceinline__ double lead(double v) {
    return v;
}
__device__ __forceinline__ float lead(dsf v) {
    return v.hi;
}
template <typename R>
__device__ __forceinline__ R from_d(double v) {
    if constexpr (std::is_same_v<R, dsf>) {
        const float h = float(v);
        return {h, float(v - h)};
    } else {
        return v;
    }
}
template <typename T>
__device__ __forceinline__ Work<T> lift(T v) {
    if constexpr (std::is_same_v<T, float>)
        return dsf{v, 0.f};
    else
        return v;
}
template <typename T>
__device__ __forceinline__ T lower(Work<T> v) {
    if constexpr (std::is_same_v<T, float>)
        return v.hi + v.lo;
    else
        return v;
}

template <typename R>
struct Mix2 {
    R mu0, mu1, s0, s1, pro0, pro1;
};

// Variances <= DBL_EPSILON are singular, as in R, for both precisions.
template <typename R>
__device__ __forceinline__ bool mix2_valid(const Mix2<R>& p) {
    using L = decltype(lead(p.mu0));
    const L v[6] = {lead(p.mu0), lead(p.mu1),  lead(p.s0),
                    lead(p.s1),  lead(p.pro0), lead(p.pro1)};
    return isfinite(v[0]) && isfinite(v[1]) && isfinite(v[2]) &&
           isfinite(v[3]) && v[2] > L(DBL_EPSILON) && v[3] > L(DBL_EPSILON) &&
           v[4] > 0 && v[5] > 0;
}

// Xor butterfly: every lane gets the bit-identical total.
template <typename V, int N>
__device__ __forceinline__ void warp_sum(V (&v)[N]) {
#pragma unroll
    for (int off = kWarp / 2; off > 0; off >>= 1)
#pragma unroll
        for (int k = 0; k < N; ++k) v[k] = v[k] + shfl_xor(v[k], off);
}

// M-step from moments about shifts sh_k: s = (w0, w1, a0, a1, b0, b1) with
// w_k = sum z_k, a_k = sum z_k (x - sh_k), b_k = sum z_k (x - sh_k)^2; n is
// the number of observations.
template <typename R>
__device__ __forceinline__ Mix2<R> finish_mstep(const R (&s)[6], R n,
                                                bool equal_var, R sh0, R sh1) {
    const R d0 = s[2] / s[0], d1 = s[3] / s[1];
    const R ss0 = s[4] - s[2] * d0, ss1 = s[5] - s[3] * d1;
    return Mix2<R>{sh0 + d0,
                   sh1 + d1,
                   equal_var ? (ss0 + ss1) / n : ss0 / s[0],
                   equal_var ? (ss0 + ss1) / n : ss1 / s[1],
                   s[0] / n,
                   s[1] / n};
}

// Per-lane part of the E-step at p fused with the next M-step over
// x[lane::32] (with multiplicities w, if given): moments about the current
// means into s, the log-likelihood (it drives the stopping rule) into ll.
template <typename T>
__device__ __forceinline__ void em_accumulate(const T* __restrict__ x,
                                              const T* __restrict__ w, int n,
                                              int lane, const Mix2<Work<T>>& p,
                                              Work<T> (&s)[6], Work<T>& ll) {
    using R = Work<T>;
    // per-iteration constants in double: a few scalar operations
    const R c0 =
        from_d<R>(log(to_d(p.pro0)) - 0.5 * log(2 * CUDART_PI * to_d(p.s0)));
    const R c1 =
        from_d<R>(log(to_d(p.pro1)) - 0.5 * log(2 * CUDART_PI * to_d(p.s1)));
    const R h0 = from_d<R>(0.5 / to_d(p.s0)), h1 = from_d<R>(0.5 / to_d(p.s1));
    T prod = T(1);  // unweighted: product of (1 + e) <= 2, flushed every 32
    int cnt = 0;
    for (int i = lane; i < n; i += kWarp) {
        const R xi = lift(x[i]);
        const R d0 = xi - p.mu0, d1 = xi - p.mu1;
        const R l0 = c0 - d0 * d0 * h0, l1 = c1 - d1 * d1 * h1;
        const R diff = l1 - l0;
        const bool first = lead(diff) < 0;  // l0 > l1
        const R hi = first ? l0 : l1;
        const T e = exp(first ? lower<T>(diff) : -lower<T>(diff));
        T inv = T(1) / (T(1) + e);
        if (w != nullptr) {
            inv *= w[i];
            ll = ll + (hi + lift(T(log1p(e)))) * lift(w[i]);
        } else {
            ll = ll + hi;
            prod *= T(1) + e;
            if (++cnt == 32) {
                ll = ll + lift(T(log(prod)));
                prod = T(1);
                cnt = 0;
            }
        }
        const T z0 = first ? inv : e * inv, z1 = first ? e * inv : inv;
        const R t0 = d0 * z0, t1 = d1 * z1;
        s[0] = s[0] + z0;
        s[1] = s[1] + z1;
        s[2] = s[2] + t0;
        s[3] = s[3] + t1;
        s[4] = s[4] + t0 * lead(d0);
        s[5] = s[5] + t1 * lead(d1);
    }
    ll = ll + lift(T(log(prod)));
}

// float32: double-single only where float rounding moves Mclust's decisions
// (the deviations d = x - mu, the log-densities l = c - d^2 h; the parameters
// are double-single state). l1 - l0 only feeds exp and is formed in float;
// moment terms are float products, accumulated with compensation.
__device__ __forceinline__ void acc_add(dsf& a, float x) {
    const dsf s = two_sum(a.hi, x);  // errors collect in lo
    a.hi = s.hi;
    a.lo += s.lo;
}
// a + b without the low-order cancellation correction: absolute error
// ~eps^2 max(|a|, |b|), enough for log-densities
__device__ __forceinline__ dsf sloppy_add(dsf a, dsf b) {
    const dsf s = two_sum(a.hi, b.hi);
    return quick_two_sum(s.hi, s.lo + (a.lo + b.lo));
}
__device__ __forceinline__ dsf sub_from(float x, dsf m) {  // x - m
    const dsf s = two_sum(x, -m.hi);
    return quick_two_sum(s.hi, s.lo - m.lo);
}
__device__ __forceinline__ dsf square(dsf d) {
    const float p = d.hi * d.hi;
    return quick_two_sum(p, fmaf(d.hi, d.hi, -p) + 2.f * d.hi * d.lo);
}

// Compile-time double-single constant.
__host__ __device__ constexpr dsf dsc(double v) {
    return dsf(float(v), float(v - double(float(v))));
}

// Natural log of a > 0 in double-single: exact power-of-two reduction to
// [sqrt(1/2), sqrt(2)), then log(m) = 2 atanh(r), r = (m - 1) / (m + 1),
// |r| < 0.172, by its series to 2^-46 relative.
__device__ __forceinline__ dsf ds_log(dsf a) {
    int e;
    const float m = frexpf(a.hi, &e);  // a.hi = m 2^e, m in [0.5, 1)
    if (m < 0.70710678f) --e;
    const dsf x(ldexpf(a.hi, -e), ldexpf(a.lo, -e));
    const dsf r = (x - dsf(1.f)) / (x + dsf(1.f));
    const dsf r2 = r * r;
    dsf t = dsc(1.0 / 21);
    t = t * r2 + dsc(1.0 / 19);
    t = t * r2 + dsc(1.0 / 17);
    t = t * r2 + dsc(1.0 / 15);
    t = t * r2 + dsc(1.0 / 13);
    t = t * r2 + dsc(1.0 / 11);
    t = t * r2 + dsc(1.0 / 9);
    t = t * r2 + dsc(1.0 / 7);
    t = t * r2 + dsc(1.0 / 5);
    t = t * r2 + dsc(1.0 / 3);
    t = t * r2 + dsf(1.f);
    return r * t * 2.f + dsc(0.69314718055994530942) * float(e);
}

template <>
__device__ __forceinline__ void em_accumulate<float>(
    const float* __restrict__ x, const float* __restrict__ w, int n, int lane,
    const Mix2<dsf>& p, dsf (&s)[6], dsf& ll) {
    // c = log(pro) - log(2 pi s) / 2 = log(pro^2 / (2 pi s)) / 2 and
    // h = 1 / (2 s), without float64: even lanes take component 0, odd lanes
    // component 1, then exchange
    const bool odd = lane & 1;
    const dsf pro = odd ? p.pro1 : p.pro0, var = odd ? p.s1 : p.s0;
    const dsf my_c = ds_log(pro * pro / (var * dsc(2 * CUDART_PI))) * 0.5f;
    const dsf my_h = dsf(0.5f) / var;
    const dsf c0 = shfl_from(my_c, 0), c1 = shfl_from(my_c, 1);
    const dsf h0 = shfl_from(my_h, 0), h1 = shfl_from(my_h, 1);
    for (int i = lane; i < n; i += kWarp) {
        const dsf d0 = sub_from(x[i], p.mu0), d1 = sub_from(x[i], p.mu1);
        const dsf l0 = sloppy_add(c0, -(square(d0) * h0));
        const dsf l1 = sloppy_add(c1, -(square(d1) * h1));
        // l1 - l0 as (hi, lo): e = exp(-|l1 - l0|) = exp(a) (1 + b) keeps the
        // low part, which rounding the argument to float would lose
        const dsf dl = two_sum(l1.hi, -l0.hi);
        const float dlo = dl.lo + (l1.lo - l0.lo);
        const bool first = dl.hi + dlo < 0.f;  // l0 > l1
        const dsf hi = first ? l0 : l1;
        const float ea = expf(first ? dl.hi : -dl.hi);
        const float e = fmaf(ea, first ? dlo : -dlo, ea);
        const float wi = w != nullptr ? w[i] : 1.f;
        const float inv = wi / (1.f + e);
        // log-likelihood term hi + log1p(e), compensated per value
        const dsf t = w != nullptr ? hi * wi : hi;
        acc_add(ll, t.hi);
        acc_add(ll, log1pf(e) * wi);
        ll.lo += t.lo;
        const float z0 = first ? inv : e * inv, z1 = first ? e * inv : inv;
        const float m0 = fmaf(z0, d0.lo, z0 * d0.hi);
        const float m1 = fmaf(z1, d1.lo, z1 * d1.hi);
        acc_add(s[0], z0);
        acc_add(s[1], z1);
        acc_add(s[2], m0);
        acc_add(s[3], m1);
        acc_add(s[4], m0 * d0.hi);
        acc_add(s[5], m1 * d1.hi);
    }
    ll = quick_two_sum(ll.hi, ll.lo);
    for (int k = 0; k < 6; ++k) s[k] = quick_two_sum(s[k].hi, s[k].lo);
}

// EM of one model from the start values p: the log-likelihood of a subset
// start is not compared and its extra M-step is always applied. Returns the
// EM iterations (-1 if singular or degenerate) and the parameters in p. Sets
// `marginal` when a stopping or update decision was within `margin`
// (relative) of its threshold.
template <typename T>
__device__ int fit_model(const T* __restrict__ x, const T* __restrict__ w,
                         int n, Work<T> n_obs, int lane, bool equal_var,
                         bool subset, int max_iter, double margin,
                         Mix2<Work<T>>& p, double& ll, bool& marginal) {
    using R = Work<T>;
    if (!mix2_valid(p)) return -1;
    R ll_prev = R(0);
    for (int it = subset ? -1 : 0;;) {
        R s[6] = {R(0), R(0), R(0), R(0), R(0), R(0)};
        R l[1] = {R(0)};
        em_accumulate(x, w, n, lane, p, s, l[0]);
        warp_sum(s);
        warp_sum(l);
        const Mix2<R> pn = finish_mstep(s, n_obs, equal_var, p.mu0, p.mu1);
        if (!isfinite(lead(l[0]))) return -1;
        if (++it > 0) {
            // |l_k - l_{k-1}| relative to the tolerance (in float for float32)
            const auto dl = fabs(lead(l[0] - ll_prev));
            const auto tol = decltype(dl)(kTol) * (1 + fabs(lead(l[0])));
            if (it > 1 && margin > 0 && fabs(dl / tol - 1) < margin)
                marginal = true;
            if ((it > 1 && dl < tol) || it >= max_iter) {
                // pn.pro* are the column means of the final responsibilities
                const double d0 = to_d(p.pro0 - pn.pro0);
                const double d1 = to_d(p.pro1 - pn.pro1);
                const double dp = d0 * d0 + d1 * d1;
                if (!subset && margin > 0 &&
                    fabs(dp / kExtraMStepTol - 1) < margin)
                    marginal = true;
                if ((subset || dp > kExtraMStepTol) && mix2_valid(pn)) p = pn;
                ll = to_d(l[0]);
                return it;
            }
            ll_prev = l[0];
        }
        if (!mix2_valid(pn)) return -1;
        p = pn;
    }
}

// Mclust's univariate start split: the type 7 quantile at the fraction j / k
// with the smallest k >= 2 for which min < q < max (the median for untied
// data); NaN when all values are equal. Order statistics by rank counting,
// O(n^2 / 32): start vectors have at most 2000 values. Sets `tied` when
// copies of the min, of the max or of q move the threshold or the split:
// float32 values rounded from float64 can merge into such ties.
template <typename T>
__device__ T warp_threshold(const T* __restrict__ x, int n, int lane,
                            bool& tied) {
    T mn = CUDART_INF, mx = -CUDART_INF;
    for (int i = lane; i < n; i += kWarp) {
        mn = fmin(mn, x[i]);
        mx = fmax(mx, x[i]);
    }
    for (int off = kWarp / 2; off > 0; off >>= 1) {
        mn = fmin(mn, __shfl_xor_sync(~0u, mn, off));
        mx = fmax(mx, __shfl_xor_sync(~0u, mx, off));
    }
    tied = false;
    if (!(mn < mx)) return T(CUDART_NAN);
    long long a = 0, b = n;  // copies of the min; first index of the max
    for (int base = 0; base < n; base += kWarp) {
        const int i = base + lane;
        a += __popc(__ballot_sync(~0u, i < n && x[i] == mn));
        b -= __popc(__ballot_sync(~0u, i < n && x[i] == mx));
    }
    // smallest k with an integer j in ((a - 1) k / (n - 1), b k / (n - 1))
    long long k = 2, j = 1;
    for (;; ++k) {
        j = (a - 1) * k / (n - 1) + 1;
        if (j * (n - 1) < b * k) break;
    }
    const long long lo = (n - 1) * j / k, rem = (n - 1) * j % k;
    T x_lo = -CUDART_INF, x_hi = -CUDART_INF;  // order statistics lo, lo + 1
    for (int i = lane; i < n; i += kWarp) {
        int less = 0, leq = 0;
        for (int t = 0; t < n; ++t) {
            less += x[t] < x[i];
            leq += x[t] <= x[i];
        }
        if (less <= lo && lo < leq) x_lo = x[i];
        if (less <= lo + 1 && lo + 1 < leq) x_hi = x[i];
    }
    for (int off = kWarp / 2; off > 0; off >>= 1) {
        x_lo = fmax(x_lo, __shfl_xor_sync(~0u, x_lo, off));
        x_hi = fmax(x_hi, __shfl_xor_sync(~0u, x_hi, off));
    }
    const T q = rem == 0 ? x_lo : x_lo + T(rem) / T(k) * (x_hi - x_lo);
    long long at_q = 0;
    for (int base = 0; base < n; base += kWarp) {
        const int i = base + lane;
        at_q += __popc(__ballot_sync(~0u, i < n && x[i] == q));
    }
    tied = a > 1 || n - b > 1 || at_q > 1;
    return q;
}

// One warp per row of X. The start values of models E (0) and V (1) are
// init_in[f, model, 6] if given, else the M-step of the hard split of the row
// at Mclust's threshold. With init_out, only writes those start values;
// otherwise runs EM on the row (with multiplicities W, if given) and writes
// the model with the larger BIC to params[f] = (mu0, mu1, var0, var1, pro0,
// pro1), NaN if both are singular, and to flags[f] bit 0 whether either model
// hit max_iter, bit 1 whether a decision was marginal (see fit_model; also a
// BIC tie, or with margin > 0 a tied start, see warp_threshold). With
// init_out, flags (if given) get only the tied start.
template <typename T>
__global__ void mix2_kernel(const T* __restrict__ X, const T* __restrict__ W,
                            long long n_fits, int n,
                            const T* __restrict__ init_in, T* init_out,
                            bool subset, int max_iter, double margin, T* params,
                            unsigned char* flags) {
    using R = Work<T>;
    const int lane = threadIdx.x % kWarp;
    const int warps = blockDim.x / kWarp;
    for (long long f = (long long)blockIdx.x * warps + threadIdx.x / kWarp;
         f < n_fits; f += (long long)gridDim.x * warps) {
        const T* x = X + f * n;
        const T* w = W != nullptr ? W + f * n : nullptr;
        Mix2<R> p[2];
        bool tied = false;
        if (init_in != nullptr) {
            for (int m = 0; m < 2; ++m) {
                const T* ip = init_in + f * 12 + m * 6;
                p[m] = Mix2<R>{lift(ip[0]), lift(ip[1]), lift(ip[2]),
                               lift(ip[3]), lift(ip[4]), lift(ip[5])};
            }
        } else {
            const T q = warp_threshold(x, n, lane, tied);
            const R rq = lift(q);
            R s[6] = {R(0), R(0), R(0), R(0), R(0), R(0)};  // hard split x < q
            for (int i = lane; i < n; i += kWarp) {
                const int k = x[i] < q ? 0 : 1;
                const R d = lift(x[i]) - rq;
                s[k] = s[k] + T(1);
                s[2 + k] = s[2 + k] + d;
                s[4 + k] = s[4 + k] + d * d;
            }
            warp_sum(s);
            for (int m = 0; m < 2; ++m)
                p[m] = finish_mstep(s, lift(T(n)), m == 0, rq, rq);
        }
        tied = tied && margin > 0;
        if (init_out != nullptr) {
            if (lane == 0 && tied && flags != nullptr) flags[f] |= 2;
            if (lane == 0)
                for (int m = 0; m < 2; ++m) {
                    const T v[6] = {lower<T>(p[m].mu0),  lower<T>(p[m].mu1),
                                    lower<T>(p[m].s0),   lower<T>(p[m].s1),
                                    lower<T>(p[m].pro0), lower<T>(p[m].pro1)};
                    for (int k = 0; k < 6; ++k)
                        init_out[f * 12 + m * 6 + k] = v[k];
                }
            continue;
        }
        R n_obs[1] = {lift(T(n))};
        if (w != nullptr) {
            n_obs[0] = R(0);
            for (int i = lane; i < n; i += kWarp) n_obs[0] = n_obs[0] + w[i];
            warp_sum(n_obs);
        }
        int it[2];
        double ll[2];
        bool marginal = tied;
        for (int m = 0; m < 2; ++m)
            it[m] = fit_model(x, w, n, n_obs[0], lane, m == 0, subset, max_iter,
                              margin, p[m], ll[m], marginal);
        if (lane == 0) {
            const double log_n = log(to_d(n_obs[0]));
            const double bic[2] = {2 * ll[0] - 4 * log_n,
                                   2 * ll[1] - 5 * log_n};
            int sel = it[0] >= 0 ? 0 : -1;
            if (it[1] >= 0 && (sel < 0 || bic[1] > bic[0])) sel = 1;
            if (it[0] >= 0 && it[1] >= 0 &&
                fabs(bic[1] - bic[0]) < margin * 1e-4 * (1 + fabs(bic[0])))
                marginal = true;
            const Mix2<R>& r = p[sel < 0 ? 0 : sel];
            const T v[6] = {lower<T>(r.mu0),  lower<T>(r.mu1),
                            lower<T>(r.s0),   lower<T>(r.s1),
                            lower<T>(r.pro0), lower<T>(r.pro1)};
            for (int k = 0; k < 6; ++k)
                params[f * 6 + k] = sel < 0 ? T(CUDART_NAN) : v[k];
            flags[f] |=
                (it[0] >= max_iter || it[1] >= max_iter) | (marginal ? 2 : 0);
        }
    }
}

}  // namespace dsb
