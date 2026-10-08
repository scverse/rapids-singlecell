#pragma once

// Batched univariate two-component Gaussian mixtures reproducing
// mclust::Mclust(x, G = 2) for dsb. Clean-room: EM (Dempster et al. 1977),
// models E/V and BIC (Fraley & Raftery 2002; Scrucca et al. 2016); start
// values, stopping rule, singularity threshold and final update were found by
// black-box runs of Mclust (see _mixture_fits in preprocessing/_dsb.py).

#include <cooperative_groups.h>
#include <cuda_runtime.h>
#include <math_constants.h>

#include <cfloat>

namespace dsb {

namespace cg = cooperative_groups;

constexpr int kWarp = 32;
constexpr double kTol = 1e-5;  // relative log-likelihood change
// One more M-step when sum((pro - colMeans(z))^2) exceeds sqrt(DBL_EPSILON).
constexpr double kExtraMStepTol = 1.4901161193847656e-08;

template <typename T>
struct Mix2 {
    T mu0, mu1, s0, s1, pro0, pro1;
};

template <typename T>
__device__ __forceinline__ bool mix2_valid(const Mix2<T>& p) {
    const T eps = sizeof(T) == 8 ? T(DBL_EPSILON) : T(FLT_EPSILON);
    return isfinite(p.mu0) && isfinite(p.mu1) && isfinite(p.s0) &&
           isfinite(p.s1) && p.s0 > eps && p.s1 > eps && p.pro0 > T(0) &&
           p.pro1 > T(0);
}

// Xor butterfly: every lane gets the bit-identical total.
template <typename V, int N>
__device__ __forceinline__ void warp_sum(V (&v)[N]) {
#pragma unroll
    for (int off = kWarp / 2; off > 0; off >>= 1)
#pragma unroll
        for (int k = 0; k < N; ++k) v[k] += __shfl_xor_sync(~0u, v[k], off);
}

// Block-wide sum of N <= 7 doubles, broadcast to all threads.
template <int N>
__device__ __forceinline__ void block_sum(double (&v)[N], double* scratch) {
    const int warp = threadIdx.x / kWarp, lane = threadIdx.x % kWarp;
    warp_sum(v);
    __syncthreads();
    if (lane == 0)
        for (int k = 0; k < N; ++k) scratch[k * kWarp + warp] = v[k];
    __syncthreads();
    for (int k = 0; k < N; ++k)
        v[k] = lane < blockDim.x / kWarp ? scratch[k * kWarp + lane] : 0.0;
    warp_sum(v);
}

// M-step from moments about shifts sh_k: s = (w0, w1, a0, a1, b0, b1) with
// w_k = sum z_k, a_k = sum z_k (x - sh_k), b_k = sum z_k (x - sh_k)^2.
template <typename T, typename Acc>
__device__ __forceinline__ Mix2<T> finish_mstep(const Acc (&s)[6], int n,
                                                bool equal_var, Acc sh0,
                                                Acc sh1) {
    const Acc d0 = s[2] / s[0], d1 = s[3] / s[1];
    const Acc ss0 = s[4] - s[2] * d0, ss1 = s[5] - s[3] * d1;
    Mix2<T> p;
    p.mu0 = T(sh0 + d0);
    p.mu1 = T(sh1 + d1);
    p.pro0 = T(s[0] / Acc(n));
    p.pro1 = T(s[1] / Acc(n));
    p.s0 = T(equal_var ? (ss0 + ss1) / Acc(n) : ss0 / s[0]);
    p.s1 = T(equal_var ? (ss0 + ss1) / Acc(n) : ss1 / s[1]);
    return p;
}

// Per-thread part of the E-step at p fused with the next M-step over
// x[begin:end:stride]: moments about the current means into s, the
// log-likelihood (always in double; it drives the stopping rule) into ll.
template <typename T, typename Acc>
__device__ __forceinline__ void em_accumulate(const T* __restrict__ x,
                                              long long begin, long long end,
                                              int stride, const Mix2<T>& p,
                                              Acc (&s)[6], double& ll) {
    const T c0 = log(p.pro0) - T(0.5) * log(T(2 * CUDART_PI) * p.s0);
    const T c1 = log(p.pro1) - T(0.5) * log(T(2 * CUDART_PI) * p.s1);
    const T h0 = T(0.5) / p.s0, h1 = T(0.5) / p.s1;
    T prod = T(1);  // product of (1 + e) <= 2, flushed every 32 values
    int cnt = 0;
    for (long long i = begin; i < end; i += stride) {
        const T xi = x[i];
        const T d0 = xi - p.mu0, d1 = xi - p.mu1;
        const T l0 = c0 - d0 * d0 * h0, l1 = c1 - d1 * d1 * h1;
        const bool first = l0 > l1;
        const T hi = first ? l0 : l1;
        const T e = exp((first ? l1 : l0) - hi);
        const T inv = T(1) / (T(1) + e);
        const T z0 = first ? inv : e * inv, z1 = first ? e * inv : inv;
        ll += double(hi);
        prod *= T(1) + e;
        if (++cnt == 32) {
            ll += double(log(prod));
            prod = T(1);
            cnt = 0;
        }
        const Acc w0 = Acc(z0) * Acc(d0), w1 = Acc(z1) * Acc(d1);
        s[0] += Acc(z0);
        s[1] += Acc(z1);
        s[2] += w0;
        s[3] += w1;
        s[4] += w0 * Acc(d0);
        s[5] += w1 * Acc(d1);
    }
    ll += double(log(prod));
}

// EM state of one model between iterations.
template <typename T>
struct FitState {
    Mix2<T> p;  // current parameters, the reported ones once done
    double ll, ll_prev;
    int it;  // -1 = singular / degenerate
    bool skip, have_prev, done;
};

// subset: started from a subset fit, whose log-likelihood is not compared.
template <typename T>
__device__ __forceinline__ FitState<T> fit_start(const Mix2<T>& p,
                                                 bool subset) {
    const bool ok = mix2_valid(p);
    return FitState<T>{p, 0.0, 0.0, ok ? 0 : -1, subset, false, !ok};
}

// One EM iteration given the log-likelihood at st.p and the next M-step pn.
template <typename T>
__device__ __forceinline__ void fit_step(FitState<T>& st, double ll,
                                         const Mix2<T>& pn, bool subset,
                                         int max_iter) {
    if (!isfinite(ll)) {
        st.it = -1;
        st.done = true;
        return;
    }
    if (st.skip) {
        st.skip = false;
    } else {
        ++st.it;
        if ((st.have_prev && fabs(ll - st.ll_prev) < kTol * (1 + fabs(ll))) ||
            st.it >= max_iter) {
            // pn.pro* are the column means of the final responsibilities
            const double d0 = double(st.p.pro0) - double(pn.pro0);
            const double d1 = double(st.p.pro1) - double(pn.pro1);
            if ((subset || d0 * d0 + d1 * d1 > kExtraMStepTol) &&
                mix2_valid(pn))
                st.p = pn;
            st.ll = ll;
            st.done = true;
            return;
        }
        st.have_prev = true;
        st.ll_prev = ll;
    }
    if (!mix2_valid(pn)) {
        st.it = -1;
        st.done = true;
        return;
    }
    st.p = pn;
}

// Keeps E (m = 0) or V (m = 1) by BIC and writes mu0, mu1, var0, var1, pro0,
// loglik; status 0 = E, 1 = V, -1 = both singular, +2 when the selected
// model stopped at max_iter instead of converging.
template <typename T>
__device__ void write_selected(const FitState<T> (&st)[2], int n, long long f,
                               int max_iter, T* params, int* status) {
    const double log_n = log(double(n));
    int sel = st[0].it >= 0 ? 0 : -1;
    if (st[1].it >= 0 &&
        (sel < 0 || 2 * st[1].ll - 5 * log_n > 2 * st[0].ll - 4 * log_n))
        sel = 1;
    const FitState<T>& s = st[sel < 0 ? 0 : sel];
    const T v[6] = {s.p.mu0, s.p.mu1, s.p.s0, s.p.s1, s.p.pro0, T(s.ll)};
    for (int k = 0; k < 6; ++k)
        params[f * 6 + k] = sel < 0 ? T(CUDART_NAN) : v[k];
    status[f] = sel < 0 ? -1 : sel + (s.it >= max_iter ? 2 : 0);
}

// Mclust's univariate start split: the type 7 quantile at the fraction j / k
// with the smallest k >= 2 for which min < q < max (the median for untied
// data); NaN when all values are equal.
template <typename T>
__device__ T warp_threshold(const T* __restrict__ x, int n, int lane) {
    T mn = CUDART_INF, mx = -CUDART_INF;
    for (int i = lane; i < n; i += kWarp) {
        mn = fmin(mn, x[i]);
        mx = fmax(mx, x[i]);
    }
    for (int off = kWarp / 2; off > 0; off >>= 1) {
        mn = fmin(mn, __shfl_xor_sync(~0u, mn, off));
        mx = fmax(mx, __shfl_xor_sync(~0u, mx, off));
    }
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
    return rem == 0 ? x_lo : x_lo + T(rem) / T(k) * (x_hi - x_lo);
}

// One warp per row of X (n <= 2000). Writes the start split to q. With
// `init`, only writes the E and V start parameters init[f, model, 6] (subset
// start of long vectors); otherwise fits and selects.
template <typename T>
__global__ void mix2_warp_kernel(const T* __restrict__ X, long long n_fits,
                                 int n, int max_iter, T* params, int* status,
                                 T* q, T* init) {
    const int lane = threadIdx.x % kWarp;
    const int warps = blockDim.x / kWarp;
    for (long long f = (long long)blockIdx.x * warps + threadIdx.x / kWarp;
         f < n_fits; f += (long long)gridDim.x * warps) {
        const T* x = X + f * n;
        const T qf = warp_threshold(x, n, lane);
        if (lane == 0) q[f] = qf;
        T s[6] = {T(0), T(0), T(0), T(0), T(0), T(0)};  // hard split x < q
        for (int i = lane; i < n; i += kWarp) {
            const int k = x[i] < qf ? 0 : 1;
            const T d = x[i] - qf;
            s[k] += T(1);
            s[2 + k] += d;
            s[4 + k] += d * d;
        }
        warp_sum(s);
        FitState<T> st[2];
        for (int m = 0; m < 2; ++m) {
            const Mix2<T> p0 = finish_mstep<T, T>(s, n, m == 0, qf, qf);
            if (init != nullptr) {
                const T v[6] = {p0.mu0, p0.mu1, p0.s0, p0.s1, p0.pro0, p0.pro1};
                if (lane == 0)
                    for (int k = 0; k < 6; ++k) init[f * 12 + m * 6 + k] = v[k];
                continue;
            }
            st[m] = fit_start(p0, false);
            while (!st[m].done) {
                T a[6] = {T(0), T(0), T(0), T(0), T(0), T(0)};
                double ll[1] = {0.0};
                em_accumulate<T, T>(x, lane, n, kWarp, st[m].p, a, ll[0]);
                warp_sum(a);
                warp_sum(ll);
                fit_step(
                    st[m], ll[0],
                    finish_mstep<T, T>(a, n, m == 0, st[m].p.mu0, st[m].p.mu1),
                    false, max_iter);
            }
        }
        if (init == nullptr && lane == 0)
            write_selected(st, n, f, max_iter, params, status);
    }
}

// Long vectors, launched cooperatively from start parameters init[f, model].
// Every iteration splits each unfinished (fit, model) pair into n_chunks
// chunks; blocks reduce (pair, chunk) items into partials, and after a grid
// barrier each pair is advanced from its partials summed in chunk order
// (deterministic).
template <typename T>
__global__ void mix2_coop_kernel(const T* __restrict__ X, long long n_fits,
                                 int n, int max_iter, long long chunk,
                                 long long n_chunks, const T* __restrict__ init,
                                 FitState<T>* states, double* partials,
                                 T* params, int* status) {
    cg::grid_group grid = cg::this_grid();
    __shared__ double scratch[7 * kWarp];
    __shared__ int active;
    const long long n_pairs = 2 * n_fits, n_items = n_pairs * n_chunks;
    const long long gtid = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    const long long gsize = (long long)gridDim.x * blockDim.x;
    for (long long pr = gtid; pr < n_pairs; pr += gsize) {
        const T* ip = init + pr * 6;
        states[pr] =
            fit_start(Mix2<T>{ip[0], ip[1], ip[2], ip[3], ip[4], ip[5]}, true);
    }
    grid.sync();
    while (true) {
        for (long long item = blockIdx.x; item < n_items; item += gridDim.x) {
            const long long pr = item / n_chunks;
            const long long begin = item % n_chunks * chunk;
            if (states[pr].done) continue;  // uniform: read after a barrier
            T a[6] = {T(0), T(0), T(0), T(0), T(0), T(0)};  // few values each
            double s[7] = {0.0};
            em_accumulate<T, T>(X + pr / 2 * n, begin + threadIdx.x,
                                begin + chunk < n ? begin + chunk : n,
                                blockDim.x, states[pr].p, a, s[0]);
            for (int k = 0; k < 6; ++k) s[k + 1] = double(a[k]);
            block_sum(s, scratch);
            if (threadIdx.x == 0)
                for (int k = 0; k < 7; ++k) partials[item * 7 + k] = s[k];
        }
        grid.sync();
        for (long long pr = gtid; pr < n_pairs; pr += gsize) {
            FitState<T> st = states[pr];
            if (st.done) continue;
            double ll = 0.0, s[6] = {0.0};
            for (long long c = 0; c < n_chunks; ++c) {
                const double* in = partials + (pr * n_chunks + c) * 7;
                ll += in[0];
                for (int k = 0; k < 6; ++k) s[k] += in[k + 1];
            }
            fit_step(
                st, ll,
                finish_mstep<T, double>(s, n, pr % 2 == 0, st.p.mu0, st.p.mu1),
                true, max_iter);
            states[pr] = st;
        }
        grid.sync();
        if (threadIdx.x == 0) {
            active = 0;
            for (long long pr = 0; pr < n_pairs; ++pr)
                active |= !states[pr].done;
        }
        __syncthreads();
        if (!active) break;
    }
    for (long long f = gtid; f < n_fits; f += gsize) {
        const FitState<T> st[2] = {states[2 * f], states[2 * f + 1]};
        write_selected(st, n, f, max_iter, params, status);
    }
}

}  // namespace dsb
