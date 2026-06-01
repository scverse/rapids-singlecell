#pragma once

#include <cuda_runtime.h>

// Batched log-domain Sinkhorn kernels (nanobind port of the validated CuPy
// RawKernels). One launch processes all pairs in a batch; the Python driver
// (_sinkhorn.run_async) calls these per iteration on per-device streams to form
// an async multi-stream work queue.
//
//   update_g : one thread per (pair, column j), serial reduce over the small N.
//   update_f : one block per (pair, row i), the block cooperatively reduces
//              over the large M axis. Orient the larger group as M (caller's
//              job; the OT cost is symmetric) so both updates stay parallel.
//   auto_eps : per-pair epsilon = scale * std(C) over the real (masked) region.
//
// Masks are bool (true = real point). Padded potentials are set to a large
// negative sentinel so they drop out of every log-sum-exp; convergence and the
// reg_ot reduction (done in CuPy) mask them out too. Per-pair ``conv`` flags
// let the update kernels short-circuit converged pairs.

namespace sinkhorn {

template <typename T>
__device__ __forceinline__ T sk_neg();
template <>
__device__ __forceinline__ float sk_neg<float>() {
    return -1e30f;
}
template <>
__device__ __forceinline__ double sk_neg<double>() {
    return -1e300;
}

// Update one running (max, sum) log-sum-exp state with a new term ``m``.
template <typename T>
__device__ __forceinline__ void lse_acc(T m, T& rmax, T& rsum) {
    if (m > rmax) {
        rsum = rsum * exp(rmax - m) + T(1);
        rmax = m;
    } else {
        rsum += exp(m - rmax);
    }
}

// Merge a second log-sum-exp state (m2, s2) into (m1, s1) in place. The
// identity state is (sk_neg, 0). Used by the block reduction in update_f.
template <typename T>
__device__ __forceinline__ void lse_merge(T& m1, T& s1, T m2, T s2) {
    T mc = m1 > m2 ? m1 : m2;
    T sc = 0;
    if (m1 > sk_neg<T>()) sc += s1 * exp(m1 - mc);
    if (m2 > sk_neg<T>()) sc += s2 * exp(m2 - mc);
    m1 = (mc <= sk_neg<T>()) ? sk_neg<T>() : mc;
    s1 = (mc <= sk_neg<T>()) ? T(0) : sc;
}

// eps[b] = scale * std(C) over the real region; floored at ``floor``.
// total[b] = (# real rows) * (# real cols). Grid: B. Block: BLOCK.
template <typename T, int BLOCK>
__global__ void auto_eps_kernel(const T* __restrict__ cost,
                                const bool* __restrict__ mask_a,
                                const bool* __restrict__ mask_b,
                                const T* __restrict__ total, T scale, T floor,
                                T* __restrict__ eps, int N, int M) {
    const int b = blockIdx.x;
    const T* Cb = cost + (size_t)b * N * M;
    const bool* ma = mask_a + (size_t)b * N;
    const bool* mb = mask_b + (size_t)b * M;
    T s1 = 0, s2 = 0;
    for (int i = 0; i < N; ++i) {
        if (!ma[i]) continue;
        const T* Ci = Cb + (size_t)i * M;
        for (int j = threadIdx.x; j < M; j += BLOCK) {
            if (!mb[j]) continue;
            T c = Ci[j];
            s1 += c;
            s2 += c * c;
        }
    }
    // Block reduction of (s1, s2): warp shuffles, then one cross-warp pass.
#pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1) {
        s1 += __shfl_down_sync(0xffffffff, s1, offset);
        s2 += __shfl_down_sync(0xffffffff, s2, offset);
    }
    __shared__ T r1[32];
    __shared__ T r2[32];
    if ((threadIdx.x & 31) == 0) {
        r1[threadIdx.x >> 5] = s1;
        r2[threadIdx.x >> 5] = s2;
    }
    __syncthreads();
    if (threadIdx.x < 32) {
        T v1 = (threadIdx.x < (blockDim.x >> 5)) ? r1[threadIdx.x] : (T)0;
        T v2 = (threadIdx.x < (blockDim.x >> 5)) ? r2[threadIdx.x] : (T)0;
#pragma unroll
        for (int offset = 16; offset > 0; offset >>= 1) {
            v1 += __shfl_down_sync(0xffffffff, v1, offset);
            v2 += __shfl_down_sync(0xffffffff, v2, offset);
        }
        if (threadIdx.x == 0) {
            T tot = fmax(total[b], T(1));
            T mean = v1 / tot;
            T var = fmax(v2 / tot - mean * mean, T(0));
            T e = scale * sqrt(var);
            eps[b] = fmax(e, floor);
        }
    }
}

// g[b, j] = eps * (log_b - LSE_i (-C[i, j] + f[i]) / eps).
// Grid: (B, ceil(M / BLOCK)). One thread per (pair, column).
// omega is the over-relaxation factor: the new potential is blended as
// (1 - omega) * g_old + omega * g_sinkhorn. omega == 1 is plain Sinkhorn
// (the old value is not even read); omega in (1, 2) accelerates convergence.
template <typename T, int BLOCK>
__global__ void update_g_kernel(
    const T* __restrict__ cost, const bool* __restrict__ mask_a,
    const bool* __restrict__ mask_b, const T* __restrict__ f,
    const T* __restrict__ eps, const T* __restrict__ log_b,
    const int* __restrict__ conv, T* __restrict__ g, int N, int M, T omega) {
    const int b = blockIdx.x;
    if (conv[b]) return;
    const int j = blockIdx.y * blockDim.x + threadIdx.x;
    if (j >= M) return;
    const bool* mb = mask_b + (size_t)b * M;
    T* gb = g + (size_t)b * M;
    if (!mb[j]) {
        gb[j] = sk_neg<T>();
        return;
    }
    const T* Cb = cost + (size_t)b * N * M;
    const bool* ma = mask_a + (size_t)b * N;
    const T* fb = f + (size_t)b * N;
    const T inv_eps = T(1) / eps[b];
    T rmax = sk_neg<T>(), rsum = 0;
    for (int i = 0; i < N; ++i) {
        if (!ma[i]) continue;
        lse_acc<T>((-Cb[(size_t)i * M + j] + fb[i]) * inv_eps, rmax, rsum);
    }
    T lse = rsum > T(0) ? log(rsum) + rmax : sk_neg<T>();
    T new_g = eps[b] * (log_b[b] - lse);
    gb[j] = (omega == T(1)) ? new_g : (T(1) - omega) * gb[j] + omega * new_g;
}

// f[b, i] = eps * (log_a - LSE_j (-C[i, j] + g[j]) / eps).
// Grid: B * N. One block per (pair, row); block cooperatively reduces over M.
// omega is the over-relaxation factor (see update_g_kernel); omega == 1 is
// plain Sinkhorn and does not read the old f.
template <typename T, int BLOCK>
__global__ void update_f_kernel(
    const T* __restrict__ cost, const bool* __restrict__ mask_a,
    const bool* __restrict__ mask_b, const T* __restrict__ g,
    const T* __restrict__ eps, const T* __restrict__ log_a,
    const int* __restrict__ conv, T* __restrict__ f, int N, int M, T omega) {
    const int bi = blockIdx.x;
    const int b = bi / N;
    if (conv[b]) return;
    const int i = bi % N;
    const bool* ma = mask_a + (size_t)b * N;
    T* fb = f + (size_t)b * N;
    if (!ma[i]) {
        if (threadIdx.x == 0) fb[i] = sk_neg<T>();
        return;
    }
    const T* Cbi = cost + (size_t)b * N * M + (size_t)i * M;
    const bool* mb = mask_b + (size_t)b * M;
    const T* gb = g + (size_t)b * M;
    const T inv_eps = T(1) / eps[b];
    T rmax = sk_neg<T>(), rsum = 0;
    for (int j = threadIdx.x; j < M; j += BLOCK) {
        if (!mb[j]) continue;
        lse_acc<T>((-Cbi[j] + gb[j]) * inv_eps, rmax, rsum);
    }
    // Block log-sum-exp reduction of (rmax, rsum): warp shuffles, then one
    // cross-warp pass. Merging states (m1, s1) and (m2, s2):
    //   mc = max(m1, m2);  sc = s1*exp(m1-mc) + s2*exp(m2-mc)  (skip sk_neg).
    // The identity state is (sk_neg, 0), used to pad inactive lanes.
#pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1) {
        T om = __shfl_down_sync(0xffffffff, rmax, offset);
        T os = __shfl_down_sync(0xffffffff, rsum, offset);
        lse_merge<T>(rmax, rsum, om, os);
    }
    __shared__ T wmax[32];
    __shared__ T wsum[32];
    if ((threadIdx.x & 31) == 0) {
        wmax[threadIdx.x >> 5] = rmax;
        wsum[threadIdx.x >> 5] = rsum;
    }
    __syncthreads();
    if (threadIdx.x < 32) {
        const int nwarps = blockDim.x >> 5;
        T vm = (threadIdx.x < nwarps) ? wmax[threadIdx.x] : sk_neg<T>();
        T vs = (threadIdx.x < nwarps) ? wsum[threadIdx.x] : T(0);
#pragma unroll
        for (int offset = 16; offset > 0; offset >>= 1) {
            T om = __shfl_down_sync(0xffffffff, vm, offset);
            T os = __shfl_down_sync(0xffffffff, vs, offset);
            lse_merge<T>(vm, vs, om, os);
        }
        if (threadIdx.x == 0) {
            T lse = vs > T(0) ? log(vs) + vm : sk_neg<T>();
            T new_f = eps[b] * (log_a[b] - lse);
            fb[i] = (omega == T(1)) ? new_f
                                    : (T(1) - omega) * fb[i] + omega * new_f;
        }
    }
}

// Build a padded squared-Euclidean cost tensor in one fused pass (no cuBLAS):
//   cost[b, i, j] = sum_d (emb[cidx_l[b, i], d] - emb[cidx_r[b, j], d])^2
// (the 2-Wasserstein cost). Block is TILE x TILE. The D features are streamed
// in FEAT_TILE-wide chunks: each chunk caches TILE rows x FEAT_TILE features of
// both sides in shared memory (reused TILE-fold) and accumulates into a
// per-thread running sum, so shared memory stays bounded at 2 * TILE *
// FEAT_TILE * sizeof(T) regardless of D (the previous full-D cache overflowed
// shared memory for large D). Padded slots use clamped indices (masked out by
// the solver).
template <typename T, int TILE, int FEAT_TILE>
__global__ void pairwise_cost_kernel(const T* __restrict__ emb,
                                     const int* __restrict__ cidx_l,
                                     const int* __restrict__ cidx_r,
                                     T* __restrict__ cost, int N, int M,
                                     int D) {
    const int b = blockIdx.x;
    const int i0 = blockIdx.y * TILE;
    const int j0 = blockIdx.z * TILE;
    const int ti = threadIdx.y;
    const int tj = threadIdx.x;
    // Padded row stride (FEAT_TILE + 1, odd) so the column reads Ys[tj * FS +
    // d] hit distinct banks: a stride that is a multiple of 32 would serialize
    // the 16 columns of a tile into one bank.
    constexpr int FS = FEAT_TILE + 1;
    extern __shared__ __align__(sizeof(double)) unsigned char smem_raw[];
    T* Xs = reinterpret_cast<T*>(smem_raw);  // TILE * FS
    T* Ys = Xs + TILE * FS;                  // TILE * FS
    const int tid = ti * TILE + tj;
    const int nthreads = TILE * TILE;
    const int gi = i0 + ti, gj = j0 + tj;
    T s = T(0);
    for (int f0 = 0; f0 < D; f0 += FEAT_TILE) {
        const int fcount = D - f0 < FEAT_TILE ? D - f0 : FEAT_TILE;
        for (int r = 0; r < TILE; ++r) {
            const int rl =
                cidx_l[(size_t)b * N + (i0 + r < N ? i0 + r : N - 1)];
            for (int d = tid; d < fcount; d += nthreads)
                Xs[r * FS + d] = emb[(size_t)rl * D + f0 + d];
            const int rr =
                cidx_r[(size_t)b * M + (j0 + r < M ? j0 + r : M - 1)];
            for (int d = tid; d < fcount; d += nthreads)
                Ys[r * FS + d] = emb[(size_t)rr * D + f0 + d];
        }
        __syncthreads();
        if (gi < N && gj < M) {
            for (int d = 0; d < fcount; ++d) {
                const T diff = Xs[ti * FS + d] - Ys[tj * FS + d];
                s += diff * diff;
            }
        }
        __syncthreads();
    }
    if (gi < N && gj < M) cost[((size_t)b * N + gi) * M + gj] = s;
}

}  // namespace sinkhorn
