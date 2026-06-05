#pragma once

#include <cstdint>

#include <cuda_runtime.h>

// Batched log-domain Sinkhorn kernels -- ragged (flat, no-padding) layout.
// Everything is stored flat with per-pair int64 offsets (cost_off/f_off/g_off)
// and int32 sizes (n/m); no masks. row2pair/col2pair map a flat row/column to
// its pair so update_f runs one block per global row and update_g one thread
// per global column. Per-pair conv[b] short-circuits converged pairs.

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

// eps[b] = scale * std(C) over pair b's contiguous cost block; floored.
// Grid: B (one block per pair). Block: BLOCK.
template <typename T, int BLOCK>
__global__ void auto_eps_kernel(const T* __restrict__ cost,
                                const int64_t* __restrict__ cost_off,
                                const int* __restrict__ n,
                                const int* __restrict__ m, T scale, T floor,
                                T* __restrict__ eps) {
    const int b = blockIdx.x;
    const T* Cb = cost + cost_off[b];
    const int64_t size = (int64_t)n[b] * m[b];
    T s1 = 0, s2 = 0;
    for (int64_t k = threadIdx.x; k < size; k += BLOCK) {
        const T c = Cb[k];
        s1 += c;
        s2 += c * c;
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
        const int nwarps = blockDim.x >> 5;
        T v1 = (threadIdx.x < nwarps) ? r1[threadIdx.x] : (T)0;
        T v2 = (threadIdx.x < nwarps) ? r2[threadIdx.x] : (T)0;
#pragma unroll
        for (int offset = 16; offset > 0; offset >>= 1) {
            v1 += __shfl_down_sync(0xffffffff, v1, offset);
            v2 += __shfl_down_sync(0xffffffff, v2, offset);
        }
        if (threadIdx.x == 0) {
            const T sz = (T)(size > 0 ? size : 1);
            const T mean = v1 / sz;
            const T var = fmax(v2 / sz - mean * mean, T(0));
            eps[b] = fmax(scale * sqrt(var), floor);
        }
    }
}

// g[b, j] = eps * (log_b - LSE_i (-C[i, j] + f[i]) / eps).
// Grid over the flat columns (one thread per global column); col2pair maps it
// to its pair. omega is the over-relaxation factor (omega == 1 = plain
// Sinkhorn, does not read the old g).
template <typename T>
__global__ void update_g_kernel(
    const T* __restrict__ cost, const int64_t* __restrict__ cost_off,
    const int* __restrict__ n, const int* __restrict__ m,
    const T* __restrict__ f, const int64_t* __restrict__ f_off,
    T* __restrict__ g, const int64_t* __restrict__ g_off,
    const int* __restrict__ col2pair, const T* __restrict__ eps,
    const T* __restrict__ log_b, const int* __restrict__ conv, T omega,
    int64_t total_cols) {
    const int64_t c = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (c >= total_cols) return;
    const int b = col2pair[c];
    if (conv[b]) return;
    const int j = (int)(c - g_off[b]);
    const int n_b = n[b], m_b = m[b];
    const T* Cb = cost + cost_off[b];
    const T* fb = f + f_off[b];
    const T inv_eps = T(1) / eps[b];
    T rmax = sk_neg<T>(), rsum = 0;
    for (int i = 0; i < n_b; ++i)
        lse_acc<T>((-Cb[(int64_t)i * m_b + j] + fb[i]) * inv_eps, rmax, rsum);
    const T lse = rsum > T(0) ? log(rsum) + rmax : sk_neg<T>();
    const T new_g = eps[b] * (log_b[b] - lse);
    T* gb = g + g_off[b];
    gb[j] = (omega == T(1)) ? new_g : (T(1) - omega) * gb[j] + omega * new_g;
}

// f[b, i] = eps * (log_a - LSE_j (-C[i, j] + g[j]) / eps).
// Grid = total flat rows (one block per global row); row2pair maps it to its
// pair; the block cooperatively reduces over that pair's m columns.
template <typename T, int BLOCK>
__global__ void update_f_kernel(
    const T* __restrict__ cost, const int64_t* __restrict__ cost_off,
    const int* __restrict__ m, const T* __restrict__ g,
    const int64_t* __restrict__ g_off, T* __restrict__ f,
    const int64_t* __restrict__ f_off, const int* __restrict__ row2pair,
    const T* __restrict__ eps, const T* __restrict__ log_a,
    const int* __restrict__ conv, T omega, int64_t total_rows) {
    const int64_t r = blockIdx.x;
    if (r >= total_rows) return;
    const int b = row2pair[r];
    if (conv[b]) return;
    const int64_t i = r - f_off[b];
    const int m_b = m[b];
    const T* Cbi = cost + cost_off[b] + i * m_b;
    const T* gb = g + g_off[b];
    const T inv_eps = T(1) / eps[b];
    T rmax = sk_neg<T>(), rsum = 0;
    for (int j = threadIdx.x; j < m_b; j += BLOCK)
        lse_acc<T>((-Cbi[j] + gb[j]) * inv_eps, rmax, rsum);
    // Block log-sum-exp reduction: warp shuffles, then one cross-warp pass.
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
            const T lse = vs > T(0) ? log(vs) + vm : sk_neg<T>();
            const T new_f = eps[b] * (log_a[b] - lse);
            T* fb = f + f_off[b];
            fb[i] = (omega == T(1)) ? new_f
                                    : (T(1) - omega) * fb[i] + omega * new_f;
        }
    }
}

// Fused gather + squared-Euclidean cost: block t does the TILE x TILE tile
// (tile_pair/tile_i0/tile_j0) of one pair, streaming D in FEAT_TILE chunks so
// shared memory is independent of D. The shared row stride MUST be FEAT_TILE+1
// (odd) -- a multiple-of-32 stride collapses the tile columns into one bank.
template <typename T, int TILE, int FEAT_TILE>
__global__ void pairwise_cost_kernel(
    const T* __restrict__ emb, const int* __restrict__ cidx_l,
    const int64_t* __restrict__ f_off, const int* __restrict__ cidx_r,
    const int64_t* __restrict__ g_off, const int* __restrict__ n,
    const int* __restrict__ m, const int64_t* __restrict__ cost_off,
    const int* __restrict__ tile_pair, const int* __restrict__ tile_i0,
    const int* __restrict__ tile_j0, T* __restrict__ cost, int D) {
    const int t = blockIdx.x;
    const int b = tile_pair[t];
    const int i0 = tile_i0[t];
    const int j0 = tile_j0[t];
    const int n_b = n[b], m_b = m[b];
    const int* cl = cidx_l + f_off[b];
    const int* cr = cidx_r + g_off[b];
    const int ti = threadIdx.y;
    const int tj = threadIdx.x;
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
            const int rl = cl[i0 + r < n_b ? i0 + r : n_b - 1];
            for (int d = tid; d < fcount; d += nthreads)
                Xs[r * FS + d] = emb[(int64_t)rl * D + f0 + d];
            const int rr = cr[j0 + r < m_b ? j0 + r : m_b - 1];
            for (int d = tid; d < fcount; d += nthreads)
                Ys[r * FS + d] = emb[(int64_t)rr * D + f0 + d];
        }
        __syncthreads();
        if (gi < n_b && gj < m_b) {
            for (int d = 0; d < fcount; ++d) {
                const T diff = Xs[ti * FS + d] - Ys[tj * FS + d];
                s += diff * diff;
            }
        }
        __syncthreads();
    }
    if (gi < n_b && gj < m_b) cost[cost_off[b] + (int64_t)gi * m_b + gj] = s;
}

// Per-pair convergence: one block per pair (no atomics -- each owns its
// conv[b]). Reduces the pair's f/g segments for max single-iteration change and
// max magnitude, sets conv[b] if change/(scale+1) < tol. Caller snapshots
// f_prev/g_prev.
template <typename T, int BLOCK>
__global__ void converged_kernel(
    const T* __restrict__ f, const T* __restrict__ f_prev,
    const int64_t* __restrict__ f_off, const int* __restrict__ n,
    const T* __restrict__ g, const T* __restrict__ g_prev,
    const int64_t* __restrict__ g_off, const int* __restrict__ m, T tol,
    int* __restrict__ conv) {
    const int b = blockIdx.x;
    if (conv[b]) return;
    const int n_b = n[b], m_b = m[b];
    const T* fb = f + f_off[b];
    const T* fp = f_prev + f_off[b];
    const T* gb = g + g_off[b];
    const T* gp = g_prev + g_off[b];
    T change = 0, scale = 0;
    for (int i = threadIdx.x; i < n_b; i += BLOCK) {
        const T v = fb[i];
        change = fmax(change, fabs(v - fp[i]));
        scale = fmax(scale, fabs(v));
    }
    for (int j = threadIdx.x; j < m_b; j += BLOCK) {
        const T v = gb[j];
        change = fmax(change, fabs(v - gp[j]));
        scale = fmax(scale, fabs(v));
    }
    // Block max-reduction: warp shuffles, then one cross-warp pass.
#pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1) {
        change = fmax(change, __shfl_down_sync(0xffffffff, change, offset));
        scale = fmax(scale, __shfl_down_sync(0xffffffff, scale, offset));
    }
    __shared__ T sc[32];
    __shared__ T ss[32];
    if ((threadIdx.x & 31) == 0) {
        sc[threadIdx.x >> 5] = change;
        ss[threadIdx.x >> 5] = scale;
    }
    __syncthreads();
    if (threadIdx.x < 32) {
        const int nwarps = blockDim.x >> 5;
        T c = (threadIdx.x < nwarps) ? sc[threadIdx.x] : T(0);
        T s = (threadIdx.x < nwarps) ? ss[threadIdx.x] : T(0);
#pragma unroll
        for (int offset = 16; offset > 0; offset >>= 1) {
            c = fmax(c, __shfl_down_sync(0xffffffff, c, offset));
            s = fmax(s, __shfl_down_sync(0xffffffff, s, offset));
        }
        if (threadIdx.x == 0 && c / (s + T(1)) < tol) conv[b] = 1;
    }
}

}  // namespace sinkhorn
