#pragma once

#include <cstdint>
#include <type_traits>

#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <mma.h>

// Batched log-domain Sinkhorn kernels -- ragged (flat, no-padding) layout.
// Everything is stored flat with per-pair int64 offsets (cost_off/f_off/g_off)
// and int32 sizes (n/m); no masks. row2pair/col2pair map a flat row/column to
// its pair. Per-pair conv[b] short-circuits converged pairs.

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

// Normalize lane sums to the warp maximum, then sum into lane zero.
template <typename T>
__device__ __forceinline__ void warp_lse(T& rmax, T& rsum) {
    T warp_max = rmax;
#pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1)
        warp_max =
            fmax(warp_max, __shfl_xor_sync(0xffffffff, warp_max, offset));
    rsum *= exp(rmax - warp_max);
    rmax = warp_max;
#pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1)
        rsum += __shfl_down_sync(0xffffffff, rsum, offset);
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
            // A NaN ground cost makes only this pair's objective undefined.
            eps[b] = isnan(v1) ? v1 : fmax(scale * sqrt(var), floor);
        }
    }
}

// g[b, j] = eps * (log_b - LSE_i (-C[i, j] + f[i]) / eps).
// A warp shares a column in small batches. Large batches have enough columns
// to assign one per thread and coalesce reads across neighboring columns.
template <typename T, bool WARP_PER_COL>
__global__ void update_g_kernel(
    const T* __restrict__ cost, const int64_t* __restrict__ cost_off,
    const int* __restrict__ n, const int* __restrict__ m,
    const T* __restrict__ f, const int64_t* __restrict__ f_off,
    T* __restrict__ g, const int64_t* __restrict__ g_off,
    const int* __restrict__ col2pair, const T* __restrict__ eps,
    const T* __restrict__ log_b, const int* __restrict__ conv, T omega,
    int64_t total_cols, T* __restrict__ g_prev) {
    constexpr int lanes = WARP_PER_COL ? 32 : 1;
    const int lane = threadIdx.x % lanes;
    const int64_t c = ((int64_t)blockIdx.x * blockDim.x + threadIdx.x) / lanes;
    if (c >= total_cols) return;
    const int b = col2pair[c];
    if (conv[b]) return;
    const int j = (int)(c - g_off[b]);
    const int n_b = n[b], m_b = m[b];
    const T* Cb = cost + cost_off[b];
    const T* fb = f + f_off[b];
    const T inv_eps = T(1) / eps[b];
    T rmax = sk_neg<T>(), rsum = 0;
    for (int i = lane; i < n_b; i += lanes)
        lse_acc<T>((-Cb[(int64_t)i * m_b + j] + fb[i]) * inv_eps, rmax, rsum);
    if constexpr (WARP_PER_COL) {
        warp_lse(rmax, rsum);
        if (lane != 0) return;
    }
    const T lse = rsum > T(0) ? log(rsum) + rmax : sk_neg<T>();
    const T new_g = eps[b] * (log_b[b] - lse);
    T* gb = g + g_off[b];
    if (g_prev) g_prev[c] = gb[j];
    gb[j] = (omega == T(1)) ? new_g : (T(1) - omega) * gb[j] + omega * new_g;
}

// f[b, i] = eps * (log_a - LSE_j (-C[i, j] + g[j]) / eps).
// A warp handles a short row; a full block shares the reduction for long rows.
template <typename T, int ROW_THREADS>
__global__ void update_f_kernel(
    const T* __restrict__ cost, const int64_t* __restrict__ cost_off,
    const int* __restrict__ m, const T* __restrict__ g,
    const int64_t* __restrict__ g_off, T* __restrict__ f,
    const int64_t* __restrict__ f_off, const int* __restrict__ row2pair,
    const T* __restrict__ eps, const T* __restrict__ log_a,
    const int* __restrict__ conv, T omega, int64_t total_rows,
    T* __restrict__ f_prev) {
    const int64_t r =
        ((int64_t)blockIdx.x * blockDim.x + threadIdx.x) / ROW_THREADS;
    if (r >= total_rows) return;
    const int b = row2pair[r];
    if (conv[b]) return;
    const int64_t i = r - f_off[b];
    const int m_b = m[b];
    const T* Cbi = cost + cost_off[b] + i * m_b;
    const T* gb = g + g_off[b];
    const T inv_eps = T(1) / eps[b];
    T rmax = sk_neg<T>(), rsum = 0;
    const int lane = threadIdx.x % ROW_THREADS;
    for (int j = lane; j < m_b; j += ROW_THREADS)
        lse_acc<T>((gb[j] - Cbi[j]) * inv_eps, rmax, rsum);
    warp_lse(rmax, rsum);
    if constexpr (ROW_THREADS > 32) {
        constexpr int nwarps = ROW_THREADS / 32;
        __shared__ T wmax[nwarps], wsum[nwarps];
        if (lane % 32 == 0) {
            wmax[lane / 32] = rmax;
            wsum[lane / 32] = rsum;
        }
        __syncthreads();
        if (lane >= 32) return;
        rmax = lane < nwarps ? wmax[lane] : sk_neg<T>();
        rsum = lane < nwarps ? wsum[lane] : T(0);
        warp_lse(rmax, rsum);
    }
    if (lane == 0) {
        const T lse = rsum > T(0) ? log(rsum) + rmax : sk_neg<T>();
        const T new_f = eps[b] * (log_a[b] - lse);
        if (f_prev) f_prev[r] = f[r];
        f[r] = (omega == T(1)) ? new_f : (T(1) - omega) * f[r] + omega * new_f;
    }
}

#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 750
// Eight warps split the features of a 16x16 cost tile. Each warp stages its
// own 16x16 half operands, then reuses the space for its FP32 dot products.
template <typename Out>
__device__ __forceinline__ void tensor_cost_tile(
    const float* emb, const int* cl, const int* cr, int rows, int cols,
    int stride, int D, Out* cost, float* scratch, int scale_exponent) {
    namespace wmma = nvcuda::wmma;
    const int tid = threadIdx.y * 16 + threadIdx.x;
    const int warp = tid / 32, lane = tid % 32;
    float* tile = scratch + warp * 256;
    float* norms = scratch + 8 * 256;
    auto* operands = reinterpret_cast<__half*>(tile);
    wmma::fragment<wmma::accumulator, 16, 16, 16, float> dot;
    wmma::fill_fragment(dot, 0.f);
    float xnorm[4]{}, ynorm[4]{};
    const float* xp[4];
    const float* yp[4];
#pragma unroll
    for (int r = 0; r < 4; ++r) {
        const int row = 4 * r + lane / 8;
        xp[r] = emb + (int64_t)cl[min(row, rows - 1)] * D;
        yp[r] = emb + (int64_t)cr[min(row, cols - 1)] * D;
    }
    for (int k = 16 * warp; k < D; k += 128) {
#pragma unroll
        for (int r = 0; r < 4; ++r) {
#pragma unroll
            for (int h = 0; h < 2; ++h) {
                const int d = k + 2 * (lane % 8) + h;
                const __half x = __float2half_rn(d < D ? xp[r][d] : 0.f);
                const __half y = __float2half_rn(d < D ? yp[r][d] : 0.f);
                const float xf = __half2float(x), yf = __half2float(y);
                xnorm[r] += xf * xf;
                ynorm[r] += yf * yf;
                operands[2 * (32 * r + lane) + h] = x;
                operands[256 + 2 * (32 * r + lane) + h] = y;
            }
        }
        __syncwarp();
        wmma::fragment<wmma::matrix_a, 16, 16, 16, __half, wmma::row_major> a;
        wmma::fragment<wmma::matrix_b, 16, 16, 16, __half, wmma::col_major> b;
        wmma::load_matrix_sync(a, operands, 16);
        wmma::load_matrix_sync(b, operands + 256, 16);
        wmma::mma_sync(dot, a, b, dot);
        __syncwarp();
    }
    wmma::store_matrix_sync(tile, dot, 16, wmma::mem_row_major);
#pragma unroll
    for (int r = 0; r < 4; ++r) {
#pragma unroll
        for (int s = 4; s; s >>= 1) {
            xnorm[r] += __shfl_down_sync(0xffffffff, xnorm[r], s, 8);
            ynorm[r] += __shfl_down_sync(0xffffffff, ynorm[r], s, 8);
        }
        if (lane % 8 == 0) {
            const int row = 4 * r + lane / 8;
            norms[warp * 32 + row] = xnorm[r];
            norms[warp * 32 + 16 + row] = ynorm[r];
        }
    }
    __syncthreads();
    const int i = threadIdx.y, j = threadIdx.x;
    if (i >= rows || j >= cols) return;
    float xy = 0.f, norm = 0.f;
#pragma unroll
    for (int w = 0; w < 8; ++w) {
        xy += scratch[256 * w + tid];
        norm += norms[32 * w + i] + norms[32 * w + 16 + j];
    }
    float value = norm - 2.f * xy;
    // Use direct FP32 differences for cancellation, half overflow, and values
    // near the half subnormal range (the smallest normal is 2^-14).
    if (!(value > fmaxf(0.05f * norm, D * 0x1p-28f)) || !isfinite(value)) {
        value = 0.f;
        for (int d = 0; d < D; ++d) {
            const float diff =
                emb[(int64_t)cl[i] * D + d] - emb[(int64_t)cr[j] * D + d];
            value += diff * diff;
        }
    }
    cost[(int64_t)i * stride + j] =
        ldexp(static_cast<Out>(value), -2 * scale_exponent);
}
#endif

// Fused gather + squared-Euclidean cost: block t does the TILE x TILE tile
// (tile_pair/tile_i0/tile_j0) of one pair, streaming D in FEAT_TILE chunks so
// shared memory is independent of D. The shared row stride MUST be FEAT_TILE+1
// (odd) -- a multiple-of-32 stride collapses the tile columns into one bank.
template <typename T, typename Out, int TILE, int FEAT_TILE, bool TENSOR_CORES>
__global__ void pairwise_cost_kernel(
    const T* __restrict__ emb, const int* __restrict__ cidx_l,
    const int64_t* __restrict__ f_off, const int* __restrict__ cidx_r,
    const int64_t* __restrict__ g_off, const int* __restrict__ n,
    const int* __restrict__ m, const int64_t* __restrict__ cost_off,
    const int* __restrict__ tile_pair, const int* __restrict__ tile_i0,
    const int* __restrict__ tile_j0, Out* __restrict__ cost, int D,
    int scale_exponent) {
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
    extern __shared__ __align__(32) unsigned char smem_raw[];
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 750
    if constexpr (TENSOR_CORES && std::is_same_v<T, float>) {
        static_assert(TILE == 16 && 2 * TILE * FS >= 8 * (256 + 32));
        tensor_cost_tile(emb, cl + i0, cr + j0, n_b - i0, m_b - j0, m_b, D,
                         cost + cost_off[b] + (int64_t)i0 * m_b + j0,
                         reinterpret_cast<float*>(smem_raw), scale_exponent);
        return;
    }
#endif
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
    if (gi < n_b && gj < m_b) {
        Out value = s;
        if constexpr (TENSOR_CORES) value = ldexp(value, -2 * scale_exponent);
        cost[cost_off[b] + (int64_t)gi * m_b + gj] = value;
    }
}

// Per-pair convergence: one block per pair (no atomics -- each owns its
// conv[b]). Reduces the pair's f/g segments for max single-iteration change and
// max magnitude, sets conv[b] if change/(scale+1) < tol. The last update in a
// chunk saves f_prev/g_prev.
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
