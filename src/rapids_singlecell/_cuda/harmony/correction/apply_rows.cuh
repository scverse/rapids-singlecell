#pragma once

// Fused Harmony correction apply: for every cell i of a segment of group g,
//   z_d = X_id - sum_k R_ik W[g * w_group_stride + k * w_k_stride + d]
// (one fma chain over k = 0, 1, ..., K - 1 from 0), then optionally
// z *= min(rsqrt(sum_d z_d^2), 1e12) summed as add_rows_normalize_kernel does
// (lane l: fma chain over z_l^2, z_{l+32}^2, ...; xor butterfly 16 .. 1; the
// same bits for float with nvcc 13.4). Each element depends only on its row
// and W[g]: results do not depend on how rows are split into segments, shards
// or CTAs, nor on the launch configuration. CTAs take runs of row tiles of one
// segment; R tiles are staged in shared memory by cp.async, W[g] stays there
// (in k chunks when it does not fit; partial sums then wait in Z).

#include <cuda_bf16.h>
#include <cuda_pipeline.h>
#include <cuda_runtime.h>

#include <algorithm>
#include <cstddef>
#include <type_traits>

namespace harmony_apply {
namespace detail {

constexpr int NT = 256;  // 8 warps; warp w owns rows w * RM .. w * RM + RM - 1
constexpr int CPS = 4;   // CTAs per segment, each a contiguous run of tiles

template <typename T>
struct alignas(4 * sizeof(T)) V4 {  // 4 consecutive k: one 128-bit load
    T x[4];
};

// rows x n bytes (row stride sld) -> rows x dn bytes (zero padded) by cp.async
// in pieces of the widest V <= v dividing n, dn and the row starts (2: plain).
template <int V = 16>
__device__ __forceinline__ void stage(int v, unsigned char* dst, int dn,
                                      const void* r, size_t sld, int rows,
                                      int n) {
    if constexpr (V > 2)
        if (v < V) return stage<V / 2>(v, dst, dn, r, sld, rows, n);
    auto src = reinterpret_cast<const unsigned char*>(r);
    auto piece = [](unsigned char* d, const unsigned char* s) {
        if constexpr (V == 2)
            *reinterpret_cast<short*>(d) = *reinterpret_cast<const short*>(s);
        else
            __pipeline_memcpy_async(d, s, V);
    };
    if (sld == (size_t)n && n == dn) {  // contiguous
        for (int i = threadIdx.x * V; i < rows * n; i += NT * V)
            piece(dst + i, src + i);
        return;
    }
    for (int row = threadIdx.x >> 5; row < rows; row += NT / 32) {
        for (int o = (threadIdx.x & 31) * V; o < n; o += 32 * V)
            piece(dst + row * dn + o, src + row * sld + o);
        for (int o = n + (threadIdx.x & 31) * 2; o < dn; o += 64)
            *reinterpret_cast<short*>(dst + row * dn + o) = 0;
    }
}

// Lane partial sums of squares -> min(rsqrt(row sum), 1e12), NaN kept.
template <typename T>
__device__ __forceinline__ T inv_norm(T sq) {
    for (int o = 16; o > 0; o >>= 1) sq += __shfl_xor_sync(0xffffffff, sq, o);
    T s = rsqrt(sq);
    return s > T(1e12) ? T(1e12) : s;
}

// CHUNKED: W in several k chunks (kc < K rounded up to 4)
template <typename T, typename RT, int SLOTS, bool CHUNKED>
__global__ void __launch_bounds__(NT)
    apply_rows_kernel(const T* __restrict__ X, const RT* __restrict__ R,
                      int ldr, const T* __restrict__ W, long long wks,
                      long long wgs, int D, int K,
                      const int* __restrict__ seg_start,
                      const int* __restrict__ seg_group, bool normalize,
                      T* __restrict__ Z, int kc, int v) {
    constexpr int RM = SLOTS < 8 ? 4 : 2, TM = 8 * RM, DW = 32 * SLOTS;
    constexpr bool CONV = !std::is_same_v<T, RT>;
    static_assert(!CONV || std::is_same_v<T, float> &&
                               std::is_same_v<RT, __nv_bfloat16>);
    extern __shared__ __align__(16) unsigned char sm[];
    const int seg = blockIdx.x / CPS, part = blockIdx.x % CPS;
    const int start = seg_start[seg], end = seg_start[seg + 1];
    const int n_tiles = (end - start + TM - 1) / TM;
    const int t0 = part * n_tiles / CPS, t1 = (part + 1) * n_tiles / CPS;
    if (t0 == t1) return;
    const int lane = threadIdx.x & 31, warp = threadIdx.x >> 5;
    const int K4 = (K + 3) & ~3, WS = min(D, DW);
    const T* wg = W + seg_group[seg] * wgs;
    // shared: 2 stages of R tiles (TM x kc as RT) | CONV: the tile as T | W
    // chunk, element (k, j) at ((k / 4) * WS + j) * 4 + k % 4, + slack read by
    // lanes past WS (results unused)
    const size_t SB = (size_t)TM * kc * sizeof(RT);
    T* rw = reinterpret_cast<T*>(sm + 2 * SB);
    T* ws = rw + (CONV ? TM * kc : 0);

    for (int d0 = 0; d0 < D; d0 += DW)
        for (int k0 = 0; k0 < K4; k0 += kc) {
            const int kn = min(kc, K4 - k0), wn = min(WS, D - d0);
            const bool first = !CHUNKED || k0 == 0,
                       last = !CHUNKED || k0 + kn == K4;
            __syncthreads();  // previous chunk done with ws and the stages
            for (int i = threadIdx.x; i < kn * wn; i += NT) {
                int k = i / wn, d = i - k * wn;
                T* p = ws + ((k >> 2) * WS + d) * 4 + (k & 3);
                if (k0 + k < K)
                    __pipeline_memcpy_async(p, wg + (k0 + k) * wks + d0 + d,
                                            sizeof(T));
                else  // R pad +0: fma(+0, -0, acc) == acc, also for acc = -0
                    *p = T(-0.0);
            }
            const int dn = kn * sizeof(RT), n = min(kn, K - k0) * sizeof(RT);
            auto load = [&](int t) {
                const int r0 = start + t * TM;
                if (t < t1)
                    stage(v, sm + (t - t0) % 2 * SB, dn,
                          R + (size_t)r0 * ldr + k0, ldr * sizeof(RT),
                          min(TM, end - r0), n);
                __pipeline_commit();
            };
            load(t0);
            for (int t = t0; t < t1; ++t) {
                __pipeline_wait_prior(0);
                __syncthreads();
                load(t + 1);
                const int o = warp * RM * kn;
                const RT* st =
                    reinterpret_cast<const RT*>(sm + (t - t0) % 2 * SB) + o;
                if constexpr (CONV) {  // widen this warp's rows (exact)
                    auto b = reinterpret_cast<const uint2*>(st);
                    for (int i = lane; i < RM * kn / 4; i += 32)
                        reinterpret_cast<uint4*>(rw + o)[i] =
                            make_uint4(b[i].x << 16, b[i].x & ~0xffffu,
                                       b[i].y << 16, b[i].y & ~0xffffu);
                    __syncwarp();
                }
                auto r = reinterpret_cast<const V4<T>*>(CONV ? rw + o
                                                             : (const T*)st);
                const int r0 = start + t * TM + warp * RM,
                          nr = min(RM, end - r0);
                auto w = reinterpret_cast<const V4<T>*>(ws) + lane;
                T acc[RM][SLOTS], x[RM][SLOTS];
#pragma unroll
                for (int j = 0; j < RM; ++j)
#pragma unroll
                    for (int c = 0; c < SLOTS; ++c) {
                        int d = d0 + lane + 32 * c;
                        size_t i = (size_t)(r0 + j) * D + d;
                        bool ok = j < nr && d < D;
                        acc[j][c] = ok && !first ? Z[i] : T(0);
                        x[j][c] = ok && last ? X[i] : T(0);
                    }
            // acc[j][c] += R[j][k] W[k][lane + 32 c], k ascending
#pragma unroll 2
                for (int g = 0; g < kn / 4; ++g) {
                    V4<T> rv[RM], wv[SLOTS];
#pragma unroll
                    for (int j = 0; j < RM; ++j) rv[j] = r[j * kn / 4 + g];
#pragma unroll
                    for (int c = 0; c < SLOTS; ++c) wv[c] = w[g * WS + 32 * c];
#pragma unroll
                    for (int q = 0; q < 4; ++q)
#pragma unroll
                        for (int j = 0; j < RM; ++j)
#pragma unroll
                            for (int c = 0; c < SLOTS; ++c)
                                acc[j][c] =
                                    fma(rv[j].x[q], wv[c].x[q], acc[j][c]);
                }
#pragma unroll
                for (int j = 0; j < RM; ++j) {
                    if (j >= nr) break;
                    T sq = T(0), scale = T(1);
#pragma unroll
                    for (int c = 0; c < SLOTS; ++c)
                        if (last && d0 + lane + 32 * c < D) {
                            acc[j][c] = x[j][c] - acc[j][c];
                            sq = fma(acc[j][c], acc[j][c], sq);
                        }
                    if (last && normalize && D <= DW) scale = inv_norm(sq);
                    T* z = Z + (size_t)(r0 + j) * D + d0 + lane;
#pragma unroll
                    for (int c = 0; c < SLOTS; ++c)
                        if (d0 + lane + 32 * c < D)
                            z[32 * c] = acc[j][c] * scale;
                }
            }
        }

    if (D > DW && normalize)  // rows this lane wrote, add_rows_normalize order
        for (int t = t0; t < t1; ++t)
            for (int j = 0, r = start + t * TM + warp * RM;
                 j < RM && r + j < end; ++j) {
                T* z = Z + (size_t)(r + j) * D;
                T sq = T(0);
                for (int d = lane; d < D; d += 32) sq = fma(z[d], z[d], sq);
                const T scale = inv_norm(sq);
                for (int d = lane; d < D; d += 32) z[d] *= scale;
            }
}

template <typename T, typename RT, bool CHUNKED>
auto kernel_for(int slots) {
    return slots == 1   ? apply_rows_kernel<T, RT, 1, CHUNKED>
           : slots == 2 ? apply_rows_kernel<T, RT, 2, CHUNKED>
           : slots == 4 ? apply_rows_kernel<T, RT, 4, CHUNKED>
                        : apply_rows_kernel<T, RT, 8, CHUNKED>;
}

// apply_rows with slots in {1, 2, 4, 8} (columns per lane / 32), W in chunks
// of at most kc clusters (rounded up to 4) and at most `budget` bytes of
// shared memory; results depend on none of them.
template <typename T, typename RT>
cudaError_t launch(const T* X, const RT* R, int ldr, const T* W, long long wks,
                   long long wgs, int D, int K, const int* seg_start,
                   const int* seg_group, int n_seg, bool normalize, T* Z,
                   int slots, int kc, size_t budget, cudaStream_t s) {
    // shared T per cluster: 2 R stages (or 2 bfloat16 + widened), W column
    const int k4 = (K + 3) & ~3, dw = 32 * slots, ws = std::min(D, dw);
    const int per = 16 * (slots < 8 ? 4 : 2) + ws, slack = (dw - ws) * 4;
    kc = std::min(
        {(kc + 3) & ~3, k4, ((int)(budget / sizeof(T)) - slack) / per & ~3});
    if (kc <= 0) return cudaErrorInvalidValue;
    if (n_seg <= 0) return cudaSuccess;
    const size_t smem = ((size_t)per * kc + slack) * sizeof(T);
    auto kernel = kc < k4 ? kernel_for<T, RT, true>(slots)
                          : kernel_for<T, RT, false>(slots);
    if (smem > 48 * 1024 &&
        cudaFuncSetAttribute(
            kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, (int)smem))
        return cudaGetLastError();
    // widest copy dividing every R row start, length and stage row
    size_t a =
        reinterpret_cast<size_t>(R) | 16 | ((size_t)ldr | kc | K) * sizeof(RT);
    kernel<<<n_seg * CPS, NT, smem, s>>>(X, R, ldr, W, wks, wgs, D, K,
                                         seg_start, seg_group, normalize, Z, kc,
                                         (int)(a & (~a + 1)));
    return cudaGetLastError();
}

}  // namespace detail

// Z (rows x D) for segments [seg_start[s], seg_start[s + 1]) (local rows) of
// groups seg_group[s]; see the top of this file. R(i, k) = R[i * ldr + k]
// (T, or bfloat16 with T = float; ldr >= K); W[g] is K x D with element
// strides w_k_stride and w_group_stride. Z must not alias X, R or W.
template <typename T, typename RT>
cudaError_t apply_rows(const T* X, const RT* R, int ldr, const T* W,
                       long long w_k_stride, long long w_group_stride, int D,
                       int K, const int* seg_start, const int* seg_group,
                       int n_seg, bool normalize, T* Z, cudaStream_t s) {
    if (D <= 0 || K <= 0 || ldr < K) return cudaErrorInvalidValue;
    if (reinterpret_cast<size_t>(R) % sizeof(RT))
        return cudaErrorMisalignedAddress;
    int dev, optin;
    if (cudaGetDevice(&dev) ||
        cudaDeviceGetAttribute(&optin, cudaDevAttrMaxSharedMemoryPerBlockOptin,
                               dev))
        return cudaGetLastError();
    // 96 KB: opt-in limit of sm_86, sm_89, sm_120 (sm_75: 64 KB). Speed only:
    // W resident if it fits, else in the largest chunks.
    const int slots = D <= 32 ? 1 : D <= 64 ? 2 : D <= 128 ? 4 : 8;
    return detail::launch(X, R, ldr, W, w_k_stride, w_group_stride, D, K,
                          seg_start, seg_group, n_seg, normalize, Z, slots, K,
                          std::min<size_t>(96 * 1024, optin), s);
}

}  // namespace harmony_apply
