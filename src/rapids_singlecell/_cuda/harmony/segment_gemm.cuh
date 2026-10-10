// Deterministic, shard-invariant segmented product for Harmony (centroids
// Z_norm^T R, correction RHS R_b^T X_b): acc[g] += fixed_point(sum over the
// cells i of group g of A_i^T R_i), A (n x D) row-major, R (n x K) of row
// stride ldr. Cells are sorted by group; segments are runs of one group cut at
// multiples of SEG cells and of the shard unit (see _segments), so shards,
// which split at unit multiples, pass their own segment tables. One CTA (per
// output tile) reduces a segment in a fixed order in T: rows go in chunks of
// ch, each in nsplit slices of cps rows; slice t accumulates its rows of every
// chunk in row order, then the slice sums are added in order t = 0, 1, ...;
// (ch, cps) depend only on (D, K, sizeof(T), sizeof(RT)). Segment partials
// are rounded once to fixed point and added with exact integer atomics: acc is
// bitwise independent of launch order, CTA count, GPU, ldr and of the shard
// split (int64 sum of the shards' acc).
#pragma once
#include <cuda_pipeline.h>
#include <cuda_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <type_traits>

namespace harmony_segments {

constexpr int kThreads = 256, kSmemBudget = 48 * 1024, kLimbBits = 30;

// group_scale[g] = 2^(61 - e + 30 (limbs - 1)), bound <= 2^e; `bound` (the
// same on every shard) >= sum over the group of |A_id| |R_ik| for all (k, d):
// N_total for the centroids (|Z| <= 1, R in [0, 1]), n_b max|X| for the RHS of
// batch b. The top limb stays below 2^62; one unit of the last limb is 2^-61,
// 2^-91, 2^-121 of the bound for 1, 2, 3 limbs.
inline double scale_for_bound(double bound, int limbs) {
    int e = 0;
    std::frexp(bound > 0 && std::isfinite(bound) ? bound : 1.0, &e);
    e = std::max(-800, std::min(1000, e));
    return std::ldexp(1.0, 61 - e + kLimbBits * (limbs - 1));
}

struct Plan {
    int td, tk, ch, nsplit, cps;       // TD x TK outputs per thread, order
    int dgt, kgt, n_dy, n_ky;          // dgt x kgt threads per tile, tiles
    int sa, sr, stage_a, stage, smem;  // smem layout, set per launch
};

// Pure function of (D, K, sizeof(T), sizeof(RT)). fp32: least issue cost of a
// row (FMAs + smem loads of the active warps), one tile if possible ({10, 8}
// only if nothing else fits); fp64 (FMA bound): fewest padded FMAs.
inline Plan make_plan(int D, int K, int sT, int sR) {
    auto cdiv = [](long long a, long long b) { return int((a + b - 1) / b); };
    auto up16 = [](long long b) { return (b + 15) / 16 * 16; };
    Plan p{8, sT == 8 ? 4 : 8};
    double best = 1e300;
    if (sT == 8)
        for (int td : {8, 10, 5, 4})
            if (cdiv(D, td) * td < best) best = cdiv(D, td) * td, p.td = td;
    constexpr int configs[][2] = {{8, 8}, {16, 4}, {10, 4}, {10, 8}};
    for (auto [td, tk] : configs) {
        const long long n = (long long)cdiv(D, td) * cdiv(K, tk);
        if (sT == 8 || n > kThreads || (td * tk == 80 && best < 1e300))
            continue;
        const int ns = kThreads / n, vec = !(td % 2 || D % 2 || K % tk);
        const int loads = vec ? td / 2 + tk / 4 : td + 2 * tk;
        const double c = cdiv(n * ns, 32) * double(td * tk + loads + 6) / ns;
        if (c < best) best = c, p.td = td, p.tk = tk;
    }
    // output tiles of <= kThreads threads, about square when tiled
    const int DG = cdiv(D, p.td), KG = cdiv(K, p.tk);
    p.dgt = std::min(DG, std::max(16, kThreads / KG));
    p.n_dy = cdiv(DG, p.dgt);
    p.dgt = cdiv(DG, p.n_dy);
    p.n_ky = cdiv(KG, kThreads / p.dgt);
    p.kgt = cdiv(KG, p.n_ky);
    // ch = 64 / 2^i, the largest whose 2 stages fit (bytes per row: contiguous
    // rows, else a tile row + alignment slack)
    const long long ra = p.n_dy > 1 ? p.dgt * p.td * sT + 31 : D * sT,
                    rr = p.n_ky > 1 ? p.kgt * p.tk * sR + 31 : K * sR;
    for (p.ch = 64; p.ch > 1; p.ch /= 2)
        if (2 * (up16(p.ch * ra + 32 + p.td * sT) + up16(p.ch * rr + 32)) <=
            kSmemBudget)
            break;
    const int nto = p.dgt * p.kgt, red = nto * p.td * p.tk * sT;
    p.nsplit = std::min({kThreads / nto, p.ch, std::max(1, kSmemBudget / red)});
    if (p.n_dy * p.n_ky > 1) p.nsplit = 1;  // tiles: plain row order
    p.cps = cdiv(p.ch, p.nsplit);
    p.nsplit = cdiv(p.ch, p.cps);
    return p;
}

namespace detail {

// v[0 .. M) = p[0 .. M) as T, by aligned loads of N elements (<= 16 bytes)
template <int N, typename T, typename E, int M>
__device__ __forceinline__ void ldv(T (&v)[M], const E* p) {
    constexpr int W = M % N ? 1 : N * sizeof(E) > 16 ? 16 / sizeof(E) : N;
    struct alignas(W * sizeof(E)) U {
        E e[W];
    };
#pragma unroll
    for (int i = 0; i < M; i += W) {
        const U u = reinterpret_cast<const U*>(p)[i / W];
#pragma unroll
        for (int j = 0; j < W; ++j) v[i + j] = T(u.e[j]);
    }
}

// Stage rows [base, base + rows), columns [c0, c0 + w) of M (row stride ld,
// rows first .. last - 1 of W columns readable) as 16-byte cp.async of aligned
// blocks (element-wise at the readable ends): element (r, c) lands at byte
// (M + base ld + c0) % 16 + (r ss + c) sizeof(E) (ss == ld if contiguous, else
// ss >= w + 16 bytes and ss = ld mod 16 bytes).
template <typename E>
__device__ __forceinline__ void stage_copy(char* dst, const E* M, long long ld,
                                           int W, int c0, int w, int ss,
                                           int base, int rows, int first,
                                           int last) {
    auto at = [&](long long i, int j) {
        return reinterpret_cast<const char*>(M + i * ld + j);
    };
    const char *lo = at(first, 0), *hi = at(last - 1, W), *src = at(base, c0);
    dst += uintptr_t(src) % 16;
    if (ld == w) w *= rows, rows = 1;  // contiguous rows: one range
    const int len = w * sizeof(E),
              lg = rows > 1 ? min(8, 32 - __clz((len + 14) / 16)) : 8;
    // 2^lg threads per row
    for (int r = threadIdx.x >> lg; r < rows; r += kThreads >> lg) {
        const char *b = src + r * ld * sizeof(E), *s = b - uintptr_t(b) % 16;
        char* d = dst + r * ss * sizeof(E) - uintptr_t(b) % 16;
        for (int k = 16 * (threadIdx.x & ((1 << lg) - 1)); s + k < b + len;
             k += 16 << lg)
            if (s + k >= lo && s + k + 16 <= hi)
                __pipeline_memcpy_async(d + k, s + k, 16);
            else
                for (int o = k; o < k + 16; o += sizeof(E))
                    if (s + o >= b && s + o < b + len)
                        *reinterpret_cast<E*>(d + o) =
                            *reinterpret_cast<const E*>(s + o);
    }
}

// Add v 2^S, rounded to an integer W (half away from zero), as 30-bit limbs:
// sign-magnitude digits, top limb first (|W| < 2^62 2^(30 (limbs - 1)) by the
// bound). Integer only: |v| = w 2^t exactly. Internal linkage keeps -rdc
// builds free of call-ABI spills.
template <typename T>
static __device__ __noinline__ void emit(T v, int S, unsigned long long* a,
                                         size_t ls, int limbs) {
    constexpr int mb = sizeof(T) == 4 ? 23 : 52, eb = sizeof(T) == 4 ? 8 : 11;
    unsigned long long u = 0;
    memcpy(&u, &v, sizeof(T));
    const int be = int(u >> mb) & ((1 << eb) - 1);
    const int t = max(be, 1) - (1 << (eb - 1)) + 1 - mb + S;  // |v| 2^S = w 2^t
    unsigned long long w = (u & ((1ull << mb) - 1)) | (be ? 1ull << mb : 0);
    if (t < 0) w = -t > 63 ? 0 : (w + (1ull << (-t - 1))) >> -t;
    // digit l = (|W| >> 30 (limbs - 1 - l)) mod 2^30, |W| = w 2^max(t, 0)
    for (int l = 0; w && l < limbs; ++l) {
        const int sh = max(t, 0) - kLimbBits * (limbs - 1 - l);
        unsigned long long d =
            sh >= 0 ? (sh > 63 ? 0 : w << sh) : (sh < -63 ? 0 : w >> -sh);
        if (l) d &= (1ull << kLimbBits) - 1;
        if (d) atomicAdd(a + l * ls, u >> (mb + eb) ? 0ull - d : d);
    }
}

template <typename T, typename RT, int TD, int TK>
__global__ void __launch_bounds__(kThreads, 2)
    segment_kernel(const T* __restrict__ A, const RT* __restrict__ R, int D,
                   int K, int ldr, const int* __restrict__ seg_start,
                   const int* __restrict__ seg_group, int n_seg,
                   const double* __restrict__ group_scale,
                   unsigned long long* __restrict__ acc, int limbs, Plan p) {
    extern __shared__ __align__(16) char smem[];
    const int d0 = blockIdx.y / p.n_ky * p.dgt * TD;
    const int k0 = blockIdx.y % p.n_ky * p.kgt * TK;
    const int dv = min(p.dgt * TD, D - d0), kv = min(p.kgt * TK, K - k0);
    const int start = seg_start[blockIdx.x], end = seg_start[blockIdx.x + 1];
    const int first = seg_start[0], last = seg_start[n_seg];  // readable rows
    auto load = [&](int c) {  // chunk c into stage c % 2
        const int base = start + c * p.ch, rows = min(p.ch, end - base);
        char* st = smem + (c & 1) * p.stage;
        if (rows > 0) {
            stage_copy(st, A, D, D, d0, dv, p.sa, base, rows, first, last);
            stage_copy(st + p.stage_a, R, ldr, K, k0, kv, p.sr, base, rows,
                       first, last);
        }
        __pipeline_commit();
    };
    load(0);
    const int nto = p.dgt * p.kgt, sp = threadIdx.x / nto;
    const int o = threadIdx.x % nto, dg = o / p.kgt, kg = o % p.kgt;
    // R columns of thread kg: TK kg + q (aligned smem vector loads) or kg + kgt
    // q (conflict free); only the owner of an output changes, never its sum
    // order
    const bool vec = p.n_dy * p.n_ky == 1 && D % 2 == 0 && K % TK == 0 &&
                     ldr % 4 == 0 && uintptr_t(A) % (2 * sizeof(T)) == 0 &&
                     uintptr_t(R) % (4 * sizeof(RT)) == 0;
    auto col = [&](int k, int q) { return vec ? TK * k + q : k + p.kgt * q; };
    T s[TD][TK] = {};
    auto rows = [&](auto V, const T* a, const RT* r, int n) {
        int off[TK];  // clamped into the tile (scalar path)
#pragma unroll
        for (int q = 0; q < TK; ++q) off[q] = min(kg + p.kgt * q, kv - 1);
#pragma unroll 2
        for (int i = 0; i < n; ++i, a += p.sa, r += p.sr) {
            T av[TD], rv[TK];
            ldv<decltype(V)::value ? 2 : 1>(av, a);
            if constexpr (decltype(V)::value)
                ldv<4>(rv, r);
            else
#pragma unroll
                for (int q = 0; q < TK; ++q) rv[q] = T(r[off[q]]);
#pragma unroll
            for (int j = 0; j < TD; ++j)
#pragma unroll
                for (int q = 0; q < TK; ++q)
                    s[j][q] = fma(av[j], rv[q], s[j][q]);
        }
    };
    for (int c = 0; c * p.ch < end - start; ++c) {
        __pipeline_wait_prior(0);
        __syncthreads();
        load(c + 1);
        const int base = start + c * p.ch, i0 = sp * p.cps;
        const int n = min(p.cps, min(p.ch, end - base) - i0);
        if (sp >= p.nsplit || n <= 0) continue;
        const char* st = smem + (c & 1) * p.stage;
        const T* a = reinterpret_cast<const T*>(
            st + uintptr_t(A + (size_t)base * D + d0) % 16);
        const RT* r = reinterpret_cast<const RT*>(
            st + p.stage_a + uintptr_t(R + (size_t)base * ldr + k0) % 16);
        a += i0 * p.sa + TD * dg;  // A columns TD dg + j (broadcast)
        r += i0 * p.sr;
        if (vec)
            rows(std::true_type{}, a, r + TK * kg, n);
        else
            rows(std::false_type{}, a, r, n);
    }
    __pipeline_wait_prior(0);
    __syncthreads();

    const int g = seg_group ? seg_group[blockIdx.x] : 0,
              S = ilogb(group_scale[g]);
    const size_t ls = (size_t)K * D;
    unsigned long long* out = acc + g * limbs * ls + (size_t)k0 * D + d0;
    auto put = [&](T v, int d, int k) {
        if (d < dv && k < kv) emit(v, S, out + (size_t)k * D + d, ls, limbs);
    };
    if (p.nsplit == 1 && sp == 0)
#pragma unroll
        for (int q = 0; q < TK; ++q)
#pragma unroll
            for (int j = 0; j < TD; ++j) put(s[j][q], TD * dg + j, col(kg, q));
    if (p.nsplit == 1) return;
    T* red = reinterpret_cast<T*>(smem);  // [split][j TK + q][o]
    const int per = nto * TD * TK;
    if (sp < p.nsplit)
#pragma unroll
        for (int e = 0; e < TD * TK; ++e)
            red[sp * per + e * nto + o] = s[e / TK][e % TK];
    __syncthreads();
    // split partials added in order t = 0, 1, ...
    for (int i = threadIdx.x; i < per; i += kThreads) {
        T v = red[i];
        for (int t = 1; t < p.nsplit; ++t) v += red[t * per + i];
        const int e = i / nto, oo = i % nto;
        put(v, TD * (oo / p.kgt) + e / TK, col(oo % p.kgt, e % TK));
    }
}

template <typename T>
__global__ void finalize_kernel(const long long* __restrict__ acc, int n_groups,
                                int D, int K, int limbs,
                                const double* __restrict__ group_scale,
                                T* __restrict__ out, long long out_group_stride,
                                int ld_out) {
    const long long kd = (long long)K * D;
    for (long long i = blockIdx.x * (long long)blockDim.x + threadIdx.x;
         i < n_groups * kd; i += (long long)gridDim.x * blockDim.x) {
        const int g = int(i / kd);
        const long long e = i - g * kd;
        double v = 0.0;
        for (int l = 0; l < limbs; ++l)
            v = ldexp(v, kLimbBits) + (double)acc[(g * limbs + l) * kd + e];
        out[g * out_group_stride + e / D * ld_out + e % D] =
            T(ldexp(v, -ilogb(group_scale[g])));
    }
}

}  // namespace detail

// acc += fixed point of sum_{i in group g} A_i^T R_i; acc: n_groups x limbs x
// K x D int64 ([g][l][k][d]), zeroed by the caller, summed as int64 across
// shards; value = sum_l acc[g][l][k][d] 2^(30 (limbs - 1 - l)) /
// group_scale[g]. A, R: rows seg_start[0] .. seg_start[n_seg] of this shard;
// R(i, k) = R[i ldr + k] (ldr >= K, the bytes between rows readable). seg_start
// (n_seg + 1, local rows), seg_group (n_seg; null: group 0), group_scale
// (n_groups): device pointers. Sums in T (R converted to T), finite inputs,
// limbs in [1, 3]. Stream-ordered, no allocation, no sync.
template <typename T, typename RT>
cudaError_t segment_rtz(const T* A, const RT* R, int D, int K, int ldr,
                        const int* seg_start, const int* seg_group, int n_seg,
                        const double* group_scale, long long* acc, int limbs,
                        cudaStream_t s) {
    if (D < 1 || K < 1 || ldr < K || limbs < 1 || limbs > 3)
        return cudaErrorInvalidValue;
    if (n_seg <= 0) return cudaSuccess;
    Plan p = make_plan(D, K, sizeof(T), sizeof(RT));
    // smem row strides (elements): the global one for contiguous rows, else the
    // tile width + 16 bytes, congruent to the global stride mod 16 bytes
    auto up16 = [](long long b) { return int((b + 15) / 16 * 16); };
    auto row = [](long long ld, long long w, int b) {
        return int(ld == w ? w : w + (16 + (ld - w) * b % 16) / b);
    };
    p.sa = row(D, p.n_dy > 1 ? p.dgt * p.td : D, sizeof(T));
    p.sr = row(ldr, p.n_ky > 1 ? p.kgt * p.tk : K, sizeof(RT));
    p.stage_a = up16(p.ch * p.sa * sizeof(T) + 32 + p.td * sizeof(T));
    p.stage = p.stage_a + up16(p.ch * p.sr * sizeof(RT) + 32);
    const int red = p.nsplit * p.dgt * p.kgt * p.td * p.tk * sizeof(T);
    p.smem = std::max(2 * p.stage, p.nsplit > 1 ? red : 0);
    auto launch = [&](auto kernel) {
        if (p.smem > kSmemBudget)  // strided R rows only
            cudaFuncSetAttribute(
                kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, p.smem);
        kernel<<<dim3(n_seg, p.n_dy * p.n_ky), kThreads, p.smem, s>>>(
            A, R, D, K, ldr, seg_start, seg_group, n_seg, group_scale,
            reinterpret_cast<unsigned long long*>(acc), limbs, p);
        return cudaGetLastError();
    };
#define HS_CASE(TD, TK)           \
    if (p.td == TD && p.tk == TK) \
        return launch(detail::segment_kernel<T, RT, TD, TK>);
    if constexpr (sizeof(T) == 8) {
        HS_CASE(8, 4) HS_CASE(10, 4) HS_CASE(5, 4) HS_CASE(4, 4)
    } else {
        HS_CASE(8, 8) HS_CASE(16, 4) HS_CASE(10, 4) HS_CASE(10, 8)
    }
#undef HS_CASE
    return cudaErrorInvalidConfiguration;
}

// out[g out_group_stride + k ld_out + d] = acc value of (g, k, d) as T. For the
// correction RHS into Phi_t_diag_R_X_all (K, n_batches + 1, D) at row b + 1:
// out = base + D, out_group_stride = D, ld_out = (n_batches + 1) D.
template <typename T>
cudaError_t finalize_rtz(const long long* acc, int n_groups, int D, int K,
                         int limbs, const double* group_scale, T* out,
                         long long out_group_stride, int ld_out,
                         cudaStream_t s) {
    const long long total = (long long)n_groups * K * D;
    if (total <= 0) return cudaSuccess;
    const int blocks = (int)std::min<long long>((total + 255) / 256, 4096);
    detail::finalize_kernel<T><<<blocks, 256, 0, s>>>(
        acc, n_groups, D, K, limbs, group_scale, out, out_group_stride, ld_out);
    return cudaGetLastError();
}

}  // namespace harmony_segments
