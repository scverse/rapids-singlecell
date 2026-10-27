#pragma once

#include <algorithm>
#include <cmath>

#include <cuda_runtime.h>

// Exact fixed-point sums, shared by the clustering and the correction.
// Sums over cells go through int64 fixed point: value * 2^s rounded to an
// integer, with s from a global bound so the sum cannot overflow. Integer
// addition is exact, so a sum does not depend on tiling, launch shape or how
// cells are split across GPUs.
template <typename T>
__device__ __forceinline__ long long to_fixed(T value, T scale) {
    if constexpr (sizeof(T) == 4)
        return __float2ll_rn(value * scale);
    else
        return __double2ll_rn(value * scale);
}

// Assignment value in [0, 1] to fixed point; float32 counts use at most 31
// fractional bits, so the conversion is 32-bit.
template <typename T>
__device__ __forceinline__ long long count_to_fixed(T value, T scale) {
    if constexpr (sizeof(T) == 4)
        return __float2uint_rn(value * scale);
    else
        return __double2ll_rn(value * scale);
}

// Largest s with bound * 2^s < 2^62.
inline int fixed_shift(double bound) {
    return 62 - (int)std::ceil(std::log2(std::max(bound, 1.0)));
}

// Fixed-point scales (see to_fixed) from global bounds: R values <= 1, so a
// count is at most n_total; a cell's objective term is at most
// 4 + sigma log K in magnitude.
template <typename T>
struct FixedScales {
    T count, objective;
    double count_inv, objective_inv;
    FixedScales(long long n_total, double sigma, int n_clusters) {
        int c = std::min(fixed_shift((double)n_total + 1),
                         sizeof(T) == 4 ? 31 : 62);
        int o = fixed_shift(
            (double)n_total *
            (5.0 + std::abs(sigma) * std::log(std::max(n_clusters, 2))));
        count = (T)std::ldexp(1.0, c), count_inv = std::ldexp(1.0, -c);
        objective = (T)std::ldexp(1.0, o), objective_inv = std::ldexp(1.0, -o);
    }
};

// Cluster counts per group, kept exactly: `total` (fixed point) is O over all
// cells. Optionally folds a block's new counts in (total += new - stored,
// stored = new; without stored total += new), then writes total minus the
// held-out block's counts `hold` to `out` in T and starts `clear`, the next
// assignment's counts, from next - cur - after (recounted counts, see
// fused_clustering_loop_body; null: zero), zeroing cur.
template <typename T>
__global__ void update_counts_kernel(long long* total, long long* fold_stored,
                                     const long long* fold_new,
                                     const long long* hold, long long* clear,
                                     double inv_scale, T* __restrict__ out,
                                     int n, long long* cur = nullptr,
                                     const long long* next = nullptr,
                                     const long long* after = nullptr) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    long long t = total[i];
    if (fold_new) {
        long long v = fold_new[i];
        t += fold_stored ? v - fold_stored[i] : v;
        if (fold_stored) fold_stored[i] = v;
        total[i] = t;
    }
    if (hold) t -= hold[i];
    out[i] = (T)((double)t * inv_scale);
    if (clear) {
        long long c = (next ? next[i] : 0) - (after ? after[i] : 0);
        if (cur) c -= cur[i], cur[i] = 0;
        clear[i] = c;
    }
}

// counts[seg_group[s] K + k] += the fixed-point counts of column k of R (row
// stride ldr) over segment s: the clustering's exact counts, per segment
// group, whatever the segments or GPUs.
template <typename T>
__global__ void segment_counts_kernel(const T* R, int ldr, int K,
                                      const int* seg_start,
                                      const int* seg_group, int n_seg, T scale,
                                      unsigned long long* counts) {
    for (long long p = blockIdx.x * (long long)blockDim.x + threadIdx.x;
         p < (long long)n_seg * K; p += (long long)gridDim.x * blockDim.x) {
        int s = (int)(p / K), k = (int)(p % K);
        long long sum = 0;
        for (int i = seg_start[s]; i < seg_start[s + 1]; ++i)
            sum += count_to_fixed(R[(size_t)i * ldr + k], scale);
        if (sum)
            atomicAdd(counts + (size_t)seg_group[s] * K + k,
                      (unsigned long long)sum);
    }
}
