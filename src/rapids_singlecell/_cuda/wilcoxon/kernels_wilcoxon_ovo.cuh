#pragma once

#include <cuda_runtime.h>

#include "wilcoxon_block_reduce.cuh"
#include "wilcoxon_fast_common.cuh"

// Bitonic sort of `n` floats in shared memory, ascending. `n` MUST be a power
// of two; pad the tail with +INF before calling. Grid-stride: any blockDim
// works.

__device__ __forceinline__ void bitonic_sort_smem(float* s, int n) {
    for (int k = 2; k <= n; k <<= 1) {
        for (int j = k >> 1; j > 0; j >>= 1) {
            for (int i = threadIdx.x; i < n; i += blockDim.x) {
                int ixj = i ^ j;
                if (ixj > i) {
                    bool asc = ((i & k) == 0);
                    float a = s[i], b = s[ixj];
                    if (asc ? (a > b) : (a < b)) {
                        s[i] = b;
                        s[ixj] = a;
                    }
                }
            }
            __syncthreads();
        }
    }
}

// Sorted-array bounds over [lo, hi). lower: first idx with arr[idx] >= v (count
// of elements < v). upper: first idx with arr[idx] > v (count <= v). Advanced
// `lo` exploits per-thread-stride monotonicity; works on global or shared arr.

__device__ __forceinline__ int sorted_lower_bound(const float* arr, int lo,
                                                  int hi, float v) {
    while (lo < hi) {
        int m = lo + ((hi - lo) >> 1);
        if (arr[m] < v)
            lo = m + 1;
        else
            hi = m;
    }
    return lo;
}

__device__ __forceinline__ int sorted_upper_bound(const float* arr, int lo,
                                                  int hi, float v) {
    while (lo < hi) {
        int m = lo + ((hi - lo) >> 1);
        if (arr[m] <= v)
            lo = m + 1;
        else
            hi = m;
    }
    return lo;
}

// Mid-rank of `v` in the merged (ref, grp) arrays. Advances the four
// incremental bounds (pass 0,0,0,0 for a fresh search); reports per-array equal
// counts for tie correction.
struct OvoRank {
    double mid_rank;
    int n_eq_ref;
    int n_eq_grp;
};

__device__ __forceinline__ OvoRank ovo_mid_rank(const float* ref, int n_ref,
                                                const float* grp, int n_grp,
                                                float v, int& ref_lb,
                                                int& ref_ub, int& grp_lb,
                                                int& grp_ub) {
    int n_lt_ref = sorted_lower_bound(ref, ref_lb, n_ref, v);
    ref_lb = n_lt_ref;
    ref_ub = sorted_upper_bound(ref, ref_ub > n_lt_ref ? ref_ub : n_lt_ref,
                                n_ref, v);
    int n_eq_ref = ref_ub - n_lt_ref;

    int n_lt_grp = sorted_lower_bound(grp, grp_lb, n_grp, v);
    grp_lb = n_lt_grp;
    grp_ub = sorted_upper_bound(grp, grp_ub > n_lt_grp ? grp_ub : n_lt_grp,
                                n_grp, v);
    int n_eq_grp = grp_ub - n_lt_grp;

    OvoRank r;
    r.mid_rank = (double)(n_lt_ref + n_lt_grp) +
                 ((double)(n_eq_ref + n_eq_grp) + 1.0) / 2.0;
    r.n_eq_ref = n_eq_ref;
    r.n_eq_grp = n_eq_grp;
    return r;
}

// Amortized tie correction for LARGE/HUGE bands (group is SORTED). Adds only
// the group-only / ref-overlap delta on the precomputed ref base
// ref_tie_sums[col], like MEDIUM. Iterates the group's UNIQUE values only (one
// ref binary search each) so the ref is NOT rescanned per group: O(n_grp_unique
// * log n_ref) vs O(n_ref)/group. Bit-identical: same per-value (t^3 - t)
// terms, reassociated against the shared ref base.
__device__ __forceinline__ void compute_tie_delta_sorted_grp(
    const float* ref_col, int n_ref, const float* grp_col, int n_grp,
    double ref_base, double* warp_buf, double* out) {
    double local = 0.0;
    for (int i = threadIdx.x; i < n_grp; i += blockDim.x) {
        // run-start of a unique value in the sorted group
        if (i == 0 || grp_col[i] != grp_col[i - 1]) {
            float v = grp_col[i];
            int gub = sorted_upper_bound(grp_col, i + 1, n_grp, v);
            double cg = (double)(gub - i);
            int rlo = sorted_lower_bound(ref_col, 0, n_ref, v);
            int rub = sorted_upper_bound(ref_col, rlo, n_ref, v);
            double cr = (double)(rub - rlo);
            double group_tie = (cg > 1.0) ? (cg * cg * cg - cg) : 0.0;
            local += group_tie;
            if (cr > 0.0) {
                double combined = cr + cg;
                double ref_tie = (cr > 1.0) ? (cr * cr * cr - cr) : 0.0;
                local += combined * combined * combined - combined - ref_tie -
                         group_tie;
            }
        }
    }
    double tie = wilcoxon_block_sum(local, warp_buf);
    if (threadIdx.x == 0)
        *out = finalize_tie_corr(n_ref + n_grp, ref_base + tie);
}

// No-tie fast path (tie_correct=False, default). Ranks each group value against
// the sorted REFERENCE only, via the Mann-Whitney U identity:
//   R_g = n_grp(n_grp+1)/2 + Σ_{g values}(#ref_below + 0.5·#ref_equal)
// Group-internal ranks collapse to the closed form, so the group needs NO sort
// (each value binary-searches the sorted ref) -- skips the group segmented
// sort, ~half of dense-OVO time. rank_sums are exact half-integers => matches
// the tiered path bit-for-bit. Grid (n_cols, n_groups). grp_dense is UNSORTED.
__global__ void ovo_rank_dense_vs_ref_kernel(
    const float* __restrict__ ref_sorted, const float* __restrict__ grp_dense,
    const int* __restrict__ grp_offsets, double* __restrict__ rank_sums,
    int n_ref, int n_all_grp, int n_cols, int n_groups) {
    int col = blockIdx.x;
    int grp = blockIdx.y;
    if (col >= n_cols || grp >= n_groups) return;

    int g_start = grp_offsets[grp];
    int n_grp = grp_offsets[grp + 1] - g_start;
    if (n_grp == 0) {
        if (threadIdx.x == 0) rank_sums[(size_t)grp * n_cols + col] = 0.0;
        return;
    }
    const float* ref_col = ref_sorted + (long long)col * n_ref;
    const float* grp_col = grp_dense + (long long)col * n_all_grp + g_start;

    double local_sum = 0.0;
    for (int i = threadIdx.x; i < n_grp; i += blockDim.x) {
        float v = grp_col[i];
        int n_lt = sorted_lower_bound(ref_col, 0, n_ref, v);
        int n_eq = sorted_upper_bound(ref_col, n_lt, n_ref, v) - n_lt;
        local_sum += (double)n_lt + 0.5 * (double)n_eq;
    }
    __shared__ double warp_buf[32];
    double total = wilcoxon_block_sum(local_sum, warp_buf);
    if (threadIdx.x == 0) {
        rank_sums[(size_t)grp * n_cols + col] =
            total + (double)n_grp * ((double)n_grp + 1.0) / 2.0;
    }
}

// LARGE/HUGE pre-sorted rank kernel. Grid (n_cols, n_groups); each thread
// carries lower/upper bounds across its stride (sorted-grp_col monotonicity).
// SMEM_SORT=true (LARGE, groups <= OVO_LARGE_MAX): load unsorted group into
// dynamic smem (large_padded floats) + bitonic-sort. =false (HUGE): read a
// CUB-segmented-sorted group from global. Post-sort body (incremental mid-ranks
// + amortized ref-tie delta) is shared. Each group owns its rank_sums/tie_corr
// row, so size-gated co-launch (skip_n_grp_le) never aliases.
template <bool SMEM_SORT>
__global__ void ovo_rank_sorted_kernel(
    const float* __restrict__ ref_sorted, const float* __restrict__ grp_in,
    const int* __restrict__ grp_offsets,
    const double* __restrict__ ref_tie_sums, double* __restrict__ rank_sums,
    double* __restrict__ tie_corr, int n_ref, int n_all_grp, int n_cols,
    int n_groups, bool compute_tie_corr, int large_padded, int skip_n_grp_le) {
    int col = blockIdx.x;
    int grp = blockIdx.y;
    if (col >= n_cols || grp >= n_groups) return;

    int g_start = grp_offsets[grp];
    int n_grp = grp_offsets[grp + 1] - g_start;
    if (n_grp <= skip_n_grp_le) return;
    if (n_grp == 0) {
        if (threadIdx.x == 0) {
            rank_sums[grp * n_cols + col] = 0.0;
            if (compute_tie_corr) tie_corr[grp * n_cols + col] = 1.0;
        }
        return;
    }

    const float* ref_col = ref_sorted + (long long)col * n_ref;
    __shared__ double warp_buf[32];
    const float* grp_col;
    if constexpr (SMEM_SORT) {
        extern __shared__ float grp_smem[];
        const float* src = grp_in + (long long)col * n_all_grp + g_start;
        for (int i = threadIdx.x; i < n_grp; i += blockDim.x)
            grp_smem[i] = src[i];
        for (int i = n_grp + threadIdx.x; i < large_padded; i += blockDim.x)
            grp_smem[i] = __int_as_float(0x7f800000);  // +INF pad
        __syncthreads();
        bitonic_sort_smem(grp_smem, large_padded);
        grp_col = grp_smem;
    } else {
        (void)large_padded;
        grp_col =
            grp_in + (long long)col * n_all_grp + g_start;  // CUB-presorted
    }

    int ref_lb = 0, ref_ub = 0, grp_lb = 0, grp_ub = 0;
    double local_sum = 0.0;
    for (int i = threadIdx.x; i < n_grp; i += blockDim.x) {
        OvoRank r = ovo_mid_rank(ref_col, n_ref, grp_col, n_grp, grp_col[i],
                                 ref_lb, ref_ub, grp_lb, grp_ub);
        local_sum += r.mid_rank;
    }
    double total = wilcoxon_block_sum(local_sum, warp_buf);
    if (threadIdx.x == 0) rank_sums[grp * n_cols + col] = total;

    if (!compute_tie_corr) return;
    __syncthreads();
    // grp_col is sorted: amortize the ref tie contribution via the precomputed
    // base instead of rescanning the ref per group.
    compute_tie_delta_sorted_grp(ref_col, n_ref, grp_col, n_grp,
                                 ref_tie_sums[col], warp_buf,
                                 &tie_corr[grp * n_cols + col]);
}

// MEDIUM-band helper: tie contribution of the sorted reference alone (one block
// per column). The rank kernels use this base and add only group-only/overlap
// deltas from the group values.
__global__ void ref_tie_sum_kernel(const float* __restrict__ ref_sorted,
                                   double* __restrict__ ref_tie_sums, int n_ref,
                                   int n_cols) {
    int col = blockIdx.x;
    if (col >= n_cols) return;
    const float* ref_col = ref_sorted + (long long)col * n_ref;

    double local_tie = 0.0;
    for (int i = threadIdx.x; i < n_ref; i += blockDim.x) {
        if (i == 0 || ref_col[i] != ref_col[i - 1]) {
            float v = ref_col[i];
            int cnt = sorted_upper_bound(ref_col, i + 1, n_ref, v) - i;
            if (cnt > 1) {
                double t = (double)cnt;
                local_tie += t * t * t - t;
            }
        }
    }

    __shared__ double warp_buf[32];
    double total = wilcoxon_block_sum(local_tie, warp_buf);
    if (threadIdx.x == 0) ref_tie_sums[col] = total;
}

// MEDIUM-band fused kernel: no-sort direct rank for groups in (skip_n_grp_le,
// max_n_grp_le]. Ranks = ref binary searches + an in-group scan over unsorted
// shared values. Tie correction starts from ref_tie_sums[col] and adds only
// group-only / ref-overlap deltas.
__global__ void ovo_rank_medium_kernel(
    const float* __restrict__ ref_sorted, const float* __restrict__ grp_dense,
    const int* __restrict__ grp_offsets,
    const double* __restrict__ ref_tie_sums, double* __restrict__ rank_sums,
    double* __restrict__ tie_corr, int n_ref, int n_all_grp, int n_cols,
    int n_groups, bool compute_tie_corr, int skip_n_grp_le, int max_n_grp_le) {
    int col = blockIdx.x;
    int grp = blockIdx.y;
    if (col >= n_cols || grp >= n_groups) return;

    int g_start = grp_offsets[grp];
    int g_end = grp_offsets[grp + 1];
    int n_grp = g_end - g_start;
    if (n_grp <= skip_n_grp_le || n_grp > max_n_grp_le) return;

    extern __shared__ char smem_raw[];
    float* grp_smem = (float*)smem_raw;
    double* warp_buf = (double*)(smem_raw + max_n_grp_le * sizeof(float));

    const float* grp_col = grp_dense + (long long)col * n_all_grp + g_start;
    for (int i = threadIdx.x; i < n_grp; i += blockDim.x)
        grp_smem[i] = grp_col[i];
    __syncthreads();

    const float* ref_col = ref_sorted + (long long)col * n_ref;
    double local_sum = 0.0;
    double local_tie_delta = 0.0;

    for (int i = threadIdx.x; i < n_grp; i += blockDim.x) {
        float v = grp_smem[i];

        int n_lt_ref = sorted_lower_bound(ref_col, 0, n_ref, v);
        int n_eq_ref =
            sorted_upper_bound(ref_col, n_lt_ref, n_ref, v) - n_lt_ref;

        int n_lt_grp = 0;
        int n_eq_grp = 0;
        bool first_in_grp = true;
        for (int j = 0; j < n_grp; ++j) {
            float w = grp_smem[j];
            if (w < v) ++n_lt_grp;
            if (w == v) {
                ++n_eq_grp;
                if (j < i) first_in_grp = false;
            }
        }

        local_sum += (double)(n_lt_ref + n_lt_grp) +
                     ((double)(n_eq_ref + n_eq_grp) + 1.0) / 2.0;

        if (compute_tie_corr && first_in_grp) {
            double cg = (double)n_eq_grp;
            double cr = (double)n_eq_ref;
            double group_tie = (cg > 1.0) ? (cg * cg * cg - cg) : 0.0;
            local_tie_delta += group_tie;
            if (cr > 0.0) {
                double combined = cr + cg;
                double ref_tie = (cr > 1.0) ? (cr * cr * cr - cr) : 0.0;
                local_tie_delta += combined * combined * combined - combined -
                                   ref_tie - group_tie;
            }
        }
    }

    double total = wilcoxon_block_sum(local_sum, warp_buf);
    if (threadIdx.x == 0) rank_sums[grp * n_cols + col] = total;

    if (!compute_tie_corr) return;
    __syncthreads();

    double tie_delta = wilcoxon_block_sum(local_tie_delta, warp_buf);
    if (threadIdx.x == 0)
        tie_corr[grp * n_cols + col] =
            finalize_tie_corr(n_ref + n_grp, ref_tie_sums[col] + tie_delta);
}

// WARP (≤32) and SMALL (33–64) tiers were removed; MEDIUM is now the smallest
// tier, covering all groups ≤ OVO_MEDIUM_MAX. Removed kernels archived with
// restore steps in .claude/wilcoxon-warp-small-tiers-removed.md.
