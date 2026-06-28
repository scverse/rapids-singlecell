#pragma once

#include <cuda_runtime.h>

#include "wilcoxon_block_reduce.cuh"
#include "wilcoxon_fast_common.cuh"

// Bitonic sort of power-of-two `n` floats in shared memory, ascending.
// Pad the tail with +INF before calling; any blockDim works.

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

// Sorted-array bounds over [lo, hi): lower is first >= v, upper first > v.
// Advanced `lo` exploits monotonic strides; global/shared arrays both work.

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

// Mid-rank of `v` in merged (ref, grp) arrays with incremental bounds.
// Also reports equal counts per array for tie correction.
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

// Amortized tie correction for sorted LARGE/HUGE groups.
// Only unique group values update the precomputed ref tie base.
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

// No-tie fast path: group-internal ranks collapse to the U closed form.
// Each unsorted group value binary-searches the sorted reference; no group
// sort.
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

// LARGE/HUGE rank kernel; LARGE smem-sorts, HUGE reads CUB-sorted groups.
// Post-sort mid-rank/tie body is shared and each group owns its output row.
template <bool SMEM_SORT>
__global__ void ovo_rank_sorted_kernel(const float* __restrict__ ref_sorted,
                                       const float* __restrict__ grp_in,
                                       const int* __restrict__ grp_offsets,
                                       const double* __restrict__ ref_tie_sums,
                                       double* __restrict__ rank_sums,
                                       double* __restrict__ tie_corr, int n_ref,
                                       int n_all_grp, int n_cols, int n_groups,
                                       bool compute_tie_corr, int large_padded,
                                       int skip_n_grp_le, int skip_n_grp_gt) {
    int col = blockIdx.x;
    int grp = blockIdx.y;
    if (col >= n_cols || grp >= n_groups) return;

    int g_start = grp_offsets[grp];
    int n_grp = grp_offsets[grp + 1] - g_start;
    if (n_grp <= skip_n_grp_le) return;
    if (n_grp > skip_n_grp_gt) return;
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

// LARGE analytic-zero path for nonnegative sparse data.
// Sort only stored positives; zeros are handled from counts.
__global__ void ovo_rank_smem_analytic_kernel(
    const float* __restrict__ ref_sorted, const float* __restrict__ grp_in,
    const int* __restrict__ grp_offsets,
    const double* __restrict__ ref_tie_sums, double* __restrict__ rank_sums,
    double* __restrict__ tie_corr, int n_ref, int n_all_grp, int n_cols,
    int n_groups, bool compute_tie_corr, int skip_n_grp_le, int skip_n_grp_gt) {
    int col = blockIdx.x;
    int grp = blockIdx.y;
    if (col >= n_cols || grp >= n_groups) return;

    int g_start = grp_offsets[grp];
    int n_grp = grp_offsets[grp + 1] - g_start;
    if (n_grp <= skip_n_grp_le) return;
    if (n_grp > skip_n_grp_gt) return;
    if (n_grp == 0) {
        if (threadIdx.x == 0) {
            rank_sums[grp * n_cols + col] = 0.0;
            if (compute_tie_corr) tie_corr[grp * n_cols + col] = 1.0;
        }
        return;
    }

    const float* ref_col = ref_sorted + (long long)col * n_ref;
    const float* grp_col = grp_in + (long long)col * n_all_grp + g_start;

    extern __shared__ float grp_smem[];
    __shared__ double warp_buf[32];
    __shared__ int sh_nnz;
    __shared__ int sh_ref_zeros;
    if (threadIdx.x == 0) {
        sh_nnz = 0;
        sh_ref_zeros = sorted_upper_bound(ref_col, 0, n_ref, 0.0f);
    }
    __syncthreads();

    for (int i = threadIdx.x; i < n_grp; i += blockDim.x) {
        float v = grp_col[i];
        if (v > 0.0f) grp_smem[atomicAdd(&sh_nnz, 1)] = v;
    }
    __syncthreads();
    int nnz = sh_nnz;
    int ref_zeros = sh_ref_zeros;
    int n_grp_zero = n_grp - nnz;
    int total_zero = ref_zeros + n_grp_zero;

    // Pad positives to a power of two for bitonic_sort_smem.
    int padded = 1;
    while (padded < nnz) padded <<= 1;
    for (int i = nnz + threadIdx.x; i < padded; i += blockDim.x)
        grp_smem[i] = __int_as_float(0x7f800000);
    __syncthreads();
    if (nnz > 1) bitonic_sort_smem(grp_smem, padded);
    __syncthreads();

    // Positive ranks are shifted by group zeros, which sort before positives.
    double zero_rank =
        (total_zero > 0) ? ((double)total_zero + 1.0) / 2.0 : 0.0;
    double local_sum =
        (threadIdx.x == 0) ? (double)n_grp_zero * zero_rank : 0.0;
    int ref_lb = 0, ref_ub = 0, grp_lb = 0, grp_ub = 0;
    for (int i = threadIdx.x; i < nnz; i += blockDim.x) {
        OvoRank r = ovo_mid_rank(ref_col, n_ref, grp_smem, nnz, grp_smem[i],
                                 ref_lb, ref_ub, grp_lb, grp_ub);
        local_sum += r.mid_rank + (double)n_grp_zero;
    }
    double total = wilcoxon_block_sum(local_sum, warp_buf);
    if (threadIdx.x == 0) rank_sums[grp * n_cols + col] = total;

    if (!compute_tie_corr) return;
    __syncthreads();

    // Add nonzero tie deltas; zero ties are added below via T(ref+grp)-T(ref).
    double local_tie = 0.0;
    for (int i = threadIdx.x; i < nnz; i += blockDim.x) {
        if (i == 0 || grp_smem[i] != grp_smem[i - 1]) {
            float v = grp_smem[i];
            int gub = sorted_upper_bound(grp_smem, i + 1, nnz, v);
            double cg = (double)(gub - i);
            int rlo = sorted_lower_bound(ref_col, 0, n_ref, v);
            int rub = sorted_upper_bound(ref_col, rlo, n_ref, v);
            double cr = (double)(rub - rlo);
            double group_tie = (cg > 1.0) ? (cg * cg * cg - cg) : 0.0;
            local_tie += group_tie;
            if (cr > 0.0) {
                double comb = cr + cg;
                double ref_tie = (cr > 1.0) ? (cr * cr * cr - cr) : 0.0;
                local_tie += comb * comb * comb - comb - ref_tie - group_tie;
            }
        }
    }
    double tie = wilcoxon_block_sum(local_tie, warp_buf);
    if (threadIdx.x == 0) {
        double zd = 0.0;
        if (total_zero > 1)
            zd += (double)total_zero * total_zero * total_zero - total_zero;
        if (ref_zeros > 1)
            zd -= (double)ref_zeros * ref_zeros * ref_zeros - ref_zeros;
        tie_corr[grp * n_cols + col] =
            finalize_tie_corr(n_ref + n_grp, ref_tie_sums[col] + tie + zd);
    }
}

// Compact HUGE-band positives and emit [base, base + nnz) segments.
__global__ void compact_huge_nonzeros_kernel(
    const float* __restrict__ grp_dense, const int* __restrict__ grp_offsets,
    const int* __restrict__ group_ids, float* __restrict__ grp_nz,
    int* __restrict__ seg_begins, int* __restrict__ seg_ends, int n_all_grp,
    int n_sort_groups, int sb_cols) {
    int col = blockIdx.x;
    int local = blockIdx.y;
    if (col >= sb_cols || local >= n_sort_groups) return;
    int g = group_ids[local];
    int g_start = grp_offsets[g];
    int n_grp = grp_offsets[g + 1] - g_start;
    size_t base = (size_t)col * n_all_grp + g_start;
    int f = col * n_sort_groups + local;

    __shared__ int cnt;
    if (threadIdx.x == 0) cnt = 0;
    __syncthreads();
    for (int i = threadIdx.x; i < n_grp; i += blockDim.x) {
        float v = grp_dense[base + i];
        if (v > 0.0f) grp_nz[base + atomicAdd(&cnt, 1)] = v;
    }
    __syncthreads();
    if (threadIdx.x == 0) {
        seg_begins[f] = (int)base;
        seg_ends[f] = (int)base + cnt;
    }
}

// Rank HUGE-band groups from sorted positives plus the zero block.
__global__ void ovo_rank_huge_analytic_kernel(
    const float* __restrict__ ref_sorted,
    const float* __restrict__ grp_nz_sorted,
    const int* __restrict__ grp_offsets, const int* __restrict__ group_ids,
    const int* __restrict__ seg_begins, const int* __restrict__ seg_ends,
    const double* __restrict__ ref_tie_sums, double* __restrict__ rank_sums,
    double* __restrict__ tie_corr, int n_ref, int n_all_grp, int n_cols,
    int n_sort_groups, bool compute_tie_corr) {
    int col = blockIdx.x;
    int local = blockIdx.y;
    if (col >= n_cols || local >= n_sort_groups) return;
    int grp = group_ids[local];
    int g_start = grp_offsets[grp];
    int n_grp = grp_offsets[grp + 1] - g_start;
    int f = col * n_sort_groups + local;
    int b = seg_begins[f];
    int nnz = seg_ends[f] - b;
    const float* nz = grp_nz_sorted + b;
    const float* ref_col = ref_sorted + (long long)col * n_ref;

    __shared__ double warp_buf[32];
    __shared__ int sh_ref_zeros;
    if (threadIdx.x == 0)
        sh_ref_zeros = sorted_upper_bound(ref_col, 0, n_ref, 0.0f);
    __syncthreads();
    int ref_zeros = sh_ref_zeros;
    int n_grp_zero = n_grp - nnz;
    int total_zero = ref_zeros + n_grp_zero;
    double zero_rank =
        (total_zero > 0) ? ((double)total_zero + 1.0) / 2.0 : 0.0;

    double local_sum =
        (threadIdx.x == 0) ? (double)n_grp_zero * zero_rank : 0.0;
    int ref_lb = 0, ref_ub = 0, grp_lb = 0, grp_ub = 0;
    for (int i = threadIdx.x; i < nnz; i += blockDim.x) {
        OvoRank r = ovo_mid_rank(ref_col, n_ref, nz, nnz, nz[i], ref_lb, ref_ub,
                                 grp_lb, grp_ub);
        local_sum += r.mid_rank + (double)n_grp_zero;
    }
    double total = wilcoxon_block_sum(local_sum, warp_buf);
    if (threadIdx.x == 0) rank_sums[grp * n_cols + col] = total;

    if (!compute_tie_corr) return;
    __syncthreads();
    double local_tie = 0.0;
    for (int i = threadIdx.x; i < nnz; i += blockDim.x) {
        if (i == 0 || nz[i] != nz[i - 1]) {
            float v = nz[i];
            int gub = sorted_upper_bound(nz, i + 1, nnz, v);
            double cg = (double)(gub - i);
            int rlo = sorted_lower_bound(ref_col, 0, n_ref, v);
            int rub = sorted_upper_bound(ref_col, rlo, n_ref, v);
            double cr = (double)(rub - rlo);
            double group_tie = (cg > 1.0) ? (cg * cg * cg - cg) : 0.0;
            local_tie += group_tie;
            if (cr > 0.0) {
                double comb = cr + cg;
                double ref_tie = (cr > 1.0) ? (cr * cr * cr - cr) : 0.0;
                local_tie += comb * comb * comb - comb - ref_tie - group_tie;
            }
        }
    }
    double tie = wilcoxon_block_sum(local_tie, warp_buf);
    if (threadIdx.x == 0) {
        double zd = 0.0;
        if (total_zero > 1)
            zd += (double)total_zero * total_zero * total_zero - total_zero;
        if (ref_zeros > 1)
            zd -= (double)ref_zeros * ref_zeros * ref_zeros - ref_zeros;
        tie_corr[grp * n_cols + col] =
            finalize_tie_corr(n_ref + n_grp, ref_tie_sums[col] + tie + zd);
    }
}

// MEDIUM tie helper: sorted-reference contribution, one block per column.
// Rank kernels add only group-only/ref-overlap deltas.
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

// MEDIUM fused kernel: ref binary searches plus in-group scan over smem values.
// Tie correction starts from ref_tie_sums[col] and adds only group deltas.
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

// MEDIUM analytic-zero path for nonnegative sparse data.
__global__ void ovo_rank_medium_analytic_kernel(
    const float* __restrict__ ref_sorted, const float* __restrict__ grp_dense,
    const int* __restrict__ grp_offsets,
    const double* __restrict__ ref_tie_sums, double* __restrict__ rank_sums,
    double* __restrict__ tie_corr, int n_ref, int n_all_grp, int n_cols,
    int n_groups, bool compute_tie_corr, int skip_n_grp_le, int max_n_grp_le) {
    int col = blockIdx.x;
    int grp = blockIdx.y;
    if (col >= n_cols || grp >= n_groups) return;

    int g_start = grp_offsets[grp];
    int n_grp = grp_offsets[grp + 1] - g_start;
    if (n_grp <= skip_n_grp_le || n_grp > max_n_grp_le) return;

    extern __shared__ char smem_raw[];
    float* grp_smem = (float*)smem_raw;
    double* warp_buf = (double*)(smem_raw + max_n_grp_le * sizeof(float));
    __shared__ int sh_nnz;
    __shared__ int sh_ref_zeros;

    const float* ref_col = ref_sorted + (long long)col * n_ref;
    const float* grp_col = grp_dense + (long long)col * n_all_grp + g_start;
    if (threadIdx.x == 0) {
        sh_nnz = 0;
        sh_ref_zeros = sorted_upper_bound(ref_col, 0, n_ref, 0.0f);
    }
    __syncthreads();
    for (int i = threadIdx.x; i < n_grp; i += blockDim.x) {
        float v = grp_col[i];
        if (v > 0.0f) grp_smem[atomicAdd(&sh_nnz, 1)] = v;
    }
    __syncthreads();
    int nnz = sh_nnz;
    int ref_zeros = sh_ref_zeros;
    int n_grp_zero = n_grp - nnz;
    int total_zero = ref_zeros + n_grp_zero;
    double zero_rank =
        (total_zero > 0) ? ((double)total_zero + 1.0) / 2.0 : 0.0;

    double local_sum =
        (threadIdx.x == 0) ? (double)n_grp_zero * zero_rank : 0.0;
    double local_tie = 0.0;
    for (int i = threadIdx.x; i < nnz; i += blockDim.x) {
        float v = grp_smem[i];
        int n_lt_ref = sorted_lower_bound(ref_col, 0, n_ref, v);
        int n_eq_ref =
            sorted_upper_bound(ref_col, n_lt_ref, n_ref, v) - n_lt_ref;
        int n_lt_grp = 0;
        int n_eq_grp = 0;
        bool first_in_grp = true;
        for (int j = 0; j < nnz; ++j) {
            float w = grp_smem[j];
            if (w < v) ++n_lt_grp;
            if (w == v) {
                ++n_eq_grp;
                if (j < i) first_in_grp = false;
            }
        }
        local_sum += (double)(n_lt_ref + n_lt_grp + n_grp_zero) +
                     ((double)(n_eq_ref + n_eq_grp) + 1.0) / 2.0;
        if (compute_tie_corr && first_in_grp) {
            double cg = (double)n_eq_grp;
            double cr = (double)n_eq_ref;
            double group_tie = (cg > 1.0) ? (cg * cg * cg - cg) : 0.0;
            local_tie += group_tie;
            if (cr > 0.0) {
                double combined = cr + cg;
                double ref_tie = (cr > 1.0) ? (cr * cr * cr - cr) : 0.0;
                local_tie += combined * combined * combined - combined -
                             ref_tie - group_tie;
            }
        }
    }
    double total = wilcoxon_block_sum(local_sum, warp_buf);
    if (threadIdx.x == 0) rank_sums[grp * n_cols + col] = total;

    if (!compute_tie_corr) return;
    __syncthreads();
    double tie = wilcoxon_block_sum(local_tie, warp_buf);
    if (threadIdx.x == 0) {
        double zd = 0.0;
        if (total_zero > 1)
            zd += (double)total_zero * total_zero * total_zero - total_zero;
        if (ref_zeros > 1)
            zd -= (double)ref_zeros * ref_zeros * ref_zeros - ref_zeros;
        tie_corr[grp * n_cols + col] =
            finalize_tie_corr(n_ref + n_grp, ref_tie_sums[col] + tie + zd);
    }
}
