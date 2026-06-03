#pragma once

#include <cuda_runtime.h>

#include "wilcoxon_fast_common.cuh"

// ============================================================================
// Warp reduction helper (sum doubles across block via warp_buf)
// ============================================================================

__device__ __forceinline__ double block_reduce_sum(double val,
                                                   double* warp_buf) {
#pragma unroll
    for (int off = 16; off > 0; off >>= 1)
        val += __shfl_down_sync(0xffffffff, val, off);
    int lane = threadIdx.x & 31;
    int wid = threadIdx.x >> 5;
    if (lane == 0) warp_buf[wid] = val;
    __syncthreads();
    if (threadIdx.x < 32) {
        double v2 = (threadIdx.x < ((blockDim.x + 31) >> 5))
                        ? warp_buf[threadIdx.x]
                        : 0.0;
#pragma unroll
        for (int off = 16; off > 0; off >>= 1)
            v2 += __shfl_down_sync(0xffffffff, v2, off);
        return v2;  // only lane 0 of warp 0 has the final result
    }
    return 0.0;
}

// ============================================================================
// Parallel tie correction — all threads collaborate.
//
// For each unique value in the combined sorted (ref, grp) arrays, accumulate
// t^3 - t where t = count of that value.  Uses two passes:
//   1. Iterate unique values in ref_col, count in both arrays.
//   2. Iterate unique values in grp_col that do NOT appear in ref_col.
//
// Incremental binary search bounds exploit monotonicity within each thread's
// stride to reduce total search work.
//
// Caller must __syncthreads() before calling.  warp_buf is reused for
// reduction (32 doubles, shared memory).
// ============================================================================

__device__ __forceinline__ void compute_tie_correction_parallel(
    const float* ref_col, int n_ref, const float* grp_col, int n_grp,
    double* warp_buf, double* out) {
    double local_tie = 0.0;

    // Pass 1: unique values in ref_col
    int grp_lb = 0, grp_ub = 0;
    for (int i = threadIdx.x; i < n_ref; i += blockDim.x) {
        if (i == 0 || ref_col[i] != ref_col[i - 1]) {
            float v = ref_col[i];

            // Count in ref: upper_bound from i+1
            int lo = i + 1, hi = n_ref;
            while (lo < hi) {
                int m = lo + ((hi - lo) >> 1);
                if (ref_col[m] <= v)
                    lo = m + 1;
                else
                    hi = m;
            }
            int cnt_ref = lo - i;

            // Count in grp: incremental lower/upper bound
            lo = grp_lb;
            hi = n_grp;
            while (lo < hi) {
                int m = lo + ((hi - lo) >> 1);
                if (grp_col[m] < v)
                    lo = m + 1;
                else
                    hi = m;
            }
            int lb = lo;
            grp_lb = lb;

            lo = (grp_ub > lb) ? grp_ub : lb;
            hi = n_grp;
            while (lo < hi) {
                int m = lo + ((hi - lo) >> 1);
                if (grp_col[m] <= v)
                    lo = m + 1;
                else
                    hi = m;
            }
            int cnt_grp = lo - lb;
            grp_ub = lo;

            int cnt = cnt_ref + cnt_grp;
            if (cnt > 1) {
                double t = (double)cnt;
                local_tie += t * t * t - t;
            }
        }
    }

    // Pass 2: unique values in grp_col that are absent from ref_col
    int ref_lb = 0;
    for (int i = threadIdx.x; i < n_grp; i += blockDim.x) {
        if (i == 0 || grp_col[i] != grp_col[i - 1]) {
            float v = grp_col[i];

            // Incremental lower_bound in ref
            int lo = ref_lb, hi = n_ref;
            while (lo < hi) {
                int m = lo + ((hi - lo) >> 1);
                if (ref_col[m] < v)
                    lo = m + 1;
                else
                    hi = m;
            }
            ref_lb = lo;

            if (lo >= n_ref || ref_col[lo] != v) {
                // Value not in ref — count in grp only (upper_bound from i+1)
                lo = i + 1;
                hi = n_grp;
                while (lo < hi) {
                    int m = lo + ((hi - lo) >> 1);
                    if (grp_col[m] <= v)
                        lo = m + 1;
                    else
                        hi = m;
                }
                int cnt = lo - i;
                if (cnt > 1) {
                    double t = (double)cnt;
                    local_tie += t * t * t - t;
                }
            }
        }
    }

    // Block-wide reduction
    double tie_sum = block_reduce_sum(local_tie, warp_buf);
    if (threadIdx.x == 0) {
        int n = n_ref + n_grp;
        double dn = (double)n;
        double denom = dn * dn * dn - dn;
        *out = (denom > 0.0) ? (1.0 - tie_sum / denom) : 1.0;
    }
}

// ============================================================================
// Batched rank sums — pre-sorted (binary search, no shared memory sort)
// Used by the OVO streaming pipeline in wilcoxon_streaming.cu.
//
// Incremental binary search: each thread carries forward lower/upper bound
// positions across loop iterations, exploiting the monotonicity of the
// sorted grp_col values within each thread's stride.
// ============================================================================

__global__ void ovo_rank_huge_kernel(
    const float* __restrict__ ref_sorted, const float* __restrict__ grp_sorted,
    const int* __restrict__ grp_offsets, double* __restrict__ rank_sums,
    double* __restrict__ tie_corr, int n_ref, int n_all_grp, int n_cols,
    int n_groups, bool compute_tie_corr, int skip_n_grp_le /*= 0*/) {
    int col = blockIdx.x;
    int grp = blockIdx.y;
    if (col >= n_cols || grp >= n_groups) return;

    int g_start = grp_offsets[grp];
    int g_end = grp_offsets[grp + 1];
    int n_grp = g_end - g_start;

    // Size-gated dispatch (see ovo_rank_large_kernel for the contract).
    if (n_grp <= skip_n_grp_le) return;

    if (n_grp == 0) {
        if (threadIdx.x == 0) {
            rank_sums[grp * n_cols + col] = 0.0;
            if (compute_tie_corr) tie_corr[grp * n_cols + col] = 1.0;
        }
        return;
    }

    const float* ref_col = ref_sorted + (long long)col * n_ref;
    const float* grp_col = grp_sorted + (long long)col * n_all_grp + g_start;

    // Incremental binary search bounds (advance monotonically per thread)
    int ref_lb = 0, ref_ub = 0;
    int grp_lb = 0, grp_ub = 0;
    double local_sum = 0.0;

    for (int i = threadIdx.x; i < n_grp; i += blockDim.x) {
        float v = grp_col[i];
        int lo, hi;

        // Lower bound in ref (from ref_lb)
        lo = ref_lb;
        hi = n_ref;
        while (lo < hi) {
            int m = lo + ((hi - lo) >> 1);
            if (ref_col[m] < v)
                lo = m + 1;
            else
                hi = m;
        }
        int n_lt_ref = lo;
        ref_lb = n_lt_ref;

        // Upper bound in ref (from max(ref_ub, n_lt_ref))
        lo = (ref_ub > n_lt_ref) ? ref_ub : n_lt_ref;
        hi = n_ref;
        while (lo < hi) {
            int m = lo + ((hi - lo) >> 1);
            if (ref_col[m] <= v)
                lo = m + 1;
            else
                hi = m;
        }
        int n_eq_ref = lo - n_lt_ref;
        ref_ub = lo;

        // Lower bound in grp (from grp_lb)
        lo = grp_lb;
        hi = n_grp;
        while (lo < hi) {
            int m = lo + ((hi - lo) >> 1);
            if (grp_col[m] < v)
                lo = m + 1;
            else
                hi = m;
        }
        int n_lt_grp = lo;
        grp_lb = n_lt_grp;

        // Upper bound in grp (from max(grp_ub, n_lt_grp))
        lo = (grp_ub > n_lt_grp) ? grp_ub : n_lt_grp;
        hi = n_grp;
        while (lo < hi) {
            int m = lo + ((hi - lo) >> 1);
            if (grp_col[m] <= v)
                lo = m + 1;
            else
                hi = m;
        }
        int n_eq_grp = lo - n_lt_grp;
        grp_ub = lo;

        local_sum += (double)(n_lt_ref + n_lt_grp) +
                     ((double)(n_eq_ref + n_eq_grp) + 1.0) / 2.0;
    }

    __shared__ double warp_buf[32];
    double total = block_reduce_sum(local_sum, warp_buf);
    if (threadIdx.x == 0) rank_sums[grp * n_cols + col] = total;

    if (!compute_tie_corr) return;
    __syncthreads();

    compute_tie_correction_parallel(ref_col, n_ref, grp_col, n_grp, warp_buf,
                                    &tie_corr[grp * n_cols + col]);
}

// ============================================================================
// Tier 1 fused kernel: smem bitonic sort + binary search rank sums
// For small groups (< ~2K cells).  No CUB, no global memory sort buffers.
// Grid: (n_cols, n_groups), Block: min(large_padded, 512)
// Shared memory: large_padded floats + 32 doubles (warp reduction)
// ============================================================================

__global__ void ovo_rank_large_kernel(
    const float* __restrict__ ref_sorted,  // F-order (n_ref, n_cols) sorted
    const float* __restrict__ grp_dense,   // F-order (n_all_grp, n_cols)
                                           // unsorted
    const int* __restrict__ grp_offsets,   // (n_groups + 1,)
    double* __restrict__ rank_sums,        // (n_groups, n_cols) row-major
    double* __restrict__ tie_corr,         // (n_groups, n_cols) row-major
    int n_ref, int n_all_grp, int n_cols, int n_groups, bool compute_tie_corr,
    int large_padded, int skip_n_grp_le /*= 0*/) {
    int col = blockIdx.x;
    int grp = blockIdx.y;
    if (col >= n_cols || grp >= n_groups) return;

    int g_start = grp_offsets[grp];
    int g_end = grp_offsets[grp + 1];
    int n_grp = g_end - g_start;

    // Size-gated dispatch: when co-launched with the Tier 0 warp kernel we
    // skip groups it's already handling.  Each group owns its own
    // rank_sums row, so the two kernels' writes never alias.
    if (n_grp <= skip_n_grp_le) return;

    if (n_grp == 0) {
        if (threadIdx.x == 0) {
            rank_sums[grp * n_cols + col] = 0.0;
            if (compute_tie_corr) tie_corr[grp * n_cols + col] = 1.0;
        }
        return;
    }

    // Shared memory: [large_padded floats | 32 doubles for warp reduction]
    extern __shared__ char smem_raw[];
    float* grp_smem = (float*)smem_raw;
    double* warp_buf = (double*)(smem_raw + large_padded * sizeof(float));

    // Load group data into shared memory, pad with +INF
    const float* grp_col = grp_dense + (long long)col * n_all_grp + g_start;
    for (int i = threadIdx.x; i < n_grp; i += blockDim.x)
        grp_smem[i] = grp_col[i];
    for (int i = n_grp + threadIdx.x; i < large_padded; i += blockDim.x)
        grp_smem[i] = __int_as_float(0x7f800000);  // +INF
    __syncthreads();

    // Bitonic sort in shared memory
    for (int k = 2; k <= large_padded; k <<= 1) {
        for (int j = k >> 1; j > 0; j >>= 1) {
            for (int i = threadIdx.x; i < large_padded; i += blockDim.x) {
                int ixj = i ^ j;
                if (ixj > i) {
                    bool asc = ((i & k) == 0);
                    float a = grp_smem[i], b = grp_smem[ixj];
                    if (asc ? (a > b) : (a < b)) {
                        grp_smem[i] = b;
                        grp_smem[ixj] = a;
                    }
                }
            }
            __syncthreads();
        }
    }

    // Binary search each sorted grp element against sorted ref
    // Incremental bounds: values are monotonic within each thread's stride
    const float* ref_col = ref_sorted + (long long)col * n_ref;
    int ref_lb = 0, ref_ub = 0;
    int grp_lb = 0, grp_ub = 0;
    double local_sum = 0.0;

    for (int i = threadIdx.x; i < n_grp; i += blockDim.x) {
        float v = grp_smem[i];
        int lo, hi;

        lo = ref_lb;
        hi = n_ref;
        while (lo < hi) {
            int m = lo + ((hi - lo) >> 1);
            if (ref_col[m] < v)
                lo = m + 1;
            else
                hi = m;
        }
        int n_lt_ref = lo;
        ref_lb = n_lt_ref;

        lo = (ref_ub > n_lt_ref) ? ref_ub : n_lt_ref;
        hi = n_ref;
        while (lo < hi) {
            int m = lo + ((hi - lo) >> 1);
            if (ref_col[m] <= v)
                lo = m + 1;
            else
                hi = m;
        }
        int n_eq_ref = lo - n_lt_ref;
        ref_ub = lo;

        lo = grp_lb;
        hi = n_grp;
        while (lo < hi) {
            int m = lo + ((hi - lo) >> 1);
            if (grp_smem[m] < v)
                lo = m + 1;
            else
                hi = m;
        }
        int n_lt_grp = lo;
        grp_lb = n_lt_grp;

        lo = (grp_ub > n_lt_grp) ? grp_ub : n_lt_grp;
        hi = n_grp;
        while (lo < hi) {
            int m = lo + ((hi - lo) >> 1);
            if (grp_smem[m] <= v)
                lo = m + 1;
            else
                hi = m;
        }
        int n_eq_grp = lo - n_lt_grp;
        grp_ub = lo;

        local_sum += (double)(n_lt_ref + n_lt_grp) +
                     ((double)(n_eq_ref + n_eq_grp) + 1.0) / 2.0;
    }

    // Block reduction → write rank_sums
    double total = block_reduce_sum(local_sum, warp_buf);
    if (threadIdx.x == 0) rank_sums[grp * n_cols + col] = total;

    if (!compute_tie_corr) return;
    __syncthreads();

    // Parallel tie correction (grp_smem is sorted shared memory)
    compute_tie_correction_parallel(ref_col, n_ref, grp_smem, n_grp, warp_buf,
                                    &tie_corr[grp * n_cols + col]);
}

// ============================================================================
// Tier 2 helper: tie contribution of the sorted reference alone.
// One block per column.  The medium unsorted-rank kernel uses this as a base
// and only adds group-only/overlap deltas from the unsorted group values.
// ============================================================================

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
            int lo = i + 1, hi = n_ref;
            while (lo < hi) {
                int m = lo + ((hi - lo) >> 1);
                if (ref_col[m] <= v)
                    lo = m + 1;
                else
                    hi = m;
            }
            int cnt = lo - i;
            if (cnt > 1) {
                double t = (double)cnt;
                local_tie += t * t * t - t;
            }
        }
    }

    __shared__ double warp_buf[32];
    double total = block_reduce_sum(local_tie, warp_buf);
    if (threadIdx.x == 0) ref_tie_sums[col] = total;
}

__global__ void ovo_rank_small_kernel(
    const float* __restrict__ ref_sorted, const float* __restrict__ grp_dense,
    const int* __restrict__ grp_offsets,
    const double* __restrict__ ref_tie_sums, double* __restrict__ rank_sums,
    double* __restrict__ tie_corr, int n_ref, int n_all_grp, int n_cols,
    int n_groups, bool compute_tie_corr, int skip_n_grp_le) {
    int col = blockIdx.x;
    int grp = blockIdx.y;
    if (col >= n_cols || grp >= n_groups) return;

    int g_start = grp_offsets[grp];
    int g_end = grp_offsets[grp + 1];
    int n_grp = g_end - g_start;
    if (n_grp <= skip_n_grp_le || n_grp > OVO_SMALL_MAX) return;

    __shared__ float grp_smem[OVO_SMALL_MAX];
    __shared__ double warp_buf[WARP_REDUCE_BUF];

    const float* grp_col = grp_dense + (long long)col * n_all_grp + g_start;
    const float POS_INF = __int_as_float(0x7f800000);
    if (threadIdx.x < OVO_SMALL_MAX) {
        grp_smem[threadIdx.x] =
            (threadIdx.x < n_grp) ? grp_col[threadIdx.x] : POS_INF;
    }
    __syncthreads();

    for (int k = 2; k <= OVO_SMALL_MAX; k <<= 1) {
        for (int j = k >> 1; j > 0; j >>= 1) {
            int i = threadIdx.x;
            int ixj = i ^ j;
            if (i < OVO_SMALL_MAX && ixj > i) {
                bool asc = ((i & k) == 0);
                float a = grp_smem[i], b = grp_smem[ixj];
                if (asc ? (a > b) : (a < b)) {
                    grp_smem[i] = b;
                    grp_smem[ixj] = a;
                }
            }
            __syncthreads();
        }
    }

    const float* ref_col = ref_sorted + (long long)col * n_ref;
    double local_sum = 0.0;
    double local_tie_delta = 0.0;

    if (threadIdx.x < n_grp) {
        float v = grp_smem[threadIdx.x];

        int lo = 0, hi = n_ref;
        while (lo < hi) {
            int m = lo + ((hi - lo) >> 1);
            if (ref_col[m] < v)
                lo = m + 1;
            else
                hi = m;
        }
        int n_lt_ref = lo;
        hi = n_ref;
        while (lo < hi) {
            int m = lo + ((hi - lo) >> 1);
            if (ref_col[m] <= v)
                lo = m + 1;
            else
                hi = m;
        }
        int n_eq_ref = lo - n_lt_ref;

        lo = 0;
        hi = n_grp;
        while (lo < hi) {
            int m = lo + ((hi - lo) >> 1);
            if (grp_smem[m] < v)
                lo = m + 1;
            else
                hi = m;
        }
        int n_lt_grp = lo;
        hi = n_grp;
        while (lo < hi) {
            int m = lo + ((hi - lo) >> 1);
            if (grp_smem[m] <= v)
                lo = m + 1;
            else
                hi = m;
        }
        int n_eq_grp = lo - n_lt_grp;

        local_sum += (double)(n_lt_ref + n_lt_grp) +
                     ((double)(n_eq_ref + n_eq_grp) + 1.0) / 2.0;

        if (compute_tie_corr &&
            (threadIdx.x == 0 || v != grp_smem[threadIdx.x - 1])) {
            double combined = (double)(n_eq_ref + n_eq_grp);
            if (combined > 1.0) {
                local_tie_delta += combined * combined * combined - combined;
            }
            if (n_eq_ref > 1) {
                double cr = (double)n_eq_ref;
                local_tie_delta -= cr * cr * cr - cr;
            }
        }
    }

    double total = block_reduce_sum(local_sum, warp_buf);
    if (threadIdx.x == 0) rank_sums[grp * n_cols + col] = total;

    if (!compute_tie_corr) return;
    __syncthreads();

    double tie_delta = block_reduce_sum(local_tie_delta, warp_buf);
    if (threadIdx.x == 0) {
        int n = n_ref + n_grp;
        double dn = (double)n;
        double denom = dn * dn * dn - dn;
        double tie_sum = ref_tie_sums[col] + tie_delta;
        tie_corr[grp * n_cols + col] =
            (denom > 0.0) ? (1.0 - tie_sum / denom) : 1.0;
    }
}

// ============================================================================
// Tier 2 fused kernel: no-sort direct rank for medium groups.
//
// Avoids the smem bitonic sort for groups in (skip_n_grp_le,
// max_n_grp_le].  Ranks are computed from ref binary searches plus an
// in-group scan over unsorted shared values.  Tie correction starts from
// ref_tie_sums[col] and adds only group-only / ref-overlap deltas.
// ============================================================================

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

        int lo = 0, hi = n_ref;
        while (lo < hi) {
            int m = lo + ((hi - lo) >> 1);
            if (ref_col[m] < v)
                lo = m + 1;
            else
                hi = m;
        }
        int n_lt_ref = lo;

        hi = n_ref;
        while (lo < hi) {
            int m = lo + ((hi - lo) >> 1);
            if (ref_col[m] <= v)
                lo = m + 1;
            else
                hi = m;
        }
        int n_eq_ref = lo - n_lt_ref;

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

    double total = block_reduce_sum(local_sum, warp_buf);
    if (threadIdx.x == 0) rank_sums[grp * n_cols + col] = total;

    if (!compute_tie_corr) return;
    __syncthreads();

    double tie_delta = block_reduce_sum(local_tie_delta, warp_buf);
    if (threadIdx.x == 0) {
        int n = n_ref + n_grp;
        double dn = (double)n;
        double denom = dn * dn * dn - dn;
        double tie_sum = ref_tie_sums[col] + tie_delta;
        tie_corr[grp * n_cols + col] =
            (denom > 0.0) ? (1.0 - tie_sum / denom) : 1.0;
    }
}

// ============================================================================
// Warp-scoped tie correction for Tier 0.
//
// Sorted values live in a 32-lane register (one per lane, with unused lanes
// carrying +INF).  Walks unique values via lane-step differentials and
// counts ties across the sorted ref column via binary search.  All the
// sync is __syncwarp — no smem, no __syncthreads.
// ============================================================================

__device__ __forceinline__ double warp_tie_sum(const float* ref_col, int n_ref,
                                               float v_lane, int n_grp,
                                               unsigned int active_mask) {
    int lane = threadIdx.x & 31;
    double local_tie = 0.0;

    // Pass 1: for each unique value in ref_col, count occurrences in ref and
    // in the sorted group (held in register v_lane across 32 lanes).
    for (int base = 0; base < n_ref; base += 32) {
        int i = base + lane;
        bool in_ref_lane = (i < n_ref);
        float v = in_ref_lane ? ref_col[i] : 0.0f;
        bool is_first = in_ref_lane && ((i == 0) || (v != ref_col[i - 1]));
        int cnt_ref = 0;
        if (is_first) {
            // Count in ref: upper_bound from i+1
            int lo = i + 1, hi = n_ref;
            while (lo < hi) {
                int m = lo + ((hi - lo) >> 1);
                if (ref_col[m] <= v)
                    lo = m + 1;
                else
                    hi = m;
            }
            cnt_ref = lo - i;
        }

        // Count in grp: look up how many lanes hold v_lane == v.  All lanes
        // execute the shuffle loop; only lanes owning a unique ref value use
        // the result.
        int cnt_grp = 0;
#pragma unroll
        for (int lane_i = 0; lane_i < OVO_WARP_MAX; ++lane_i) {
            float vi = __shfl_sync(0xffffffff, v_lane, lane_i);
            if (is_first && lane_i < n_grp && vi == v) ++cnt_grp;
        }

        if (is_first) {
            int cnt = cnt_ref + cnt_grp;
            if (cnt > 1) {
                double t = (double)cnt;
                local_tie += t * t * t - t;
            }
        }
    }

    // Pass 2: unique values in grp that are absent from ref.
    // Walk lanes 0..n_grp-1; for each lane whose v differs from prev lane's,
    // binary-search ref for v.  If not present, count consecutive matching
    // lanes (tie block).
    if (lane < n_grp) {
        float v = v_lane;
        float prev_lane_v =
            __shfl_sync(active_mask, v_lane, (lane > 0) ? lane - 1 : 0);
        float v_prev =
            (lane > 0) ? prev_lane_v : __int_as_float(0xff800000);  // -INF
        bool first_in_grp = (lane == 0) || (v != v_prev);
        bool in_ref = false;
        if (first_in_grp) {
            // Binary search in ref.
            int lo = 0, hi = n_ref;
            while (lo < hi) {
                int m = lo + ((hi - lo) >> 1);
                if (ref_col[m] < v)
                    lo = m + 1;
                else
                    hi = m;
            }
            in_ref = (lo < n_ref) && (ref_col[lo] == v);
        }

        // Count how many lanes ≥ this lane hold the same v.  Keep the shuffle
        // uniform across active lanes even though only unique, ref-absent
        // group values consume the count.
        int cnt = 0;
#pragma unroll
        for (int lane_i = 0; lane_i < OVO_WARP_MAX; ++lane_i) {
            int src_lane = (lane_i < n_grp) ? lane_i : 0;
            float vi = __shfl_sync(active_mask, v_lane, src_lane);
            if (first_in_grp && !in_ref && lane_i >= lane && lane_i < n_grp &&
                vi == v) {
                ++cnt;
            }
        }
        if (first_in_grp && !in_ref && cnt > 1) {
            double t = (double)cnt;
            local_tie += t * t * t - t;
        }
    }

    // Warp reduce.
#pragma unroll
    for (int off = 16; off > 0; off >>= 1)
        local_tie += __shfl_down_sync(0xffffffff, local_tie, off);
    return local_tie;  // meaningful on lane 0.
}

__device__ __forceinline__ double warp_tie_delta(const float* ref_col,
                                                 int n_ref, float v_lane,
                                                 int n_grp,
                                                 unsigned int active_mask) {
    int lane = threadIdx.x & 31;
    double local_delta = 0.0;

    if (lane < n_grp) {
        float v = v_lane;
        float prev_lane_v =
            __shfl_sync(active_mask, v_lane, (lane > 0) ? lane - 1 : 0);
        float v_prev =
            (lane > 0) ? prev_lane_v : __int_as_float(0xff800000);  // -INF
        bool first_in_grp = (lane == 0) || (v != v_prev);

        int cnt_grp = 0;
#pragma unroll
        for (int lane_i = 0; lane_i < OVO_WARP_MAX; ++lane_i) {
            int src_lane = (lane_i < n_grp) ? lane_i : 0;
            float vi = __shfl_sync(active_mask, v_lane, src_lane);
            if (lane_i < n_grp && vi == v) ++cnt_grp;
        }

        if (first_in_grp) {
            int lo = 0, hi = n_ref;
            while (lo < hi) {
                int m = lo + ((hi - lo) >> 1);
                if (ref_col[m] < v)
                    lo = m + 1;
                else
                    hi = m;
            }
            int ref_lb = lo;
            hi = n_ref;
            while (lo < hi) {
                int m = lo + ((hi - lo) >> 1);
                if (ref_col[m] <= v)
                    lo = m + 1;
                else
                    hi = m;
            }
            int cnt_ref = lo - ref_lb;

            double combined = (double)(cnt_ref + cnt_grp);
            if (combined > 1.0) {
                local_delta += combined * combined * combined - combined;
            }
            if (cnt_ref > 1) {
                double cr = (double)cnt_ref;
                local_delta -= cr * cr * cr - cr;
            }
        }
    }

#pragma unroll
    for (int off = 16; off > 0; off >>= 1)
        local_delta += __shfl_down_sync(0xffffffff, local_delta, off);
    return local_delta;  // meaningful on lane 0.
}

// ============================================================================
// Tier 0 fused kernel: warp-per-(col, group) pair, 8 warps packed per block.
//
// Each warp independently:
//   1. Loads ≤ 32 group values into a single register (one per lane,
//      padded with +INF).
//   2. Bitonic-sorts via __shfl_xor_sync — no smem, no __syncthreads.
//   3. Binary-searches into sorted ref for each lane's value and
//      accumulates the rank-sum term.
//   4. Warp-shuffle reduces to lane 0 and writes rank_sums / tie_corr.
//
// 8 (col, group) pairs per block cuts block count 8× vs the block-per-pair
// Tier 1, and the lack of __syncthreads / smem sort lets each warp run
// independently at full throughput.
//
// Grid: (n_cols, ceil(n_groups / 8)), Block: 256.
// ============================================================================

__global__ void ovo_rank_warp_kernel(const float* __restrict__ ref_sorted,
                                     const float* __restrict__ grp_dense,
                                     const int* __restrict__ grp_offsets,
                                     const double* __restrict__ ref_tie_sums,
                                     double* __restrict__ rank_sums,
                                     double* __restrict__ tie_corr, int n_ref,
                                     int n_all_grp, int n_cols, int n_groups,
                                     bool compute_tie_corr) {
    constexpr int WARPS_PER_BLOCK = 8;
    int warp_id = threadIdx.x >> 5;
    int lane = threadIdx.x & 31;

    int col = blockIdx.x;
    int grp = blockIdx.y * WARPS_PER_BLOCK + warp_id;
    if (col >= n_cols || grp >= n_groups) return;

    int g_start = grp_offsets[grp];
    int g_end = grp_offsets[grp + 1];
    int n_grp = g_end - g_start;

    // This kernel only handles groups that fit in a single warp (one value
    // per lane).  Larger groups are delegated to Tier 1/3 in a co-launched
    // kernel; since each group owns its own row in rank_sums/tie_corr, the
    // two kernels interlace into the output without conflict.
    if (n_grp > OVO_WARP_MAX) return;

    if (n_grp == 0) {
        if (lane == 0) {
            rank_sums[grp * n_cols + col] = 0.0;
            if (compute_tie_corr) tie_corr[grp * n_cols + col] = 1.0;
        }
        return;
    }

    // One value per lane, pad with +INF so sort pushes them to the end.
    const float POS_INF = __int_as_float(0x7f800000);
    const float* grp_col = grp_dense + (long long)col * n_all_grp + g_start;
    float x = (lane < n_grp) ? grp_col[lane] : POS_INF;
    unsigned int active_mask = __ballot_sync(0xffffffff, lane < n_grp);

    // Warp-shuffle bitonic sort (ascending) — 32 elements in registers.
    for (int k = 1; k <= 16; k <<= 1) {
        for (int j = k; j > 0; j >>= 1) {
            float y = __shfl_xor_sync(0xffffffff, x, j);
            bool asc = (((lane & (k << 1)) == 0));
            bool take_min = (((lane & j) == 0) == asc);
            x = take_min ? fminf(x, y) : fmaxf(x, y);
        }
    }

    // After sort, x[lane] holds the lane-th smallest group value (lanes
    // ≥ n_grp hold +INF).  Binary-search each value into the sorted ref.
    const float* ref_col = ref_sorted + (long long)col * n_ref;
    double local_sum = 0.0;

    if (lane < n_grp) {
        float v = x;
        // Lower bound in ref.
        int lo = 0, hi = n_ref;
        while (lo < hi) {
            int m = lo + ((hi - lo) >> 1);
            if (ref_col[m] < v)
                lo = m + 1;
            else
                hi = m;
        }
        int n_lt_ref = lo;
        // Upper bound in ref.
        hi = n_ref;
        while (lo < hi) {
            int m = lo + ((hi - lo) >> 1);
            if (ref_col[m] <= v)
                lo = m + 1;
            else
                hi = m;
        }
        int n_eq_ref = lo - n_lt_ref;

        // In-group counts: in the sorted warp-register x, count lanes < this
        // one that hold strictly less, and lanes with equal value.
        int n_lt_grp = 0;
        int n_eq_grp_offset = 0;  // tied lanes strictly before this one
        int n_eq_grp_after = 1;   // count self
#pragma unroll
        for (int lane_i = 0; lane_i < OVO_WARP_MAX; ++lane_i) {
            if (lane_i >= n_grp) continue;
            float vi = __shfl_sync(active_mask, v, lane_i);
            if (lane_i < lane) {
                if (vi < v)
                    ++n_lt_grp;
                else if (vi == v)
                    ++n_eq_grp_offset;
            } else if (lane_i > lane) {
                if (vi == v) ++n_eq_grp_after;
            }
        }
        int n_eq_grp_total = n_eq_grp_offset + n_eq_grp_after;
        // Contribution: rank = n_lt_ref + n_lt_grp + (n_eq_ref +
        // n_eq_grp_total + 1) / 2, but we sum per lane so each tie lane
        // gets the same mid-rank.  This matches the Tier 1 accumulation.
        local_sum = (double)(n_lt_ref + n_lt_grp) +
                    ((double)(n_eq_ref + n_eq_grp_total) + 1.0) / 2.0;
    }

    // Warp reduce.
#pragma unroll
    for (int off = 16; off > 0; off >>= 1)
        local_sum += __shfl_down_sync(0xffffffff, local_sum, off);
    if (lane == 0) rank_sums[grp * n_cols + col] = local_sum;

    if (!compute_tie_corr) return;

    // Warp-scoped tie correction.
    double tie_sum;
    if (ref_tie_sums != nullptr) {
        tie_sum = ref_tie_sums[col] +
                  warp_tie_delta(ref_col, n_ref, x, n_grp, active_mask);
    } else {
        tie_sum = warp_tie_sum(ref_col, n_ref, x, n_grp, active_mask);
    }
    if (lane == 0) {
        int n = n_ref + n_grp;
        double dn = (double)n;
        double denom = dn * dn * dn - dn;
        tie_corr[grp * n_cols + col] =
            (denom > 0.0) ? (1.0 - tie_sum / denom) : 1.0;
    }
}
