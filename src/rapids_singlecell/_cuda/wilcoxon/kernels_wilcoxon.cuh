#pragma once

#include <cuda_runtime.h>

#include "wilcoxon_block_reduce.cuh"
#include "wilcoxon_ovr_tie_walk.cuh"

// Dense OVR rank kernel. sorted_vals/sorted_row_idx are F-order arrays from a
// segmented SortPairs. One block per column; walks sorted tie runs and
// accumulates average ranks per group without materializing a rank matrix.
// The `use_gmem` flag (set by ovr_smem_config) selects shared- vs
// global-memory group accumulators -- CRITICAL: the use_gmem path is REQUIRED
// when n_groups is large (does NOT fit in smem) and must not be removed.
__global__ void rank_sums_from_sorted_kernel(
    const float* __restrict__ sorted_vals,
    const int* __restrict__ sorted_row_idx, const int* __restrict__ group_codes,
    double* __restrict__ rank_sums, double* __restrict__ tie_corr, int n_rows,
    int n_cols, int n_groups, bool compute_tie_corr, bool use_gmem) {
    int col = blockIdx.x;
    if (col >= n_cols) return;

    extern __shared__ double smem[];

    double* grp_sums;
    if (use_gmem) {
        grp_sums = rank_sums + (size_t)col;
    } else {
        grp_sums = smem;
        for (int g = threadIdx.x; g < n_groups; g += blockDim.x) {
            grp_sums[g] = 0.0;
        }
        __syncthreads();
    }

    const float* sv = sorted_vals + (size_t)col * n_rows;
    const int* si = sorted_row_idx + (size_t)col * n_rows;

    int chunk = (n_rows + blockDim.x - 1) / blockDim.x;
    int my_start = threadIdx.x * chunk;
    int my_end = my_start + chunk;
    if (my_end > n_rows) my_end = n_rows;

    int acc_stride = use_gmem ? n_cols : 1;
    double local_tie_sum = ovr_walk_tie_runs<int>(
        sv, si, group_codes, grp_sums, acc_stride, n_groups, my_start, my_end,
        /*seg_floor=*/0, /*seg_ceil=*/n_rows, /*rank_offset=*/0.0,
        compute_tie_corr);

    __syncthreads();

    if (!use_gmem) {
        for (int g = threadIdx.x; g < n_groups; g += blockDim.x) {
            rank_sums[(size_t)g * n_cols + col] = grp_sums[g];
        }
    }

    if (compute_tie_corr) {
        int warp_buf_off = use_gmem ? 0 : n_groups;
        double* warp_buf = smem + warp_buf_off;
        double tie_sum = wilcoxon_block_sum(local_tie_sum, warp_buf);
        if (threadIdx.x == 0) {
            double n = (double)n_rows;
            double denom = n * n * n - n;
            tie_corr[col] = (denom > 0.0) ? (1.0 - tie_sum / denom) : 1.0;
        }
    }
}
