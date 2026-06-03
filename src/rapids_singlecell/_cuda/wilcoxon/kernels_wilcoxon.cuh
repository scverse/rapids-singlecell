#pragma once

#include <cuda_runtime.h>

__device__ __forceinline__ double wilcoxon_block_sum(double val,
                                                     double* warp_buf) {
#pragma unroll
    for (int off = 16; off > 0; off >>= 1)
        val += __shfl_down_sync(0xffffffff, val, off);
    int lane = threadIdx.x & 31;
    int wid = threadIdx.x >> 5;
    if (lane == 0) warp_buf[wid] = val;
    __syncthreads();
    if (threadIdx.x < 32) {
        double v = (threadIdx.x < ((blockDim.x + 31) >> 5))
                       ? warp_buf[threadIdx.x]
                       : 0.0;
#pragma unroll
        for (int off = 16; off > 0; off >>= 1)
            v += __shfl_down_sync(0xffffffff, v, off);
        return v;
    }
    return 0.0;
}

/**
 * OVR dense rank-sum kernel for data sorted by column.
 *
 * sorted_vals and sorted_row_idx are F-order arrays from a segmented
 * SortPairs. One block owns one column, walks tie runs, and accumulates the
 * average ranks per group without materializing a full rank matrix.
 */
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

    double local_tie_sum = 0.0;
    int acc_stride = use_gmem ? n_cols : 1;

    int i = my_start;
    while (i < my_end) {
        double val = sv[i];

        int tie_local_end = i + 1;
        while (tie_local_end < my_end && sv[tie_local_end] == val) {
            ++tie_local_end;
        }

        int tie_global_start = i;
        if (i == my_start && i > 0 && sv[i - 1] == val) {
            int lo = 0;
            int hi = i;
            while (lo < hi) {
                int mid = lo + ((hi - lo) >> 1);
                if (sv[mid] < val)
                    lo = mid + 1;
                else
                    hi = mid;
            }
            tie_global_start = lo;
        }

        int tie_global_end = tie_local_end;
        if (tie_local_end == my_end && tie_local_end < n_rows &&
            sv[tie_local_end] == val) {
            int lo = tie_local_end;
            int hi = n_rows - 1;
            while (lo < hi) {
                int mid = hi - ((hi - lo) >> 1);
                if (sv[mid] > val)
                    hi = mid - 1;
                else
                    lo = mid;
            }
            tie_global_end = lo + 1;
        }

        int total_tie = tie_global_end - tie_global_start;
        double avg_rank = (double)(tie_global_start + tie_global_end + 1) / 2.0;

        for (int j = i; j < tie_local_end; ++j) {
            int grp = group_codes[si[j]];
            if (grp < n_groups) {
                atomicAdd(&grp_sums[grp * acc_stride], avg_rank);
            }
        }

        if (compute_tie_corr && tie_global_start >= my_start && total_tie > 1) {
            double t = (double)total_tie;
            local_tie_sum += t * t * t - t;
        }

        i = tie_local_end;
    }

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
