#pragma once

#include <cuda_runtime.h>

// Walk this thread's chunk [my_start, my_end) of a sorted column, accumulating
// tie-averaged ranks into grp_sums (atomic, strided by acc_stride). Ties that
// straddle a chunk boundary are expanded to their global extent within
// [seg_floor, seg_ceil) by binary search. `rank_offset` shifts every rank (the
// sparse path uses it to account for implicit leading zeros). Returns this
// thread's tie-correction sum (sum of t^3 - t over tie blocks it owns).
template <typename IndexT>
__device__ __forceinline__ double ovr_walk_tie_runs(
    const float* sv, const IndexT* si, const int* group_codes, double* grp_sums,
    int acc_stride, int n_groups, int my_start, int my_end, int seg_floor,
    int seg_ceil, double rank_offset, bool compute_tie_corr) {
    double local_tie_sum = 0.0;
    int i = my_start;
    while (i < my_end) {
        float val = sv[i];

        int tie_local_end = i + 1;
        while (tie_local_end < my_end && sv[tie_local_end] == val)
            ++tie_local_end;

        int tie_global_start = i;
        if (i == my_start && i > seg_floor && sv[i - 1] == val) {
            // tie spans into a prior chunk: find global tie start.
            int lo = seg_floor, hi = i;
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
        if (tie_local_end == my_end && tie_local_end < seg_ceil &&
            sv[tie_local_end] == val) {
            int lo = tie_local_end, hi = seg_ceil - 1;
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
        double avg_rank =
            rank_offset +
            ((double)tie_global_start + (double)tie_global_end + 1.0) / 2.0;

        for (int j = i; j < tie_local_end; ++j) {
            int grp = group_codes[si[j]];
            if (grp >= 0 && grp < n_groups) {
                atomicAdd(&grp_sums[(size_t)grp * acc_stride], avg_rank);
            }
        }

        if (compute_tie_corr && tie_global_start >= my_start && total_tie > 1) {
            double t = (double)total_tie;
            local_tie_sum += t * t * t - t;
        }

        i = tie_local_end;
    }
    return local_tie_sum;
}
