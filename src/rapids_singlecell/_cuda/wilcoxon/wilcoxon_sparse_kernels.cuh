#pragma once

#include <cuda_runtime.h>

#include "wilcoxon_block_reduce.cuh"
#include "wilcoxon_ovr_tie_walk.cuh"

// Sparse OVR rank with implicit zeros inserted analytically between sorted
// negative and positive stored values. CRITICAL: keep the gmem fallback for
// large n_groups.
template <typename IndexT = int>
__global__ void rank_sums_sparse_ovr_kernel(
    const float* __restrict__ sorted_vals,
    const IndexT* __restrict__ sorted_row_idx,
    const int* __restrict__ col_seg_offsets,
    const int* __restrict__ group_codes, const double* __restrict__ group_sizes,
    double* __restrict__ rank_sums, double* __restrict__ tie_corr,
    double* __restrict__ nz_count_scratch, int n_rows, int sb_cols,
    int n_groups, bool compute_tie_corr, bool use_gmem) {
    int col = blockIdx.x;
    if (col >= sb_cols) return;

    int seg_start = col_seg_offsets[col];
    int seg_end = col_seg_offsets[col + 1];
    int nnz_stored = seg_end - seg_start;

    const float* sv = sorted_vals + seg_start;
    const IndexT* si = sorted_row_idx + seg_start;

    extern __shared__ double smem[];
    double* grp_sums;
    double* grp_nz_count;
    // Accumulator stride: 1 for shared mem (dense per-block), sb_cols for
    // gmem (row-major layout (n_groups, sb_cols) shared across blocks).
    int acc_stride;

    if (use_gmem) {
        // rank_sums doubles as accumulator (pre-zeroed by caller).
        grp_sums = rank_sums + (size_t)col;
        grp_nz_count = nz_count_scratch + (size_t)col;
        acc_stride = sb_cols;
    } else {
        grp_sums = smem;
        grp_nz_count = smem + n_groups;
        acc_stride = 1;
        for (int g = threadIdx.x; g < n_groups; g += blockDim.x) {
            grp_sums[g] = 0.0;
            grp_nz_count[g] = 0.0;
        }
        __syncthreads();
    }

    // Split sorted stored values around zero. Implicit zeros join the explicit
    // zero run between [0, neg_end) and [pos_start, nnz_stored).
    __shared__ int sh_neg_end;
    __shared__ int sh_pos_start;
    if (threadIdx.x == 0) {
        int lo = 0, hi = nnz_stored;
        while (lo < hi) {
            int mid = lo + ((hi - lo) >> 1);
            if (sv[mid] < 0.0f)
                lo = mid + 1;
            else
                hi = mid;
        }
        sh_neg_end = lo;

        lo = 0;
        hi = nnz_stored;
        while (lo < hi) {
            int mid = lo + ((hi - lo) >> 1);
            if (sv[mid] <= 0.0f)
                lo = mid + 1;
            else
                hi = mid;
        }
        sh_pos_start = lo;
    }
    __syncthreads();

    int neg_end = sh_neg_end;
    int pos_start = sh_pos_start;
    int n_stored_zero = pos_start - neg_end;
    int n_implicit_zero = n_rows - nnz_stored;
    int total_zero = n_implicit_zero + n_stored_zero;
    double zero_avg_rank =
        (total_zero > 0) ? (double)neg_end + (total_zero + 1.0) / 2.0 : 0.0;

    // Positive ranks use the stored index plus the inserted implicit zeros.
    int offset_pos = n_implicit_zero;

    // Count stored negatives and positives per group. Explicit and implicit
    // zeros both receive the analytic zero contribution below.
    for (int i = threadIdx.x; i < neg_end; i += blockDim.x) {
        int grp = group_codes[si[i]];
        if (grp >= 0 && grp < n_groups) {
            atomicAdd(&grp_nz_count[(size_t)grp * acc_stride], 1.0);
        }
    }
    for (int i = pos_start + threadIdx.x; i < nnz_stored; i += blockDim.x) {
        int grp = group_codes[si[i]];
        if (grp >= 0 && grp < n_groups) {
            atomicAdd(&grp_nz_count[(size_t)grp * acc_stride], 1.0);
        }
    }
    __syncthreads();

    // Analytic zero contribution: each group's zeros all get zero_avg_rank.
    for (int g = threadIdx.x; g < n_groups; g += blockDim.x) {
        double n_zero_in_g =
            group_sizes[g] - grp_nz_count[(size_t)g * acc_stride];
        grp_sums[(size_t)g * acc_stride] = n_zero_in_g * zero_avg_rank;
    }
    __syncthreads();

    // Walk stored negatives and positives separately so the implicit-zero tie
    // occupies its exact position between them.
    int n_neg = neg_end;
    int neg_chunk = (n_neg + blockDim.x - 1) / blockDim.x;
    int my_neg_start = threadIdx.x * neg_chunk;
    int my_neg_end = my_neg_start + neg_chunk;
    if (my_neg_end > neg_end) my_neg_end = neg_end;

    double local_tie_sum = ovr_walk_tie_runs<IndexT>(
        sv, si, group_codes, grp_sums, acc_stride, n_groups, my_neg_start,
        my_neg_end, /*seg_floor=*/0, /*seg_ceil=*/neg_end,
        /*rank_offset=*/0.0, compute_tie_corr);

    int n_pos = nnz_stored - pos_start;
    int chunk = (n_pos + blockDim.x - 1) / blockDim.x;
    int my_start = pos_start + threadIdx.x * chunk;
    int my_end = my_start + chunk;
    if (my_end > nnz_stored) my_end = nnz_stored;

    local_tie_sum += ovr_walk_tie_runs<IndexT>(
        sv, si, group_codes, grp_sums, acc_stride, n_groups, my_start, my_end,
        /*seg_floor=*/pos_start, /*seg_ceil=*/nnz_stored,
        /*rank_offset=*/(double)offset_pos, compute_tie_corr);

    __syncthreads();

    // Write rank sums to global output (smem path only — gmem path is direct)
    if (!use_gmem) {
        for (int g = threadIdx.x; g < n_groups; g += blockDim.x) {
            rank_sums[(size_t)g * sb_cols + col] = grp_sums[g];
        }
    }

    // Tie correction: warp + block reduction
    if (compute_tie_corr) {
        // Single zero tie block contributes once.
        if (threadIdx.x == 0 && total_zero > 1) {
            double tz = (double)total_zero;
            local_tie_sum += tz * tz * tz - tz;
        }

        // smem path: warp buf after both accumulator arrays (2 * n_groups).
        // gmem path: accumulators are in gmem, warp buf starts at smem[0].
        int warp_buf_off = use_gmem ? 0 : 2 * n_groups;
        double* warp_buf = smem + warp_buf_off;

        double v = wilcoxon_block_sum(local_tie_sum, warp_buf);
        if (threadIdx.x == 0) tie_corr[col] = finalize_tie_corr(n_rows, v);
    }
}

// Shared sparse-OVR rank launch for all sparse OVR implementations.
// CRITICAL: keep the gmem fallback for large-n_groups perturbation DE.
template <typename ValT = int>
static inline void launch_ovr_sparse_rank(
    const float* sorted_vals, const ValT* sorted_row_idx,
    const int* col_seg_offsets, const int* group_codes,
    const double* group_sizes, double* rank_sums, double* tie_corr,
    double* nz_count_scratch, int n_rows, int sb_cols, int n_groups, int tpb,
    size_t smem_bytes, bool compute_tie_corr, bool use_gmem,
    cudaStream_t stream) {
    if (use_gmem) {
        cudaMemsetAsync(rank_sums, 0,
                        (size_t)n_groups * sb_cols * sizeof(double), stream);
        cudaMemsetAsync(nz_count_scratch, 0,
                        (size_t)n_groups * sb_cols * sizeof(double), stream);
    }
    rank_sums_sparse_ovr_kernel<ValT><<<sb_cols, tpb, smem_bytes, stream>>>(
        sorted_vals, sorted_row_idx, col_seg_offsets, group_codes, group_sizes,
        rank_sums, tie_corr, nz_count_scratch, n_rows, sb_cols, n_groups,
        compute_tie_corr, use_gmem);
    CUDA_CHECK_LAST_ERROR(rank_sums_sparse_ovr_kernel);
}

// CRITICAL: sparse stats gmem fallback is load-bearing for large n_groups.
// It selects the global accumulator when smem would exceed the per-block limit.
static size_t cast_accumulate_smem_config(int n_groups, bool compute_nnz,
                                          bool compute_totals, bool& use_gmem) {
    int n_arrays = 1 + (compute_nnz ? 1 : 0);
    size_t need = (size_t)n_arrays * n_groups * sizeof(double);
    if (compute_totals) need += WARP_REDUCE_BUF * sizeof(double);
    if (need <= wilcoxon_max_smem_per_block()) {
        use_gmem = false;
        return need;
    }
    use_gmem = true;
    return compute_totals ? WARP_REDUCE_BUF * sizeof(double) : 0;
}

// Shared cast+accumulate loop for sparse-OVR stats kernels.
// Casts to f32 for sort and atomically accumulates f64 sums/nnz.
template <typename InT, typename IndexT>
__device__ __forceinline__ void accumulate_group_stats(
    const InT* data_in, float* data_f32_out, const IndexT* indices,
    int seg_start, int seg_end, const int* group_codes, double* sums,
    double* nnz, int acc_stride, int n_groups, bool compute_nnz,
    bool compute_totals, double& local_total_sum, double& local_total_nnz) {
    for (int i = seg_start + threadIdx.x; i < seg_end; i += blockDim.x) {
        InT v_in = data_in[i];
        double v = (double)v_in;
        data_f32_out[i] = (float)v_in;
        if (compute_totals) {
            local_total_sum += v;
            if (compute_nnz && v != 0.0) local_total_nnz += 1.0;
        }
        int row = (int)indices[i];
        int g = group_codes[row];
        if (g >= 0 && g < n_groups) {
            atomicAdd(&sums[(size_t)g * acc_stride], v);
            if (compute_nnz && v != 0.0)
                atomicAdd(&nnz[(size_t)g * acc_stride], 1.0);
        }
    }
}

/** Pre-sort cast-and-accumulate kernel for sparse OVR streaming.
 *  Writes f32 sort keys and accumulates explicit-value sums/nnz in f64. */
template <typename InT, typename IndexT = int>
__global__ void ovr_cast_and_accumulate_sparse_kernel(
    const InT* __restrict__ data_in, float* __restrict__ data_f32_out,
    const IndexT* __restrict__ indices, const int* __restrict__ col_seg_offsets,
    const int* __restrict__ group_codes, double* __restrict__ group_sums,
    double* __restrict__ group_nnz, double* __restrict__ total_sums,
    double* __restrict__ total_nnz, int sb_cols, int n_groups,
    bool compute_nnz = true, bool compute_totals = false) {
    int col = blockIdx.x;
    if (col >= sb_cols) return;

    int seg_start = col_seg_offsets[col];
    int seg_end = col_seg_offsets[col + 1];

    // Packed layout matching cast_accumulate_smem_config ((1+compute_nnz)*
    // n_groups doubles).
    extern __shared__ double smem[];
    double* s_sum = smem;
    double* s_nnz = smem + n_groups;
    double* warp_buf = smem + (size_t)(1 + (compute_nnz ? 1 : 0)) * n_groups;

    for (int g = threadIdx.x; g < n_groups; g += blockDim.x) {
        s_sum[g] = 0.0;
        if (compute_nnz) s_nnz[g] = 0.0;
    }
    __syncthreads();

    double local_total_sum = 0.0;
    double local_total_nnz = 0.0;
    accumulate_group_stats<InT, IndexT>(
        data_in, data_f32_out, indices, seg_start, seg_end, group_codes, s_sum,
        s_nnz, /*acc_stride=*/1, n_groups, compute_nnz, compute_totals,
        local_total_sum, local_total_nnz);
    __syncthreads();

    if (compute_totals) {
        double total = wilcoxon_block_sum(local_total_sum, warp_buf);
        if (threadIdx.x == 0) total_sums[col] = total;
        __syncthreads();
        if (compute_nnz) {
            double nnz_total = wilcoxon_block_sum(local_total_nnz, warp_buf);
            if (threadIdx.x == 0) total_nnz[col] = nnz_total;
            __syncthreads();
        }
    }

    for (int g = threadIdx.x; g < n_groups; g += blockDim.x) {
        group_sums[(size_t)g * sb_cols + col] = s_sum[g];
        if (compute_nnz) {
            group_nnz[(size_t)g * sb_cols + col] = s_nnz[g];
        }
    }
}

// CRITICAL: gmem stats accumulator for n_groups too large for smem.
// Required for Perturb-seq-scale group counts.
template <typename InT, typename IndexT = int>
__global__ void ovr_cast_and_accumulate_sparse_global_kernel(
    const InT* __restrict__ data_in, float* __restrict__ data_f32_out,
    const IndexT* __restrict__ indices, const int* __restrict__ col_seg_offsets,
    const int* __restrict__ group_codes, double* __restrict__ group_sums,
    double* __restrict__ group_nnz, double* __restrict__ total_sums,
    double* __restrict__ total_nnz, int sb_cols, int n_groups,
    bool compute_nnz = true, bool compute_totals = false) {
    int col = blockIdx.x;
    if (col >= sb_cols) return;

    int seg_start = col_seg_offsets[col];
    int seg_end = col_seg_offsets[col + 1];

    extern __shared__ double warp_buf[];
    double local_total_sum = 0.0;
    double local_total_nnz = 0.0;
    accumulate_group_stats<InT, IndexT>(
        data_in, data_f32_out, indices, seg_start, seg_end, group_codes,
        group_sums + col, group_nnz + col,
        /*acc_stride=*/sb_cols, n_groups, compute_nnz, compute_totals,
        local_total_sum, local_total_nnz);
    if (compute_totals) {
        double total = wilcoxon_block_sum(local_total_sum, warp_buf);
        if (threadIdx.x == 0) total_sums[col] = total;
        __syncthreads();
        if (compute_nnz) {
            double nnz_total = wilcoxon_block_sum(local_total_nnz, warp_buf);
            if (threadIdx.x == 0) total_nnz[col] = nnz_total;
        }
    }
}

template <typename InT, typename IndexT = int>
static void launch_ovr_cast_and_accumulate_sparse(
    const InT* d_data_orig, float* d_data_f32, const IndexT* d_indices,
    const int* d_col_offsets, const int* d_group_codes, double* d_group_sums,
    double* d_group_nnz, double* d_total_sums, double* d_total_nnz, int sb_cols,
    int n_groups, bool compute_nnz, bool compute_totals, int tpb,
    size_t smem_cast, bool use_gmem, cudaStream_t stream) {
    if (use_gmem) {
        size_t stats_items = (size_t)n_groups * sb_cols;
        cudaMemsetAsync(d_group_sums, 0, stats_items * sizeof(double), stream);
        if (compute_nnz) {
            cudaMemsetAsync(d_group_nnz, 0, stats_items * sizeof(double),
                            stream);
        }
        ovr_cast_and_accumulate_sparse_global_kernel<InT, IndexT>
            <<<sb_cols, tpb, smem_cast, stream>>>(
                d_data_orig, d_data_f32, d_indices, d_col_offsets,
                d_group_codes, d_group_sums, d_group_nnz, d_total_sums,
                d_total_nnz, sb_cols, n_groups, compute_nnz, compute_totals);
        CUDA_CHECK_LAST_ERROR(ovr_cast_and_accumulate_sparse_global_kernel);
    } else {
        ovr_cast_and_accumulate_sparse_kernel<InT, IndexT>
            <<<sb_cols, tpb, smem_cast, stream>>>(
                d_data_orig, d_data_f32, d_indices, d_col_offsets,
                d_group_codes, d_group_sums, d_group_nnz, d_total_sums,
                d_total_nnz, sb_cols, n_groups, compute_nnz, compute_totals);
        CUDA_CHECK_LAST_ERROR(ovr_cast_and_accumulate_sparse_kernel);
    }
}
