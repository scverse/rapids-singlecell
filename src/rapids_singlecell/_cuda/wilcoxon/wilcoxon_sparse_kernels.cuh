#pragma once

#include <cuda_runtime.h>

#include "wilcoxon_block_reduce.cuh"
#include "wilcoxon_ovr_tie_walk.cuh"

/**
 * Sparse-aware OVR rank-sum kernel for nonnegative sorted stored values.
 *
 * Sparse rank_genes_groups now rejects explicit negative sparse values before
 * reaching CUDA, so after CUB sort each column segment is:
 *   [stored_zeros..., positives...]
 *
 * Implicit zeros (n_rows - nnz_stored) join stored zeros as the first tie
 * block.  The kernel ranks only stored positive values and adds each group's
 * zero contribution analytically.
 *
 * Full sorted array (conceptual):
 *   [ALL_zeros (stored+implicit)..., positives...]
 *
 * Rank offsets:
 *   positive at stored pos i : full pos = i + n_implicit_zero
 *   zeros                    : avg rank = (total_zero + 1) / 2
 *
 * Shared-memory layout (doubles):
 *   grp_sums[n_groups]      rank-sum accumulators
 *   grp_nz_count[n_groups]  nonzero-per-group counters
 *   warp_buf[32]            tie-correction reduction scratch
 *
 * n_rows is the ranking population, including rows whose group code is the
 * n_groups sentinel. Sentinel rows contribute to the "rest" distribution and
 * tie-correction denominator but do not receive rank-sum accumulation.
 *
 * Grid: (sb_cols,)   Block: (tpb,)
 */
// HEADLINE sparse-OVR optimization (OVR-only). Ranks ONLY stored positive
// values; all zeros (stored + implicit n_rows-nnz) are treated as one leading
// tie block ranked analytically at (total_zero+1)/2, and each group's zero
// contribution is applied in closed form. Cost is O(nnz log nnz) per column,
// not O(n_rows log n_rows). The `use_gmem` flag selects shared- vs
// global-memory accumulators (see sparse_ovr_smem_config) -- CRITICAL: the
// use_gmem path is REQUIRED for large n_groups (Perturb-seq) and must not be
// removed. Validity relies on the upstream rejection of explicit negative
// sparse values, which guarantees zeros form the first tie block.
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

    // pos_start = first index where sv[i] > 0 (stored zeros precede positives).
    __shared__ int sh_pos_start;
    if (threadIdx.x == 0) {
        int lo = 0, hi = nnz_stored;
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

    int pos_start = sh_pos_start;
    int n_stored_zero = pos_start;
    int n_implicit_zero = n_rows - nnz_stored;
    int total_zero = n_implicit_zero + n_stored_zero;
    double zero_avg_rank = (total_zero > 0) ? (total_zero + 1.0) / 2.0 : 0.0;

    // Rank offset for positive stored values:
    //   full_pos(i) = i + n_implicit_zero  for i >= pos_start
    // So avg_rank for tie group [a,b) of positives:
    //   = n_implicit_zero + (a + b + 1) / 2
    int offset_pos = n_implicit_zero;

    // Count stored positives per group.
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

    // Walk stored positives and compute tie-averaged ranks.
    int n_pos = nnz_stored - pos_start;
    int chunk = (n_pos + blockDim.x - 1) / blockDim.x;
    int my_start = pos_start + threadIdx.x * chunk;
    int my_end = my_start + chunk;
    if (my_end > nnz_stored) my_end = nnz_stored;

    double local_tie_sum = ovr_walk_tie_runs<IndexT>(
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
        if (threadIdx.x == 0) {
            double n = (double)n_rows;
            double denom = n * n * n - n;
            tie_corr[col] = (denom > 0.0) ? (1.0 - v / denom) : 1.0;
        }
    }
}

// Shared sparse-OVR rank launch, used by all four sparse OVR impls (they differ
// only in how they produce the sorted nonzeros and how they scatter results).
// Optionally zeroes the global-memory accumulators, then launches the
// analytic-zero rank kernel. use_gmem is the CRITICAL large-n_groups /
// perturbation fallback (see sparse_ovr_smem_config) — DO NOT drop the gmem
// branch. ValT is the sorted-row-index type (int everywhere today).
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

// CRITICAL — DO NOT REMOVE the gmem branch (large n_groups / perturbation DE).
//
// Decide smem-vs-gmem for the sparse-OVR stats cast-and-accumulate kernel
// (sums / sq-sums / nnz). Needs n_arrays*n_groups doubles in smem; when that
// exceeds the per-block limit, use_gmem=true selects
// ovr_cast_and_accumulate_sparse_global_kernel, which accumulates directly in
// global memory. Same large-n_groups workloads that drive
// sparse_ovr_smem_config to gmem also drive this one; both fallbacks are
// load-bearing, not dead.
static size_t cast_accumulate_smem_config(int n_groups, bool compute_nnz,
                                          bool& use_gmem) {
    int n_arrays = 1 + (compute_nnz ? 1 : 0);
    size_t need = (size_t)n_arrays * n_groups * sizeof(double);
    if (need <= wilcoxon_max_smem_per_block()) {
        use_gmem = false;
        return need;
    }
    use_gmem = true;
    return 0;
}

// Shared cast+accumulate loop for the two sparse-OVR stats kernels. Casts each
// stored value to f32 (data_f32_out) and atomically accumulates per-group sums
// (and nonzero counts) into sums/nnz, strided by acc_stride (1 for a per-block
// smem buffer, sb_cols for the global row-major layout).
template <typename InT, typename IndexT>
__device__ __forceinline__ void accumulate_group_stats(
    const InT* data_in, float* data_f32_out, const IndexT* indices,
    int seg_start, int seg_end, const int* group_codes, double* sums,
    double* nnz, int acc_stride, int n_groups, bool compute_nnz) {
    for (int i = seg_start + threadIdx.x; i < seg_end; i += blockDim.x) {
        InT v_in = data_in[i];
        double v = (double)v_in;
        data_f32_out[i] = (float)v_in;
        int row = (int)indices[i];
        int g = group_codes[row];
        if (g >= 0 && g < n_groups) {
            atomicAdd(&sums[(size_t)g * acc_stride], v);
            if (compute_nnz && v != 0.0)
                atomicAdd(&nnz[(size_t)g * acc_stride], 1.0);
        }
    }
}

/**
 * Pre-sort cast-and-accumulate kernel for sparse OVR host streaming.
 *
 * Sub-batch CSC data is laid out contiguously: values for column c live
 * at positions [col_seg_offsets[c], col_seg_offsets[c+1]).  For each
 * stored value, read the native-dtype InT, write a float32 copy for the
 * CUB sort, and accumulate per-group sum/sum-sq/nnz in float64.  Implicit
 * zeros contribute nothing to any of these stats.
 *
 * Block-per-column layout (grid: (sb_cols,), block: (tpb,)).
 * Shared memory: 3 * n_groups doubles.
 */
template <typename InT, typename IndexT = int>
__global__ void ovr_cast_and_accumulate_sparse_kernel(
    const InT* __restrict__ data_in, float* __restrict__ data_f32_out,
    const IndexT* __restrict__ indices, const int* __restrict__ col_seg_offsets,
    const int* __restrict__ group_codes, double* __restrict__ group_sums,
    double* __restrict__ group_nnz, int sb_cols, int n_groups,
    bool compute_nnz = true) {
    int col = blockIdx.x;
    if (col >= sb_cols) return;

    int seg_start = col_seg_offsets[col];
    int seg_end = col_seg_offsets[col + 1];

    // Packed layout matching cast_accumulate_smem_config, which sizes the
    // dynamic smem as (1 + compute_nnz) * n_groups doubles.
    extern __shared__ double smem[];
    double* s_sum = smem;
    double* s_nnz = smem + n_groups;

    for (int g = threadIdx.x; g < n_groups; g += blockDim.x) {
        s_sum[g] = 0.0;
        if (compute_nnz) s_nnz[g] = 0.0;
    }
    __syncthreads();

    accumulate_group_stats<InT, IndexT>(
        data_in, data_f32_out, indices, seg_start, seg_end, group_codes, s_sum,
        s_nnz, /*acc_stride=*/1, n_groups, compute_nnz);
    __syncthreads();

    for (int g = threadIdx.x; g < n_groups; g += blockDim.x) {
        group_sums[(size_t)g * sb_cols + col] = s_sum[g];
        if (compute_nnz) {
            group_nnz[(size_t)g * sb_cols + col] = s_nnz[g];
        }
    }
}

// CRITICAL — DO NOT REMOVE. Global-memory variant of the stats accumulator,
// selected by cast_accumulate_smem_config when n_groups is too large for the
// smem version. Required for Perturb-seq-scale n_groups; the smem kernel cannot
// launch when its (n_arrays*n_groups) double buffer exceeds the per-block
// limit.
template <typename InT, typename IndexT = int>
__global__ void ovr_cast_and_accumulate_sparse_global_kernel(
    const InT* __restrict__ data_in, float* __restrict__ data_f32_out,
    const IndexT* __restrict__ indices, const int* __restrict__ col_seg_offsets,
    const int* __restrict__ group_codes, double* __restrict__ group_sums,
    double* __restrict__ group_nnz, int sb_cols, int n_groups,
    bool compute_nnz = true) {
    int col = blockIdx.x;
    if (col >= sb_cols) return;

    int seg_start = col_seg_offsets[col];
    int seg_end = col_seg_offsets[col + 1];

    accumulate_group_stats<InT, IndexT>(
        data_in, data_f32_out, indices, seg_start, seg_end, group_codes,
        group_sums + col, group_nnz + col,
        /*acc_stride=*/sb_cols, n_groups, compute_nnz);
}

template <typename InT, typename IndexT = int>
static void launch_ovr_cast_and_accumulate_sparse(
    const InT* d_data_orig, float* d_data_f32, const IndexT* d_indices,
    const int* d_col_offsets, const int* d_group_codes, double* d_group_sums,
    double* d_group_nnz, int sb_cols, int n_groups, bool compute_nnz, int tpb,
    size_t smem_cast, bool use_gmem, cudaStream_t stream) {
    if (use_gmem) {
        size_t stats_items = (size_t)n_groups * sb_cols;
        cudaMemsetAsync(d_group_sums, 0, stats_items * sizeof(double), stream);
        if (compute_nnz) {
            cudaMemsetAsync(d_group_nnz, 0, stats_items * sizeof(double),
                            stream);
        }
        ovr_cast_and_accumulate_sparse_global_kernel<InT, IndexT>
            <<<sb_cols, tpb, 0, stream>>>(d_data_orig, d_data_f32, d_indices,
                                          d_col_offsets, d_group_codes,
                                          d_group_sums, d_group_nnz, sb_cols,
                                          n_groups, compute_nnz);
        CUDA_CHECK_LAST_ERROR(ovr_cast_and_accumulate_sparse_global_kernel);
    } else {
        ovr_cast_and_accumulate_sparse_kernel<InT, IndexT>
            <<<sb_cols, tpb, smem_cast, stream>>>(
                d_data_orig, d_data_f32, d_indices, d_col_offsets,
                d_group_codes, d_group_sums, d_group_nnz, sb_cols, n_groups,
                compute_nnz);
        CUDA_CHECK_LAST_ERROR(ovr_cast_and_accumulate_sparse_kernel);
    }
}
