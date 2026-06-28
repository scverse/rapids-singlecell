#pragma once

#include <climits>

#include <cub/device/device_segmented_radix_sort.cuh>

#include "../sparse_extract/sparse_extract.cuh"

/** Build CUB segmented-sort ranges for HUGE-band groups.
 *  Ranges point into the original dense group layout. */
__global__ void build_huge_seg_offsets_kernel(
    const int* __restrict__ grp_offsets, const int* __restrict__ group_ids,
    int* __restrict__ begins, int* __restrict__ ends, int n_all_grp,
    int n_sort_groups, int sb_cols) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = sb_cols * n_sort_groups;
    if (idx >= total) return;

    int c = idx / n_sort_groups;
    int local = idx % n_sort_groups;
    int g = group_ids[local];
    int base = c * n_all_grp;
    begins[idx] = base + grp_offsets[g];
    ends[idx] = base + grp_offsets[g + 1];
}

template <typename T>
__global__ void dense_ovo_group_stats_kernel(
    const T* __restrict__ ref_dense, const T* __restrict__ grp_dense,
    const int* __restrict__ grp_codes, double* __restrict__ group_sums,
    double* __restrict__ group_sum_sq, double* __restrict__ group_nnz,
    int n_ref, int n_all_grp, int sb_cols, int n_groups, bool compute_nnz) {
    int col = blockIdx.x;
    if (col >= sb_cols) return;

    int ref_slot = n_groups;
    const T* ref_col = ref_dense + (size_t)col * n_ref;
    const T* grp_col = grp_dense + (size_t)col * n_all_grp;

    for (int row = threadIdx.x; row < n_ref; row += blockDim.x) {
        double v = (double)ref_col[row];
        atomicAdd(&group_sums[(size_t)ref_slot * sb_cols + col], v);
        atomicAdd(&group_sum_sq[(size_t)ref_slot * sb_cols + col], v * v);
        if (compute_nnz && v != 0.0) {
            atomicAdd(&group_nnz[(size_t)ref_slot * sb_cols + col], 1.0);
        }
    }

    for (int row = threadIdx.x; row < n_all_grp; row += blockDim.x) {
        int g = grp_codes[row];
        if (g < 0 || g >= n_groups) continue;
        double v = (double)grp_col[row];
        atomicAdd(&group_sums[(size_t)g * sb_cols + col], v);
        atomicAdd(&group_sum_sq[(size_t)g * sb_cols + col], v * v);
        if (compute_nnz && v != 0.0) {
            atomicAdd(&group_nnz[(size_t)g * sb_cols + col], 1.0);
        }
    }
}

/** Sizing knobs for LARGE/HUGE dispatch.
 *  LARGE uses fused smem sort; HUGE uses CUB sort plus pre-sorted rank. */
struct OvoTierPlan {
    int max_grp_size = 0;
    bool run_medium = false;    // MEDIUM band: any group ≤ OVO_MEDIUM_MAX
    bool run_large = false;     // LARGE band: (OVO_MEDIUM_MAX, OVO_LARGE_MAX]
    bool run_huge = false;      // HUGE band: > OVO_LARGE_MAX
    bool above_medium = false;  // at least one group exceeds OVO_MEDIUM_MAX
    int huge_skip_le = OVO_MEDIUM_MAX;  // HUGE/CUB ranks groups > this
    int large_padded = 0;
    int large_tpb = 0;
    size_t large_smem = 0;
};

// MEDIUM, LARGE, and HUGE bands are launched independently.
static OvoTierPlan make_ovo_tier_plan(const int* h_grp_offsets, int n_groups) {
    OvoTierPlan c;
    for (int g = 0; g < n_groups; g++) {
        int sz = h_grp_offsets[g + 1] - h_grp_offsets[g];
        if (sz > c.max_grp_size) c.max_grp_size = sz;
        if (sz <= OVO_MEDIUM_MAX)
            c.run_medium = true;
        else if (sz <= OVO_LARGE_MAX)
            c.run_large = true;
        else
            c.run_huge = true;
    }
    c.above_medium = c.run_large || c.run_huge;
    c.huge_skip_le = OVO_LARGE_MAX;

    // Size smem for the largest LARGE-band group.
    if (c.run_large) {
        int large_max = std::min(c.max_grp_size, OVO_LARGE_MAX);
        c.large_padded = 1;
        while (c.large_padded < large_max) c.large_padded <<= 1;
        c.large_tpb = std::min(c.large_padded, MAX_THREADS_PER_BLOCK);
        c.large_smem = (size_t)c.large_padded * sizeof(float);
        // Fall back to CUB if the smem kernel would not launch.
        if (c.large_smem > wilcoxon_max_smem_per_block()) {
            c.run_large = false;
            c.run_huge = c.above_medium;
            c.huge_skip_le = OVO_MEDIUM_MAX;
        }
    }
    return c;
}

static std::vector<int> make_sort_group_ids(const int* h_grp_offsets,
                                            int n_groups, int skip_n_grp_le) {
    std::vector<int> ids;
    ids.reserve(n_groups);
    for (int g = 0; g < n_groups; ++g) {
        int sz = h_grp_offsets[g + 1] - h_grp_offsets[g];
        if (skip_n_grp_le > 0 && sz <= skip_n_grp_le) continue;
        ids.push_back(g);
    }
    return ids;
}

static inline void launch_ref_tie_sums(const float* ref_sorted,
                                       double* ref_tie_sums, int n_ref,
                                       int sb_cols, cudaStream_t stream) {
    ref_tie_sum_kernel<<<sb_cols, UTIL_BLOCK_SIZE, 0, stream>>>(
        ref_sorted, ref_tie_sums, n_ref, sb_cols);
    CUDA_CHECK_LAST_ERROR(ref_tie_sum_kernel);
}

static inline void launch_ovo_medium(
    const float* ref_sorted, const float* grp_dense, const int* grp_offsets,
    const double* ref_tie_sums, double* rank_sums, double* tie_corr, int n_ref,
    int n_all_grp, int sb_cols, int K, bool compute_tie_corr, int skip_n_grp_le,
    bool analytic_zeros, cudaStream_t stream) {
    constexpr int tpb = 256;
    size_t smem = (size_t)OVO_MEDIUM_MAX * sizeof(float) +
                  WARP_REDUCE_BUF * sizeof(double);
    dim3 grid(sb_cols, K);
    if (analytic_zeros) {
        ovo_rank_medium_analytic_kernel<<<grid, tpb, smem, stream>>>(
            ref_sorted, grp_dense, grp_offsets, ref_tie_sums, rank_sums,
            tie_corr, n_ref, n_all_grp, sb_cols, K, compute_tie_corr,
            skip_n_grp_le, OVO_MEDIUM_MAX);
        CUDA_CHECK_LAST_ERROR(ovo_rank_medium_analytic_kernel);
    } else {
        ovo_rank_medium_kernel<<<grid, tpb, smem, stream>>>(
            ref_sorted, grp_dense, grp_offsets, ref_tie_sums, rank_sums,
            tie_corr, n_ref, n_all_grp, sb_cols, K, compute_tie_corr,
            skip_n_grp_le, OVO_MEDIUM_MAX);
        CUDA_CHECK_LAST_ERROR(ovo_rank_medium_kernel);
    }
}

// Per-stream scratch for ovo_dispatch_tiers (one set per CUDA stream).
// grp_sorted/grp_seg_*/grp_cub_temp are HUGE-band only; may be null otherwise.
struct OvoTierScratch {
    double* ref_tie_sums;  // [sb_cols] pre-computed reference tie sums, or null
    double* sub_rank_sums;  // [n_groups * sb_cols] rank-sum output accumulator
    double* sub_tie_corr;   // [n_groups * sb_cols] tie-correction output
    float* grp_sorted;      // HUGE: [n_all_grp * sb_cols] sorted group values
    int* grp_seg_offsets;   // HUGE: CUB segment begins
    int* grp_seg_ends;      // HUGE: CUB segment ends
    uint8_t* grp_cub_temp;  // HUGE: CUB scratch
    float* grp_nz = nullptr;  // HUGE analytic CUB input
};

// Single OVO ranking engine shared by dense and all sparse host/device paths.
// Callers differ only in how they produce ref_sorted and grp_dense.
static inline void ovo_dispatch_tiers(
    const float* ref_sorted, const float* grp_dense, const int* grp_offsets,
    const OvoTierPlan& plan, const OvoTierScratch& sc,
    const int* d_sort_group_ids, int n_sort_groups, size_t grp_cub_temp_bytes,
    int sb_grp_items_actual, int tpb_rank, int n_ref, int n_all_grp,
    int sb_cols, int n_groups, bool compute_tie_corr, bool analytic_zeros,
    cudaStream_t stream) {
    // No-tie fast path: rank unsorted group values vs sorted ref (U-identity).
    // Skips group sort and all tier kernels.
    if (!compute_tie_corr) {
        constexpr int VS_REF_BLOCK = 256;
        dim3 grid(sb_cols, n_groups);
        ovo_rank_dense_vs_ref_kernel<<<grid, VS_REF_BLOCK, 0, stream>>>(
            ref_sorted, grp_dense, grp_offsets, sc.sub_rank_sums, n_ref,
            n_all_grp, sb_cols, n_groups);
        CUDA_CHECK_LAST_ERROR(ovo_rank_dense_vs_ref_kernel);
        return;
    }
    // One reference tie base is shared by all active tiers.
    if (compute_tie_corr) {
        launch_ref_tie_sums(ref_sorted, sc.ref_tie_sums, n_ref, sb_cols,
                            stream);
    }
    if (plan.run_medium) {
        launch_ovo_medium(ref_sorted, grp_dense, grp_offsets, sc.ref_tie_sums,
                          sc.sub_rank_sums, sc.sub_tie_corr, n_ref, n_all_grp,
                          sb_cols, n_groups, compute_tie_corr, /*skip=*/0,
                          analytic_zeros, stream);
    }

    if (plan.run_large) {
        dim3 grid(sb_cols, n_groups);
        if (analytic_zeros) {
            ovo_rank_smem_analytic_kernel<<<grid, plan.large_tpb,
                                            plan.large_smem, stream>>>(
                ref_sorted, grp_dense, grp_offsets, sc.ref_tie_sums,
                sc.sub_rank_sums, sc.sub_tie_corr, n_ref, n_all_grp, sb_cols,
                n_groups, compute_tie_corr, /*skip_n_grp_le=*/OVO_MEDIUM_MAX,
                /*skip_n_grp_gt=*/OVO_LARGE_MAX);
            CUDA_CHECK_LAST_ERROR(ovo_rank_smem_analytic_kernel);
        } else {
            ovo_rank_sorted_kernel<true>
                <<<grid, plan.large_tpb, plan.large_smem, stream>>>(
                    ref_sorted, grp_dense, grp_offsets, sc.ref_tie_sums,
                    sc.sub_rank_sums, sc.sub_tie_corr, n_ref, n_all_grp,
                    sb_cols, n_groups, compute_tie_corr, plan.large_padded,
                    /*skip_n_grp_le=*/OVO_MEDIUM_MAX,
                    /*skip_n_grp_gt=*/OVO_LARGE_MAX);
            CUDA_CHECK_LAST_ERROR(ovo_rank_sorted_kernel);
        }
    }
    if (plan.run_huge) {
        int sb_grp_seg =
            checked_int_product((size_t)n_sort_groups, (size_t)sb_cols,
                                "OVO active group segment count");
        if (analytic_zeros) {
            // CUB sorts only compacted positives.
            dim3 cgrid(sb_cols, n_sort_groups);
            compact_huge_nonzeros_kernel<<<cgrid, UTIL_BLOCK_SIZE, 0, stream>>>(
                grp_dense, grp_offsets, d_sort_group_ids, sc.grp_nz,
                sc.grp_seg_offsets, sc.grp_seg_ends, n_all_grp, n_sort_groups,
                sb_cols);
            CUDA_CHECK_LAST_ERROR(compact_huge_nonzeros_kernel);

            cub_segmented_sortkeys(sc.grp_cub_temp, grp_cub_temp_bytes,
                                   sc.grp_nz, sc.grp_sorted,
                                   sb_grp_items_actual, sb_grp_seg,
                                   sc.grp_seg_offsets, sc.grp_seg_ends, stream,
                                   "OVO huge-tier non-zero segmented sort");

            dim3 grid(sb_cols, n_sort_groups);
            ovo_rank_huge_analytic_kernel<<<grid, tpb_rank, 0, stream>>>(
                ref_sorted, sc.grp_sorted, grp_offsets, d_sort_group_ids,
                sc.grp_seg_offsets, sc.grp_seg_ends, sc.ref_tie_sums,
                sc.sub_rank_sums, sc.sub_tie_corr, n_ref, n_all_grp, sb_cols,
                n_sort_groups, compute_tie_corr);
            CUDA_CHECK_LAST_ERROR(ovo_rank_huge_analytic_kernel);
            return;
        }
        int blk = (sb_grp_seg + UTIL_BLOCK_SIZE - 1) / UTIL_BLOCK_SIZE;
        build_huge_seg_offsets_kernel<<<blk, UTIL_BLOCK_SIZE, 0, stream>>>(
            grp_offsets, d_sort_group_ids, sc.grp_seg_offsets, sc.grp_seg_ends,
            n_all_grp, n_sort_groups, sb_cols);
        CUDA_CHECK_LAST_ERROR(build_huge_seg_offsets_kernel);

        cub_segmented_sortkeys(sc.grp_cub_temp, grp_cub_temp_bytes, grp_dense,
                               sc.grp_sorted, sb_grp_items_actual, sb_grp_seg,
                               sc.grp_seg_offsets, sc.grp_seg_ends, stream,
                               "OVO huge-tier group segmented sort");

        dim3 grid(sb_cols, n_groups);
        ovo_rank_sorted_kernel<false><<<grid, tpb_rank, 0, stream>>>(
            ref_sorted, sc.grp_sorted, grp_offsets, sc.ref_tie_sums,
            sc.sub_rank_sums, sc.sub_tie_corr, n_ref, n_all_grp, sb_cols,
            n_groups, compute_tie_corr, /*large_padded=*/0,
            /*skip_n_grp_le=*/plan.huge_skip_le, /*skip_n_grp_gt=*/INT_MAX);
        CUDA_CHECK_LAST_ERROR(ovo_rank_sorted_kernel);
    }
}
