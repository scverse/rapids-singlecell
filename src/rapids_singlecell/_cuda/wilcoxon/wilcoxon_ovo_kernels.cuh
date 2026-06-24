#pragma once

#include <cub/device/device_segmented_radix_sort.cuh>

#include "../sparse_extract/sparse_extract.cuh"

/**
 * Build CUB segmented-sort ranges for HUGE-band groups. Ranges point into the
 * original dense group layout so the presorted rank kernel reads normal
 * per-group positions.
 */
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

/**
 * Sizing knobs for LARGE-band dispatch: when the largest group fits in shared
 * memory, a fused bitonic-sort + binary-search kernel handles the group per
 * block; otherwise fall back to the HUGE band (CUB segmented sort + pre-sorted
 * rank kernel).
 */
struct OvoTierPlan {
    int max_grp_size = 0;
    bool run_medium = false;    // MEDIUM band: any group ≤ OVO_MEDIUM_MAX
    bool run_large = false;     // LARGE band: (OVO_MEDIUM_MAX, OVO_LARGE_MAX]
    bool above_medium = false;  // at least one group exceeds OVO_MEDIUM_MAX
    int large_padded = 0;
    int large_tpb = 0;
    size_t large_smem = 0;
};

// Single source of truth for OVO tier dispatch (used by the dense path AND all
// four sparse OVO impls, which extract ref+group rows to dense then call this).
// Scans group sizes once; returns which size bands to co-launch (by max group):
//   MEDIUM (<=512):  ovo_rank_medium_kernel (no sort; O(n^2) in-group count)
//   LARGE  (<=2500): ovo_rank_large_kernel  (fused smem bitonic sort)
//   HUGE   (>2500):  CUB segmented sort + ovo_rank_huge_kernel (presorted rank)
// MEDIUM is the smallest tier (the WARP/SMALL sub-tiers were removed -- no
// measurable speedup on real data; archived in
// .claude/wilcoxon-warp-small-tiers-removed.md). MEDIUM co-launches with LARGE
// or HUGE; the upper tier skips groups ≤ OVO_MEDIUM_MAX (skip_n_grp_le). LARGE
// is device-adapted: if its smem would exceed the per-block limit it falls back
// to HUGE.
static OvoTierPlan make_ovo_tier_plan(const int* h_grp_offsets, int n_groups) {
    OvoTierPlan c;
    for (int g = 0; g < n_groups; g++) {
        int sz = h_grp_offsets[g + 1] - h_grp_offsets[g];
        if (sz > c.max_grp_size) c.max_grp_size = sz;
        if (sz <= OVO_MEDIUM_MAX) c.run_medium = true;
        if (sz > OVO_MEDIUM_MAX) c.above_medium = true;
    }

    // run_large: the fused smem-sort fast path for groups > MEDIUM but ≤ LARGE.
    c.run_large = c.above_medium && (c.max_grp_size <= OVO_LARGE_MAX);
    if (c.run_large) {
        c.large_padded = 1;
        while (c.large_padded < c.max_grp_size) c.large_padded <<= 1;
        c.large_tpb = std::min(c.large_padded, MAX_THREADS_PER_BLOCK);
        c.large_smem = (size_t)c.large_padded * sizeof(float) +
                       WARP_REDUCE_BUF * sizeof(double);
        // Device-adapt: if the fused-sort buffer exceeds the per-block smem
        // limit, fall back to HUGE (no smem cap) instead of launching a kernel
        // that would fail. Inert at the current ~16.6KB threshold; guards
        // against threshold/device-limit changes.
        if (c.large_smem > wilcoxon_max_smem_per_block()) {
            c.run_large = false;
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
    cudaStream_t stream) {
    constexpr int tpb = 256;
    size_t smem = (size_t)OVO_MEDIUM_MAX * sizeof(float) +
                  WARP_REDUCE_BUF * sizeof(double);
    dim3 grid(sb_cols, K);
    ovo_rank_medium_kernel<<<grid, tpb, smem, stream>>>(
        ref_sorted, grp_dense, grp_offsets, ref_tie_sums, rank_sums, tie_corr,
        n_ref, n_all_grp, sb_cols, K, compute_tie_corr, skip_n_grp_le,
        OVO_MEDIUM_MAX);
    CUDA_CHECK_LAST_ERROR(ovo_rank_medium_kernel);
}

// Per-stream scratch consumed by ovo_dispatch_tiers (one set per CUDA stream).
// grp_sorted/grp_seg_*/grp_cub_temp are only needed for the HUGE band and may
// be null otherwise.
struct OvoTierScratch {
    double* ref_tie_sums;  // [sb_cols] pre-computed reference tie sums, or null
    double* sub_rank_sums;  // [n_groups * sb_cols] rank-sum output accumulator
    double* sub_tie_corr;   // [n_groups * sb_cols] tie-correction output
    float* grp_sorted;      // HUGE: [n_all_grp * sb_cols] sorted group values
    int* grp_seg_offsets;   // HUGE: CUB segment begins
    int* grp_seg_ends;      // HUGE: CUB segment ends
    uint8_t* grp_cub_temp;  // HUGE: CUB scratch
};

// SINGLE OVO ranking engine, shared by the dense path and all four sparse OVO
// impls (host/device CSC/CSR). Given a sorted reference slice and a dense group
// slice for one column sub-batch, runs the size-banded dispatch from `plan`
// (see make_ovo_tier_plan): co-launch MEDIUM for groups ≤512, then LARGE (fused
// smem sort) OR HUGE (CUB segmented sort) for the rest. Callers differ only in
// how they produce ref_sorted / grp_dense.
static inline void ovo_dispatch_tiers(
    const float* ref_sorted, const float* grp_dense, const int* grp_offsets,
    const OvoTierPlan& plan, const OvoTierScratch& sc,
    const int* d_sort_group_ids, int n_sort_groups, size_t grp_cub_temp_bytes,
    int sb_grp_items_actual, int tpb_rank, int n_ref, int n_all_grp,
    int sb_cols, int n_groups, bool compute_tie_corr, cudaStream_t stream) {
    // No-tie fast path (tie_correct=False, the default): rank each group value
    // vs the sorted reference only (U-identity), skipping the group sort and
    // all tiers. grp_dense is unsorted here, which is exactly what this kernel
    // wants.
    if (!compute_tie_corr) {
        constexpr int VS_REF_BLOCK = 256;
        dim3 grid(sb_cols, n_groups);
        ovo_rank_dense_vs_ref_kernel<<<grid, VS_REF_BLOCK, 0, stream>>>(
            ref_sorted, grp_dense, grp_offsets, sc.sub_rank_sums, n_ref,
            n_all_grp, sb_cols, n_groups);
        CUDA_CHECK_LAST_ERROR(ovo_rank_dense_vs_ref_kernel);
        return;
    }
    bool run_large = plan.above_medium && plan.run_large;
    bool run_huge = plan.above_medium && !run_large;

    // All tiers (MEDIUM/LARGE/HUGE) share the precomputed reference tie base,
    // so compute it once per column whenever correcting.
    if (compute_tie_corr) {
        launch_ref_tie_sums(ref_sorted, sc.ref_tie_sums, n_ref, sb_cols,
                            stream);
    }
    // MEDIUM is the smallest tier: it handles every group ≤ OVO_MEDIUM_MAX
    // (skip_n_grp_le = 0). LARGE/HUGE then take the groups above MEDIUM.
    if (plan.run_medium) {
        launch_ovo_medium(ref_sorted, grp_dense, grp_offsets, sc.ref_tie_sums,
                          sc.sub_rank_sums, sc.sub_tie_corr, n_ref, n_all_grp,
                          sb_cols, n_groups, compute_tie_corr, /*skip=*/0,
                          stream);
    }

    int upper_skip_le = plan.above_medium ? OVO_MEDIUM_MAX : 0;
    if (plan.above_medium && run_large) {
        dim3 grid(sb_cols, n_groups);
        ovo_rank_large_kernel<<<grid, plan.large_tpb, plan.large_smem,
                                stream>>>(
            ref_sorted, grp_dense, grp_offsets, sc.ref_tie_sums,
            sc.sub_rank_sums, sc.sub_tie_corr, n_ref, n_all_grp, sb_cols,
            n_groups, compute_tie_corr, plan.large_padded, upper_skip_le);
        CUDA_CHECK_LAST_ERROR(ovo_rank_large_kernel);
    } else if (run_huge) {
        int sb_grp_seg =
            checked_int_product((size_t)n_sort_groups, (size_t)sb_cols,
                                "OVO active group segment count");
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
        ovo_rank_huge_kernel<<<grid, tpb_rank, 0, stream>>>(
            ref_sorted, sc.grp_sorted, grp_offsets, sc.ref_tie_sums,
            sc.sub_rank_sums, sc.sub_tie_corr, n_ref, n_all_grp, sb_cols,
            n_groups, compute_tie_corr, upper_skip_le);
        CUDA_CHECK_LAST_ERROR(ovo_rank_huge_kernel);
    }
}
