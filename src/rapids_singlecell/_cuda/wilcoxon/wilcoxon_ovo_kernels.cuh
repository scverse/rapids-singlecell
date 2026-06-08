#pragma once

#include <cub/device/device_segmented_radix_sort.cuh>

/**
 * Build CUB segmented-sort ranges only for groups in the HUGE band.
 * Group ids are relative to grp_offsets, and ranges still point into the
 * original dense group layout so the presorted rank kernel can read from the
 * normal per-group positions.
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
 * Extract specific rows from CSC into dense F-order, using a row lookup map.
 * row_map[original_row] = output_row_index (or -1 to skip).
 * One block per column, threads scatter matching nonzeros.
 * Output must be pre-zeroed.
 */
template <typename IndexT = int, typename IndptrT = int>
__global__ void csc_extract_mapped_kernel(const float* __restrict__ data,
                                          const IndexT* __restrict__ indices,
                                          const IndptrT* __restrict__ indptr,
                                          const int* __restrict__ row_map,
                                          float* __restrict__ out, int n_target,
                                          int col_start) {
    int col_local = blockIdx.x;
    int col = col_start + col_local;

    IndptrT start = indptr[col];
    IndptrT end = indptr[col + 1];

    for (IndptrT p = start + threadIdx.x; p < end; p += blockDim.x) {
        int out_row = row_map[(int)indices[p]];
        if (out_row >= 0) {
            out[(long long)col_local * n_target + out_row] = data[p];
        }
    }
}

/**
 * LARGE-band dispatch: when the largest group fits in shared memory, a fused
 * bitonic-sort + binary-search kernel handles the whole group per block.
 * Otherwise we fall back to the HUGE band (CUB segmented sort plus the
 * pre-sorted rank kernel).  This struct bundles the sizing knobs derived from
 * the host-side group offsets so each streaming impl can drop a 15-line prep
 * block.
 */
struct OvoTierPlan {
    int max_grp_size = 0;
    int min_grp_size = 0;
    bool run_warp = false;  // any group fits in one warp (≤ OVO_WARP_MAX)
    bool run_large =
        false;  // any group needs > WARP but fits the LARGE smem-sort band
    bool above_warp = false;    // at least one group exceeds OVO_WARP_MAX
    bool run_small = false;     // SMALL band: (OVO_WARP_MAX, OVO_SMALL_MAX]
    bool run_medium = false;    // MEDIUM band: (OVO_SMALL_MAX, OVO_MEDIUM_MAX]
    bool above_medium = false;  // at least one group exceeds OVO_MEDIUM_MAX
    int large_padded = 0;
    int large_tpb = 0;
    size_t large_smem = 0;
};

// Single source of truth for OVO tier dispatch (used by the dense path AND all
// four sparse OVO impls, which extract ref+group rows to dense then call this).
// Scans group sizes once; returns which size bands to co-launch (by max group):
//   WARP   (<=32):   ovo_rank_warp_kernel   (warp-shuffle sort, in registers)
//   SMALL  (<=64):   ovo_rank_small_kernel  (fixed 64-element smem sort)
//   MEDIUM (<=512):  ovo_rank_medium_kernel (no sort; O(n^2) in-group count)
//   LARGE  (<=2500): ovo_rank_large_kernel  (fused smem bitonic sort)
//   HUGE   (>2500):  CUB segmented sort + ovo_rank_huge_kernel (presorted rank)
// Bands cooperate via skip_n_grp_le (a larger band skips groups a smaller one
// already handled). LARGE is device-adapted: if its smem would exceed the
// per-block limit it falls back to HUGE.
static OvoTierPlan make_ovo_tier_plan(const int* h_grp_offsets, int n_groups) {
    OvoTierPlan c;
    c.min_grp_size = INT_MAX;
    for (int g = 0; g < n_groups; g++) {
        int sz = h_grp_offsets[g + 1] - h_grp_offsets[g];
        if (sz > c.max_grp_size) c.max_grp_size = sz;
        if (sz < c.min_grp_size) c.min_grp_size = sz;
        if (sz > OVO_WARP_MAX && sz <= OVO_SMALL_MAX) {
            c.run_small = true;
        }
        if (sz > OVO_SMALL_MAX && sz <= OVO_MEDIUM_MAX) {
            c.run_medium = true;
        }
        if (sz > OVO_MEDIUM_MAX) c.above_medium = true;
    }
    if (n_groups == 0) c.min_grp_size = 0;

    // run_warp: WARP kernel is worth running (at least one group small
    // enough to benefit from the warp path).
    c.run_warp = (c.min_grp_size <= OVO_WARP_MAX);
    // above_warp: at least one group needs a non-WARP kernel.
    c.above_warp = (c.max_grp_size > OVO_WARP_MAX);
    // run_large: the fused smem-sort fast path (groups > WARP but ≤ LARGE).
    c.run_large = c.above_warp && (c.max_grp_size <= OVO_LARGE_MAX);
    if (c.run_large) {
        c.large_padded = 1;
        while (c.large_padded < c.max_grp_size) c.large_padded <<= 1;
        c.large_tpb = std::min(c.large_padded, MAX_THREADS_PER_BLOCK);
        c.large_smem = (size_t)c.large_padded * sizeof(float) +
                       WARP_REDUCE_BUF * sizeof(double);
        // Adapt to the device: if the fused-sort buffer would exceed the
        // per-block shared-memory limit, fall back to the HUGE-band CUB
        // segmented sort (no smem cap) rather than launching a kernel that
        // would fail. Never triggers at the current threshold (~16.6KB), but
        // keeps the dispatch correct if the threshold or device limit changes.
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

// WARP kernel launcher: 8 warps × 32 threads per block, one (col, group)
// pair per warp.  grid.y covers ceil(K/8) pair rows.
static inline void launch_ovo_warp(const float* ref_sorted,
                                   const float* grp_dense,
                                   const int* grp_offsets,
                                   const double* ref_tie_sums,
                                   double* rank_sums, double* tie_corr,
                                   int n_ref, int n_all_grp, int sb_cols, int K,
                                   bool compute_tie_corr, cudaStream_t stream) {
    constexpr int WARPS_PER_BLOCK = 8;
    dim3 grid(sb_cols, (K + WARPS_PER_BLOCK - 1) / WARPS_PER_BLOCK);
    ovo_rank_warp_kernel<<<grid, 256, 0, stream>>>(
        ref_sorted, grp_dense, grp_offsets, ref_tie_sums, rank_sums, tie_corr,
        n_ref, n_all_grp, sb_cols, K, compute_tie_corr);
    CUDA_CHECK_LAST_ERROR(ovo_rank_warp_kernel);
}

static inline void launch_ref_tie_sums(const float* ref_sorted,
                                       double* ref_tie_sums, int n_ref,
                                       int sb_cols, cudaStream_t stream) {
    ref_tie_sum_kernel<<<sb_cols, UTIL_BLOCK_SIZE, 0, stream>>>(
        ref_sorted, ref_tie_sums, n_ref, sb_cols);
    CUDA_CHECK_LAST_ERROR(ref_tie_sum_kernel);
}

static inline void launch_ovo_small(
    const float* ref_sorted, const float* grp_dense, const int* grp_offsets,
    const double* ref_tie_sums, double* rank_sums, double* tie_corr, int n_ref,
    int n_all_grp, int sb_cols, int K, bool compute_tie_corr, int skip_n_grp_le,
    cudaStream_t stream) {
    dim3 grid(sb_cols, K);
    ovo_rank_small_kernel<<<grid, OVO_SMALL_MAX, 0, stream>>>(
        ref_sorted, grp_dense, grp_offsets, ref_tie_sums, rank_sums, tie_corr,
        n_ref, n_all_grp, sb_cols, K, compute_tie_corr, skip_n_grp_le);
    CUDA_CHECK_LAST_ERROR(ovo_rank_small_kernel);
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
// impls (host/device CSC/CSR). Given an already-sorted reference slice and a
// dense group slice for one column sub-batch, it runs the size-banded dispatch
// from `plan` (see make_ovo_tier_plan): co-launch WARP/SMALL/MEDIUM for small
// groups, then LARGE (fused smem sort) OR HUGE (CUB segmented sort) for the
// rest. Pure host-side code motion: the kernel launches are identical to the
// previous inline copies, so results and performance are unchanged. The five
// callers differ only in how they produce ref_sorted / grp_dense.
static inline void ovo_dispatch_tiers(
    const float* ref_sorted, const float* grp_dense, const int* grp_offsets,
    const OvoTierPlan& plan, const OvoTierScratch& sc,
    const int* d_sort_group_ids, int n_sort_groups, size_t grp_cub_temp_bytes,
    int sb_grp_items_actual, int tpb_rank, int n_ref, int n_all_grp,
    int sb_cols, int n_groups, bool compute_tie_corr, cudaStream_t stream) {
    bool run_large = plan.above_medium && plan.run_large;
    bool run_huge = plan.above_medium && !run_large;

    int skip_le = 0;
    if (compute_tie_corr &&
        (plan.run_warp || plan.run_small || plan.run_medium)) {
        launch_ref_tie_sums(ref_sorted, sc.ref_tie_sums, n_ref, sb_cols,
                            stream);
    }
    if (plan.run_warp) {
        launch_ovo_warp(ref_sorted, grp_dense, grp_offsets, sc.ref_tie_sums,
                        sc.sub_rank_sums, sc.sub_tie_corr, n_ref, n_all_grp,
                        sb_cols, n_groups, compute_tie_corr, stream);
        if (plan.above_warp) skip_le = OVO_WARP_MAX;
    }
    if (plan.run_small) {
        launch_ovo_small(ref_sorted, grp_dense, grp_offsets, sc.ref_tie_sums,
                         sc.sub_rank_sums, sc.sub_tie_corr, n_ref, n_all_grp,
                         sb_cols, n_groups, compute_tie_corr, skip_le, stream);
        if (plan.max_grp_size > OVO_SMALL_MAX) skip_le = OVO_SMALL_MAX;
    }
    if (plan.run_medium) {
        launch_ovo_medium(ref_sorted, grp_dense, grp_offsets, sc.ref_tie_sums,
                          sc.sub_rank_sums, sc.sub_tie_corr, n_ref, n_all_grp,
                          sb_cols, n_groups, compute_tie_corr, skip_le, stream);
    }

    int upper_skip_le = plan.above_medium ? OVO_MEDIUM_MAX : skip_le;
    if (plan.above_medium && run_large) {
        dim3 grid(sb_cols, n_groups);
        ovo_rank_large_kernel<<<grid, plan.large_tpb, plan.large_smem,
                                stream>>>(
            ref_sorted, grp_dense, grp_offsets, sc.sub_rank_sums,
            sc.sub_tie_corr, n_ref, n_all_grp, sb_cols, n_groups,
            compute_tie_corr, plan.large_padded, upper_skip_le);
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

        size_t temp = grp_cub_temp_bytes;
        cuda_check(cub::DeviceSegmentedRadixSort::SortKeys(
                       sc.grp_cub_temp, temp, grp_dense, sc.grp_sorted,
                       sb_grp_items_actual, sb_grp_seg, sc.grp_seg_offsets,
                       sc.grp_seg_ends, BEGIN_BIT, END_BIT, stream),
                   "OVO huge-tier group segmented sort");

        dim3 grid(sb_cols, n_groups);
        ovo_rank_huge_kernel<<<grid, tpb_rank, 0, stream>>>(
            ref_sorted, sc.grp_sorted, grp_offsets, sc.sub_rank_sums,
            sc.sub_tie_corr, n_ref, n_all_grp, sb_cols, n_groups,
            compute_tie_corr, upper_skip_le);
        CUDA_CHECK_LAST_ERROR(ovo_rank_huge_kernel);
    }
}
