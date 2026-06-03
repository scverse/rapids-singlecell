#pragma once

/**
 * Build CUB segmented-sort ranges only for groups that Tier 3 will rank.
 * Group ids are relative to grp_offsets, and ranges still point into the
 * original dense group layout so the presorted rank kernel can read from the
 * normal per-group positions.
 */
__global__ void build_tier3_seg_begin_end_offsets_kernel(
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
 * Tier 1 dispatch: when the largest group fits in shared memory, a fused
 * bitonic-sort + binary-search kernel handles the whole group per block.
 * Otherwise we fall back to CUB segmented sort plus the pre-sorted rank
 * kernel.  This struct bundles the sizing knobs derived from the host-side
 * group offsets so each streaming impl can drop a 15-line prep block.
 */
struct Tier1Config {
    int max_grp_size = 0;
    int min_grp_size = 0;
    bool use_tier0 =
        false;  // any group fits in one warp (≤ TIER0_GROUP_THRESHOLD)
    bool use_tier1 =
        false;  // any group needs > tier0 but fits in tier1 smem sort
    bool any_above_t0 =
        false;  // at least one group exceeds TIER0_GROUP_THRESHOLD
    bool any_tier0_64 = false;  // any group needs Tier 0.5: (T0, T0_64]
    bool any_tier2 = false;     // any group needs Tier 2: (T0_64, T2]
    bool any_above_t2 =
        false;  // at least one group exceeds TIER2_GROUP_THRESHOLD
    int padded_grp_size = 0;
    int tier1_tpb = 0;
    size_t tier1_smem = 0;
};

// SINGLE source of truth for OVO tier dispatch, used by EVERY OVO path:
// dense (wilcoxon.cu) AND all four sparse OVO impls (host/device CSC/CSR),
// which extract ref+group rows to dense per sub-batch and then call this.
// Scans per-group sizes once and returns which size-gated tiers to co-launch:
//   Tier 0   (<=32, TIER0_GROUP_THRESHOLD):    ovo_warp_sort_rank_kernel
//   Tier 0.5 (<=64, TIER0_64_GROUP_THRESHOLD): ovo_small64_sort_rank_kernel
//   Tier 2   (<=512, TIER2_GROUP_THRESHOLD):   ovo_medium_unsorted_rank_kernel
//   Tier 1   (<=2500, TIER1_GROUP_THRESHOLD):  ovo_fused_sort_rank_kernel (smem
//   sort) Tier 3   (>2500):                          CUB segmented sort +
//   batched_rank_sums_presorted_kernel
// Tiers cooperate via skip_n_grp_le: a larger tier skips groups a smaller tier
// already handled. Tier 1 is device-adapted: if its fused-sort smem footprint
// would exceed wilcoxon_max_smem_per_block() it is disabled in favor of Tier 3.
static Tier1Config make_tier1_config(const int* h_grp_offsets, int n_groups) {
    Tier1Config c;
    c.min_grp_size = INT_MAX;
    for (int g = 0; g < n_groups; g++) {
        int sz = h_grp_offsets[g + 1] - h_grp_offsets[g];
        if (sz > c.max_grp_size) c.max_grp_size = sz;
        if (sz < c.min_grp_size) c.min_grp_size = sz;
        if (sz > TIER0_GROUP_THRESHOLD && sz <= TIER0_64_GROUP_THRESHOLD) {
            c.any_tier0_64 = true;
        }
        if (sz > TIER0_64_GROUP_THRESHOLD && sz <= TIER2_GROUP_THRESHOLD) {
            c.any_tier2 = true;
        }
        if (sz > TIER2_GROUP_THRESHOLD) c.any_above_t2 = true;
    }
    if (n_groups == 0) c.min_grp_size = 0;

    // use_tier0: Tier 0 kernel is worth running (at least one group small
    // enough to benefit from the warp path).
    c.use_tier0 = (c.min_grp_size <= TIER0_GROUP_THRESHOLD);
    // any_above_t0: at least one group needs a non-Tier-0 kernel.
    c.any_above_t0 = (c.max_grp_size > TIER0_GROUP_THRESHOLD);
    // use_tier1: the fused smem-sort fast path (for groups > T0 but ≤ T1).
    c.use_tier1 = c.any_above_t0 && (c.max_grp_size <= TIER1_GROUP_THRESHOLD);
    if (c.use_tier1) {
        c.padded_grp_size = 1;
        while (c.padded_grp_size < c.max_grp_size) c.padded_grp_size <<= 1;
        c.tier1_tpb = std::min(c.padded_grp_size, MAX_THREADS_PER_BLOCK);
        c.tier1_smem = (size_t)c.padded_grp_size * sizeof(float) +
                       WARP_REDUCE_BUF * sizeof(double);
        // Adapt to the device: if the fused-sort buffer would exceed the
        // per-block shared-memory limit, fall back to the tier-3 CUB segmented
        // sort (which has no smem cap) rather than launching a kernel that
        // would fail. Never triggers at the current threshold (~16.6KB), but
        // keeps the dispatch correct if the threshold or device limit changes.
        if (c.tier1_smem > wilcoxon_max_smem_per_block()) {
            c.use_tier1 = false;
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

// Tier 0 kernel launcher: 8 warps × 32 threads per block, one (col, group)
// pair per warp.  grid.y covers ceil(K/8) pair rows.
static inline void launch_tier0(const float* ref_sorted, const float* grp_dense,
                                const int* grp_offsets,
                                const double* ref_tie_sums, double* rank_sums,
                                double* tie_corr, int n_ref, int n_all_grp,
                                int sb_cols, int K, bool compute_tie_corr,
                                cudaStream_t stream) {
    constexpr int WARPS_PER_BLOCK = 8;
    dim3 grid(sb_cols, (K + WARPS_PER_BLOCK - 1) / WARPS_PER_BLOCK);
    ovo_warp_sort_rank_kernel<<<grid, 256, 0, stream>>>(
        ref_sorted, grp_dense, grp_offsets, ref_tie_sums, rank_sums, tie_corr,
        n_ref, n_all_grp, sb_cols, K, compute_tie_corr);
    CUDA_CHECK_LAST_ERROR(ovo_warp_sort_rank_kernel);
}

static inline void launch_ref_tie_sums(const float* ref_sorted,
                                       double* ref_tie_sums, int n_ref,
                                       int sb_cols, cudaStream_t stream) {
    ref_tie_sum_kernel<<<sb_cols, UTIL_BLOCK_SIZE, 0, stream>>>(
        ref_sorted, ref_tie_sums, n_ref, sb_cols);
    CUDA_CHECK_LAST_ERROR(ref_tie_sum_kernel);
}

static inline void launch_tier0_64(
    const float* ref_sorted, const float* grp_dense, const int* grp_offsets,
    const double* ref_tie_sums, double* rank_sums, double* tie_corr, int n_ref,
    int n_all_grp, int sb_cols, int K, bool compute_tie_corr, int skip_n_grp_le,
    cudaStream_t stream) {
    dim3 grid(sb_cols, K);
    ovo_small64_sort_rank_kernel<<<grid, TIER0_64_GROUP_THRESHOLD, 0, stream>>>(
        ref_sorted, grp_dense, grp_offsets, ref_tie_sums, rank_sums, tie_corr,
        n_ref, n_all_grp, sb_cols, K, compute_tie_corr, skip_n_grp_le);
    CUDA_CHECK_LAST_ERROR(ovo_small64_sort_rank_kernel);
}

static inline void launch_tier2_medium(
    const float* ref_sorted, const float* grp_dense, const int* grp_offsets,
    const double* ref_tie_sums, double* rank_sums, double* tie_corr, int n_ref,
    int n_all_grp, int sb_cols, int K, bool compute_tie_corr, int skip_n_grp_le,
    cudaStream_t stream) {
    constexpr int tpb = 256;
    size_t smem = (size_t)TIER2_GROUP_THRESHOLD * sizeof(float) +
                  WARP_REDUCE_BUF * sizeof(double);
    dim3 grid(sb_cols, K);
    ovo_medium_unsorted_rank_kernel<<<grid, tpb, smem, stream>>>(
        ref_sorted, grp_dense, grp_offsets, ref_tie_sums, rank_sums, tie_corr,
        n_ref, n_all_grp, sb_cols, K, compute_tie_corr, skip_n_grp_le,
        TIER2_GROUP_THRESHOLD);
    CUDA_CHECK_LAST_ERROR(ovo_medium_unsorted_rank_kernel);
}
