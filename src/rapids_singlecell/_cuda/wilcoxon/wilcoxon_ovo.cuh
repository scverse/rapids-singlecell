#pragma once

#include "wilcoxon_common.cuh"

// Bitonic sort of power-of-two `n` floats in shared memory, ascending.
// Pad the tail with +INF before calling; any blockDim works.

__device__ __forceinline__ void bitonic_sort_smem(float* s, int n) {
    for (int k = 2; k <= n; k <<= 1) {
        for (int j = k >> 1; j > 0; j >>= 1) {
            for (int i = threadIdx.x; i < n; i += blockDim.x) {
                int ixj = i ^ j;
                if (ixj > i) {
                    bool asc = ((i & k) == 0);
                    float a = s[i], b = s[ixj];
                    if (asc ? (a > b) : (a < b)) {
                        s[i] = b;
                        s[ixj] = a;
                    }
                }
            }
            __syncthreads();
        }
    }
}

// Rank sum and tie correction of one (column, group) from the group's
// nonzeros: sorted (binary search) or, for small counts, unsorted (O(nnz^2)
// scan). Zeros form one tie block between negatives and positives, and every
// nonzero adds grp_zero (it sits above the group's zeros or raises their
// ranks), so signed and dense data take the same path as sparse data.
__device__ __forceinline__ void ovo_rank_nonzeros(
    const float* ref, int n_ref, const float* nz, int nnz, int n_grp,
    bool sorted, double ref_tie, double* warp_buf, double* rank_out,
    double* tie_out) {
    int ref_neg = sorted_lower_bound(ref, 0, n_ref, 0.0f);
    int ref_zero = sorted_upper_bound(ref, ref_neg, n_ref, 0.0f) - ref_neg;
    int grp_zero = n_grp - nnz;
    double local_sum = 0.0;
    double local_tie = 0.0;
    if (threadIdx.x == 0) {
        local_sum =
            (double)grp_zero *
            ((double)ref_neg + ((double)(ref_zero + grp_zero) + 1.0) / 2.0);
        local_tie = tie3(ref_zero + grp_zero) - tie3(ref_zero);
    }
    int ref_lb = 0, grp_lb = 0;  // monotone search starts when sorted
    for (int i = threadIdx.x; i < nnz; i += blockDim.x) {
        float v = nz[i];
        int lt_r = sorted_lower_bound(ref, sorted ? ref_lb : 0, n_ref, v);
        int cr = sorted_upper_bound(ref, lt_r, n_ref, v) - lt_r;
        int lt_g = 0, cg = 0;
        bool first = true;
        if (sorted) {
            ref_lb = lt_r;
            lt_g = grp_lb = sorted_lower_bound(nz, grp_lb, nnz, v);
            cg = sorted_upper_bound(nz, i + 1, nnz, v) - lt_g;
            first = lt_g == i;
        } else {
            for (int j = 0; j < nnz; ++j) {
                float w = nz[j];
                lt_g += w < v;
                if (w == v) {
                    ++cg;
                    first &= j >= i;
                }
            }
        }
        local_sum +=
            (double)(lt_r + lt_g + grp_zero) + ((double)(cr + cg) + 1.0) / 2.0;
        if (first) local_tie += tie3(cg + cr) - tie3(cr);
    }
    double total = wilcoxon_block_sum(local_sum, warp_buf);
    if (threadIdx.x == 0) *rank_out = total;
    __syncthreads();
    double tie = wilcoxon_block_sum(local_tie, warp_buf);
    if (threadIdx.x == 0)
        *tie_out = finalize_tie_corr(n_ref + n_grp, ref_tie + tie);
}

// Warp-aggregated compaction of src's nonzeros into dst (unordered); *count
// must start at zero.
__device__ __forceinline__ void compact_nonzeros(const float* src, int n,
                                                 float* dst, int* count) {
    int lane = threadIdx.x & 31;
    for (int base = 0; base < n; base += blockDim.x) {
        int i = base + threadIdx.x;
        float v = i < n ? src[i] : 0.0f;
        unsigned mask = __ballot_sync(0xffffffffu, v != 0.0f);
        int slot = 0;
        if (lane == 0 && mask) slot = atomicAdd(count, __popc(mask));
        slot = __shfl_sync(0xffffffffu, slot, 0) +
               __popc(mask & ((1u << lane) - 1u));
        if (v != 0.0f) dst[slot] = v;
    }
}

// No-tie path: group-internal ranks collapse to the U closed form, so each
// unsorted group value only searches the sorted reference (zeros share one
// search).
__global__ void ovo_rank_dense_vs_ref_kernel(
    const float* __restrict__ ref_sorted, const float* __restrict__ grp_dense,
    const int* __restrict__ grp_offsets, double* __restrict__ rank_sums,
    int n_ref, int n_all_grp, int ld) {
    int col = blockIdx.x;
    int grp = blockIdx.y;

    int g_start = grp_offsets[grp];
    int n_grp = grp_offsets[grp + 1] - g_start;
    if (n_grp == 0) {
        if (threadIdx.x == 0) rank_sums[(size_t)grp * ld + col] = 0.0;
        return;
    }
    const float* ref_col = ref_sorted + (long long)col * n_ref;
    const float* grp_col = grp_dense + (long long)col * n_all_grp + g_start;

    __shared__ double warp_buf[WARP_REDUCE_BUF];
    __shared__ int ref_neg, ref_zero;
    if (threadIdx.x == 0) {
        ref_neg = sorted_lower_bound(ref_col, 0, n_ref, 0.0f);
        ref_zero = sorted_upper_bound(ref_col, ref_neg, n_ref, 0.0f) - ref_neg;
    }
    __syncthreads();

    double local_sum = 0.0;
    int n_zero = 0;
    for (int i = threadIdx.x; i < n_grp; i += blockDim.x) {
        float v = grp_col[i];
        if (v == 0.0f) {
            n_zero++;
            continue;
        }
        int n_lt = sorted_lower_bound(ref_col, 0, n_ref, v);
        int n_eq = sorted_upper_bound(ref_col, n_lt, n_ref, v) - n_lt;
        local_sum += (double)n_lt + 0.5 * (double)n_eq;
    }
    local_sum += (double)n_zero * ((double)ref_neg + 0.5 * (double)ref_zero);
    double total = wilcoxon_block_sum(local_sum, warp_buf);
    if (threadIdx.x == 0) {
        rank_sums[(size_t)grp * ld + col] =
            total + (double)n_grp * ((double)n_grp + 1.0) / 2.0;
    }
}

// Tie path for groups with lo <= n_grp <= hi: compact the group's nonzeros
// into shared memory, bitonic-sort them when there are more than
// OVO_MEDIUM_MAX, and rank against the sorted reference.
__global__ void ovo_rank_smem_kernel(const float* __restrict__ ref_sorted,
                                     const float* __restrict__ grp_dense,
                                     const int* __restrict__ grp_offsets,
                                     const double* __restrict__ ref_tie_sums,
                                     double* __restrict__ rank_sums,
                                     double* __restrict__ tie_corr, int n_ref,
                                     int n_all_grp, int ld, int lo, int hi) {
    int col = blockIdx.x;
    int grp = blockIdx.y;
    int g_start = grp_offsets[grp];
    int n_grp = grp_offsets[grp + 1] - g_start;
    if (n_grp < lo || n_grp > hi) return;

    extern __shared__ float nz[];
    __shared__ double warp_buf[WARP_REDUCE_BUF];
    __shared__ int sh_nnz;
    if (threadIdx.x == 0) sh_nnz = 0;
    __syncthreads();
    compact_nonzeros(grp_dense + (size_t)col * n_all_grp + g_start, n_grp, nz,
                     &sh_nnz);
    __syncthreads();
    int nnz = sh_nnz;
    bool sorted = nnz > OVO_MEDIUM_MAX;
    if (sorted) {
        int padded = 1;
        while (padded < nnz) padded <<= 1;
        for (int i = nnz + threadIdx.x; i < padded; i += blockDim.x)
            nz[i] = __int_as_float(0x7f800000);  // +INF pad
        __syncthreads();
        bitonic_sort_smem(nz, padded);
    }

    size_t out = (size_t)grp * ld + col;
    ovo_rank_nonzeros(ref_sorted + (size_t)col * n_ref, n_ref, nz, nnz, n_grp,
                      sorted, ref_tie_sums[col], warp_buf, rank_sums + out,
                      tie_corr + out);
}

// HUGE groups: compact each (column, group) segment's nonzeros to the start
// of its slot and emit [begin, end) for the CUB segmented sort.
__global__ void ovo_compact_nonzeros_kernel(const float* __restrict__ grp_dense,
                                            const int* __restrict__ grp_offsets,
                                            const int* __restrict__ group_ids,
                                            float* __restrict__ grp_nz,
                                            int* __restrict__ seg_begins,
                                            int* __restrict__ seg_ends,
                                            int n_all_grp, int n_sort_groups) {
    int col = blockIdx.x;
    int local = blockIdx.y;
    int g = group_ids[local];
    int g_start = grp_offsets[g];
    int n_grp = grp_offsets[g + 1] - g_start;
    size_t base = (size_t)col * n_all_grp + g_start;

    __shared__ int cnt;
    if (threadIdx.x == 0) cnt = 0;
    __syncthreads();
    compact_nonzeros(grp_dense + base, n_grp, grp_nz + base, &cnt);
    __syncthreads();
    if (threadIdx.x == 0) {
        int f = col * n_sort_groups + local;
        seg_begins[f] = (int)base;
        seg_ends[f] = (int)base + cnt;
    }
}

// HUGE groups: rank the CUB-sorted nonzero segments.
__global__ void ovo_rank_sorted_nz_kernel(
    const float* __restrict__ ref_sorted, const float* __restrict__ grp_sorted,
    const int* __restrict__ grp_offsets, const int* __restrict__ group_ids,
    const int* __restrict__ seg_begins, const int* __restrict__ seg_ends,
    const double* __restrict__ ref_tie_sums, double* __restrict__ rank_sums,
    double* __restrict__ tie_corr, int n_ref, int ld, int n_sort_groups) {
    int col = blockIdx.x;
    int local = blockIdx.y;
    int grp = group_ids[local];
    int n_grp = grp_offsets[grp + 1] - grp_offsets[grp];
    int f = col * n_sort_groups + local;
    int begin = seg_begins[f];

    __shared__ double warp_buf[WARP_REDUCE_BUF];
    size_t out = (size_t)grp * ld + col;
    ovo_rank_nonzeros(ref_sorted + (size_t)col * n_ref, n_ref,
                      grp_sorted + begin, seg_ends[f] - begin, n_grp, true,
                      ref_tie_sums[col], warp_buf, rank_sums + out,
                      tie_corr + out);
}

// Sorted-reference tie base, one block per column; group kernels add deltas.
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
            int cnt = sorted_upper_bound(ref_col, i + 1, n_ref, v) - i;
            if (cnt > 1) local_tie += tie3(cnt);
        }
    }

    __shared__ double warp_buf[WARP_REDUCE_BUF];
    double total = wilcoxon_block_sum(local_tie, warp_buf);
    if (threadIdx.x == 0) ref_tie_sums[col] = total;
}

// Per-stream OVO scratch. HUGE fields stay null unless HUGE groups are ranked.
struct OvoScratch {
    double* ref_tie_sums = nullptr;  // [sb_cols]
    float* grp_nz = nullptr;         // compacted HUGE nonzeros
    float* grp_sorted = nullptr;     // CUB-sorted HUGE nonzeros
    int* seg_begins = nullptr;
    int* seg_ends = nullptr;
    uint8_t* cub_temp = nullptr;
};

// Tie bands shared by every OVO layout. Groups up to OVO_MEDIUM_MAX and up to
// OVO_LARGE_MAX sort their nonzeros in shared memory (two launch shapes);
// larger groups compact nonzeros for one CUB segmented sort.
struct OvoRanker {
    bool tie = false;
    int n_groups = 0;
    int max_grp_size = 0;
    int medium_max = -1;  // largest group <= OVO_MEDIUM_MAX, -1 if none
    int large_max = -1;   // largest group in (OVO_MEDIUM_MAX, OVO_LARGE_MAX]
    std::vector<int> huge_ids;
    const int* d_huge_ids = nullptr;

    OvoRanker(const int* h_offsets, int n_groups_, bool tie_)
        : tie(tie_), n_groups(n_groups_) {
        for (int g = 0; g < n_groups; g++) {
            int sz = h_offsets[g + 1] - h_offsets[g];
            max_grp_size = std::max(max_grp_size, sz);
            if (sz <= OVO_MEDIUM_MAX)
                medium_max = std::max(medium_max, sz);
            else if (sz <= OVO_LARGE_MAX)
                large_max = std::max(large_max, sz);
            else if (tie)
                huge_ids.push_back(g);
        }
    }

    int n_huge() const {
        return (int)huge_ids.size();
    }

    size_t cub_bytes(size_t grp_items, int sb_cols) const {
        if (huge_ids.empty()) return 0;
        return cub_segmented_sortkeys_temp_bytes(
            checked_cub_items(grp_items, "OVO group sub-batch"),
            checked_int_product((size_t)n_huge(), (size_t)sb_cols,
                                "OVO HUGE segment count"));
    }

    // Per-stream scratch bytes, for stream-count budget clamps.
    size_t scratch_bytes(size_t grp_items, int sb_cols) const {
        if (!tie) return 0;
        size_t bytes = (size_t)sb_cols * sizeof(double);
        if (!huge_ids.empty())
            bytes += 2 * grp_items * sizeof(float) +
                     2 * (size_t)n_huge() * sb_cols * sizeof(int) +
                     cub_bytes(grp_items, sb_cols);
        return bytes;
    }

    void upload_huge_ids(RmmScratchPool& pool) {
        if (huge_ids.empty()) return;
        int* d = pool.alloc<int>(huge_ids.size());
        cuda_check(cudaMemcpy(d, huge_ids.data(), huge_ids.size() * sizeof(int),
                              cudaMemcpyHostToDevice),
                   "OVO HUGE group ids H2D");
        d_huge_ids = d;
    }

    OvoScratch alloc(RmmScratchPool& pool, size_t grp_items,
                     int sb_cols) const {
        OvoScratch sc;
        if (!tie) return sc;
        sc.ref_tie_sums = pool.alloc<double>(sb_cols);
        if (huge_ids.empty()) return sc;
        size_t n_seg = (size_t)n_huge() * sb_cols;
        sc.grp_nz = pool.alloc<float>(grp_items);
        sc.grp_sorted = pool.alloc<float>(grp_items);
        sc.seg_begins = pool.alloc<int>(n_seg);
        sc.seg_ends = pool.alloc<int>(n_seg);
        sc.cub_temp = pool.alloc<uint8_t>(cub_bytes(grp_items, sb_cols));
        return sc;
    }

    // Rank one column sub-batch: sorted ref [n_ref, sb_cols] F-order against
    // dense groups [n_all_grp, sb_cols] F-order split by d_offsets; results
    // go to [group * ld + col].
    void rank(const float* ref_sorted, const float* grp_dense,
              const int* d_offsets, const OvoScratch& sc, double* rank_sums,
              double* tie_corr, int ld, int n_ref, int n_all_grp, int sb_cols,
              cudaStream_t stream) const {
        if (!tie) {
            ovo_rank_dense_vs_ref_kernel<<<dim3(sb_cols, n_groups),
                                           UTIL_BLOCK_SIZE, 0, stream>>>(
                ref_sorted, grp_dense, d_offsets, rank_sums, n_ref, n_all_grp,
                ld);
            CUDA_CHECK_LAST_ERROR(ovo_rank_dense_vs_ref_kernel);
            return;
        }
        ref_tie_sum_kernel<<<sb_cols, UTIL_BLOCK_SIZE, 0, stream>>>(
            ref_sorted, sc.ref_tie_sums, n_ref, sb_cols);
        CUDA_CHECK_LAST_ERROR(ref_tie_sum_kernel);
        auto launch_band = [&](int lo, int hi, int band_max, int max_tpb) {
            if (band_max < lo) return;
            int padded = 1;
            while (padded < band_max) padded <<= 1;
            int tpb = std::max(WARP_SIZE, std::min(padded, max_tpb));
            ovo_rank_smem_kernel<<<dim3(sb_cols, n_groups), tpb,
                                   (size_t)padded * sizeof(float), stream>>>(
                ref_sorted, grp_dense, d_offsets, sc.ref_tie_sums, rank_sums,
                tie_corr, n_ref, n_all_grp, ld, lo, hi);
            CUDA_CHECK_LAST_ERROR(ovo_rank_smem_kernel);
        };
        launch_band(0, OVO_MEDIUM_MAX, medium_max, UTIL_BLOCK_SIZE);
        launch_band(OVO_MEDIUM_MAX + 1, OVO_LARGE_MAX, large_max,
                    MAX_THREADS_PER_BLOCK);
        if (huge_ids.empty()) return;

        int n_seg = checked_int_product((size_t)n_huge(), (size_t)sb_cols,
                                        "OVO HUGE segment count");
        dim3 grid(sb_cols, n_huge());
        ovo_compact_nonzeros_kernel<<<grid, UTIL_BLOCK_SIZE, 0, stream>>>(
            grp_dense, d_offsets, d_huge_ids, sc.grp_nz, sc.seg_begins,
            sc.seg_ends, n_all_grp, n_huge());
        CUDA_CHECK_LAST_ERROR(ovo_compact_nonzeros_kernel);
        int grp_items = checked_int_product((size_t)n_all_grp, (size_t)sb_cols,
                                            "OVO group sub-batch");
        cub_segmented_sortkeys(sc.cub_temp, cub_bytes(grp_items, sb_cols),
                               sc.grp_nz, sc.grp_sorted, grp_items, n_seg,
                               sc.seg_begins, sc.seg_ends, stream,
                               "OVO HUGE nonzero segmented sort");
        int tpb =
            round_up_to_warp(std::min(max_grp_size, MAX_THREADS_PER_BLOCK));
        ovo_rank_sorted_nz_kernel<<<grid, tpb, 0, stream>>>(
            ref_sorted, sc.grp_sorted, d_offsets, d_huge_ids, sc.seg_begins,
            sc.seg_ends, sc.ref_tie_sums, rank_sums, tie_corr, n_ref, ld,
            n_huge());
        CUDA_CHECK_LAST_ERROR(ovo_rank_sorted_nz_kernel);
    }
};
