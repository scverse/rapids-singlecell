#pragma once

#include <algorithm>
#include <cstdint>
#include <limits>
#include <vector>

#include <cuda_runtime.h>

#include <cub/device/device_segmented_radix_sort.cuh>

#include "../nb_types.h"     // for CUDA_CHECK_LAST_ERROR
#include "../rmm_scratch.h"  // rmm_allocate, RmmScratchPool, ScopedCudaBuffer
#include "../sparse_extract/sparse_extract.cuh"  // csr_extract_dense* kernels
#include "../streaming/streaming.cuh"

constexpr int WARP_SIZE = 32;
constexpr int MAX_THREADS_PER_BLOCK = 512;
constexpr int N_STREAMS = 4;
constexpr int SUB_BATCH_COLS = 64;
constexpr int BEGIN_BIT = 0;
constexpr int END_BIT = 32;
// Scratch slots for warp-level reduction (one slot per warp, 32 warps max).
constexpr int WARP_REDUCE_BUF = 32;
// MEDIUM band cap: groups up to this size use unsorted O(n^2) in-group-count
// rank (no smem sort). Tier dispatch: make_ovo_tier_plan.
constexpr int OVO_MEDIUM_MAX = 512;
// LARGE band cap (fused smem-sort kernel); beyond it -> HUGE (CUB segmented
// sort).
constexpr int OVO_LARGE_MAX = 2500;
// Per-stream dense slab budget (f32 items): 128M*4B=512MB slab + 512MB sorted
// copy ≈ 1GB/stream. Sub-batching keeps (n_g * eff_sb_cols) <= this.
constexpr size_t GROUP_DENSE_BUDGET_ITEMS = 128 * 1024 * 1024;

// Budget-aware OVO-host pack sizing. Per-stream device scratch that does NOT
// scale with pack nnz: dense + sorted slabs (each <= GROUP_DENSE_BUDGET) plus
// rank/tie/seg/cub headroom. Reserved per target stream when bounding pack nnz
// so the resident packs + sorted ref cache fit device free.
constexpr size_t OVO_PACK_FIXED_PER_STREAM =
    4 * GROUP_DENSE_BUDGET_ITEMS * sizeof(float);  // ~2 GB
// Floor for the budget-derived pack-nnz cap: avoid pathological over-splitting
// into thousands of tiny packs when device memory is very tight.
constexpr size_t OVO_MIN_PACK_NNZ = 64 * 1024 * 1024;  // 64M nnz

// Host->device staging-ring slot cap (nnz). Bounds the page-locked footprint:
// a pack's device buffer is filled in row-blocks of <= this many nonzeros, so
// the cold pin stays small instead of seconds when pack nnz is large. 32M nnz
// (128MB vals + 128MB cols/slot) is the joint sweet spot across scales: it
// crushes the whole-pack pin at 2M (~2.7x) yet stays well clear of a sharp
// large-scale slowdown seen with much smaller blocks at multi-billion nnz.
constexpr size_t STAGE_RING_NNZ_CAP = 32 * 1024 * 1024;

// Query CUB segmented-radix-sort scratch size. Float keys, int values/offsets.
static inline size_t cub_segmented_sortkeys_temp_bytes(int num_items,
                                                       int num_segments) {
    size_t bytes = 0;
    auto* fk = reinterpret_cast<float*>(1);
    auto* doff = reinterpret_cast<int*>(1);
    cuda_check(cub::DeviceSegmentedRadixSort::SortKeys(
                   nullptr, bytes, fk, fk, num_items, num_segments, doff,
                   doff + 1, BEGIN_BIT, END_BIT),
               "CUB SortKeys temp-size query");
    return bytes;
}

template <typename ValT = int>
static inline size_t cub_segmented_sortpairs_temp_bytes(int num_items,
                                                        int num_segments) {
    size_t bytes = 0;
    auto* fk = reinterpret_cast<float*>(1);
    auto* v = reinterpret_cast<ValT*>(1);
    auto* off = reinterpret_cast<int*>(1);
    cuda_check(cub::DeviceSegmentedRadixSort::SortPairs(
                   nullptr, bytes, fk, fk, v, v, num_items, num_segments, off,
                   off + 1, BEGIN_BIT, END_BIT),
               "CUB SortPairs temp-size query");
    return bytes;
}

// Launch wrappers. begin/end offset arrays may be contiguous (off, off+1) or
// distinct (starts, ends).
static inline void cub_segmented_sortkeys(
    void* d_temp, size_t temp_bytes, const float* keys_in, float* keys_out,
    int num_items, int num_segments, const int* begin_offsets,
    const int* end_offsets, cudaStream_t stream, const char* what) {
    cuda_check(
        cub::DeviceSegmentedRadixSort::SortKeys(
            d_temp, temp_bytes, keys_in, keys_out, num_items, num_segments,
            begin_offsets, end_offsets, BEGIN_BIT, END_BIT, stream),
        what);
}

template <typename ValT = int>
static inline void cub_segmented_sortpairs(
    void* d_temp, size_t temp_bytes, const float* keys_in, float* keys_out,
    const ValT* vals_in, ValT* vals_out, int num_items, int num_segments,
    const int* begin_offsets, const int* end_offsets, cudaStream_t stream,
    const char* what) {
    cuda_check(cub::DeviceSegmentedRadixSort::SortPairs(
                   d_temp, temp_bytes, keys_in, keys_out, vals_in, vals_out,
                   num_items, num_segments, begin_offsets, end_offsets,
                   BEGIN_BIT, END_BIT, stream),
               what);
}

// Universal CUDA static per-block shared-memory floor; safe fallback if the
// device query fails.
constexpr size_t WILCOXON_FALLBACK_SMEM_PER_BLOCK = 48 * 1024;

// CRITICAL: per-block smem limit (cached per device) powering every smem/gmem
// and tier decision (ovr_smem_config, sparse_ovr_smem_config,
// cast_accumulate_smem_config, make_ovo_tier_plan). DO NOT hardcode a smem
// value in place of this call -- gmem-fallback thresholds (e.g. sparse OVR
// ~3056 groups) auto-scale with the GPU. Falls back to 48 KB if the query
// fails.
static inline size_t wilcoxon_max_smem_per_block() {
    int device = 0;
    if (cudaGetDevice(&device) != cudaSuccess) {
        return WILCOXON_FALLBACK_SMEM_PER_BLOCK;
    }
    static thread_local int cached_dev = -1;
    static thread_local size_t cached_smem = 0;
    if (device == cached_dev) return cached_smem;
    int max_smem = 0;
    if (cudaDeviceGetAttribute(&max_smem, cudaDevAttrMaxSharedMemoryPerBlock,
                               device) != cudaSuccess) {
        return WILCOXON_FALLBACK_SMEM_PER_BLOCK;
    }
    cached_dev = device;
    cached_smem = (size_t)max_smem;
    return cached_smem;
}

// Max per-batch nnz: a batch is sorted in one CUB segmented call (int32 item
// count) and addressed with int offsets, so it must stay below INT_MAX.
constexpr size_t SAFE_BATCH_NNZ = STREAMING_SAFE_BATCH_NNZ;

static inline int round_up_to_warp(int n) {
    int rounded = ((n + WARP_SIZE - 1) / WARP_SIZE) * WARP_SIZE;
    return (rounded < MAX_THREADS_PER_BLOCK) ? rounded : MAX_THREADS_PER_BLOCK;
}

/** Per-row stats codes for a pack of K groups. From pack_grp_offsets (size K+1,
 *  relative to pack start), write stats_codes[r] = base_slot + group_idx(r) via
 *  binary search over the K+1 offsets. */
__global__ void fill_pack_stats_codes_kernel(
    const int* __restrict__ pack_grp_offsets, int* __restrict__ stats_codes,
    int K, int base_slot) {
    int r = blockIdx.x * blockDim.x + threadIdx.x;
    int pack_n_rows = pack_grp_offsets[K];
    if (r >= pack_n_rows) return;
    int lo = 0, hi = K;
    while (lo < hi) {
        int m = lo + ((hi - lo) >> 1);
        if (pack_grp_offsets[m + 1] <= r)
            lo = m + 1;
        else
            hi = m;
    }
    stats_codes[r] = base_slot + lo;
}

// Per-group stats over an already-compact CSR (accumulate half of the mapped
// gather kernel, decoupled for host-staged data). slot = stats_codes[r] or
// fixed_slot; slot outside [0,n_groups_stats) is skipped.
__global__ void csr_compact_accumulate_kernel(
    const float* __restrict__ d_data_f32, const int* __restrict__ d_indices,
    const int* __restrict__ d_indptr, const int* __restrict__ d_stats_codes,
    int fixed_slot, double* __restrict__ group_sums,
    double* __restrict__ group_nnz, int n_target_rows, int n_cols,
    int n_groups_stats, bool compute_sums, bool compute_nnz) {
    int r = blockIdx.x;
    if (r >= n_target_rows) return;
    int slot = (d_stats_codes != nullptr) ? d_stats_codes[r] : fixed_slot;
    if (slot < 0 || slot >= n_groups_stats) return;
    int rs = d_indptr[r];
    int re = d_indptr[r + 1];
    for (int i = rs + threadIdx.x; i < re; i += blockDim.x) {
        int c = d_indices[i];
        double v = (double)d_data_f32[i];
        if (compute_sums) atomicAdd(&group_sums[(size_t)slot * n_cols + c], v);
        if (compute_nnz && v != 0.0)
            atomicAdd(&group_nnz[(size_t)slot * n_cols + c], 1.0);
    }
}
