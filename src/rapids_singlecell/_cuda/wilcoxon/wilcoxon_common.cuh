#pragma once

#include <algorithm>
#include <climits>
#include <cstdint>
#include <limits>
#include <vector>

#include <cuda_runtime.h>

#include <cub/device/device_segmented_radix_sort.cuh>

#include "../nb_types.h"
#include "../rmm_scratch.h"
#include "../sparse_extract/sparse_extract.cuh"
#include "../streaming/streaming.cuh"

constexpr int WARP_SIZE = 32;
constexpr int MAX_THREADS_PER_BLOCK = 512;
constexpr int N_STREAMS = 4;
constexpr int SUB_BATCH_COLS = 64;
// Scratch slots for warp-level reduction (one slot per warp, 32 warps max).
constexpr int WARP_REDUCE_BUF = 32;
// OVO tie bands: groups up to OVO_MEDIUM_MAX and up to OVO_LARGE_MAX rank in
// shared memory (nonzeros scanned up to OVO_MEDIUM_MAX, else bitonic-sorted);
// larger groups go through one CUB segmented sort.
constexpr int OVO_MEDIUM_MAX = 512;
constexpr int OVO_LARGE_MAX = 2500;
// Columns per OVO dense slab when a wide host CSR batch is ranked in chunks;
// the rest of the budget goes to the gathered batch (fewer host passes).
constexpr int OVO_SLAB_COLS = 256;
// Max per-batch nnz: a batch is sorted in one CUB segmented call (int32 item
// count) and addressed with int offsets, so it must stay below INT_MAX.
constexpr size_t SAFE_BATCH_NNZ = STREAMING_SAFE_BATCH_NNZ;

// Query CUB segmented-radix-sort scratch size. Float keys, int values/offsets.
static inline size_t cub_segmented_sortkeys_temp_bytes(int num_items,
                                                       int num_segments) {
    size_t bytes = 0;
    auto* fk = reinterpret_cast<float*>(1);
    auto* doff = reinterpret_cast<int*>(1);
    cuda_check(cub::DeviceSegmentedRadixSort::SortKeys(nullptr, bytes, fk, fk,
                                                       num_items, num_segments,
                                                       doff, doff + 1, 0, 32),
               "CUB SortKeys temp-size query");
    return bytes;
}

static inline size_t cub_segmented_sortpairs_temp_bytes(int num_items,
                                                        int num_segments) {
    size_t bytes = 0;
    auto* fk = reinterpret_cast<float*>(1);
    auto* v = reinterpret_cast<int*>(1);
    auto* off = reinterpret_cast<int*>(1);
    cuda_check(cub::DeviceSegmentedRadixSort::SortPairs(
                   nullptr, bytes, fk, fk, v, v, num_items, num_segments, off,
                   off + 1, 0, 32),
               "CUB SortPairs temp-size query");
    return bytes;
}

// Launch wrappers. begin/end offset arrays may be contiguous (off, off+1) or
// distinct (starts, ends).
static inline void cub_segmented_sortkeys(
    void* d_temp, size_t temp_bytes, const float* keys_in, float* keys_out,
    int num_items, int num_segments, const int* begin_offsets,
    const int* end_offsets, cudaStream_t stream, const char* what) {
    cuda_check(cub::DeviceSegmentedRadixSort::SortKeys(
                   d_temp, temp_bytes, keys_in, keys_out, num_items,
                   num_segments, begin_offsets, end_offsets, 0, 32, stream),
               what);
}

static inline void cub_segmented_sortpairs(
    void* d_temp, size_t temp_bytes, const float* keys_in, float* keys_out,
    const int* vals_in, int* vals_out, int num_items, int num_segments,
    const int* begin_offsets, const int* end_offsets, cudaStream_t stream,
    const char* what) {
    cuda_check(
        cub::DeviceSegmentedRadixSort::SortPairs(
            d_temp, temp_bytes, keys_in, keys_out, vals_in, vals_out, num_items,
            num_segments, begin_offsets, end_offsets, 0, 32, stream),
        what);
}

static inline int round_up_to_warp(int n) {
    int rounded = ((n + WARP_SIZE - 1) / WARP_SIZE) * WARP_SIZE;
    return (rounded < MAX_THREADS_PER_BLOCK) ? rounded : MAX_THREADS_PER_BLOCK;
}

// Sum `v` across the 32 lanes of a warp via shuffle-down; result on lane 0.
__device__ __forceinline__ double warp_reduce_sum(double v) {
#pragma unroll
    for (int off = 16; off > 0; off >>= 1)
        v += __shfl_down_sync(0xffffffff, v, off);
    return v;
}

// Block-wide sum of `val` using one shared double per warp.
// Result is returned on thread 0; other threads get 0.0.
__device__ __forceinline__ double wilcoxon_block_sum(double val,
                                                     double* warp_buf) {
    val = warp_reduce_sum(val);
    int lane = threadIdx.x & 31;
    int wid = threadIdx.x >> 5;
    if (lane == 0) warp_buf[wid] = val;
    __syncthreads();
    if (threadIdx.x < 32) {
        double v = (threadIdx.x < ((blockDim.x + 31) >> 5))
                       ? warp_buf[threadIdx.x]
                       : 0.0;
        return warp_reduce_sum(v);
    }
    return 0.0;
}

// Final tie-correction factor: 1 - sum(t^3 - t) / (n^3 - n), or 1.0 when the
// ranking population n_total is too small for a correction.
__device__ __forceinline__ double finalize_tie_corr(int n_total,
                                                    double tie_sum) {
    double dn = (double)n_total;
    double denom = dn * dn * dn - dn;
    return (denom > 0.0) ? (1.0 - tie_sum / denom) : 1.0;
}

// Sorted-array bounds over [lo, hi): lower is first >= v, upper first > v.
// Advanced `lo` exploits monotonic strides; global/shared arrays both work.

__device__ __forceinline__ int sorted_lower_bound(const float* arr, int lo,
                                                  int hi, float v) {
    while (lo < hi) {
        int m = lo + ((hi - lo) >> 1);
        if (arr[m] < v)
            lo = m + 1;
        else
            hi = m;
    }
    return lo;
}

__device__ __forceinline__ int sorted_upper_bound(const float* arr, int lo,
                                                  int hi, float v) {
    while (lo < hi) {
        int m = lo + ((hi - lo) >> 1);
        if (arr[m] <= v)
            lo = m + 1;
        else
            hi = m;
    }
    return lo;
}

// One tie run's t^3 - t (zero for t <= 1).
// Value a stored entry adds to the group sums: x itself, or expm1(scale * x)
// for means in expression space (inverts log1p with base e^scale).
__device__ __forceinline__ double stat_value(double x, double expm1_scale) {
    return expm1_scale != 0.0 ? expm1(expm1_scale * x) : x;
}

__device__ __forceinline__ double tie3(double t) {
    return t * t * t - t;
}
