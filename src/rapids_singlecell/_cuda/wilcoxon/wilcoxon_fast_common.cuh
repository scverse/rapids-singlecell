#pragma once

#include <algorithm>
#include <cstdint>
#include <limits>
#include <stdexcept>
#include <string>
#include <memory>
#include <thread>
#include <vector>

#include <cuda_runtime.h>

#include <cub/device/device_segmented_radix_sort.cuh>

#include "../nb_types.h"     // for CUDA_CHECK_LAST_ERROR
#include "../rmm_scratch.h"  // rmm_allocate, RmmScratchPool, ScopedCudaBuffer
#include "../sparse_extract/sparse_extract.cuh"  // csr_extract_dense* kernels

// Host thread count for CPU-side CSR passes: hardware concurrency, capped.
static inline int host_worker_count() {
    unsigned hw = std::thread::hardware_concurrency();
    return (int)std::min<unsigned>(hw ? hw : 4u, 32u);
}

// Run fn(chunk, r0, r1) over a partition of [0, n); `chunk` = 0-based worker
// index. fn runs concurrently: read-only shared state, disjoint output ranges
// (keyed by chunk or [r0,r1)). Returns chunks used; serial for small n.
template <typename F>
static inline int host_parallel_chunks(int n, F fn) {
    if (n <= 0) return 0;
    int n_threads = host_worker_count();
    if (n_threads <= 1 || n < 4096) {
        fn(0, 0, n);
        return 1;
    }
    int chunk = (n + n_threads - 1) / n_threads;
    std::vector<std::thread> pool;
    pool.reserve(n_threads);
    for (int t = 0; t < n_threads; t++) {
        int r0 = t * chunk;
        if (r0 >= n) break;
        int r1 = std::min(n, r0 + chunk);
        pool.emplace_back([&fn, t, r0, r1]() { fn(t, r0, r1); });
    }
    int used = (int)pool.size();
    for (std::thread& th : pool) th.join();
    return used;
}

// Run fn(r0, r1) over a partition of [0, n) across hardware threads (serial for
// small n). Concurrent: read-only shared state, disjoint output ranges.
template <typename F>
static inline void host_parallel_ranges(int n, F fn) {
    host_parallel_chunks(n, [&fn](int, int r0, int r1) { fn(r0, r1); });
}

constexpr int WARP_SIZE = 32;
constexpr int MAX_THREADS_PER_BLOCK = 512;
constexpr int N_STREAMS = 4;
constexpr int SUB_BATCH_COLS = 64;
constexpr int BEGIN_BIT = 0;
constexpr int END_BIT = 32;
// Default thread-per-block for utility kernels.
constexpr int UTIL_BLOCK_SIZE = 256;
// Scratch slots for warp-level reduction (one slot per warp, 32 warps max).
constexpr int WARP_REDUCE_BUF = 32;

// Stream-count clamps: never use more streams than column batches, nor more
// than the per-stream memory budget allows.
static inline int clamp_streams_by_cols(int n_cols, int sub_batch_cols) {
    int n = N_STREAMS;
    if (n_cols < n * sub_batch_cols)
        n = (n_cols + sub_batch_cols - 1) / sub_batch_cols;
    return n;
}

static inline int clamp_streams_by_budget(int n_streams,
                                          size_t per_stream_bytes,
                                          size_t budget) {
    while (n_streams > 1 && (size_t)n_streams * per_stream_bytes > budget)
        n_streams--;
    return n_streams;
}

// Scatter a [rows, sb_cols] device sub-batch (row-major doubles, src stride
// sb_cols) into `dst` (stride n_cols). `dst` must point at the dest column
// offset (e.g. out + col).
static inline void scatter_cols_2d(double* dst, const double* src, int rows,
                                   int n_cols, int sb_cols,
                                   cudaStream_t stream) {
    cudaMemcpy2DAsync(dst, n_cols * sizeof(double), src,
                      sb_cols * sizeof(double), sb_cols * sizeof(double), rows,
                      cudaMemcpyDeviceToDevice, stream);
}
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

static inline int checked_cub_items(size_t count, const char* context) {
    if (count > (size_t)std::numeric_limits<int>::max()) {
        throw std::runtime_error(std::string(context) +
                                 " exceeds CUB int item limit");
    }
    return (int)count;
}

static inline int checked_int_span(size_t count, const char* context) {
    if (count > (size_t)std::numeric_limits<int>::max()) {
        throw std::runtime_error(std::string(context) +
                                 " exceeds int32 offset limit");
    }
    return (int)count;
}

static inline int checked_int_product(size_t a, size_t b, const char* context) {
    if (a != 0 && b > (size_t)std::numeric_limits<int>::max() / a) {
        throw std::runtime_error(std::string(context) +
                                 " exceeds int32 item limit");
    }
    return (int)(a * b);
}

// Precompute per-batch CSC column offsets rebased to each batch's ptr_start,
// laid out [n_batches][sub_batch_cols+1], upload once (from `pool`). Avoids a
// per-batch H2D from a transient host buffer.
template <typename IndptrT>
static inline int* precompute_csc_batch_offsets(const IndptrT* h_indptr,
                                                int n_cols, int sub_batch_cols,
                                                int n_batches,
                                                RmmScratchPool& pool,
                                                const char* what) {
    std::vector<int> h_all_offsets((size_t)n_batches * (sub_batch_cols + 1), 0);
    for (int b = 0; b < n_batches; b++) {
        int col_start = b * sub_batch_cols;
        int rem = n_cols - col_start;
        int sb = (sub_batch_cols < rem) ? sub_batch_cols : rem;
        IndptrT ptr_start = h_indptr[col_start];
        int* off = &h_all_offsets[(size_t)b * (sub_batch_cols + 1)];
        for (int i = 0; i <= sb; i++)
            off[i] = checked_int_span(
                (size_t)(h_indptr[col_start + i] - ptr_start), what);
    }
    int* d_all_offsets =
        pool.alloc<int>((size_t)n_batches * (sub_batch_cols + 1));
    cudaMemcpy(d_all_offsets, h_all_offsets.data(),
               h_all_offsets.size() * sizeof(int), cudaMemcpyHostToDevice);
    return d_all_offsets;
}

// Max per-batch nnz: a batch is sorted in one CUB segmented call (int32 item
// count) and addressed with int offsets, so it must stay below INT_MAX.
constexpr size_t SAFE_BATCH_NNZ = 2000000000;  // < INT_MAX

// Halve sub_batch_cols until the densest window holds <= cap nonzeros, keeping
// every batch's nnz within int32 for CUB and bounding per-stream transpose/sort
// scratch. col_nnz(i) = nnz of column i. Worst case returns 1 (single column,
// nnz <= n_rows).
template <typename ColNnz>
static inline int cap_sub_batch_by_nnz(int n_cols, int sub_batch_cols,
                                       size_t cap, ColNnz col_nnz) {
    if (cap < 1) cap = 1;
    auto max_window = [&](int s) {
        size_t mx = 0;
        for (int c = 0; c < n_cols; c += s) {
            int e = std::min(c + s, n_cols);
            size_t sum = 0;
            for (int i = c; i < e; i++) sum += col_nnz(i);
            if (sum > mx) mx = sum;
        }
        return mx;
    };
    while (sub_batch_cols > 1 && max_window(sub_batch_cols) > cap)
        sub_batch_cols = (sub_batch_cols + 1) / 2;
    return sub_batch_cols;
}

// RAII guard for cudaHostRegister: unregisters on scope exit (incl. exception
// unwind), preventing leaked host pinning on stream-sync failures.
struct HostRegisterGuard {
    void* ptr = nullptr;

    HostRegisterGuard() = default;
    HostRegisterGuard(void* p, size_t bytes, unsigned int flags = 0,
                      bool best_effort = false) {
        if (p && bytes > 0) {
            cudaError_t err = cudaHostRegister(p, bytes, flags);
            if (err != cudaSuccess) {
                // Already-registered = owned elsewhere; use it without
                // unregistering. Other failures make mapped reads unsafe, so
                // surface them -- unless best_effort (pin is only a speedup;
                // unpinned H2D still works).
                if (err == cudaErrorHostMemoryAlreadyRegistered ||
                    best_effort) {
                    cudaGetLastError();  // clear sticky error flag
                } else {
                    throw std::runtime_error(
                        std::string("cudaHostRegister failed (") +
                        std::to_string((size_t)bytes) +
                        " bytes, flags=" + std::to_string(flags) +
                        "): " + cudaGetErrorString(err));
                }
            } else {
                ptr = p;
            }
        }
    }
    ~HostRegisterGuard() {
        if (ptr) cudaHostUnregister(ptr);
    }
    HostRegisterGuard(const HostRegisterGuard&) = delete;
    HostRegisterGuard& operator=(const HostRegisterGuard&) = delete;
    HostRegisterGuard(HostRegisterGuard&& other) noexcept : ptr(other.ptr) {
        other.ptr = nullptr;
    }
    HostRegisterGuard& operator=(HostRegisterGuard&& other) noexcept {
        if (this != &other) {
            if (ptr) cudaHostUnregister(ptr);
            ptr = other.ptr;
            other.ptr = nullptr;
        }
        return *this;
    }
};

// RAII for CUDA streams/events: reclaim on every path (incl. exception unwind).
// Stream dtor SYNCHRONIZES before destroying. CRITICAL ordering: declare the
// RmmScratchPool BEFORE these guards so streams (destroyed first) drain
// in-flight kernels before the pool (destroyed last) frees the scratch they
// read.
struct ScopedCudaStream {
    cudaStream_t stream = nullptr;

    ScopedCudaStream() = default;
    explicit ScopedCudaStream(unsigned int flags) {
        cuda_check(cudaStreamCreateWithFlags(&stream, flags),
                   "cudaStreamCreateWithFlags");
    }
    ~ScopedCudaStream() {
        if (stream) {
            cudaStreamSynchronize(stream);  // drain before teardown
            cudaStreamDestroy(stream);
        }
    }
    operator cudaStream_t() const {
        return stream;
    }
    cudaStream_t get() const {
        return stream;
    }
    ScopedCudaStream(const ScopedCudaStream&) = delete;
    ScopedCudaStream& operator=(const ScopedCudaStream&) = delete;
};

struct ScopedCudaStreams {
    std::vector<cudaStream_t> streams;

    // `flags` is explicit so call sites keep their original stream semantics.
    ScopedCudaStreams(int n, unsigned int flags) {
        streams.reserve(n > 0 ? (size_t)n : 0);
        for (int i = 0; i < n; ++i) {
            cudaStream_t s = nullptr;
            cudaError_t err = cudaStreamCreateWithFlags(&s, flags);
            if (err != cudaSuccess) {
                // dtor won't run on ctor throw; reclaim what we made.
                for (cudaStream_t prev : streams) {
                    cudaStreamSynchronize(prev);
                    cudaStreamDestroy(prev);
                }
                throw std::runtime_error(
                    std::string("cudaStreamCreateWithFlags failed: ") +
                    cudaGetErrorString(err));
            }
            streams.push_back(s);
        }
    }
    ~ScopedCudaStreams() {
        for (cudaStream_t s : streams) {
            if (!s) continue;
            cudaStreamSynchronize(s);  // drain before teardown
            cudaStreamDestroy(s);
        }
    }
    cudaStream_t operator[](int i) const {
        return streams[i];
    }
    int size() const {
        return (int)streams.size();
    }
    ScopedCudaStreams(const ScopedCudaStreams&) = delete;
    ScopedCudaStreams& operator=(const ScopedCudaStreams&) = delete;
};

// Drain every stream, surfacing the first async error with a context label.
static inline void sync_streams(const ScopedCudaStreams& streams,
                                const char* what) {
    for (int i = 0; i < streams.size(); ++i) {
        cudaError_t err = cudaStreamSynchronize(streams[i]);
        if (err != cudaSuccess)
            throw std::runtime_error(std::string("CUDA error in ") + what +
                                     ": " + cudaGetErrorString(err));
    }
}

struct ScopedCudaEvent {
    cudaEvent_t event = nullptr;

    ScopedCudaEvent() = default;
    explicit ScopedCudaEvent(unsigned int flags) {
        cuda_check(cudaEventCreateWithFlags(&event, flags),
                   "cudaEventCreateWithFlags");
    }
    ~ScopedCudaEvent() {
        if (event) cudaEventDestroy(event);
    }
    void record(cudaStream_t stream) {
        cuda_check(cudaEventRecord(event, stream), "cudaEventRecord");
    }
    cudaEvent_t get() const {
        return event;
    }
    ScopedCudaEvent(const ScopedCudaEvent&) = delete;
    ScopedCudaEvent& operator=(const ScopedCudaEvent&) = delete;
};

static inline int round_up_to_warp(int n) {
    int rounded = ((n + WARP_SIZE - 1) / WARP_SIZE) * WARP_SIZE;
    return (rounded < MAX_THREADS_PER_BLOCK) ? rounded : MAX_THREADS_PER_BLOCK;
}

// Per-stream pinned host staging (f32 vals + int32 cols) with a per-slot event,
// so a CPU gather into slot s overlaps GPU compute: wait(s) blocks only until
// slot s's prior H2D drained, not the whole pipeline.
struct HostStagingRing {
    std::vector<std::unique_ptr<float[]>> vals;
    std::vector<std::unique_ptr<int[]>> cols;
    std::vector<HostRegisterGuard> pin_v, pin_c;
    std::vector<cudaEvent_t> evt;
    std::vector<char> used;
    HostStagingRing(int n_streams, size_t nnz)
        : vals(n_streams),
          cols(n_streams),
          pin_v(n_streams),
          pin_c(n_streams),
          evt(n_streams, nullptr),
          used(n_streams, 0) {
        size_t n = nnz ? nnz : 1;
        for (int s = 0; s < n_streams; s++) {
            vals[s].reset(new float[n]);
            cols[s].reset(new int[n]);
            pin_v[s] = HostRegisterGuard(vals[s].get(), n * sizeof(float));
            pin_c[s] = HostRegisterGuard(cols[s].get(), n * sizeof(int));
            cuda_check(
                cudaEventCreateWithFlags(&evt[s], cudaEventDisableTiming),
                "HostStagingRing event create");
        }
    }
    ~HostStagingRing() {
        for (cudaEvent_t e : evt)
            if (e) cudaEventDestroy(e);
    }
    void wait(int s) {
        if (used[s])
            cuda_check(cudaEventSynchronize(evt[s]), "HostStagingRing reuse");
    }
    void record(int s, cudaStream_t stream) {
        cuda_check(cudaEventRecord(evt[s], stream), "HostStagingRing record");
        used[s] = true;
    }
    HostStagingRing(const HostStagingRing&) = delete;
    HostStagingRing& operator=(const HostStagingRing&) = delete;
};

/** Fill linear segment offsets [0, stride, ..., n_segments*stride] on-device.
 */
__global__ void fill_linear_offsets_kernel(int* __restrict__ out,
                                           int n_segments, int stride) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i <= n_segments) out[i] = i * stride;
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

/** Rebase a slice of indptr: out[i] = indptr[col+i] - indptr[col]. Grid-strided
 *  (arbitrary `count`). Templated so 64-bit global indptrs produce 32-bit
 *  pack-local indptrs (per-pack nnz fits int32 via the memory budget). */
template <typename IdxIn, typename IdxOut>
__global__ void rebase_indptr_kernel(const IdxIn* __restrict__ indptr,
                                     IdxOut* __restrict__ out, int col,
                                     int count) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < count) out[i] = (IdxOut)(indptr[col + i] - indptr[col]);
}

// Threaded host gather of selected rows into compact staging (f32 vals + int32
// cols) at disjoint per-row offsets (compact_indptr - base) -> race-free.
// No-pin alternative to the mapped gather kernel: only the compacted slice
// crosses the bus.
template <typename InT, typename IndexT, typename IndptrT, typename CompactT>
static void host_gather_rows_compact(const InT* h_data, const IndexT* h_indices,
                                     const IndptrT* h_indptr,
                                     const int* row_ids,
                                     const CompactT* compact_indptr,
                                     CompactT base, int n_target,
                                     float* stage_vals, int* stage_cols) {
    host_parallel_ranges(n_target, [&](int i0, int i1) {
        for (int i = i0; i < i1; i++) {
            int r = row_ids[i];
            IndptrT rs = h_indptr[r];
            int nnz = (int)(h_indptr[r + 1] - rs);
            size_t ds = (size_t)(compact_indptr[i] - base);
            for (int k = 0; k < nnz; k++) {
                stage_vals[ds + k] = (float)h_data[rs + k];
                stage_cols[ds + k] = (int)h_indices[rs + k];
            }
        }
    });
}

// Threaded host cast-copy of a contiguous nnz slice into staging (f32 + int32).
// CSC analogue of host_gather_rows_compact: contiguous column batch, no gather.
// nnz fits int32 (batch-bounded).
template <typename InT, typename IndexT>
static void host_cast_copy_slice(const InT* h_data, const IndexT* h_indices,
                                 size_t start, int nnz, float* stage_vals,
                                 int* stage_cols) {
    host_parallel_ranges(nnz, [&](int k0, int k1) {
        for (int k = k0; k < k1; k++) {
            stage_vals[k] = (float)h_data[start + k];
            stage_cols[k] = (int)h_indices[start + k];
        }
    });
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

/** Fill linear segment offsets [0, stride, ...] on the supplied stream (avoids
 *  serializing multi-stream pipelines). */
static inline void upload_linear_offsets(int* d_offsets, int n_segments,
                                         int stride, cudaStream_t stream) {
    int count = n_segments + 1;
    int blk = (count + UTIL_BLOCK_SIZE - 1) / UTIL_BLOCK_SIZE;
    fill_linear_offsets_kernel<<<blk, UTIL_BLOCK_SIZE, 0, stream>>>(
        d_offsets, n_segments, stride);
    CUDA_CHECK_LAST_ERROR(fill_linear_offsets_kernel);
}
