#pragma once

#include <algorithm>
#include <cstdint>
#include <limits>
#include <stdexcept>
#include <string>
#include <thread>
#include <vector>

#include <cuda_runtime.h>

#include <cub/device/device_segmented_radix_sort.cuh>

#include "../nb_types.h"     // for CUDA_CHECK_LAST_ERROR
#include "../rmm_scratch.h"  // rmm_allocate, RmmScratchPool, ScopedCudaBuffer

// Host thread count for CPU-side CSR passes: hardware concurrency, capped.
static inline int host_worker_count() {
    unsigned hw = std::thread::hardware_concurrency();
    return (int)std::min<unsigned>(hw ? hw : 4u, 32u);
}

// Run fn(chunk, r0, r1) over a contiguous partition of [0, n); `chunk` is the
// 0-based worker index (for per-thread scratch). fn runs concurrently, so it
// must only read shared state and write disjoint output ranges (keyed by chunk
// or by [r0,r1)). Returns the number of chunks used. Serial for small n.
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

// Run fn(r0, r1) over a contiguous partition of [0, n) across hardware threads
// (serial for small n). fn is invoked concurrently, so it must only read shared
// state and write disjoint output ranges. Used for host-side CSR gathers.
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
// WARP band: warp-per-(col,group) fused kernel. Each warp sorts+ranks one
// pair entirely in registers (warp-shuffle bitonic, no smem, no __syncthreads).
// Blocks pack 8 warps to amortise launch overhead. Fast route for
// perturbation-style workloads where most groups have a few dozen cells.
constexpr int OVO_WARP_MAX = 32;
// SMALL band: groups slightly larger than one warp. One compact smem sort
// block per (col, group), avoiding the heavier MEDIUM-band in-group scan.
constexpr int OVO_SMALL_MAX = 64;
// MEDIUM band: unsorted direct-rank kernel. Avoiding a full smem bitonic sort
// wins here despite the O(n^2) in-group count.
constexpr int OVO_MEDIUM_MAX = 512;
// Max group size for the fused smem-sort rank kernel (the LARGE band).
// Beyond this, fall back to the HUGE band: CUB segmented sort + rank kernel.
constexpr int OVO_LARGE_MAX = 2500;
// Per-stream dense slab budget (float32 items). Sub-batching keeps
// (n_g × eff_sb_cols) ≤ this. 128M × 4B = 512 MB slab + same for sorted copy
// ≈ 1 GB / stream. Bigger = fewer launches; smaller = less per-stream memory.
constexpr size_t GROUP_DENSE_BUDGET_ITEMS = 128 * 1024 * 1024;

// Query CUB device-segmented-radix-sort scratch size with a dummy launch.
// Every Wilcoxon sort uses float keys and (for SortPairs) int values/offsets.
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

// Launch wrappers for the queries above. begin/end offset arrays may be
// contiguous (off, off + 1) or distinct (starts, ends).
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

// CRITICAL device-limit query that powers every smem/gmem and tier decision.
// Returns the per-block shared-memory limit (cached per device). Consumed by
// ovr_smem_config, sparse_ovr_smem_config, cast_accumulate_smem_config, and
// make_ovo_tier_plan to decide when accumulators/sorts no longer fit in smem
// and must fall back to global memory or CUB. DO NOT hardcode a smem value in
// place of this call -- the gmem-fallback thresholds (e.g. sparse OVR ~3056
// groups) auto-scale with the GPU because of it; falls back to 48 KB if the
// query fails.
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

// Largest per-batch nonzero count we let a column batch reach. A batch is
// sorted in a single CUB segmented call (int32 item count) and addressed with
// int offsets, so it must stay below INT_MAX with margin.
constexpr size_t SAFE_BATCH_NNZ = 2000000000;  // < INT_MAX

// Shrink a column sub-batch (halving) until the densest contiguous window of
// `sub_batch_cols` columns holds <= cap nonzeros, keeping every batch's nnz
// within int32 for CUB and bounding the per-stream transpose/sort scratch.
// `col_nnz(i)` returns the nonzero count of column i. Worst case returns 1
// (a single column, whose nnz is <= n_rows).
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

// ---------------------------------------------------------------------------
// RAII guard for cudaHostRegister.  Unregisters on scope exit even when an
// exception unwinds — prevents leaked host pinning on stream-sync failures.
// ---------------------------------------------------------------------------
struct HostRegisterGuard {
    void* ptr = nullptr;

    HostRegisterGuard() = default;
    HostRegisterGuard(void* p, size_t bytes, unsigned int flags = 0) {
        if (p && bytes > 0) {
            cudaError_t err = cudaHostRegister(p, bytes, flags);
            if (err != cudaSuccess) {
                // Already-registered memory belongs to another owner; use it
                // without unregistering here. Other failures mean mapped reads
                // would be unsafe, so surface them immediately.
                if (err == cudaErrorHostMemoryAlreadyRegistered) {
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

// RAII for CUDA streams/events: reclaim on every path (incl. exception unwind),
// fixing the leak when a throwing call skips a trailing manual destroy. The
// stream dtor SYNCHRONIZES before destroying. Convention: declare the
// RmmScratchPool BEFORE these guards so the streams (destroyed first) drain
// their in-flight kernels before the pool (destroyed last) frees the scratch
// those kernels read -- safe on the normal and exception-unwind paths alike.
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

/** Fill linear segment offsets [0, stride, 2*stride, ..., n_segments*stride]
 *  on-device.  One thread per output slot. */
__global__ void fill_linear_offsets_kernel(int* __restrict__ out,
                                           int n_segments, int stride) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i <= n_segments) out[i] = i * stride;
}

/** Fill per-row stats codes for a pack of K groups.
 *  Given pack_grp_offsets (size K+1, relative to pack start), write
 *  stats_codes[r] = base_slot + group_idx_of_row_r for r in [0, pack_n_rows).
 *  Binary search within the K+1 offsets. */
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

/** Rebase a slice of indptr: out[i] = indptr[col + i] - indptr[col].
 *  Grid-strided: supports arbitrary `count` (no single-block thread limit).
 *  Templated so that 64-bit global indptrs can produce 32-bit pack-local
 *  indptrs (per-pack nnz always fits in int32 thanks to the memory budget).
 */
template <typename IdxIn, typename IdxOut>
__global__ void rebase_indptr_kernel(const IdxIn* __restrict__ indptr,
                                     IdxOut* __restrict__ out, int col,
                                     int count) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < count) out[i] = (IdxOut)(indptr[col + i] - indptr[col]);
}

/** Fused gather + cast-to-float32 + stats accumulation, reading from mapped
 *  pinned host memory.  Block-per-row; threads in the block cooperate on the
 *  row's nnz.  Each nnz is read from host over PCIe exactly once — no
 *  intermediate native-dtype GPU buffer, no second GPU pass.
 *
 *  h_data / h_indices: device-accessible pointers into mapped pinned host
 *                     memory (cudaHostRegisterMapped).
 *  d_indptr_full: full-matrix indptr on device.
 *  d_row_ids:    rows to gather (size n_target_rows).
 *  d_out_indptr: pre-computed compacted indptr, size n_target_rows+1 with
 *                out_indptr[i+1] - out_indptr[i] equal to the source row's
 *                nnz.
 *
 *  Slot dispatch:
 *    d_stats_codes != nullptr → slot = d_stats_codes[r]; otherwise slot =
 *    fixed_slot (used for the Ref phase where every row maps to the same
 *    slot).  slot ∉ [0, n_groups_stats) skips accumulation.
 */
template <typename InT, typename IndexT, typename IndptrT>
__global__ void csr_gather_cast_accumulate_mapped_kernel(
    const InT* __restrict__ h_data, const IndexT* __restrict__ h_indices,
    const IndptrT* __restrict__ d_indptr_full,
    const int* __restrict__ d_row_ids, const int* __restrict__ d_out_indptr,
    const int* __restrict__ d_stats_codes, int fixed_slot,
    float* __restrict__ d_out_data_f32, int* __restrict__ d_out_indices,
    double* __restrict__ group_sums, double* __restrict__ group_nnz,
    int n_target_rows, int n_cols, int n_groups_stats, bool compute_sums,
    bool compute_nnz) {
    int r = blockIdx.x;
    if (r >= n_target_rows) return;
    int src_row = d_row_ids[r];
    IndptrT rs = d_indptr_full[src_row];
    IndptrT re = d_indptr_full[src_row + 1];
    int row_nnz = (int)(re - rs);
    int ds = d_out_indptr[r];
    int slot = (d_stats_codes != nullptr) ? d_stats_codes[r] : fixed_slot;
    bool accumulate = (slot >= 0 && slot < n_groups_stats);
    for (int i = threadIdx.x; i < row_nnz; i += blockDim.x) {
        InT v_in = h_data[rs + i];
        int c = (int)h_indices[rs + i];
        double v = (double)v_in;
        d_out_data_f32[ds + i] = (float)v_in;
        d_out_indices[ds + i] = c;
        if (accumulate) {
            if (compute_sums) {
                atomicAdd(&group_sums[(size_t)slot * n_cols + c], v);
            }
            if (compute_nnz && v != 0.0) {
                atomicAdd(&group_nnz[(size_t)slot * n_cols + c], 1.0);
            }
        }
    }
}

/** Fill linear segment offsets [0, stride, 2*stride, ...] on device.
 *  Runs on the supplied stream so it doesn't serialize multi-stream pipelines.
 */
static inline void upload_linear_offsets(int* d_offsets, int n_segments,
                                         int stride, cudaStream_t stream) {
    int count = n_segments + 1;
    int blk = (count + UTIL_BLOCK_SIZE - 1) / UTIL_BLOCK_SIZE;
    fill_linear_offsets_kernel<<<blk, UTIL_BLOCK_SIZE, 0, stream>>>(
        d_offsets, n_segments, stride);
    CUDA_CHECK_LAST_ERROR(fill_linear_offsets_kernel);
}

// ============================================================================
// CSR → dense F-order extraction (templated on data type)
// ============================================================================

template <typename T, typename IndptrT = int>
__global__ void csr_extract_dense_kernel(const T* __restrict__ data,
                                         const int* __restrict__ indices,
                                         const IndptrT* __restrict__ indptr,
                                         const int* __restrict__ row_ids,
                                         T* __restrict__ out, int n_target,
                                         int col_start, int col_stop) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= n_target) return;

    int row = row_ids[tid];
    IndptrT rs = indptr[row];
    IndptrT re = indptr[row + 1];

    IndptrT lo = rs, hi = re;
    while (lo < hi) {
        IndptrT m = lo + ((hi - lo) >> 1);
        if (indices[m] < col_start)
            lo = m + 1;
        else
            hi = m;
    }

    for (IndptrT p = lo; p < re; ++p) {
        int c = indices[p];
        if (c >= col_stop) break;
        out[(long long)(c - col_start) * n_target + tid] = data[p];
    }
}

template <typename T>
__global__ void csr_extract_dense_identity_rows_unsorted_kernel(
    const T* __restrict__ data, const int* __restrict__ indices,
    const int* __restrict__ indptr, T* __restrict__ out, int n_target,
    int col_start, int col_stop) {
    int row = blockIdx.x;
    if (row >= n_target) return;

    int rs = indptr[row];
    int re = indptr[row + 1];

    for (int p = rs + threadIdx.x; p < re; p += blockDim.x) {
        int c = indices[p];
        if (c >= col_start && c < col_stop) {
            out[(long long)(c - col_start) * n_target + row] = data[p];
        }
    }
}
