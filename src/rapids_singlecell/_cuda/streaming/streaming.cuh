#pragma once

#include <algorithm>
#include <limits>
#include <memory>
#include <stdexcept>
#include <string>
#include <thread>
#include <tuple>
#include <type_traits>
#include <vector>

#include <cuda_runtime.h>

#include "../nb_types.h"
#include "../rmm_scratch.h"

// Default thread-per-block for utility kernels shared by streaming pipelines.
constexpr int UTIL_BLOCK_SIZE = 256;
constexpr int DEFAULT_STREAMING_STREAMS = 4;
// Max per-batch nnz for segmented CUDA primitives that take int32 item counts.
constexpr size_t STREAMING_SAFE_BATCH_NNZ = 2000000000;  // < INT_MAX
// Above this host span, avoid whole-array cudaHostRegister and use bounded
// staging. Moderate arrays keep the lower-overhead direct async-copy path.
constexpr size_t HOST_STREAMING_DIRECT_PIN_LIMIT_BYTES =
    16ULL * 1024ULL * 1024ULL * 1024ULL;

// Host thread count for CPU-side staging passes: hardware concurrency, capped.
static inline int host_worker_count() {
    unsigned hw = std::thread::hardware_concurrency();
    return (int)std::min<unsigned>(hw ? hw : 4u, 32u);
}

// Run fn(chunk, r0, r1) over partitions of [0, n), serial for small n.
// Concurrent callers must use read-only shared state and disjoint outputs.
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

template <typename DeviceValueT, typename DeviceIndexT = int,
          typename AccumT = double>
struct SparseWindowDTypes {
    using value_type = DeviceValueT;
    using index_type = DeviceIndexT;
    using accum_type = AccumT;

    static constexpr size_t bytes_per_nnz =
        sizeof(value_type) + sizeof(index_type);
};

using WilcoxonSparseWindowDTypes = SparseWindowDTypes<float, int, double>;

template <typename DTypes>
static inline size_t sparse_window_nnz_bytes(size_t nnz) {
    return nnz * DTypes::bytes_per_nnz;
}

template <typename DTypes>
static inline size_t sparse_window_accum_bytes(size_t count) {
    return count * sizeof(typename DTypes::accum_type);
}

static inline void host_clear_id_map(int* id_map, int n_items) {
    std::fill(id_map, id_map + n_items, -1);
}

static inline void host_build_id_map(const int* ids, int n_ids, int* id_map,
                                     int n_items, const char* what) {
    host_clear_id_map(id_map, n_items);
    for (int local = 0; local < n_ids; local++) {
        int id = ids[local];
        if (id < 0 || id >= n_items) {
            throw std::runtime_error(std::string(what) +
                                     " id is out of bounds");
        }
        id_map[id] = local;
    }
}

static inline void host_build_contiguous_id_map(int first, int count,
                                                int* id_map, int n_items,
                                                const char* what) {
    if (first < 0 || count < 0 || first > n_items - count) {
        throw std::runtime_error(std::string(what) +
                                 " contiguous id window is out of bounds");
    }
    host_clear_id_map(id_map, n_items);
    for (int local = 0; local < count; local++) id_map[first + local] = local;
}

// Stream-count clamps: never use more streams than column batches, nor more
// than the per-stream memory budget allows.
static inline int clamp_streams_by_cols(
    int n_cols, int sub_batch_cols,
    int max_streams = DEFAULT_STREAMING_STREAMS) {
    int n = max_streams;
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

// Scatter row-major [rows, sb_cols] into destination stride n_cols.
// `dst` must already point at the destination column offset.
static inline void scatter_cols_2d(double* dst, const double* src, int rows,
                                   int n_cols, int sb_cols,
                                   cudaStream_t stream) {
    cudaMemcpy2DAsync(dst, n_cols * sizeof(double), src,
                      sb_cols * sizeof(double), sb_cols * sizeof(double), rows,
                      cudaMemcpyDeviceToDevice, stream);
}

// Halve sub_batch_cols until the densest window holds <= cap nonzeros.
// Keeps CUB item counts and per-stream scratch bounded; worst case returns 1.
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

struct ColumnBatchPlan {
    int sub_batch_cols = 0;
    int n_batches = 0;
    size_t max_nnz = 0;
    std::vector<int> offsets;
    std::vector<size_t> nnz;
};

struct HostCompactSparseWindowPlan {
    int major_count = 0;
    size_t nnz = 0;
    std::vector<int> indptr;
};

struct DenseColumnBatchPlan {
    int sub_batch_cols = 0;
    int n_batches = 0;
    size_t max_items = 0;
};

static inline DenseColumnBatchPlan plan_dense_column_batches(
    int n_rows, int n_cols, int sub_batch_cols, size_t cap, const char* what) {
    DenseColumnBatchPlan plan;
    if (sub_batch_cols < 1) sub_batch_cols = 1;
    if (cap < 1) cap = 1;
    checked_cub_items((size_t)n_rows, what);

    size_t max_cols =
        n_rows > 0 ? cap / (size_t)n_rows : (size_t)sub_batch_cols;
    if (max_cols < 1) max_cols = 1;
    if ((size_t)sub_batch_cols > max_cols) sub_batch_cols = (int)max_cols;

    plan.sub_batch_cols = sub_batch_cols;
    plan.n_batches = (n_cols + sub_batch_cols - 1) / sub_batch_cols;
    plan.max_items = (size_t)n_rows * (size_t)sub_batch_cols;
    checked_cub_items(plan.max_items, what);
    return plan;
}

template <typename CountAt>
static inline ColumnBatchPlan plan_column_batches_from_counts(
    int n_cols, int sub_batch_cols, size_t cap, CountAt count_at,
    const char* what) {
    ColumnBatchPlan plan;
    plan.sub_batch_cols =
        cap_sub_batch_by_nnz(n_cols, sub_batch_cols, cap, count_at);
    plan.n_batches = (n_cols + plan.sub_batch_cols - 1) / plan.sub_batch_cols;
    plan.offsets.assign((size_t)plan.n_batches * (plan.sub_batch_cols + 1), 0);
    plan.nnz.assign(plan.n_batches, 0);
    for (int b = 0; b < plan.n_batches; b++) {
        int col_start = b * plan.sub_batch_cols;
        int sb = std::min(plan.sub_batch_cols, n_cols - col_start);
        int* off = &plan.offsets[(size_t)b * (plan.sub_batch_cols + 1)];
        for (int i = 0; i < sb; i++) {
            off[i + 1] = checked_int_span(
                (size_t)off[i] + (size_t)count_at(col_start + i), what);
        }
        plan.nnz[b] = (size_t)off[sb];
        if (plan.nnz[b] > plan.max_nnz) plan.max_nnz = plan.nnz[b];
    }
    return plan;
}

template <typename IndptrT>
static inline ColumnBatchPlan plan_csc_column_batches(const IndptrT* h_indptr,
                                                      int n_cols,
                                                      int sub_batch_cols,
                                                      size_t cap,
                                                      const char* what) {
    return plan_column_batches_from_counts(
        n_cols, sub_batch_cols, cap,
        [&](int c) { return (size_t)(h_indptr[c + 1] - h_indptr[c]); }, what);
}

static inline int* upload_batch_offsets(const ColumnBatchPlan& plan,
                                        RmmScratchPool& pool) {
    int* d_all_offsets = pool.alloc<int>(plan.offsets.size());
    cudaMemcpy(d_all_offsets, plan.offsets.data(),
               plan.offsets.size() * sizeof(int), cudaMemcpyHostToDevice);
    return d_all_offsets;
}

template <typename CountAt>
static HostCompactSparseWindowPlan plan_compact_sparse_window(
    int major_count, CountAt count_at, const char* what) {
    HostCompactSparseWindowPlan plan;
    plan.major_count = major_count;
    plan.indptr.assign((size_t)major_count + 1, 0);
    if (major_count <= 0) return plan;

    std::vector<size_t> counts(major_count, 0);
    host_parallel_ranges(major_count, [&](int i0, int i1) {
        for (int i = i0; i < i1; i++) counts[i] = count_at(i);
    });

    size_t run = 0;
    for (int i = 0; i < major_count; i++) {
        plan.indptr[i] = checked_int_span(run, what);
        run += counts[i];
    }
    plan.indptr[major_count] = checked_int_span(run, what);
    plan.nnz = run;
    return plan;
}

template <typename IndexT, typename IndptrT, typename RowToLocal>
static HostCompactSparseWindowPlan plan_csc_rows_window(
    const IndexT* h_indices, const IndptrT* h_indptr, int col_start,
    int n_window_cols, RowToLocal row_to_local, const char* what) {
    return plan_compact_sparse_window(
        n_window_cols,
        [&](int local_col) {
            int col = col_start + local_col;
            size_t count = 0;
            for (IndptrT p = h_indptr[col]; p < h_indptr[col + 1]; p++) {
                if (row_to_local((int)h_indices[p]) >= 0) count++;
            }
            return count;
        },
        what);
}

template <typename IndexT, typename IndptrT, typename ColToLocal>
static HostCompactSparseWindowPlan plan_csr_cols_window(
    const IndexT* h_indices, const IndptrT* h_indptr, const int* row_ids,
    int n_window_rows, ColToLocal col_to_local, const char* what) {
    return plan_compact_sparse_window(
        n_window_rows,
        [&](int local_row) {
            int row = row_ids ? row_ids[local_row] : local_row;
            size_t count = 0;
            for (IndptrT p = h_indptr[row]; p < h_indptr[row + 1]; p++) {
                if (col_to_local((int)h_indices[p]) >= 0) count++;
            }
            return count;
        },
        what);
}

template <typename IndexT, typename IndptrT>
static HostCompactSparseWindowPlan plan_csc_rows_window_from_map(
    const IndexT* h_indices, const IndptrT* h_indptr, int col_start,
    int n_window_cols, const int* row_map, const char* what) {
    return plan_csc_rows_window(
        h_indices, h_indptr, col_start, n_window_cols,
        [&](int row) { return row_map[row]; }, what);
}

template <typename IndexT, typename IndptrT>
static HostCompactSparseWindowPlan plan_csr_cols_window_from_map(
    const IndexT* h_indices, const IndptrT* h_indptr, const int* row_ids,
    int n_window_rows, const int* col_map, const char* what) {
    return plan_csr_cols_window(
        h_indices, h_indptr, row_ids, n_window_rows,
        [&](int col) { return col_map[col]; }, what);
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
                // Already-registered memory is owned elsewhere; use as-is.
                // Other failures are fatal unless pinning is only a speedup.
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

// RAII for CUDA streams/events: stream destruction synchronizes first.
// Declare RmmScratchPool before guards so streams drain before scratch frees.
struct ScopedCudaStream {
    cudaStream_t stream = nullptr;

    ScopedCudaStream() = default;
    explicit ScopedCudaStream(unsigned int flags) {
        cuda_check(cudaStreamCreateWithFlags(&stream, flags),
                   "cudaStreamCreateWithFlags");
    }
    ~ScopedCudaStream() {
        if (stream) {
            cudaStreamSynchronize(stream);
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
            cudaStreamSynchronize(s);
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

template <typename T>
struct PinnedRingArray {
    std::vector<std::unique_ptr<T[]>> data;
    std::vector<HostRegisterGuard> pins;

    PinnedRingArray() = default;
    PinnedRingArray(int n_slots, size_t count) : data(n_slots), pins(n_slots) {
        size_t n = count ? count : 1;
        for (int s = 0; s < n_slots; s++) {
            data[s].reset(new T[n]);
            pins[s] = HostRegisterGuard(data[s].get(), n * sizeof(T));
        }
    }
    T* get(int slot) {
        return data[slot].get();
    }
    const T* get(int slot) const {
        return data[slot].get();
    }
};

// Per-slot pinned host staging with events for CPU/GPU overlap.
// Arrays share item capacity; use another ring for differently-sized metadata.
template <typename... Ts>
struct PinnedRing {
    std::tuple<PinnedRingArray<Ts>...> arrays;
    std::vector<cudaEvent_t> evt;
    std::vector<char> used;
    int n_slots = 0;
    size_t capacity = 0;

    PinnedRing(int n_slots_, size_t count)
        : arrays(PinnedRingArray<Ts>(n_slots_, count)...),
          evt(n_slots_, nullptr),
          used(n_slots_, 0) {
        n_slots = n_slots_;
        capacity = count ? count : 1;
        for (int s = 0; s < n_slots; s++) {
            cuda_check(
                cudaEventCreateWithFlags(&evt[s], cudaEventDisableTiming),
                "PinnedRing event create");
        }
    }
    ~PinnedRing() {
        for (size_t s = 0; s < evt.size(); ++s) {
            cudaEvent_t e = evt[s];
            if (!e) continue;
            if (s < used.size() && used[s]) cudaEventSynchronize(e);
            cudaEventDestroy(e);
        }
    }
    void wait(int s) {
        if (used[s])
            cuda_check(cudaEventSynchronize(evt[s]), "PinnedRing reuse");
    }
    void record(int s, cudaStream_t stream) {
        cuda_check(cudaEventRecord(evt[s], stream), "PinnedRing record");
        used[s] = true;
    }
    template <size_t I>
    typename std::tuple_element<I, std::tuple<Ts...>>::type* get(int slot) {
        return std::get<I>(arrays).get(slot);
    }
    template <size_t I>
    const typename std::tuple_element<I, std::tuple<Ts...>>::type* get(
        int slot) const {
        return std::get<I>(arrays).get(slot);
    }
    PinnedRing(const PinnedRing&) = delete;
    PinnedRing& operator=(const PinnedRing&) = delete;
};

template <typename DTypes>
using SparseWindowStagingRing =
    PinnedRing<typename DTypes::value_type, typename DTypes::index_type>;

using HostStagingRing = SparseWindowStagingRing<WilcoxonSparseWindowDTypes>;

/** Fill linear segment offsets [0, stride, ..., n_segments*stride] on-device.
 */
__global__ void fill_linear_offsets_kernel(int* __restrict__ out,
                                           int n_segments, int stride) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i <= n_segments) out[i] = i * stride;
}

/** Rebase indptr slice to a local origin, grid-strided for arbitrary count.
 *  64-bit global indptrs may produce 32-bit pack-local indptrs. */
template <typename IdxIn, typename IdxOut>
__global__ void rebase_indptr_kernel(const IdxIn* __restrict__ indptr,
                                     IdxOut* __restrict__ out, int col,
                                     int count) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < count) out[i] = (IdxOut)(indptr[col + i] - indptr[col]);
}

// Threaded selected-row gather into compact staging at disjoint offsets.
// No-pin alternative: only the compacted slice crosses the bus.
template <typename StageValT, typename StageIndexT, typename InT,
          typename IndexT, typename IndptrT, typename CompactT>
static void host_gather_rows_compact_as(
    const InT* h_data, const IndexT* h_indices, const IndptrT* h_indptr,
    const int* row_ids, const CompactT* compact_indptr, CompactT base,
    int n_target, StageValT* stage_vals, StageIndexT* stage_cols) {
    host_parallel_ranges(n_target, [&](int i0, int i1) {
        for (int i = i0; i < i1; i++) {
            int r = row_ids[i];
            IndptrT rs = h_indptr[r];
            int nnz = (int)(h_indptr[r + 1] - rs);
            size_t ds = (size_t)(compact_indptr[i] - base);
            for (int k = 0; k < nnz; k++) {
                stage_vals[ds + k] = (StageValT)h_data[rs + k];
                stage_cols[ds + k] = (StageIndexT)h_indices[rs + k];
            }
        }
    });
}

template <typename InT, typename IndexT, typename IndptrT, typename CompactT>
static void host_gather_rows_compact(const InT* h_data, const IndexT* h_indices,
                                     const IndptrT* h_indptr,
                                     const int* row_ids,
                                     const CompactT* compact_indptr,
                                     CompactT base, int n_target,
                                     float* stage_vals, int* stage_cols) {
    host_gather_rows_compact_as<float, int>(h_data, h_indices, h_indptr,
                                            row_ids, compact_indptr, base,
                                            n_target, stage_vals, stage_cols);
}

// Threaded host cast-copy of a contiguous nnz slice into staging.
// CSC analogue of row gather: contiguous column batch, bounded int32 nnz.
template <typename StageValT, typename StageIndexT, typename InT,
          typename IndexT>
static void host_copy_slice_as(const InT* h_data, const IndexT* h_indices,
                               size_t start, int nnz, StageValT* stage_vals,
                               StageIndexT* stage_cols) {
    host_parallel_ranges(nnz, [&](int k0, int k1) {
        for (int k = k0; k < k1; k++) {
            stage_vals[k] = (StageValT)h_data[start + k];
            stage_cols[k] = (StageIndexT)h_indices[start + k];
        }
    });
}

template <typename InT, typename IndexT>
static void host_copy_slice(const InT* h_data, const IndexT* h_indices,
                            size_t start, int nnz, InT* stage_vals,
                            IndexT* stage_cols) {
    host_copy_slice_as<InT, IndexT>(h_data, h_indices, start, nnz, stage_vals,
                                    stage_cols);
}

template <typename InT, typename IndexT>
static void host_cast_copy_slice(const InT* h_data, const IndexT* h_indices,
                                 size_t start, int nnz, float* stage_vals,
                                 int* stage_cols) {
    host_copy_slice_as<float, int>(h_data, h_indices, start, nnz, stage_vals,
                                   stage_cols);
}

// Threaded host gather of selected dense rows and contiguous columns.
// Output staging is always F-order [n_window_rows, n_window_cols].
template <typename StageT, typename InT>
static void host_materialize_dense_rows_window_as(
    const InT* h_X, bool f_order, int n_full_rows, int n_full_cols,
    const int* row_ids, int n_window_rows, int col_start, int n_window_cols,
    StageT* stage) {
    int total =
        checked_int_product((size_t)n_window_rows, (size_t)n_window_cols,
                            "dense host row-window items");
    host_parallel_ranges(total, [&](int i0, int i1) {
        for (int idx = i0; idx < i1; idx++) {
            int local_col = idx / n_window_rows;
            int local_row = idx - local_col * n_window_rows;
            int row = row_ids ? row_ids[local_row] : local_row;
            int col = col_start + local_col;
            size_t src = f_order ? (size_t)col * n_full_rows + row
                                 : (size_t)row * n_full_cols + col;
            stage[(size_t)local_col * n_window_rows + local_row] =
                (StageT)h_X[src];
        }
    });
}

template <typename InT>
static void host_materialize_dense_rows_window(const InT* h_X, bool f_order,
                                               int n_full_rows, int n_full_cols,
                                               const int* row_ids,
                                               int n_window_rows, int col_start,
                                               int n_window_cols, InT* stage) {
    host_materialize_dense_rows_window_as<InT>(
        h_X, f_order, n_full_rows, n_full_cols, row_ids, n_window_rows,
        col_start, n_window_cols, stage);
}

// Cross-axis CSC materialization: filter a contiguous column window by selected
// rows and emit compact CSC with local row ids.
template <typename StageValT, typename StageIndexT, typename InT,
          typename IndexT, typename IndptrT, typename RowToLocal>
static void host_materialize_csc_rows_window_as(
    const InT* h_data, const IndexT* h_indices, const IndptrT* h_indptr,
    int col_start, int n_window_cols, const int* compact_indptr,
    RowToLocal row_to_local, StageValT* stage_vals, StageIndexT* stage_rows) {
    host_parallel_ranges(n_window_cols, [&](int c0, int c1) {
        for (int local_col = c0; local_col < c1; local_col++) {
            int col = col_start + local_col;
            size_t dst = (size_t)compact_indptr[local_col];
            for (IndptrT p = h_indptr[col]; p < h_indptr[col + 1]; p++) {
                int local_row = row_to_local((int)h_indices[p]);
                if (local_row < 0) continue;
                stage_vals[dst] = (StageValT)h_data[p];
                stage_rows[dst] = (StageIndexT)local_row;
                dst++;
            }
        }
    });
}

template <typename InT, typename IndexT, typename IndptrT>
static void host_materialize_csc_rows_window(
    const InT* h_data, const IndexT* h_indices, const IndptrT* h_indptr,
    int col_start, int n_window_cols, const int* compact_indptr,
    const int* row_map, float* stage_vals, int* stage_rows) {
    host_materialize_csc_rows_window_as<float, int>(
        h_data, h_indices, h_indptr, col_start, n_window_cols, compact_indptr,
        [&](int row) { return row_map[row]; }, stage_vals, stage_rows);
}

// Cross-axis CSR materialization: filter selected rows by selected columns and
// emit compact CSR with local column ids.
template <typename StageValT, typename StageIndexT, typename InT,
          typename IndexT, typename IndptrT, typename ColToLocal>
static void host_materialize_csr_cols_window_as(
    const InT* h_data, const IndexT* h_indices, const IndptrT* h_indptr,
    const int* row_ids, int n_window_rows, const int* compact_indptr,
    ColToLocal col_to_local, StageValT* stage_vals, StageIndexT* stage_cols) {
    host_parallel_ranges(n_window_rows, [&](int r0, int r1) {
        for (int local_row = r0; local_row < r1; local_row++) {
            int row = row_ids ? row_ids[local_row] : local_row;
            size_t dst = (size_t)compact_indptr[local_row];
            for (IndptrT p = h_indptr[row]; p < h_indptr[row + 1]; p++) {
                int local_col = col_to_local((int)h_indices[p]);
                if (local_col < 0) continue;
                stage_vals[dst] = (StageValT)h_data[p];
                stage_cols[dst] = (StageIndexT)local_col;
                dst++;
            }
        }
    });
}

template <typename InT, typename IndexT, typename IndptrT>
static void host_materialize_csr_cols_window(
    const InT* h_data, const IndexT* h_indices, const IndptrT* h_indptr,
    const int* row_ids, int n_window_rows, const int* compact_indptr,
    const int* col_map, float* stage_vals, int* stage_cols) {
    host_materialize_csr_cols_window_as<float, int>(
        h_data, h_indices, h_indptr, row_ids, n_window_rows, compact_indptr,
        [&](int col) { return col_map[col]; }, stage_vals, stage_cols);
}

// Optimized CSR -> contiguous-column-window materialization for sorted rows.
// The per-row cursor examines each nonzero once across the full stream.
template <typename StageValT, typename StageIndexT, typename InT,
          typename IndexT, typename IndptrT>
static int host_materialize_csr_column_interval_cursor_as(
    const InT* h_data, const IndexT* h_indices, const IndptrT* h_indptr,
    int n_rows, int col_start, int col_end, IndptrT* cursor, int* row_counts,
    int* compact_indptr, StageValT* stage_vals, StageIndexT* stage_cols,
    const char* what) {
    host_parallel_ranges(n_rows, [&](int r0, int r1) {
        for (int r = r0; r < r1; r++) {
            const IndexT* row_base = h_indices + h_indptr[r];
            const IndexT* lo = row_base + cursor[r];
            const IndexT* hi = h_indices + h_indptr[r + 1];
            if (lo < hi && *lo < (IndexT)col_start) {
                lo = std::lower_bound(lo, hi, (IndexT)col_start);
                cursor[r] = (IndptrT)(lo - row_base);
            }
            row_counts[r] =
                (int)(std::lower_bound(lo, hi, (IndexT)col_end) - lo);
        }
    });

    compact_indptr[0] = 0;
    for (int r = 0; r < n_rows; r++) {
        compact_indptr[r + 1] = checked_int_span(
            (size_t)compact_indptr[r] + (size_t)row_counts[r], what);
    }
    int batch_nnz = compact_indptr[n_rows];

    host_parallel_ranges(n_rows, [&](int r0, int r1) {
        for (int r = r0; r < r1; r++) {
            IndptrT base = h_indptr[r] + cursor[r];
            size_t dst = (size_t)compact_indptr[r];
            int count = row_counts[r];
            for (int k = 0; k < count; k++) {
                stage_vals[dst + k] = (StageValT)h_data[base + k];
                stage_cols[dst + k] = (StageIndexT)h_indices[base + k];
            }
            cursor[r] += count;
        }
    });
    return batch_nnz;
}

template <typename InT, typename IndexT, typename IndptrT>
static int host_materialize_csr_column_interval_cursor(
    const InT* h_data, const IndexT* h_indices, const IndptrT* h_indptr,
    int n_rows, int col_start, int col_end, IndptrT* cursor, int* row_counts,
    int* compact_indptr, InT* stage_vals, int* stage_cols, const char* what) {
    return host_materialize_csr_column_interval_cursor_as<InT, int>(
        h_data, h_indices, h_indptr, n_rows, col_start, col_end, cursor,
        row_counts, compact_indptr, stage_vals, stage_cols, what);
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
