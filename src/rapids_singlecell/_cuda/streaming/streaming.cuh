#pragma once

#include <algorithm>
#include <condition_variable>
#include <functional>
#include <limits>
#include <memory>
#include <mutex>
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
// Multi-GPU callers can set a per-calling-thread limit so concurrent device
// workers do not each consume the full host thread pool.
inline thread_local int host_worker_limit = 0;

static inline int set_host_worker_limit(int limit) {
    int previous = host_worker_limit;
    host_worker_limit = std::max(0, limit);
    return previous;
}

static inline int host_worker_count() {
    unsigned hw = std::thread::hardware_concurrency();
    int workers = (int)std::min<unsigned>(hw ? hw : 4u, 32u);
    if (host_worker_limit > 0) workers = std::min(workers, host_worker_limit);
    return workers;
}

// Reuse staging workers for the lifetime of one calling thread. Host
// Wilcoxon runs each device shard on its own Python executor thread, so this
// naturally gives every shard an independent pool while avoiding thousands
// of create/join cycles across bounded staging blocks.
class HostWorkerPool {
   public:
    explicit HostWorkerPool(int n_threads) : n_threads_(n_threads) {
        workers_.reserve(n_threads_);
        try {
            for (int thread_index = 0; thread_index < n_threads_;
                 thread_index++) {
                workers_.emplace_back(
                    [this, thread_index]() { worker_loop(thread_index); });
            }
        } catch (...) {
            {
                std::lock_guard<std::mutex> lock(mutex_);
                stopping_ = true;
            }
            ready_.notify_all();
            for (std::thread& worker : workers_) worker.join();
            throw;
        }
    }

    HostWorkerPool(const HostWorkerPool&) = delete;
    HostWorkerPool& operator=(const HostWorkerPool&) = delete;

    ~HostWorkerPool() {
        {
            std::lock_guard<std::mutex> lock(mutex_);
            stopping_ = true;
        }
        ready_.notify_all();
        for (std::thread& worker : workers_) worker.join();
    }

    int size() const {
        return n_threads_;
    }

    int run(int n, std::function<void(int, int, int)> fn) {
        int chunk = (n + n_threads_ - 1) / n_threads_;
        int used = (n + chunk - 1) / chunk;

        std::unique_lock<std::mutex> lock(mutex_);
        active_fn_ = std::move(fn);
        active_n_ = n;
        active_chunk_ = chunk;
        active_threads_ = used;
        completed_threads_ = 0;
        generation_++;
        ready_.notify_all();
        finished_.wait(
            lock, [this]() { return completed_threads_ == active_threads_; });
        active_fn_ = {};
        return used;
    }

   private:
    void worker_loop(int thread_index) {
        size_t observed_generation = 0;
        for (;;) {
            std::unique_lock<std::mutex> lock(mutex_);
            ready_.wait(lock, [this, observed_generation]() {
                return stopping_ || generation_ != observed_generation;
            });
            if (stopping_) return;
            observed_generation = generation_;
            if (thread_index >= active_threads_) continue;

            int r0 = thread_index * active_chunk_;
            int r1 = std::min(active_n_, r0 + active_chunk_);
            auto* fn = &active_fn_;
            lock.unlock();
            (*fn)(thread_index, r0, r1);
            lock.lock();
            completed_threads_++;
            if (completed_threads_ == active_threads_) finished_.notify_one();
        }
    }

    int n_threads_;
    std::vector<std::thread> workers_;
    std::mutex mutex_;
    std::condition_variable ready_;
    std::condition_variable finished_;
    std::function<void(int, int, int)> active_fn_;
    int active_n_ = 0;
    int active_chunk_ = 0;
    int active_threads_ = 0;
    int completed_threads_ = 0;
    size_t generation_ = 0;
    bool stopping_ = false;
};

static inline HostWorkerPool& host_worker_pool(int n_threads) {
    thread_local std::unique_ptr<HostWorkerPool> pool;
    if (!pool || pool->size() != n_threads) {
        pool = std::make_unique<HostWorkerPool>(n_threads);
    }
    return *pool;
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
    return host_worker_pool(n_threads).run(
        n,
        [&fn](int thread_index, int r0, int r1) { fn(thread_index, r0, r1); });
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

struct DenseColumnBatchPlan {
    int sub_batch_cols = 0;
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
struct CudaHostFree {
    void operator()(T* ptr) const noexcept {
        if (ptr) cudaFreeHost(ptr);
    }
};

template <typename T>
struct PinnedRingArray {
    std::vector<std::unique_ptr<T[]>> data;
    std::vector<std::unique_ptr<T, CudaHostFree<T>>> write_combined_data;
    std::vector<HostRegisterGuard> pins;

    PinnedRingArray(int n_slots, size_t count, bool write_combined)
        : data(n_slots), write_combined_data(n_slots), pins(n_slots) {
        size_t n = count ? count : 1;
        for (int s = 0; s < n_slots; s++) {
            if (write_combined) {
                T* ptr = nullptr;
                cuda_check(cudaHostAlloc((void**)&ptr, n * sizeof(T),
                                         cudaHostAllocWriteCombined),
                           "PinnedRing write-combined allocation");
                write_combined_data[s].reset(ptr);
            } else {
                data[s].reset(new T[n]);
                pins[s] = HostRegisterGuard(data[s].get(), n * sizeof(T));
            }
        }
    }
    T* get(int slot) {
        T* ptr = write_combined_data[slot].get();
        return ptr ? ptr : data[slot].get();
    }
};

// Per-slot pinned host staging with events for CPU/GPU overlap.
// Arrays share item capacity; use another ring for differently-sized metadata.
template <typename... Ts>
struct PinnedRing {
    std::tuple<PinnedRingArray<Ts>...> arrays;
    std::vector<cudaEvent_t> evt;
    std::vector<char> used;
    size_t capacity = 0;

    PinnedRing(int n_slots_, size_t count, bool write_combined = false)
        : arrays(PinnedRingArray<Ts>(n_slots_, count, write_combined)...),
          evt(n_slots_, nullptr),
          used(n_slots_, 0) {
        capacity = count ? count : 1;
        for (int s = 0; s < n_slots_; s++) {
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
