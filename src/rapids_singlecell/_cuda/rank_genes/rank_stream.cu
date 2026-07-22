// Host-streaming aggregation (sum/sq_sum/count) and binned histograms.
// Additive-scatter reductions (no per-column ordering, unlike Wilcoxon) →
// stream host row-blocks (CSR/C-dense) or column-windows (CSC/F-dense) into the
// device output, reusing the existing device kernels; only a window is ever
// on-GPU.
#include <cuda_runtime.h>
#include <nanobind/stl/optional.h>

#include <optional>
#include <vector>

#include "../aggr/kernels_aggr.cuh"
#include "../nb_types.h"
#include "../streaming/streaming.cuh"
#include "../wilcoxon_binned/kernels_wilcoxon_binned.cuh"

using namespace nb::literals;

namespace {

constexpr int AGGR_BLOCK = 256;
constexpr int HIST_BLOCK = 256;
constexpr int N_STREAMS = DEFAULT_STREAMING_STREAMS;
constexpr int STREAM_ROW_CAP = 1 << 20;  // bound the per-block indptr buffer
constexpr int DEFAULT_SUB_BATCH = 4096;  // rows (CSR/C-dense) or cols (CSC/F)

// Greedy contiguous blocks bounded by nnz_cap and seg_cap (>=1 segment).
template <typename IdxT>
std::vector<int> plan_nnz_blocks(const IdxT* indptr, int n, size_t nnz_cap,
                                 int seg_cap) {
    std::vector<int> bounds;
    bounds.push_back(0);
    int seg = 0;
    while (seg < n) {
        int start = seg;
        size_t base = (size_t)indptr[start];
        int stop = start + 1;
        while (stop < n && (stop - start) < seg_cap &&
               (size_t)indptr[stop + 1] - base <= nnz_cap) {
            stop++;
        }
        bounds.push_back(stop);
        seg = stop;
    }
    return bounds;
}

// Rebase indptr[col .. col+count) to a local origin on-device.
template <typename IdxT>
void rebase_block_indptr(const IdxT* d_indptr_full, IdxT* d_out, int col,
                         int count, cudaStream_t stream) {
    int blk = (count + UTIL_BLOCK_SIZE - 1) / UTIL_BLOCK_SIZE;
    rebase_indptr_kernel<IdxT, IdxT>
        <<<blk, UTIL_BLOCK_SIZE, 0, stream>>>(d_indptr_full, d_out, col, count);
    CUDA_CHECK_LAST_ERROR(rebase_indptr_kernel);
}

// Transpose a C-order [rows, cols] device block into an F-order [rows, cols]
// block (needed only for C-order dense histograms, which read whole columns).
template <typename T>
__global__ void transpose_c_to_f_kernel(const T* __restrict__ src,
                                        T* __restrict__ dst, int rows,
                                        int cols) {
    size_t n = (size_t)rows * cols;
    for (size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x; i < n;
         i += (size_t)gridDim.x * blockDim.x) {
        int r = (int)(i / cols);
        int c = (int)(i % cols);
        dst[(size_t)c * rows + r] = src[i];
    }
}

// Per-stream device scratch for the sparse streamers.
template <typename InT, typename IdxT>
struct SparseStreamBufs {
    InT* d_data = nullptr;
    IdxT* d_indices = nullptr;
    IdxT* d_indptr = nullptr;  // rebased, seg_cap+1
};

// Stream nnz-bounded blocks of a host CSR/CSC through N_STREAMS streams,
// calling consume(slot, d_data, d_indices, d_indptr_rebased, seg0, n_seg,
// block_nnz, stream) per block. slot == stream index (keeps per-stream buffer
// reuse ordered).
template <typename InT, typename IdxT, typename Consume>
void stream_sparse_blocks(const InT* h_data, const IdxT* h_indices,
                          const IdxT* h_indptr, int n_seg, int sub_batch,
                          Consume&& consume) {
    if (n_seg <= 0) return;
    size_t total_nnz = (size_t)h_indptr[n_seg];

    size_t budget = rmm_available_device_bytes(0.8);
    size_t bytes_per_nnz = sizeof(InT) + sizeof(IdxT);
    size_t nnz_cap = STREAMING_SAFE_BATCH_NNZ;
    size_t mem_cap = budget / (size_t)N_STREAMS / bytes_per_nnz;
    if (mem_cap > 0 && mem_cap < nnz_cap) nnz_cap = mem_cap;
    if (nnz_cap < 1) nnz_cap = 1;

    int seg_cap = sub_batch > 0 ? sub_batch : DEFAULT_SUB_BATCH;
    if (seg_cap > STREAM_ROW_CAP) seg_cap = STREAM_ROW_CAP;

    std::vector<int> bounds =
        plan_nnz_blocks(h_indptr, n_seg, nnz_cap, seg_cap);
    int n_blocks = (int)bounds.size() - 1;

    size_t max_nnz = 0;
    for (int b = 0; b < n_blocks; b++) {
        size_t bnnz =
            (size_t)h_indptr[bounds[b + 1]] - (size_t)h_indptr[bounds[b]];
        if (bnnz > max_nnz) max_nnz = bnnz;
    }
    if (max_nnz < 1) max_nnz = 1;

    RmmScratchPool pool;
    IdxT* d_indptr_full = pool.alloc<IdxT>((size_t)n_seg + 1);
    std::vector<SparseStreamBufs<InT, IdxT>> bufs(N_STREAMS);
    for (int s = 0; s < N_STREAMS; s++) {
        bufs[s].d_data = pool.alloc<InT>(max_nnz);
        bufs[s].d_indices = pool.alloc<IdxT>(max_nnz);
        bufs[s].d_indptr = pool.alloc<IdxT>((size_t)seg_cap + 1);
    }
    ScopedCudaStreams streams(N_STREAMS, cudaStreamDefault);
    // Pinned staging ring (host cast-copy pageable->pinned, then async
    // pinned->device): beats whole-array register and pageable staging, and
    // gives true async DMA that overlaps across devices (mirrors wilcoxon).
    (void)total_nnz;
    PinnedRing<InT, IdxT> stage(N_STREAMS, max_nnz);

    cuda_check(
        cudaMemcpy(d_indptr_full, h_indptr, ((size_t)n_seg + 1) * sizeof(IdxT),
                   cudaMemcpyHostToDevice),
        "stream_sparse_blocks indptr H2D");
    cudaDeviceSynchronize();

    for (int b = 0; b < n_blocks; b++) {
        int seg0 = bounds[b];
        int n_block_seg = bounds[b + 1] - seg0;
        size_t ptr_start = (size_t)h_indptr[seg0];
        size_t block_nnz = (size_t)h_indptr[seg0 + n_block_seg] - ptr_start;
        int s = b % N_STREAMS;
        cudaStream_t stream = streams[s];
        auto& buf = bufs[s];

        stage.wait(s);  // slot free once its prior async copy finished
        if (block_nnz > 0) {
            InT* h_vals = stage.template get<0>(s);
            IdxT* h_idx = stage.template get<1>(s);
            host_copy_slice_as<InT, IdxT>(h_data, h_indices, ptr_start,
                                          (int)block_nnz, h_vals, h_idx);
            cuda_check(
                cudaMemcpyAsync(buf.d_data, h_vals, block_nnz * sizeof(InT),
                                cudaMemcpyHostToDevice, stream),
                "stream_sparse_blocks data H2D");
            cuda_check(
                cudaMemcpyAsync(buf.d_indices, h_idx, block_nnz * sizeof(IdxT),
                                cudaMemcpyHostToDevice, stream),
                "stream_sparse_blocks indices H2D");
        }
        stage.record(s, stream);  // slot reusable after these copies finish
        rebase_block_indptr(d_indptr_full, buf.d_indptr, seg0, n_block_seg + 1,
                            stream);
        consume(s, buf.d_data, buf.d_indices, buf.d_indptr, seg0, n_block_seg,
                (int)block_nnz, stream);
    }
    sync_streams(streams, "stream_sparse_blocks");
}

// Aggregation MASK dispatch (mirrors aggr.cu; kernels from kernels_aggr.cuh).
#define RANK_STREAM_AGGR_DISPATCH(active, LAUNCH)                        \
    switch (active) {                                                    \
        case 1:                                                          \
            LAUNCH(1);                                                   \
            break;                                                       \
        case 2:                                                          \
            LAUNCH(2);                                                   \
            break;                                                       \
        case 3:                                                          \
            LAUNCH(3);                                                   \
            break;                                                       \
        case 4:                                                          \
            LAUNCH(4);                                                   \
            break;                                                       \
        case 5:                                                          \
            LAUNCH(5);                                                   \
            break;                                                       \
        case 6:                                                          \
            LAUNCH(6);                                                   \
            break;                                                       \
        case 7:                                                          \
            LAUNCH(7);                                                   \
            break;                                                       \
        default:                                                         \
            throw std::runtime_error(                                    \
                "rank_stream aggr: at least one output plane required"); \
    }

int aggr_active_mask(double* ps, double* pc, double* pq) {
    return (ps ? AGGR_SUM : 0) | (pc ? AGGR_COUNT : 0) | (pq ? AGGR_SQSUM : 0);
}

// Return the caller's cell mask, or an all-ones mask allocated from `pool`
// (which must outlive the streaming call that reads it).
template <typename Device>
const bool* alloc_or_mask(
    const std::optional<gpu_array_c<const bool, Device>>& mask, int n_cells,
    RmmScratchPool& pool) {
    if (mask) return mask->data();
    bool* ones = pool.alloc<bool>((size_t)n_cells);
    cudaMemset(ones, 1, (size_t)n_cells);
    return ones;
}

// ---- aggregation: CSR host (row-block stream, full-output accumulation) ----
template <typename T, typename IdxT, typename Device>
void def_aggr_csr_host(nb::module_& m) {
    m.def(
        "aggr_csr_host",
        [](host_array<const T> data, host_array<const IdxT> indices,
           host_array<const IdxT> indptr, gpu_array_c<const int, Device> cats,
           std::optional<gpu_array_c<double, Device>> out_sum,
           std::optional<gpu_array_c<double, Device>> out_count,
           std::optional<gpu_array_c<double, Device>> out_sqsum,
           std::optional<gpu_array_c<const bool, Device>> mask, int n_cells,
           int n_genes, int sub_batch_rows) {
            double* ps = out_sum ? out_sum->data() : nullptr;
            double* pc = out_count ? out_count->data() : nullptr;
            double* pq = out_sqsum ? out_sqsum->data() : nullptr;
            int active = aggr_active_mask(ps, pc, pq);
            nb_require((int)indptr.shape(0) == n_cells + 1,
                       "aggr_csr_host: indptr length must be n_cells+1");
            const int* d_cats = cats.data();
            RmmScratchPool mask_pool;
            const bool* d_mask = alloc_or_mask(mask, n_cells, mask_pool);
            auto consume = [&](int s, const T* d_data, const IdxT* d_indices,
                               const IdxT* d_indptr, int seg0, int n_block_seg,
                               int block_nnz, cudaStream_t stream) {
                (void)s;
                (void)block_nnz;
                dim3 block(AGGR_BLOCK);
                dim3 grid((unsigned)n_block_seg);
#define LAUNCH(M)                                                              \
    csr_aggr_kernel<T, IdxT, M><<<grid, block, 0, stream>>>(                   \
        d_indptr, d_indices, d_data, ps, pc, pq, d_cats + seg0, d_mask + seg0, \
        (size_t)n_block_seg, (size_t)n_genes);                                 \
    CUDA_CHECK_LAST_ERROR(csr_aggr_kernel)
                RANK_STREAM_AGGR_DISPATCH(active, LAUNCH);
#undef LAUNCH
            };
            stream_sparse_blocks<T, IdxT>(data.data(), indices.data(),
                                          indptr.data(), n_cells,
                                          sub_batch_rows, consume);
        },
        "data"_a, "indices"_a, "indptr"_a, "cats"_a, nb::kw_only(),
        "out_sum"_a = nb::none(), "out_count"_a = nb::none(),
        "out_sqsum"_a = nb::none(), "mask"_a = nb::none(), "n_cells"_a,
        "n_genes"_a, "sub_batch_rows"_a = DEFAULT_SUB_BATCH,
        nb::call_guard<nb::gil_scoped_release>());
}

// ---- aggregation: CSC host (column-window stream, per-window out + scatter)
// ----
template <typename T, typename IdxT, typename Device>
void def_aggr_csc_host(nb::module_& m) {
    m.def(
        "aggr_csc_host",
        [](host_array<const T> data, host_array<const IdxT> indices,
           host_array<const IdxT> indptr, gpu_array_c<const int, Device> cats,
           std::optional<gpu_array_c<double, Device>> out_sum,
           std::optional<gpu_array_c<double, Device>> out_count,
           std::optional<gpu_array_c<double, Device>> out_sqsum,
           std::optional<gpu_array_c<const bool, Device>> mask, int n_cells,
           int n_genes, int sub_batch_cols) {
            double* ps = out_sum ? out_sum->data() : nullptr;
            double* pc = out_count ? out_count->data() : nullptr;
            double* pq = out_sqsum ? out_sqsum->data() : nullptr;
            int active = aggr_active_mask(ps, pc, pq);
            nb_require((int)indptr.shape(0) == n_genes + 1,
                       "aggr_csc_host: indptr length must be n_genes+1");
            int n_cats = out_sum     ? (int)out_sum->shape(0)
                         : out_count ? (int)out_count->shape(0)
                                     : (int)out_sqsum->shape(0);
            const int* d_cats = cats.data();
            int sb = sub_batch_cols > 0 ? sub_batch_cols : DEFAULT_SUB_BATCH;
            RmmScratchPool wpool;
            const bool* d_mask = alloc_or_mask(mask, n_cells, wpool);
            // Per-stream window output planes, scattered into the full output.
            double* w_sum[N_STREAMS] = {nullptr};
            double* w_count[N_STREAMS] = {nullptr};
            double* w_sqsum[N_STREAMS] = {nullptr};
            for (int s = 0; s < N_STREAMS; s++) {
                if (ps) w_sum[s] = wpool.alloc<double>((size_t)n_cats * sb);
                if (pc) w_count[s] = wpool.alloc<double>((size_t)n_cats * sb);
                if (pq) w_sqsum[s] = wpool.alloc<double>((size_t)n_cats * sb);
            }
            auto consume = [&](int s, const T* d_data, const IdxT* d_indices,
                               const IdxT* d_indptr, int seg0, int n_block_seg,
                               int block_nnz, cudaStream_t stream) {
                (void)block_nnz;
                double* ws = ps ? w_sum[s] : nullptr;
                double* wc = pc ? w_count[s] : nullptr;
                double* wq = pq ? w_sqsum[s] : nullptr;
                size_t plane = (size_t)n_cats * n_block_seg;
                if (ws) cudaMemsetAsync(ws, 0, plane * sizeof(double), stream);
                if (wc) cudaMemsetAsync(wc, 0, plane * sizeof(double), stream);
                if (wq) cudaMemsetAsync(wq, 0, plane * sizeof(double), stream);
                dim3 block(AGGR_BLOCK);
                dim3 grid((unsigned)n_block_seg);
#define LAUNCH(M)                                                \
    csc_aggr_kernel<T, IdxT, M><<<grid, block, 0, stream>>>(     \
        d_indptr, d_indices, d_data, ws, wc, wq, d_cats, d_mask, \
        (size_t)n_cells, (size_t)n_block_seg);                   \
    CUDA_CHECK_LAST_ERROR(csc_aggr_kernel)
                RANK_STREAM_AGGR_DISPATCH(active, LAUNCH);
#undef LAUNCH
                if (ws)
                    scatter_cols_2d(ps + seg0, ws, n_cats, n_genes, n_block_seg,
                                    stream);
                if (wc)
                    scatter_cols_2d(pc + seg0, wc, n_cats, n_genes, n_block_seg,
                                    stream);
                if (wq)
                    scatter_cols_2d(pq + seg0, wq, n_cats, n_genes, n_block_seg,
                                    stream);
            };
            stream_sparse_blocks<T, IdxT>(data.data(), indices.data(),
                                          indptr.data(), n_genes, sb, consume);
        },
        "data"_a, "indices"_a, "indptr"_a, "cats"_a, nb::kw_only(),
        "out_sum"_a = nb::none(), "out_count"_a = nb::none(),
        "out_sqsum"_a = nb::none(), "mask"_a = nb::none(), "n_cells"_a,
        "n_genes"_a, "sub_batch_cols"_a = DEFAULT_SUB_BATCH,
        nb::call_guard<nb::gil_scoped_release>());
}

// ---- aggregation: dense host (C-order row-block, full out; F-order
// column-window + scatter) ----
template <typename T, typename Device, typename HostArray, bool FOrder>
void def_aggr_dense_host(nb::module_& m) {
    m.def(
        "aggr_dense_host",
        [](HostArray X, gpu_array_c<const int, Device> cats,
           std::optional<gpu_array_c<double, Device>> out_sum,
           std::optional<gpu_array_c<double, Device>> out_count,
           std::optional<gpu_array_c<double, Device>> out_sqsum,
           std::optional<gpu_array_c<const bool, Device>> mask, int sub_batch) {
            double* ps = out_sum ? out_sum->data() : nullptr;
            double* pc = out_count ? out_count->data() : nullptr;
            double* pq = out_sqsum ? out_sqsum->data() : nullptr;
            int active = aggr_active_mask(ps, pc, pq);
            int n_cells = (int)X.shape(0);
            int n_genes = (int)X.shape(1);
            int n_cats = out_sum     ? (int)out_sum->shape(0)
                         : out_count ? (int)out_count->shape(0)
                                     : (int)out_sqsum->shape(0);
            const T* h_X = X.data();
            const int* d_cats = cats.data();
            int sb = sub_batch > 0 ? sub_batch : DEFAULT_SUB_BATCH;
            int axis =
                FOrder ? n_genes : n_cells;  // stream cols (F) or rows (C)
            if (sb > axis) sb = axis < 1 ? 1 : axis;

            size_t budget = rmm_available_device_bytes(0.8);
            size_t other = FOrder ? (size_t)n_cells : (size_t)n_genes;
            size_t max_items = (size_t)sb * other;
            int n_streams = N_STREAMS;

            RmmScratchPool pool;
            const bool* d_mask = alloc_or_mask(mask, n_cells, pool);
            std::vector<T*> d_block(n_streams, nullptr);
            std::vector<double*> w_sum(n_streams, nullptr),
                w_count(n_streams, nullptr), w_sqsum(n_streams, nullptr);
            for (int s = 0; s < n_streams; s++) {
                d_block[s] = pool.alloc<T>(max_items);
                if (FOrder) {
                    if (ps) w_sum[s] = pool.alloc<double>((size_t)n_cats * sb);
                    if (pc)
                        w_count[s] = pool.alloc<double>((size_t)n_cats * sb);
                    if (pq)
                        w_sqsum[s] = pool.alloc<double>((size_t)n_cats * sb);
                }
            }
            (void)budget;
            ScopedCudaStreams streams(n_streams, cudaStreamDefault);
            // Pageable async copies (see stream_sparse_blocks) — no per-call
            // pin.
            cudaDeviceSynchronize();

            int b = 0;
            for (int p = 0; p < axis; p += sb, b++) {
                int n_block = (p + sb <= axis) ? sb : (axis - p);
                int s = b % n_streams;
                cudaStream_t stream = streams[s];
                if (FOrder) {
                    // contiguous column window [p, p+n_block): h_X + p*n_cells
                    cuda_check(
                        cudaMemcpyAsync(d_block[s], h_X + (size_t)p * n_cells,
                                        (size_t)n_block * n_cells * sizeof(T),
                                        cudaMemcpyHostToDevice, stream),
                        "aggr_dense_host F window H2D");
                    double* ws = ps ? w_sum[s] : nullptr;
                    double* wc = pc ? w_count[s] : nullptr;
                    double* wq = pq ? w_sqsum[s] : nullptr;
                    size_t plane = (size_t)n_cats * n_block;
                    if (ws)
                        cudaMemsetAsync(ws, 0, plane * sizeof(double), stream);
                    if (wc)
                        cudaMemsetAsync(wc, 0, plane * sizeof(double), stream);
                    if (wq)
                        cudaMemsetAsync(wq, 0, plane * sizeof(double), stream);
                    dim3 block(AGGR_BLOCK);
                    dim3 grid(
                        strided_grid((long long)n_cells * n_block, AGGR_BLOCK));
#define LAUNCH(M)                                                            \
    dense_aggr_kernel_F<T, M>                                                \
        <<<grid, block, 0, stream>>>(d_block[s], ws, wc, wq, d_cats, d_mask, \
                                     (size_t)n_cells, (size_t)n_block);      \
    CUDA_CHECK_LAST_ERROR(dense_aggr_kernel_F)
                    RANK_STREAM_AGGR_DISPATCH(active, LAUNCH);
#undef LAUNCH
                    if (ws)
                        scatter_cols_2d(ps + p, ws, n_cats, n_genes, n_block,
                                        stream);
                    if (wc)
                        scatter_cols_2d(pc + p, wc, n_cats, n_genes, n_block,
                                        stream);
                    if (wq)
                        scatter_cols_2d(pq + p, wq, n_cats, n_genes, n_block,
                                        stream);
                } else {
                    // contiguous row block [p, p+n_block): h_X + p*n_genes
                    cuda_check(
                        cudaMemcpyAsync(d_block[s], h_X + (size_t)p * n_genes,
                                        (size_t)n_block * n_genes * sizeof(T),
                                        cudaMemcpyHostToDevice, stream),
                        "aggr_dense_host C row-block H2D");
                    dim3 block(AGGR_BLOCK);
                    dim3 grid(
                        strided_grid((long long)n_block * n_genes, AGGR_BLOCK));
#define LAUNCH(M)                                                        \
    dense_aggr_kernel_C<T, M><<<grid, block, 0, stream>>>(               \
        d_block[s], ps, pc, pq, d_cats + p, d_mask + p, (size_t)n_block, \
        (size_t)n_genes);                                                \
    CUDA_CHECK_LAST_ERROR(dense_aggr_kernel_C)
                    RANK_STREAM_AGGR_DISPATCH(active, LAUNCH);
#undef LAUNCH
                }
            }
            sync_streams(streams, "aggr_dense_host");
        },
        "X"_a, "cats"_a, nb::kw_only(), "out_sum"_a = nb::none(),
        "out_count"_a = nb::none(), "out_sqsum"_a = nb::none(),
        "mask"_a = nb::none(), "sub_batch"_a = DEFAULT_SUB_BATCH,
        nb::call_guard<nb::gil_scoped_release>());
}

// ---- histogram: CSR host (row-block stream, single pass into the full hist)
// ---- csr_hist_kernel filters the gene window per row; bin 0 is filled
// Python-side.
template <typename T, typename IdxT, typename Device>
void def_hist_csr_host(nb::module_& m) {
    m.def(
        "hist_csr_host",
        [](host_array<const T> data, host_array<const IdxT> indices,
           host_array<const IdxT> indptr, gpu_array_c<const int, Device> gcodes,
           gpu_array_c<unsigned int, Device> hist,
           std::optional<gpu_array_c<double, Device>> group_sums,
           std::optional<gpu_array_c<double, Device>> group_nnz, int n_cells,
           int n_genes, int n_groups, int n_bins, double bin_low,
           double inv_bin_width, int col_start, int col_stop,
           int sub_batch_rows) {
            if (col_stop < 0) col_stop = n_genes;
            nb_require((int)indptr.shape(0) == n_cells + 1,
                       "hist_csr_host: indptr length must be n_cells+1");
            nb_require(
                col_start >= 0 && col_start <= col_stop && col_stop <= n_genes,
                "hist_csr_host: invalid column range");
            const int* d_gcodes = gcodes.data();
            unsigned int* d_hist = hist.data();
            int window = col_stop - col_start;
            // Fused means: accumulate group sums (+nnz) on the same window
            // (single full-width pass only, so each value is added once).
            double* d_gsum = group_sums ? group_sums->data() : nullptr;
            double* d_gnnz = group_nnz ? group_nnz->data() : nullptr;
            int aggr_mask = (d_gsum ? AGGR_SUM : 0) | (d_gnnz ? AGGR_COUNT : 0);
            RmmScratchPool mask_pool;
            bool* d_mask = nullptr;
            if (aggr_mask) {
                d_mask = mask_pool.alloc<bool>((size_t)n_cells);
                cudaMemset(d_mask, 1, (size_t)n_cells);
            }
            auto consume = [&](int s, const T* d_data, const IdxT* d_indices,
                               const IdxT* d_indptr, int seg0, int n_block_seg,
                               int block_nnz, cudaStream_t stream) {
                (void)s;
                (void)block_nnz;
                // gene_start=col_start, n_genes=window → fills the chunk hist.
                csr_hist_kernel<T, IdxT>
                    <<<(unsigned)n_block_seg, HIST_BLOCK, 0, stream>>>(
                        d_data, d_indices, d_indptr, d_gcodes + seg0, d_hist,
                        n_block_seg, window, n_groups, n_bins, bin_low,
                        inv_bin_width, col_start);
                CUDA_CHECK_LAST_ERROR(csr_hist_kernel);
                if (aggr_mask) {
                    dim3 grid((unsigned)n_block_seg);
                    dim3 block(AGGR_BLOCK);
#define LAUNCH(M)                                                              \
    csr_aggr_kernel<T, IdxT, M><<<grid, block, 0, stream>>>(                   \
        d_indptr, d_indices, d_data, d_gsum, d_gnnz, nullptr, d_gcodes + seg0, \
        d_mask + seg0, (size_t)n_block_seg, (size_t)n_genes);                  \
    CUDA_CHECK_LAST_ERROR(csr_aggr_kernel)
                    RANK_STREAM_AGGR_DISPATCH(aggr_mask, LAUNCH);
#undef LAUNCH
                }
            };
            stream_sparse_blocks<T, IdxT>(data.data(), indices.data(),
                                          indptr.data(), n_cells,
                                          sub_batch_rows, consume);
        },
        "data"_a, "indices"_a, "indptr"_a, "gcodes"_a, "hist"_a, nb::kw_only(),
        "group_sums"_a = nb::none(), "group_nnz"_a = nb::none(), "n_cells"_a,
        "n_genes"_a, "n_groups"_a, "n_bins"_a, "bin_low"_a, "inv_bin_width"_a,
        "col_start"_a = 0, "col_stop"_a = -1,
        "sub_batch_rows"_a = DEFAULT_SUB_BATCH,
        nb::call_guard<nb::gil_scoped_release>());
}

// ---- histogram: CSC host (column-window stream into the chunk hist) ----
// Chunk hist (col_stop-col_start, n_groups, nbt); sub-blocks write at their
// leading-dim offset (gene_start=0 on the rebased view).
template <typename T, typename IdxT, typename Device>
void def_hist_csc_host(nb::module_& m) {
    m.def(
        "hist_csc_host",
        [](host_array<const T> data, host_array<const IdxT> indices,
           host_array<const IdxT> indptr, gpu_array_c<const int, Device> gcodes,
           gpu_array_c<unsigned int, Device> hist,
           std::optional<gpu_array_c<double, Device>> group_sums,
           std::optional<gpu_array_c<double, Device>> group_nnz, int n_cells,
           int n_genes, int n_groups, int n_bins, double bin_low,
           double inv_bin_width, int col_start, int col_stop,
           int sub_batch_cols) {
            nb_require(
                col_start >= 0 && col_start <= col_stop && col_stop <= n_genes,
                "hist_csc_host: invalid column range");
            int nbt = n_bins + 1;
            const int* d_gcodes = gcodes.data();
            unsigned int* d_hist = hist.data();
            int window = col_stop - col_start;
            // Fused means: accumulate into group_sums[:, col] via full n_genes
            // stride + the window's global column offset.
            double* d_gsum = group_sums ? group_sums->data() : nullptr;
            double* d_gnnz = group_nnz ? group_nnz->data() : nullptr;
            int aggr_mask = (d_gsum ? AGGR_SUM : 0) | (d_gnnz ? AGGR_COUNT : 0);
            RmmScratchPool mask_pool;
            bool* d_mask = nullptr;
            if (aggr_mask) {
                d_mask = mask_pool.alloc<bool>((size_t)n_cells);
                cudaMemset(d_mask, 1, (size_t)n_cells);
            }
            auto consume = [&](int s, const T* d_data, const IdxT* d_indices,
                               const IdxT* d_indptr, int seg0, int n_block_seg,
                               int block_nnz, cudaStream_t stream) {
                (void)s;
                (void)block_nnz;
                unsigned int* hist_off = d_hist + (size_t)seg0 * n_groups * nbt;
                csc_hist_kernel<T, IdxT>
                    <<<(unsigned)n_block_seg, HIST_BLOCK, 0, stream>>>(
                        d_data, d_indices, d_indptr, d_gcodes, hist_off,
                        n_cells, n_block_seg, n_groups, n_bins, bin_low,
                        inv_bin_width, 0);
                CUDA_CHECK_LAST_ERROR(csc_hist_kernel);
                if (aggr_mask) {
                    size_t off = (size_t)(col_start + seg0);
                    double* gsum = d_gsum ? d_gsum + off : nullptr;
                    double* gnnz = d_gnnz ? d_gnnz + off : nullptr;
                    dim3 grid((unsigned)n_block_seg);
                    dim3 block(AGGR_BLOCK);
#define LAUNCH(M)                                                           \
    csc_aggr_kernel<T, IdxT, M><<<grid, block, 0, stream>>>(                \
        d_indptr, d_indices, d_data, gsum, gnnz, nullptr, d_gcodes, d_mask, \
        (size_t)n_cells, (size_t)n_genes);                                  \
    CUDA_CHECK_LAST_ERROR(csc_aggr_kernel)
                    RANK_STREAM_AGGR_DISPATCH(aggr_mask, LAUNCH);
#undef LAUNCH
                }
            };
            stream_sparse_blocks<T, IdxT>(data.data(), indices.data(),
                                          indptr.data() + col_start, window,
                                          sub_batch_cols, consume);
        },
        "data"_a, "indices"_a, "indptr"_a, "gcodes"_a, "hist"_a, nb::kw_only(),
        "group_sums"_a = nb::none(), "group_nnz"_a = nb::none(), "n_cells"_a,
        "n_genes"_a, "n_groups"_a, "n_bins"_a, "bin_low"_a, "inv_bin_width"_a,
        "col_start"_a, "col_stop"_a, "sub_batch_cols"_a = DEFAULT_SUB_BATCH,
        nb::call_guard<nb::gil_scoped_release>());
}

// ---- histogram: dense host (column-window; dense_hist needs whole columns.
// F-order: contiguous. C-order: strided 2D H2D to a C block, transposed to F)
// ----
template <typename T, typename Device, typename HostArray, bool FOrder>
void def_hist_dense_host(nb::module_& m) {
    m.def(
        "hist_dense_host",
        [](HostArray X, gpu_array_c<const int, Device> gcodes,
           gpu_array_c<unsigned int, Device> hist,
           std::optional<gpu_array_c<double, Device>> group_sums,
           std::optional<gpu_array_c<double, Device>> group_nnz, int n_groups,
           int n_bins, double bin_low, double inv_bin_width, int col_start,
           int col_stop, int sub_batch_cols) {
            int n_cells = (int)X.shape(0);
            int n_genes_full = (int)X.shape(1);
            nb_require(col_start >= 0 && col_start <= col_stop &&
                           col_stop <= n_genes_full,
                       "hist_dense_host: invalid column range");
            int nbt = n_bins + 1;
            const T* h_X = X.data();
            const int* d_gcodes = gcodes.data();
            unsigned int* d_hist = hist.data();
            double* d_gsum = group_sums ? group_sums->data() : nullptr;
            double* d_gnnz = group_nnz ? group_nnz->data() : nullptr;
            int aggr_mask = (d_gsum ? AGGR_SUM : 0) | (d_gnnz ? AGGR_COUNT : 0);
            int window = col_stop - col_start;
            if (window <= 0) return;
            RmmScratchPool mask_pool;
            bool* d_mask = nullptr;
            if (aggr_mask) {
                d_mask = mask_pool.alloc<bool>((size_t)n_cells);
                cudaMemset(d_mask, 1, (size_t)n_cells);
            }
            int sb = sub_batch_cols > 0 ? sub_batch_cols : DEFAULT_SUB_BATCH;
            if (sb > window) sb = window;
            size_t budget = rmm_available_device_bytes(0.8);
            size_t per_col = (size_t)n_cells * sizeof(T) * (FOrder ? 1 : 2);
            size_t mem_cols =
                budget / (size_t)N_STREAMS / (per_col ? per_col : 1);
            if (mem_cols >= 1 && (size_t)sb > mem_cols) sb = (int)mem_cols;
            if (sb < 1) sb = 1;

            // dense_aggr_kernel_F ties data layout to n_genes → fused sum uses
            // a per-window buffer scattered into group_sums[:, c].
            int n_sum_rows = d_gsum ? (int)group_sums->shape(0)
                                    : (d_gnnz ? (int)group_nnz->shape(0) : 0);
            RmmScratchPool pool;
            std::vector<T*> d_block(N_STREAMS, nullptr);
            std::vector<T*> d_blockF(N_STREAMS, nullptr);
            std::vector<double*> w_gsum(N_STREAMS, nullptr);
            std::vector<double*> w_gnnz(N_STREAMS, nullptr);
            for (int s = 0; s < N_STREAMS; s++) {
                d_block[s] = pool.alloc<T>((size_t)sb * n_cells);
                if (!FOrder) d_blockF[s] = pool.alloc<T>((size_t)sb * n_cells);
                if (d_gsum)
                    w_gsum[s] = pool.alloc<double>((size_t)n_sum_rows * sb);
                if (d_gnnz)
                    w_gnnz[s] = pool.alloc<double>((size_t)n_sum_rows * sb);
            }
            ScopedCudaStreams streams(N_STREAMS, cudaStreamDefault);
            // Pageable async copies (see stream_sparse_blocks) — no per-call
            // pin.
            cudaDeviceSynchronize();

            int b = 0;
            for (int c = col_start; c < col_stop; c += sb, b++) {
                int wc = (c + sb <= col_stop) ? sb : (col_stop - c);
                int s = b % N_STREAMS;
                cudaStream_t stream = streams[s];
                const T* dev_cols;
                if (FOrder) {
                    cuda_check(
                        cudaMemcpyAsync(d_block[s], h_X + (size_t)c * n_cells,
                                        (size_t)wc * n_cells * sizeof(T),
                                        cudaMemcpyHostToDevice, stream),
                        "hist_dense_host F window H2D");
                    dev_cols = d_block[s];
                } else {
                    cuda_check(cudaMemcpy2DAsync(
                                   d_block[s], (size_t)wc * sizeof(T), h_X + c,
                                   (size_t)n_genes_full * sizeof(T),
                                   (size_t)wc * sizeof(T), n_cells,
                                   cudaMemcpyHostToDevice, stream),
                               "hist_dense_host C window H2D");
                    dim3 tb(UTIL_BLOCK_SIZE);
                    dim3 tg(
                        strided_grid((long long)n_cells * wc, UTIL_BLOCK_SIZE));
                    transpose_c_to_f_kernel<T><<<tg, tb, 0, stream>>>(
                        d_block[s], d_blockF[s], n_cells, wc);
                    CUDA_CHECK_LAST_ERROR(transpose_c_to_f_kernel);
                    dev_cols = d_blockF[s];
                }
                unsigned int* hist_off =
                    d_hist + (size_t)(c - col_start) * n_groups * nbt;
                dense_hist_kernel<T><<<(unsigned)wc, HIST_BLOCK, 0, stream>>>(
                    dev_cols, d_gcodes, hist_off, n_cells, wc, n_groups, n_bins,
                    bin_low, inv_bin_width);
                CUDA_CHECK_LAST_ERROR(dense_hist_kernel);
                if (aggr_mask) {
                    // Per-window sum (n_genes=wc) then scatter into
                    // group_sums[:, c].
                    double* ws = d_gsum ? w_gsum[s] : nullptr;
                    double* wn = d_gnnz ? w_gnnz[s] : nullptr;
                    size_t plane = (size_t)n_sum_rows * wc;
                    if (ws)
                        cudaMemsetAsync(ws, 0, plane * sizeof(double), stream);
                    if (wn)
                        cudaMemsetAsync(wn, 0, plane * sizeof(double), stream);
                    dim3 ab(AGGR_BLOCK);
                    dim3 ag(strided_grid((long long)n_cells * wc, AGGR_BLOCK));
#define LAUNCH(M)                                                            \
    dense_aggr_kernel_F<T, M>                                                \
        <<<ag, ab, 0, stream>>>(dev_cols, ws, wn, nullptr, d_gcodes, d_mask, \
                                (size_t)n_cells, (size_t)wc);                \
    CUDA_CHECK_LAST_ERROR(dense_aggr_kernel_F)
                    RANK_STREAM_AGGR_DISPATCH(aggr_mask, LAUNCH);
#undef LAUNCH
                    if (ws)
                        scatter_cols_2d(d_gsum + c, ws, n_sum_rows,
                                        n_genes_full, wc, stream);
                    if (wn)
                        scatter_cols_2d(d_gnnz + c, wn, n_sum_rows,
                                        n_genes_full, wc, stream);
                }
            }
            sync_streams(streams, "hist_dense_host");
        },
        "X"_a, "gcodes"_a, "hist"_a, nb::kw_only(), "group_sums"_a = nb::none(),
        "group_nnz"_a = nb::none(), "n_groups"_a, "n_bins"_a, "bin_low"_a,
        "inv_bin_width"_a, "col_start"_a, "col_stop"_a,
        "sub_batch_cols"_a = DEFAULT_SUB_BATCH,
        nb::call_guard<nb::gil_scoped_release>());
}

template <typename Device>
void register_bindings(nb::module_& m) {
    def_aggr_csr_host<float, int, Device>(m);
    def_aggr_csr_host<double, int, Device>(m);
    def_aggr_csr_host<float, long long, Device>(m);
    def_aggr_csr_host<double, long long, Device>(m);

    def_aggr_csc_host<float, int, Device>(m);
    def_aggr_csc_host<double, int, Device>(m);
    def_aggr_csc_host<float, long long, Device>(m);
    def_aggr_csc_host<double, long long, Device>(m);

    def_aggr_dense_host<float, Device, host_array_c2<const float>, false>(m);
    def_aggr_dense_host<float, Device, host_array_f2<const float>, true>(m);
    def_aggr_dense_host<double, Device, host_array_c2<const double>, false>(m);
    def_aggr_dense_host<double, Device, host_array_f2<const double>, true>(m);

    def_hist_csr_host<float, int, Device>(m);
    def_hist_csr_host<double, int, Device>(m);
    def_hist_csr_host<float, long long, Device>(m);
    def_hist_csr_host<double, long long, Device>(m);

    def_hist_csc_host<float, int, Device>(m);
    def_hist_csc_host<double, int, Device>(m);
    def_hist_csc_host<float, long long, Device>(m);
    def_hist_csc_host<double, long long, Device>(m);

    def_hist_dense_host<float, Device, host_array_c2<const float>, false>(m);
    def_hist_dense_host<float, Device, host_array_f2<const float>, true>(m);
    def_hist_dense_host<double, Device, host_array_c2<const double>, false>(m);
    def_hist_dense_host<double, Device, host_array_f2<const double>, true>(m);
}

}  // namespace

NB_MODULE(_rank_stream_cuda, m) {
    m.def("_set_host_worker_limit", &set_host_worker_limit, "limit"_a);
    REGISTER_GPU_BINDINGS(register_bindings, m);
}
