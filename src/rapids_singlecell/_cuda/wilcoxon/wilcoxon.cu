#include <cstdint>
#include <type_traits>
#include <vector>

#include "wilcoxon_sparse.cuh"

using namespace nb::literals;

// nb::numpy does not restrict the device, so pin host overloads to the CPU.
template <typename T, typename Order>
using host_dense = nb::ndarray<T, nb::ndim<2>, Order, nb::device::cpu>;
template <typename T>
using host_vec = nb::ndarray<T, nb::ndim<1>, nb::c_contig, nb::device::cpu>;

// Dense batch width in [1, min(width, n_cols)] whose per-stream scratch
// (col_bytes per column) fits 80% of the free device memory.
static int fit_dense_width(int width, int n_cols, size_t col_bytes) {
    size_t cols =
        rmm_available_device_bytes(0.8) / std::max<size_t>(col_bytes, 1);
    return (int)std::max<size_t>(
        1, std::min({(size_t)width, (size_t)n_cols, cols}));
}

// Device view of dense column batch [col, col + sb_cols) of X (n_rows x ld):
// device input is used in place; host input is copied into `stage` (F-order
// contiguously, C-order as a 2D copy). Returns the C-order row stride.
template <typename T>
static int dense_batch(const T* X, bool on_host, bool f_order, int ld,
                       int n_rows, int col, int sb_cols, T* stage,
                       const T*& src, cudaStream_t stream) {
    src = X + (f_order ? (size_t)col * n_rows : (size_t)col);
    if (!on_host) return ld;
    if (f_order)
        cudaMemcpyAsync(stage, src, (size_t)n_rows * sb_cols * sizeof(T),
                        cudaMemcpyHostToDevice, stream);
    else
        cudaMemcpy2DAsync(stage, (size_t)sb_cols * sizeof(T), src,
                          (size_t)ld * sizeof(T), (size_t)sb_cols * sizeof(T),
                          n_rows, cudaMemcpyHostToDevice, stream);
    src = stage;
    return sb_cols;
}

// Dense OVR over column batches [col_start, col_start + n_cols) of X
// (n_rows x ld, F- or C-order, host or device). Host batches stream through
// async copies; F-order float32 batches feed the sort without a cast copy.
// f64 group stats read the native values; outputs are written in place at
// [group * n_cols + col] (stats outputs must start zeroed).
template <typename T>
static void ovr_dense(const T* X, bool on_host, bool f_order, int ld,
                      int col_start, const int* group_codes,
                      const double* group_sizes, double* rank_sums,
                      double* tie_corr, double* group_sums, double* group_nnz,
                      double* total_sums, double* total_nnz, int n_rows,
                      int n_cols, int n_groups, bool compute_tie_corr,
                      bool compute_nnz, bool compute_totals, int sub_batch_cols,
                      double expm1_scale, cudaStream_t upstream) {
    if (n_rows == 0 || n_cols == 0 || n_groups == 0) return;
    if (sub_batch_cols <= 0) sub_batch_cols = SUB_BATCH_COLS;
    const bool direct_keys = f_order && std::is_same<T, float>::value;
    // Per item: host stage, keys in/out, rows in/out and CUB's pair buffers.
    sub_batch_cols = fit_dense_width(
        sub_batch_cols, n_cols,
        (size_t)n_rows *
            ((on_host ? sizeof(T) : 0) + (direct_keys ? 0 : sizeof(float)) +
             sizeof(float) + 2 * sizeof(int) + 2 * sizeof(float)));

    DenseColumnBatchPlan batches = plan_dense_column_batches(
        n_rows, n_cols, sub_batch_cols, SAFE_BATCH_NNZ, "Dense OVR sub-batch");
    sub_batch_cols = batches.sub_batch_cols;
    size_t sub_items = batches.max_items;
    size_t cub_temp_bytes = cub_segmented_sortpairs_temp_bytes(
        checked_cub_items(sub_items, "Dense OVR sub-batch"), sub_batch_cols);
    size_t per_stream = sub_items * ((on_host ? sizeof(T) : 0) +
                                     (direct_keys ? 0 : sizeof(float)) +
                                     sizeof(float) + sizeof(int)) +
                        cub_temp_bytes;
    int n_streams =
        clamp_streams_by_budget(clamp_streams_by_cols(n_cols, sub_batch_cols),
                                per_stream, rmm_available_device_bytes(0.8));

    // pool first: streams drain before it frees their scratch (see guard doc).
    RmmScratchPool pool;
    ScopedCudaStreams streams(n_streams, cudaStreamNonBlocking);
    // CUB only reads the iota rows and linear offsets: fill once, share.
    int* rows_in = pool.alloc<int>(sub_items);
    int* offsets = pool.alloc<int>(sub_batch_cols + 1);
    fill_row_indices_kernel<<<sub_batch_cols, UTIL_BLOCK_SIZE, 0, upstream>>>(
        rows_in, n_rows, sub_batch_cols);
    CUDA_CHECK_LAST_ERROR(fill_row_indices_kernel);
    upload_linear_offsets(offsets, sub_batch_cols, n_rows, upstream);
    ScopedCudaEvent inputs_ready(cudaEventDisableTiming);
    inputs_ready.record(upstream);

    struct StreamBuf {
        T* stage;
        float* keys_in;
        float* keys_out;
        int* rows_out;
        uint8_t* cub_temp;
    };
    std::vector<StreamBuf> bufs(n_streams);
    for (int s = 0; s < n_streams; ++s) {
        cuda_check(cudaStreamWaitEvent(streams[s], inputs_ready.get(), 0),
                   "dense OVR wait for inputs");
        bufs[s].stage = on_host ? pool.alloc<T>(sub_items) : nullptr;
        bufs[s].keys_in = direct_keys ? nullptr : pool.alloc<float>(sub_items);
        bufs[s].keys_out = pool.alloc<float>(sub_items);
        bufs[s].rows_out = pool.alloc<int>(sub_items);
        bufs[s].cub_temp = pool.alloc<uint8_t>(cub_temp_bytes);
    }

    for (int col = 0, b = 0; col < n_cols; col += sub_batch_cols, ++b) {
        int sb_cols = std::min(sub_batch_cols, n_cols - col);
        int sb_items = n_rows * sb_cols;
        cudaStream_t stream = streams[b % n_streams];
        auto& buf = bufs[b % n_streams];
        const T* src;
        int row_stride =
            dense_batch(X, on_host, f_order, ld, n_rows, col_start + col,
                        sb_cols, buf.stage, src, stream);
        const float* keys = reinterpret_cast<const float*>(src);
        if (!direct_keys) {
            launch_dense_gather(src, nullptr, buf.keys_in, n_rows, n_rows,
                                sb_cols, f_order, row_stride, stream);
            keys = buf.keys_in;
        }
        dense_group_accumulate_kernel<T>
            <<<dim3((n_rows + 31) / 32, (sb_cols + 31) / 32), dim3(32, 8), 0,
               stream>>>(src, group_codes, group_sums + col, group_nnz + col,
                         total_sums + col, total_nnz + col, n_cols, n_rows,
                         sb_cols, n_groups, f_order, row_stride, compute_nnz,
                         compute_totals, expm1_scale);
        CUDA_CHECK_LAST_ERROR(dense_group_accumulate_kernel);
        cub_segmented_sortpairs(buf.cub_temp, cub_temp_bytes, keys,
                                buf.keys_out, rows_in, buf.rows_out, sb_items,
                                sb_cols, offsets, offsets + 1, stream,
                                "dense OVR segmented sort");
        launch_ovr_rank<int>(buf.keys_out, buf.rows_out, offsets, group_codes,
                             group_sizes, rank_sums + col, tie_corr + col,
                             n_cols, n_rows, sb_cols, n_groups,
                             compute_tie_corr, stream);
    }
    sync_streams(streams, "dense OVR");
}

// Dense OVO over column batches of X (layout as in ovr_dense): gather the
// reference and group rows of each batch into F-order float slabs, sort the
// reference and rank the groups against it. f64 stats are per-code group
// sums / nnz of the native values ([code * n_cols + col], zeroed).
template <typename T>
static void ovo_dense(const T* X, bool on_host, bool f_order, int ld,
                      int col_start, const int* ref_rows, int n_ref,
                      const int* grp_rows, int n_all_grp,
                      const int* grp_offsets, const int* h_offsets,
                      int n_groups, const int* group_codes, int n_stat_groups,
                      double* rank_sums, double* tie_corr, double* group_sums,
                      double* group_nnz, int n_rows, int n_cols,
                      bool compute_tie_corr, bool compute_nnz,
                      int sub_batch_cols, double expm1_scale,
                      cudaStream_t upstream) {
    if (n_cols == 0 || n_ref == 0 || n_all_grp == 0 || n_groups == 0) return;
    if (sub_batch_cols <= 0) sub_batch_cols = SUB_BATCH_COLS;
    OvoRanker ranker(h_offsets, n_groups, compute_tie_corr);
    sub_batch_cols =
        fit_dense_width(sub_batch_cols, n_cols,
                        (on_host ? (size_t)n_rows * sizeof(T) : 0) +
                            (3 * (size_t)n_ref + n_all_grp) * sizeof(float) +
                            ranker.scratch_bytes(n_all_grp, 1));

    DenseColumnBatchPlan batches = plan_dense_column_batches(
        n_rows, n_cols, sub_batch_cols, SAFE_BATCH_NNZ, "Dense OVO sub-batch");
    sub_batch_cols = batches.sub_batch_cols;
    size_t ref_items = (size_t)n_ref * sub_batch_cols;
    size_t grp_items = (size_t)n_all_grp * sub_batch_cols;
    size_t ref_cub_bytes = cub_segmented_sortkeys_temp_bytes(
        checked_cub_items(ref_items, "Dense OVO reference sub-batch"),
        sub_batch_cols);
    size_t per_stream = (on_host ? batches.max_items * sizeof(T) : 0) +
                        (2 * ref_items + grp_items) * sizeof(float) +
                        ref_cub_bytes +
                        ranker.scratch_bytes(grp_items, sub_batch_cols);
    int n_streams =
        clamp_streams_by_budget(clamp_streams_by_cols(n_cols, sub_batch_cols),
                                per_stream, rmm_available_device_bytes(0.8));

    // pool first: streams drain before it frees their scratch (see guard doc).
    RmmScratchPool pool;
    ScopedCudaStreams streams(n_streams, cudaStreamNonBlocking);
    ranker.upload_huge_ids(pool);
    int* ref_offsets = pool.alloc<int>(sub_batch_cols + 1);
    upload_linear_offsets(ref_offsets, sub_batch_cols, n_ref, upstream);
    ScopedCudaEvent inputs_ready(cudaEventDisableTiming);
    inputs_ready.record(upstream);

    struct StreamBuf {
        T* stage;
        float* ref;
        float* ref_sorted;
        float* grp;
        uint8_t* ref_cub;
        OvoScratch ovo;
    };
    std::vector<StreamBuf> bufs(n_streams);
    for (int s = 0; s < n_streams; ++s) {
        cuda_check(cudaStreamWaitEvent(streams[s], inputs_ready.get(), 0),
                   "dense OVO wait for inputs");
        bufs[s].stage = on_host ? pool.alloc<T>(batches.max_items) : nullptr;
        bufs[s].ref = pool.alloc<float>(ref_items);
        bufs[s].ref_sorted = pool.alloc<float>(ref_items);
        bufs[s].grp = pool.alloc<float>(grp_items);
        bufs[s].ref_cub = pool.alloc<uint8_t>(ref_cub_bytes);
        bufs[s].ovo = ranker.alloc(pool, grp_items, sub_batch_cols);
    }

    for (int col = 0, b = 0; col < n_cols; col += sub_batch_cols, ++b) {
        int sb_cols = std::min(sub_batch_cols, n_cols - col);
        cudaStream_t stream = streams[b % n_streams];
        auto& buf = bufs[b % n_streams];
        const T* src;
        int row_stride =
            dense_batch(X, on_host, f_order, ld, n_rows, col_start + col,
                        sb_cols, buf.stage, src, stream);
        launch_dense_gather(src, ref_rows, buf.ref, n_ref, n_rows, sb_cols,
                            f_order, row_stride, stream);
        launch_dense_gather(src, grp_rows, buf.grp, n_all_grp, n_rows, sb_cols,
                            f_order, row_stride, stream);
        dense_group_accumulate_kernel<T>
            <<<dim3((n_rows + 31) / 32, (sb_cols + 31) / 32), dim3(32, 8), 0,
               stream>>>(src, group_codes, group_sums + col, group_nnz + col,
                         nullptr, nullptr, n_cols, n_rows, sb_cols,
                         n_stat_groups, f_order, row_stride, compute_nnz, false,
                         expm1_scale);
        CUDA_CHECK_LAST_ERROR(dense_group_accumulate_kernel);
        cub_segmented_sortkeys(buf.ref_cub, ref_cub_bytes, buf.ref,
                               buf.ref_sorted, n_ref * sb_cols, sb_cols,
                               ref_offsets, ref_offsets + 1, stream,
                               "dense OVO ref segmented sort");
        ranker.rank(buf.ref_sorted, buf.grp, grp_offsets, buf.ovo,
                    rank_sums + col, tie_corr + col, n_cols, n_ref, n_all_grp,
                    sb_cols, stream);
    }
    sync_streams(streams, "dense OVO");
}

// The bindings take n_groups x n_cols from rank_sums and the cell count from
// the input; check what the kernels index by those sizes so a bad call raises
// instead of reading or writing out of bounds. OVO stats also hold the
// reference group, OVO tie_corr is per group, and OVR totals are per column.
template <typename Out>
static void check_shapes(const Out& rank_sums, const Out& tie_corr,
                         const Out& group_sums, const Out& group_nnz,
                         const Out* total_sums, const Out* total_nnz,
                         size_t n_codes, size_t n_rows, bool ovo, bool tie,
                         bool nnz) {
    size_t g = rank_sums.shape(0), c = rank_sums.shape(1);
    size_t stat_groups = group_sums.ndim() == 2 ? group_sums.shape(0) : 0;
    nb_require(rank_sums.ndim() == 2 && n_codes == n_rows &&
                   group_sums.size() == stat_groups * c &&
                   (ovo ? stat_groups > g : stat_groups == g) &&
                   (!nnz || group_nnz.size() == group_sums.size()) &&
                   (!tie || tie_corr.size() == (ovo ? g * c : c)) &&
                   (!total_sums || (total_sums->size() == c &&
                                    (!nnz || total_nnz->size() == c))),
               "Wilcoxon: group_codes, statistics or tie arrays do not match "
               "rank_sums and the cell count");
}

// OVO group offsets [n_groups + 1] to the host, checked against the rows.
template <typename Ints>
static std::vector<int> ovo_offsets(const Ints& grp_offsets, size_t n_groups,
                                    size_t n_grp_rows, cudaStream_t stream) {
    nb_require(grp_offsets.shape(0) == n_groups + 1,
               "Wilcoxon OVO: grp_offsets must have n_groups + 1 entries");
    std::vector<int> h(n_groups + 1);
    cuda_check(cudaMemcpyAsync(h.data(), grp_offsets.data(),
                               (n_groups + 1) * sizeof(int),
                               cudaMemcpyDeviceToHost, stream),
               "OVO group offsets D2H");
    cuda_check(cudaStreamSynchronize(stream), "OVO group offsets sync");
    nb_require(h[0] == 0 && (size_t)h[n_groups] == n_grp_rows,
               "Wilcoxon OVO: grp_offsets must span the group rows");
    return h;
}

// Sparse inputs: data and indices pair up, indptr spans the compressed axis
// and host CSR row spans cover every row.
template <typename Data, typename Index>
static void check_sparse(const Data& data, const Index& indices,
                         const Index& indptr, const Index& row_starts,
                         const Index& row_stops, bool csc, bool on_host,
                         size_t n_rows, size_t n_cols) {
    nb_require(
        data.shape(0) == indices.shape(0) &&
            indptr.shape(0) == (csc ? n_cols : n_rows) + 1 &&
            (csc || !on_host ||
             (row_starts.shape(0) == n_rows && row_stops.shape(0) == n_rows)),
        "Wilcoxon sparse: data, indices, indptr or row spans do not "
        "match the matrix shape");
}

template <typename T, typename Array, bool OnHost, bool FOrder, typename Device>
static void def_ovr_dense(nb::module_& m) {
    m.def(
        "ovr_dense",
        [](Array X, gpu_array_c<const int, Device> group_codes,
           gpu_array_c<const double, Device> group_sizes,
           gpu_array_c<double, Device> rank_sums,
           gpu_array_c<double, Device> tie_corr,
           gpu_array_c<double, Device> group_sums,
           gpu_array_c<double, Device> group_nnz,
           gpu_array_c<double, Device> total_sums,
           gpu_array_c<double, Device> total_nnz, bool compute_tie_corr,
           bool compute_nnz, bool compute_totals, int col_start,
           int sub_batch_cols, double expm1_scale, std::uintptr_t stream) {
            check_shapes(rank_sums, tie_corr, group_sums, group_nnz,
                         compute_totals ? &total_sums : nullptr, &total_nnz,
                         group_codes.shape(0), X.shape(0), false,
                         compute_tie_corr, compute_nnz);
            ovr_dense<T>(X.data(), OnHost, FOrder, (int)X.shape(1), col_start,
                         group_codes.data(), group_sizes.data(),
                         rank_sums.data(), tie_corr.data(), group_sums.data(),
                         group_nnz.data(), total_sums.data(), total_nnz.data(),
                         (int)X.shape(0), (int)rank_sums.shape(1),
                         (int)rank_sums.shape(0), compute_tie_corr, compute_nnz,
                         compute_totals, sub_batch_cols, expm1_scale,
                         (cudaStream_t)stream);
        },
        "X"_a, "group_codes"_a, "group_sizes"_a, "rank_sums"_a, "tie_corr"_a,
        "group_sums"_a, "group_nnz"_a, "total_sums"_a, "total_nnz"_a,
        nb::kw_only(), "compute_tie_corr"_a, "compute_nnz"_a,
        "compute_totals"_a, "col_start"_a = 0,
        "sub_batch_cols"_a = SUB_BATCH_COLS, "expm1_scale"_a = 0.0,
        "stream"_a = 0, nb::call_guard<nb::gil_scoped_release>());
}

template <typename T, typename Array, bool OnHost, bool FOrder, typename Device>
static void def_ovo_dense(nb::module_& m) {
    m.def(
        "ovo_dense",
        [](Array X, gpu_array_c<const int, Device> ref_rows,
           gpu_array_c<const int, Device> grp_rows,
           gpu_array_c<const int, Device> grp_offsets,
           gpu_array_c<const int, Device> group_codes,
           gpu_array_c<double, Device> rank_sums,
           gpu_array_c<double, Device> tie_corr,
           gpu_array_c<double, Device> group_sums,
           gpu_array_c<double, Device> group_nnz, bool compute_tie_corr,
           bool compute_nnz, int col_start, int sub_batch_cols,
           double expm1_scale, std::uintptr_t stream) {
            check_shapes<gpu_array_c<double, Device>>(
                rank_sums, tie_corr, group_sums, group_nnz, nullptr, nullptr,
                group_codes.shape(0), X.shape(0), true, compute_tie_corr,
                compute_nnz);
            std::vector<int> h_offsets =
                ovo_offsets(grp_offsets, rank_sums.shape(0), grp_rows.shape(0),
                            (cudaStream_t)stream);
            ovo_dense<T>(
                X.data(), OnHost, FOrder, (int)X.shape(1), col_start,
                ref_rows.data(), (int)ref_rows.shape(0), grp_rows.data(),
                (int)grp_rows.shape(0), grp_offsets.data(), h_offsets.data(),
                (int)rank_sums.shape(0), group_codes.data(),
                (int)group_sums.shape(0), rank_sums.data(), tie_corr.data(),
                group_sums.data(), group_nnz.data(), (int)X.shape(0),
                (int)rank_sums.shape(1), compute_tie_corr, compute_nnz,
                sub_batch_cols, expm1_scale, (cudaStream_t)stream);
        },
        "X"_a, "ref_rows"_a, "grp_rows"_a, "grp_offsets"_a, "group_codes"_a,
        "rank_sums"_a, "tie_corr"_a, "group_sums"_a, "group_nnz"_a,
        nb::kw_only(), "compute_tie_corr"_a, "compute_nnz"_a, "col_start"_a = 0,
        "sub_batch_cols"_a = SUB_BATCH_COLS, "expm1_scale"_a = 0.0,
        "stream"_a = 0, nb::call_guard<nb::gil_scoped_release>());
}

template <typename IndexT, typename IndptrT, typename Device>
static void def_csr_column_range_indptr_device(nb::module_& m) {
    m.def(
        "csr_column_range_indptr_device",
        [](gpu_array_c<const IndexT, Device> indices,
           gpu_array_c<const IndptrT, Device> indptr,
           gpu_array_c<IndptrT, Device> local_indptr, int col_start,
           int col_stop, std::uintptr_t stream) -> int64_t {
            nb_require(indptr.ndim() == 1 && indices.ndim() == 1 &&
                           local_indptr.ndim() == 1,
                       "csr_column_range_indptr_device: arrays must be 1D");
            nb_require(
                indptr.shape(0) >= 1 &&
                    indptr.shape(0) <= (size_t)std::numeric_limits<int>::max(),
                "csr_column_range_indptr_device: invalid indptr length");
            nb_require(local_indptr.shape(0) == indptr.shape(0),
                       "csr_column_range_indptr_device: local_indptr must "
                       "match indptr length");
            nb_require(col_start >= 0 && col_start <= col_stop,
                       "csr_column_range_indptr_device: invalid column range");

            int n_rows = (int)indptr.shape(0) - 1;
            cudaStream_t cuda_stream = (cudaStream_t)stream;
            cuda_check(cudaMemsetAsync(local_indptr.data(), 0, sizeof(IndptrT),
                                       cuda_stream),
                       "CSR column range local indptr zero");
            IndptrT local_nnz = 0;
            if (n_rows > 0) {
                constexpr int BLOCK_SIZE = 256;
                unsigned int grid = strided_grid(n_rows, BLOCK_SIZE);
                csr_column_range_count_kernel<<<grid, BLOCK_SIZE, 0,
                                                cuda_stream>>>(
                    indices.data(), indptr.data(), local_indptr.data(), n_rows,
                    col_start, col_stop);
                CUDA_CHECK_LAST_ERROR(csr_column_range_count_kernel);

                size_t scan_bytes = 0;
                cuda_check(cub::DeviceScan::InclusiveSum(
                               nullptr, scan_bytes, local_indptr.data() + 1,
                               local_indptr.data() + 1, n_rows, cuda_stream),
                           "CSR column range scan temp-size query");
                ScopedCudaBuffer scan_temp(scan_bytes);
                cuda_check(
                    cub::DeviceScan::InclusiveSum(
                        scan_temp.data(), scan_bytes, local_indptr.data() + 1,
                        local_indptr.data() + 1, n_rows, cuda_stream),
                    "CSR column range indptr scan");
                cuda_check(
                    cudaMemcpyAsync(&local_nnz, local_indptr.data() + n_rows,
                                    sizeof(IndptrT), cudaMemcpyDeviceToHost,
                                    cuda_stream),
                    "CSR column range nnz D2H");
                cuda_check(cudaStreamSynchronize(cuda_stream),
                           "CSR column range indptr sync");
            } else {
                cuda_check(cudaStreamSynchronize(cuda_stream),
                           "CSR column range empty indptr sync");
            }
            nb_require(local_nnz >= 0,
                       "csr_column_range_indptr_device: local nnz overflow");
            return (int64_t)local_nnz;
        },
        "indices"_a, "indptr"_a, "local_indptr"_a, nb::kw_only(), "col_start"_a,
        "col_stop"_a, "stream"_a = 0);
}

template <typename InT, typename IndexT, typename IndptrT, typename Device>
static void def_csr_column_range_gather_device(nb::module_& m) {
    m.def(
        "csr_column_range_gather_device",
        [](gpu_array_c<const InT, Device> data,
           gpu_array_c<const IndexT, Device> indices,
           gpu_array_c<const IndptrT, Device> indptr,
           gpu_array_c<const IndptrT, Device> local_indptr,
           gpu_array_c<InT, Device> local_data,
           gpu_array_c<IndexT, Device> local_indices, int col_start,
           int col_stop, std::uintptr_t stream) {
            nb_require(data.ndim() == 1 && indices.ndim() == 1 &&
                           indptr.ndim() == 1 && local_indptr.ndim() == 1 &&
                           local_data.ndim() == 1 && local_indices.ndim() == 1,
                       "csr_column_range_gather_device: arrays must be 1D");
            nb_require(data.shape(0) == indices.shape(0),
                       "csr_column_range_gather_device: data and indices "
                       "lengths must match");
            nb_require(
                indptr.shape(0) >= 1 &&
                    indptr.shape(0) <= (size_t)std::numeric_limits<int>::max(),
                "csr_column_range_gather_device: invalid indptr length");
            nb_require(local_indptr.shape(0) == indptr.shape(0),
                       "csr_column_range_gather_device: local_indptr must "
                       "match indptr length");
            size_t local_nnz = local_data.shape(0);
            nb_require(local_nnz == local_indices.shape(0),
                       "csr_column_range_gather_device: output lengths must "
                       "match");
            nb_require(col_start >= 0 && col_start <= col_stop,
                       "csr_column_range_gather_device: invalid column range");

            int n_rows = (int)indptr.shape(0) - 1;
            if (n_rows == 0 || local_nnz == 0) return;
            constexpr int BLOCK_SIZE = 256;
            unsigned int grid = strided_grid(n_rows, 1);
            csr_column_range_gather_kernel<<<grid, BLOCK_SIZE, 0,
                                             (cudaStream_t)stream>>>(
                data.data(), indices.data(), indptr.data(), local_indptr.data(),
                local_data.data(), local_indices.data(), n_rows, col_start,
                col_stop);
            CUDA_CHECK_LAST_ERROR(csr_column_range_gather_kernel);
        },
        "data"_a, "indices"_a, "indptr"_a, "local_indptr"_a, "local_data"_a,
        "local_indices"_a, nb::kw_only(), "col_start"_a, "col_stop"_a,
        "stream"_a = 0);
}

template <typename IndexT, typename IndptrT>
static void def_csr_row_boundaries_host(nb::module_& m) {
    m.def(
        "csr_row_boundaries_host",
        [](host_vec<const IndexT> h_indices, host_vec<const IndptrT> h_indptr,
           host_vec<const int> h_col_cuts, host_array_c2<IndptrT> h_boundaries,
           int n_cols) {
            nb_require(n_cols >= 0,
                       "csr_row_boundaries_host: n_cols must be nonnegative");
            nb_require((int)h_indptr.shape(0) >= 1,
                       "csr_row_boundaries_host: indptr must not be empty");
            int n_rows = (int)h_indptr.shape(0) - 1;
            int n_cuts = (int)h_col_cuts.shape(0);
            nb_require((int)h_boundaries.shape(0) == n_cuts &&
                           (int)h_boundaries.shape(1) == n_rows,
                       "csr_row_boundaries_host: boundaries shape must be "
                       "(n_cuts, n_rows)");
            for (int cut = 0; cut < n_cuts; cut++) {
                int value = h_col_cuts.data()[cut];
                nb_require(value >= 0 && value <= n_cols,
                           "csr_row_boundaries_host: cuts must be within "
                           "[0, n_cols]");
                if (cut > 0) {
                    nb_require(value >= h_col_cuts.data()[cut - 1],
                               "csr_row_boundaries_host: cuts must be sorted");
                }
            }
            const IndexT* indices = h_indices.data();
            const IndptrT* indptr = h_indptr.data();
            IndptrT* boundaries = h_boundaries.data();
            host_parallel_ranges(n_rows, [&](int r0, int r1) {
                for (int row = r0; row < r1; row++) {
                    const IndexT* lo = indices + indptr[row];
                    const IndexT* hi = indices + indptr[row + 1];
                    for (int cut = 0; cut < n_cuts; cut++) {
                        lo = interp_lower_bound(lo, hi,
                                                (IndexT)h_col_cuts.data()[cut]);
                        boundaries[(size_t)cut * n_rows + row] =
                            (IndptrT)(lo - indices);
                    }
                }
            });
        },
        "h_indices"_a, "h_indptr"_a, "h_col_cuts"_a, "h_boundaries"_a,
        nb::kw_only(), "n_cols"_a, nb::call_guard<nb::gil_scoped_release>());
}

template <typename IndexT>
static void def_has_canonical_format_host(nb::module_& m) {
    m.def(
        "has_canonical_format_host",
        [](host_vec<const IndexT> h_indptr, host_vec<const IndexT> h_indices) {
            size_t n = h_indptr.shape(0);
            nb_require(n >= 1 && h_indptr.data()[0] >= 0 &&
                           (size_t)h_indptr.data()[n - 1] <= h_indices.shape(0),
                       "has_canonical_format_host: indptr must index into "
                       "indices");
            return has_canonical_format_host(h_indptr.data(), h_indices.data(),
                                             (int64_t)n - 1);
        },
        "indptr"_a.noconvert(), "indices"_a.noconvert(),
        nb::call_guard<nb::gil_scoped_release>());
}

template <typename InT, typename IndexT, bool OnHost, typename Ranker>
static void run_sparse(bool csc, const InT* data, const IndexT* indices,
                       const IndexT* indptr, const IndexT* row_starts,
                       const IndexT* row_stops, int col_start, int n_rows,
                       int n_cols, int sub_batch_cols, const Ranker& rank,
                       const SparseStats& stats) {
    if (sub_batch_cols <= 0) sub_batch_cols = SUB_BATCH_COLS;
    if (csc)
        sparse_csc(data, indices, indptr, OnHost, n_rows, n_cols,
                   sub_batch_cols, rank, stats);
    else if constexpr (OnHost)
        sparse_csr_host(data, indices, indptr, row_starts, row_stops, col_start,
                        n_rows, n_cols, sub_batch_cols, rank, stats);
    else if constexpr (std::is_same_v<Ranker, OvoBatchRanker>)
        ovo_csr_device(data, indices, indptr, n_rows, n_cols, sub_batch_cols,
                       rank, stats);
    else
        sparse_csr_device(data, indices, indptr, n_rows, n_cols, sub_batch_cols,
                          rank, stats);
}

// Host and device overloads differ only in where the sparse arrays live.
template <typename T, bool OnHost, typename Device>
using sparse_vec =
    std::conditional_t<OnHost, host_vec<T>, gpu_array_c<T, Device>>;

// Row spans (host CSR column shards) are ignored elsewhere. Outputs are
// [group * n_cols + col]; stats are per group code and must start zeroed.
template <typename InT, typename IndexT, bool OnHost, typename Device>
static void def_sparse(nb::module_& m) {
    using Data = sparse_vec<const InT, OnHost, Device>;
    using Index = sparse_vec<const IndexT, OnHost, Device>;
    using Ints = gpu_array_c<const int, Device>;
    using Out = gpu_array_c<double, Device>;
    m.def(
        "ovr_sparse",
        [](Data data, Index indices, Index indptr, Index row_starts,
           Index row_stops, Ints group_codes,
           gpu_array_c<const double, Device> group_sizes, Out rank_sums,
           Out tie_corr, Out group_sums, Out group_nnz, Out total_sums,
           Out total_nnz, bool csc, bool compute_tie_corr, bool compute_nnz,
           bool compute_totals, int col_start, int sub_batch_cols,
           double expm1_scale) {
            int n_rows = (int)group_codes.shape(0);
            int n_groups = (int)rank_sums.shape(0);
            int n_cols = (int)rank_sums.shape(1);
            check_shapes(rank_sums, tie_corr, group_sums, group_nnz,
                         compute_totals ? &total_sums : nullptr, &total_nnz,
                         group_codes.shape(0), group_codes.shape(0), false,
                         compute_tie_corr, compute_nnz);
            check_sparse(data, indices, indptr, row_starts, row_stops, csc,
                         OnHost, n_rows, n_cols);
            OvrBatchRanker rank{group_codes.data(), group_sizes.data(),
                                n_groups,           n_rows,
                                compute_tie_corr,   rank_sums.data(),
                                tie_corr.data(),    n_cols};
            SparseStats stats{group_codes.data(),
                              n_groups,
                              group_sums.data(),
                              group_nnz.data(),
                              total_sums.data(),
                              total_nnz.data(),
                              n_cols,
                              compute_nnz,
                              compute_totals,
                              expm1_scale};
            run_sparse<InT, IndexT, OnHost>(
                csc, data.data(), indices.data(), indptr.data(),
                row_starts.data(), row_stops.data(), col_start, n_rows, n_cols,
                sub_batch_cols, rank, stats);
        },
        "data"_a, "indices"_a, "indptr"_a, "row_starts"_a, "row_stops"_a,
        "group_codes"_a, "group_sizes"_a, "rank_sums"_a, "tie_corr"_a,
        "group_sums"_a, "group_nnz"_a, "total_sums"_a, "total_nnz"_a,
        nb::kw_only(), "csc"_a, "compute_tie_corr"_a, "compute_nnz"_a,
        "compute_totals"_a, "col_start"_a = 0,
        "sub_batch_cols"_a = SUB_BATCH_COLS, "expm1_scale"_a = 0.0,
        nb::call_guard<nb::gil_scoped_release>());
    m.def(
        "ovo_sparse",
        [](Data data, Index indices, Index indptr, Index row_starts,
           Index row_stops, Ints group_codes, Ints ref_map, Ints grp_map,
           Ints ref_rows, Ints grp_rows, Ints grp_offsets, Out rank_sums,
           Out tie_corr, Out group_sums, Out group_nnz, int n_ref, bool csc,
           bool compute_tie_corr, bool compute_nnz, int col_start,
           int sub_batch_cols, double expm1_scale) {
            int n_groups = (int)rank_sums.shape(0);
            int n_cols = (int)rank_sums.shape(1);
            size_t n_rows = ref_map.shape(0);
            check_shapes<Out>(rank_sums, tie_corr, group_sums, group_nnz,
                              nullptr, nullptr, group_codes.shape(0), n_rows,
                              true, compute_tie_corr, compute_nnz);
            check_sparse(data, indices, indptr, row_starts, row_stops, csc,
                         OnHost, n_rows, n_cols);
            nb_require(grp_map.shape(0) == n_rows &&
                           ref_rows.shape(0) == (size_t)n_ref,
                       "Wilcoxon OVO: row maps or reference rows do not match");
            std::vector<int> h_offsets =
                ovo_offsets(grp_offsets, n_groups, grp_rows.shape(0), 0);
            if (n_ref == 0 || h_offsets[n_groups] == 0) return;
            OvoRanker ranker(h_offsets.data(), n_groups, compute_tie_corr);
            RmmScratchPool pool;
            ranker.upload_huge_ids(pool);
            OvoBatchRanker rank{
                ref_map.data(),     grp_map.data(), ref_rows.data(),
                grp_rows.data(),    n_ref,          h_offsets[n_groups],
                grp_offsets.data(), &ranker,        rank_sums.data(),
                tie_corr.data(),    n_cols};
            SparseStats stats{group_codes.data(),
                              (int)group_sums.shape(0),
                              group_sums.data(),
                              group_nnz.data(),
                              nullptr,
                              nullptr,
                              n_cols,
                              compute_nnz,
                              false,
                              expm1_scale};
            run_sparse<InT, IndexT, OnHost>(
                csc, data.data(), indices.data(), indptr.data(),
                row_starts.data(), row_stops.data(), col_start, (int)n_rows,
                n_cols, sub_batch_cols, rank, stats);
        },
        "data"_a, "indices"_a, "indptr"_a, "row_starts"_a, "row_stops"_a,
        "group_codes"_a, "ref_map"_a, "grp_map"_a, "ref_rows"_a, "grp_rows"_a,
        "grp_offsets"_a, "rank_sums"_a, "tie_corr"_a, "group_sums"_a,
        "group_nnz"_a, nb::kw_only(), "n_ref"_a, "csc"_a, "compute_tie_corr"_a,
        "compute_nnz"_a, "col_start"_a = 0, "sub_batch_cols"_a = SUB_BATCH_COLS,
        "expm1_scale"_a = 0.0, nb::call_guard<nb::gil_scoped_release>());
}

template <typename Device>
void register_bindings(nb::module_& m) {
    m.doc() = "CUDA kernels for the Wilcoxon rank-sum test";

    def_ovr_dense<float, host_dense<const float, nb::c_contig>, true, false,
                  Device>(m);
    def_ovr_dense<float, host_dense<const float, nb::f_contig>, true, true,
                  Device>(m);
    def_ovr_dense<double, host_dense<const double, nb::c_contig>, true, false,
                  Device>(m);
    def_ovr_dense<double, host_dense<const double, nb::f_contig>, true, true,
                  Device>(m);
    def_ovr_dense<float, gpu_array_c<const float, Device>, false, false,
                  Device>(m);
    def_ovr_dense<float, gpu_array_f<const float, Device>, false, true, Device>(
        m);
    def_ovr_dense<double, gpu_array_c<const double, Device>, false, false,
                  Device>(m);
    def_ovr_dense<double, gpu_array_f<const double, Device>, false, true,
                  Device>(m);
    def_ovo_dense<float, host_dense<const float, nb::c_contig>, true, false,
                  Device>(m);
    def_ovo_dense<float, host_dense<const float, nb::f_contig>, true, true,
                  Device>(m);
    def_ovo_dense<double, host_dense<const double, nb::c_contig>, true, false,
                  Device>(m);
    def_ovo_dense<double, host_dense<const double, nb::f_contig>, true, true,
                  Device>(m);
    def_ovo_dense<float, gpu_array_c<const float, Device>, false, false,
                  Device>(m);
    def_ovo_dense<float, gpu_array_f<const float, Device>, false, true, Device>(
        m);
    def_ovo_dense<double, gpu_array_c<const double, Device>, false, false,
                  Device>(m);
    def_ovo_dense<double, gpu_array_f<const double, Device>, false, true,
                  Device>(m);

    def_sparse<float, int, false, Device>(m);
    def_sparse<double, int, false, Device>(m);
    def_sparse<float, int64_t, false, Device>(m);
    def_sparse<double, int64_t, false, Device>(m);
    def_sparse<float, int, true, Device>(m);
    def_sparse<double, int, true, Device>(m);
    def_sparse<float, int64_t, true, Device>(m);
    def_sparse<double, int64_t, true, Device>(m);

    // Multi-GPU device CSR column shards.
    def_csr_column_range_indptr_device<int, int, Device>(m);
    def_csr_column_range_indptr_device<int64_t, int64_t, Device>(m);
    def_csr_column_range_gather_device<float, int, int, Device>(m);
    def_csr_column_range_gather_device<double, int, int, Device>(m);
    def_csr_column_range_gather_device<float, int64_t, int64_t, Device>(m);
    def_csr_column_range_gather_device<double, int64_t, int64_t, Device>(m);
}

NB_MODULE(_wilcoxon_cuda, m) {
    m.def("_set_host_worker_limit", &set_host_worker_limit, "limit"_a);
    register_scratch_allocator(m);
    def_csr_row_boundaries_host<int, int>(m);
    def_csr_row_boundaries_host<int64_t, int64_t>(m);
    def_has_canonical_format_host<int>(m);
    def_has_canonical_format_host<int64_t>(m);
    REGISTER_GPU_BINDINGS(register_bindings, m);
}
