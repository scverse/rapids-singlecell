#pragma once

#include <cstdint>
#include <cstring>
#include <new>

#include <cub/device/device_scan.cuh>

constexpr int OVR_HOST_CSR_U16_COLUMN_CAPACITY =
    (int)std::numeric_limits<uint16_t>::max() + 1;
constexpr size_t OVR_HOST_CSR_RANGE_STAGE_ITEMS = (size_t)32 * 1024 * 1024;
constexpr int OVR_HOST_CSR_RANGE_STAGE_SLOTS = 3;
constexpr int OVR_HOST_CSR_RANGE_METADATA_SLOTS = 2;
constexpr size_t OVR_HOST_CSR_RANGE_COUNT_CACHE_LIMIT_BYTES =
    (size_t)384 * 1024 * 1024;

// Host-streaming CSC OVR: sort only stored nonzeros per column.
// GPU memory is O(max_batch_nnz), not O(n_rows * n_cols).
template <typename InT, typename IndexT, typename IndptrT>
static void ovr_sparse_csc_host_streaming_impl(
    const InT* h_data, const IndexT* h_indices, const IndptrT* h_indptr,
    const int* h_group_codes, const double* h_group_sizes, double* d_rank_sums,
    double* d_tie_corr, double* d_group_sums, double* d_group_nnz,
    double* d_total_sums, double* d_total_nnz, int n_rows, int n_cols,
    int n_groups, bool compute_tie_corr, bool compute_nnz, bool compute_totals,
    int sub_batch_cols) {
    if (n_rows == 0 || n_cols == 0) return;

    // Bound each batch's nnz: CUB item counts stay within int32 + per-stream
    // sort buffers fit the budget (column counts free from CSC indptr).
    size_t cap = SAFE_BATCH_NNZ;
    {
        constexpr size_t BYTES_PER_NNZ =
            sizeof(InT) + 2 * sizeof(float) + 2 * sizeof(IndexT) + 8;
        size_t mem_cap =
            rmm_available_device_bytes(0.8) / (size_t)N_STREAMS / BYTES_PER_NNZ;
        if (mem_cap > 0 && mem_cap < cap) cap = mem_cap;
    }

    ColumnBatchPlan batches =
        plan_csc_column_batches(h_indptr, n_cols, sub_batch_cols, cap,
                                "OVR host CSC rebased column offsets");
    sub_batch_cols = batches.sub_batch_cols;
    int n_streams = clamp_streams_by_cols(n_cols, sub_batch_cols);
    size_t max_nnz = batches.max_nnz;

    size_t cub_temp_bytes = 0;
    if (max_nnz > 0) {
        int max_nnz_i32 =
            checked_cub_items(max_nnz, "OVR host CSC sparse sub-batch nnz");
        cub_temp_bytes =
            cub_segmented_sortpairs_temp_bytes(max_nnz_i32, sub_batch_cols);
    }

    // pool first: streams drain before it frees their scratch (see guard doc).
    RmmScratchPool pool;
    size_t total_nnz = (size_t)h_indptr[n_cols];
    size_t direct_pin_bytes = total_nnz * (sizeof(InT) + sizeof(IndexT));
    bool use_bounded_stage =
        direct_pin_bytes > HOST_STREAMING_DIRECT_PIN_LIMIT_BYTES;
    HostRegisterGuard pin_data;
    HostRegisterGuard pin_indices;
    std::unique_ptr<PinnedRing<InT, IndexT>> stage;
    if (use_bounded_stage) {
        stage.reset(new PinnedRing<InT, IndexT>(n_streams, max_nnz));
    } else {
        pin_data = HostRegisterGuard(const_cast<InT*>(h_data),
                                     total_nnz * sizeof(InT));
        pin_indices = HostRegisterGuard(const_cast<IndexT*>(h_indices),
                                        total_nnz * sizeof(IndexT));
    }
    ScopedCudaStreams streams(n_streams, cudaStreamDefault);
    int* d_group_codes = pool.alloc<int>(n_rows);
    double* d_group_sizes = pool.alloc<double>(n_groups);
    struct StreamBuf {
        InT* d_sparse_data_orig;
        float* d_sparse_data_f32;
        IndexT* d_sparse_indices;
        int* idx_i32;  // int32 sort-val scratch; only used when IndexT != int
        int* d_seg_offsets;
        float* keys_out;
        int* vals_out;
        uint8_t* cub_temp;
        double* d_rank_sums;
        double* d_tie_corr;
        double* d_group_sums;
        double* d_group_nnz;
        double* d_total_sums;
        double* d_total_nnz;
        double* d_nz_scratch;  // gmem-only; non-null when rank_use_gmem
    };
    std::vector<StreamBuf> bufs(n_streams);
    for (int s = 0; s < n_streams; s++) {
        bufs[s].d_sparse_data_orig = pool.alloc<InT>(max_nnz);
        bufs[s].d_sparse_data_f32 = pool.alloc<float>(max_nnz);
        bufs[s].d_sparse_indices = pool.alloc<IndexT>(max_nnz);
        bufs[s].idx_i32 =
            (sizeof(IndexT) > sizeof(int)) ? pool.alloc<int>(max_nnz) : nullptr;
        bufs[s].d_seg_offsets = pool.alloc<int>(sub_batch_cols + 1);
        bufs[s].keys_out = pool.alloc<float>(max_nnz);
        bufs[s].vals_out = pool.alloc<int>(max_nnz);
        bufs[s].cub_temp = pool.alloc<uint8_t>(cub_temp_bytes);
        bufs[s].d_rank_sums =
            pool.alloc<double>((size_t)n_groups * sub_batch_cols);
        bufs[s].d_tie_corr = pool.alloc<double>(sub_batch_cols);
        bufs[s].d_group_sums =
            pool.alloc<double>((size_t)n_groups * sub_batch_cols);
        bufs[s].d_group_nnz =
            compute_nnz ? pool.alloc<double>((size_t)n_groups * sub_batch_cols)
                        : nullptr;
        bufs[s].d_total_sums =
            compute_totals ? pool.alloc<double>(sub_batch_cols) : nullptr;
        bufs[s].d_total_nnz = (compute_totals && compute_nnz)
                                  ? pool.alloc<double>(sub_batch_cols)
                                  : nullptr;
    }

    cudaMemcpy(d_group_codes, h_group_codes, n_rows * sizeof(int),
               cudaMemcpyHostToDevice);
    cudaMemcpy(d_group_sizes, h_group_sizes, n_groups * sizeof(double),
               cudaMemcpyHostToDevice);

    // Pre-compute rebased per-batch offsets, upload once (no per-batch H2D).
    int* d_all_offsets = upload_batch_offsets(batches, pool);

    int tpb = UTIL_BLOCK_SIZE;
    bool rank_use_gmem = false;
    size_t smem_bytes = sparse_ovr_smem_config(n_groups, rank_use_gmem);
    bool cast_use_gmem = false;
    size_t smem_cast = cast_accumulate_smem_config(
        n_groups, compute_nnz, compute_totals, cast_use_gmem);

    // gmem mode: rank kernel accumulates into rank_sums directly, needs a
    // per-stream nz_count scratch buffer sized (n_groups, sb_cols).
    for (int s = 0; s < n_streams; s++) {
        if (rank_use_gmem) {
            bufs[s].d_nz_scratch =
                pool.alloc<double>((size_t)n_groups * sub_batch_cols);
        } else {
            bufs[s].d_nz_scratch = nullptr;
        }
    }

    cudaDeviceSynchronize();

    int col = 0;
    int batch_idx = 0;
    while (col < n_cols) {
        int sb_cols = std::min(sub_batch_cols, n_cols - col);
        int s = batch_idx % n_streams;
        auto stream = streams[s];
        auto& buf = bufs[s];

        IndptrT ptr_start = h_indptr[col];
        IndptrT ptr_end = h_indptr[col + sb_cols];
        int batch_nnz = checked_int_span((size_t)(ptr_end - ptr_start),
                                         "OVR host CSC active batch nnz");

        if (use_bounded_stage) {
            // Bounded staging: copy native values/indices into a small pinned
            // slot instead of page-locking the whole host CSC.
            stage->wait(s);
            if (batch_nnz > 0) {
                host_copy_slice_as<InT, IndexT>(
                    h_data, h_indices, (size_t)ptr_start, batch_nnz,
                    stage->template get<0>(s), stage->template get<1>(s));
                cudaMemcpyAsync(buf.d_sparse_data_orig,
                                stage->template get<0>(s),
                                (size_t)batch_nnz * sizeof(InT),
                                cudaMemcpyHostToDevice, stream);
                cudaMemcpyAsync(buf.d_sparse_indices, stage->template get<1>(s),
                                (size_t)batch_nnz * sizeof(IndexT),
                                cudaMemcpyHostToDevice, stream);
            }
            stage->record(s, stream);
        } else if (batch_nnz > 0) {
            cudaMemcpyAsync(buf.d_sparse_data_orig, h_data + ptr_start,
                            (size_t)batch_nnz * sizeof(InT),
                            cudaMemcpyHostToDevice, stream);
            cudaMemcpyAsync(buf.d_sparse_indices, h_indices + ptr_start,
                            (size_t)batch_nnz * sizeof(IndexT),
                            cudaMemcpyHostToDevice, stream);
        }

        // Row indices are the sort values; downcast int64 -> int32 at the
        // device boundary (values < n_rows < 2^31) so sort + rank stay int32.
        int* idx32;
        if constexpr (sizeof(IndexT) > sizeof(int)) {
            if (batch_nnz > 0) {
                int cblk = (batch_nnz + tpb - 1) / tpb;
                cast_array_kernel<IndexT, int><<<cblk, tpb, 0, stream>>>(
                    buf.d_sparse_indices, buf.idx_i32, (size_t)batch_nnz);
                CUDA_CHECK_LAST_ERROR(cast_array_kernel);
            }
            idx32 = buf.idx_i32;
        } else {
            idx32 = buf.d_sparse_indices;
        }

        int* src = d_all_offsets + (size_t)batch_idx * (sub_batch_cols + 1);
        cudaMemcpyAsync(buf.d_seg_offsets, src, (sb_cols + 1) * sizeof(int),
                        cudaMemcpyDeviceToDevice, stream);

        // Cast to float32 for sort + accumulate stats in float64
        launch_ovr_cast_and_accumulate_sparse<InT>(
            buf.d_sparse_data_orig, buf.d_sparse_data_f32, idx32,
            buf.d_seg_offsets, d_group_codes, buf.d_group_sums, buf.d_group_nnz,
            buf.d_total_sums, buf.d_total_nnz, sb_cols, n_groups, compute_nnz,
            compute_totals, tpb, smem_cast, cast_use_gmem, stream);

        // Sort only stored nonzeros (float32 keys)
        if (batch_nnz > 0) {
            cub_segmented_sortpairs(buf.cub_temp, cub_temp_bytes,
                                    buf.d_sparse_data_f32, buf.keys_out, idx32,
                                    buf.vals_out, batch_nnz, sb_cols,
                                    buf.d_seg_offsets, buf.d_seg_offsets + 1,
                                    stream, "host CSC OVR segmented sort");
        }

        launch_ovr_sparse_rank<int>(
            buf.keys_out, buf.vals_out, buf.d_seg_offsets, d_group_codes,
            d_group_sizes, buf.d_rank_sums, buf.d_tie_corr, buf.d_nz_scratch,
            n_rows, sb_cols, n_groups, tpb, smem_bytes, compute_tie_corr,
            rank_use_gmem, stream);

        scatter_cols_2d(d_rank_sums + col, buf.d_rank_sums, n_groups, n_cols,
                        sb_cols, stream);
        if (compute_tie_corr) {
            cudaMemcpyAsync(d_tie_corr + col, buf.d_tie_corr,
                            sb_cols * sizeof(double), cudaMemcpyDeviceToDevice,
                            stream);
        }
        scatter_cols_2d(d_group_sums + col, buf.d_group_sums, n_groups, n_cols,
                        sb_cols, stream);
        if (compute_nnz) {
            scatter_cols_2d(d_group_nnz + col, buf.d_group_nnz, n_groups,
                            n_cols, sb_cols, stream);
        }
        if (compute_totals) {
            cudaMemcpyAsync(d_total_sums + col, buf.d_total_sums,
                            sb_cols * sizeof(double), cudaMemcpyDeviceToDevice,
                            stream);
            if (compute_nnz) {
                cudaMemcpyAsync(d_total_nnz + col, buf.d_total_nnz,
                                sb_cols * sizeof(double),
                                cudaMemcpyDeviceToDevice, stream);
            }
        }

        col += sb_cols;
        batch_idx++;
    }

    sync_streams(streams, "sparse host CSC streaming");
}

// Plan CSR range batches without a full per-column histogram. Sorted row spans
// let each shard find batch boundaries with lower_bound, so planning touches
// O(n_rows * n_batches) indices instead of every source nonzero.
struct OvrCsrRangeBatchPlan {
    ColumnBatchPlan columns;
    // Batch-major. Empty means the bounded-cache fallback is active.
    std::vector<int> row_counts;
};

template <typename IndexT, typename IndptrT>
static OvrCsrRangeBatchPlan plan_ovr_csr_range_batches(
    const IndexT* h_indices, const IndptrT* h_indptr, const IndptrT* row_starts,
    const IndptrT* row_stops, int n_rows, int input_col_start, int n_cols,
    int sub_batch_cols, size_t cap, const char* what) {
    if (sub_batch_cols < 1) sub_batch_cols = 1;
    if (cap < 1) cap = 1;

    size_t range_nnz = 0;
    for (int r = 0; r < n_rows; r++) {
        if (row_starts[r] < h_indptr[r] || row_stops[r] < row_starts[r] ||
            row_stops[r] > h_indptr[r + 1]) {
            throw std::runtime_error(std::string(what) +
                                     " has an invalid row span");
        }
        size_t row_nnz = (size_t)(row_stops[r] - row_starts[r]);
        if (row_nnz > std::numeric_limits<size_t>::max() - range_nnz) {
            throw std::runtime_error(std::string(what) +
                                     " total nnz overflows size_t");
        }
        range_nnz += row_nnz;
    }
    size_t min_batches = range_nnz == 0 ? 0 : 1 + (range_nnz - 1) / cap;
    bool count_cache_allocation_failed = false;

    for (;;) {
        OvrCsrRangeBatchPlan range_plan;
        ColumnBatchPlan& plan = range_plan.columns;
        plan.sub_batch_cols = sub_batch_cols;
        plan.n_batches = 1 + (n_cols - 1) / plan.sub_batch_cols;
        plan.nnz.assign(plan.n_batches, 0);

        // Skip widths that cannot fit even under a perfectly even nnz split.
        // The accepted width still goes through the exact per-row validator.
        if (plan.sub_batch_cols > 1 && (size_t)plan.n_batches < min_batches) {
            sub_batch_cols = std::max(1, sub_batch_cols / 2);
            continue;
        }

        int n_workers = host_worker_count();
        bool cache_counts = !count_cache_allocation_failed;
        constexpr size_t max_count_cache_items =
            OVR_HOST_CSR_RANGE_COUNT_CACHE_LIMIT_BYTES / sizeof(int);
        if ((size_t)plan.n_batches > max_count_cache_items / (size_t)n_rows) {
            cache_counts = false;
        }
        if (cache_counts) {
            try {
                range_plan.row_counts.resize((size_t)plan.n_batches *
                                             (size_t)n_rows);
            } catch (const std::bad_alloc&) {
                range_plan.row_counts.clear();
                count_cache_allocation_failed = true;
                cache_counts = false;
            }
        }

        std::vector<std::vector<size_t>> local_counts(
            n_workers, std::vector<size_t>(plan.n_batches, 0));
        std::vector<unsigned char> invalid_cached_count(n_workers, 0);
        int used = host_parallel_chunks(n_rows, [&](int w, int r0, int r1) {
            std::vector<size_t>& counts = local_counts[w];
            for (int r = r0; r < r1; r++) {
                const IndexT* lo = h_indices + row_starts[r];
                const IndexT* hi = h_indices + row_stops[r];
                if (lo < hi && *lo < (IndexT)input_col_start) {
                    lo = std::lower_bound(lo, hi, (IndexT)input_col_start);
                }
                for (int b = 0; b < plan.n_batches; b++) {
                    int local_start = b * plan.sub_batch_cols;
                    int width =
                        std::min(plan.sub_batch_cols, n_cols - local_start);
                    IndexT input_col_end =
                        (IndexT)(input_col_start + local_start + width);
                    const IndexT* next =
                        std::lower_bound(lo, hi, input_col_end);
                    size_t row_count = (size_t)(next - lo);
                    counts[b] += row_count;
                    if (cache_counts) {
                        if (row_count <=
                            (size_t)std::numeric_limits<int>::max()) {
                            range_plan.row_counts[(size_t)b * (size_t)n_rows +
                                                  (size_t)r] = (int)row_count;
                        } else {
                            invalid_cached_count[w] = 1;
                        }
                    }
                    lo = next;
                }
            }
        });

        bool fits = true;
        for (int b = 0; b < plan.n_batches; b++) {
            size_t batch_nnz = 0;
            for (int w = 0; w < used; w++) batch_nnz += local_counts[w][b];
            plan.nnz[b] = batch_nnz;
            plan.max_nnz = std::max(plan.max_nnz, batch_nnz);
            if (batch_nnz > cap) fits = false;
        }
        if (cache_counts &&
            std::any_of(invalid_cached_count.begin(),
                        invalid_cached_count.begin() + used,
                        [](unsigned char invalid) { return invalid != 0; })) {
            range_plan.row_counts.clear();
        }

        if (fits || plan.sub_batch_cols == 1) {
            for (size_t batch_nnz : plan.nnz) checked_int_span(batch_nnz, what);
            return range_plan;
        }
        sub_batch_cols = std::max(1, sub_batch_cols / 2);
    }
}

// Count one range batch before materializing it so the compact row offsets can
// be copied once while values/indices are transferred in bounded row blocks.
template <typename IndexT, typename IndptrT>
static int prepare_ovr_csr_range_batch(
    const IndexT* h_indices, const IndptrT* h_indptr, const IndptrT* row_stops,
    int n_rows, int col_start, int col_end, IndptrT* cursor,
    int* fallback_row_counts, const int* planned_row_counts,
    int* compact_indptr, const char* what) {
    const int* row_counts = planned_row_counts;
    if (row_counts == nullptr) {
        host_parallel_ranges(n_rows, [&](int r0, int r1) {
            for (int r = r0; r < r1; r++) {
                const IndexT* row_base = h_indices + h_indptr[r];
                const IndexT* lo = row_base + cursor[r];
                const IndexT* hi = h_indices + row_stops[r];
                if (lo < hi && *lo < (IndexT)col_start) {
                    lo = std::lower_bound(lo, hi, (IndexT)col_start);
                    cursor[r] = (IndptrT)(lo - row_base);
                }
                fallback_row_counts[r] =
                    (int)(std::lower_bound(lo, hi, (IndexT)col_end) - lo);
            }
        });
        row_counts = fallback_row_counts;
    }

    compact_indptr[0] = 0;
    for (int r = 0; r < n_rows; r++) {
        compact_indptr[r + 1] = checked_int_span(
            (size_t)compact_indptr[r] + (size_t)row_counts[r], what);
    }
    return compact_indptr[n_rows];
}

// Materialize a row-aligned slice of a prepared range batch. Column counts are
// derived from the compacted device indices so this host pass only copies and
// rebases each item.
template <typename StageIndexT, typename InT, typename IndexT, typename IndptrT>
static void materialize_ovr_csr_range_row_block(
    const InT* h_data, const IndexT* h_indices, const IndptrT* h_indptr,
    int row_begin, int row_end, int stage_col_offset, IndptrT* cursor,
    const int* row_counts, const int* compact_indptr, int block_nnz_begin,
    InT* stage_vals, StageIndexT* stage_cols) {
    auto materialize_rows = [&](int r0, int r1) {
        for (int r = r0; r < r1; r++) {
            IndptrT base = h_indptr[r] + cursor[r];
            size_t dst = (size_t)(compact_indptr[r] - block_nnz_begin);
            int count = row_counts[r];
            std::memcpy(stage_vals + dst, h_data + base,
                        (size_t)count * sizeof(InT));
            const IndexT* __restrict__ source_cols = h_indices + base;
            StageIndexT* __restrict__ dest_cols = stage_cols + dst;
            for (int k = 0; k < count; k++) {
                dest_cols[k] =
                    (StageIndexT)(source_cols[k] - (IndexT)stage_col_offset);
            }
            cursor[r] += count;
        }
    };

    int block_rows = row_end - row_begin;
    int n_workers = host_worker_count();
    if (n_workers <= 1 || block_rows < 4096) {
        materialize_rows(row_begin, row_end);
        return;
    }

    // Equal-row chunks leave stragglers when cell nnz varies. The compact CSR
    // offsets provide an exact nnz prefix; adding one work unit per row keeps
    // the prefix strictly increasing across empty rows. Snap equal-work targets
    // to row boundaries so every worker's writes remain disjoint.
    int block_nnz_end = compact_indptr[row_end];
    size_t block_nnz = (size_t)(block_nnz_end - block_nnz_begin);
    size_t total_work = block_nnz + (size_t)block_rows;
    std::vector<int> worker_row_bounds;
    worker_row_bounds.reserve((size_t)n_workers + 1);
    worker_row_bounds.push_back(row_begin);
    for (int w = 1; w < n_workers; w++) {
        size_t target =
            total_work / (size_t)n_workers * (size_t)w +
            total_work % (size_t)n_workers * (size_t)w / (size_t)n_workers;
        int lo = row_begin;
        int hi = row_end;
        while (lo < hi) {
            int mid = lo + (hi - lo) / 2;
            size_t completed = (size_t)(compact_indptr[mid] - block_nnz_begin) +
                               (size_t)(mid - row_begin);
            if (completed < target)
                lo = mid + 1;
            else
                hi = mid;
        }
        if (lo > worker_row_bounds.back() && lo < row_end)
            worker_row_bounds.push_back(lo);
    }
    worker_row_bounds.push_back(row_end);
    int n_partitions = (int)worker_row_bounds.size() - 1;
    host_worker_pool(n_workers).run(n_partitions, [&](int, int p0, int p1) {
        for (int p = p0; p < p1; p++)
            materialize_rows(worker_row_bounds[p], worker_row_bounds[p + 1]);
    });
}

// Host CSR range OVR: sorted row spans let cursors advance once, so each nnz is
// gathered/transferred once.
template <typename StageIndexT, typename InT, typename IndexT, typename IndptrT>
static void ovr_sparse_csr_host_range_impl_typed(
    const InT* h_data, const IndexT* h_indices, const IndptrT* h_indptr,
    const int* h_group_codes, const double* h_group_sizes, double* d_rank_sums,
    double* d_tie_corr, double* d_group_sums, double* d_group_nnz,
    double* d_total_sums, double* d_total_nnz, int n_rows, int n_cols,
    int n_groups, bool compute_tie_corr, bool compute_nnz, bool compute_totals,
    int sub_batch_cols, int input_col_start, const IndptrT* row_starts,
    const IndptrT* row_stops) {
    if (n_rows == 0 || n_cols == 0) return;

    RmmScratchPool pool;
    int tpb = UTIL_BLOCK_SIZE;
    size_t budget = rmm_available_device_bytes(0.8);

    // Column batches must satisfy int32 CUB limits and device memory budget.
    // Per-nnz scratch covers mini-CSR gather, CSC accum, sort output, and CUB.
    constexpr size_t BYTES_PER_NNZ = 2 * sizeof(InT)  // gather val + csc val
                                     + 2 * sizeof(float)    // f32 key in + out
                                     + sizeof(StageIndexT)  // gather col
                                     + 2 * sizeof(int)  // csc + sorted row ids
                                     + 12;              // CUB temp headroom
    size_t cap = SAFE_BATCH_NNZ;
    size_t mem_cap = budget / BYTES_PER_NNZ;
    if (mem_cap > 0 && mem_cap < cap) cap = mem_cap;

    OvrCsrRangeBatchPlan range_batches = plan_ovr_csr_range_batches(
        h_indices, h_indptr, row_starts, row_stops, n_rows, input_col_start,
        n_cols, sub_batch_cols, cap, "rowstream range batch nnz");
    ColumnBatchPlan batches = std::move(range_batches.columns);

    sub_batch_cols = batches.sub_batch_cols;
    int n_batches = batches.n_batches;
    size_t max_batch_nnz = batches.max_nnz;

    size_t cub_temp_bytes = 0;
    if (max_batch_nnz > 0) {
        int mb_i32 =
            checked_cub_items(max_batch_nnz, "rowstream sub-batch nnz");
        cub_temp_bytes =
            cub_segmented_sortpairs_temp_bytes(mb_i32, sub_batch_cols);
        auto* histogram = reinterpret_cast<int*>(1);
        size_t scan_temp_bytes = 0;
        cuda_check(
            cub::DeviceScan::ExclusiveSum(nullptr, scan_temp_bytes, histogram,
                                          histogram, sub_batch_cols + 1),
            "rowstream range scan temp-size query");
        cub_temp_bytes = std::max(cub_temp_bytes, scan_temp_bytes);
    }
    bool rank_use_gmem = false;
    size_t smem_bytes = sparse_ovr_smem_config(n_groups, rank_use_gmem);
    bool cast_use_gmem = false;
    size_t smem_cast = cast_accumulate_smem_config(
        n_groups, compute_nnz, compute_totals, cast_use_gmem);

    // Host gather staging is pinned; full CSR stays pageable on CPU.
    // Only the compacted column interval crosses the bus.
    size_t stage_nnz = max_batch_nnz ? max_batch_nnz : 1;
    size_t range_stage_target =
        std::max(OVR_HOST_CSR_RANGE_STAGE_ITEMS, (size_t)sub_batch_cols);
    size_t gather_stage_nnz = std::min(stage_nnz, range_stage_target);
    int gather_stage_slots = OVR_HOST_CSR_RANGE_STAGE_SLOTS;
    PinnedRing<InT, StageIndexT> gather_stage(gather_stage_slots,
                                              gather_stage_nnz, true);
    size_t metadata_items = (size_t)n_rows + 1;
    int metadata_slots = OVR_HOST_CSR_RANGE_METADATA_SLOTS;
    PinnedRing<int> indptr_stage(metadata_slots, metadata_items);
    std::vector<IndptrT> cursor(n_rows, 0);  // offset within each (sorted) row
    host_parallel_ranges(n_rows, [&](int r0, int r1) {
        for (int r = r0; r < r1; r++) {
            const IndexT* row_base = h_indices + h_indptr[r];
            const IndexT* lo = h_indices + row_starts[r];
            const IndexT* hi = h_indices + row_stops[r];
            if (lo < hi && *lo < (IndexT)input_col_start) {
                lo = std::lower_bound(lo, hi, (IndexT)input_col_start);
            }
            cursor[r] = (IndptrT)(lo - row_base);
        }
    });

    int* d_group_codes = pool.alloc<int>(n_rows);
    double* d_group_sizes = pool.alloc<double>(n_groups);
    cudaMemcpy(d_group_codes, h_group_codes, n_rows * sizeof(int),
               cudaMemcpyHostToDevice);
    cudaMemcpy(d_group_sizes, h_group_sizes, n_groups * sizeof(double),
               cudaMemcpyHostToDevice);

    InT* d_gather_vals = pool.alloc<InT>(max_batch_nnz);
    StageIndexT* d_gather_cols = pool.alloc<StageIndexT>(max_batch_nnz);
    int* d_gather_indptr = pool.alloc<int>(n_rows + 1);
    int* d_col_offsets[OVR_HOST_CSR_RANGE_METADATA_SLOTS] = {
        pool.alloc<int>(sub_batch_cols + 1),
        pool.alloc<int>(sub_batch_cols + 1)};
    int* write_pos = pool.alloc<int>(sub_batch_cols);
    InT* csc_vals_orig = pool.alloc<InT>(max_batch_nnz);
    float* csc_vals_f32 = pool.alloc<float>(max_batch_nnz);
    int* csc_row_idx = pool.alloc<int>(max_batch_nnz);
    float* keys_out = pool.alloc<float>(max_batch_nnz);
    int* vals_out = pool.alloc<int>(max_batch_nnz);
    uint8_t* cub_temp = pool.alloc<uint8_t>(cub_temp_bytes);
    double* sub_rank_sums =
        pool.alloc<double>((size_t)n_groups * sub_batch_cols);
    double* sub_tie_corr = pool.alloc<double>(sub_batch_cols);
    double* sub_group_sums =
        pool.alloc<double>((size_t)n_groups * sub_batch_cols);
    double* sub_group_nnz =
        compute_nnz ? pool.alloc<double>((size_t)n_groups * sub_batch_cols)
                    : nullptr;
    double* sub_total_sums =
        compute_totals ? pool.alloc<double>(sub_batch_cols) : nullptr;
    double* sub_total_nnz = (compute_totals && compute_nnz)
                                ? pool.alloc<double>(sub_batch_cols)
                                : nullptr;
    double* d_nz_scratch =
        rank_use_gmem ? pool.alloc<double>((size_t)n_groups * sub_batch_cols)
                      : nullptr;
    std::vector<std::unique_ptr<ScopedCudaEvent>> batch_ready;
    std::vector<std::unique_ptr<ScopedCudaEvent>> gather_consumed;
    ScopedCudaStream row_stream(cudaStreamDefault);
    cudaStream_t stream = row_stream.get();
    ScopedCudaStream range_copy_stream(cudaStreamNonBlocking);
    cudaStream_t copy_stream = range_copy_stream.get();
    batch_ready.reserve(n_batches);
    gather_consumed.reserve(n_batches);
    for (int b = 0; b < n_batches; b++) {
        batch_ready.emplace_back(
            std::make_unique<ScopedCudaEvent>(cudaEventDisableTiming));
        gather_consumed.emplace_back(
            std::make_unique<ScopedCudaEvent>(cudaEventDisableTiming));
    }
    int device = 0;
    int n_sms = 0;
    cuda_check(cudaGetDevice(&device), "rowstream histogram get device");
    cuda_check(
        cudaDeviceGetAttribute(&n_sms, cudaDevAttrMultiProcessorCount, device),
        "rowstream histogram get SM count");
    int range_histogram_blocks = std::max(1, 4 * n_sms);

    // One ascending column pass; sorted-row cursors make transfer one-shot.
    // Threaded gather counts row runs, prefix-sums, then copies disjoint
    // ranges. The GPU derives CSC column offsets from the staged column ids.
    std::vector<int> g_count;
    if (range_batches.row_counts.empty()) g_count.resize(n_rows);
    size_t range_stage_sequence = 0;

    int col = 0;
    for (int b = 0; b < n_batches; b++) {
        int sb_cols = std::min(sub_batch_cols, n_cols - col);
        int input_col = input_col_start + col;
        int input_col_end = input_col + sb_cols;
        int metadata_slot = b % OVR_HOST_CSR_RANGE_METADATA_SLOTS;
        int* active_col_offsets = d_col_offsets[metadata_slot];
        if (b > 0) {
            cuda_check(cudaStreamWaitEvent(copy_stream,
                                           gather_consumed[b - 1]->get(), 0),
                       "rowstream copy wait for gather consumption");
        }
        indptr_stage.wait(metadata_slot);
        int* h_gather_indptr = indptr_stage.template get<0>(metadata_slot);
        const int* planned_row_counts =
            range_batches.row_counts.empty()
                ? nullptr
                : range_batches.row_counts.data() + (size_t)b * (size_t)n_rows;
        int* fallback_row_counts = g_count.empty() ? nullptr : g_count.data();

        int batch_nnz = prepare_ovr_csr_range_batch(
            h_indices, h_indptr, row_stops, n_rows, input_col, input_col_end,
            cursor.data(), fallback_row_counts, planned_row_counts,
            h_gather_indptr, "rowstream range gather nnz");
        if ((size_t)batch_nnz != batches.nnz[b]) {
            throw std::runtime_error(
                "rowstream planned and materialized batch nnz differ");
        }
        const int* batch_row_counts = planned_row_counts != nullptr
                                          ? planned_row_counts
                                          : fallback_row_counts;

        int row_begin = 0;
        while (row_begin < n_rows && batch_nnz > 0) {
            int block_nnz_begin = h_gather_indptr[row_begin];
            size_t block_nnz_limit =
                std::min((size_t)batch_nnz,
                         (size_t)block_nnz_begin + gather_stage.capacity);
            int block_nnz_limit_i32 = checked_int_span(
                block_nnz_limit, "rowstream range stage block limit");
            const int* row_limit = std::upper_bound(
                h_gather_indptr + row_begin + 1, h_gather_indptr + n_rows + 1,
                block_nnz_limit_i32);
            int row_end = (int)(row_limit - h_gather_indptr) - 1;
            if (row_end <= row_begin) {
                throw std::runtime_error(
                    "rowstream range row exceeds pinned stage capacity");
            }
            int block_nnz = h_gather_indptr[row_end] - block_nnz_begin;
            if (block_nnz > 0) {
                int stage_slot = (int)(range_stage_sequence %
                                       OVR_HOST_CSR_RANGE_STAGE_SLOTS);
                gather_stage.wait(stage_slot);
                InT* h_gather_vals = gather_stage.template get<0>(stage_slot);
                StageIndexT* h_gather_cols =
                    gather_stage.template get<1>(stage_slot);
                materialize_ovr_csr_range_row_block<StageIndexT>(
                    h_data, h_indices, h_indptr, row_begin, row_end,
                    input_col_start, cursor.data(), batch_row_counts,
                    h_gather_indptr, block_nnz_begin, h_gather_vals,
                    h_gather_cols);
                cuda_check(cudaMemcpyAsync(d_gather_vals + block_nnz_begin,
                                           h_gather_vals,
                                           (size_t)block_nnz * sizeof(InT),
                                           cudaMemcpyHostToDevice, copy_stream),
                           "rowstream range gathered vals H2D");
                cuda_check(cudaMemcpyAsync(
                               d_gather_cols + block_nnz_begin, h_gather_cols,
                               (size_t)block_nnz * sizeof(StageIndexT),
                               cudaMemcpyHostToDevice, copy_stream),
                           "rowstream range gathered cols H2D");
                gather_stage.record(stage_slot, copy_stream);
                range_stage_sequence++;
            }
            row_begin = row_end;
        }

        cuda_check(cudaMemcpyAsync(d_gather_indptr, h_gather_indptr,
                                   (n_rows + 1) * sizeof(int),
                                   cudaMemcpyHostToDevice, copy_stream),
                   "rowstream range indptr H2D");
        indptr_stage.record(metadata_slot, copy_stream);
        if (batch_nnz > 0) {
            cuda_check(
                cudaMemsetAsync(active_col_offsets, 0,
                                (sb_cols + 1) * sizeof(int), copy_stream),
                "rowstream range histogram reset");
            int histogram_blocks = std::min(
                range_histogram_blocks, 1 + (batch_nnz - 1) / UTIL_BLOCK_SIZE);
            compact_col_histogram_kernel<<<histogram_blocks, UTIL_BLOCK_SIZE,
                                           (size_t)sb_cols * sizeof(int),
                                           copy_stream>>>(
                d_gather_cols, active_col_offsets, batch_nnz, col, sb_cols);
            CUDA_CHECK_LAST_ERROR(compact_col_histogram_kernel);
        } else {
            cuda_check(
                cudaMemsetAsync(active_col_offsets, 0,
                                (sb_cols + 1) * sizeof(int), copy_stream),
                "rowstream empty range column offsets");
        }
        batch_ready[b]->record(copy_stream);
        cuda_check(cudaStreamWaitEvent(stream, batch_ready[b]->get(), 0),
                   "rowstream compute wait for gathered batch");
        if (batch_nnz > 0) {
            size_t scan_temp_bytes = cub_temp_bytes;
            cuda_check(cub::DeviceScan::ExclusiveSum(
                           cub_temp, scan_temp_bytes, active_col_offsets,
                           active_col_offsets, sb_cols + 1, stream),
                       "rowstream range column offset scan");
        }
        cuda_check(cudaMemcpyAsync(write_pos, active_col_offsets,
                                   sb_cols * sizeof(int),
                                   cudaMemcpyDeviceToDevice, stream),
                   "rowstream range write positions D2D");

        // Scatter mini-CSR into the column-batch CSC accumulator.
        csr_scatter_to_csc_kernel<InT, StageIndexT, int>
            <<<(n_rows + tpb - 1) / tpb, tpb, 0, stream>>>(
                d_gather_vals, d_gather_cols, d_gather_indptr, write_pos,
                csc_vals_orig, csc_row_idx, n_rows, col, col + sb_cols, 0);
        CUDA_CHECK_LAST_ERROR(csr_scatter_to_csc_kernel);
        gather_consumed[b]->record(stream);

        launch_ovr_cast_and_accumulate_sparse<InT>(
            csc_vals_orig, csc_vals_f32, csc_row_idx, active_col_offsets,
            d_group_codes, sub_group_sums, sub_group_nnz, sub_total_sums,
            sub_total_nnz, sb_cols, n_groups, compute_nnz, compute_totals, tpb,
            smem_cast, cast_use_gmem, stream);
        if (batch_nnz > 0) {
            cub_segmented_sortpairs(
                cub_temp, cub_temp_bytes, csc_vals_f32, keys_out, csc_row_idx,
                vals_out, batch_nnz, sb_cols, active_col_offsets,
                active_col_offsets + 1, stream, "rowstream segmented sort");
        }
        launch_ovr_sparse_rank<int>(keys_out, vals_out, active_col_offsets,
                                    d_group_codes, d_group_sizes, sub_rank_sums,
                                    sub_tie_corr, d_nz_scratch, n_rows, sb_cols,
                                    n_groups, tpb, smem_bytes, compute_tie_corr,
                                    rank_use_gmem, stream);

        cudaMemcpy2DAsync(d_rank_sums + col, n_cols * sizeof(double),
                          sub_rank_sums, sb_cols * sizeof(double),
                          sb_cols * sizeof(double), n_groups,
                          cudaMemcpyDeviceToDevice, stream);
        if (compute_tie_corr)
            cudaMemcpyAsync(d_tie_corr + col, sub_tie_corr,
                            sb_cols * sizeof(double), cudaMemcpyDeviceToDevice,
                            stream);
        cudaMemcpy2DAsync(d_group_sums + col, n_cols * sizeof(double),
                          sub_group_sums, sb_cols * sizeof(double),
                          sb_cols * sizeof(double), n_groups,
                          cudaMemcpyDeviceToDevice, stream);
        if (compute_nnz)
            cudaMemcpy2DAsync(d_group_nnz + col, n_cols * sizeof(double),
                              sub_group_nnz, sb_cols * sizeof(double),
                              sb_cols * sizeof(double), n_groups,
                              cudaMemcpyDeviceToDevice, stream);
        if (compute_totals) {
            cudaMemcpyAsync(d_total_sums + col, sub_total_sums,
                            sb_cols * sizeof(double), cudaMemcpyDeviceToDevice,
                            stream);
            if (compute_nnz) {
                cudaMemcpyAsync(d_total_nnz + col, sub_total_nnz,
                                sb_cols * sizeof(double),
                                cudaMemcpyDeviceToDevice, stream);
            }
        }
        col += sb_cols;
    }
    cuda_check(cudaStreamSynchronize(stream), "rowstream sync");
}

// Host CSR OVR range: never stage/map the full CSR on the target device. The
// shard scans only its global column interval and writes local-width outputs in
// the original within-range order.
template <typename InT, typename IndexT, typename IndptrT>
static void ovr_sparse_csr_host_range_impl(
    const InT* h_data, const IndexT* h_indices, const IndptrT* h_indptr,
    const int* h_group_codes, const double* h_group_sizes, double* d_rank_sums,
    double* d_tie_corr, double* d_group_sums, double* d_group_nnz,
    double* d_total_sums, double* d_total_nnz, int n_rows, int col_start,
    int n_cols, int n_groups, bool compute_tie_corr, bool compute_nnz,
    bool compute_totals, int sub_batch_cols, const IndptrT* row_starts,
    const IndptrT* row_stops) {
    if (n_cols <= OVR_HOST_CSR_U16_COLUMN_CAPACITY) {
        ovr_sparse_csr_host_range_impl_typed<uint16_t>(
            h_data, h_indices, h_indptr, h_group_codes, h_group_sizes,
            d_rank_sums, d_tie_corr, d_group_sums, d_group_nnz, d_total_sums,
            d_total_nnz, n_rows, n_cols, n_groups, compute_tie_corr,
            compute_nnz, compute_totals, sub_batch_cols, col_start, row_starts,
            row_stops);
        return;
    }
    ovr_sparse_csr_host_range_impl_typed<int>(
        h_data, h_indices, h_indptr, h_group_codes, h_group_sizes, d_rank_sums,
        d_tie_corr, d_group_sums, d_group_nnz, d_total_sums, d_total_nnz,
        n_rows, n_cols, n_groups, compute_tie_corr, compute_nnz, compute_totals,
        sub_batch_cols, col_start, row_starts, row_stops);
}

// Sparse-aware CSC OVR streaming: sort only stored nonzeros.

template <typename IndexT = int, typename IndptrT = int>
static void ovr_sparse_csc_streaming_impl(
    const float* csc_data, const IndexT* csc_indices, const IndptrT* csc_indptr,
    const int* group_codes, const double* group_sizes, double* rank_sums,
    double* tie_corr, int n_rows, int n_cols, int n_groups,
    bool compute_tie_corr, int sub_batch_cols) {
    if (n_rows == 0 || n_cols == 0) return;

    // Read indptr to host for batch planning.
    std::vector<IndptrT> h_indptr(n_cols + 1);
    cudaMemcpy(h_indptr.data(), csc_indptr, (n_cols + 1) * sizeof(IndptrT),
               cudaMemcpyDeviceToHost);

    // Bound each batch's nnz: CUB item counts within int32 + sort buffers fit.
    constexpr size_t BYTES_PER_NNZ = 2 * sizeof(float) + 2 * sizeof(int) + 8;
    size_t cap = SAFE_BATCH_NNZ;
    size_t mem_cap =
        rmm_available_device_bytes(0.8) / (size_t)N_STREAMS / BYTES_PER_NNZ;
    if (mem_cap > 0 && mem_cap < cap) cap = mem_cap;
    ColumnBatchPlan batches =
        plan_csc_column_batches(h_indptr.data(), n_cols, sub_batch_cols, cap,
                                "OVR device CSC rebased column offsets");
    sub_batch_cols = batches.sub_batch_cols;
    int n_streams = clamp_streams_by_cols(n_cols, sub_batch_cols);
    size_t max_nnz = batches.max_nnz;

    size_t cub_temp_bytes = 0;
    if (max_nnz > 0) {
        int max_nnz_i32 =
            checked_cub_items(max_nnz, "OVR device CSC sparse sub-batch nnz");
        cub_temp_bytes =
            cub_segmented_sortpairs_temp_bytes(max_nnz_i32, sub_batch_cols);
    }

    // pool first: streams drain before it frees their scratch (see guard doc).
    RmmScratchPool pool;
    ScopedCudaStreams streams(n_streams, cudaStreamDefault);

    int tpb = UTIL_BLOCK_SIZE;
    bool rank_use_gmem = false;
    size_t smem_bytes = sparse_ovr_smem_config(n_groups, rank_use_gmem);

    struct StreamBuf {
        float* keys_out;
        int* vals_out;
        int* idx_i32;  // int32 sort-val scratch; only used when IndexT != int
        int* seg_offsets;
        uint8_t* cub_temp;
        double* sub_rank_sums;
        double* sub_tie_corr;
        double* d_nz_scratch;  // gmem-only
    };
    std::vector<StreamBuf> bufs(n_streams);
    for (int s = 0; s < n_streams; s++) {
        bufs[s].keys_out = pool.alloc<float>(max_nnz);
        bufs[s].vals_out = pool.alloc<int>(max_nnz);
        bufs[s].idx_i32 =
            (sizeof(IndexT) > sizeof(int)) ? pool.alloc<int>(max_nnz) : nullptr;
        bufs[s].seg_offsets = pool.alloc<int>(sub_batch_cols + 1);
        bufs[s].cub_temp = pool.alloc<uint8_t>(cub_temp_bytes);
        bufs[s].sub_rank_sums =
            pool.alloc<double>((size_t)n_groups * sub_batch_cols);
        bufs[s].sub_tie_corr = pool.alloc<double>(sub_batch_cols);
        bufs[s].d_nz_scratch =
            rank_use_gmem
                ? pool.alloc<double>((size_t)n_groups * sub_batch_cols)
                : nullptr;
    }

    cudaDeviceSynchronize();

    int col = 0;
    int batch_idx = 0;
    while (col < n_cols) {
        int sb_cols = std::min(sub_batch_cols, n_cols - col);
        int s = batch_idx % n_streams;
        auto stream = streams[s];
        auto& buf = bufs[s];

        IndptrT ptr_start = h_indptr[col];
        IndptrT ptr_end = h_indptr[col + sb_cols];
        int batch_nnz = checked_int_span((size_t)(ptr_end - ptr_start),
                                         "OVR device CSC active batch nnz");

        // Rebase segment offsets on GPU (avoids host pinned-buffer race).
        {
            int count = sb_cols + 1;
            int blk = (count + UTIL_BLOCK_SIZE - 1) / UTIL_BLOCK_SIZE;
            rebase_indptr_kernel<<<blk, UTIL_BLOCK_SIZE, 0, stream>>>(
                csc_indptr, buf.seg_offsets, col, count);
            CUDA_CHECK_LAST_ERROR(rebase_indptr_kernel);
        }

        // Sort stored values; row indices become int32 sort values here.
        // This keeps sort/rank int32 while preserving int64 sparse buffers.
        if (batch_nnz > 0) {
            const int* idx_src;
            if constexpr (sizeof(IndexT) > sizeof(int)) {
                int cblk = (batch_nnz + UTIL_BLOCK_SIZE - 1) / UTIL_BLOCK_SIZE;
                cast_array_kernel<IndexT, int>
                    <<<cblk, UTIL_BLOCK_SIZE, 0, stream>>>(
                        csc_indices + ptr_start, buf.idx_i32,
                        (size_t)batch_nnz);
                CUDA_CHECK_LAST_ERROR(cast_array_kernel);
                idx_src = buf.idx_i32;
            } else {
                idx_src = csc_indices + ptr_start;
            }
            cub_segmented_sortpairs(buf.cub_temp, cub_temp_bytes,
                                    csc_data + ptr_start, buf.keys_out, idx_src,
                                    buf.vals_out, batch_nnz, sb_cols,
                                    buf.seg_offsets, buf.seg_offsets + 1,
                                    stream, "device CSC OVR segmented sort");
        }

        // Sparse rank kernel (handles implicit zeros analytically)
        launch_ovr_sparse_rank<int>(buf.keys_out, buf.vals_out, buf.seg_offsets,
                                    group_codes, group_sizes, buf.sub_rank_sums,
                                    buf.sub_tie_corr, buf.d_nz_scratch, n_rows,
                                    sb_cols, n_groups, tpb, smem_bytes,
                                    compute_tie_corr, rank_use_gmem, stream);

        scatter_cols_2d(rank_sums + col, buf.sub_rank_sums, n_groups, n_cols,
                        sb_cols, stream);
        if (compute_tie_corr) {
            cudaMemcpyAsync(tie_corr + col, buf.sub_tie_corr,
                            sb_cols * sizeof(double), cudaMemcpyDeviceToDevice,
                            stream);
        }

        col += sb_cols;
        batch_idx++;
    }

    sync_streams(streams, "sparse ovr streaming");
}

// Sparse-aware CSR OVR streaming with partial CSR->CSC transpose per batch.
// Histogram plans batches; each batch transposes, sorts nnz only, then ranks.
template <typename IndexT = int, typename IndptrT = int>
static void ovr_sparse_csr_streaming_impl(
    const float* csr_data, const IndexT* csr_indices, const IndptrT* csr_indptr,
    const int* group_codes, const double* group_sizes, double* rank_sums,
    double* tie_corr, int n_rows, int n_cols, int n_groups,
    bool compute_tie_corr, int sub_batch_cols) {
    if (n_rows == 0 || n_cols == 0) return;

    // ---- Phase 0: count nnz per column via histogram ----
    RmmScratchPool pool;
    unsigned int* d_col_counts = pool.alloc<unsigned int>(n_cols);
    cudaMemset(d_col_counts, 0, n_cols * sizeof(unsigned int));
    {
        int blocks = (n_rows + UTIL_BLOCK_SIZE - 1) / UTIL_BLOCK_SIZE;
        csr_col_histogram_kernel<<<blocks, UTIL_BLOCK_SIZE>>>(
            csr_indices, csr_indptr, d_col_counts, n_rows, n_cols);
        CUDA_CHECK_LAST_ERROR(csr_col_histogram_kernel);
    }
    std::vector<unsigned int> h_col_counts(n_cols);
    cudaMemcpy(h_col_counts.data(), d_col_counts, n_cols * sizeof(unsigned int),
               cudaMemcpyDeviceToHost);

    // Bound each batch's nnz: CUB item counts within int32 + transpose/sort
    // buffers fit.
    constexpr size_t BYTES_PER_NNZ = 2 * sizeof(float) + 2 * sizeof(int) + 8;
    size_t cap = SAFE_BATCH_NNZ;
    size_t mem_cap =
        rmm_available_device_bytes(0.8) / (size_t)N_STREAMS / BYTES_PER_NNZ;
    if (mem_cap > 0 && mem_cap < cap) cap = mem_cap;
    ColumnBatchPlan batches = plan_column_batches_from_counts(
        n_cols, sub_batch_cols, cap,
        [&](int c) { return (size_t)h_col_counts[c]; },
        "OVR device CSR rebased column offsets");
    sub_batch_cols = batches.sub_batch_cols;
    int n_batches = batches.n_batches;
    size_t max_batch_nnz = batches.max_nnz;

    // Upload all batch offsets in one H2D.
    int* d_all_offsets = upload_batch_offsets(batches, pool);

    // ---- Phase 1: per-stream buffers ----
    size_t cub_temp_bytes = 0;
    if (max_batch_nnz > 0) {
        int max_batch_nnz_i32 = checked_cub_items(
            max_batch_nnz, "OVR device CSR sparse sub-batch nnz");
        cub_temp_bytes = cub_segmented_sortpairs_temp_bytes(max_batch_nnz_i32,
                                                            sub_batch_cols);
    }

    int n_streams = N_STREAMS;
    if (n_batches < n_streams) n_streams = n_batches;

    // CSR path needs 4 sort arrays per stream (scatter intermediates + CUB
    // output); fit stream count to available GPU memory.
    bool rank_use_gmem = false;
    size_t smem_bytes = sparse_ovr_smem_config(n_groups, rank_use_gmem);
    size_t per_stream_bytes =
        max_batch_nnz * (2 * sizeof(float) + 2 * sizeof(int)) +
        (sub_batch_cols + 1 + sub_batch_cols) * sizeof(int) + cub_temp_bytes +
        (size_t)n_groups * sub_batch_cols * sizeof(double) +
        sub_batch_cols * sizeof(double);
    if (rank_use_gmem) {
        // gmem fallback (n_groups too large for smem): per-stream d_nz_scratch,
        // same size as sub_rank_sums.
        per_stream_bytes += (size_t)n_groups * sub_batch_cols * sizeof(double);
    }

    size_t budget = rmm_available_device_bytes(0.8);
    n_streams = clamp_streams_by_budget(n_streams, per_stream_bytes, budget);

    ScopedCudaStreams streams(n_streams, cudaStreamDefault);

    int tpb = UTIL_BLOCK_SIZE;
    int scatter_blocks = (n_rows + tpb - 1) / tpb;

    struct StreamBuf {
        int* col_offsets;  // CSC-style offsets
        int* write_pos;    // atomic write counters
        float* csc_vals;   // transposed values
        int* csc_row_idx;  // transposed row indices
        float* keys_out;   // CUB sort output
        int* vals_out;     // CUB sort output
        uint8_t* cub_temp;
        double* sub_rank_sums;
        double* sub_tie_corr;
        double* d_nz_scratch;  // gmem-only
    };
    std::vector<StreamBuf> bufs(n_streams);
    for (int s = 0; s < n_streams; s++) {
        bufs[s].col_offsets = pool.alloc<int>(sub_batch_cols + 1);
        bufs[s].write_pos = pool.alloc<int>(sub_batch_cols);
        bufs[s].csc_vals = pool.alloc<float>(max_batch_nnz);
        bufs[s].csc_row_idx = pool.alloc<int>(max_batch_nnz);
        bufs[s].keys_out = pool.alloc<float>(max_batch_nnz);
        bufs[s].vals_out = pool.alloc<int>(max_batch_nnz);
        bufs[s].cub_temp = pool.alloc<uint8_t>(cub_temp_bytes);
        bufs[s].sub_rank_sums =
            pool.alloc<double>((size_t)n_groups * sub_batch_cols);
        bufs[s].sub_tie_corr = pool.alloc<double>(sub_batch_cols);
        bufs[s].d_nz_scratch =
            rank_use_gmem
                ? pool.alloc<double>((size_t)n_groups * sub_batch_cols)
                : nullptr;
    }

    cudaDeviceSynchronize();

    // ---- Phase 2: stream loop ----
    int col = 0;
    for (int b = 0; b < n_batches; b++) {
        int sb_cols = std::min(sub_batch_cols, n_cols - col);
        int s = b % n_streams;
        auto stream = streams[s];
        auto& buf = bufs[s];
        int batch_nnz =
            checked_int_span(batches.nnz[b], "OVR device CSR active batch nnz");

        int* src = d_all_offsets + (size_t)b * (sub_batch_cols + 1);
        cudaMemcpyAsync(buf.col_offsets, src, (sb_cols + 1) * sizeof(int),
                        cudaMemcpyDeviceToDevice, stream);

        // write_pos = col_offsets[0..sb_cols-1] (same D2D source).
        cudaMemcpyAsync(buf.write_pos, src, sb_cols * sizeof(int),
                        cudaMemcpyDeviceToDevice, stream);

        if (batch_nnz > 0) {
            // Scatter CSR -> CSC for this sub-batch.
            csr_scatter_to_csc_kernel<<<scatter_blocks, tpb, 0, stream>>>(
                csr_data, csr_indices, csr_indptr, buf.write_pos, buf.csc_vals,
                buf.csc_row_idx, n_rows, col, col + sb_cols);
            CUDA_CHECK_LAST_ERROR(csr_scatter_to_csc_kernel);

            // Sort only the nonzeros.
            cub_segmented_sortpairs(buf.cub_temp, cub_temp_bytes, buf.csc_vals,
                                    buf.keys_out, buf.csc_row_idx, buf.vals_out,
                                    batch_nnz, sb_cols, buf.col_offsets,
                                    buf.col_offsets + 1, stream,
                                    "device CSR OVR segmented sort");
        }

        // Sparse rank kernel (handles implicit zeros analytically)
        launch_ovr_sparse_rank<int>(buf.keys_out, buf.vals_out, buf.col_offsets,
                                    group_codes, group_sizes, buf.sub_rank_sums,
                                    buf.sub_tie_corr, buf.d_nz_scratch, n_rows,
                                    sb_cols, n_groups, tpb, smem_bytes,
                                    compute_tie_corr, rank_use_gmem, stream);

        scatter_cols_2d(rank_sums + col, buf.sub_rank_sums, n_groups, n_cols,
                        sb_cols, stream);
        if (compute_tie_corr) {
            cudaMemcpyAsync(tie_corr + col, buf.sub_tie_corr,
                            sb_cols * sizeof(double), cudaMemcpyDeviceToDevice,
                            stream);
        }

        col += sb_cols;
    }

    sync_streams(streams, "sparse CSR ovr streaming");
}
