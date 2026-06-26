#pragma once

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
                host_copy_slice(h_data, h_indices, (size_t)ptr_start, batch_nnz,
                                stage->template get<0>(s),
                                stage->template get<1>(s));
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

// Host CSR rowstream OVR for matrices too large to stage on the GPU.
// Sorted rows let cursors advance once, so each nnz is gathered/transferred
// once.
template <typename InT, typename IndexT, typename IndptrT>
static void ovr_sparse_csr_host_rowstream_impl(
    const InT* h_data, const IndexT* h_indices, const IndptrT* h_indptr,
    const int* h_group_codes, const double* h_group_sizes, double* d_rank_sums,
    double* d_tie_corr, double* d_group_sums, double* d_group_nnz,
    double* d_total_sums, double* d_total_nnz, int n_rows, int n_cols,
    int n_groups, bool compute_tie_corr, bool compute_nnz, bool compute_totals,
    int sub_batch_cols) {
    if (n_rows == 0 || n_cols == 0) return;
    size_t total_nnz = (size_t)h_indptr[n_rows];

    RmmScratchPool pool;
    int tpb = UTIL_BLOCK_SIZE;
    size_t budget = rmm_available_device_bytes(0.8);

    // Host column histogram; each worker counts privately, then merges.
    std::vector<size_t> h_col_counts(n_cols, 0);
    {
        int n_workers = host_worker_count();
        std::vector<std::vector<size_t>> local(n_workers,
                                               std::vector<size_t>(n_cols, 0));
        int used = host_parallel_chunks(n_rows, [&](int w, int r0, int r1) {
            std::vector<size_t>& lc = local[w];
            for (IndptrT p = h_indptr[r0]; p < h_indptr[r1]; p++)
                lc[(size_t)h_indices[p]]++;
        });
        for (int w = 0; w < used; w++)
            for (int c = 0; c < n_cols; c++) h_col_counts[c] += local[w][c];
    }

    // Column batches must satisfy int32 CUB limits and device memory budget.
    // Per-nnz scratch covers mini-CSR gather, CSC accum, sort output, and CUB.
    constexpr size_t BYTES_PER_NNZ = 2 * sizeof(InT)  // gather val + csc val
                                     + 2 * sizeof(float)  // f32 key in + out
                                     + 3 * sizeof(int)    // gather col + 2 rows
                                     + 12;                // CUB temp headroom
    size_t cap = SAFE_BATCH_NNZ;
    size_t mem_cap = budget / BYTES_PER_NNZ;
    if (mem_cap > 0 && mem_cap < cap) cap = mem_cap;
    ColumnBatchPlan batches = plan_column_batches_from_counts(
        n_cols, sub_batch_cols, cap, [&](int c) { return h_col_counts[c]; },
        "rowstream rebased column offsets");
    sub_batch_cols = batches.sub_batch_cols;
    int n_batches = batches.n_batches;
    size_t max_batch_nnz = batches.max_nnz;

    size_t cub_temp_bytes = 0;
    if (max_batch_nnz > 0) {
        int mb_i32 =
            checked_cub_items(max_batch_nnz, "rowstream sub-batch nnz");
        cub_temp_bytes =
            cub_segmented_sortpairs_temp_bytes(mb_i32, sub_batch_cols);
    }
    bool rank_use_gmem = false;
    size_t smem_bytes = sparse_ovr_smem_config(n_groups, rank_use_gmem);
    bool cast_use_gmem = false;
    size_t smem_cast = cast_accumulate_smem_config(
        n_groups, compute_nnz, compute_totals, cast_use_gmem);

    // Host gather staging is pinned; full CSR stays pageable on CPU.
    // Only the compacted column interval crosses the bus.
    size_t stage_nnz = max_batch_nnz ? max_batch_nnz : 1;
    PinnedRing<InT, int> gather_stage(1, stage_nnz);
    PinnedRing<int> indptr_stage(1, (size_t)n_rows + 1);
    std::vector<IndptrT> cursor(n_rows, 0);  // offset within each (sorted) row

    int* d_group_codes = pool.alloc<int>(n_rows);
    double* d_group_sizes = pool.alloc<double>(n_groups);
    cudaMemcpy(d_group_codes, h_group_codes, n_rows * sizeof(int),
               cudaMemcpyHostToDevice);
    cudaMemcpy(d_group_sizes, h_group_sizes, n_groups * sizeof(double),
               cudaMemcpyHostToDevice);

    InT* d_gather_vals = pool.alloc<InT>(max_batch_nnz);
    int* d_gather_cols = pool.alloc<int>(max_batch_nnz);
    int* d_gather_indptr = pool.alloc<int>(n_rows + 1);
    int* col_offsets = pool.alloc<int>(sub_batch_cols + 1);
    int* write_pos = pool.alloc<int>(sub_batch_cols);
    int* d_all_offsets = upload_batch_offsets(batches, pool);
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
    ScopedCudaStream row_stream(cudaStreamDefault);
    cudaStream_t stream = row_stream.get();

    // One ascending column pass; sorted-row cursors make transfer one-shot.
    // Threaded gather counts row runs, prefix-sums, then copies disjoint
    // ranges.
    std::vector<int> g_count(n_rows);
    int col = 0;
    for (int b = 0; b < n_batches; b++) {
        int sb_cols = std::min(sub_batch_cols, n_cols - col);
        int col_end = col + sb_cols;
        gather_stage.wait(0);
        indptr_stage.wait(0);
        InT* h_gather_vals = gather_stage.template get<0>(0);
        int* h_gather_cols = gather_stage.template get<1>(0);
        int* h_gather_indptr = indptr_stage.template get<0>(0);

        int batch_nnz = host_materialize_csr_column_interval_cursor(
            h_data, h_indices, h_indptr, n_rows, col, col_end, cursor.data(),
            g_count.data(), h_gather_indptr, h_gather_vals, h_gather_cols,
            "rowstream gather nnz");

        int* off = d_all_offsets + (size_t)b * (sub_batch_cols + 1);
        cudaMemcpyAsync(col_offsets, off, (sb_cols + 1) * sizeof(int),
                        cudaMemcpyDeviceToDevice, stream);
        cudaMemcpyAsync(write_pos, off, sb_cols * sizeof(int),
                        cudaMemcpyDeviceToDevice, stream);

        // Bulk H2D of this batch's compacted nonzeros (1x transfer).
        if (batch_nnz > 0) {
            cuda_check(cudaMemcpyAsync(d_gather_vals, h_gather_vals,
                                       (size_t)batch_nnz * sizeof(InT),
                                       cudaMemcpyHostToDevice, stream),
                       "rowstream gathered vals H2D");
            cuda_check(cudaMemcpyAsync(d_gather_cols, h_gather_cols,
                                       (size_t)batch_nnz * sizeof(int),
                                       cudaMemcpyHostToDevice, stream),
                       "rowstream gathered cols H2D");
        }
        cudaMemcpyAsync(d_gather_indptr, h_gather_indptr,
                        (n_rows + 1) * sizeof(int), cudaMemcpyHostToDevice,
                        stream);
        gather_stage.record(0, stream);
        indptr_stage.record(0, stream);

        // Scatter mini-CSR into the column-batch CSC accumulator.
        csr_scatter_to_csc_kernel<InT, int, int>
            <<<(n_rows + tpb - 1) / tpb, tpb, 0, stream>>>(
                d_gather_vals, d_gather_cols, d_gather_indptr, write_pos,
                csc_vals_orig, csc_row_idx, n_rows, col, col_end, 0);
        CUDA_CHECK_LAST_ERROR(csr_scatter_to_csc_kernel);

        launch_ovr_cast_and_accumulate_sparse<InT>(
            csc_vals_orig, csc_vals_f32, csc_row_idx, col_offsets,
            d_group_codes, sub_group_sums, sub_group_nnz, sub_total_sums,
            sub_total_nnz, sb_cols, n_groups, compute_nnz, compute_totals, tpb,
            smem_cast, cast_use_gmem, stream);
        if (batch_nnz > 0) {
            cub_segmented_sortpairs(cub_temp, cub_temp_bytes, csc_vals_f32,
                                    keys_out, csc_row_idx, vals_out, batch_nnz,
                                    sb_cols, col_offsets, col_offsets + 1,
                                    stream, "rowstream segmented sort");
        }
        launch_ovr_sparse_rank<int>(
            keys_out, vals_out, col_offsets, d_group_codes, d_group_sizes,
            sub_rank_sums, sub_tie_corr, d_nz_scratch, n_rows, sb_cols,
            n_groups, tpb, smem_bytes, compute_tie_corr, rank_use_gmem, stream);

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

// Host CSR sparse OVR stream: keep CSR on host and batch CSR->CSC scatter.
// Avoids full sparse upload and whole-matrix CSR->CSC conversion.
template <typename InT, typename IndexT, typename IndptrT>
static void ovr_sparse_csr_host_streaming_impl(
    const InT* h_data, const IndexT* h_indices, const IndptrT* h_indptr,
    const int* h_group_codes, const double* h_group_sizes, double* d_rank_sums,
    double* d_tie_corr, double* d_group_sums, double* d_group_nnz,
    double* d_total_sums, double* d_total_nnz, int n_rows, int n_cols,
    int n_groups, bool compute_tie_corr, bool compute_nnz, bool compute_totals,
    int sub_batch_cols) {
    if (n_rows == 0 || n_cols == 0) return;

    // Declared before pool/streams: on exception unwind streams drain (kernels
    // finish reading mapped host memory) before unregistration.
    HostRegisterGuard pin_data;
    HostRegisterGuard pin_indices;

    RmmScratchPool pool;
    size_t total_nnz = (size_t)h_indptr[n_rows];

    size_t budget = rmm_available_device_bytes(0.8);

    int tpb = UTIL_BLOCK_SIZE;
    size_t data_bytes = total_nnz * sizeof(InT);
    size_t idx_bytes = total_nnz * sizeof(IndexT);

    // Too large to stage on device: per-batch scatter would fall back to
    // bus-latency-bound zero-copy reads. Page the CSR through in row blocks.
    if (total_nnz > 0 && data_bytes + idx_bytes > (budget * 3) / 4) {
        ovr_sparse_csr_host_rowstream_impl<InT, IndexT, IndptrT>(
            h_data, h_indices, h_indptr, h_group_codes, h_group_sizes,
            d_rank_sums, d_tie_corr, d_group_sums, d_group_nnz, d_total_sums,
            d_total_nnz, n_rows, n_cols, n_groups, compute_tie_corr,
            compute_nnz, compute_totals, sub_batch_cols);
        return;
    }

    IndptrT* d_indptr_full = pool.alloc<IndptrT>(n_rows + 1);
    cudaMemcpy(d_indptr_full, h_indptr, (n_rows + 1) * sizeof(IndptrT),
               cudaMemcpyHostToDevice);

    // Stage indices first when they fit so histogram/scatter read at HBM speed.
    // Data is staged too only if data plus one stream buffer still fits.
    IndexT* d_indices = nullptr;
    bool indices_staged = total_nnz > 0 && idx_bytes <= budget / 2;
    if (total_nnz > 0) {
        if (indices_staged) {
            d_indices = pool.alloc<IndexT>(total_nnz);
            cuda_check(cudaMemcpy(d_indices, h_indices, idx_bytes,
                                  cudaMemcpyHostToDevice),
                       "OVR host CSR stage indices H2D");
        } else {
            pin_indices = HostRegisterGuard(const_cast<IndexT*>(h_indices),
                                            idx_bytes, cudaHostRegisterMapped);
            cuda_check(
                cudaHostGetDevicePointer((void**)&d_indices,
                                         const_cast<IndexT*>(h_indices), 0),
                "OVR host CSR map indices");
        }
    }

    // Count per-column nnz on GPU; CSR has no native column structure.
    // Only n_cols counts are copied back for batch planning.
    std::vector<unsigned int> h_col_counts(n_cols, 0);
    if (total_nnz > 0) {
        unsigned int* d_col_counts = pool.alloc<unsigned int>(n_cols);
        cudaMemset(d_col_counts, 0, n_cols * sizeof(unsigned int));
        int hist_blocks = (n_rows + tpb - 1) / tpb;
        csr_col_histogram_kernel<IndexT, IndptrT><<<hist_blocks, tpb>>>(
            d_indices, d_indptr_full, d_col_counts, n_rows, n_cols);
        CUDA_CHECK_LAST_ERROR(csr_col_histogram_kernel);
        cuda_check(
            cudaMemcpy(h_col_counts.data(), d_col_counts,
                       n_cols * sizeof(unsigned int), cudaMemcpyDeviceToHost),
            "OVR host CSR column-count D2H");
    }

    // Each batch uses one CUB segmented sort and per-stream CSR->CSC scratch.
    // Shrink sub_batch_cols until item counts and memory budget both fit.
    constexpr size_t BYTES_PER_NNZ = sizeof(InT) + sizeof(float) +
                                     2 * sizeof(int) + 8;  // buffers + CUB temp
    size_t batch_nnz_cap = SAFE_BATCH_NNZ;
    size_t mem_cap = budget / (size_t)N_STREAMS / BYTES_PER_NNZ;
    if (mem_cap > 0 && mem_cap < batch_nnz_cap) batch_nnz_cap = mem_cap;
    ColumnBatchPlan batches = plan_column_batches_from_counts(
        n_cols, sub_batch_cols, batch_nnz_cap,
        [&](int c) { return (size_t)h_col_counts[c]; },
        "OVR host CSR rebased column offsets");
    sub_batch_cols = batches.sub_batch_cols;
    int n_batches = batches.n_batches;
    size_t max_batch_nnz = batches.max_nnz;
    int* d_all_offsets = upload_batch_offsets(batches, pool);

    // ---- Phase 1: per-stream bounded work buffer size + stream count ----
    size_t cub_temp_bytes = 0;
    if (max_batch_nnz > 0) {
        int max_batch_nnz_i32 = checked_cub_items(
            max_batch_nnz, "OVR host CSR sparse sub-batch nnz");
        cub_temp_bytes = cub_segmented_sortpairs_temp_bytes(max_batch_nnz_i32,
                                                            sub_batch_cols);
    }

    bool rank_use_gmem = false;
    size_t smem_bytes = sparse_ovr_smem_config(n_groups, rank_use_gmem);
    bool cast_use_gmem = false;
    size_t smem_cast = cast_accumulate_smem_config(
        n_groups, compute_nnz, compute_totals, cast_use_gmem);

    size_t per_stream_bytes =
        max_batch_nnz * (sizeof(InT) + sizeof(float) + 2 * sizeof(int)) +
        (sub_batch_cols + 1 + sub_batch_cols) * sizeof(int) + cub_temp_bytes +
        2 * (size_t)n_groups * sub_batch_cols * sizeof(double) +
        sub_batch_cols * sizeof(double);
    if (compute_nnz) {
        per_stream_bytes += (size_t)n_groups * sub_batch_cols * sizeof(double);
    }
    if (compute_totals) {
        per_stream_bytes += sub_batch_cols * sizeof(double);
        if (compute_nnz) {
            per_stream_bytes += sub_batch_cols * sizeof(double);
        }
    }
    if (rank_use_gmem) {
        per_stream_bytes += (size_t)n_groups * sub_batch_cols * sizeof(double);
    }

    // Stage data when indices are resident and one transpose stream still fits.
    // Otherwise values stay mapped zero-copy for bounded-memory streaming.
    size_t resident = indices_staged ? idx_bytes : 0;
    bool data_staged = total_nnz > 0 && indices_staged &&
                       resident + data_bytes + per_stream_bytes <= budget;

    int n_streams = N_STREAMS;
    if (n_batches < n_streams) n_streams = n_batches;
    size_t stream_budget = budget - resident - (data_staged ? data_bytes : 0);
    n_streams =
        clamp_streams_by_budget(n_streams, per_stream_bytes, stream_budget);

    ScopedCudaStreams streams(n_streams, cudaStreamDefault);

    InT* d_data = nullptr;
    if (total_nnz > 0) {
        if (data_staged) {
            d_data = pool.alloc<InT>(total_nnz);
            cuda_check(
                cudaMemcpy(d_data, h_data, data_bytes, cudaMemcpyHostToDevice),
                "OVR host CSR stage data H2D");
        } else {
            pin_data = HostRegisterGuard(const_cast<InT*>(h_data), data_bytes,
                                         cudaHostRegisterMapped);
            cuda_check(cudaHostGetDevicePointer((void**)&d_data,
                                                const_cast<InT*>(h_data), 0),
                       "OVR host CSR map data");
        }
    }

    int* d_group_codes = pool.alloc<int>(n_rows);
    double* d_group_sizes = pool.alloc<double>(n_groups);
    cudaMemcpy(d_group_codes, h_group_codes, n_rows * sizeof(int),
               cudaMemcpyHostToDevice);
    cudaMemcpy(d_group_sizes, h_group_sizes, n_groups * sizeof(double),
               cudaMemcpyHostToDevice);

    int scatter_blocks = (n_rows + tpb - 1) / tpb;

    struct StreamBuf {
        int* col_offsets;
        int* write_pos;
        InT* csc_vals_orig;
        float* csc_vals_f32;
        int* csc_row_idx;
        float* keys_out;
        int* vals_out;
        uint8_t* cub_temp;
        double* sub_rank_sums;
        double* sub_tie_corr;
        double* sub_group_sums;
        double* sub_group_nnz;
        double* sub_total_sums;
        double* sub_total_nnz;
        double* d_nz_scratch;
    };
    std::vector<StreamBuf> bufs(n_streams);
    for (int s = 0; s < n_streams; s++) {
        bufs[s].col_offsets = pool.alloc<int>(sub_batch_cols + 1);
        bufs[s].write_pos = pool.alloc<int>(sub_batch_cols);
        bufs[s].csc_vals_orig = pool.alloc<InT>(max_batch_nnz);
        bufs[s].csc_vals_f32 = pool.alloc<float>(max_batch_nnz);
        bufs[s].csc_row_idx = pool.alloc<int>(max_batch_nnz);
        bufs[s].keys_out = pool.alloc<float>(max_batch_nnz);
        bufs[s].vals_out = pool.alloc<int>(max_batch_nnz);
        bufs[s].cub_temp = pool.alloc<uint8_t>(cub_temp_bytes);
        bufs[s].sub_rank_sums =
            pool.alloc<double>((size_t)n_groups * sub_batch_cols);
        bufs[s].sub_tie_corr = pool.alloc<double>(sub_batch_cols);
        bufs[s].sub_group_sums =
            pool.alloc<double>((size_t)n_groups * sub_batch_cols);
        bufs[s].sub_group_nnz =
            compute_nnz ? pool.alloc<double>((size_t)n_groups * sub_batch_cols)
                        : nullptr;
        bufs[s].sub_total_sums =
            compute_totals ? pool.alloc<double>(sub_batch_cols) : nullptr;
        bufs[s].sub_total_nnz = (compute_totals && compute_nnz)
                                    ? pool.alloc<double>(sub_batch_cols)
                                    : nullptr;
        bufs[s].d_nz_scratch =
            rank_use_gmem
                ? pool.alloc<double>((size_t)n_groups * sub_batch_cols)
                : nullptr;
    }

    cudaDeviceSynchronize();

    // ---- Phase 2: bounded CSR->CSC scatter + GPU rank batches ----
    int col = 0;
    for (int b = 0; b < n_batches; b++) {
        int sb_cols = std::min(sub_batch_cols, n_cols - col);
        int s = b % n_streams;
        auto stream = streams[s];
        auto& buf = bufs[s];
        int batch_nnz =
            checked_int_span(batches.nnz[b], "OVR host CSR active batch nnz");

        int* src = d_all_offsets + (size_t)b * (sub_batch_cols + 1);
        cudaMemcpyAsync(buf.col_offsets, src, (sb_cols + 1) * sizeof(int),
                        cudaMemcpyDeviceToDevice, stream);
        cudaMemcpyAsync(buf.write_pos, src, sb_cols * sizeof(int),
                        cudaMemcpyDeviceToDevice, stream);

        if (batch_nnz > 0) {
            csr_scatter_to_csc_kernel<InT, IndexT, IndptrT>
                <<<scatter_blocks, tpb, 0, stream>>>(
                    d_data, d_indices, d_indptr_full, buf.write_pos,
                    buf.csc_vals_orig, buf.csc_row_idx, n_rows, col,
                    col + sb_cols);
            CUDA_CHECK_LAST_ERROR(csr_scatter_to_csc_kernel);
        }

        launch_ovr_cast_and_accumulate_sparse<InT>(
            buf.csc_vals_orig, buf.csc_vals_f32, buf.csc_row_idx,
            buf.col_offsets, d_group_codes, buf.sub_group_sums,
            buf.sub_group_nnz, buf.sub_total_sums, buf.sub_total_nnz, sb_cols,
            n_groups, compute_nnz, compute_totals, tpb, smem_cast,
            cast_use_gmem, stream);

        if (batch_nnz > 0) {
            cub_segmented_sortpairs(
                buf.cub_temp, cub_temp_bytes, buf.csc_vals_f32, buf.keys_out,
                buf.csc_row_idx, buf.vals_out, batch_nnz, sb_cols,
                buf.col_offsets, buf.col_offsets + 1, stream,
                "host CSR OVR segmented sort");
        }

        launch_ovr_sparse_rank<int>(
            buf.keys_out, buf.vals_out, buf.col_offsets, d_group_codes,
            d_group_sizes, buf.sub_rank_sums, buf.sub_tie_corr,
            buf.d_nz_scratch, n_rows, sb_cols, n_groups, tpb, smem_bytes,
            compute_tie_corr, rank_use_gmem, stream);

        scatter_cols_2d(d_rank_sums + col, buf.sub_rank_sums, n_groups, n_cols,
                        sb_cols, stream);
        if (compute_tie_corr) {
            cudaMemcpyAsync(d_tie_corr + col, buf.sub_tie_corr,
                            sb_cols * sizeof(double), cudaMemcpyDeviceToDevice,
                            stream);
        }
        scatter_cols_2d(d_group_sums + col, buf.sub_group_sums, n_groups,
                        n_cols, sb_cols, stream);
        if (compute_nnz) {
            scatter_cols_2d(d_group_nnz + col, buf.sub_group_nnz, n_groups,
                            n_cols, sb_cols, stream);
        }
        if (compute_totals) {
            cudaMemcpyAsync(d_total_sums + col, buf.sub_total_sums,
                            sb_cols * sizeof(double), cudaMemcpyDeviceToDevice,
                            stream);
            if (compute_nnz) {
                cudaMemcpyAsync(d_total_nnz + col, buf.sub_total_nnz,
                                sb_cols * sizeof(double),
                                cudaMemcpyDeviceToDevice, stream);
            }
        }

        col += sb_cols;
    }

    sync_streams(streams, "sparse host CSR streaming");
}

// Sign-safe sparse OVR path: sparse window -> dense f32 tile -> dense rank.

constexpr int SPARSE_DENSE_OVR_CHUNK_COLS = 512;

static void launch_ovr_dense_rank_window(
    const float* dense, const int* group_codes, double* rank_sums,
    double* tie_corr, int out_col, int n_rows, int n_cols_total,
    int window_cols, int n_groups, bool compute_tie_corr,
    int rank_sub_batch_cols, cudaStream_t upstream_stream) {
    if (n_rows == 0 || window_cols == 0 || n_groups == 0) return;
    if (rank_sub_batch_cols <= 0) rank_sub_batch_cols = SUB_BATCH_COLS;

    DenseColumnBatchPlan batches = plan_dense_column_batches(
        n_rows, window_cols, rank_sub_batch_cols, SAFE_BATCH_NNZ,
        "Sparse-dense OVR rank sub-batch");
    rank_sub_batch_cols = batches.sub_batch_cols;
    int n_streams = clamp_streams_by_cols(window_cols, rank_sub_batch_cols);
    size_t sub_items = batches.max_items;
    int sub_items_i32 =
        checked_cub_items(sub_items, "Sparse-dense OVR rank sub-batch");
    size_t cub_temp_bytes =
        cub_segmented_sortpairs_temp_bytes(sub_items_i32, rank_sub_batch_cols);

    RmmScratchPool pool;
    ScopedCudaStreams streams(n_streams, cudaStreamNonBlocking);
    ScopedCudaEvent inputs_ready(cudaEventDisableTiming);
    inputs_ready.record(upstream_stream);
    for (int s = 0; s < n_streams; s++) {
        cuda_check(cudaStreamWaitEvent(streams[s], inputs_ready.get(), 0),
                   "wait on sparse-dense OVR tile");
    }

    struct StreamBuf {
        float* keys_out;
        int* vals_in;
        int* vals_out;
        int* seg_offsets;
        uint8_t* cub_temp;
        double* sub_rank_sums;
        double* sub_tie_corr;
    };
    std::vector<StreamBuf> bufs(n_streams);
    for (int s = 0; s < n_streams; s++) {
        bufs[s].keys_out = pool.alloc<float>(sub_items);
        bufs[s].vals_in = pool.alloc<int>(sub_items);
        bufs[s].vals_out = pool.alloc<int>(sub_items);
        bufs[s].seg_offsets = pool.alloc<int>(rank_sub_batch_cols + 1);
        bufs[s].cub_temp = pool.alloc<uint8_t>(cub_temp_bytes);
        bufs[s].sub_rank_sums =
            pool.alloc<double>((size_t)n_groups * rank_sub_batch_cols);
        bufs[s].sub_tie_corr = pool.alloc<double>(rank_sub_batch_cols);
    }

    int tpb_rank = round_up_to_warp(n_rows);
    bool use_gmem = false;
    size_t smem_rank = ovr_smem_config(n_groups, use_gmem);

    int col = 0;
    int batch_idx = 0;
    while (col < window_cols) {
        int sb_cols = std::min(rank_sub_batch_cols, window_cols - col);
        int sb_items =
            checked_int_product((size_t)n_rows, (size_t)sb_cols,
                                "Sparse-dense OVR active rank sub-batch");
        int s = batch_idx % n_streams;
        cudaStream_t stream = streams[s];
        auto& buf = bufs[s];

        upload_linear_offsets(buf.seg_offsets, sb_cols, n_rows, stream);
        fill_row_indices_kernel<<<sb_cols, UTIL_BLOCK_SIZE, 0, stream>>>(
            buf.vals_in, n_rows, sb_cols);
        CUDA_CHECK_LAST_ERROR(fill_row_indices_kernel);

        const float* keys_in = dense + (size_t)col * n_rows;
        cub_segmented_sortpairs(
            buf.cub_temp, cub_temp_bytes, keys_in, buf.keys_out, buf.vals_in,
            buf.vals_out, sb_items, sb_cols, buf.seg_offsets,
            buf.seg_offsets + 1, stream, "sparse-dense OVR segmented sort");

        if (use_gmem) {
            cuda_check(cudaMemsetAsync(
                           buf.sub_rank_sums, 0,
                           (size_t)n_groups * sb_cols * sizeof(double), stream),
                       "sparse-dense OVR gmem rank_sums memset");
        }
        rank_sums_from_sorted_kernel<<<sb_cols, tpb_rank, smem_rank, stream>>>(
            buf.keys_out, buf.vals_out, group_codes, buf.sub_rank_sums,
            buf.sub_tie_corr, n_rows, sb_cols, n_groups, compute_tie_corr,
            use_gmem);
        CUDA_CHECK_LAST_ERROR(rank_sums_from_sorted_kernel);

        scatter_cols_2d(rank_sums + out_col + col, buf.sub_rank_sums, n_groups,
                        n_cols_total, sb_cols, stream);
        if (compute_tie_corr) {
            cudaMemcpyAsync(tie_corr + out_col + col, buf.sub_tie_corr,
                            sb_cols * sizeof(double), cudaMemcpyDeviceToDevice,
                            stream);
        }

        col += sb_cols;
        batch_idx++;
    }

    sync_streams(streams, "sparse-dense OVR rank");
}

template <typename IndexT = int, typename IndptrT = int>
static void ovr_dense_csr_streaming_impl(
    const float* csr_data, const IndexT* csr_indices, const IndptrT* csr_indptr,
    const int* group_codes, double* rank_sums, double* tie_corr, int n_rows,
    int n_cols, int n_groups, bool compute_tie_corr, int chunk_cols,
    int rank_sub_batch_cols) {
    if (n_rows == 0 || n_cols == 0 || n_groups == 0) return;
    if (chunk_cols <= 0) chunk_cols = SPARSE_DENSE_OVR_CHUNK_COLS;

    DenseColumnBatchPlan chunks =
        plan_dense_column_batches(n_rows, n_cols, chunk_cols, SAFE_BATCH_NNZ,
                                  "Device CSR sparse-dense OVR column chunk");
    chunk_cols = chunks.sub_batch_cols;
    size_t max_dense_items = (size_t)n_rows * (size_t)chunk_cols;

    RmmScratchPool pool;
    ScopedCudaStream extract_stream(cudaStreamDefault);
    cudaStream_t stream = extract_stream.get();
    float* dense = pool.alloc<float>(max_dense_items);

    for (int col = 0; col < n_cols; col += chunk_cols) {
        int sb_cols = std::min(chunk_cols, n_cols - col);
        size_t dense_items = (size_t)n_rows * (size_t)sb_cols;
        cudaMemsetAsync(dense, 0, dense_items * sizeof(float), stream);
        int blocks = (n_rows + UTIL_BLOCK_SIZE - 1) / UTIL_BLOCK_SIZE;
        csr_tile_to_dense_kernel<float, IndptrT, IndexT, float>
            <<<blocks, UTIL_BLOCK_SIZE, 0, stream>>>(csr_indptr, csr_indices,
                                                     csr_data, dense, col,
                                                     col + sb_cols, n_rows);
        CUDA_CHECK_LAST_ERROR(csr_tile_to_dense_kernel);
        launch_ovr_dense_rank_window(
            dense, group_codes, rank_sums, tie_corr, col, n_rows, n_cols,
            sb_cols, n_groups, compute_tie_corr, rank_sub_batch_cols, stream);
    }
}

template <typename IndexT = int, typename IndptrT = int>
static void ovr_dense_csc_streaming_impl(
    const float* csc_data, const IndexT* csc_indices, const IndptrT* csc_indptr,
    const int* group_codes, double* rank_sums, double* tie_corr, int n_rows,
    int n_cols, int n_groups, bool compute_tie_corr, int chunk_cols,
    int rank_sub_batch_cols) {
    if (n_rows == 0 || n_cols == 0 || n_groups == 0) return;
    if (chunk_cols <= 0) chunk_cols = SPARSE_DENSE_OVR_CHUNK_COLS;

    DenseColumnBatchPlan chunks =
        plan_dense_column_batches(n_rows, n_cols, chunk_cols, SAFE_BATCH_NNZ,
                                  "Device CSC sparse-dense OVR column chunk");
    chunk_cols = chunks.sub_batch_cols;
    size_t max_dense_items = (size_t)n_rows * (size_t)chunk_cols;

    RmmScratchPool pool;
    ScopedCudaStream extract_stream(cudaStreamDefault);
    cudaStream_t stream = extract_stream.get();
    float* dense = pool.alloc<float>(max_dense_items);

    for (int col = 0; col < n_cols; col += chunk_cols) {
        int sb_cols = std::min(chunk_cols, n_cols - col);
        size_t dense_items = (size_t)n_rows * (size_t)sb_cols;
        cudaMemsetAsync(dense, 0, dense_items * sizeof(float), stream);
        csc_tile_to_dense_kernel<float, IndptrT, IndexT, float>
            <<<sb_cols, UTIL_BLOCK_SIZE, 0, stream>>>(csc_indptr, csc_indices,
                                                      csc_data, dense, col,
                                                      col + sb_cols, n_rows);
        CUDA_CHECK_LAST_ERROR(csc_tile_to_dense_kernel);
        launch_ovr_dense_rank_window(
            dense, group_codes, rank_sums, tie_corr, col, n_rows, n_cols,
            sb_cols, n_groups, compute_tie_corr, rank_sub_batch_cols, stream);
    }
}

template <typename InT, typename IndexT, typename IndptrT>
static void ovr_dense_csc_host_streaming_impl(
    const InT* h_data, const IndexT* h_indices, const IndptrT* h_indptr,
    const int* h_group_codes, double* d_rank_sums, double* d_tie_corr,
    double* d_group_sums, double* d_group_nnz, double* d_total_sums,
    double* d_total_nnz, int n_rows, int n_cols, int n_groups,
    bool compute_tie_corr, bool compute_stats, bool compute_nnz,
    bool compute_totals, int chunk_cols, int rank_sub_batch_cols) {
    if (n_rows == 0 || n_cols == 0 || n_groups == 0) return;
    if (chunk_cols <= 0) chunk_cols = SPARSE_DENSE_OVR_CHUNK_COLS;
    compute_nnz = compute_stats && compute_nnz && d_group_nnz != nullptr;
    compute_totals = compute_stats && compute_totals && d_total_sums != nullptr;

    DenseColumnBatchPlan dense_chunks =
        plan_dense_column_batches(n_rows, n_cols, chunk_cols, SAFE_BATCH_NNZ,
                                  "Host CSC sparse-dense OVR column chunk");
    chunk_cols = dense_chunks.sub_batch_cols;
    ColumnBatchPlan batches =
        plan_csc_column_batches(h_indptr, n_cols, chunk_cols, SAFE_BATCH_NNZ,
                                "Host CSC sparse-dense OVR offsets");
    chunk_cols = batches.sub_batch_cols;
    size_t max_nnz = batches.max_nnz;
    size_t max_dense_items = (size_t)n_rows * (size_t)chunk_cols;

    RmmScratchPool pool;
    PinnedRing<InT, IndexT> stage(1, max_nnz ? max_nnz : 1);
    ScopedCudaStream extract_stream(cudaStreamDefault);
    cudaStream_t stream = extract_stream.get();

    int* d_group_codes = pool.alloc<int>(n_rows);
    cudaMemcpy(d_group_codes, h_group_codes, n_rows * sizeof(int),
               cudaMemcpyHostToDevice);
    int* d_all_offsets = upload_batch_offsets(batches, pool);

    InT* d_sparse_data_orig = pool.alloc<InT>(max_nnz ? max_nnz : 1);
    float* d_sparse_data_f32 = pool.alloc<float>(max_nnz ? max_nnz : 1);
    IndexT* d_sparse_indices = pool.alloc<IndexT>(max_nnz ? max_nnz : 1);
    int* idx_i32 = (sizeof(IndexT) > sizeof(int))
                       ? pool.alloc<int>(max_nnz ? max_nnz : 1)
                       : nullptr;
    int* d_indptr = pool.alloc<int>(chunk_cols + 1);
    float* dense = pool.alloc<float>(max_dense_items);
    double* sub_group_sums =
        compute_stats ? pool.alloc<double>((size_t)n_groups * chunk_cols)
                      : nullptr;
    double* sub_group_nnz =
        compute_nnz ? pool.alloc<double>((size_t)n_groups * chunk_cols)
                    : nullptr;
    double* sub_total_sums =
        compute_totals ? pool.alloc<double>(chunk_cols) : nullptr;
    double* sub_total_nnz = (compute_totals && compute_nnz)
                                ? pool.alloc<double>(chunk_cols)
                                : nullptr;

    bool cast_use_gmem = false;
    size_t smem_cast = cast_accumulate_smem_config(
        n_groups, compute_nnz, compute_totals, cast_use_gmem);

    int col = 0;
    for (int b = 0; b < batches.n_batches; b++) {
        int sb_cols = std::min(chunk_cols, n_cols - col);
        IndptrT ptr_start = h_indptr[col];
        IndptrT ptr_end = h_indptr[col + sb_cols];
        int batch_nnz = checked_int_span((size_t)(ptr_end - ptr_start),
                                         "Host CSC sparse-dense active nnz");

        stage.wait(0);
        if (batch_nnz > 0) {
            host_copy_slice(h_data, h_indices, (size_t)ptr_start, batch_nnz,
                            stage.template get<0>(0), stage.template get<1>(0));
            cudaMemcpyAsync(d_sparse_data_orig, stage.template get<0>(0),
                            (size_t)batch_nnz * sizeof(InT),
                            cudaMemcpyHostToDevice, stream);
            cudaMemcpyAsync(d_sparse_indices, stage.template get<1>(0),
                            (size_t)batch_nnz * sizeof(IndexT),
                            cudaMemcpyHostToDevice, stream);
        }
        stage.record(0, stream);

        int* idx32;
        if constexpr (sizeof(IndexT) > sizeof(int)) {
            if (batch_nnz > 0) {
                int cblk = (batch_nnz + UTIL_BLOCK_SIZE - 1) / UTIL_BLOCK_SIZE;
                cast_array_kernel<IndexT, int>
                    <<<cblk, UTIL_BLOCK_SIZE, 0, stream>>>(
                        d_sparse_indices, idx_i32, (size_t)batch_nnz);
                CUDA_CHECK_LAST_ERROR(cast_array_kernel);
            }
            idx32 = idx_i32;
        } else {
            idx32 = reinterpret_cast<int*>(d_sparse_indices);
        }

        int* src = d_all_offsets + (size_t)b * (chunk_cols + 1);
        cudaMemcpyAsync(d_indptr, src, (sb_cols + 1) * sizeof(int),
                        cudaMemcpyDeviceToDevice, stream);

        launch_ovr_cast_and_accumulate_sparse<InT>(
            d_sparse_data_orig, d_sparse_data_f32, idx32, d_indptr,
            d_group_codes, sub_group_sums, sub_group_nnz, sub_total_sums,
            sub_total_nnz, sb_cols, n_groups, compute_nnz, compute_totals,
            UTIL_BLOCK_SIZE, smem_cast, cast_use_gmem, stream);

        size_t dense_items = (size_t)n_rows * (size_t)sb_cols;
        cudaMemsetAsync(dense, 0, dense_items * sizeof(float), stream);
        csc_tile_to_dense_kernel<float, int, int, float>
            <<<sb_cols, UTIL_BLOCK_SIZE, 0, stream>>>(
                d_indptr, idx32, d_sparse_data_f32, dense, 0, sb_cols, n_rows);
        CUDA_CHECK_LAST_ERROR(csc_tile_to_dense_kernel);

        if (compute_stats) {
            scatter_cols_2d(d_group_sums + col, sub_group_sums, n_groups,
                            n_cols, sb_cols, stream);
            if (compute_nnz) {
                scatter_cols_2d(d_group_nnz + col, sub_group_nnz, n_groups,
                                n_cols, sb_cols, stream);
            }
            if (compute_totals) {
                cudaMemcpyAsync(d_total_sums + col, sub_total_sums,
                                sb_cols * sizeof(double),
                                cudaMemcpyDeviceToDevice, stream);
                if (compute_nnz) {
                    cudaMemcpyAsync(d_total_nnz + col, sub_total_nnz,
                                    sb_cols * sizeof(double),
                                    cudaMemcpyDeviceToDevice, stream);
                }
            }
        }

        launch_ovr_dense_rank_window(
            dense, d_group_codes, d_rank_sums, d_tie_corr, col, n_rows, n_cols,
            sb_cols, n_groups, compute_tie_corr, rank_sub_batch_cols, stream);
        col += sb_cols;
    }
}

template <typename InT, typename IndexT, typename IndptrT>
static void ovr_dense_csr_host_streaming_impl(
    const InT* h_data, const IndexT* h_indices, const IndptrT* h_indptr,
    const int* h_group_codes, double* d_rank_sums, double* d_tie_corr,
    double* d_group_sums, double* d_group_nnz, double* d_total_sums,
    double* d_total_nnz, int n_rows, int n_cols, int n_groups,
    bool compute_tie_corr, bool compute_stats, bool compute_nnz,
    bool compute_totals, int chunk_cols, int rank_sub_batch_cols) {
    if (n_rows == 0 || n_cols == 0 || n_groups == 0) return;
    if (chunk_cols <= 0) chunk_cols = SPARSE_DENSE_OVR_CHUNK_COLS;
    compute_nnz = compute_stats && compute_nnz && d_group_nnz != nullptr;
    compute_totals = compute_stats && compute_totals && d_total_sums != nullptr;

    DenseColumnBatchPlan dense_chunks =
        plan_dense_column_batches(n_rows, n_cols, chunk_cols, SAFE_BATCH_NNZ,
                                  "Host CSR sparse-dense OVR column chunk");
    chunk_cols = dense_chunks.sub_batch_cols;

    std::vector<size_t> h_col_counts(n_cols, 0);
    {
        int n_workers = host_worker_count();
        std::vector<std::vector<size_t>> local(n_workers,
                                               std::vector<size_t>(n_cols, 0));
        int used = host_parallel_chunks(n_rows, [&](int w, int r0, int r1) {
            std::vector<size_t>& lc = local[w];
            for (IndptrT p = h_indptr[r0]; p < h_indptr[r1]; p++)
                lc[(size_t)h_indices[p]]++;
        });
        for (int w = 0; w < used; w++)
            for (int c = 0; c < n_cols; c++) h_col_counts[c] += local[w][c];
    }

    size_t cap = SAFE_BATCH_NNZ;
    ColumnBatchPlan batches = plan_column_batches_from_counts(
        n_cols, chunk_cols, cap, [&](int c) { return h_col_counts[c]; },
        "Host CSR sparse-dense OVR offsets");
    chunk_cols = batches.sub_batch_cols;
    size_t max_batch_nnz = batches.max_nnz;
    size_t max_dense_items = (size_t)n_rows * (size_t)chunk_cols;

    RmmScratchPool pool;
    ScopedCudaStream extract_stream(cudaStreamDefault);
    cudaStream_t stream = extract_stream.get();
    PinnedRing<InT, int> gather_stage(1, max_batch_nnz ? max_batch_nnz : 1);
    PinnedRing<int> indptr_stage(1, (size_t)n_rows + 1);
    std::vector<IndptrT> cursor(n_rows, 0);
    std::vector<int> row_counts(n_rows);

    int* d_group_codes = pool.alloc<int>(n_rows);
    cudaMemcpy(d_group_codes, h_group_codes, n_rows * sizeof(int),
               cudaMemcpyHostToDevice);
    int* d_all_offsets = upload_batch_offsets(batches, pool);

    InT* d_gather_vals = pool.alloc<InT>(max_batch_nnz ? max_batch_nnz : 1);
    int* d_gather_cols = pool.alloc<int>(max_batch_nnz ? max_batch_nnz : 1);
    int* d_gather_indptr = pool.alloc<int>(n_rows + 1);
    int* col_offsets = pool.alloc<int>(chunk_cols + 1);
    int* write_pos = pool.alloc<int>(chunk_cols);
    InT* csc_vals_orig = pool.alloc<InT>(max_batch_nnz ? max_batch_nnz : 1);
    float* csc_vals_f32 = pool.alloc<float>(max_batch_nnz ? max_batch_nnz : 1);
    int* csc_row_idx = pool.alloc<int>(max_batch_nnz ? max_batch_nnz : 1);
    float* dense = pool.alloc<float>(max_dense_items);
    double* sub_group_sums =
        compute_stats ? pool.alloc<double>((size_t)n_groups * chunk_cols)
                      : nullptr;
    double* sub_group_nnz =
        compute_nnz ? pool.alloc<double>((size_t)n_groups * chunk_cols)
                    : nullptr;
    double* sub_total_sums =
        compute_totals ? pool.alloc<double>(chunk_cols) : nullptr;
    double* sub_total_nnz = (compute_totals && compute_nnz)
                                ? pool.alloc<double>(chunk_cols)
                                : nullptr;

    bool cast_use_gmem = false;
    size_t smem_cast = cast_accumulate_smem_config(
        n_groups, compute_nnz, compute_totals, cast_use_gmem);

    int col = 0;
    for (int b = 0; b < batches.n_batches; b++) {
        int sb_cols = std::min(chunk_cols, n_cols - col);
        int col_end = col + sb_cols;
        gather_stage.wait(0);
        indptr_stage.wait(0);
        InT* h_gather_vals = gather_stage.template get<0>(0);
        int* h_gather_cols = gather_stage.template get<1>(0);
        int* h_gather_indptr = indptr_stage.template get<0>(0);

        int batch_nnz = host_materialize_csr_column_interval_cursor(
            h_data, h_indices, h_indptr, n_rows, col, col_end, cursor.data(),
            row_counts.data(), h_gather_indptr, h_gather_vals, h_gather_cols,
            "Host CSR sparse-dense gather nnz");

        int* src = d_all_offsets + (size_t)b * (chunk_cols + 1);
        cudaMemcpyAsync(col_offsets, src, (sb_cols + 1) * sizeof(int),
                        cudaMemcpyDeviceToDevice, stream);
        cudaMemcpyAsync(write_pos, src, sb_cols * sizeof(int),
                        cudaMemcpyDeviceToDevice, stream);

        if (batch_nnz > 0) {
            cudaMemcpyAsync(d_gather_vals, h_gather_vals,
                            (size_t)batch_nnz * sizeof(InT),
                            cudaMemcpyHostToDevice, stream);
            cudaMemcpyAsync(d_gather_cols, h_gather_cols,
                            (size_t)batch_nnz * sizeof(int),
                            cudaMemcpyHostToDevice, stream);
        }
        cudaMemcpyAsync(d_gather_indptr, h_gather_indptr,
                        (n_rows + 1) * sizeof(int), cudaMemcpyHostToDevice,
                        stream);
        gather_stage.record(0, stream);
        indptr_stage.record(0, stream);

        if (batch_nnz > 0) {
            csr_scatter_to_csc_kernel<InT, int, int>
                <<<(n_rows + UTIL_BLOCK_SIZE - 1) / UTIL_BLOCK_SIZE,
                   UTIL_BLOCK_SIZE, 0, stream>>>(
                    d_gather_vals, d_gather_cols, d_gather_indptr, write_pos,
                    csc_vals_orig, csc_row_idx, n_rows, col, col_end, 0);
            CUDA_CHECK_LAST_ERROR(csr_scatter_to_csc_kernel);
        }

        launch_ovr_cast_and_accumulate_sparse<InT>(
            csc_vals_orig, csc_vals_f32, csc_row_idx, col_offsets,
            d_group_codes, sub_group_sums, sub_group_nnz, sub_total_sums,
            sub_total_nnz, sb_cols, n_groups, compute_nnz, compute_totals,
            UTIL_BLOCK_SIZE, smem_cast, cast_use_gmem, stream);

        size_t dense_items = (size_t)n_rows * (size_t)sb_cols;
        cudaMemsetAsync(dense, 0, dense_items * sizeof(float), stream);
        csc_tile_to_dense_kernel<float, int, int, float>
            <<<sb_cols, UTIL_BLOCK_SIZE, 0, stream>>>(col_offsets, csc_row_idx,
                                                      csc_vals_f32, dense, 0,
                                                      sb_cols, n_rows);
        CUDA_CHECK_LAST_ERROR(csc_tile_to_dense_kernel);

        if (compute_stats) {
            scatter_cols_2d(d_group_sums + col, sub_group_sums, n_groups,
                            n_cols, sb_cols, stream);
            if (compute_nnz) {
                scatter_cols_2d(d_group_nnz + col, sub_group_nnz, n_groups,
                                n_cols, sb_cols, stream);
            }
            if (compute_totals) {
                cudaMemcpyAsync(d_total_sums + col, sub_total_sums,
                                sb_cols * sizeof(double),
                                cudaMemcpyDeviceToDevice, stream);
                if (compute_nnz) {
                    cudaMemcpyAsync(d_total_nnz + col, sub_total_nnz,
                                    sb_cols * sizeof(double),
                                    cudaMemcpyDeviceToDevice, stream);
                }
            }
        }

        launch_ovr_dense_rank_window(
            dense, d_group_codes, d_rank_sums, d_tie_corr, col, n_rows, n_cols,
            sb_cols, n_groups, compute_tie_corr, rank_sub_batch_cols, stream);
        col += sb_cols;
    }
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
