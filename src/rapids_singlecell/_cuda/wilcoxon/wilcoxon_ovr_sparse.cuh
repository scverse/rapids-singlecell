#pragma once

/**
 * Sparse-aware host-streaming CSC OVR pipeline.
 *
 * Like ovr_streaming_csc_host_impl but sorts only stored nonzeros per column
 * instead of extracting dense blocks.  GPU memory is O(max_batch_nnz) instead
 * of O(sub_batch * n_rows), and sort work is proportional to nnz, not n_rows.
 */
template <typename InT, typename IndexT, typename IndptrT>
static void ovr_sparse_csc_host_streaming_impl(
    const InT* h_data, const IndexT* h_indices, const IndptrT* h_indptr,
    const int* h_group_codes, const double* h_group_sizes, double* d_rank_sums,
    double* d_tie_corr, double* d_group_sums, double* d_group_nnz, int n_rows,
    int n_cols, int n_groups, bool compute_tie_corr, bool compute_nnz,
    int sub_batch_cols) {
    if (n_rows == 0 || n_cols == 0) return;

    // Bound each column batch's nnz so CUB item counts stay within int32 and
    // the per-stream sort buffers fit the memory budget (column counts come
    // free from the CSC indptr).
    {
        constexpr size_t BYTES_PER_NNZ =
            sizeof(InT) + 2 * sizeof(float) + 2 * sizeof(IndexT) + 8;
        size_t cap = SAFE_BATCH_NNZ;
        size_t mem_cap =
            rmm_available_device_bytes(0.8) / (size_t)N_STREAMS / BYTES_PER_NNZ;
        if (mem_cap > 0 && mem_cap < cap) cap = mem_cap;
        sub_batch_cols = cap_sub_batch_by_nnz(
            n_cols, sub_batch_cols, cap,
            [&](int c) { return (size_t)(h_indptr[c + 1] - h_indptr[c]); });
    }

    int n_streams = clamp_streams_by_cols(n_cols, sub_batch_cols);

    size_t max_nnz = 0;
    for (int col = 0; col < n_cols; col += sub_batch_cols) {
        int sb_cols = std::min(sub_batch_cols, n_cols - col);
        size_t nnz = (size_t)(h_indptr[col + sb_cols] - h_indptr[col]);
        if (nnz > max_nnz) max_nnz = nnz;
    }

    size_t cub_temp_bytes = 0;
    if (max_nnz > 0) {
        int max_nnz_i32 =
            checked_cub_items(max_nnz, "OVR host CSC sparse sub-batch nnz");
        cub_temp_bytes =
            cub_segmented_sortpairs_temp_bytes(max_nnz_i32, sub_batch_cols);
    }

    // pool first: streams drain before it frees their scratch (see guard doc).
    RmmScratchPool pool;
    // Pin host inputs before the streams so on an exception unwind the streams
    // drain before the buffers are unregistered (mirrors the safe CSR order).
    size_t total_nnz = (size_t)h_indptr[n_cols];
    HostRegisterGuard _pin_data(const_cast<InT*>(h_data),
                                total_nnz * sizeof(InT));
    HostRegisterGuard _pin_indices(const_cast<IndexT*>(h_indices),
                                   total_nnz * sizeof(IndexT));
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
    }

    cudaMemcpy(d_group_codes, h_group_codes, n_rows * sizeof(int),
               cudaMemcpyHostToDevice);
    cudaMemcpy(d_group_sizes, h_group_sizes, n_groups * sizeof(double),
               cudaMemcpyHostToDevice);

    // Pre-compute rebased per-batch offsets and upload once (avoids per-batch
    // H2D copy from a transient host buffer).
    int n_batches = (n_cols + sub_batch_cols - 1) / sub_batch_cols;
    int* d_all_offsets = precompute_csc_batch_offsets(
        h_indptr, n_cols, sub_batch_cols, n_batches, pool,
        "OVR host CSC rebased column offsets");

    int tpb = UTIL_BLOCK_SIZE;
    bool rank_use_gmem = false;
    size_t smem_bytes = sparse_ovr_smem_config(n_groups, rank_use_gmem);
    bool cast_use_gmem = false;
    size_t smem_cast =
        cast_accumulate_smem_config(n_groups, compute_nnz, cast_use_gmem);

    // In gmem mode the sparse rank kernel accumulates into rank_sums directly
    // and needs a per-stream nz_count scratch buffer sized (n_groups, sb_cols).
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

        // H2D: this column range's sparse data (native dtype)
        if (batch_nnz > 0) {
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
            sb_cols, n_groups, compute_nnz, tpb, smem_cast, cast_use_gmem,
            stream);

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

        col += sb_cols;
        batch_idx++;
    }

    sync_streams(streams, "sparse host CSC streaming");
}

// ============================================================================
// Sparse-aware host-streaming CSR OVR pipeline.
// ============================================================================

/**
 * Out-of-core OVR for a host CSR too large to stage on the GPU.
 *
 * Column indices are sorted within each row, so a per-row cursor (init 0) lets
 * us walk the matrix ONCE: for each ascending column batch [col, col_end) every
 * row resumes where the previous batch stopped and emits its run of columns in
 * range. The cursor advances monotonically, so each nonzero is read on the host
 * and bulk-transferred exactly once over the matrix's lifetime -- the gathered,
 * already-compacted per-batch slice is the only thing that crosses the bus (a
 * true 1x transfer, versus re-streaming the whole CSR per batch). The column
 * histogram is counted on the host; the full CSR is never page-locked (the
 * gather reads it on the CPU). Single stream: the per-batch CSC accumulator
 * plus the gather buffers fill the device.
 */
template <typename InT, typename IndexT, typename IndptrT>
static void ovr_sparse_csr_host_rowstream_impl(
    const InT* h_data, const IndexT* h_indices, const IndptrT* h_indptr,
    const int* h_group_codes, const double* h_group_sizes, double* d_rank_sums,
    double* d_tie_corr, double* d_group_sums, double* d_group_nnz, int n_rows,
    int n_cols, int n_groups, bool compute_tie_corr, bool compute_nnz,
    int sub_batch_cols) {
    if (n_rows == 0 || n_cols == 0) return;
    size_t total_nnz = (size_t)h_indptr[n_rows];

    RmmScratchPool pool;
    int tpb = UTIL_BLOCK_SIZE;
    size_t budget = rmm_available_device_bytes(0.8);

    // ---- Phase 0: column histogram on the host, threaded by row range. Each
    // worker counts into a private array (no false sharing), merged after. ----
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

    // ---- Column batch size: int32 CUB limit + device buffers that fit the
    // budget. Per-nnz device footprint: gathered mini-CSR (val + col) + CSC
    // accumulator (val + f32 + row) + sort outputs (key + row) + CUB temp. ----
    constexpr size_t BYTES_PER_NNZ = 2 * sizeof(InT)  // gather val + csc val
                                     + 2 * sizeof(float)  // f32 key in + out
                                     + 3 * sizeof(int)    // gather col + 2 rows
                                     + 12;                // CUB temp headroom
    size_t cap = SAFE_BATCH_NNZ;
    size_t mem_cap = budget / BYTES_PER_NNZ;
    if (mem_cap > 0 && mem_cap < cap) cap = mem_cap;
    sub_batch_cols = cap_sub_batch_by_nnz(
        n_cols, sub_batch_cols, cap, [&](int c) { return h_col_counts[c]; });

    int n_batches = (n_cols + sub_batch_cols - 1) / sub_batch_cols;
    size_t max_batch_nnz = 0;
    std::vector<int> h_all_offsets((size_t)n_batches * (sub_batch_cols + 1), 0);
    std::vector<size_t> h_batch_nnz(n_batches);
    for (int b = 0; b < n_batches; b++) {
        int col_start = b * sub_batch_cols;
        int sb = std::min(sub_batch_cols, n_cols - col_start);
        int* off = &h_all_offsets[(size_t)b * (sub_batch_cols + 1)];
        for (int i = 0; i < sb; i++)
            off[i + 1] =
                checked_int_span((size_t)off[i] + h_col_counts[col_start + i],
                                 "rowstream rebased column offsets");
        h_batch_nnz[b] = (size_t)off[sb];
        if (h_batch_nnz[b] > max_batch_nnz) max_batch_nnz = h_batch_nnz[b];
    }

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
    size_t smem_cast =
        cast_accumulate_smem_config(n_groups, compute_nnz, cast_use_gmem);

    // ---- Host gather staging (pinned for fast bulk H2D) + per-row cursor.
    // Full CSR is NOT page-locked: gather reads it on the CPU, only the
    // compacted per-batch slice crosses the bus. ----
    size_t stage_nnz = max_batch_nnz ? max_batch_nnz : 1;
    std::vector<InT> h_gather_vals(stage_nnz);
    std::vector<int> h_gather_cols(stage_nnz);
    std::vector<int> h_gather_indptr(n_rows + 1);
    HostRegisterGuard pin_gvals(h_gather_vals.data(), stage_nnz * sizeof(InT));
    HostRegisterGuard pin_gcols(h_gather_cols.data(), stage_nnz * sizeof(int));
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
    double* d_nz_scratch =
        rank_use_gmem ? pool.alloc<double>((size_t)n_groups * sub_batch_cols)
                      : nullptr;

    // ---- One linear pass, column-batched. The per-row cursor advances
    // monotonically (indices sorted + batches ascending), so each nonzero is
    // read and transferred exactly once -- no whole-matrix re-streaming. Gather
    // is threaded: count each row's run, prefix-sum to per-row output offsets,
    // copy rows in parallel into disjoint staging ranges. ----
    std::vector<int> g_count(n_rows);
    int col = 0;
    for (int b = 0; b < n_batches; b++) {
        int sb_cols = std::min(sub_batch_cols, n_cols - col);
        int col_end = col + sb_cols;

        // Count this batch's run per row: sorted indices -> binary search from
        // the cursor for the first column >= col_end.
        host_parallel_ranges(n_rows, [&](int r0, int r1) {
            for (int r = r0; r < r1; r++) {
                const IndexT* lo = h_indices + h_indptr[r] + cursor[r];
                const IndexT* hi = h_indices + h_indptr[r + 1];
                g_count[r] =
                    (int)(std::lower_bound(lo, hi, (IndexT)col_end) - lo);
            }
        });
        // Prefix sum -> per-row output offsets (gathered mini-CSR row pointer).
        h_gather_indptr[0] = 0;
        for (int r = 0; r < n_rows; r++)
            h_gather_indptr[r + 1] = checked_int_span(
                (size_t)h_gather_indptr[r] + (size_t)g_count[r],
                "rowstream gather nnz");
        int batch_nnz = h_gather_indptr[n_rows];
        // Copy each row's run into its slot and advance its cursor (disjoint
        // output ranges -> race-free).
        host_parallel_ranges(n_rows, [&](int r0, int r1) {
            for (int r = r0; r < r1; r++) {
                IndptrT base = h_indptr[r] + cursor[r];
                size_t gpos = (size_t)h_gather_indptr[r];
                int cnt = g_count[r];
                for (int k = 0; k < cnt; k++) {
                    h_gather_vals[gpos + k] = h_data[base + k];
                    h_gather_cols[gpos + k] = (int)h_indices[base + k];
                }
                cursor[r] += cnt;
            }
        });

        int* off = &h_all_offsets[(size_t)b * (sub_batch_cols + 1)];
        cudaMemcpy(col_offsets, off, (sb_cols + 1) * sizeof(int),
                   cudaMemcpyHostToDevice);
        cudaMemcpy(write_pos, off, sb_cols * sizeof(int),
                   cudaMemcpyHostToDevice);

        // Bulk H2D of this batch's compacted nonzeros (1x transfer).
        if (batch_nnz > 0) {
            cuda_check(cudaMemcpy(d_gather_vals, h_gather_vals.data(),
                                  (size_t)batch_nnz * sizeof(InT),
                                  cudaMemcpyHostToDevice),
                       "rowstream gathered vals H2D");
            cuda_check(cudaMemcpy(d_gather_cols, h_gather_cols.data(),
                                  (size_t)batch_nnz * sizeof(int),
                                  cudaMemcpyHostToDevice),
                       "rowstream gathered cols H2D");
        }
        cudaMemcpy(d_gather_indptr, h_gather_indptr.data(),
                   (n_rows + 1) * sizeof(int), cudaMemcpyHostToDevice);

        // Scatter mini-CSR into the column-batch CSC accumulator.
        csr_scatter_to_csc_kernel<InT, int, int>
            <<<(n_rows + tpb - 1) / tpb, tpb>>>(
                d_gather_vals, d_gather_cols, d_gather_indptr, write_pos,
                csc_vals_orig, csc_row_idx, n_rows, col, col_end, 0);
        CUDA_CHECK_LAST_ERROR(csr_scatter_to_csc_kernel);

        launch_ovr_cast_and_accumulate_sparse<InT>(
            csc_vals_orig, csc_vals_f32, csc_row_idx, col_offsets,
            d_group_codes, sub_group_sums, sub_group_nnz, sb_cols, n_groups,
            compute_nnz, tpb, smem_cast, cast_use_gmem, 0);
        if (batch_nnz > 0) {
            cub_segmented_sortpairs(cub_temp, cub_temp_bytes, csc_vals_f32,
                                    keys_out, csc_row_idx, vals_out, batch_nnz,
                                    sb_cols, col_offsets, col_offsets + 1, 0,
                                    "rowstream segmented sort");
        }
        launch_ovr_sparse_rank<int>(
            keys_out, vals_out, col_offsets, d_group_codes, d_group_sizes,
            sub_rank_sums, sub_tie_corr, d_nz_scratch, n_rows, sb_cols,
            n_groups, tpb, smem_bytes, compute_tie_corr, rank_use_gmem, 0);

        cudaMemcpy2D(d_rank_sums + col, n_cols * sizeof(double), sub_rank_sums,
                     sb_cols * sizeof(double), sb_cols * sizeof(double),
                     n_groups, cudaMemcpyDeviceToDevice);
        if (compute_tie_corr)
            cudaMemcpy(d_tie_corr + col, sub_tie_corr, sb_cols * sizeof(double),
                       cudaMemcpyDeviceToDevice);
        cudaMemcpy2D(d_group_sums + col, n_cols * sizeof(double),
                     sub_group_sums, sb_cols * sizeof(double),
                     sb_cols * sizeof(double), n_groups,
                     cudaMemcpyDeviceToDevice);
        if (compute_nnz)
            cudaMemcpy2D(d_group_nnz + col, n_cols * sizeof(double),
                         sub_group_nnz, sb_cols * sizeof(double),
                         sb_cols * sizeof(double), n_groups,
                         cudaMemcpyDeviceToDevice);
        col += sb_cols;
    }
    cuda_check(cudaDeviceSynchronize(), "rowstream sync");
}

/**
 * Host CSR variant of the sparse OVR stream.
 *
 * The CSR input stays in host memory.  We count columns once on the CPU, then
 * use mapped pinned CSR arrays for bounded per-column-batch CSR->CSC scatter
 * on the GPU.  This avoids both a full host->device sparse upload and any
 * whole-matrix CSR->CSC conversion.
 */
template <typename InT, typename IndexT, typename IndptrT>
static void ovr_sparse_csr_host_streaming_impl(
    const InT* h_data, const IndexT* h_indices, const IndptrT* h_indptr,
    const int* h_group_codes, const double* h_group_sizes, double* d_rank_sums,
    double* d_tie_corr, double* d_group_sums, double* d_group_nnz, int n_rows,
    int n_cols, int n_groups, bool compute_tie_corr, bool compute_nnz,
    int sub_batch_cols) {
    if (n_rows == 0 || n_cols == 0) return;

    // Declared before the pool/streams so on exception unwind the streams
    // drain (kernels finish reading any mapped host memory) before it is
    // unregistered.
    HostRegisterGuard pin_data;
    HostRegisterGuard pin_indices;

    RmmScratchPool pool;
    size_t total_nnz = (size_t)h_indptr[n_rows];

    size_t budget = rmm_available_device_bytes(0.8);

    int tpb = UTIL_BLOCK_SIZE;
    size_t data_bytes = total_nnz * sizeof(InT);
    size_t idx_bytes = total_nnz * sizeof(IndexT);

    // When the matrix is too large to stage on the device, the per-batch
    // scatter would fall back to bus-latency-bound zero-copy reads. Page the
    // CSR through the GPU in row blocks (pinned bulk H2D) instead.
    if (total_nnz > 0 && data_bytes + idx_bytes > (budget * 3) / 4) {
        ovr_sparse_csr_host_rowstream_impl<InT, IndexT, IndptrT>(
            h_data, h_indices, h_indptr, h_group_codes, h_group_sizes,
            d_rank_sums, d_tie_corr, d_group_sums, d_group_nnz, n_rows, n_cols,
            n_groups, compute_tie_corr, compute_nnz, sub_batch_cols);
        return;
    }

    IndptrT* d_indptr_full = pool.alloc<IndptrT>(n_rows + 1);
    cudaMemcpy(d_indptr_full, h_indptr, (n_rows + 1) * sizeof(IndptrT),
               cudaMemcpyHostToDevice);

    // Stage the indices on the device when they fit, so the per-column
    // histogram and the per-batch CSR->CSC scatter read them at HBM speed
    // rather than over the bus. Indices are needed by both, so they are staged
    // first; the (equally sized) data array is staged later only if it too
    // fits. A bulk pageable copy is driver-staged -- no host registration.
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

    // ---- Phase 0: per-column nnz counts on the GPU ----
    // CSR has no column structure, so counting on the CPU is a serial pass over
    // every nonzero. Histogram the device-accessible indices instead; only the
    // n_cols counts come back for the per-batch prefix sums.
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

    // Each column batch is sorted in one CUB segmented call (int32 item count)
    // and its CSR->CSC transpose lives in per-stream scratch (~BYTES_PER_NNZ
    // per stored nonzero). Shrink sub_batch_cols until the densest window fits
    // BOTH the int32 limit AND a per-stream slice of the budget, so very tall
    // matrices neither overflow CUB nor exhaust memory.
    constexpr size_t BYTES_PER_NNZ = sizeof(InT) + sizeof(float) +
                                     2 * sizeof(int) + 8;  // buffers + CUB temp
    size_t batch_nnz_cap = SAFE_BATCH_NNZ;
    size_t mem_cap = budget / (size_t)N_STREAMS / BYTES_PER_NNZ;
    if (mem_cap > 0 && mem_cap < batch_nnz_cap) batch_nnz_cap = mem_cap;
    sub_batch_cols =
        cap_sub_batch_by_nnz(n_cols, sub_batch_cols, batch_nnz_cap,
                             [&](int c) { return (size_t)h_col_counts[c]; });

    int n_batches = (n_cols + sub_batch_cols - 1) / sub_batch_cols;
    size_t max_batch_nnz = 0;
    std::vector<int> h_all_offsets((size_t)n_batches * (sub_batch_cols + 1), 0);
    std::vector<size_t> h_batch_nnz(n_batches);
    for (int b = 0; b < n_batches; b++) {
        int col_start = b * sub_batch_cols;
        int sb_cols = std::min(sub_batch_cols, n_cols - col_start);
        int* off = &h_all_offsets[(size_t)b * (sub_batch_cols + 1)];
        for (int i = 0; i < sb_cols; i++)
            off[i + 1] = checked_int_span(
                (size_t)off[i] + (size_t)h_col_counts[col_start + i],
                "OVR host CSR rebased column offsets");
        h_batch_nnz[b] = (size_t)off[sb_cols];
        if (h_batch_nnz[b] > max_batch_nnz) max_batch_nnz = h_batch_nnz[b];
    }

    int* d_all_offsets =
        pool.alloc<int>((size_t)n_batches * (sub_batch_cols + 1));
    cudaMemcpy(d_all_offsets, h_all_offsets.data(),
               h_all_offsets.size() * sizeof(int), cudaMemcpyHostToDevice);

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
    size_t smem_cast =
        cast_accumulate_smem_config(n_groups, compute_nnz, cast_use_gmem);

    size_t per_stream_bytes =
        max_batch_nnz * (sizeof(InT) + sizeof(float) + 2 * sizeof(int)) +
        (sub_batch_cols + 1 + sub_batch_cols) * sizeof(int) + cub_temp_bytes +
        2 * (size_t)n_groups * sub_batch_cols * sizeof(double) +
        sub_batch_cols * sizeof(double);
    if (compute_nnz) {
        per_stream_bytes += (size_t)n_groups * sub_batch_cols * sizeof(double);
    }
    if (rank_use_gmem) {
        per_stream_bytes += (size_t)n_groups * sub_batch_cols * sizeof(double);
    }

    // Stage the data array too when the indices are already resident and the
    // data + at least one stream's transpose buffers still fit; the scatter
    // then reads values at HBM speed. Otherwise the data stays mapped
    // zero-copy (bounded for matrices too large to stage).
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
            checked_int_span(h_batch_nnz[b], "OVR host CSR active batch nnz");

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
            buf.sub_group_nnz, sb_cols, n_groups, compute_nnz, tpb, smem_cast,
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

        col += sb_cols;
    }

    sync_streams(streams, "sparse host CSR streaming");
}

// ============================================================================
// Sparse-aware CSC OVR streaming (sort only stored nonzeros)
// ============================================================================

template <typename IndexT = int, typename IndptrT = int>
static void ovr_sparse_csc_streaming_impl(
    const float* csc_data, const IndexT* csc_indices, const IndptrT* csc_indptr,
    const int* group_codes, const double* group_sizes, double* rank_sums,
    double* tie_corr, int n_rows, int n_cols, int n_groups,
    bool compute_tie_corr, int sub_batch_cols) {
    if (n_rows == 0 || n_cols == 0) return;

    // Read indptr to host for batch planning
    std::vector<IndptrT> h_indptr(n_cols + 1);
    cudaMemcpy(h_indptr.data(), csc_indptr, (n_cols + 1) * sizeof(IndptrT),
               cudaMemcpyDeviceToHost);

    // Bound each column batch's nnz so CUB item counts stay within int32 and
    // the per-stream sort buffers fit the budget.
    {
        constexpr size_t BYTES_PER_NNZ =
            2 * sizeof(float) + 2 * sizeof(int) + 8;
        size_t cap = SAFE_BATCH_NNZ;
        size_t mem_cap =
            rmm_available_device_bytes(0.8) / (size_t)N_STREAMS / BYTES_PER_NNZ;
        if (mem_cap > 0 && mem_cap < cap) cap = mem_cap;
        sub_batch_cols = cap_sub_batch_by_nnz(
            n_cols, sub_batch_cols, cap,
            [&](int c) { return (size_t)(h_indptr[c + 1] - h_indptr[c]); });
    }

    int n_streams = clamp_streams_by_cols(n_cols, sub_batch_cols);

    size_t max_nnz = 0;
    for (int col = 0; col < n_cols; col += sub_batch_cols) {
        int sb_cols = std::min(sub_batch_cols, n_cols - col);
        size_t nnz = (size_t)(h_indptr[col + sb_cols] - h_indptr[col]);
        if (nnz > max_nnz) max_nnz = nnz;
    }

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

        // Compute rebased segment offsets on GPU (avoids host pinned-buffer
        // race)
        {
            int count = sb_cols + 1;
            int blk = (count + UTIL_BLOCK_SIZE - 1) / UTIL_BLOCK_SIZE;
            rebase_indptr_kernel<<<blk, UTIL_BLOCK_SIZE, 0, stream>>>(
                csc_indptr, buf.seg_offsets, col, count);
            CUDA_CHECK_LAST_ERROR(rebase_indptr_kernel);
        }

        // Sort only stored values (keys=data, vals=row_indices). Row indices
        // always fit int32 (n_rows < 2^31); downcast int64 input here so the
        // sort + rank stay int32 (half the val buffer) -- the device boundary.
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

// ============================================================================
// Sparse-aware CSR OVR streaming (partial CSR→CSC transpose per sub-batch)
// ============================================================================

/**
 * Sparse-aware OVR streaming pipeline for GPU CSR data.
 *
 * Phase 0: One histogram kernel counts nnz per column. D2H + host prefix sums
 *          give exact per-batch nnz and max_batch_nnz for buffer sizing.
 * Phase 1: Allocate per-stream buffers sized to max_batch_nnz.
 * Phase 2: For each sub-batch: scatter CSR→CSC (partial transpose via
 *          atomics) → CUB sort only nonzeros → sparse rank kernel.
 *
 * Compared to the dense CSR path, sort work drops by ~1/sparsity.
 */
template <typename IndexT = int, typename IndptrT = int>
static void ovr_sparse_csr_streaming_impl(
    const float* csr_data, const IndexT* csr_indices, const IndptrT* csr_indptr,
    const int* group_codes, const double* group_sizes, double* rank_sums,
    double* tie_corr, int n_rows, int n_cols, int n_groups,
    bool compute_tie_corr, int sub_batch_cols) {
    if (n_rows == 0 || n_cols == 0) return;

    // ---- Phase 0: Planning — count nnz per column via histogram ----
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

    // Bound each column batch's nnz so CUB item counts stay within int32 and
    // the per-stream transpose/sort buffers fit the budget.
    {
        constexpr size_t BYTES_PER_NNZ =
            2 * sizeof(float) + 2 * sizeof(int) + 8;
        size_t cap = SAFE_BATCH_NNZ;
        size_t mem_cap =
            rmm_available_device_bytes(0.8) / (size_t)N_STREAMS / BYTES_PER_NNZ;
        if (mem_cap > 0 && mem_cap < cap) cap = mem_cap;
        sub_batch_cols = cap_sub_batch_by_nnz(
            n_cols, sub_batch_cols, cap,
            [&](int c) { return (size_t)h_col_counts[c]; });
    }

    // Per-batch prefix sums on host
    int n_batches = (n_cols + sub_batch_cols - 1) / sub_batch_cols;
    size_t max_batch_nnz = 0;

    // Flat array: n_batches × (sub_batch_cols + 1) offsets
    std::vector<int> h_all_offsets((size_t)n_batches * (sub_batch_cols + 1), 0);
    std::vector<size_t> h_batch_nnz(n_batches);

    for (int b = 0; b < n_batches; b++) {
        int col_start = b * sub_batch_cols;
        int sb_cols = std::min(sub_batch_cols, n_cols - col_start);
        int* off = &h_all_offsets[(size_t)b * (sub_batch_cols + 1)];
        off[0] = 0;
        for (int i = 0; i < sb_cols; i++)
            off[i + 1] = checked_int_span(
                (size_t)off[i] + (size_t)h_col_counts[col_start + i],
                "OVR device CSR rebased column offsets");
        h_batch_nnz[b] = (size_t)off[sb_cols];
        if (h_batch_nnz[b] > max_batch_nnz) max_batch_nnz = h_batch_nnz[b];
    }

    // Upload all batch offsets in one H2D (~20 KB)
    int* d_all_offsets =
        pool.alloc<int>((size_t)n_batches * (sub_batch_cols + 1));
    cudaMemcpy(d_all_offsets, h_all_offsets.data(),
               h_all_offsets.size() * sizeof(int), cudaMemcpyHostToDevice);

    // ---- Phase 1: Allocate per-stream buffers ----
    size_t cub_temp_bytes = 0;
    if (max_batch_nnz > 0) {
        int max_batch_nnz_i32 = checked_cub_items(
            max_batch_nnz, "OVR device CSR sparse sub-batch nnz");
        cub_temp_bytes = cub_segmented_sortpairs_temp_bytes(max_batch_nnz_i32,
                                                            sub_batch_cols);
    }

    int n_streams = N_STREAMS;
    if (n_batches < n_streams) n_streams = n_batches;

    // CSR path needs 4 sort arrays per stream (scatter intermediates +
    // CUB output).  Fit stream count to available GPU memory.
    bool rank_use_gmem = false;
    size_t smem_bytes = sparse_ovr_smem_config(n_groups, rank_use_gmem);
    size_t per_stream_bytes =
        max_batch_nnz * (2 * sizeof(float) + 2 * sizeof(int)) +
        (sub_batch_cols + 1 + sub_batch_cols) * sizeof(int) + cub_temp_bytes +
        (size_t)n_groups * sub_batch_cols * sizeof(double) +
        sub_batch_cols * sizeof(double);
    if (rank_use_gmem) {
        // gmem rank fallback (n_groups too large for smem): per-stream
        // d_nz_scratch accumulator, same size as sub_rank_sums.
        per_stream_bytes += (size_t)n_groups * sub_batch_cols * sizeof(double);
    }

    size_t budget = rmm_available_device_bytes(0.8);
    n_streams = clamp_streams_by_budget(n_streams, per_stream_bytes, budget);

    ScopedCudaStreams streams(n_streams, cudaStreamDefault);

    int tpb = UTIL_BLOCK_SIZE;
    int scatter_blocks = (n_rows + tpb - 1) / tpb;

    struct StreamBuf {
        int* col_offsets;  // [sub_batch_cols + 1]  CSC-style offsets
        int* write_pos;    // [sub_batch_cols]      atomic write counters
        float* csc_vals;   // [max_batch_nnz]       transposed values
        int* csc_row_idx;  // [max_batch_nnz]       transposed row indices
        float* keys_out;   // [max_batch_nnz]       CUB sort output
        int* vals_out;     // [max_batch_nnz]       CUB sort output
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

    // ---- Phase 2: Stream loop ----
    int col = 0;
    for (int b = 0; b < n_batches; b++) {
        int sb_cols = std::min(sub_batch_cols, n_cols - col);
        int s = b % n_streams;
        auto stream = streams[s];
        auto& buf = bufs[s];
        int batch_nnz =
            checked_int_span(h_batch_nnz[b], "OVR device CSR active batch nnz");

        int* src = d_all_offsets + (size_t)b * (sub_batch_cols + 1);
        cudaMemcpyAsync(buf.col_offsets, src, (sb_cols + 1) * sizeof(int),
                        cudaMemcpyDeviceToDevice, stream);

        // write_pos = col_offsets[0..sb_cols-1] (same D2D source)
        cudaMemcpyAsync(buf.write_pos, src, sb_cols * sizeof(int),
                        cudaMemcpyDeviceToDevice, stream);

        if (batch_nnz > 0) {
            // Scatter CSR → CSC for this sub-batch
            csr_scatter_to_csc_kernel<<<scatter_blocks, tpb, 0, stream>>>(
                csr_data, csr_indices, csr_indptr, buf.write_pos, buf.csc_vals,
                buf.csc_row_idx, n_rows, col, col + sb_cols);
            CUDA_CHECK_LAST_ERROR(csr_scatter_to_csc_kernel);

            // Sort only the nonzeros
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
