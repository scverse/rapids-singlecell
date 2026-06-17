#pragma once

/**
 * CSR-direct OVO streaming pipeline.
 *
 * One C++ call does everything.  Reference rows are extracted and sorted once
 * across all columns, then each group sub-batch ranks against that cached
 * reference slice.  This mirrors the fast host-CSR path and avoids redoing the
 * reference dense extraction + segmented sort for every column sub-batch.
 */
template <typename IndptrT = int>
static void ovo_streaming_csr_impl(
    const float* csr_data, const int* csr_indices, const IndptrT* csr_indptr,
    const int* ref_row_ids, const int* grp_row_ids, const int* grp_offsets,
    double* rank_sums, double* tie_corr, int n_ref, int n_all_grp, int n_cols,
    int n_groups, bool compute_tie_corr, int sub_batch_cols) {
    if (n_cols == 0 || n_ref == 0 || n_all_grp == 0) return;

    // Cap sub_batch_cols so the dense group slab (n_all_grp × sub_batch_cols,
    // sorted in one CUB call) stays within int32. n_all_grp is a cell count, so
    // it drives the cap; the reference side is chunked separately below.
    {
        size_t cap = n_all_grp > 0 ? SAFE_BATCH_NNZ / (size_t)n_all_grp
                                   : (size_t)sub_batch_cols;
        if (cap < 1) cap = 1;
        if ((size_t)sub_batch_cols > cap) sub_batch_cols = (int)cap;
    }

    std::vector<int> h_offsets(n_groups + 1);
    cudaMemcpy(h_offsets.data(), grp_offsets, (n_groups + 1) * sizeof(int),
               cudaMemcpyDeviceToHost);
    auto t1 = make_ovo_tier_plan(h_offsets.data(), n_groups);
    int max_grp_size = t1.max_grp_size;
    bool run_large = t1.above_medium && t1.run_large;
    bool run_huge = t1.above_medium && !run_large;
    std::vector<int> h_sort_group_ids;
    int n_sort_groups = n_groups;
    if (run_huge) {
        h_sort_group_ids =
            make_sort_group_ids(h_offsets.data(), n_groups, OVO_MEDIUM_MAX);
        n_sort_groups = (int)h_sort_group_ids.size();
    }

    int n_streams = N_STREAMS;
    if (n_cols < n_streams * sub_batch_cols)
        n_streams = (n_cols + sub_batch_cols - 1) / sub_batch_cols;

    size_t sub_grp_items = (size_t)n_all_grp * sub_batch_cols;

    size_t max_ref_cols = 2147483647LL / (size_t)n_ref;
    if (max_ref_cols == 0) {
        throw std::runtime_error(
            "OVO device CSR reference group exceeds CUB int item limit");
    }
    int ref_cache_cols = std::min(n_cols, (int)max_ref_cols);
    {
        // Reference cache holds 2 floats/col/ref-row; size it to ~a third of
        // what the joint allocator can serve (leaving room for group buffers).
        size_t bytes_per_col = (size_t)n_ref * sizeof(float) * 2;
        size_t target_bytes = rmm_available_device_bytes(1.0 / 3.0);
        if (bytes_per_col > 0 && target_bytes >= bytes_per_col) {
            size_t mem_cols = target_bytes / bytes_per_col;
            if (mem_cols > 0 && mem_cols < (size_t)ref_cache_cols) {
                ref_cache_cols = (int)mem_cols;
            }
        }
    }
    if (ref_cache_cols < 1) ref_cache_cols = 1;

    RmmScratchPool pool;

    size_t cub_temp_bytes = 0;
    if (run_huge) {
        size_t cub_grp_bytes = 0;
        int sub_grp_items_i32 =
            checked_cub_items(sub_grp_items, "OVO device CSR group sub-batch");
        int max_grp_seg =
            checked_int_product((size_t)n_sort_groups, (size_t)sub_batch_cols,
                                "OVO device CSR group segment count");
        cub_grp_bytes =
            cub_segmented_sortkeys_temp_bytes(sub_grp_items_i32, max_grp_seg);
        cub_temp_bytes = cub_grp_bytes;
    }

    ScopedCudaStreams streams(n_streams, cudaStreamDefault);
    ScopedCudaStream ref_stream(cudaStreamNonBlocking);

    int* d_sort_group_ids = nullptr;
    if (run_huge) {
        d_sort_group_ids = pool.alloc<int>(h_sort_group_ids.size());
        cudaMemcpy(d_sort_group_ids, h_sort_group_ids.data(),
                   h_sort_group_ids.size() * sizeof(int),
                   cudaMemcpyHostToDevice);
    }

    struct StreamBuf {
        float* grp_dense;
        float* grp_sorted;
        int* grp_seg_offsets;
        int* grp_seg_ends;
        uint8_t* cub_temp;
        double* ref_tie_sums;
        double* sub_rank_sums;
        double* sub_tie_corr;
    };
    std::vector<StreamBuf> bufs(n_streams);
    for (int s = 0; s < n_streams; s++) {
        bufs[s].grp_dense = pool.alloc<float>(sub_grp_items);
        bufs[s].cub_temp =
            run_huge ? pool.alloc<uint8_t>(cub_temp_bytes) : nullptr;
        bufs[s].ref_tie_sums =
            (compute_tie_corr && (t1.run_warp || t1.run_small || t1.run_medium))
                ? pool.alloc<double>(sub_batch_cols)
                : nullptr;
        bufs[s].sub_rank_sums =
            pool.alloc<double>((size_t)n_groups * sub_batch_cols);
        bufs[s].sub_tie_corr =
            pool.alloc<double>((size_t)n_groups * sub_batch_cols);
        if (run_huge) {
            bufs[s].grp_sorted = pool.alloc<float>(sub_grp_items);
            int max_seg = checked_int_product(
                (size_t)n_sort_groups, (size_t)sub_batch_cols,
                "OVO device CSR group segment buffer");
            bufs[s].grp_seg_offsets = pool.alloc<int>(max_seg);
            bufs[s].grp_seg_ends = pool.alloc<int>(max_seg);
        } else {
            bufs[s].grp_sorted = nullptr;
            bufs[s].grp_seg_offsets = nullptr;
            bufs[s].grp_seg_ends = nullptr;
        }
    }

    int tpb_extract = round_up_to_warp(std::max(n_ref, n_all_grp));
    int tpb_rank =
        round_up_to_warp(std::min(max_grp_size, MAX_THREADS_PER_BLOCK));

    for (int cache_col = 0; cache_col < n_cols; cache_col += ref_cache_cols) {
        int cache_cols = std::min(ref_cache_cols, n_cols - cache_col);
        size_t cache_ref_items = (size_t)n_ref * cache_cols;
        int cache_ref_items_i32 = checked_cub_items(
            cache_ref_items, "OVO device CSR reference cache");

        ScopedCudaBuffer ref_dense_buf(cache_ref_items * sizeof(float));
        ScopedCudaBuffer ref_sorted_buf(cache_ref_items * sizeof(float));
        ScopedCudaBuffer ref_seg_offsets_buf((size_t)(cache_cols + 1) *
                                             sizeof(int));
        float* d_ref_dense = (float*)ref_dense_buf.data();
        float* d_ref_sorted = (float*)ref_sorted_buf.data();
        int* d_ref_seg_offsets = (int*)ref_seg_offsets_buf.data();

        cudaMemsetAsync(d_ref_dense, 0, cache_ref_items * sizeof(float),
                        ref_stream);
        int tpb_ref_extract = round_up_to_warp(n_ref);
        int ref_blk = (n_ref + tpb_ref_extract - 1) / tpb_ref_extract;
        csr_extract_dense_kernel<<<ref_blk, tpb_ref_extract, 0, ref_stream>>>(
            csr_data, csr_indices, csr_indptr, ref_row_ids, d_ref_dense, n_ref,
            cache_col, cache_col + cache_cols);
        CUDA_CHECK_LAST_ERROR(csr_extract_dense_kernel);

        upload_linear_offsets(d_ref_seg_offsets, cache_cols, n_ref, ref_stream);

        size_t ref_cub_bytes =
            cub_segmented_sortkeys_temp_bytes(cache_ref_items_i32, cache_cols);
        ScopedCudaBuffer ref_cub_temp_buf(ref_cub_bytes);
        size_t ref_temp = ref_cub_bytes;
        cuda_check(
            cub::DeviceSegmentedRadixSort::SortKeys(
                ref_cub_temp_buf.data(), ref_temp, d_ref_dense, d_ref_sorted,
                cache_ref_items_i32, cache_cols, d_ref_seg_offsets,
                d_ref_seg_offsets + 1, BEGIN_BIT, END_BIT, ref_stream),
            "device CSR OVO ref segmented sort");
        cuda_check(cudaStreamSynchronize(ref_stream),
                   "device CSR OVO ref sort sync");

        int col = cache_col;
        int cache_stop = cache_col + cache_cols;
        int batch_idx = 0;
        while (col < cache_stop) {
            int sb_cols = std::min(sub_batch_cols, cache_stop - col);
            int sb_grp_items_actual =
                checked_int_product((size_t)n_all_grp, (size_t)sb_cols,
                                    "OVO device CSR active group sub-batch");
            int s = batch_idx % n_streams;
            auto stream = streams[s];
            auto& buf = bufs[s];
            const float* ref_sub =
                d_ref_sorted + (size_t)(col - cache_col) * n_ref;

            cudaMemsetAsync(buf.grp_dense, 0,
                            sb_grp_items_actual * sizeof(float), stream);
            {
                int blk = (n_all_grp + tpb_extract - 1) / tpb_extract;
                csr_extract_dense_kernel<<<blk, tpb_extract, 0, stream>>>(
                    csr_data, csr_indices, csr_indptr, grp_row_ids,
                    buf.grp_dense, n_all_grp, col, col + sb_cols);
                CUDA_CHECK_LAST_ERROR(csr_extract_dense_kernel);
            }

            OvoTierScratch sc{buf.ref_tie_sums,    buf.sub_rank_sums,
                              buf.sub_tie_corr,    buf.grp_sorted,
                              buf.grp_seg_offsets, buf.grp_seg_ends,
                              buf.cub_temp};
            ovo_dispatch_tiers(ref_sub, buf.grp_dense, grp_offsets, t1, sc,
                               d_sort_group_ids, n_sort_groups, cub_temp_bytes,
                               sb_grp_items_actual, tpb_rank, n_ref, n_all_grp,
                               sb_cols, n_groups, compute_tie_corr, stream);

            cudaMemcpy2DAsync(rank_sums + col, n_cols * sizeof(double),
                              buf.sub_rank_sums, sb_cols * sizeof(double),
                              sb_cols * sizeof(double), n_groups,
                              cudaMemcpyDeviceToDevice, stream);
            if (compute_tie_corr) {
                cudaMemcpy2DAsync(tie_corr + col, n_cols * sizeof(double),
                                  buf.sub_tie_corr, sb_cols * sizeof(double),
                                  sb_cols * sizeof(double), n_groups,
                                  cudaMemcpyDeviceToDevice, stream);
            }

            col += sb_cols;
            batch_idx++;
        }

        for (int s = 0; s < n_streams; s++) {
            cudaError_t err = cudaStreamSynchronize(streams[s]);
            if (err != cudaSuccess)
                throw std::runtime_error(
                    std::string("CUDA error in OVO device CSR streaming: ") +
                    cudaGetErrorString(err));
        }
    }
}

/**
 * CSC-direct OVO streaming pipeline.
 *
 * Like the CSR variant, but extracts rows via lookup maps so it can operate on
 * native CSC input without converting the whole matrix.
 */
template <typename IndptrT = int>
static void ovo_streaming_csc_impl(
    const float* csc_data, const int* csc_indices, const IndptrT* csc_indptr,
    const int* ref_row_map, const int* grp_row_map, const int* grp_offsets,
    double* rank_sums, double* tie_corr, int n_ref, int n_all_grp, int n_cols,
    int n_groups, bool compute_tie_corr, int sub_batch_cols) {
    if (n_cols == 0 || n_ref == 0 || n_all_grp == 0) return;

    // Cap sub_batch_cols so both dense slabs (n_ref × sub_batch_cols and
    // n_all_grp × sub_batch_cols, each sorted in one CUB call) stay within
    // int32. These row counts are cell counts, so they drive the cap.
    {
        size_t max_rows = (size_t)std::max(n_ref, n_all_grp);
        size_t cap =
            max_rows > 0 ? SAFE_BATCH_NNZ / max_rows : (size_t)sub_batch_cols;
        if (cap < 1) cap = 1;
        if ((size_t)sub_batch_cols > cap) sub_batch_cols = (int)cap;
    }

    std::vector<int> h_offsets(n_groups + 1);
    cudaMemcpy(h_offsets.data(), grp_offsets, (n_groups + 1) * sizeof(int),
               cudaMemcpyDeviceToHost);
    auto t1 = make_ovo_tier_plan(h_offsets.data(), n_groups);
    int max_grp_size = t1.max_grp_size;
    bool run_large = t1.above_medium && t1.run_large;
    bool run_huge = t1.above_medium && !run_large;
    std::vector<int> h_sort_group_ids;
    int n_sort_groups = n_groups;
    if (run_huge) {
        h_sort_group_ids =
            make_sort_group_ids(h_offsets.data(), n_groups, OVO_MEDIUM_MAX);
        n_sort_groups = (int)h_sort_group_ids.size();
    }

    int n_streams = N_STREAMS;
    if (n_cols < n_streams * sub_batch_cols)
        n_streams = (n_cols + sub_batch_cols - 1) / sub_batch_cols;

    size_t sub_ref_items = (size_t)n_ref * sub_batch_cols;
    size_t sub_grp_items = (size_t)n_all_grp * sub_batch_cols;
    int sub_ref_items_i32 =
        checked_cub_items(sub_ref_items, "OVO device CSC reference sub-batch");
    int sub_grp_items_i32 =
        checked_cub_items(sub_grp_items, "OVO device CSC group sub-batch");

    size_t cub_ref_bytes =
        cub_segmented_sortkeys_temp_bytes(sub_ref_items_i32, sub_batch_cols);
    size_t cub_temp_bytes = cub_ref_bytes;
    if (run_huge) {
        size_t cub_grp_bytes = 0;
        int max_grp_seg =
            checked_int_product((size_t)n_sort_groups, (size_t)sub_batch_cols,
                                "OVO device CSC group segment count");
        cub_grp_bytes =
            cub_segmented_sortkeys_temp_bytes(sub_grp_items_i32, max_grp_seg);
        cub_temp_bytes = std::max(cub_ref_bytes, cub_grp_bytes);
    }

    // pool first: streams drain before it frees their scratch (see guard doc).
    RmmScratchPool pool;
    ScopedCudaStreams streams(n_streams, cudaStreamDefault);
    int* d_sort_group_ids = nullptr;
    if (run_huge) {
        d_sort_group_ids = pool.alloc<int>(h_sort_group_ids.size());
        cudaMemcpy(d_sort_group_ids, h_sort_group_ids.data(),
                   h_sort_group_ids.size() * sizeof(int),
                   cudaMemcpyHostToDevice);
    }

    struct StreamBuf {
        float* ref_dense;
        float* ref_sorted;
        float* grp_dense;
        float* grp_sorted;
        int* ref_seg_offsets;
        int* grp_seg_offsets;
        int* grp_seg_ends;
        uint8_t* cub_temp;
        double* ref_tie_sums;
        double* sub_rank_sums;
        double* sub_tie_corr;
    };
    std::vector<StreamBuf> bufs(n_streams);
    for (int s = 0; s < n_streams; s++) {
        bufs[s].ref_dense = pool.alloc<float>(sub_ref_items);
        bufs[s].ref_sorted = pool.alloc<float>(sub_ref_items);
        bufs[s].grp_dense = pool.alloc<float>(sub_grp_items);
        bufs[s].ref_seg_offsets = pool.alloc<int>(sub_batch_cols + 1);
        bufs[s].cub_temp = pool.alloc<uint8_t>(cub_temp_bytes);
        bufs[s].ref_tie_sums =
            (compute_tie_corr && (t1.run_warp || t1.run_small || t1.run_medium))
                ? pool.alloc<double>(sub_batch_cols)
                : nullptr;
        bufs[s].sub_rank_sums =
            pool.alloc<double>((size_t)n_groups * sub_batch_cols);
        bufs[s].sub_tie_corr =
            pool.alloc<double>((size_t)n_groups * sub_batch_cols);
        if (run_huge) {
            bufs[s].grp_sorted = pool.alloc<float>(sub_grp_items);
            int max_grp_seg = checked_int_product(
                (size_t)n_sort_groups, (size_t)sub_batch_cols,
                "OVO device CSC group segment buffer");
            bufs[s].grp_seg_offsets = pool.alloc<int>(max_grp_seg);
            bufs[s].grp_seg_ends = pool.alloc<int>(max_grp_seg);
        } else {
            bufs[s].grp_sorted = nullptr;
            bufs[s].grp_seg_offsets = nullptr;
            bufs[s].grp_seg_ends = nullptr;
        }
    }

    int tpb_rank =
        round_up_to_warp(std::min(max_grp_size, MAX_THREADS_PER_BLOCK));

    int col = 0;
    int batch_idx = 0;
    while (col < n_cols) {
        int sb_cols = std::min(sub_batch_cols, n_cols - col);
        int sb_ref_items_actual =
            checked_int_product((size_t)n_ref, (size_t)sb_cols,
                                "OVO device CSC active reference sub-batch");
        int sb_grp_items_actual =
            checked_int_product((size_t)n_all_grp, (size_t)sb_cols,
                                "OVO device CSC active group sub-batch");
        int s = batch_idx % n_streams;
        auto stream = streams[s];
        auto& buf = bufs[s];

        cudaMemsetAsync(buf.ref_dense, 0, sb_ref_items_actual * sizeof(float),
                        stream);
        csc_extract_mapped_kernel<<<sb_cols, UTIL_BLOCK_SIZE, 0, stream>>>(
            csc_data, csc_indices, csc_indptr, ref_row_map, buf.ref_dense,
            n_ref, col);
        CUDA_CHECK_LAST_ERROR(csc_extract_mapped_kernel);
        upload_linear_offsets(buf.ref_seg_offsets, sb_cols, n_ref, stream);
        {
            size_t temp = cub_temp_bytes;
            cuda_check(cub::DeviceSegmentedRadixSort::SortKeys(
                           buf.cub_temp, temp, buf.ref_dense, buf.ref_sorted,
                           sb_ref_items_actual, sb_cols, buf.ref_seg_offsets,
                           buf.ref_seg_offsets + 1, BEGIN_BIT, END_BIT, stream),
                       "device CSC OVO ref segmented sort");
        }

        cudaMemsetAsync(buf.grp_dense, 0, sb_grp_items_actual * sizeof(float),
                        stream);
        csc_extract_mapped_kernel<<<sb_cols, UTIL_BLOCK_SIZE, 0, stream>>>(
            csc_data, csc_indices, csc_indptr, grp_row_map, buf.grp_dense,
            n_all_grp, col);
        CUDA_CHECK_LAST_ERROR(csc_extract_mapped_kernel);

        OvoTierScratch sc{buf.ref_tie_sums,    buf.sub_rank_sums,
                          buf.sub_tie_corr,    buf.grp_sorted,
                          buf.grp_seg_offsets, buf.grp_seg_ends,
                          buf.cub_temp};
        ovo_dispatch_tiers(buf.ref_sorted, buf.grp_dense, grp_offsets, t1, sc,
                           d_sort_group_ids, n_sort_groups, cub_temp_bytes,
                           sb_grp_items_actual, tpb_rank, n_ref, n_all_grp,
                           sb_cols, n_groups, compute_tie_corr, stream);

        cudaMemcpy2DAsync(rank_sums + col, n_cols * sizeof(double),
                          buf.sub_rank_sums, sb_cols * sizeof(double),
                          sb_cols * sizeof(double), n_groups,
                          cudaMemcpyDeviceToDevice, stream);
        if (compute_tie_corr) {
            cudaMemcpy2DAsync(tie_corr + col, n_cols * sizeof(double),
                              buf.sub_tie_corr, sb_cols * sizeof(double),
                              sb_cols * sizeof(double), n_groups,
                              cudaMemcpyDeviceToDevice, stream);
        }

        col += sb_cols;
        batch_idx++;
    }

    for (int s = 0; s < n_streams; s++) {
        cudaError_t err = cudaStreamSynchronize(streams[s]);
        if (err != cudaSuccess)
            throw std::runtime_error(
                std::string("CUDA error in OVO device CSC streaming: ") +
                cudaGetErrorString(err));
    }
}
