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

    std::vector<int> h_offsets(n_groups + 1);
    cudaMemcpy(h_offsets.data(), grp_offsets, (n_groups + 1) * sizeof(int),
               cudaMemcpyDeviceToHost);
    auto t1 = make_tier1_config(h_offsets.data(), n_groups);
    int max_grp_size = t1.max_grp_size;
    bool use_tier1 = t1.any_above_t2 && t1.use_tier1;
    bool needs_tier3 = t1.any_above_t2 && !use_tier1;
    int padded_grp_size = t1.padded_grp_size;
    int tier1_tpb = t1.tier1_tpb;
    size_t tier1_smem = t1.tier1_smem;
    std::vector<int> h_sort_group_ids;
    int n_sort_groups = n_groups;
    if (needs_tier3) {
        h_sort_group_ids = make_sort_group_ids(h_offsets.data(), n_groups,
                                               TIER2_GROUP_THRESHOLD);
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
    size_t free_bytes = 0;
    size_t total_bytes = 0;
    if (cudaMemGetInfo(&free_bytes, &total_bytes) == cudaSuccess) {
        size_t bytes_per_col = (size_t)n_ref * sizeof(float) * 2;
        size_t target_bytes = free_bytes / 3;
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
    if (needs_tier3) {
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

    std::vector<cudaStream_t> streams(n_streams);
    for (int i = 0; i < n_streams; i++) cudaStreamCreate(&streams[i]);
    cudaStream_t ref_stream;
    cudaStreamCreateWithFlags(&ref_stream, cudaStreamNonBlocking);

    int* d_sort_group_ids = nullptr;
    if (needs_tier3) {
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
            needs_tier3 ? pool.alloc<uint8_t>(cub_temp_bytes) : nullptr;
        bufs[s].ref_tie_sums =
            (compute_tie_corr &&
             (t1.use_tier0 || t1.any_tier0_64 || t1.any_tier2))
                ? pool.alloc<double>(sub_batch_cols)
                : nullptr;
        bufs[s].sub_rank_sums =
            pool.alloc<double>((size_t)n_groups * sub_batch_cols);
        bufs[s].sub_tie_corr =
            pool.alloc<double>((size_t)n_groups * sub_batch_cols);
        if (needs_tier3) {
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
        cub::DeviceSegmentedRadixSort::SortKeys(
            ref_cub_temp_buf.data(), ref_temp, d_ref_dense, d_ref_sorted,
            cache_ref_items_i32, cache_cols, d_ref_seg_offsets,
            d_ref_seg_offsets + 1, BEGIN_BIT, END_BIT, ref_stream);
        cudaStreamSynchronize(ref_stream);

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

            int skip_le = 0;
            bool run_tier0 = t1.use_tier0;
            bool run_tier0_64 = t1.any_tier0_64;
            bool run_tier2 = t1.any_tier2;
            if (compute_tie_corr && (run_tier0 || run_tier0_64 || run_tier2)) {
                launch_ref_tie_sums(ref_sub, buf.ref_tie_sums, n_ref, sb_cols,
                                    stream);
            }
            if (run_tier0) {
                launch_tier0(ref_sub, buf.grp_dense, grp_offsets,
                             buf.ref_tie_sums, buf.sub_rank_sums,
                             buf.sub_tie_corr, n_ref, n_all_grp, sb_cols,
                             n_groups, compute_tie_corr, stream);
                if (t1.any_above_t0) skip_le = TIER0_GROUP_THRESHOLD;
            }
            if (run_tier0_64) {
                launch_tier0_64(ref_sub, buf.grp_dense, grp_offsets,
                                buf.ref_tie_sums, buf.sub_rank_sums,
                                buf.sub_tie_corr, n_ref, n_all_grp, sb_cols,
                                n_groups, compute_tie_corr, skip_le, stream);
                if (t1.max_grp_size > TIER0_64_GROUP_THRESHOLD) {
                    skip_le = TIER0_64_GROUP_THRESHOLD;
                }
            }
            if (run_tier2) {
                launch_tier2_medium(
                    ref_sub, buf.grp_dense, grp_offsets, buf.ref_tie_sums,
                    buf.sub_rank_sums, buf.sub_tie_corr, n_ref, n_all_grp,
                    sb_cols, n_groups, compute_tie_corr, skip_le, stream);
            }

            int upper_skip_le =
                t1.any_above_t2 ? TIER2_GROUP_THRESHOLD : skip_le;
            if (t1.any_above_t2 && use_tier1) {
                dim3 grid(sb_cols, n_groups);
                ovo_fused_sort_rank_kernel<<<grid, tier1_tpb, tier1_smem,
                                             stream>>>(
                    ref_sub, buf.grp_dense, grp_offsets, buf.sub_rank_sums,
                    buf.sub_tie_corr, n_ref, n_all_grp, sb_cols, n_groups,
                    compute_tie_corr, padded_grp_size, upper_skip_le);
                CUDA_CHECK_LAST_ERROR(ovo_fused_sort_rank_kernel);
            } else if (needs_tier3) {
                int sb_grp_seg = checked_int_product(
                    (size_t)n_sort_groups, (size_t)sb_cols,
                    "OVO device CSR active group segment count");
                {
                    int blk =
                        (sb_grp_seg + UTIL_BLOCK_SIZE - 1) / UTIL_BLOCK_SIZE;
                    build_tier3_seg_begin_end_offsets_kernel<<<
                        blk, UTIL_BLOCK_SIZE, 0, stream>>>(
                        grp_offsets, d_sort_group_ids, buf.grp_seg_offsets,
                        buf.grp_seg_ends, n_all_grp, n_sort_groups, sb_cols);
                    CUDA_CHECK_LAST_ERROR(
                        build_tier3_seg_begin_end_offsets_kernel);
                }
                {
                    size_t temp = cub_temp_bytes;
                    cub::DeviceSegmentedRadixSort::SortKeys(
                        buf.cub_temp, temp, buf.grp_dense, buf.grp_sorted,
                        sb_grp_items_actual, sb_grp_seg, buf.grp_seg_offsets,
                        buf.grp_seg_ends, BEGIN_BIT, END_BIT, stream);
                }
                {
                    dim3 grid(sb_cols, n_groups);
                    batched_rank_sums_presorted_kernel<<<grid, tpb_rank, 0,
                                                         stream>>>(
                        ref_sub, buf.grp_sorted, grp_offsets, buf.sub_rank_sums,
                        buf.sub_tie_corr, n_ref, n_all_grp, sb_cols, n_groups,
                        compute_tie_corr, upper_skip_le);
                    CUDA_CHECK_LAST_ERROR(batched_rank_sums_presorted_kernel);
                }
            }

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
    for (int s = 0; s < n_streams; s++) cudaStreamDestroy(streams[s]);
    cudaStreamDestroy(ref_stream);
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

    std::vector<int> h_offsets(n_groups + 1);
    cudaMemcpy(h_offsets.data(), grp_offsets, (n_groups + 1) * sizeof(int),
               cudaMemcpyDeviceToHost);
    auto t1 = make_tier1_config(h_offsets.data(), n_groups);
    int max_grp_size = t1.max_grp_size;
    bool use_tier1 = t1.any_above_t2 && t1.use_tier1;
    bool needs_tier3 = t1.any_above_t2 && !use_tier1;
    int padded_grp_size = t1.padded_grp_size;
    int tier1_tpb = t1.tier1_tpb;
    size_t tier1_smem = t1.tier1_smem;
    std::vector<int> h_sort_group_ids;
    int n_sort_groups = n_groups;
    if (needs_tier3) {
        h_sort_group_ids = make_sort_group_ids(h_offsets.data(), n_groups,
                                               TIER2_GROUP_THRESHOLD);
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
    if (needs_tier3) {
        size_t cub_grp_bytes = 0;
        int max_grp_seg =
            checked_int_product((size_t)n_sort_groups, (size_t)sub_batch_cols,
                                "OVO device CSC group segment count");
        cub_grp_bytes =
            cub_segmented_sortkeys_temp_bytes(sub_grp_items_i32, max_grp_seg);
        cub_temp_bytes = std::max(cub_ref_bytes, cub_grp_bytes);
    }

    std::vector<cudaStream_t> streams(n_streams);
    for (int i = 0; i < n_streams; i++) cudaStreamCreate(&streams[i]);

    RmmScratchPool pool;
    int* d_sort_group_ids = nullptr;
    if (needs_tier3) {
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
            (compute_tie_corr &&
             (t1.use_tier0 || t1.any_tier0_64 || t1.any_tier2))
                ? pool.alloc<double>(sub_batch_cols)
                : nullptr;
        bufs[s].sub_rank_sums =
            pool.alloc<double>((size_t)n_groups * sub_batch_cols);
        bufs[s].sub_tie_corr =
            pool.alloc<double>((size_t)n_groups * sub_batch_cols);
        if (needs_tier3) {
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
            cub::DeviceSegmentedRadixSort::SortKeys(
                buf.cub_temp, temp, buf.ref_dense, buf.ref_sorted,
                sb_ref_items_actual, sb_cols, buf.ref_seg_offsets,
                buf.ref_seg_offsets + 1, BEGIN_BIT, END_BIT, stream);
        }

        cudaMemsetAsync(buf.grp_dense, 0, sb_grp_items_actual * sizeof(float),
                        stream);
        csc_extract_mapped_kernel<<<sb_cols, UTIL_BLOCK_SIZE, 0, stream>>>(
            csc_data, csc_indices, csc_indptr, grp_row_map, buf.grp_dense,
            n_all_grp, col);
        CUDA_CHECK_LAST_ERROR(csc_extract_mapped_kernel);

        int skip_le = 0;
        bool run_tier0 = t1.use_tier0;
        bool run_tier0_64 = t1.any_tier0_64;
        bool run_tier2 = t1.any_tier2;
        if (compute_tie_corr && (run_tier0 || run_tier0_64 || run_tier2)) {
            launch_ref_tie_sums(buf.ref_sorted, buf.ref_tie_sums, n_ref,
                                sb_cols, stream);
        }
        if (run_tier0) {
            launch_tier0(buf.ref_sorted, buf.grp_dense, grp_offsets,
                         buf.ref_tie_sums, buf.sub_rank_sums, buf.sub_tie_corr,
                         n_ref, n_all_grp, sb_cols, n_groups, compute_tie_corr,
                         stream);
            if (t1.any_above_t0) skip_le = TIER0_GROUP_THRESHOLD;
        }
        if (run_tier0_64) {
            launch_tier0_64(buf.ref_sorted, buf.grp_dense, grp_offsets,
                            buf.ref_tie_sums, buf.sub_rank_sums,
                            buf.sub_tie_corr, n_ref, n_all_grp, sb_cols,
                            n_groups, compute_tie_corr, skip_le, stream);
            if (t1.max_grp_size > TIER0_64_GROUP_THRESHOLD) {
                skip_le = TIER0_64_GROUP_THRESHOLD;
            }
        }
        if (run_tier2) {
            launch_tier2_medium(buf.ref_sorted, buf.grp_dense, grp_offsets,
                                buf.ref_tie_sums, buf.sub_rank_sums,
                                buf.sub_tie_corr, n_ref, n_all_grp, sb_cols,
                                n_groups, compute_tie_corr, skip_le, stream);
        }

        int upper_skip_le = t1.any_above_t2 ? TIER2_GROUP_THRESHOLD : skip_le;
        if (t1.any_above_t2 && use_tier1) {
            dim3 grid(sb_cols, n_groups);
            ovo_fused_sort_rank_kernel<<<grid, tier1_tpb, tier1_smem, stream>>>(
                buf.ref_sorted, buf.grp_dense, grp_offsets, buf.sub_rank_sums,
                buf.sub_tie_corr, n_ref, n_all_grp, sb_cols, n_groups,
                compute_tie_corr, padded_grp_size, upper_skip_le);
            CUDA_CHECK_LAST_ERROR(ovo_fused_sort_rank_kernel);
        } else if (needs_tier3) {
            int sb_grp_seg = checked_int_product(
                (size_t)n_sort_groups, (size_t)sb_cols,
                "OVO device CSC active group segment count");
            {
                int blk = (sb_grp_seg + UTIL_BLOCK_SIZE - 1) / UTIL_BLOCK_SIZE;
                build_tier3_seg_begin_end_offsets_kernel<<<blk, UTIL_BLOCK_SIZE,
                                                           0, stream>>>(
                    grp_offsets, d_sort_group_ids, buf.grp_seg_offsets,
                    buf.grp_seg_ends, n_all_grp, n_sort_groups, sb_cols);
                CUDA_CHECK_LAST_ERROR(build_tier3_seg_begin_end_offsets_kernel);
            }
            {
                size_t temp = cub_temp_bytes;
                cub::DeviceSegmentedRadixSort::SortKeys(
                    buf.cub_temp, temp, buf.grp_dense, buf.grp_sorted,
                    sb_grp_items_actual, sb_grp_seg, buf.grp_seg_offsets,
                    buf.grp_seg_ends, BEGIN_BIT, END_BIT, stream);
            }
            {
                dim3 grid(sb_cols, n_groups);
                batched_rank_sums_presorted_kernel<<<grid, tpb_rank, 0,
                                                     stream>>>(
                    buf.ref_sorted, buf.grp_sorted, grp_offsets,
                    buf.sub_rank_sums, buf.sub_tie_corr, n_ref, n_all_grp,
                    sb_cols, n_groups, compute_tie_corr, upper_skip_le);
                CUDA_CHECK_LAST_ERROR(batched_rank_sums_presorted_kernel);
            }
        }

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
    for (int s = 0; s < n_streams; s++) cudaStreamDestroy(streams[s]);
}
