#pragma once

/**
 * Host-streaming CSC OVO pipeline.
 *
 * CSC arrays live on host.  Only the sparse data for each sub-batch of
 * columns is transferred to GPU.  Row maps + group offsets are uploaded once.
 * Results are written back to host per sub-batch.
 */
template <typename InT, typename IndexT, typename IndptrT>
static void ovo_streaming_csc_host_impl(
    const InT* h_data, const IndexT* h_indices, const IndptrT* h_indptr,
    const int* h_ref_row_map, const int* h_grp_row_map,
    const int* h_grp_offsets, const int* h_stats_codes, double* d_rank_sums,
    double* d_tie_corr, double* d_group_sums, double* d_group_nnz, int n_ref,
    int n_all_grp, int n_rows, int n_cols, int n_groups, int n_groups_stats,
    bool compute_tie_corr, bool compute_nnz, int sub_batch_cols) {
    if (n_cols == 0 || n_ref == 0 || n_all_grp == 0) return;

    // Cap sub_batch_cols so neither the dense ref/group slabs (rows ×
    // sub_batch_cols, sorted in one CUB call) nor the per-column-batch nnz
    // exceed int32. rows here are cell counts, so they dominate the dense cap.
    {
        size_t max_rows = (size_t)std::max(n_ref, n_all_grp);
        size_t dense_cap =
            max_rows > 0 ? SAFE_BATCH_NNZ / max_rows : (size_t)sub_batch_cols;
        if (dense_cap < 1) dense_cap = 1;
        if ((size_t)sub_batch_cols > dense_cap) sub_batch_cols = (int)dense_cap;
        sub_batch_cols = cap_sub_batch_by_nnz(
            n_cols, sub_batch_cols, SAFE_BATCH_NNZ,
            [&](int c) { return (size_t)(h_indptr[c + 1] - h_indptr[c]); });
    }

    auto t1 = make_ovo_tier_plan(h_grp_offsets, n_groups);
    int max_grp_size = t1.max_grp_size;
    bool run_large = t1.above_medium && t1.run_large;
    bool run_huge = t1.above_medium && !run_large;
    std::vector<int> h_sort_group_ids;
    int n_sort_groups = n_groups;
    if (run_huge) {
        h_sort_group_ids =
            make_sort_group_ids(h_grp_offsets, n_groups, OVO_MEDIUM_MAX);
        n_sort_groups = (int)h_sort_group_ids.size();
    }

    int n_streams = clamp_streams_by_cols(n_cols, sub_batch_cols);

    size_t sub_ref_items = (size_t)n_ref * sub_batch_cols;
    size_t sub_grp_items = (size_t)n_all_grp * sub_batch_cols;
    int sub_ref_items_i32 =
        checked_cub_items(sub_ref_items, "OVO host CSC reference sub-batch");
    int sub_grp_items_i32 =
        checked_cub_items(sub_grp_items, "OVO host CSC group sub-batch");

    // CUB temp
    size_t cub_ref_bytes =
        cub_segmented_sortkeys_temp_bytes(sub_ref_items_i32, sub_batch_cols);
    size_t cub_temp_bytes = cub_ref_bytes;
    if (run_huge) {
        int max_grp_seg =
            checked_int_product((size_t)n_sort_groups, (size_t)sub_batch_cols,
                                "OVO host CSC group segment count");
        size_t cub_grp_bytes =
            cub_segmented_sortkeys_temp_bytes(sub_grp_items_i32, max_grp_seg);
        cub_temp_bytes = std::max(cub_ref_bytes, cub_grp_bytes);
    }

    // Max nnz across any sub-batch for sparse transfer buffer sizing
    size_t max_nnz = 0;
    for (int c = 0; c < n_cols; c += sub_batch_cols) {
        int sb = std::min(sub_batch_cols, n_cols - c);
        size_t nnz = (size_t)(h_indptr[c + sb] - h_indptr[c]);
        if (nnz > max_nnz) max_nnz = nnz;
    }

    // Reduce the stream count so the per-stream scratch fits the memory budget.
    // The dense ref/group slabs scale with n_ref/n_all_grp (cell counts), so at
    // scale a fixed N_STREAMS would exceed GPU memory and thrash/OOM.
    {
        size_t per_stream =
            max_nnz * (sizeof(InT) + sizeof(float) + sizeof(IndexT)) +
            2 * sub_ref_items * sizeof(float) +
            (run_huge ? 2 : 1) * sub_grp_items * sizeof(float) +
            2 * (size_t)n_groups * sub_batch_cols * sizeof(double) +
            (compute_nnz ? 2 : 1) * (size_t)n_groups_stats * sub_batch_cols *
                sizeof(double) +
            cub_temp_bytes;
        size_t budget = rmm_available_device_bytes(0.8);
        n_streams = clamp_streams_by_budget(n_streams, per_stream, budget);
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

    int n_batches = (n_cols + sub_batch_cols - 1) / sub_batch_cols;
    int* d_all_offsets = precompute_csc_batch_offsets(
        h_indptr, n_cols, sub_batch_cols, n_batches, pool,
        "OVO host CSC rebased column offsets");

    // Row maps + group offsets + stats codes (uploaded once)
    int* d_ref_row_map = pool.alloc<int>(n_rows);
    int* d_grp_row_map = pool.alloc<int>(n_rows);
    int* d_grp_offsets = pool.alloc<int>(n_groups + 1);
    int* d_stats_codes = pool.alloc<int>(n_rows);
    int* d_sort_group_ids = nullptr;
    cudaMemcpy(d_ref_row_map, h_ref_row_map, n_rows * sizeof(int),
               cudaMemcpyHostToDevice);
    cudaMemcpy(d_grp_row_map, h_grp_row_map, n_rows * sizeof(int),
               cudaMemcpyHostToDevice);
    cudaMemcpy(d_grp_offsets, h_grp_offsets, (n_groups + 1) * sizeof(int),
               cudaMemcpyHostToDevice);
    cudaMemcpy(d_stats_codes, h_stats_codes, n_rows * sizeof(int),
               cudaMemcpyHostToDevice);
    if (run_huge) {
        d_sort_group_ids = pool.alloc<int>(h_sort_group_ids.size());
        cudaMemcpy(d_sort_group_ids, h_sort_group_ids.data(),
                   h_sort_group_ids.size() * sizeof(int),
                   cudaMemcpyHostToDevice);
    }

    struct StreamBuf {
        InT* d_sparse_data_orig;
        float* d_sparse_data_f32;
        IndexT* d_sparse_indices;
        int* d_indptr;
        float* ref_dense;
        float* ref_sorted;
        float* grp_dense;
        float* grp_sorted;
        int* ref_seg_offsets;
        int* grp_seg_offsets;
        int* grp_seg_ends;
        uint8_t* cub_temp;
        double* ref_tie_sums;
        double* d_rank_sums;
        double* d_tie_corr;
        double* d_group_sums;
        double* d_group_nnz;
    };
    std::vector<StreamBuf> bufs(n_streams);
    for (int s = 0; s < n_streams; s++) {
        bufs[s].d_sparse_data_orig = pool.alloc<InT>(max_nnz);
        bufs[s].d_sparse_data_f32 = pool.alloc<float>(max_nnz);
        bufs[s].d_sparse_indices = pool.alloc<IndexT>(max_nnz);
        bufs[s].d_indptr = pool.alloc<int>(sub_batch_cols + 1);
        bufs[s].ref_dense = pool.alloc<float>(sub_ref_items);
        bufs[s].ref_sorted = pool.alloc<float>(sub_ref_items);
        bufs[s].grp_dense = pool.alloc<float>(sub_grp_items);
        bufs[s].ref_seg_offsets = pool.alloc<int>(sub_batch_cols + 1);
        bufs[s].cub_temp = pool.alloc<uint8_t>(cub_temp_bytes);
        // LARGE/HUGE now share the ref tie base too: allocate whenever
        // correcting.
        bufs[s].ref_tie_sums =
            compute_tie_corr ? pool.alloc<double>(sub_batch_cols) : nullptr;
        bufs[s].d_rank_sums =
            pool.alloc<double>((size_t)n_groups * sub_batch_cols);
        bufs[s].d_tie_corr =
            pool.alloc<double>((size_t)n_groups * sub_batch_cols);
        bufs[s].d_group_sums =
            pool.alloc<double>((size_t)n_groups_stats * sub_batch_cols);
        bufs[s].d_group_nnz = pool.alloc<double>(
            compute_nnz ? (size_t)n_groups_stats * sub_batch_cols : 1);
        if (run_huge) {
            bufs[s].grp_sorted = pool.alloc<float>(sub_grp_items);
            int max_grp_seg = checked_int_product(
                (size_t)n_sort_groups, (size_t)sub_batch_cols,
                "OVO host CSC stream group segment count");
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
    bool cast_use_gmem = false;
    size_t smem_cast =
        cast_accumulate_smem_config(n_groups_stats, compute_nnz, cast_use_gmem);

    int col = 0;
    int batch_idx = 0;
    while (col < n_cols) {
        int sb_cols = std::min(sub_batch_cols, n_cols - col);
        int sb_ref_actual =
            checked_int_product((size_t)n_ref, (size_t)sb_cols,
                                "OVO host CSC active reference sub-batch");
        int sb_grp_actual =
            checked_int_product((size_t)n_all_grp, (size_t)sb_cols,
                                "OVO host CSC active group sub-batch");
        int s = batch_idx % n_streams;
        auto stream = streams[s];
        auto& buf = bufs[s];

        // H2D: sparse data for this column range (native dtype)
        IndptrT ptr_start = h_indptr[col];
        IndptrT ptr_end = h_indptr[col + sb_cols];
        size_t nnz = (size_t)(ptr_end - ptr_start);
        checked_int_span(nnz, "OVO host CSC active batch nnz");
        cudaMemcpyAsync(buf.d_sparse_data_orig, h_data + ptr_start,
                        nnz * sizeof(InT), cudaMemcpyHostToDevice, stream);
        cudaMemcpyAsync(buf.d_sparse_indices, h_indices + ptr_start,
                        nnz * sizeof(IndexT), cudaMemcpyHostToDevice, stream);
        int* src = d_all_offsets + (size_t)batch_idx * (sub_batch_cols + 1);
        cudaMemcpyAsync(buf.d_indptr, src, (sb_cols + 1) * sizeof(int),
                        cudaMemcpyDeviceToDevice, stream);

        // Cast to float32 for sort + accumulate stats in float64
        launch_ovr_cast_and_accumulate_sparse<InT, IndexT>(
            buf.d_sparse_data_orig, buf.d_sparse_data_f32, buf.d_sparse_indices,
            buf.d_indptr, d_stats_codes, buf.d_group_sums, buf.d_group_nnz,
            sb_cols, n_groups_stats, compute_nnz, UTIL_BLOCK_SIZE, smem_cast,
            cast_use_gmem, stream);

        // Extract ref from CSC via row_map, sort
        cudaMemsetAsync(buf.ref_dense, 0, sb_ref_actual * sizeof(float),
                        stream);
        csc_extract_mapped_kernel<<<sb_cols, UTIL_BLOCK_SIZE, 0, stream>>>(
            buf.d_sparse_data_f32, buf.d_sparse_indices, buf.d_indptr,
            d_ref_row_map, buf.ref_dense, n_ref, 0);
        CUDA_CHECK_LAST_ERROR(csc_extract_mapped_kernel);
        upload_linear_offsets(buf.ref_seg_offsets, sb_cols, n_ref, stream);
        cub_segmented_sortkeys(buf.cub_temp, cub_temp_bytes, buf.ref_dense,
                               buf.ref_sorted, sb_ref_actual, sb_cols,
                               buf.ref_seg_offsets, buf.ref_seg_offsets + 1,
                               stream, "host CSC OVO ref segmented sort");

        // Extract grp from CSC via row_map
        cudaMemsetAsync(buf.grp_dense, 0, sb_grp_actual * sizeof(float),
                        stream);
        csc_extract_mapped_kernel<<<sb_cols, UTIL_BLOCK_SIZE, 0, stream>>>(
            buf.d_sparse_data_f32, buf.d_sparse_indices, buf.d_indptr,
            d_grp_row_map, buf.grp_dense, n_all_grp, 0);
        CUDA_CHECK_LAST_ERROR(csc_extract_mapped_kernel);

        // Tier dispatch: sort grp + rank
        OvoTierScratch sc{buf.ref_tie_sums,    buf.d_rank_sums,
                          buf.d_tie_corr,      buf.grp_sorted,
                          buf.grp_seg_offsets, buf.grp_seg_ends,
                          buf.cub_temp};
        ovo_dispatch_tiers(buf.ref_sorted, buf.grp_dense, d_grp_offsets, t1, sc,
                           d_sort_group_ids, n_sort_groups, cub_temp_bytes,
                           sb_grp_actual, tpb_rank, n_ref, n_all_grp, sb_cols,
                           n_groups, compute_tie_corr, stream);

        // D2D: scatter sub-batch results into caller's GPU buffers
        scatter_cols_2d(d_rank_sums + col, buf.d_rank_sums, n_groups, n_cols,
                        sb_cols, stream);
        if (compute_tie_corr) {
            scatter_cols_2d(d_tie_corr + col, buf.d_tie_corr, n_groups, n_cols,
                            sb_cols, stream);
        }
        scatter_cols_2d(d_group_sums + col, buf.d_group_sums, n_groups_stats,
                        n_cols, sb_cols, stream);
        if (compute_nnz) {
            scatter_cols_2d(d_group_nnz + col, buf.d_group_nnz, n_groups_stats,
                            n_cols, sb_cols, stream);
        }

        col += sb_cols;
        batch_idx++;
    }

    sync_streams(streams, "wilcoxon streaming");
}

/**
 * Host CSR OVO pipeline — zero-copy mapped full-CSR with GPU-side row gather.
 *
 * Setup: pin the full host CSR with cudaHostRegisterMapped, upload the full
 * indptr (small) + row_ids + pre-computed compacted indptrs.  Each pack
 * gathers only its rows over PCIe via a UVA kernel — the full matrix is never
 * transferred to GPU.
 *
 * Phase 1 (Ref): fused gather + cast + stats over ref rows; segmented sort
 *                to d_ref_sorted (cached for the whole run).
 * Phase 2 (per pack, round-robin across N_STREAMS):
 *   1. rebase per-pack output indptr from the pre-uploaded global compacted
 *      indptr.
 *   2. rebase per-pack group offsets + build per-row stats codes.
 *   3. csr_gather_cast_accumulate_mapped_kernel — one PCIe pass, writes
 *      compacted f32 data + indices and accumulates per-group stats.
 *   4. Per sub-batch: extract dense → sort → rank vs ref_sorted → scatter.
 *
 * Memory: d_ref_sorted (n_ref × n_cols × 4B) + N_STREAMS pack buffers sized
 * for max_pack_rows × sb_cols (dense) and max_pack_nnz (compacted CSR).
 * Full CSR stays on host (pinned-mapped).
 */
template <typename InT, typename IndexT, typename IndptrT>
static void ovo_streaming_csr_host_impl(
    const InT* h_data, const IndexT* h_indices, const IndptrT* h_indptr,
    int n_full_rows, const int* h_ref_row_ids, int n_ref,
    const int* h_grp_row_ids, const int* h_grp_offsets, int n_all_grp,
    int n_test, double* d_rank_sums, double* d_tie_corr, double* d_group_sums,
    double* d_group_nnz, int n_cols, int n_groups_stats, bool compute_tie_corr,
    bool compute_nnz, bool compute_sums, int sub_batch_cols) {
    if (n_cols == 0 || n_ref == 0 || n_test == 0 || n_all_grp == 0) return;

    // Pre-compute compacted indptrs on host (O(n_ref + n_all_grp)).
    // Use IndptrT for the global compacted indptr because the grp side can
    // exceed 2^31 nnz on very large / dense matrices.  Ref always fits in
    // int32 since n_ref × n_cols ≪ 2B; keeping int32 there matches the
    // downstream CUB segmented-sort temp sizing.
    std::vector<int> h_ref_indptr_compact(n_ref + 1);
    h_ref_indptr_compact[0] = 0;
    for (int i = 0; i < n_ref; i++) {
        int r = h_ref_row_ids[i];
        IndptrT row_nnz = h_indptr[r + 1] - h_indptr[r];
        if ((size_t)row_nnz > (size_t)std::numeric_limits<int>::max()) {
            throw std::runtime_error(
                "OVO host CSR reference row exceeds int32 compacted nnz limit");
        }
        int nnz_i = (int)row_nnz;
        if ((size_t)h_ref_indptr_compact[i] + (size_t)nnz_i >
            (size_t)std::numeric_limits<int>::max()) {
            throw std::runtime_error(
                "OVO host CSR reference compacted nnz exceeds int32 limit");
        }
        h_ref_indptr_compact[i + 1] = h_ref_indptr_compact[i] + nnz_i;
    }
    int ref_nnz = h_ref_indptr_compact[n_ref];

    // grp: compacted indptr over concatenated test-group rows (IndptrT).
    std::vector<IndptrT> h_grp_indptr_compact(n_all_grp + 1);
    h_grp_indptr_compact[0] = 0;
    for (int i = 0; i < n_all_grp; i++) {
        int r = h_grp_row_ids[i];
        IndptrT nnz_i = h_indptr[r + 1] - h_indptr[r];
        h_grp_indptr_compact[i + 1] = h_grp_indptr_compact[i] + nnz_i;
    }

    // Build packs (same rule as grp_impl, but uses compacted indptr)
    struct Pack {
        int first;
        int end;
        int n_rows;
        size_t nnz;
        int sb_cols;
    };
    std::vector<Pack> packs;
    int max_pack_rows = 0;
    size_t max_pack_nnz = 0;
    int max_pack_K = 0;
    int max_pack_items = 0;
    int max_pack_sb_cols = sub_batch_cols;
    {
        int target_packs = N_STREAMS;
        int target_rows = (n_all_grp + target_packs - 1) / target_packs;
        if (target_rows < 1) target_rows = 1;
        size_t budget_cap_rows =
            GROUP_DENSE_BUDGET_ITEMS / (size_t)sub_batch_cols;
        if ((size_t)target_rows > budget_cap_rows)
            target_rows = (int)budget_cap_rows;
        // Also bound each pack's compacted nnz: it feeds int32 CUB item counts
        // and int offsets, so a dense pack must stay under INT_MAX. This splits
        // dense perturbation groups across more packs.
        constexpr size_t SAFE_PACK_NNZ = 1500000000;  // < INT_MAX, CUB-safe

        int cur_first = 0;
        int cur_rows = 0;
        size_t cur_nnz = 0;
        for (int g = 0; g < n_test; g++) {
            int n_g = h_grp_offsets[g + 1] - h_grp_offsets[g];
            size_t nnz_g = (size_t)(h_grp_indptr_compact[h_grp_offsets[g + 1]] -
                                    h_grp_indptr_compact[h_grp_offsets[g]]);
            int new_rows = cur_rows + n_g;
            bool can_add =
                (cur_rows == 0) ||
                (new_rows <= target_rows && cur_nnz + nnz_g <= SAFE_PACK_NNZ);
            if (!can_add) {
                size_t sb_size =
                    std::min((size_t)n_cols,
                             GROUP_DENSE_BUDGET_ITEMS / (size_t)cur_rows);
                if (sb_size < (size_t)sub_batch_cols) sb_size = sub_batch_cols;
                packs.push_back(
                    {cur_first, g, cur_rows, cur_nnz, (int)sb_size});
                cur_first = g;
                cur_rows = n_g;
                cur_nnz = nnz_g;
            } else {
                cur_rows = new_rows;
                cur_nnz += nnz_g;
            }
        }
        if (cur_rows > 0) {
            size_t sb_size = std::min(
                (size_t)n_cols, GROUP_DENSE_BUDGET_ITEMS / (size_t)cur_rows);
            if (sb_size < (size_t)sub_batch_cols) sb_size = sub_batch_cols;
            packs.push_back(
                {cur_first, n_test, cur_rows, cur_nnz, (int)sb_size});
        }
    }
    for (const Pack& pk : packs) {
        int K = pk.end - pk.first;
        if (pk.n_rows > max_pack_rows) max_pack_rows = pk.n_rows;
        if (pk.nnz > max_pack_nnz) max_pack_nnz = pk.nnz;
        if (K > max_pack_K) max_pack_K = K;
        int pack_items =
            checked_int_product((size_t)pk.n_rows, (size_t)pk.sb_cols,
                                "OVO host CSR pack dense slab");
        if (pack_items > max_pack_items) max_pack_items = pack_items;
        checked_int_span(pk.nnz, "OVO host CSR pack compacted nnz");
        if (pk.sb_cols > max_pack_sb_cols) max_pack_sb_cols = pk.sb_cols;
    }
    size_t max_sub_items = (size_t)max_pack_items;
    if (max_pack_rows == 0) return;

    RmmScratchPool pool;

    if (compute_sums) {
        cudaMemsetAsync(d_group_sums, 0,
                        (size_t)n_groups_stats * n_cols * sizeof(double));
    }
    if (compute_nnz) {
        cudaMemsetAsync(d_group_nnz, 0,
                        (size_t)n_groups_stats * n_cols * sizeof(double));
    }

    // Pin full host data + indices as MAPPED (zero-copy accessible)
    size_t full_nnz = (size_t)h_indptr[n_full_rows];
    HostRegisterGuard _pin_data(const_cast<InT*>(h_data),
                                full_nnz * sizeof(InT), cudaHostRegisterMapped);
    HostRegisterGuard _pin_indices(const_cast<IndexT*>(h_indices),
                                   full_nnz * sizeof(IndexT),
                                   cudaHostRegisterMapped);

    // Get device-accessible pointers (UVA makes these equal to host ptrs on
    // Linux x86-64, but the API is the safe/portable way).
    InT* d_data_zc = nullptr;
    IndexT* d_indices_zc = nullptr;
    if (full_nnz > 0) {
        cudaError_t e1 = cudaHostGetDevicePointer((void**)&d_data_zc,
                                                  const_cast<InT*>(h_data), 0);
        cudaError_t e2 = cudaHostGetDevicePointer(
            (void**)&d_indices_zc, const_cast<IndexT*>(h_indices), 0);
        if (e1 != cudaSuccess || e2 != cudaSuccess) {
            throw std::runtime_error(
                std::string("cudaHostGetDevicePointer failed: ") +
                cudaGetErrorString(e1 != cudaSuccess ? e1 : e2));
        }
    }

    // Upload full indptr (keep native IndptrT — can exceed int32)
    IndptrT* d_indptr_full = pool.alloc<IndptrT>(n_full_rows + 1);
    cudaMemcpy(d_indptr_full, h_indptr, (n_full_rows + 1) * sizeof(IndptrT),
               cudaMemcpyHostToDevice);

    // Upload row_ids + compacted indptrs + group boundaries
    int* d_ref_row_ids = pool.alloc<int>(n_ref);
    int* d_grp_row_ids = pool.alloc<int>(n_all_grp);
    IndptrT* d_grp_indptr_compact = pool.alloc<IndptrT>(n_all_grp + 1);
    int* d_grp_offsets_full = pool.alloc<int>(n_test + 1);
    cudaMemcpy(d_ref_row_ids, h_ref_row_ids, n_ref * sizeof(int),
               cudaMemcpyHostToDevice);
    cudaMemcpy(d_grp_row_ids, h_grp_row_ids, n_all_grp * sizeof(int),
               cudaMemcpyHostToDevice);
    cudaMemcpy(d_grp_indptr_compact, h_grp_indptr_compact.data(),
               (n_all_grp + 1) * sizeof(IndptrT), cudaMemcpyHostToDevice);
    cudaMemcpy(d_grp_offsets_full, h_grp_offsets, (n_test + 1) * sizeof(int),
               cudaMemcpyHostToDevice);

    // Phase 1: Ref setup (scoped scratch, ref_sorted persists).
    // The full-width sorted reference cache d_ref_sorted is [n_ref × n_cols],
    // but it is built one COLUMN CHUNK at a time so each CUB segmented sort
    // stays within int32 (n_ref × ref_chunk_cols items) and the dense extract
    // scratch is bounded to a chunk instead of the whole [n_ref × n_cols] slab.
    // This is what lets large references (n_ref × n_cols > INT_MAX) work.
    size_t ref_items = (size_t)n_ref * (size_t)n_cols;
    if (ref_items > std::numeric_limits<size_t>::max() / (2 * sizeof(float))) {
        throw std::runtime_error(
            "OVO host CSR dense reference cache size overflows size_t");
    }
    size_t ref_avail = rmm_available_device_bytes(0.9);
    if (ref_avail > 0 && ref_items * sizeof(float) > ref_avail) {
        throw std::runtime_error(
            "OVO host CSR sorted reference cache requires more GPU memory than "
            "is available; use native CSC/device sparse input or reduce "
            "genes/reference size");
    }
    int ref_chunk_cols =
        n_ref > 0
            ? (int)std::min((size_t)n_cols, SAFE_BATCH_NNZ / (size_t)n_ref)
            : n_cols;
    if (ref_chunk_cols < 1) ref_chunk_cols = 1;
    size_t ref_chunk_items = (size_t)n_ref * (size_t)ref_chunk_cols;
    int ref_chunk_items_i32 =
        checked_cub_items(ref_chunk_items, "OVO host CSR ref column chunk");
    float* d_ref_sorted = pool.alloc<float>(ref_items);
    ScopedCudaStream ref_stream(cudaStreamNonBlocking);
    {
        ScopedCudaBuffer ref_data_f32_buf(ref_nnz * sizeof(float));
        ScopedCudaBuffer ref_indices_buf(ref_nnz * sizeof(int));
        ScopedCudaBuffer ref_indptr_buf((n_ref + 1) * sizeof(int));
        ScopedCudaBuffer ref_dense_buf(ref_chunk_items * sizeof(float));
        ScopedCudaBuffer ref_seg_buf((ref_chunk_cols + 1) * sizeof(int));

        float* d_ref_data_f32 = (float*)ref_data_f32_buf.data();
        int* d_ref_indices = (int*)ref_indices_buf.data();
        int* d_ref_indptr = (int*)ref_indptr_buf.data();
        float* d_ref_dense = (float*)ref_dense_buf.data();
        int* d_ref_seg = (int*)ref_seg_buf.data();

        cudaMemcpy(d_ref_indptr, h_ref_indptr_compact.data(),
                   (n_ref + 1) * sizeof(int), cudaMemcpyHostToDevice);

        // Fused gather + cast + stats for ref (fixed slot = n_test): one PCIe
        // pass, no intermediate native-dtype buffer, all-column stats once.
        if (n_ref > 0 && ref_nnz > 0) {
            csr_gather_cast_accumulate_mapped_kernel<InT, IndexT, IndptrT>
                <<<n_ref, UTIL_BLOCK_SIZE, 0, ref_stream>>>(
                    d_data_zc, d_indices_zc, d_indptr_full, d_ref_row_ids,
                    d_ref_indptr, /*d_stats_codes=*/nullptr,
                    /*fixed_slot=*/n_test, d_ref_data_f32, d_ref_indices,
                    d_group_sums, d_group_nnz, n_ref, n_cols, n_groups_stats,
                    compute_sums, compute_nnz);
            CUDA_CHECK_LAST_ERROR(csr_gather_cast_accumulate_mapped_kernel);
        }

        size_t ref_cub_bytes = cub_segmented_sortkeys_temp_bytes(
            ref_chunk_items_i32, ref_chunk_cols);
        ScopedCudaBuffer cub_temp_buf(ref_cub_bytes);

        // Extract + segment-sort the reference one column chunk at a time.
        for (int cs = 0; cs < n_cols; cs += ref_chunk_cols) {
            int ce = std::min(cs + ref_chunk_cols, n_cols);
            int cc = ce - cs;
            size_t chunk_items = (size_t)n_ref * (size_t)cc;
            cudaMemsetAsync(d_ref_dense, 0, chunk_items * sizeof(float),
                            ref_stream);
            csr_extract_dense_identity_rows_unsorted_kernel<float>
                <<<n_ref, UTIL_BLOCK_SIZE, 0, ref_stream>>>(
                    d_ref_data_f32, d_ref_indices, d_ref_indptr, d_ref_dense,
                    n_ref, cs, ce);
            CUDA_CHECK_LAST_ERROR(
                csr_extract_dense_identity_rows_unsorted_kernel);
            upload_linear_offsets(d_ref_seg, cc, n_ref, ref_stream);
            cub_segmented_sortkeys(
                cub_temp_buf.data(), ref_cub_bytes, d_ref_dense,
                d_ref_sorted + (size_t)cs * (size_t)n_ref, (int)chunk_items, cc,
                d_ref_seg, d_ref_seg + 1, ref_stream,
                "host CSR OVO ref segmented sort");
        }
        cuda_check(cudaStreamSynchronize(ref_stream),
                   "host CSR OVO ref sort sync");
    }  // ref scratch drops here

    // Phase 2: Per-pack streaming
    auto t1 = make_ovo_tier_plan(h_grp_offsets, n_test);
    bool may_need_cub = (t1.max_grp_size > OVO_LARGE_MAX);

    constexpr int MAX_GROUP_STREAMS = 4;
    int n_streams = MAX_GROUP_STREAMS;
    if (n_test < n_streams) n_streams = n_test;
    if (n_streams < 1) n_streams = 1;
    if ((int)packs.size() < n_streams) n_streams = (int)packs.size();
    if (n_streams < 1) n_streams = 1;

    size_t cub_grp_bytes = 0;
    if (may_need_cub && max_sub_items > 0) {
        int max_sub_items_i32 =
            checked_cub_items(max_sub_items, "OVO host CSR group pack");
        int max_segments =
            checked_int_product((size_t)max_pack_K, (size_t)max_pack_sb_cols,
                                "OVO host CSR max group segment count");
        cub_grp_bytes =
            cub_segmented_sortkeys_temp_bytes(max_sub_items_i32, max_segments);
    }

    ScopedCudaStreams streams(n_streams, cudaStreamDefault);

    struct StreamBuf {
        float* d_grp_data_f32;
        int* d_grp_indices;
        int* d_grp_indptr;
        int* d_pack_grp_offsets;
        int* d_pack_stats_codes;
        float* d_grp_dense;
        float* d_grp_sorted;
        double* d_ref_tie_sums;
        int* d_sort_group_ids;
        int* d_grp_seg_offsets;
        int* d_grp_seg_ends;
        uint8_t* cub_temp;
        double* d_rank_sums;
        double* d_tie_corr;
    };
    std::vector<StreamBuf> bufs(n_streams);
    int max_pack_kernel_seg =
        checked_int_product((size_t)max_pack_K, (size_t)max_pack_sb_cols,
                            "OVO host CSR pack segment buffer");
    for (int s = 0; s < n_streams; s++) {
        bufs[s].d_grp_data_f32 = pool.alloc<float>(max_pack_nnz);
        bufs[s].d_grp_indices = pool.alloc<int>(max_pack_nnz);
        bufs[s].d_grp_indptr = pool.alloc<int>(max_pack_rows + 1);
        bufs[s].d_pack_grp_offsets = pool.alloc<int>(max_pack_K + 1);
        bufs[s].d_pack_stats_codes = pool.alloc<int>(max_pack_rows);
        bufs[s].d_grp_dense = pool.alloc<float>(max_sub_items);
        bufs[s].d_ref_tie_sums = pool.alloc<double>(max_pack_sb_cols);
        bufs[s].d_rank_sums =
            pool.alloc<double>((size_t)max_pack_K * max_pack_sb_cols);
        bufs[s].d_tie_corr =
            pool.alloc<double>((size_t)max_pack_K * max_pack_sb_cols);
        if (may_need_cub) {
            bufs[s].d_grp_sorted = pool.alloc<float>(max_sub_items);
            bufs[s].d_sort_group_ids = pool.alloc<int>(max_pack_K);
            bufs[s].d_grp_seg_offsets = pool.alloc<int>(max_pack_kernel_seg);
            bufs[s].d_grp_seg_ends = pool.alloc<int>(max_pack_kernel_seg);
            bufs[s].cub_temp = pool.alloc<uint8_t>(cub_grp_bytes);
        } else {
            bufs[s].d_grp_sorted = nullptr;
            bufs[s].d_sort_group_ids = nullptr;
            bufs[s].d_grp_seg_offsets = nullptr;
            bufs[s].d_grp_seg_ends = nullptr;
            bufs[s].cub_temp = nullptr;
        }
    }

    for (int p = 0; p < (int)packs.size(); p++) {
        const Pack& pack = packs[p];
        int K = pack.end - pack.first;
        if (K == 0 || pack.n_rows == 0) continue;
        OvoTierPlan pack_t1 = make_ovo_tier_plan(h_grp_offsets + pack.first, K);
        int pack_tpb_rank = round_up_to_warp(
            std::min(pack_t1.max_grp_size, MAX_THREADS_PER_BLOCK));
        // HUGE skips groups MEDIUM already handled (≤ OVO_MEDIUM_MAX).
        int pack_huge_skip_le = OVO_MEDIUM_MAX;
        std::vector<int> h_sort_group_ids;
        int pack_n_sort_groups = K;
        if (pack_t1.above_medium && !pack_t1.run_large) {
            h_sort_group_ids = make_sort_group_ids(h_grp_offsets + pack.first,
                                                   K, pack_huge_skip_le);
            pack_n_sort_groups = (int)h_sort_group_ids.size();
        }

        int s = p % n_streams;
        cudaStream_t stream = streams[s];
        auto& buf = bufs[s];

        if (pack_t1.above_medium && !pack_t1.run_large) {
            cudaMemcpyAsync(buf.d_sort_group_ids, h_sort_group_ids.data(),
                            h_sort_group_ids.size() * sizeof(int),
                            cudaMemcpyHostToDevice, stream);
        }

        int row_start = h_grp_offsets[pack.first];
        int pack_rows = pack.n_rows;
        int pack_sb = pack.sb_cols;

        // Rebase pack's output indptr from pre-uploaded global compacted indptr
        // (IndptrT → int32: pack nnz is bounded by GROUP_DENSE_BUDGET so fits).
        {
            int count = pack_rows + 1;
            int blk = (count + UTIL_BLOCK_SIZE - 1) / UTIL_BLOCK_SIZE;
            rebase_indptr_kernel<IndptrT, int>
                <<<blk, UTIL_BLOCK_SIZE, 0, stream>>>(
                    d_grp_indptr_compact, buf.d_grp_indptr, row_start, count);
            CUDA_CHECK_LAST_ERROR(rebase_indptr_kernel);
        }

        // Build per-pack group offsets on GPU — needed for stats codes before
        // the fused gather kernel can run.
        {
            int count = K + 1;
            int blk = (count + UTIL_BLOCK_SIZE - 1) / UTIL_BLOCK_SIZE;
            rebase_indptr_kernel<int, int><<<blk, UTIL_BLOCK_SIZE, 0, stream>>>(
                d_grp_offsets_full, buf.d_pack_grp_offsets, pack.first, count);
            CUDA_CHECK_LAST_ERROR(rebase_indptr_kernel);
        }

        {
            int blk = (pack_rows + UTIL_BLOCK_SIZE - 1) / UTIL_BLOCK_SIZE;
            fill_pack_stats_codes_kernel<<<blk, UTIL_BLOCK_SIZE, 0, stream>>>(
                buf.d_pack_grp_offsets, buf.d_pack_stats_codes, K, pack.first);
            CUDA_CHECK_LAST_ERROR(fill_pack_stats_codes_kernel);
        }

        // Fused gather + cast + stats for the pack: one PCIe pass (reads mapped
        // host via UVA), no intermediate native-dtype buffer.
        if (pack.nnz > 0) {
            csr_gather_cast_accumulate_mapped_kernel<InT, IndexT, IndptrT>
                <<<pack_rows, UTIL_BLOCK_SIZE, 0, stream>>>(
                    d_data_zc, d_indices_zc, d_indptr_full,
                    d_grp_row_ids + row_start, buf.d_grp_indptr,
                    buf.d_pack_stats_codes, /*fixed_slot=*/-1,
                    buf.d_grp_data_f32, buf.d_grp_indices, d_group_sums,
                    d_group_nnz, pack_rows, n_cols, n_groups_stats,
                    compute_sums, compute_nnz);
            CUDA_CHECK_LAST_ERROR(csr_gather_cast_accumulate_mapped_kernel);
        }

        int col = 0;
        while (col < n_cols) {
            int sb_cols = std::min(pack_sb, n_cols - col);
            int sb_items =
                checked_int_product((size_t)pack_rows, (size_t)sb_cols,
                                    "OVO host CSR active group sub-batch");

            cudaMemsetAsync(buf.d_grp_dense, 0, sb_items * sizeof(float),
                            stream);
            csr_extract_dense_identity_rows_unsorted_kernel<float>
                <<<pack_rows, UTIL_BLOCK_SIZE, 0, stream>>>(
                    buf.d_grp_data_f32, buf.d_grp_indices, buf.d_grp_indptr,
                    buf.d_grp_dense, pack_rows, col, col + sb_cols);
            CUDA_CHECK_LAST_ERROR(
                csr_extract_dense_identity_rows_unsorted_kernel);

            const float* ref_sub = d_ref_sorted + (size_t)col * n_ref;

            OvoTierScratch sc{buf.d_ref_tie_sums,    buf.d_rank_sums,
                              buf.d_tie_corr,        buf.d_grp_sorted,
                              buf.d_grp_seg_offsets, buf.d_grp_seg_ends,
                              buf.cub_temp};
            ovo_dispatch_tiers(ref_sub, buf.d_grp_dense, buf.d_pack_grp_offsets,
                               pack_t1, sc, buf.d_sort_group_ids,
                               pack_n_sort_groups, cub_grp_bytes, sb_items,
                               pack_tpb_rank, n_ref, pack_rows, sb_cols, K,
                               compute_tie_corr, stream);

            scatter_cols_2d(d_rank_sums + (size_t)pack.first * n_cols + col,
                            buf.d_rank_sums, K, n_cols, sb_cols, stream);
            if (compute_tie_corr) {
                scatter_cols_2d(d_tie_corr + (size_t)pack.first * n_cols + col,
                                buf.d_tie_corr, K, n_cols, sb_cols, stream);
            }

            col += sb_cols;
        }
    }

    sync_streams(streams, "ovo csr host streaming");
}
