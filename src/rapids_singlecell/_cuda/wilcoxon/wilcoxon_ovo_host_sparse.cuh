#pragma once

#include <cstdint>

struct OvoHostCsrPack {
    int first;
    int end;
    int n_rows;
    size_t nnz;
    int sb_cols;
};

struct OvoHostCsrPackPlan {
    std::vector<OvoHostCsrPack> packs;
    int max_pack_rows = 0;
    size_t max_pack_nnz = 0;
    int max_pack_K = 0;
    int max_pack_sb_cols = 0;
    size_t max_sub_items = 0;
};

constexpr int OVO_HOST_CSR_U16_COLUMN_CAPACITY =
    (int)std::numeric_limits<uint16_t>::max() + 1;
constexpr int OVO_HOST_CSR_REF_STAGE_RING_SLOTS = 2;

template <typename IndptrT>
static OvoHostCsrPackPlan plan_ovo_host_csr_packs(
    const int* h_grp_offsets, const IndptrT* h_grp_indptr_compact,
    int n_all_grp, int n_test, int n_cols, int n_ref, int sub_batch_cols) {
    OvoHostCsrPackPlan plan;
    plan.max_pack_sb_cols = sub_batch_cols;

    int target_packs = N_STREAMS;
    int target_rows = (n_all_grp + target_packs - 1) / target_packs;
    if (target_rows < 1) target_rows = 1;
    size_t budget_cap_rows = GROUP_DENSE_BUDGET_ITEMS / (size_t)sub_batch_cols;
    if ((size_t)target_rows > budget_cap_rows)
        target_rows = (int)budget_cap_rows;

    constexpr size_t SAFE_PACK_NNZ = 1500000000;  // < INT_MAX, CUB-safe
    size_t pack_nnz_cap = SAFE_PACK_NNZ;
    {
        int target_streams = std::min(N_STREAMS, n_test);
        if (target_streams < 1) target_streams = 1;
        size_t dev_budget = rmm_available_device_bytes(0.9);
        size_t ref_bytes = (size_t)n_ref * (size_t)n_cols * sizeof(float);
        size_t reserve = (size_t)target_streams * OVO_PACK_FIXED_PER_STREAM;
        size_t grp_avail = dev_budget > ref_bytes ? dev_budget - ref_bytes : 0;
        size_t data_avail = grp_avail > reserve ? grp_avail - reserve : 0;
        size_t cap = data_avail / ((size_t)target_streams * 2 * sizeof(float));
        if (cap < OVO_MIN_PACK_NNZ) cap = OVO_MIN_PACK_NNZ;
        if (cap < pack_nnz_cap) pack_nnz_cap = cap;
    }

    int cur_first = 0;
    int cur_rows = 0;
    size_t cur_nnz = 0;
    for (int g = 0; g < n_test; g++) {
        int n_g = h_grp_offsets[g + 1] - h_grp_offsets[g];
        size_t nnz_g = (size_t)(h_grp_indptr_compact[h_grp_offsets[g + 1]] -
                                h_grp_indptr_compact[h_grp_offsets[g]]);
        int new_rows = cur_rows + n_g;
        bool can_add =
            (n_g == 0) || (cur_rows == 0) ||
            (new_rows <= target_rows && cur_nnz + nnz_g <= pack_nnz_cap);
        if (!can_add) {
            size_t sb_size = std::min(
                (size_t)n_cols, GROUP_DENSE_BUDGET_ITEMS / (size_t)cur_rows);
            if (sb_size < (size_t)sub_batch_cols) sb_size = sub_batch_cols;
            plan.packs.push_back(
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
        size_t sb_size = std::min((size_t)n_cols,
                                  GROUP_DENSE_BUDGET_ITEMS / (size_t)cur_rows);
        if (sb_size < (size_t)sub_batch_cols) sb_size = sub_batch_cols;
        plan.packs.push_back(
            {cur_first, n_test, cur_rows, cur_nnz, (int)sb_size});
    }

    for (const OvoHostCsrPack& pk : plan.packs) {
        int K = pk.end - pk.first;
        if (pk.n_rows > plan.max_pack_rows) plan.max_pack_rows = pk.n_rows;
        if (pk.nnz > plan.max_pack_nnz) plan.max_pack_nnz = pk.nnz;
        if (K > plan.max_pack_K) plan.max_pack_K = K;
        size_t pack_items =
            (size_t)checked_int_product((size_t)pk.n_rows, (size_t)pk.sb_cols,
                                        "OVO host CSR pack dense slab");
        if (pack_items > plan.max_sub_items) plan.max_sub_items = pack_items;
        checked_int_span(pk.nnz, "OVO host CSR pack compacted nnz");
        if (pk.sb_cols > plan.max_pack_sb_cols)
            plan.max_pack_sb_cols = pk.sb_cols;
    }
    return plan;
}

/** Host-streaming CSC OVO: send only each column sub-batch to GPU.
 *  Row maps/group offsets upload once; results scatter per sub-batch. */
template <typename InT, typename IndexT, typename IndptrT>
static void ovo_streaming_csc_host_impl(
    const InT* h_data, const IndexT* h_indices, const IndptrT* h_indptr,
    const int* h_ref_row_map, const int* h_grp_row_map,
    const int* h_grp_offsets, const int* h_stats_codes, double* d_rank_sums,
    double* d_tie_corr, double* d_group_sums, double* d_group_nnz, int n_ref,
    int n_all_grp, int n_rows, int n_cols, int n_groups, int n_groups_stats,
    bool compute_tie_corr, bool compute_nnz, int sub_batch_cols,
    bool analytic_zeros) {
    if (n_cols == 0 || n_ref == 0 || n_all_grp == 0) return;

    // Cap sub_batch_cols so neither the dense ref/group slabs (rows ×
    // sub_batch_cols, one CUB call) nor per-batch nnz exceed int32.
    DenseColumnBatchPlan dense_batches = plan_dense_column_batches(
        std::max(n_ref, n_all_grp), n_cols, sub_batch_cols, SAFE_BATCH_NNZ,
        "OVO host CSC dense sub-batch");
    sub_batch_cols = dense_batches.sub_batch_cols;
    size_t sparse_cap = SAFE_BATCH_NNZ;
    ColumnBatchPlan batches =
        plan_csc_column_batches(h_indptr, n_cols, sub_batch_cols, sparse_cap,
                                "OVO host CSC rebased column offsets");
    sub_batch_cols = batches.sub_batch_cols;

    auto t1 = make_ovo_tier_plan(h_grp_offsets, n_groups);
    int max_grp_size = t1.max_grp_size;
    bool run_huge = compute_tie_corr && t1.run_huge;
    std::vector<int> h_sort_group_ids;
    int n_sort_groups = n_groups;
    if (run_huge) {
        h_sort_group_ids =
            make_sort_group_ids(h_grp_offsets, n_groups, t1.huge_skip_le);
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

    size_t max_nnz = batches.max_nnz;
    constexpr size_t window_value_bytes =
        sizeof(WilcoxonSparseWindowDTypes::value_type);
    size_t group_slab_count = run_huge ? 2 : 1;
    if (run_huge && analytic_zeros) ++group_slab_count;

    // Clamp streams so per-stream scratch fits the budget: dense slabs scale
    // with cell counts, so a fixed N_STREAMS would OOM at scale.
    {
        size_t per_stream =
            sparse_window_nnz_bytes<WilcoxonSparseWindowDTypes>(max_nnz) +
            2 * sub_ref_items * window_value_bytes +
            group_slab_count * sub_grp_items * window_value_bytes +
            sparse_window_accum_bytes<WilcoxonSparseWindowDTypes>(
                (size_t)(1 + compute_tie_corr) * n_groups * sub_batch_cols) +
            (compute_nnz ? 2 : 1) *
                sparse_window_accum_bytes<WilcoxonSparseWindowDTypes>(
                    (size_t)n_groups_stats * sub_batch_cols) +
            cub_temp_bytes;
        size_t budget = rmm_available_device_bytes(0.8);
        n_streams = clamp_streams_by_budget(n_streams, per_stream, budget);
    }

    // pool first: streams drain before it frees their scratch (RAII order).
    RmmScratchPool pool;
    // Bounded staging avoids page-locking huge host CSC arrays and gives every
    // dtype/index combination the same device footprint.
    HostStagingRing stage(n_streams, max_nnz);
    ScopedCudaStreams streams(n_streams, cudaStreamDefault);

    int* d_all_offsets = upload_batch_offsets(batches, pool);

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
        float* d_sparse_data_f32;
        int* d_sparse_indices;
        int* d_indptr;
        float* ref_dense;
        float* ref_sorted;
        float* grp_dense;
        float* grp_sorted;
        float* grp_nz;
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
        bufs[s].d_sparse_data_f32 = pool.alloc<float>(max_nnz);
        bufs[s].d_sparse_indices = pool.alloc<int>(max_nnz);
        bufs[s].d_indptr = pool.alloc<int>(sub_batch_cols + 1);
        bufs[s].ref_dense = pool.alloc<float>(sub_ref_items);
        bufs[s].ref_sorted = pool.alloc<float>(sub_ref_items);
        bufs[s].grp_dense = pool.alloc<float>(sub_grp_items);
        bufs[s].ref_seg_offsets = pool.alloc<int>(sub_batch_cols + 1);
        bufs[s].cub_temp = pool.alloc<uint8_t>(cub_temp_bytes);
        // LARGE/HUGE share the ref tie base: allocate whenever correcting.
        bufs[s].ref_tie_sums =
            compute_tie_corr ? pool.alloc<double>(sub_batch_cols) : nullptr;
        bufs[s].d_rank_sums =
            pool.alloc<double>((size_t)n_groups * sub_batch_cols);
        bufs[s].d_tie_corr =
            compute_tie_corr
                ? pool.alloc<double>((size_t)n_groups * sub_batch_cols)
                : nullptr;
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
            bufs[s].grp_nz =
                analytic_zeros ? pool.alloc<float>(sub_grp_items) : nullptr;
        } else {
            bufs[s].grp_sorted = nullptr;
            bufs[s].grp_seg_offsets = nullptr;
            bufs[s].grp_seg_ends = nullptr;
            bufs[s].grp_nz = nullptr;
        }
    }

    int tpb_rank =
        round_up_to_warp(std::min(max_grp_size, MAX_THREADS_PER_BLOCK));
    bool cast_use_gmem = false;
    size_t smem_cast = cast_accumulate_smem_config(
        n_groups_stats, compute_nnz, /*compute_totals=*/false, cast_use_gmem);

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

        IndptrT ptr_start = h_indptr[col];
        IndptrT ptr_end = h_indptr[col + sb_cols];
        size_t nnz = (size_t)(ptr_end - ptr_start);
        int nnz_i = checked_int_span(nnz, "OVO host CSC active batch nnz");

        // Cast-copy column batch into pinned staging, bulk H2D; the event lets
        // the next copy overlap compute.
        stage.wait(s);
        host_copy_slice_as<float, int>(h_data, h_indices, (size_t)ptr_start,
                                       nnz_i, stage.get<0>(s), stage.get<1>(s));
        cudaMemcpyAsync(buf.d_sparse_data_f32, stage.get<0>(s),
                        nnz * sizeof(float), cudaMemcpyHostToDevice, stream);
        cudaMemcpyAsync(buf.d_sparse_indices, stage.get<1>(s),
                        nnz * sizeof(int), cudaMemcpyHostToDevice, stream);
        stage.record(s, stream);
        int* src = d_all_offsets + (size_t)batch_idx * (sub_batch_cols + 1);
        cudaMemcpyAsync(buf.d_indptr, src, (sb_cols + 1) * sizeof(int),
                        cudaMemcpyDeviceToDevice, stream);

        // Data already f32 on device: accumulate stats (cast is f32->f32
        // no-op).
        launch_ovr_cast_and_accumulate_sparse<float, int>(
            buf.d_sparse_data_f32, buf.d_sparse_data_f32, buf.d_sparse_indices,
            buf.d_indptr, d_stats_codes, buf.d_group_sums, buf.d_group_nnz,
            nullptr, nullptr, sb_cols, n_groups_stats, compute_nnz,
            /*compute_totals=*/false, UTIL_BLOCK_SIZE, smem_cast, cast_use_gmem,
            stream);

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
                          buf.cub_temp,        buf.grp_nz};
        ovo_dispatch_tiers(buf.ref_sorted, buf.grp_dense, d_grp_offsets, t1, sc,
                           d_sort_group_ids, n_sort_groups, cub_temp_bytes,
                           sb_grp_actual, tpb_rank, n_ref, n_all_grp, sb_cols,
                           n_groups, compute_tie_corr, analytic_zeros, stream);

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

// Gather selected rows from precomputed absolute CSR spans, avoiding per-row
// lower_bound searches during planning and staging.
template <typename StageIndexT, typename InT, typename IndexT, typename IndptrT,
          typename CompactT>
static void host_gather_rows_compact_spans(
    const InT* h_data, const IndexT* h_indices, const int* row_ids,
    const IndptrT* row_starts, const IndptrT* row_stops,
    const CompactT* compact_indptr, CompactT base, int n_target, int col_start,
    float* stage_vals, StageIndexT* stage_cols) {
    host_parallel_ranges(n_target, [&](int i0, int i1) {
        for (int i = i0; i < i1; i++) {
            int row = row_ids[i];
            IndptrT src = row_starts[row];
            size_t dst = (size_t)(compact_indptr[i] - base);
            size_t count = (size_t)(row_stops[row] - src);
            for (size_t k = 0; k < count; k++) {
                stage_vals[dst + k] = (float)h_data[(size_t)src + k];
                stage_cols[dst + k] =
                    (StageIndexT)((int)h_indices[(size_t)src + k] - col_start);
            }
        }
    });
}

template <typename StageIndexT, typename InT, typename IndexT, typename IndptrT>
static void ovo_streaming_csr_host_impl_typed(
    const InT* h_data, const IndexT* h_indices, const int* h_ref_row_ids,
    int n_ref, const int* h_grp_row_ids, const int* h_grp_offsets,
    int n_all_grp, int n_test, double* d_rank_sums, double* d_tie_corr,
    double* d_group_sums, double* d_group_nnz, int col_start, int n_cols,
    int n_groups_stats, bool compute_tie_corr, bool compute_nnz,
    int sub_batch_cols, bool analytic_zeros, const IndptrT* row_starts,
    const IndptrT* row_stops) {
    if (n_cols == 0 || n_ref == 0 || n_test == 0 || n_all_grp == 0) return;
    // Compacted indptrs on host. IndptrT for grp (can exceed 2^31 nnz when
    // large/dense); ref stays int32 (n_ref × n_cols ≪ 2B, matches CUB temp).
    std::vector<int> h_ref_indptr_compact(n_ref + 1);
    size_t max_ref_row_nnz = 0;
    h_ref_indptr_compact[0] = 0;
    for (int i = 0; i < n_ref; i++) {
        int r = h_ref_row_ids[i];
        size_t row_nnz = (size_t)(row_stops[r] - row_starts[r]);
        if (row_nnz > (size_t)std::numeric_limits<int>::max()) {
            throw std::runtime_error(
                "OVO host CSR reference row exceeds int32 compacted nnz limit");
        }
        int nnz_i = (int)row_nnz;
        if (row_nnz > max_ref_row_nnz) {
            max_ref_row_nnz = row_nnz;
        }
        if ((size_t)h_ref_indptr_compact[i] + (size_t)nnz_i >
            (size_t)std::numeric_limits<int>::max()) {
            throw std::runtime_error(
                "OVO host CSR reference compacted nnz exceeds int32 limit");
        }
        h_ref_indptr_compact[i + 1] = h_ref_indptr_compact[i] + nnz_i;
    }
    int ref_nnz = h_ref_indptr_compact[n_ref];

    // grp: compacted indptr over concatenated test-group rows.
    std::vector<IndptrT> h_grp_indptr_compact(n_all_grp + 1);
    size_t max_pack_row_nnz = 0;
    h_grp_indptr_compact[0] = 0;
    for (int i = 0; i < n_all_grp; i++) {
        int r = h_grp_row_ids[i];
        IndptrT nnz_i = row_stops[r] - row_starts[r];
        max_pack_row_nnz = std::max(max_pack_row_nnz, (size_t)nnz_i);
        h_grp_indptr_compact[i + 1] = h_grp_indptr_compact[i] + nnz_i;
    }

    OvoHostCsrPackPlan pack_plan = plan_ovo_host_csr_packs(
        h_grp_offsets, h_grp_indptr_compact.data(), n_all_grp, n_test, n_cols,
        n_ref, sub_batch_cols);
    const std::vector<OvoHostCsrPack>& packs = pack_plan.packs;
    int max_pack_rows = pack_plan.max_pack_rows;
    size_t max_pack_nnz = pack_plan.max_pack_nnz;
    int max_pack_K = pack_plan.max_pack_K;
    int max_pack_sb_cols = pack_plan.max_pack_sb_cols;
    size_t max_sub_items = pack_plan.max_sub_items;
    if (max_pack_rows == 0) return;

    RmmScratchPool pool;
    ScopedCudaStream ref_stream(cudaStreamNonBlocking);

    // LFC-only calls need sums but not nnz. Those sums are block-reduced from
    // the dense values already consumed by OVO ranking, avoiding one FP64
    // atomic for every stored CSR value. Points requests retain the established
    // sparse stats accumulator because they need nnz in the same pass.
    bool fused_rank_sums = !compute_nnz;

    cudaMemsetAsync(d_group_sums, 0,
                    (size_t)n_groups_stats * n_cols * sizeof(double),
                    ref_stream);
    if (compute_nnz) {
        cudaMemsetAsync(d_group_nnz, 0,
                        (size_t)n_groups_stats * n_cols * sizeof(double),
                        ref_stream);
    }

    // Upload compacted indptrs + group boundaries. Row ids are consumed by the
    // host gather and are not needed on device.
    IndptrT* d_grp_indptr_compact = pool.alloc<IndptrT>(n_all_grp + 1);
    int* d_grp_offsets_full = pool.alloc<int>(n_test + 1);
    cudaMemcpy(d_grp_indptr_compact, h_grp_indptr_compact.data(),
               (n_all_grp + 1) * sizeof(IndptrT), cudaMemcpyHostToDevice);
    cudaMemcpy(d_grp_offsets_full, h_grp_offsets, (n_test + 1) * sizeof(int),
               cudaMemcpyHostToDevice);

    // Phase 1: ref setup with scoped scratch; sorted cache persists.
    // Build by column chunk so CUB item counts and extract scratch stay
    // bounded.
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
    {
        // Roll reference rows through up to two bounded staging slots.
        size_t ref_stage_capacity = ref_nnz ? (size_t)ref_nnz : 1;
        int ref_stage_slots = 1;
        if (ref_nnz > 0) {
            size_t bounded_capacity =
                std::max(STAGE_RING_NNZ_CAP, max_ref_row_nnz);
            ref_stage_capacity = std::min((size_t)ref_nnz, bounded_capacity);
            if ((size_t)ref_nnz >= 2 * ref_stage_capacity) {
                ref_stage_slots = OVO_HOST_CSR_REF_STAGE_RING_SLOTS;
            }
        }
        PinnedRing<float, StageIndexT> ref_stage(ref_stage_slots,
                                                 ref_stage_capacity);
        ScopedCudaBuffer ref_data_f32_buf(ref_nnz * sizeof(float));
        ScopedCudaBuffer ref_indices_buf(ref_nnz * sizeof(StageIndexT));
        ScopedCudaBuffer ref_indptr_buf((n_ref + 1) * sizeof(int));
        ScopedCudaBuffer ref_dense_buf(ref_chunk_items * sizeof(float));
        ScopedCudaBuffer ref_seg_buf((ref_chunk_cols + 1) * sizeof(int));

        float* d_ref_data_f32 = (float*)ref_data_f32_buf.data();
        StageIndexT* d_ref_indices = (StageIndexT*)ref_indices_buf.data();
        int* d_ref_indptr = (int*)ref_indptr_buf.data();
        float* d_ref_dense = (float*)ref_dense_buf.data();
        int* d_ref_seg = (int*)ref_seg_buf.data();

        cudaMemcpy(d_ref_indptr, h_ref_indptr_compact.data(),
                   (n_ref + 1) * sizeof(int), cudaMemcpyHostToDevice);

        // Host-gather ref rows into pinned staging, bulk H2D, accumulate stats.
        if (n_ref > 0 && ref_nnz > 0) {
            int ref_row_begin = 0;
            int stage_slot = 0;
            while (ref_row_begin < n_ref) {
                int block_base = h_ref_indptr_compact[ref_row_begin];
                int ref_row_end = ref_row_begin + 1;
                while (ref_row_end < n_ref &&
                       (size_t)(h_ref_indptr_compact[ref_row_end + 1] -
                                block_base) <= ref_stage_capacity) {
                    ref_row_end++;
                }
                size_t block_nnz =
                    (size_t)(h_ref_indptr_compact[ref_row_end] - block_base);
                int slot = stage_slot % ref_stage_slots;
                stage_slot++;
                ref_stage.wait(slot);
                host_gather_rows_compact_spans(
                    h_data, h_indices, h_ref_row_ids + ref_row_begin,
                    row_starts, row_stops,
                    h_ref_indptr_compact.data() + ref_row_begin, block_base,
                    ref_row_end - ref_row_begin, col_start,
                    ref_stage.template get<0>(slot),
                    ref_stage.template get<1>(slot));
                cuda_check(cudaMemcpyAsync(d_ref_data_f32 + block_base,
                                           ref_stage.template get<0>(slot),
                                           block_nnz * sizeof(float),
                                           cudaMemcpyHostToDevice, ref_stream),
                           "OVO host CSR ref staged vals H2D");
                cuda_check(cudaMemcpyAsync(d_ref_indices + block_base,
                                           ref_stage.template get<1>(slot),
                                           block_nnz * sizeof(StageIndexT),
                                           cudaMemcpyHostToDevice, ref_stream),
                           "OVO host CSR ref staged cols H2D");
                ref_stage.record(slot, ref_stream);
                ref_row_begin = ref_row_end;
            }
            if (!fused_rank_sums) {
                csr_compact_accumulate_kernel<<<n_ref, UTIL_BLOCK_SIZE, 0,
                                                ref_stream>>>(
                    d_ref_data_f32, d_ref_indices, d_ref_indptr,
                    /*d_stats_codes=*/nullptr, /*fixed_slot=*/n_test,
                    d_group_sums, d_group_nnz, n_ref, n_cols, n_groups_stats,
                    compute_nnz);
                CUDA_CHECK_LAST_ERROR(csr_compact_accumulate_kernel);
            }
        }

        size_t ref_cub_bytes = cub_segmented_sortkeys_temp_bytes(
            ref_chunk_items_i32, ref_chunk_cols);
        ScopedCudaBuffer cub_temp_buf(ref_cub_bytes);

        // Extract + segment-sort the reference per column chunk.
        for (int cs = 0; cs < n_cols; cs += ref_chunk_cols) {
            int ce = std::min(cs + ref_chunk_cols, n_cols);
            int cc = ce - cs;
            size_t chunk_items = (size_t)n_ref * (size_t)cc;
            cudaMemsetAsync(d_ref_dense, 0, chunk_items * sizeof(float),
                            ref_stream);
            csr_extract_dense_identity_rows_kernel<float, StageIndexT>
                <<<(n_ref + UTIL_BLOCK_SIZE - 1) / UTIL_BLOCK_SIZE,
                   UTIL_BLOCK_SIZE, 0, ref_stream>>>(
                    d_ref_data_f32, d_ref_indices, d_ref_indptr, d_ref_dense,
                    n_ref, cs, ce);
            CUDA_CHECK_LAST_ERROR(csr_extract_dense_identity_rows_kernel);
            if (fused_rank_sums) {
                ovo_dense_column_sum_kernel<<<cc, UTIL_BLOCK_SIZE, 0,
                                              ref_stream>>>(
                    d_ref_dense, d_group_sums + (size_t)n_test * n_cols + cs,
                    n_ref, cc);
                CUDA_CHECK_LAST_ERROR(ovo_dense_column_sum_kernel);
            }
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
    bool may_need_cub = compute_tie_corr && t1.run_huge;

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
    int max_pack_kernel_seg =
        checked_int_product((size_t)max_pack_K, (size_t)max_pack_sb_cols,
                            "OVO host CSR pack segment buffer");
    constexpr size_t window_value_bytes =
        sizeof(WilcoxonSparseWindowDTypes::value_type);
    size_t cub_group_slab_count = 0;
    if (may_need_cub) cub_group_slab_count = analytic_zeros ? 2 : 1;

    // Clamp streams to the post-ref free-memory budget.
    // Per-stream pack buffers dominate; fewer streams reduce overlap only.
    {
        using StageWindowDTypes =
            SparseWindowDTypes<float, StageIndexT, double>;
        size_t per_stream =
            sparse_window_nnz_bytes<StageWindowDTypes>(max_pack_nnz) +
            (size_t)(max_pack_rows + 1) * sizeof(int)  // grp indptr
            + (compute_nnz ? (size_t)max_pack_rows * sizeof(int) : 0) +
            (size_t)(max_pack_K + 1) * sizeof(int)  // pack grp offsets
            + max_sub_items * window_value_bytes    // grp dense
            + sparse_window_accum_bytes<WilcoxonSparseWindowDTypes>(
                  (size_t)(1 + compute_tie_corr) * max_pack_K *
                  max_pack_sb_cols)  // rank + optional tie
            + sparse_window_accum_bytes<WilcoxonSparseWindowDTypes>(
                  compute_tie_corr ? (size_t)max_pack_sb_cols : 0)  // ref tie
            +
            (may_need_cub
                 ? cub_group_slab_count * max_sub_items *
                           window_value_bytes              // grp sorted/nz
                       + (size_t)max_pack_K * sizeof(int)  // sort ids
                       + 2 * (size_t)max_pack_kernel_seg * sizeof(int)  // segs
                       + cub_grp_bytes  // cub temp
                 : 0);
        size_t budget = rmm_available_device_bytes(0.9);
        n_streams = clamp_streams_by_budget(n_streams, per_stream, budget);
    }

    ScopedCudaStreams streams(n_streams, cudaStreamDefault);

    struct StreamBuf {
        float* d_grp_data_f32;
        StageIndexT* d_grp_indices;
        int* d_grp_indptr;
        int* d_pack_grp_offsets;
        int* d_pack_stats_codes;
        float* d_grp_dense;
        float* d_grp_sorted;
        float* d_grp_nz;
        double* d_ref_tie_sums;
        int* d_sort_group_ids;
        int* d_grp_seg_offsets;
        int* d_grp_seg_ends;
        uint8_t* cub_temp;
        double* d_rank_sums;
        double* d_tie_corr;
    };
    std::vector<StreamBuf> bufs(n_streams);
    for (int s = 0; s < n_streams; s++) {
        bufs[s].d_grp_data_f32 = pool.alloc<float>(max_pack_nnz);
        bufs[s].d_grp_indices = pool.alloc<StageIndexT>(max_pack_nnz);
        bufs[s].d_grp_indptr = pool.alloc<int>(max_pack_rows + 1);
        bufs[s].d_pack_grp_offsets = pool.alloc<int>(max_pack_K + 1);
        bufs[s].d_pack_stats_codes =
            compute_nnz ? pool.alloc<int>(max_pack_rows) : nullptr;
        bufs[s].d_grp_dense = pool.alloc<float>(max_sub_items);
        bufs[s].d_ref_tie_sums =
            compute_tie_corr ? pool.alloc<double>(max_pack_sb_cols) : nullptr;
        bufs[s].d_rank_sums =
            pool.alloc<double>((size_t)max_pack_K * max_pack_sb_cols);
        bufs[s].d_tie_corr =
            compute_tie_corr
                ? pool.alloc<double>((size_t)max_pack_K * max_pack_sb_cols)
                : nullptr;
        if (may_need_cub) {
            bufs[s].d_grp_sorted = pool.alloc<float>(max_sub_items);
            bufs[s].d_sort_group_ids = pool.alloc<int>(max_pack_K);
            bufs[s].d_grp_seg_offsets = pool.alloc<int>(max_pack_kernel_seg);
            bufs[s].d_grp_seg_ends = pool.alloc<int>(max_pack_kernel_seg);
            bufs[s].cub_temp = pool.alloc<uint8_t>(cub_grp_bytes);
            bufs[s].d_grp_nz =
                analytic_zeros ? pool.alloc<float>(max_sub_items) : nullptr;
        } else {
            bufs[s].d_grp_sorted = nullptr;
            bufs[s].d_sort_group_ids = nullptr;
            bufs[s].d_grp_seg_offsets = nullptr;
            bufs[s].d_grp_seg_ends = nullptr;
            bufs[s].cub_temp = nullptr;
            bufs[s].d_grp_nz = nullptr;
        }
    }

    // Rolling pinned staging fills pack device buffers in <= stage_cap nnz
    // blocks. This keeps page-locked footprint small while extra slots overlap
    // H2D.
    size_t stage_cap =
        std::max(std::min(max_pack_nnz, STAGE_RING_NNZ_CAP), max_pack_row_nnz);
    constexpr int ring_slots = 2;
    PinnedRing<float, StageIndexT> stage(ring_slots, stage_cap);
    int stage_slot = 0;

    for (int p = 0; p < (int)packs.size(); p++) {
        const OvoHostCsrPack& pack = packs[p];
        int K = pack.end - pack.first;
        if (K == 0 || pack.n_rows == 0) continue;
        OvoTierPlan pack_t1 = make_ovo_tier_plan(h_grp_offsets + pack.first, K);
        int pack_tpb_rank = round_up_to_warp(
            std::min(pack_t1.max_grp_size, MAX_THREADS_PER_BLOCK));
        int pack_huge_skip_le = pack_t1.huge_skip_le;
        std::vector<int> h_sort_group_ids;
        int pack_n_sort_groups = K;
        bool pack_run_huge = compute_tie_corr && pack_t1.run_huge;
        if (pack_run_huge) {
            h_sort_group_ids = make_sort_group_ids(h_grp_offsets + pack.first,
                                                   K, pack_huge_skip_le);
            pack_n_sort_groups = (int)h_sort_group_ids.size();
        }

        int s = p % n_streams;
        cudaStream_t stream = streams[s];
        auto& buf = bufs[s];

        if (pack_run_huge) {
            cudaMemcpyAsync(buf.d_sort_group_ids, h_sort_group_ids.data(),
                            h_sort_group_ids.size() * sizeof(int),
                            cudaMemcpyHostToDevice, stream);
        }

        int row_start = h_grp_offsets[pack.first];
        int pack_rows = pack.n_rows;
        int pack_sb = pack.sb_cols;

        // Rebase pack's output indptr (IndptrT → int32: pack nnz is bounded by
        // GROUP_DENSE_BUDGET so fits).
        {
            int count = pack_rows + 1;
            int blk = (count + UTIL_BLOCK_SIZE - 1) / UTIL_BLOCK_SIZE;
            rebase_indptr_kernel<IndptrT, int>
                <<<blk, UTIL_BLOCK_SIZE, 0, stream>>>(
                    d_grp_indptr_compact, buf.d_grp_indptr, row_start, count);
            CUDA_CHECK_LAST_ERROR(rebase_indptr_kernel);
        }

        // Per-pack group offsets on GPU — needed for stats codes.
        {
            int count = K + 1;
            int blk = (count + UTIL_BLOCK_SIZE - 1) / UTIL_BLOCK_SIZE;
            rebase_indptr_kernel<int, int><<<blk, UTIL_BLOCK_SIZE, 0, stream>>>(
                d_grp_offsets_full, buf.d_pack_grp_offsets, pack.first, count);
            CUDA_CHECK_LAST_ERROR(rebase_indptr_kernel);
        }

        if (compute_nnz) {
            int blk = (pack_rows + UTIL_BLOCK_SIZE - 1) / UTIL_BLOCK_SIZE;
            fill_pack_stats_codes_kernel<<<blk, UTIL_BLOCK_SIZE, 0, stream>>>(
                buf.d_pack_grp_offsets, buf.d_pack_stats_codes, K, pack.first);
            CUDA_CHECK_LAST_ERROR(fill_pack_stats_codes_kernel);
        }

        // Host-gather pack rows into rolling staging blocks, then H2D by
        // offset. Stats accumulate once over the full device-resident pack.
        if (pack.nnz > 0) {
            IndptrT pack_base = h_grp_indptr_compact[row_start];
            int rb0 = 0;
            while (rb0 < pack_rows) {
                IndptrT blk_base = h_grp_indptr_compact[row_start + rb0];
                int rb1 = rb0 + 1;
                while (rb1 < pack_rows &&
                       (size_t)(h_grp_indptr_compact[row_start + rb1 + 1] -
                                blk_base) <= stage_cap)
                    rb1++;
                size_t blk_nnz =
                    (size_t)(h_grp_indptr_compact[row_start + rb1] - blk_base);
                size_t dev_off = (size_t)(blk_base - pack_base);
                int slot = stage_slot % ring_slots;
                stage_slot++;
                // wait drains a prior H2D out of this slot before we overwrite
                // it; the event lets the next gather overlap the in-flight H2D.
                stage.wait(slot);
                host_gather_rows_compact_spans(
                    h_data, h_indices, h_grp_row_ids + row_start + rb0,
                    row_starts, row_stops,
                    h_grp_indptr_compact.data() + row_start + rb0, blk_base,
                    rb1 - rb0, col_start, stage.template get<0>(slot),
                    stage.template get<1>(slot));
                cuda_check(cudaMemcpyAsync(buf.d_grp_data_f32 + dev_off,
                                           stage.template get<0>(slot),
                                           blk_nnz * sizeof(float),
                                           cudaMemcpyHostToDevice, stream),
                           "OVO host CSR pack staged vals H2D");
                cuda_check(cudaMemcpyAsync(buf.d_grp_indices + dev_off,
                                           stage.template get<1>(slot),
                                           blk_nnz * sizeof(StageIndexT),
                                           cudaMemcpyHostToDevice, stream),
                           "OVO host CSR pack staged cols H2D");
                stage.record(slot, stream);
                rb0 = rb1;
            }
            if (!fused_rank_sums) {
                csr_compact_accumulate_kernel<<<pack_rows, UTIL_BLOCK_SIZE, 0,
                                                stream>>>(
                    buf.d_grp_data_f32, buf.d_grp_indices, buf.d_grp_indptr,
                    buf.d_pack_stats_codes, /*fixed_slot=*/-1, d_group_sums,
                    d_group_nnz, pack_rows, n_cols, n_groups_stats,
                    compute_nnz);
                CUDA_CHECK_LAST_ERROR(csr_compact_accumulate_kernel);
            }
        }

        int col = 0;
        while (col < n_cols) {
            int sb_cols = std::min(pack_sb, n_cols - col);
            int sb_items =
                checked_int_product((size_t)pack_rows, (size_t)sb_cols,
                                    "OVO host CSR active group sub-batch");

            cudaMemsetAsync(buf.d_grp_dense, 0, sb_items * sizeof(float),
                            stream);
            csr_extract_dense_identity_rows_kernel<float, StageIndexT>
                <<<(pack_rows + UTIL_BLOCK_SIZE - 1) / UTIL_BLOCK_SIZE,
                   UTIL_BLOCK_SIZE, 0, stream>>>(
                    buf.d_grp_data_f32, buf.d_grp_indices, buf.d_grp_indptr,
                    buf.d_grp_dense, pack_rows, col, col + sb_cols);
            CUDA_CHECK_LAST_ERROR(csr_extract_dense_identity_rows_kernel);

            const float* ref_sub = d_ref_sorted + (size_t)col * n_ref;

            OvoTierScratch sc{buf.d_ref_tie_sums,    buf.d_rank_sums,
                              buf.d_tie_corr,        buf.d_grp_sorted,
                              buf.d_grp_seg_offsets, buf.d_grp_seg_ends,
                              buf.cub_temp,          buf.d_grp_nz};
            ovo_dispatch_tiers(
                ref_sub, buf.d_grp_dense, buf.d_pack_grp_offsets, pack_t1, sc,
                buf.d_sort_group_ids, pack_n_sort_groups, cub_grp_bytes,
                sb_items, pack_tpb_rank, n_ref, pack_rows, sb_cols, K,
                compute_tie_corr, analytic_zeros, stream,
                fused_rank_sums
                    ? d_group_sums + (size_t)pack.first * n_cols + col
                    : nullptr,
                n_cols);

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

template <typename InT, typename IndexT, typename IndptrT>
static void ovo_streaming_csr_host_impl(
    const InT* h_data, const IndexT* h_indices, const int* h_ref_row_ids,
    int n_ref, const int* h_grp_row_ids, const int* h_grp_offsets,
    int n_all_grp, int n_test, double* d_rank_sums, double* d_tie_corr,
    double* d_group_sums, double* d_group_nnz, int col_start, int n_cols,
    int n_groups_stats, bool compute_tie_corr, bool compute_nnz,
    int sub_batch_cols, bool analytic_zeros, const IndptrT* row_starts,
    const IndptrT* row_stops) {
    // Range-shard inputs rebase every column id to the supplied local interval.
    // Use uint16 staging whenever that interval fits exactly.
    bool use_u16_indices = n_cols <= OVO_HOST_CSR_U16_COLUMN_CAPACITY;
    if (use_u16_indices) {
        ovo_streaming_csr_host_impl_typed<uint16_t>(
            h_data, h_indices, h_ref_row_ids, n_ref, h_grp_row_ids,
            h_grp_offsets, n_all_grp, n_test, d_rank_sums, d_tie_corr,
            d_group_sums, d_group_nnz, col_start, n_cols, n_groups_stats,
            compute_tie_corr, compute_nnz, sub_batch_cols, analytic_zeros,
            row_starts, row_stops);
        return;
    }
    ovo_streaming_csr_host_impl_typed<int>(
        h_data, h_indices, h_ref_row_ids, n_ref, h_grp_row_ids, h_grp_offsets,
        n_all_grp, n_test, d_rank_sums, d_tie_corr, d_group_sums, d_group_nnz,
        col_start, n_cols, n_groups_stats, compute_tie_corr, compute_nnz,
        sub_batch_cols, analytic_zeros, row_starts, row_stops);
}
