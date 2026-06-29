#include <cuda_runtime.h>
#include <cub/device/device_segmented_radix_sort.cuh>

#include <algorithm>
#include <limits>
#include <type_traits>
#include <vector>

#include "../nb_types.h"

#include "kernels_wilcoxon.cuh"
#include "wilcoxon_fast_common.cuh"
#include "kernels_wilcoxon_ovo.cuh"
#include "wilcoxon_ovr_kernels.cuh"
#include "wilcoxon_ovo_kernels.cuh"

using namespace nb::literals;

static void launch_ovr_rank_dense_streaming(
    const float* block, const int* group_codes, double* rank_sums,
    double* tie_corr, int n_rows, int n_cols, int n_groups,
    bool compute_tie_corr, int sub_batch_cols, cudaStream_t upstream_stream) {
    if (n_rows == 0 || n_cols == 0 || n_groups == 0) return;
    if (sub_batch_cols <= 0) sub_batch_cols = SUB_BATCH_COLS;

    DenseColumnBatchPlan batches = plan_dense_column_batches(
        n_rows, n_cols, sub_batch_cols, SAFE_BATCH_NNZ, "Dense OVR sub-batch");
    sub_batch_cols = batches.sub_batch_cols;
    int n_streams = clamp_streams_by_cols(n_cols, sub_batch_cols);
    size_t sub_items = batches.max_items;
    int sub_items_i32 = checked_cub_items(sub_items, "Dense OVR sub-batch");

    size_t cub_temp_bytes =
        cub_segmented_sortpairs_temp_bytes(sub_items_i32, sub_batch_cols);

    // pool first: streams drain before it frees their scratch (see guard doc).
    RmmScratchPool pool;
    ScopedCudaStreams streams(n_streams, cudaStreamNonBlocking);

    ScopedCudaEvent inputs_ready(cudaEventDisableTiming);
    inputs_ready.record(upstream_stream);
    for (int i = 0; i < n_streams; ++i) {
        cuda_check(cudaStreamWaitEvent(streams[i], inputs_ready.get(), 0),
                   "wait on inputs_ready (dense OVR)");
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
    for (int s = 0; s < n_streams; ++s) {
        bufs[s].keys_out = pool.alloc<float>(sub_items);
        bufs[s].vals_in = pool.alloc<int>(sub_items);
        bufs[s].vals_out = pool.alloc<int>(sub_items);
        bufs[s].seg_offsets = pool.alloc<int>(sub_batch_cols + 1);
        bufs[s].cub_temp = pool.alloc<uint8_t>(cub_temp_bytes);
        bufs[s].sub_rank_sums =
            pool.alloc<double>((size_t)n_groups * sub_batch_cols);
        bufs[s].sub_tie_corr = pool.alloc<double>(sub_batch_cols);
    }

    int tpb_rank = round_up_to_warp(n_rows);
    bool use_gmem = false;
    size_t smem_rank = ovr_smem_config(n_groups, use_gmem);

    int col = 0;
    int batch_idx = 0;
    while (col < n_cols) {
        int sb_cols = std::min(sub_batch_cols, n_cols - col);
        int sb_items = checked_int_product((size_t)n_rows, (size_t)sb_cols,
                                           "Dense OVR active sub-batch");
        int s = batch_idx % n_streams;
        cudaStream_t stream = streams[s];
        auto& buf = bufs[s];

        upload_linear_offsets(buf.seg_offsets, sb_cols, n_rows, stream);
        fill_row_indices_kernel<<<sb_cols, UTIL_BLOCK_SIZE, 0, stream>>>(
            buf.vals_in, n_rows, sb_cols);
        CUDA_CHECK_LAST_ERROR(fill_row_indices_kernel);

        const float* keys_in = block + (size_t)col * n_rows;
        cub_segmented_sortpairs(
            buf.cub_temp, cub_temp_bytes, keys_in, buf.keys_out, buf.vals_in,
            buf.vals_out, sb_items, sb_cols, buf.seg_offsets,
            buf.seg_offsets + 1, stream, "dense OVR segmented sort");

        if (use_gmem) {
            cuda_check(cudaMemsetAsync(
                           buf.sub_rank_sums, 0,
                           (size_t)n_groups * sb_cols * sizeof(double), stream),
                       "dense OVR gmem rank_sums memset");
        }
        rank_sums_from_sorted_kernel<<<sb_cols, tpb_rank, smem_rank, stream>>>(
            buf.keys_out, buf.vals_out, group_codes, buf.sub_rank_sums,
            buf.sub_tie_corr, n_rows, sb_cols, n_groups, compute_tie_corr,
            use_gmem);
        CUDA_CHECK_LAST_ERROR(rank_sums_from_sorted_kernel);

        cuda_check(
            cudaMemcpy2DAsync(rank_sums + col, n_cols * sizeof(double),
                              buf.sub_rank_sums, sb_cols * sizeof(double),
                              sb_cols * sizeof(double), n_groups,
                              cudaMemcpyDeviceToDevice, stream),
            "dense OVR rank_sums D2D copy");
        if (compute_tie_corr) {
            cuda_check(cudaMemcpyAsync(tie_corr + col, buf.sub_tie_corr,
                                       sb_cols * sizeof(double),
                                       cudaMemcpyDeviceToDevice, stream),
                       "dense OVR tie_corr D2D copy");
        }

        col += sb_cols;
        ++batch_idx;
    }

    sync_streams(streams, "dense OVR streaming rank");
}

// Host-streaming dense OVR: pinned multi-stream batches into F-order device
// slabs. F-order copies contiguous; C-order uses 2D copy; stats accumulate in
// f64.
template <typename T>
static void launch_ovr_rank_dense_host_streaming(
    const T* h_X, bool f_order, const int* group_codes, double* rank_sums,
    double* tie_corr, double* group_sums, double* group_nnz, double* total_sums,
    double* total_nnz, int n_rows, int n_cols, int n_groups,
    bool compute_tie_corr, bool compute_nnz, bool compute_totals,
    int sub_batch_cols) {
    if (n_rows == 0 || n_cols == 0 || n_groups == 0) return;
    if (sub_batch_cols <= 0) sub_batch_cols = SUB_BATCH_COLS;
    const bool compute_stats = group_sums != nullptr;
    compute_nnz = compute_nnz && (group_nnz != nullptr);
    compute_totals = compute_stats && compute_totals && (total_sums != nullptr);
    // F-order float32 input feeds the sort directly (no cast/transpose buffer).
    const bool fast_keys = f_order && std::is_same<T, float>::value;

    DenseColumnBatchPlan batches =
        plan_dense_column_batches(n_rows, n_cols, sub_batch_cols,
                                  SAFE_BATCH_NNZ, "Dense host OVR sub-batch");
    sub_batch_cols = batches.sub_batch_cols;
    int n_streams = clamp_streams_by_cols(n_cols, sub_batch_cols);
    size_t sub_items = batches.max_items;
    int sub_items_i32 =
        checked_cub_items(sub_items, "Dense host OVR sub-batch");
    size_t cub_temp_bytes =
        cub_segmented_sortpairs_temp_bytes(sub_items_i32, sub_batch_cols);

    // Clamp stream count to device memory budget so a large matrix shrinks the
    // pipeline rather than OOMing on per-stream sort scratch.
    size_t per_stream_bytes =
        sub_items * (sizeof(T) + (fast_keys ? 0 : sizeof(float)) +
                     sizeof(float) + 2 * sizeof(int)) +
        cub_temp_bytes + (size_t)(sub_batch_cols + 1) * sizeof(int) +
        (size_t)n_groups * sub_batch_cols * sizeof(double) +
        (size_t)sub_batch_cols * sizeof(double) +
        (compute_stats ? (size_t)n_groups * sub_batch_cols * sizeof(double)
                       : 0) +
        (compute_nnz ? (size_t)n_groups * sub_batch_cols * sizeof(double) : 0) +
        (compute_totals ? (size_t)sub_batch_cols * sizeof(double) : 0) +
        (compute_totals && compute_nnz ? (size_t)sub_batch_cols * sizeof(double)
                                       : 0);
    n_streams = clamp_streams_by_budget(n_streams, per_stream_bytes,
                                        rmm_available_device_bytes(0.8));

    // pool first: streams drain before it frees their scratch (see guard doc).
    RmmScratchPool pool;
    // Best-effort pin for faster async H2D; on failure proceed unpinned.
    HostRegisterGuard _pin(const_cast<T*>(h_X),
                           (size_t)n_rows * n_cols * sizeof(T), 0,
                           /*best_effort=*/true);
    ScopedCudaStreams streams(n_streams, cudaStreamDefault);

    struct StreamBuf {
        T* d_stg;
        float* block_f32;
        float* keys_out;
        int* vals_in;
        int* vals_out;
        int* seg_offsets;
        uint8_t* cub_temp;
        double* sub_rank_sums;
        double* sub_tie_corr;
        double* sub_group_sums;
        double* sub_group_nnz;
        double* sub_total_sums;
        double* sub_total_nnz;
    };
    std::vector<StreamBuf> bufs(n_streams);
    for (int s = 0; s < n_streams; ++s) {
        bufs[s].d_stg = pool.alloc<T>(sub_items);
        bufs[s].block_f32 = fast_keys ? nullptr : pool.alloc<float>(sub_items);
        bufs[s].keys_out = pool.alloc<float>(sub_items);
        bufs[s].vals_in = pool.alloc<int>(sub_items);
        bufs[s].vals_out = pool.alloc<int>(sub_items);
        bufs[s].seg_offsets = pool.alloc<int>(sub_batch_cols + 1);
        bufs[s].cub_temp = pool.alloc<uint8_t>(cub_temp_bytes);
        bufs[s].sub_rank_sums =
            pool.alloc<double>((size_t)n_groups * sub_batch_cols);
        bufs[s].sub_tie_corr = pool.alloc<double>(sub_batch_cols);
        bufs[s].sub_group_sums =
            compute_stats
                ? pool.alloc<double>((size_t)n_groups * sub_batch_cols)
                : nullptr;
        bufs[s].sub_group_nnz =
            compute_nnz ? pool.alloc<double>((size_t)n_groups * sub_batch_cols)
                        : nullptr;
        bufs[s].sub_total_sums =
            compute_totals ? pool.alloc<double>(sub_batch_cols) : nullptr;
        bufs[s].sub_total_nnz = (compute_totals && compute_nnz)
                                    ? pool.alloc<double>(sub_batch_cols)
                                    : nullptr;
    }

    int tpb_rank = round_up_to_warp(n_rows);
    bool use_gmem = false;
    size_t smem_rank = ovr_smem_config(n_groups, use_gmem);

    cudaDeviceSynchronize();

    int col = 0;
    int batch_idx = 0;
    while (col < n_cols) {
        int sb_cols = std::min(sub_batch_cols, n_cols - col);
        int sb_items = checked_int_product((size_t)n_rows, (size_t)sb_cols,
                                           "Dense host OVR active sub-batch");
        int s = batch_idx % n_streams;
        cudaStream_t stream = streams[s];
        auto& buf = bufs[s];

        // H2D the column window (overlaps the prior batch rank).
        if (f_order) {
            cudaMemcpyAsync(buf.d_stg, h_X + (size_t)col * n_rows,
                            (size_t)sb_items * sizeof(T),
                            cudaMemcpyHostToDevice, stream);
        } else {
            cudaMemcpy2DAsync(buf.d_stg, (size_t)sb_cols * sizeof(T), h_X + col,
                              (size_t)n_cols * sizeof(T),
                              (size_t)sb_cols * sizeof(T), n_rows,
                              cudaMemcpyHostToDevice, stream);
        }

        const float* keys_in;
        if (fast_keys) {
            keys_in = reinterpret_cast<const float*>(buf.d_stg);
        } else {
            // grid-stride kernel: bounded grid covers any sb_items (<=INT_MAX)
            // with no launch-math overflow.
            const unsigned int grid = (unsigned int)std::min<size_t>(
                ((size_t)sb_items + UTIL_BLOCK_SIZE - 1) / UTIL_BLOCK_SIZE,
                65535u);
            dense_block_to_f32_kernel<T><<<grid, UTIL_BLOCK_SIZE, 0, stream>>>(
                buf.d_stg, buf.block_f32, n_rows, sb_cols, f_order);
            CUDA_CHECK_LAST_ERROR(dense_block_to_f32_kernel);
            keys_in = buf.block_f32;
        }

        upload_linear_offsets(buf.seg_offsets, sb_cols, n_rows, stream);
        fill_row_indices_kernel<<<sb_cols, UTIL_BLOCK_SIZE, 0, stream>>>(
            buf.vals_in, n_rows, sb_cols);
        CUDA_CHECK_LAST_ERROR(fill_row_indices_kernel);

        cub_segmented_sortpairs(
            buf.cub_temp, cub_temp_bytes, keys_in, buf.keys_out, buf.vals_in,
            buf.vals_out, sb_items, sb_cols, buf.seg_offsets,
            buf.seg_offsets + 1, stream, "dense host OVR segmented sort");

        // gmem rank mode atomicAdds without self-zeroing and the buffer is
        // reused round-robin, so zero it first.
        if (use_gmem) {
            cuda_check(cudaMemsetAsync(
                           buf.sub_rank_sums, 0,
                           (size_t)n_groups * sb_cols * sizeof(double), stream),
                       "dense host OVR gmem rank_sums memset");
        }
        rank_sums_from_sorted_kernel<<<sb_cols, tpb_rank, smem_rank, stream>>>(
            buf.keys_out, buf.vals_out, group_codes, buf.sub_rank_sums,
            buf.sub_tie_corr, n_rows, sb_cols, n_groups, compute_tie_corr,
            use_gmem);
        CUDA_CHECK_LAST_ERROR(rank_sums_from_sorted_kernel);

        cuda_check(
            cudaMemcpy2DAsync(rank_sums + col, n_cols * sizeof(double),
                              buf.sub_rank_sums, sb_cols * sizeof(double),
                              sb_cols * sizeof(double), n_groups,
                              cudaMemcpyDeviceToDevice, stream),
            "dense host OVR rank_sums D2D copy");
        if (compute_tie_corr) {
            cuda_check(cudaMemcpyAsync(tie_corr + col, buf.sub_tie_corr,
                                       sb_cols * sizeof(double),
                                       cudaMemcpyDeviceToDevice, stream),
                       "dense host OVR tie_corr D2D copy");
        }

        // Group sums (+nnz) for means/pts, f64 from native staging (matches
        // the Aggregate path).
        if (compute_stats) {
            cudaMemsetAsync(buf.sub_group_sums, 0,
                            (size_t)n_groups * sb_cols * sizeof(double),
                            stream);
            if (compute_nnz) {
                cudaMemsetAsync(buf.sub_group_nnz, 0,
                                (size_t)n_groups * sb_cols * sizeof(double),
                                stream);
            }
            if (compute_totals) {
                cudaMemsetAsync(buf.sub_total_sums, 0, sb_cols * sizeof(double),
                                stream);
                if (compute_nnz) {
                    cudaMemsetAsync(buf.sub_total_nnz, 0,
                                    sb_cols * sizeof(double), stream);
                }
            }
            dense_group_accumulate_kernel<T>
                <<<sb_cols, UTIL_BLOCK_SIZE, 0, stream>>>(
                    buf.d_stg, group_codes, buf.sub_group_sums,
                    compute_nnz ? buf.sub_group_nnz : buf.sub_group_sums,
                    buf.sub_total_sums,
                    compute_nnz ? buf.sub_total_nnz : buf.sub_total_sums,
                    n_rows, sb_cols, n_groups, f_order, compute_nnz,
                    compute_totals);
            CUDA_CHECK_LAST_ERROR(dense_group_accumulate_kernel);
            scatter_cols_2d(group_sums + col, buf.sub_group_sums, n_groups,
                            n_cols, sb_cols, stream);
            if (compute_nnz) {
                scatter_cols_2d(group_nnz + col, buf.sub_group_nnz, n_groups,
                                n_cols, sb_cols, stream);
            }
            if (compute_totals) {
                cudaMemcpyAsync(total_sums + col, buf.sub_total_sums,
                                sb_cols * sizeof(double),
                                cudaMemcpyDeviceToDevice, stream);
                if (compute_nnz) {
                    cudaMemcpyAsync(total_nnz + col, buf.sub_total_nnz,
                                    sb_cols * sizeof(double),
                                    cudaMemcpyDeviceToDevice, stream);
                }
            }
        }

        col += sb_cols;
        ++batch_idx;
    }

    sync_streams(streams, "dense host OVR streaming");
}

static void launch_ovo_rank_dense_tiered_unsorted_ref(
    const float* ref_data, const float* grp_data, const int* grp_offsets,
    double* rank_sums, double* tie_corr, int n_ref, int n_all_grp, int n_cols,
    int n_groups, bool compute_tie_corr, int sub_batch_cols,
    cudaStream_t upstream_stream) {
    if (n_cols == 0 || n_ref == 0 || n_all_grp == 0 || n_groups == 0) return;
    if (sub_batch_cols <= 0) sub_batch_cols = SUB_BATCH_COLS;

    std::vector<int> h_offsets(n_groups + 1);
    cuda_check(cudaStreamSynchronize(upstream_stream),
               "dense OVO sync before offsets D2H");
    cuda_check(cudaMemcpy(h_offsets.data(), grp_offsets,
                          (n_groups + 1) * sizeof(int), cudaMemcpyDeviceToHost),
               "dense OVO group offsets D2H");
    auto t1 = make_ovo_tier_plan(h_offsets.data(), n_groups);
    int max_grp_size = t1.max_grp_size;
    bool run_huge = compute_tie_corr && t1.run_huge;

    std::vector<int> h_sort_group_ids;
    int n_sort_groups = n_groups;
    if (run_huge) {
        h_sort_group_ids =
            make_sort_group_ids(h_offsets.data(), n_groups, t1.huge_skip_le);
        n_sort_groups = (int)h_sort_group_ids.size();
    }

    DenseColumnBatchPlan batches = plan_dense_column_batches(
        std::max(n_ref, n_all_grp), n_cols, sub_batch_cols, SAFE_BATCH_NNZ,
        "Dense OVO sub-batch");
    sub_batch_cols = batches.sub_batch_cols;
    int n_streams = clamp_streams_by_cols(n_cols, sub_batch_cols);

    size_t sub_ref_items = (size_t)n_ref * sub_batch_cols;
    int sub_ref_items_i32 =
        checked_cub_items(sub_ref_items, "Dense OVO reference sub-batch");

    size_t sub_grp_items = (size_t)n_all_grp * sub_batch_cols;
    int sub_grp_items_i32 =
        checked_cub_items(sub_grp_items, "Dense OVO group sub-batch");

    size_t grp_cub_temp_bytes = 0;
    if (run_huge) {
        int max_grp_seg =
            checked_int_product((size_t)n_sort_groups, (size_t)sub_batch_cols,
                                "Dense OVO group segment count");
        grp_cub_temp_bytes =
            cub_segmented_sortkeys_temp_bytes(sub_grp_items_i32, max_grp_seg);
    }
    size_t ref_cub_temp_bytes =
        cub_segmented_sortkeys_temp_bytes(sub_ref_items_i32, sub_batch_cols);

    {
        size_t per_stream =
            sub_ref_items * sizeof(float) +
            (size_t)(sub_batch_cols + 1) * sizeof(int) + ref_cub_temp_bytes +
            (run_huge ? sub_grp_items * sizeof(float) : 0) +
            (run_huge ? 2 * (size_t)n_sort_groups * sub_batch_cols * sizeof(int)
                      : 0) +
            (run_huge ? grp_cub_temp_bytes : 0) +
            (compute_tie_corr ? (size_t)sub_batch_cols * sizeof(double) : 0) +
            2 * (size_t)n_groups * sub_batch_cols * sizeof(double);
        n_streams = clamp_streams_by_budget(n_streams, per_stream,
                                            rmm_available_device_bytes(0.8));
    }

    // pool first: streams drain before it frees their scratch (see guard doc).
    RmmScratchPool pool;
    ScopedCudaStreams streams(n_streams, cudaStreamNonBlocking);

    ScopedCudaEvent inputs_ready(cudaEventDisableTiming);
    inputs_ready.record(upstream_stream);
    for (int i = 0; i < n_streams; ++i) {
        cuda_check(cudaStreamWaitEvent(streams[i], inputs_ready.get(), 0),
                   "wait on inputs_ready (dense OVO)");
    }
    int* d_sort_group_ids = nullptr;
    if (run_huge) {
        d_sort_group_ids = pool.alloc<int>(h_sort_group_ids.size());
        cuda_check(cudaMemcpy(d_sort_group_ids, h_sort_group_ids.data(),
                              h_sort_group_ids.size() * sizeof(int),
                              cudaMemcpyHostToDevice),
                   "dense OVO sort group ids H2D");
    }

    struct StreamBuf {
        float* ref_sorted;
        int* ref_seg_offsets;
        uint8_t* ref_cub_temp;
        float* grp_sorted;
        int* grp_seg_offsets;
        int* grp_seg_ends;
        uint8_t* grp_cub_temp;
        double* ref_tie_sums;
        double* sub_rank_sums;
        double* sub_tie_corr;
    };
    std::vector<StreamBuf> bufs(n_streams);
    for (int s = 0; s < n_streams; ++s) {
        bufs[s].ref_sorted = pool.alloc<float>(sub_ref_items);
        bufs[s].ref_seg_offsets = pool.alloc<int>(sub_batch_cols + 1);
        bufs[s].ref_cub_temp = pool.alloc<uint8_t>(ref_cub_temp_bytes);
        bufs[s].grp_cub_temp =
            run_huge ? pool.alloc<uint8_t>(grp_cub_temp_bytes) : nullptr;
        // All tiers share the ref tie base, so allocate whenever correcting.
        bufs[s].ref_tie_sums =
            compute_tie_corr ? pool.alloc<double>(sub_batch_cols) : nullptr;
        bufs[s].sub_rank_sums =
            pool.alloc<double>((size_t)n_groups * sub_batch_cols);
        bufs[s].sub_tie_corr =
            pool.alloc<double>((size_t)n_groups * sub_batch_cols);
        if (run_huge) {
            bufs[s].grp_sorted = pool.alloc<float>(sub_grp_items);
            int max_seg = checked_int_product((size_t)n_sort_groups,
                                              (size_t)sub_batch_cols,
                                              "Dense OVO group segment buffer");
            bufs[s].grp_seg_offsets = pool.alloc<int>(max_seg);
            bufs[s].grp_seg_ends = pool.alloc<int>(max_seg);
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
                                "Dense OVO active reference sub-batch");
        int sb_grp_items_actual =
            checked_int_product((size_t)n_all_grp, (size_t)sb_cols,
                                "Dense OVO active group sub-batch");
        int s = batch_idx % n_streams;
        cudaStream_t stream = streams[s];
        auto& buf = bufs[s];
        const float* ref_sub = ref_data + (size_t)col * n_ref;
        const float* grp_sub = grp_data + (size_t)col * n_all_grp;
        upload_linear_offsets(buf.ref_seg_offsets, sb_cols, n_ref, stream);
        cub_segmented_sortkeys(buf.ref_cub_temp, ref_cub_temp_bytes, ref_sub,
                               buf.ref_sorted, sb_ref_items_actual, sb_cols,
                               buf.ref_seg_offsets, buf.ref_seg_offsets + 1,
                               stream, "dense OVO ref segmented sort");
        ref_sub = buf.ref_sorted;

        OvoTierScratch sc{buf.ref_tie_sums,    buf.sub_rank_sums,
                          buf.sub_tie_corr,    buf.grp_sorted,
                          buf.grp_seg_offsets, buf.grp_seg_ends,
                          buf.grp_cub_temp};
        ovo_dispatch_tiers(ref_sub, grp_sub, grp_offsets, t1, sc,
                           d_sort_group_ids, n_sort_groups, grp_cub_temp_bytes,
                           sb_grp_items_actual, tpb_rank, n_ref, n_all_grp,
                           sb_cols, n_groups, compute_tie_corr,
                           /*analytic_zeros=*/false, stream);

        cuda_check(
            cudaMemcpy2DAsync(rank_sums + col, n_cols * sizeof(double),
                              buf.sub_rank_sums, sb_cols * sizeof(double),
                              sb_cols * sizeof(double), n_groups,
                              cudaMemcpyDeviceToDevice, stream),
            "dense OVO rank_sums D2D copy");
        if (compute_tie_corr) {
            cuda_check(
                cudaMemcpy2DAsync(tie_corr + col, n_cols * sizeof(double),
                                  buf.sub_tie_corr, sb_cols * sizeof(double),
                                  sb_cols * sizeof(double), n_groups,
                                  cudaMemcpyDeviceToDevice, stream),
                "dense OVO tie_corr D2D copy");
        }

        col += sb_cols;
        ++batch_idx;
    }

    sync_streams(streams, "dense OVO tiered rank");
}

template <typename T>
static void launch_ovo_rank_dense_host_streaming(
    const T* h_X, bool f_order, const int* h_ref_row_ids,
    const int* h_grp_row_ids, const int* h_grp_offsets, double* rank_sums,
    double* tie_corr, double* group_sums, double* group_sum_sq,
    double* group_nnz, int n_full_rows, int n_ref, int n_all_grp, int n_cols,
    int n_groups, int n_groups_stats, bool compute_tie_corr, bool compute_nnz,
    bool compute_stats, int sub_batch_cols) {
    if (n_cols == 0 || n_ref == 0 || n_all_grp == 0 || n_groups == 0) return;
    if (sub_batch_cols <= 0) sub_batch_cols = SUB_BATCH_COLS;
    if (compute_stats && n_groups_stats != n_groups + 1) {
        throw std::runtime_error(
            "dense OVO host stats require n_groups_stats == n_groups + 1");
    }
    if (h_grp_offsets[0] != 0 || h_grp_offsets[n_groups] != n_all_grp) {
        throw std::runtime_error(
            "dense OVO host group offsets must span n_all_grp");
    }

    auto tier_plan = make_ovo_tier_plan(h_grp_offsets, n_groups);
    int max_grp_size = tier_plan.max_grp_size;
    bool run_huge = compute_tie_corr && tier_plan.run_huge;

    std::vector<int> h_sort_group_ids;
    int n_sort_groups = n_groups;
    if (run_huge) {
        h_sort_group_ids = make_sort_group_ids(h_grp_offsets, n_groups,
                                               tier_plan.huge_skip_le);
        n_sort_groups = (int)h_sort_group_ids.size();
    }

    DenseColumnBatchPlan batches = plan_dense_column_batches(
        std::max(n_ref, n_all_grp), n_cols, sub_batch_cols, SAFE_BATCH_NNZ,
        "Dense host OVO sub-batch");
    sub_batch_cols = batches.sub_batch_cols;
    int n_streams = clamp_streams_by_cols(n_cols, sub_batch_cols);

    size_t sub_ref_items = (size_t)n_ref * sub_batch_cols;
    int sub_ref_items_i32 =
        checked_cub_items(sub_ref_items, "Dense host OVO reference sub-batch");
    size_t sub_grp_items = (size_t)n_all_grp * sub_batch_cols;
    int sub_grp_items_i32 =
        checked_cub_items(sub_grp_items, "Dense host OVO group sub-batch");
    constexpr bool fast_keys = std::is_same<T, float>::value;
    int n_stats_rows = n_groups + 1;

    size_t grp_cub_temp_bytes = 0;
    if (run_huge) {
        int max_grp_seg =
            checked_int_product((size_t)n_sort_groups, (size_t)sub_batch_cols,
                                "Dense host OVO group segment count");
        grp_cub_temp_bytes =
            cub_segmented_sortkeys_temp_bytes(sub_grp_items_i32, max_grp_seg);
    }
    size_t ref_cub_temp_bytes =
        cub_segmented_sortkeys_temp_bytes(sub_ref_items_i32, sub_batch_cols);

    {
        size_t native_items = sub_ref_items + sub_grp_items;
        size_t per_stream =
            native_items * sizeof(T) +
            (fast_keys ? 0 : native_items * sizeof(float)) +
            sub_ref_items * sizeof(float) +
            (size_t)(sub_batch_cols + 1) * sizeof(int) + ref_cub_temp_bytes +
            (run_huge ? sub_grp_items * sizeof(float) : 0) +
            (run_huge ? 2 * (size_t)n_sort_groups * sub_batch_cols * sizeof(int)
                      : 0) +
            (run_huge ? grp_cub_temp_bytes : 0) +
            (compute_tie_corr ? (size_t)sub_batch_cols * sizeof(double) : 0) +
            2 * (size_t)n_groups * sub_batch_cols * sizeof(double) +
            (compute_stats
                 ? 2 * (size_t)n_stats_rows * sub_batch_cols * sizeof(double)
                 : 0) +
            (compute_nnz
                 ? (size_t)n_stats_rows * sub_batch_cols * sizeof(double)
                 : 0);
        n_streams = clamp_streams_by_budget(n_streams, per_stream,
                                            rmm_available_device_bytes(0.8));
    }

    RmmScratchPool pool;
    PinnedRing<T, T> stage(n_streams, batches.max_items);
    ScopedCudaStreams streams(n_streams, cudaStreamDefault);

    int* d_grp_offsets = pool.alloc<int>(n_groups + 1);
    cuda_check(cudaMemcpy(d_grp_offsets, h_grp_offsets,
                          (size_t)(n_groups + 1) * sizeof(int),
                          cudaMemcpyHostToDevice),
               "dense host OVO offsets H2D");

    int* d_sort_group_ids = nullptr;
    if (run_huge) {
        d_sort_group_ids = pool.alloc<int>(h_sort_group_ids.size());
        cuda_check(cudaMemcpy(d_sort_group_ids, h_sort_group_ids.data(),
                              h_sort_group_ids.size() * sizeof(int),
                              cudaMemcpyHostToDevice),
                   "dense host OVO sort group ids H2D");
    }

    int* d_grp_codes = nullptr;
    if (compute_stats) {
        std::vector<int> h_grp_codes(n_all_grp, -1);
        for (int g = 0; g < n_groups; g++) {
            int begin = h_grp_offsets[g];
            int end = h_grp_offsets[g + 1];
            if (begin < 0 || end < begin || end > n_all_grp) {
                throw std::runtime_error(
                    "dense OVO host group offsets are invalid");
            }
            std::fill(h_grp_codes.begin() + begin, h_grp_codes.begin() + end,
                      g);
        }
        d_grp_codes = pool.alloc<int>(n_all_grp);
        cuda_check(
            cudaMemcpy(d_grp_codes, h_grp_codes.data(),
                       (size_t)n_all_grp * sizeof(int), cudaMemcpyHostToDevice),
            "dense host OVO group codes H2D");
    }

    struct StreamBuf {
        T* ref_native;
        T* grp_native;
        float* ref_f32;
        float* grp_f32;
        float* ref_sorted;
        int* ref_seg_offsets;
        uint8_t* ref_cub_temp;
        float* grp_sorted;
        int* grp_seg_offsets;
        int* grp_seg_ends;
        uint8_t* grp_cub_temp;
        double* ref_tie_sums;
        double* sub_rank_sums;
        double* sub_tie_corr;
        double* sub_group_sums;
        double* sub_group_sum_sq;
        double* sub_group_nnz;
    };
    std::vector<StreamBuf> bufs(n_streams);
    for (int s = 0; s < n_streams; s++) {
        bufs[s].ref_native = pool.alloc<T>(sub_ref_items);
        bufs[s].grp_native = pool.alloc<T>(sub_grp_items);
        bufs[s].ref_f32 =
            fast_keys ? nullptr : pool.alloc<float>(sub_ref_items);
        bufs[s].grp_f32 =
            fast_keys ? nullptr : pool.alloc<float>(sub_grp_items);
        bufs[s].ref_sorted = pool.alloc<float>(sub_ref_items);
        bufs[s].ref_seg_offsets = pool.alloc<int>(sub_batch_cols + 1);
        bufs[s].ref_cub_temp = pool.alloc<uint8_t>(ref_cub_temp_bytes);
        bufs[s].grp_cub_temp =
            run_huge ? pool.alloc<uint8_t>(grp_cub_temp_bytes) : nullptr;
        bufs[s].ref_tie_sums =
            compute_tie_corr ? pool.alloc<double>(sub_batch_cols) : nullptr;
        bufs[s].sub_rank_sums =
            pool.alloc<double>((size_t)n_groups * sub_batch_cols);
        bufs[s].sub_tie_corr =
            pool.alloc<double>((size_t)n_groups * sub_batch_cols);
        if (run_huge) {
            bufs[s].grp_sorted = pool.alloc<float>(sub_grp_items);
            int max_seg = checked_int_product((size_t)n_sort_groups,
                                              (size_t)sub_batch_cols,
                                              "Dense host OVO group segments");
            bufs[s].grp_seg_offsets = pool.alloc<int>(max_seg);
            bufs[s].grp_seg_ends = pool.alloc<int>(max_seg);
        } else {
            bufs[s].grp_sorted = nullptr;
            bufs[s].grp_seg_offsets = nullptr;
            bufs[s].grp_seg_ends = nullptr;
        }
        bufs[s].sub_group_sums =
            compute_stats
                ? pool.alloc<double>((size_t)n_stats_rows * sub_batch_cols)
                : nullptr;
        bufs[s].sub_group_sum_sq =
            compute_stats
                ? pool.alloc<double>((size_t)n_stats_rows * sub_batch_cols)
                : nullptr;
        bufs[s].sub_group_nnz =
            compute_nnz
                ? pool.alloc<double>((size_t)n_stats_rows * sub_batch_cols)
                : nullptr;
    }

    int tpb_rank =
        round_up_to_warp(std::min(max_grp_size, MAX_THREADS_PER_BLOCK));
    int tpb = UTIL_BLOCK_SIZE;
    cudaDeviceSynchronize();

    int col = 0;
    int batch_idx = 0;
    while (col < n_cols) {
        int sb_cols = std::min(sub_batch_cols, n_cols - col);
        int sb_ref_items_actual =
            checked_int_product((size_t)n_ref, (size_t)sb_cols,
                                "Dense host OVO active reference sub-batch");
        int sb_grp_items_actual =
            checked_int_product((size_t)n_all_grp, (size_t)sb_cols,
                                "Dense host OVO active group sub-batch");
        int s = batch_idx % n_streams;
        cudaStream_t stream = streams[s];
        auto& buf = bufs[s];
        stage.wait(s);
        T* h_ref_stage = stage.template get<0>(s);
        T* h_grp_stage = stage.template get<1>(s);

        host_materialize_dense_rows_window(h_X, f_order, n_full_rows, n_cols,
                                           h_ref_row_ids, n_ref, col, sb_cols,
                                           h_ref_stage);
        host_materialize_dense_rows_window(h_X, f_order, n_full_rows, n_cols,
                                           h_grp_row_ids, n_all_grp, col,
                                           sb_cols, h_grp_stage);

        cuda_check(cudaMemcpyAsync(buf.ref_native, h_ref_stage,
                                   (size_t)sb_ref_items_actual * sizeof(T),
                                   cudaMemcpyHostToDevice, stream),
                   "dense host OVO ref H2D");
        cuda_check(cudaMemcpyAsync(buf.grp_native, h_grp_stage,
                                   (size_t)sb_grp_items_actual * sizeof(T),
                                   cudaMemcpyHostToDevice, stream),
                   "dense host OVO group H2D");
        stage.record(s, stream);

        const float* ref_sub;
        const float* grp_sub;
        if (fast_keys) {
            ref_sub = reinterpret_cast<const float*>(buf.ref_native);
            grp_sub = reinterpret_cast<const float*>(buf.grp_native);
        } else {
            unsigned int ref_grid = (unsigned int)std::min<size_t>(
                ((size_t)sb_ref_items_actual + UTIL_BLOCK_SIZE - 1) /
                    UTIL_BLOCK_SIZE,
                65535u);
            dense_block_to_f32_kernel<T>
                <<<ref_grid, UTIL_BLOCK_SIZE, 0, stream>>>(
                    buf.ref_native, buf.ref_f32, n_ref, sb_cols, true);
            CUDA_CHECK_LAST_ERROR(dense_block_to_f32_kernel);
            unsigned int grp_grid = (unsigned int)std::min<size_t>(
                ((size_t)sb_grp_items_actual + UTIL_BLOCK_SIZE - 1) /
                    UTIL_BLOCK_SIZE,
                65535u);
            dense_block_to_f32_kernel<T>
                <<<grp_grid, UTIL_BLOCK_SIZE, 0, stream>>>(
                    buf.grp_native, buf.grp_f32, n_all_grp, sb_cols, true);
            CUDA_CHECK_LAST_ERROR(dense_block_to_f32_kernel);
            ref_sub = buf.ref_f32;
            grp_sub = buf.grp_f32;
        }

        upload_linear_offsets(buf.ref_seg_offsets, sb_cols, n_ref, stream);
        cub_segmented_sortkeys(buf.ref_cub_temp, ref_cub_temp_bytes, ref_sub,
                               buf.ref_sorted, sb_ref_items_actual, sb_cols,
                               buf.ref_seg_offsets, buf.ref_seg_offsets + 1,
                               stream, "dense host OVO ref segmented sort");
        ref_sub = buf.ref_sorted;

        OvoTierScratch sc{buf.ref_tie_sums,    buf.sub_rank_sums,
                          buf.sub_tie_corr,    buf.grp_sorted,
                          buf.grp_seg_offsets, buf.grp_seg_ends,
                          buf.grp_cub_temp};
        ovo_dispatch_tiers(ref_sub, grp_sub, d_grp_offsets, tier_plan, sc,
                           d_sort_group_ids, n_sort_groups, grp_cub_temp_bytes,
                           sb_grp_items_actual, tpb_rank, n_ref, n_all_grp,
                           sb_cols, n_groups, compute_tie_corr,
                           /*analytic_zeros=*/false, stream);

        cuda_check(
            cudaMemcpy2DAsync(rank_sums + col, n_cols * sizeof(double),
                              buf.sub_rank_sums, sb_cols * sizeof(double),
                              sb_cols * sizeof(double), n_groups,
                              cudaMemcpyDeviceToDevice, stream),
            "dense host OVO rank_sums D2D copy");
        if (compute_tie_corr) {
            cuda_check(
                cudaMemcpy2DAsync(tie_corr + col, n_cols * sizeof(double),
                                  buf.sub_tie_corr, sb_cols * sizeof(double),
                                  sb_cols * sizeof(double), n_groups,
                                  cudaMemcpyDeviceToDevice, stream),
                "dense host OVO tie_corr D2D copy");
        }

        if (compute_stats) {
            cuda_check(
                cudaMemsetAsync(buf.sub_group_sums, 0,
                                (size_t)n_stats_rows * sb_cols * sizeof(double),
                                stream),
                "dense host OVO group sums memset");
            cuda_check(
                cudaMemsetAsync(buf.sub_group_sum_sq, 0,
                                (size_t)n_stats_rows * sb_cols * sizeof(double),
                                stream),
                "dense host OVO group sumsq memset");
            if (compute_nnz) {
                cuda_check(cudaMemsetAsync(
                               buf.sub_group_nnz, 0,
                               (size_t)n_stats_rows * sb_cols * sizeof(double),
                               stream),
                           "dense host OVO group nnz memset");
            }
            dense_ovo_group_stats_kernel<T><<<sb_cols, tpb, 0, stream>>>(
                buf.ref_native, buf.grp_native, d_grp_codes, buf.sub_group_sums,
                buf.sub_group_sum_sq,
                compute_nnz ? buf.sub_group_nnz : buf.sub_group_sums, n_ref,
                n_all_grp, sb_cols, n_groups, compute_nnz);
            CUDA_CHECK_LAST_ERROR(dense_ovo_group_stats_kernel);
            scatter_cols_2d(group_sums + col, buf.sub_group_sums, n_stats_rows,
                            n_cols, sb_cols, stream);
            scatter_cols_2d(group_sum_sq + col, buf.sub_group_sum_sq,
                            n_stats_rows, n_cols, sb_cols, stream);
            if (compute_nnz) {
                scatter_cols_2d(group_nnz + col, buf.sub_group_nnz,
                                n_stats_rows, n_cols, sb_cols, stream);
            }
        }

        col += sb_cols;
        ++batch_idx;
    }

    sync_streams(streams, "dense host OVO streaming");
}

template <typename T, typename Device, typename HostArray, bool FOrder>
static void def_ovr_rank_dense_host_streaming(nb::module_& m) {
    m.def(
        "ovr_rank_dense_host_streaming",
        [](HostArray X, gpu_array_c<const int, Device> group_codes,
           gpu_array_c<double, Device> rank_sums,
           gpu_array_c<double, Device> tie_corr,
           gpu_array_c<double, Device> group_sums,
           gpu_array_c<double, Device> group_nnz,
           gpu_array_c<double, Device> total_sums,
           gpu_array_c<double, Device> total_nnz, int n_groups,
           bool compute_tie_corr, bool compute_nnz, bool compute_stats,
           bool compute_totals, int sub_batch_cols) {
            int n_rows = (int)X.shape(0);
            int n_cols = (int)X.shape(1);
            nb_require((int)group_codes.shape(0) == n_rows,
                       "ovr_rank_host: group_codes length must be n_rows");
            nb_require(
                (int)rank_sums.shape(0) == n_groups &&
                    (int)rank_sums.shape(1) == n_cols,
                "ovr_rank_host: rank_sums shape must be (n_groups, n_cols)");
            nb_require((int)tie_corr.shape(0) == n_cols,
                       "ovr_rank_host: tie_corr length must be n_cols");
            launch_ovr_rank_dense_host_streaming<T>(
                X.data(), FOrder, group_codes.data(), rank_sums.data(),
                tie_corr.data(), compute_stats ? group_sums.data() : nullptr,
                compute_nnz ? group_nnz.data() : nullptr,
                compute_totals ? total_sums.data() : nullptr,
                (compute_totals && compute_nnz) ? total_nnz.data() : nullptr,
                n_rows, n_cols, n_groups, compute_tie_corr, compute_nnz,
                compute_totals, sub_batch_cols);
        },
        "X"_a, "group_codes"_a, "rank_sums"_a, "tie_corr"_a, "group_sums"_a,
        "group_nnz"_a, "total_sums"_a, "total_nnz"_a, nb::kw_only(),
        "n_groups"_a, "compute_tie_corr"_a, "compute_nnz"_a, "compute_stats"_a,
        "compute_totals"_a, "sub_batch_cols"_a = SUB_BATCH_COLS);
}

template <typename T, typename Device, typename HostArray, bool FOrder>
static void def_ovo_rank_dense_host_streaming(nb::module_& m) {
    m.def(
        "ovo_rank_dense_host_streaming",
        [](HostArray X, host_array<const int> ref_row_ids,
           host_array<const int> grp_row_ids, host_array<const int> grp_offsets,
           gpu_array_c<double, Device> rank_sums,
           gpu_array_c<double, Device> tie_corr,
           gpu_array_c<double, Device> group_sums,
           gpu_array_c<double, Device> group_sum_sq,
           gpu_array_c<double, Device> group_nnz, int n_groups,
           bool compute_tie_corr, bool compute_nnz, bool compute_stats,
           int sub_batch_cols) {
            int n_full_rows = (int)X.shape(0);
            int n_cols = (int)X.shape(1);
            int n_ref = (int)ref_row_ids.shape(0);
            int n_all_grp = (int)grp_row_ids.shape(0);
            nb_require((int)grp_offsets.shape(0) == n_groups + 1,
                       "ovo_rank_host: grp_offsets length must be n_groups+1");
            nb_require(rank_sums.ndim() == 2 && tie_corr.ndim() == 2,
                       "ovo_rank_host: rank_sums/tie_corr must be 2D");
            nb_require((int)rank_sums.shape(0) == n_groups &&
                           (int)rank_sums.shape(1) == n_cols,
                       "ovo_rank_host: rank_sums shape must be "
                       "(n_groups, n_cols)");
            nb_require((int)tie_corr.shape(0) == n_groups &&
                           (int)tie_corr.shape(1) == n_cols,
                       "ovo_rank_host: tie_corr shape must be "
                       "(n_groups, n_cols)");
            int n_groups_stats = compute_stats ? (int)group_sums.shape(0) : 0;
            if (compute_stats) {
                nb_require(group_sums.ndim() == 2 && group_sum_sq.ndim() == 2,
                           "ovo_rank_host: stats outputs must be 2D");
                nb_require(n_groups_stats == n_groups + 1 &&
                               (int)group_sums.shape(1) == n_cols,
                           "ovo_rank_host: group_sums shape must be "
                           "(n_groups+1, n_cols)");
                nb_require((int)group_sum_sq.shape(0) == n_groups + 1 &&
                               (int)group_sum_sq.shape(1) == n_cols,
                           "ovo_rank_host: group_sum_sq shape must be "
                           "(n_groups+1, n_cols)");
                if (compute_nnz) {
                    nb_require(group_nnz.ndim() == 2 &&
                                   (int)group_nnz.shape(0) == n_groups + 1 &&
                                   (int)group_nnz.shape(1) == n_cols,
                               "ovo_rank_host: group_nnz shape must be "
                               "(n_groups+1, n_cols)");
                }
            }
            launch_ovo_rank_dense_host_streaming<T>(
                X.data(), FOrder, ref_row_ids.data(), grp_row_ids.data(),
                grp_offsets.data(), rank_sums.data(), tie_corr.data(),
                compute_stats ? group_sums.data() : nullptr,
                compute_stats ? group_sum_sq.data() : nullptr,
                compute_nnz ? group_nnz.data() : nullptr, n_full_rows, n_ref,
                n_all_grp, n_cols, n_groups, n_groups_stats, compute_tie_corr,
                compute_nnz, compute_stats, sub_batch_cols);
        },
        "X"_a, "ref_row_ids"_a, "grp_row_ids"_a, "grp_offsets"_a, "rank_sums"_a,
        "tie_corr"_a, "group_sums"_a, "group_sum_sq"_a, "group_nnz"_a,
        nb::kw_only(), "n_groups"_a, "compute_tie_corr"_a, "compute_nnz"_a,
        "compute_stats"_a, "sub_batch_cols"_a = SUB_BATCH_COLS);
}

template <typename Device>
void register_bindings(nb::module_& m) {
    m.doc() = "CUDA kernels for Wilcoxon rank-sum test";

    def_ovr_rank_dense_host_streaming<float, Device, host_array_c2<const float>,
                                      false>(m);
    def_ovr_rank_dense_host_streaming<float, Device, host_array_f2<const float>,
                                      true>(m);
    def_ovr_rank_dense_host_streaming<double, Device,
                                      host_array_c2<const double>, false>(m);
    def_ovr_rank_dense_host_streaming<double, Device,
                                      host_array_f2<const double>, true>(m);
    def_ovo_rank_dense_host_streaming<float, Device, host_array_c2<const float>,
                                      false>(m);
    def_ovo_rank_dense_host_streaming<float, Device, host_array_f2<const float>,
                                      true>(m);
    def_ovo_rank_dense_host_streaming<double, Device,
                                      host_array_c2<const double>, false>(m);
    def_ovo_rank_dense_host_streaming<double, Device,
                                      host_array_f2<const double>, true>(m);

    m.def(
        "ovo_rank_dense_tiered_unsorted_ref",
        [](gpu_array_f<const float, Device> ref_data,
           gpu_array_f<const float, Device> grp_data,
           gpu_array_c<const int, Device> grp_offsets,
           gpu_array_c<double, Device> rank_sums,
           gpu_array_c<double, Device> tie_corr, int n_ref, int n_all_grp,
           int n_cols, int n_groups, bool compute_tie_corr, int sub_batch_cols,
           std::uintptr_t stream) {
            nb_require(ref_data.ndim() == 2 && grp_data.ndim() == 2 &&
                           rank_sums.ndim() == 2 && tie_corr.ndim() == 2 &&
                           grp_offsets.ndim() == 1,
                       "ovo_rank: data/outputs must be 2D, grp_offsets 1D");
            nb_require((int)ref_data.shape(0) == n_ref &&
                           (int)ref_data.shape(1) == n_cols,
                       "ovo_rank: ref_data shape must be (n_ref, n_cols)");
            nb_require((int)grp_data.shape(0) == n_all_grp &&
                           (int)grp_data.shape(1) == n_cols,
                       "ovo_rank: grp_data shape must be (n_all_grp, n_cols)");
            nb_require((int)grp_offsets.shape(0) >= n_groups + 1,
                       "ovo_rank: grp_offsets length must be >= n_groups + 1");
            nb_require((int)rank_sums.shape(0) == n_groups &&
                           (int)rank_sums.shape(1) == n_cols,
                       "ovo_rank: rank_sums shape must be (n_groups, n_cols)");
            nb_require((int)tie_corr.shape(0) == n_groups &&
                           (int)tie_corr.shape(1) == n_cols,
                       "ovo_rank: tie_corr shape must be (n_groups, n_cols)");
            launch_ovo_rank_dense_tiered_unsorted_ref(
                ref_data.data(), grp_data.data(), grp_offsets.data(),
                rank_sums.data(), tie_corr.data(), n_ref, n_all_grp, n_cols,
                n_groups, compute_tie_corr, sub_batch_cols,
                (cudaStream_t)stream);
        },
        "ref_data"_a, "grp_data"_a, "grp_offsets"_a, "rank_sums"_a,
        "tie_corr"_a, nb::kw_only(), "n_ref"_a, "n_all_grp"_a, "n_cols"_a,
        "n_groups"_a, "compute_tie_corr"_a, "sub_batch_cols"_a = SUB_BATCH_COLS,
        "stream"_a = 0);

    m.def(
        "ovr_rank_dense_streaming",
        [](gpu_array_f<const float, Device> block,
           gpu_array_c<const int, Device> group_codes,
           gpu_array_c<double, Device> rank_sums,
           gpu_array_c<double, Device> tie_corr, int n_rows, int n_cols,
           int n_groups, bool compute_tie_corr, int sub_batch_cols,
           std::uintptr_t stream) {
            nb_require(block.ndim() == 2 && rank_sums.ndim() == 2 &&
                           group_codes.ndim() == 1 && tie_corr.ndim() == 1,
                       "ovr_rank: block/rank_sums 2D, group_codes/tie_corr 1D");
            nb_require(
                (int)block.shape(0) == n_rows && (int)block.shape(1) == n_cols,
                "ovr_rank: block shape must be (n_rows, n_cols)");
            nb_require((int)group_codes.shape(0) == n_rows,
                       "ovr_rank: group_codes length must be n_rows");
            nb_require((int)rank_sums.shape(0) == n_groups &&
                           (int)rank_sums.shape(1) == n_cols,
                       "ovr_rank: rank_sums shape must be (n_groups, n_cols)");
            nb_require((int)tie_corr.shape(0) == n_cols,
                       "ovr_rank: tie_corr length must be n_cols");
            launch_ovr_rank_dense_streaming(
                block.data(), group_codes.data(), rank_sums.data(),
                tie_corr.data(), n_rows, n_cols, n_groups, compute_tie_corr,
                sub_batch_cols, (cudaStream_t)stream);
        },
        "block"_a, "group_codes"_a, "rank_sums"_a, "tie_corr"_a, nb::kw_only(),
        "n_rows"_a, "n_cols"_a, "n_groups"_a, "compute_tie_corr"_a,
        "sub_batch_cols"_a = SUB_BATCH_COLS, "stream"_a = 0);
}

NB_MODULE(_wilcoxon_cuda, m) {
    REGISTER_GPU_BINDINGS(register_bindings, m);
}
