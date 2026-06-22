#include <cuda_runtime.h>
#include <cub/device/device_segmented_radix_sort.cuh>

#include <algorithm>
#include <limits>
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

    int n_streams = N_STREAMS;
    if (n_cols < n_streams * sub_batch_cols) {
        n_streams = (n_cols + sub_batch_cols - 1) / sub_batch_cols;
    }

    size_t sub_items = (size_t)n_rows * sub_batch_cols;
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

    for (int s = 0; s < n_streams; ++s) {
        cudaError_t err = cudaStreamSynchronize(streams[s]);
        if (err != cudaSuccess) {
            throw std::runtime_error(
                std::string("CUDA error in dense OVR streaming rank: ") +
                cudaGetErrorString(err));
        }
    }
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
                           sb_cols, n_groups, compute_tie_corr, stream);

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

    for (int s = 0; s < n_streams; ++s) {
        cudaError_t err = cudaStreamSynchronize(streams[s]);
        if (err != cudaSuccess) {
            throw std::runtime_error(
                std::string("CUDA error in dense OVO tiered rank: ") +
                cudaGetErrorString(err));
        }
    }
}

template <typename Device>
void register_bindings(nb::module_& m) {
    m.doc() = "CUDA kernels for Wilcoxon rank-sum test";

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
