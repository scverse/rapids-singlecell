#include <cstdint>
#include <vector>

#include <cub/device/device_segmented_radix_sort.cuh>

#include "../nb_types.h"
#include "wilcoxon_fast_common.cuh"
#include "kernels_wilcoxon.cuh"
#include "wilcoxon_sparse_kernels.cuh"
#include "wilcoxon_ovr_kernels.cuh"
#include "wilcoxon_ovr_sparse.cuh"
#include "kernels_wilcoxon_ovo.cuh"
#include "wilcoxon_ovo_kernels.cuh"
#include "wilcoxon_ovo_device_sparse.cuh"
#include "wilcoxon_ovo_host_sparse.cuh"

using namespace nb::literals;

template <typename Array>
static bool has_matrix_shape(const Array& array, int n_rows, int n_cols) {
    return array.ndim() == 2 && (int)array.shape(0) == n_rows &&
           (int)array.shape(1) == n_cols;
}

template <typename IndexT, typename IndptrT>
static void def_csr_row_boundaries_host(nb::module_& m) {
    m.def(
        "csr_row_boundaries_host",
        [](host_array<const IndexT> h_indices,
           host_array<const IndptrT> h_indptr, host_array<const int> h_col_cuts,
           host_array_c2<IndptrT> h_boundaries, int n_cols) {
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
                        lo = std::lower_bound(lo, hi,
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

template <typename InT, typename IndexT, typename IndptrT, typename Device>
static void def_ovr_sparse_csr_host(nb::module_& m) {
    m.def(
        "ovr_sparse_csr_host",
        [](host_array<const InT> h_data, host_array<const IndexT> h_indices,
           host_array<const IndptrT> h_indptr,
           host_array<const IndptrT> h_row_starts,
           host_array<const IndptrT> h_row_stops,
           host_array<const int> h_group_codes,
           host_array<double> h_group_sizes,
           gpu_array_c<double, Device> d_rank_sums,
           gpu_array_c<double, Device> d_tie_corr,
           gpu_array_c<double, Device> d_group_sums,
           gpu_array_c<double, Device> d_group_nnz,
           gpu_array_c<double, Device> d_total_sums,
           gpu_array_c<double, Device> d_total_nnz, int n_cols,
           bool compute_tie_corr, bool compute_nnz, bool compute_totals,
           int col_start, int col_stop, int sub_batch_cols) {
            nb_require((int)h_indptr.shape(0) >= 1,
                       "ovr_sparse_csr_host: indptr must not be empty");
            int n_rows = (int)h_indptr.shape(0) - 1;
            int n_groups = (int)h_group_sizes.shape(0);
            if (col_stop < 0) col_stop = n_cols;
            nb_require(n_cols >= 0,
                       "ovr_sparse_csr_host: n_cols must be nonnegative");
            nb_require(
                col_start >= 0 && col_start <= col_stop && col_stop <= n_cols,
                "ovr_sparse_csr_host: invalid input column "
                "range");
            int local_cols = col_stop - col_start;
            nb_require((int)h_data.shape(0) == (int)h_indices.shape(0),
                       "ovr_sparse_csr_host: data and indices lengths "
                       "must match");
            nb_require((int)h_row_starts.shape(0) == n_rows &&
                           (int)h_row_stops.shape(0) == n_rows,
                       "ovr_sparse_csr_host: row span lengths must be "
                       "n_rows");
            nb_require((int)h_group_codes.shape(0) == n_rows,
                       "ovr_sparse_csr_host: group_codes length must "
                       "be n_rows");
            nb_require(has_matrix_shape(d_rank_sums, n_groups, local_cols),
                       "ovr_sparse_csr_host: rank_sums shape must be "
                       "(n_groups, local_cols)");
            nb_require(d_tie_corr.ndim() == 1 &&
                           (int)d_tie_corr.shape(0) == local_cols,
                       "ovr_sparse_csr_host: tie_corr length must be "
                       "local_cols");
            nb_require(has_matrix_shape(d_group_sums, n_groups, local_cols),
                       "ovr_sparse_csr_host: group_sums shape must be "
                       "(n_groups, local_cols)");
            if (compute_nnz) {
                nb_require(has_matrix_shape(d_group_nnz, n_groups, local_cols),
                           "ovr_sparse_csr_host: group_nnz shape must be "
                           "(n_groups, local_cols)");
            }
            if (compute_totals) {
                nb_require(has_matrix_shape(d_total_sums, 1, local_cols),
                           "ovr_sparse_csr_host: total_sums shape must "
                           "be (1, local_cols)");
                if (compute_nnz) {
                    nb_require(has_matrix_shape(d_total_nnz, 1, local_cols),
                               "ovr_sparse_csr_host: total_nnz shape must be "
                               "(1, local_cols)");
                }
            }
            if (sub_batch_cols <= 0) sub_batch_cols = SUB_BATCH_COLS;
            ovr_sparse_csr_host_range_impl<InT, IndexT, IndptrT>(
                h_data.data(), h_indices.data(), h_indptr.data(),
                h_group_codes.data(), h_group_sizes.data(), d_rank_sums.data(),
                d_tie_corr.data(), d_group_sums.data(), d_group_nnz.data(),
                d_total_sums.data(), d_total_nnz.data(), n_rows, col_start,
                local_cols, n_groups, compute_tie_corr, compute_nnz,
                compute_totals, sub_batch_cols, h_row_starts.data(),
                h_row_stops.data());
        },
        "h_data"_a, "h_indices"_a, "h_indptr"_a, "h_row_starts"_a,
        "h_row_stops"_a, "h_group_codes"_a, "h_group_sizes"_a, "d_rank_sums"_a,
        "d_tie_corr"_a, "d_group_sums"_a, "d_group_nnz"_a, "d_total_sums"_a,
        "d_total_nnz"_a, nb::kw_only(), "n_cols"_a, "compute_tie_corr"_a,
        "compute_nnz"_a = true, "compute_totals"_a = false, "col_start"_a = 0,
        "col_stop"_a = -1, "sub_batch_cols"_a = SUB_BATCH_COLS,
        nb::call_guard<nb::gil_scoped_release>());
}

template <typename InT, typename IndexT, typename IndptrT, typename Device>
static void def_ovo_streaming_csr_host(nb::module_& m) {
    m.def(
        "ovo_streaming_csr_host",
        [](host_array<const InT> h_data, host_array<const IndexT> h_indices,
           host_array<const IndptrT> h_row_starts,
           host_array<const IndptrT> h_row_stops,
           host_array<const int> h_ref_row_ids,
           host_array<const int> h_grp_row_ids,
           host_array<const int> h_grp_offsets,
           gpu_array_c<double, Device> d_rank_sums,
           gpu_array_c<double, Device> d_tie_corr,
           gpu_array_c<double, Device> d_group_sums,
           gpu_array_c<double, Device> d_group_nnz, int n_cols,
           bool compute_tie_corr, bool compute_nnz, int col_start, int col_stop,
           int sub_batch_cols, bool analytic_zeros) {
            int n_full_rows = (int)h_row_starts.shape(0);
            int n_ref = (int)h_ref_row_ids.shape(0);
            int n_all_grp = (int)h_grp_row_ids.shape(0);
            int n_test = (int)d_rank_sums.shape(0);
            nb_require(d_group_sums.ndim() == 2,
                       "ovo_streaming_csr_host: group_sums must be 2D");
            int n_groups_stats = (int)d_group_sums.shape(0);
            if (col_stop < 0) col_stop = n_cols;
            nb_require(n_cols >= 0,
                       "ovo_streaming_csr_host: n_cols must be nonnegative");
            nb_require(n_groups_stats >= n_test + 1,
                       "ovo_streaming_csr_host: n_groups_stats must be at "
                       "least n_test+1");
            nb_require(
                col_start >= 0 && col_start <= col_stop && col_stop <= n_cols,
                "ovo_streaming_csr_host: invalid input column "
                "range");
            int local_cols = col_stop - col_start;
            nb_require((int)h_data.shape(0) == (int)h_indices.shape(0),
                       "ovo_streaming_csr_host: data and indices "
                       "lengths must match");
            nb_require((int)h_row_stops.shape(0) == n_full_rows,
                       "ovo_streaming_csr_host: row span lengths must match");
            nb_require((int)h_grp_offsets.shape(0) == n_test + 1,
                       "ovo_streaming_csr_host: grp_offsets length "
                       "must be n_test+1");
            nb_require(has_matrix_shape(d_rank_sums, n_test, local_cols),
                       "ovo_streaming_csr_host: rank_sums shape must be "
                       "(n_test, local_cols)");
            nb_require(has_ovo_tie_shape(d_tie_corr, n_test, local_cols,
                                         compute_tie_corr),
                       "ovo_streaming_csr_host: tie_corr must be "
                       "(n_test, local_cols) when enabled or length 1 when "
                       "disabled");
            nb_require(
                has_matrix_shape(d_group_sums, n_groups_stats, local_cols),
                "ovo_streaming_csr_host: group_sums shape must be "
                "(n_groups_stats, local_cols)");
            if (compute_nnz) {
                nb_require(
                    has_matrix_shape(d_group_nnz, n_groups_stats, local_cols),
                    "ovo_streaming_csr_host: group_nnz shape must be "
                    "(n_groups_stats, local_cols)");
            }
            if (sub_batch_cols <= 0) sub_batch_cols = SUB_BATCH_COLS;
            ovo_streaming_csr_host_impl<InT, IndexT, IndptrT>(
                h_data.data(), h_indices.data(), h_ref_row_ids.data(), n_ref,
                h_grp_row_ids.data(), h_grp_offsets.data(), n_all_grp, n_test,
                d_rank_sums.data(),
                compute_tie_corr ? d_tie_corr.data() : nullptr,
                d_group_sums.data(), d_group_nnz.data(), col_start, local_cols,
                n_groups_stats, compute_tie_corr, compute_nnz, sub_batch_cols,
                analytic_zeros, h_row_starts.data(), h_row_stops.data());
        },
        "h_data"_a, "h_indices"_a, "h_row_starts"_a, "h_row_stops"_a,
        "h_ref_row_ids"_a, "h_grp_row_ids"_a, "h_grp_offsets"_a,
        "d_rank_sums"_a, "d_tie_corr"_a, "d_group_sums"_a, "d_group_nnz"_a,
        nb::kw_only(), "n_cols"_a, "compute_tie_corr"_a, "compute_nnz"_a = true,
        "col_start"_a = 0, "col_stop"_a = -1,
        "sub_batch_cols"_a = SUB_BATCH_COLS, "analytic_zeros"_a = false,
        nb::call_guard<nb::gil_scoped_release>());
}

template <typename Device>
void register_sparse_bindings(nb::module_& m) {
    m.doc() = "Sparse-native host Wilcoxon CUDA kernels";

#define RSC_OVR_SPARSE_DEVICE_BINDING(NAME, IMPL, IndexCType, IndptrCType)    \
    m.def(                                                                    \
        NAME,                                                                 \
        [](gpu_array_c<const float, Device> data,                             \
           gpu_array_c<const IndexCType, Device> indices,                     \
           gpu_array_c<const IndptrCType, Device> indptr,                     \
           gpu_array_c<const int, Device> group_codes,                        \
           gpu_array_c<const double, Device> group_sizes,                     \
           gpu_array_c<double, Device> rank_sums,                             \
           gpu_array_c<double, Device> tie_corr, bool compute_tie_corr,       \
           int sub_batch_cols) {                                              \
            int n_rows = (int)group_codes.shape(0);                           \
            int n_groups = (int)group_sizes.shape(0);                         \
            int n_cols = (int)rank_sums.shape(1);                             \
            if (sub_batch_cols <= 0) sub_batch_cols = SUB_BATCH_COLS;         \
            IMPL(data.data(), indices.data(), indptr.data(),                  \
                 group_codes.data(), group_sizes.data(), rank_sums.data(),    \
                 tie_corr.data(), n_rows, n_cols, n_groups, compute_tie_corr, \
                 sub_batch_cols);                                             \
        },                                                                    \
        "data"_a, "indices"_a, "indptr"_a, "group_codes"_a, "group_sizes"_a,  \
        "rank_sums"_a, "tie_corr"_a, nb::kw_only(), "compute_tie_corr"_a,     \
        "sub_batch_cols"_a = SUB_BATCH_COLS)

    RSC_OVR_SPARSE_DEVICE_BINDING("ovr_sparse_csc_device",
                                  ovr_sparse_csc_streaming_impl, int, int);
    RSC_OVR_SPARSE_DEVICE_BINDING("ovr_sparse_csc_device",
                                  ovr_sparse_csc_streaming_impl, int64_t,
                                  int64_t);
    RSC_OVR_SPARSE_DEVICE_BINDING("ovr_sparse_csr_device",
                                  ovr_sparse_csr_streaming_impl, int, int);
    RSC_OVR_SPARSE_DEVICE_BINDING("ovr_sparse_csr_device",
                                  ovr_sparse_csr_streaming_impl, int64_t,
                                  int64_t);
#undef RSC_OVR_SPARSE_DEVICE_BINDING

#define RSC_OVR_SPARSE_CSC_HOST_BINDING(NAME, InT, IndexT, IndptrT)           \
    m.def(                                                                    \
        NAME,                                                                 \
        [](host_array<const InT> h_data, host_array<const IndexT> h_indices,  \
           host_array<const IndptrT> h_indptr,                                \
           host_array<const int> h_group_codes,                               \
           host_array<double> h_group_sizes,                                  \
           gpu_array_c<double, Device> d_rank_sums,                           \
           gpu_array_c<double, Device> d_tie_corr,                            \
           gpu_array_c<double, Device> d_group_sums,                          \
           gpu_array_c<double, Device> d_group_nnz,                           \
           gpu_array_c<double, Device> d_total_sums,                          \
           gpu_array_c<double, Device> d_total_nnz, bool compute_tie_corr,    \
           bool compute_nnz, bool compute_totals, int sub_batch_cols) {       \
            int n_rows = (int)h_group_codes.shape(0);                         \
            int n_cols = (int)h_indptr.shape(0) - 1;                          \
            int n_groups = (int)h_group_sizes.shape(0);                       \
            if (sub_batch_cols <= 0) sub_batch_cols = SUB_BATCH_COLS;         \
            ovr_sparse_csc_host_streaming_impl<InT, IndexT, IndptrT>(         \
                h_data.data(), h_indices.data(), h_indptr.data(),             \
                h_group_codes.data(), h_group_sizes.data(),                   \
                d_rank_sums.data(), d_tie_corr.data(), d_group_sums.data(),   \
                d_group_nnz.data(), d_total_sums.data(), d_total_nnz.data(),  \
                n_rows, n_cols, n_groups, compute_tie_corr, compute_nnz,      \
                compute_totals, sub_batch_cols);                              \
        },                                                                    \
        "h_data"_a, "h_indices"_a, "h_indptr"_a, "h_group_codes"_a,           \
        "h_group_sizes"_a, "d_rank_sums"_a, "d_tie_corr"_a, "d_group_sums"_a, \
        "d_group_nnz"_a, "d_total_sums"_a, "d_total_nnz"_a, nb::kw_only(),    \
        "compute_tie_corr"_a, "compute_nnz"_a = true,                         \
        "compute_totals"_a = false, "sub_batch_cols"_a = SUB_BATCH_COLS,      \
        nb::call_guard<nb::gil_scoped_release>())

    RSC_OVR_SPARSE_CSC_HOST_BINDING("ovr_sparse_csc_host", float, int, int);
    RSC_OVR_SPARSE_CSC_HOST_BINDING("ovr_sparse_csc_host", double, int, int);
    RSC_OVR_SPARSE_CSC_HOST_BINDING("ovr_sparse_csc_host", float, int64_t,
                                    int64_t);
    RSC_OVR_SPARSE_CSC_HOST_BINDING("ovr_sparse_csc_host", double, int64_t,
                                    int64_t);
#undef RSC_OVR_SPARSE_CSC_HOST_BINDING

#define RSC_OVO_DEVICE_BINDING(NAME, IMPL, IndexCType, IndptrCType)           \
    m.def(                                                                    \
        NAME,                                                                 \
        [](gpu_array_c<const float, Device> data,                             \
           gpu_array_c<const IndexCType, Device> indices,                     \
           gpu_array_c<const IndptrCType, Device> indptr,                     \
           gpu_array_c<const int, Device> ref_rows,                           \
           gpu_array_c<const int, Device> grp_rows,                           \
           gpu_array_c<const int, Device> grp_offsets,                        \
           gpu_array_c<double, Device> rank_sums,                             \
           gpu_array_c<double, Device> tie_corr, int n_ref, int n_all_grp,    \
           bool compute_tie_corr, int sub_batch_cols) {                       \
            int n_groups = (int)grp_offsets.shape(0) - 1;                     \
            int n_cols = (int)rank_sums.shape(1);                             \
            nb_require(has_ovo_tie_shape(tie_corr, n_groups, n_cols,          \
                                         compute_tie_corr),                   \
                       "OVO sparse device tie_corr must be (n_groups, "       \
                       "n_cols) when enabled or length 1 when disabled");     \
            if (sub_batch_cols <= 0) sub_batch_cols = SUB_BATCH_COLS;         \
            IMPL(data.data(), indices.data(), indptr.data(), ref_rows.data(), \
                 grp_rows.data(), grp_offsets.data(), rank_sums.data(),       \
                 compute_tie_corr ? tie_corr.data() : nullptr, n_ref,         \
                 n_all_grp, n_cols, n_groups, compute_tie_corr,               \
                 sub_batch_cols);                                             \
        },                                                                    \
        "data"_a, "indices"_a, "indptr"_a, "ref_rows"_a, "grp_rows"_a,        \
        "grp_offsets"_a, "rank_sums"_a, "tie_corr"_a, nb::kw_only(),          \
        "n_ref"_a, "n_all_grp"_a, "compute_tie_corr"_a,                       \
        "sub_batch_cols"_a = SUB_BATCH_COLS)

    RSC_OVO_DEVICE_BINDING("ovo_streaming_csc_device", ovo_streaming_csc_impl,
                           int, int);
    RSC_OVO_DEVICE_BINDING("ovo_streaming_csc_device", ovo_streaming_csc_impl,
                           int64_t, int64_t);
    RSC_OVO_DEVICE_BINDING("ovo_streaming_csr_device", ovo_streaming_csr_impl,
                           int, int);
    RSC_OVO_DEVICE_BINDING("ovo_streaming_csr_device", ovo_streaming_csr_impl,
                           int64_t, int64_t);
#undef RSC_OVO_DEVICE_BINDING

#define RSC_OVO_CSC_HOST_BINDING(NAME, InT, IndexT, IndptrT)                  \
    m.def(                                                                    \
        NAME,                                                                 \
        [](host_array<const InT> h_data, host_array<const IndexT> h_indices,  \
           host_array<const IndptrT> h_indptr,                                \
           host_array<const int> h_ref_row_map,                               \
           host_array<const int> h_grp_row_map,                               \
           host_array<const int> h_grp_offsets,                               \
           host_array<const int> h_stats_codes,                               \
           gpu_array_c<double, Device> d_rank_sums,                           \
           gpu_array_c<double, Device> d_tie_corr,                            \
           gpu_array_c<double, Device> d_group_sums,                          \
           gpu_array_c<double, Device> d_group_nnz, int n_ref, int n_all_grp, \
           bool compute_tie_corr, bool compute_nnz, int sub_batch_cols,       \
           bool analytic_zeros) {                                             \
            int n_rows = (int)h_ref_row_map.shape(0);                         \
            int n_cols = (int)h_indptr.shape(0) - 1;                          \
            int n_groups = (int)h_grp_offsets.shape(0) - 1;                   \
            int n_groups_stats = (int)d_group_sums.shape(0);                  \
            nb_require(has_ovo_tie_shape(d_tie_corr, n_groups, n_cols,        \
                                         compute_tie_corr),                   \
                       "OVO sparse CSC host tie_corr must be (n_groups, "     \
                       "n_cols) when enabled or length 1 when disabled");     \
            if (sub_batch_cols <= 0) sub_batch_cols = SUB_BATCH_COLS;         \
            ovo_streaming_csc_host_impl<InT, IndexT, IndptrT>(                \
                h_data.data(), h_indices.data(), h_indptr.data(),             \
                h_ref_row_map.data(), h_grp_row_map.data(),                   \
                h_grp_offsets.data(), h_stats_codes.data(),                   \
                d_rank_sums.data(),                                           \
                compute_tie_corr ? d_tie_corr.data() : nullptr,               \
                d_group_sums.data(), d_group_nnz.data(), n_ref, n_all_grp,    \
                n_rows, n_cols, n_groups, n_groups_stats, compute_tie_corr,   \
                compute_nnz, sub_batch_cols, analytic_zeros);                 \
        },                                                                    \
        "h_data"_a, "h_indices"_a, "h_indptr"_a, "h_ref_row_map"_a,           \
        "h_grp_row_map"_a, "h_grp_offsets"_a, "h_stats_codes"_a,              \
        "d_rank_sums"_a, "d_tie_corr"_a, "d_group_sums"_a, "d_group_nnz"_a,   \
        nb::kw_only(), "n_ref"_a, "n_all_grp"_a, "compute_tie_corr"_a,        \
        "compute_nnz"_a = true, "sub_batch_cols"_a = SUB_BATCH_COLS,          \
        "analytic_zeros"_a = false, nb::call_guard<nb::gil_scoped_release>())

    RSC_OVO_CSC_HOST_BINDING("ovo_streaming_csc_host", float, int, int);
    RSC_OVO_CSC_HOST_BINDING("ovo_streaming_csc_host", double, int, int);
    RSC_OVO_CSC_HOST_BINDING("ovo_streaming_csc_host", float, int64_t, int64_t);
    RSC_OVO_CSC_HOST_BINDING("ovo_streaming_csc_host", double, int64_t,
                             int64_t);
#undef RSC_OVO_CSC_HOST_BINDING

    def_ovr_sparse_csr_host<float, int, int, Device>(m);
    def_ovr_sparse_csr_host<double, int, int, Device>(m);
    def_ovr_sparse_csr_host<float, int64_t, int64_t, Device>(m);
    def_ovr_sparse_csr_host<double, int64_t, int64_t, Device>(m);

    def_ovo_streaming_csr_host<float, int, int, Device>(m);
    def_ovo_streaming_csr_host<double, int, int, Device>(m);
    def_ovo_streaming_csr_host<float, int64_t, int64_t, Device>(m);
    def_ovo_streaming_csr_host<double, int64_t, int64_t, Device>(m);
}

NB_MODULE(_wilcoxon_sparse_cuda, m) {
    m.def("_set_host_worker_limit", &set_host_worker_limit, "limit"_a);
    def_csr_row_boundaries_host<int, int>(m);
    def_csr_row_boundaries_host<int64_t, int64_t>(m);
    REGISTER_GPU_BINDINGS(register_sparse_bindings, m);
}
