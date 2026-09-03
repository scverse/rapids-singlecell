#include <cuda_runtime.h>
#include <nanobind/stl/optional.h>

#include <algorithm>
#include <optional>

#include "../nb_types.h"

#include "kernels_autocorr.cuh"

using namespace nb::literals;

constexpr int DENSE_BLOCK_DIM = 8;
constexpr int SPARSE_BLOCK_SIZE = 1024;
constexpr int ELEMENTWISE_BLOCK_SIZE = 32;

template <typename T>
static inline void launch_morans_dense(const T* data_centered,
                                       const int* adj_row_ptr,
                                       const int* adj_col_ind,
                                       const T* adj_data, T* num,
                                       size_t n_samples, size_t n_features,
                                       cudaStream_t stream) {
    dim3 block(DENSE_BLOCK_DIM, DENSE_BLOCK_DIM);
    dim3 grid(
        strided_grid(static_cast<long long>(n_features), DENSE_BLOCK_DIM),
        strided_grid_y(static_cast<long long>(n_samples), DENSE_BLOCK_DIM));
    morans_I_num_dense_kernel<<<grid, block, 0, stream>>>(
        data_centered, adj_row_ptr, adj_col_ind, adj_data, num, n_samples,
        n_features);
    CUDA_CHECK_LAST_ERROR(morans_I_num_dense_kernel);
}

template <typename T, typename AdjIdxT, typename DataIdxT>
static inline void launch_morans_sparse(
    const AdjIdxT* adj_row_ptr, const AdjIdxT* adj_col_ind, const T* adj_data,
    const DataIdxT* data_row_ptr, const DataIdxT* data_col_ind,
    const T* data_values, int n_samples, int n_features, const T* mean_array,
    T* num, cudaStream_t stream) {
    dim3 block(SPARSE_BLOCK_SIZE);
    dim3 grid(n_samples);
    morans_I_num_sparse_kernel<T, AdjIdxT, DataIdxT>
        <<<grid, block, 0, stream>>>(adj_row_ptr, adj_col_ind, adj_data,
                                     data_row_ptr, data_col_ind, data_values,
                                     n_samples, n_features, mean_array, num);
    CUDA_CHECK_LAST_ERROR(morans_I_num_sparse_kernel);
}

template <typename T>
static inline void launch_gearys_dense(const T* data, const int* adj_row_ptr,
                                       const int* adj_col_ind,
                                       const T* adj_data, T* num,
                                       size_t n_samples, size_t n_features,
                                       cudaStream_t stream) {
    dim3 block(DENSE_BLOCK_DIM, DENSE_BLOCK_DIM);
    dim3 grid(
        strided_grid(static_cast<long long>(n_features), DENSE_BLOCK_DIM),
        strided_grid_y(static_cast<long long>(n_samples), DENSE_BLOCK_DIM));
    gearys_C_num_dense_kernel<<<grid, block, 0, stream>>>(
        data, adj_row_ptr, adj_col_ind, adj_data, num, n_samples, n_features);
    CUDA_CHECK_LAST_ERROR(gearys_C_num_dense_kernel);
}

template <typename T, typename AdjIdxT, typename DataIdxT>
static inline void launch_gearys_sparse(
    const AdjIdxT* adj_row_ptr, const AdjIdxT* adj_col_ind, const T* adj_data,
    const DataIdxT* data_row_ptr, const DataIdxT* data_col_ind,
    const T* data_values, int n_samples, int n_features, T* num,
    cudaStream_t stream) {
    dim3 block(SPARSE_BLOCK_SIZE);
    dim3 grid(n_samples);
    gearys_C_num_sparse_kernel<T, AdjIdxT, DataIdxT>
        <<<grid, block, 0, stream>>>(adj_row_ptr, adj_col_ind, adj_data,
                                     data_row_ptr, data_col_ind, data_values,
                                     n_samples, n_features, num);
    CUDA_CHECK_LAST_ERROR(gearys_C_num_sparse_kernel);
}

template <typename T, typename IdxT>
static inline void launch_pre_den_sparse(const IdxT* data_col_ind,
                                         const T* data_values, long long nnz,
                                         const T* mean_array, T* den,
                                         int* counter, cudaStream_t stream) {
    dim3 block(ELEMENTWISE_BLOCK_SIZE);
    dim3 grid(strided_grid(nnz, ELEMENTWISE_BLOCK_SIZE));
    pre_den_sparse_kernel<T, IdxT><<<grid, block, 0, stream>>>(
        data_col_ind, data_values, nnz, mean_array, den, counter);
    CUDA_CHECK_LAST_ERROR(pre_den_sparse_kernel);
}

template <typename T, typename Device>
void def_morans_dense(nb::module_& m) {
    m.def(
        "morans_dense",
        [](gpu_array_c<const T, Device> data_centered,
           gpu_array_c<const int, Device> adj_row_ptr,
           gpu_array_c<const int, Device> adj_col_ind,
           gpu_array_c<const T, Device> adj_data, gpu_array_c<T, Device> num,
           size_t n_samples, size_t n_features, std::uintptr_t stream) {
            launch_morans_dense(data_centered.data(), adj_row_ptr.data(),
                                adj_col_ind.data(), adj_data.data(), num.data(),
                                n_samples, n_features, (cudaStream_t)stream);
        },
        "data_centered"_a, nb::kw_only(), "adj_row_ptr"_a, "adj_col_ind"_a,
        "adj_data"_a, "num"_a, "n_samples"_a, "n_features"_a, "stream"_a = 0);
}

template <typename T, typename AdjIdxT, typename DataIdxT, typename Device>
void def_morans_sparse(nb::module_& m) {
    m.def(
        "morans_sparse",
        [](gpu_array_c<const AdjIdxT, Device> adj_row_ptr,
           gpu_array_c<const AdjIdxT, Device> adj_col_ind,
           gpu_array_c<const T, Device> adj_data,
           gpu_array_c<const DataIdxT, Device> data_row_ptr,
           gpu_array_c<const DataIdxT, Device> data_col_ind,
           gpu_array_c<const T, Device> data_values, int n_samples,
           int n_features, gpu_array_c<const T, Device> mean_array,
           gpu_array_c<T, Device> num, std::uintptr_t stream) {
            launch_morans_sparse<T, AdjIdxT, DataIdxT>(
                adj_row_ptr.data(), adj_col_ind.data(), adj_data.data(),
                data_row_ptr.data(), data_col_ind.data(), data_values.data(),
                n_samples, n_features, mean_array.data(), num.data(),
                (cudaStream_t)stream);
        },
        "adj_row_ptr"_a, "adj_col_ind"_a, "adj_data"_a, nb::kw_only(),
        "data_row_ptr"_a, "data_col_ind"_a, "data_values"_a, "n_samples"_a,
        "n_features"_a, "mean_array"_a, "num"_a, "stream"_a = 0);
}

template <typename T, typename Device>
void def_gearys_dense(nb::module_& m) {
    m.def(
        "gearys_dense",
        [](gpu_array_c<const T, Device> data,
           gpu_array_c<const int, Device> adj_row_ptr,
           gpu_array_c<const int, Device> adj_col_ind,
           gpu_array_c<const T, Device> adj_data, gpu_array_c<T, Device> num,
           size_t n_samples, size_t n_features, std::uintptr_t stream) {
            launch_gearys_dense(data.data(), adj_row_ptr.data(),
                                adj_col_ind.data(), adj_data.data(), num.data(),
                                n_samples, n_features, (cudaStream_t)stream);
        },
        "data"_a, nb::kw_only(), "adj_row_ptr"_a, "adj_col_ind"_a, "adj_data"_a,
        "num"_a, "n_samples"_a, "n_features"_a, "stream"_a = 0);
}

template <typename T, typename AdjIdxT, typename DataIdxT, typename Device>
void def_gearys_sparse(nb::module_& m) {
    m.def(
        "gearys_sparse",
        [](gpu_array_c<const AdjIdxT, Device> adj_row_ptr,
           gpu_array_c<const AdjIdxT, Device> adj_col_ind,
           gpu_array_c<const T, Device> adj_data,
           gpu_array_c<const DataIdxT, Device> data_row_ptr,
           gpu_array_c<const DataIdxT, Device> data_col_ind,
           gpu_array_c<const T, Device> data_values, int n_samples,
           int n_features, gpu_array_c<T, Device> num, std::uintptr_t stream) {
            launch_gearys_sparse<T, AdjIdxT, DataIdxT>(
                adj_row_ptr.data(), adj_col_ind.data(), adj_data.data(),
                data_row_ptr.data(), data_col_ind.data(), data_values.data(),
                n_samples, n_features, num.data(), (cudaStream_t)stream);
        },
        "adj_row_ptr"_a, "adj_col_ind"_a, "adj_data"_a, nb::kw_only(),
        "data_row_ptr"_a, "data_col_ind"_a, "data_values"_a, "n_samples"_a,
        "n_features"_a, "num"_a, "stream"_a = 0);
}

template <typename T, typename IdxT, typename Device>
void def_pre_den_sparse(nb::module_& m) {
    m.def(
        "pre_den_sparse",
        [](gpu_array_c<const IdxT, Device> data_col_ind,
           gpu_array_c<const T, Device> data_values, long long nnz,
           gpu_array_c<const T, Device> mean_array, gpu_array_c<T, Device> den,
           gpu_array_c<int, Device> counter, std::uintptr_t stream) {
            launch_pre_den_sparse<T, IdxT>(
                data_col_ind.data(), data_values.data(), nnz, mean_array.data(),
                den.data(), counter.data(), (cudaStream_t)stream);
        },
        "data_col_ind"_a, "data_values"_a, nb::kw_only(), "nnz"_a,
        "mean_array"_a, "den"_a, "counter"_a, "stream"_a = 0);
}

// ---------------------------------------------------------------------------
// Fused Moran / Geary numerators (double accumulation, optional row perm).
// ---------------------------------------------------------------------------
constexpr int AUTOCORR_DENSE_BLOCK = 128;
constexpr int AUTOCORR_SPARSE_BLOCK = 256;
constexpr int AUTOCORR_STATS_BLOCK = 128;
constexpr int AUTOCORR_BLOCKS_PER_SM = 32;

static inline int autocorr_sm_count() {
    int device = 0;
    int sms = 1;
    cudaGetDevice(&device);
    cudaDeviceGetAttribute(&sms, cudaDevAttrMultiProcessorCount, device);
    return sms < 1 ? 1 : sms;
}

template <typename T, typename AdjIdxT, typename Device>
void def_autocorr_dense(nb::module_& m) {
    m.def(
        "autocorr_dense",
        [](gpu_array_c<const T, Device> x,
           gpu_array_c<const double, Device> means,
           gpu_array_c<const AdjIdxT, Device> adj_row_ptr,
           gpu_array_c<const AdjIdxT, Device> adj_col_ind,
           gpu_array_c<const T, Device> adj_data,
           std::optional<gpu_array_c<const int, Device>> perm,
           gpu_array_c<double, Device> num,
           std::optional<gpu_array_c<double, Device>> den, bool geary,
           std::uintptr_t stream) {
            nb_require(x.ndim() == 2, "autocorr_dense: x must be 2-D");
            const int n_samples = static_cast<int>(x.shape(0));
            const int n_features = static_cast<int>(x.shape(1));
            nb_require(static_cast<int>(means.shape(0)) == n_features &&
                           static_cast<int>(num.shape(0)) == n_features,
                       "autocorr_dense: means/num must have n_features");
            nb_require(static_cast<int>(adj_row_ptr.shape(0)) == n_samples + 1,
                       "autocorr_dense: adj_row_ptr must have n_samples+1");
            nb_require(!perm || static_cast<int>(perm->shape(0)) == n_samples,
                       "autocorr_dense: perm must have n_samples");
            nb_require(!den || static_cast<int>(den->shape(0)) == n_features,
                       "autocorr_dense: den must have n_features");
            if (n_samples == 0 || n_features == 0) return;
            const unsigned grid_x =
                (n_features + AUTOCORR_DENSE_BLOCK - 1) / AUTOCORR_DENSE_BLOCK;
            long long want_y = (static_cast<long long>(AUTOCORR_BLOCKS_PER_SM) *
                                    autocorr_sm_count() +
                                grid_x - 1) /
                               grid_x;
            long long cap_y = std::min<long long>(n_samples, max_grid_dim_y());
            const unsigned grid_y = static_cast<unsigned>(
                std::max<long long>(1, std::min(want_y, cap_y)));
            dim3 grid(grid_x, grid_y);
            dim3 block(AUTOCORR_DENSE_BLOCK);
            const int* d_perm = perm ? perm->data() : nullptr;
            double* d_den = den ? den->data() : nullptr;
            if (geary) {
                autocorr_dense_kernel<T, AdjIdxT, AUTOCORR_GEARY>
                    <<<grid, block, 0, (cudaStream_t)stream>>>(
                        x.data(), means.data(), adj_row_ptr.data(),
                        adj_col_ind.data(), adj_data.data(), d_perm, num.data(),
                        d_den, n_samples, n_features);
            } else {
                autocorr_dense_kernel<T, AdjIdxT, AUTOCORR_MORAN>
                    <<<grid, block, 0, (cudaStream_t)stream>>>(
                        x.data(), means.data(), adj_row_ptr.data(),
                        adj_col_ind.data(), adj_data.data(), d_perm, num.data(),
                        d_den, n_samples, n_features);
            }
            CUDA_CHECK_LAST_ERROR(autocorr_dense_kernel);
        },
        "x"_a, "means"_a, "adj_row_ptr"_a, "adj_col_ind"_a, "adj_data"_a,
        nb::kw_only(), "perm"_a = nb::none(), "num"_a, "den"_a = nb::none(),
        "geary"_a, "stream"_a = 0);
}

template <typename T, typename AdjIdxT, typename DataIdxT, typename Device>
void def_autocorr_sparse(nb::module_& m) {
    m.def(
        "autocorr_sparse",
        [](gpu_array_c<const AdjIdxT, Device> adj_row_ptr,
           gpu_array_c<const AdjIdxT, Device> adj_col_ind,
           gpu_array_c<const T, Device> adj_data,
           gpu_array_c<const DataIdxT, Device> data_row_ptr,
           gpu_array_c<const DataIdxT, Device> data_col_ind,
           gpu_array_c<const T, Device> data_values,
           gpu_array_c<const double, Device> means,
           std::optional<gpu_array_c<const int, Device>> perm,
           gpu_array_c<double, Device> num, int n_samples, int n_features,
           bool geary, std::uintptr_t stream) {
            nb_require(
                static_cast<int>(adj_row_ptr.shape(0)) == n_samples + 1 &&
                    static_cast<int>(data_row_ptr.shape(0)) == n_samples + 1,
                "autocorr_sparse: row pointers must have n_samples+1");
            nb_require(static_cast<int>(means.shape(0)) == n_features &&
                           static_cast<int>(num.shape(0)) == n_features,
                       "autocorr_sparse: means/num must have n_features");
            nb_require(data_col_ind.shape(0) == data_values.shape(0),
                       "autocorr_sparse: data indices/values length mismatch");
            nb_require(!perm || static_cast<int>(perm->shape(0)) == n_samples,
                       "autocorr_sparse: perm must have n_samples");
            if (n_samples == 0 || n_features == 0) return;
            dim3 grid(strided_grid(n_samples, 1));
            dim3 block(AUTOCORR_SPARSE_BLOCK);
            const int* d_perm = perm ? perm->data() : nullptr;
            if (geary) {
                autocorr_sparse_kernel<T, AdjIdxT, DataIdxT, AUTOCORR_GEARY>
                    <<<grid, block, 0, (cudaStream_t)stream>>>(
                        adj_row_ptr.data(), adj_col_ind.data(), adj_data.data(),
                        d_perm, data_row_ptr.data(), data_col_ind.data(),
                        data_values.data(), means.data(), num.data(), n_samples,
                        n_features);
            } else {
                autocorr_sparse_kernel<T, AdjIdxT, DataIdxT, AUTOCORR_MORAN>
                    <<<grid, block, 0, (cudaStream_t)stream>>>(
                        adj_row_ptr.data(), adj_col_ind.data(), adj_data.data(),
                        d_perm, data_row_ptr.data(), data_col_ind.data(),
                        data_values.data(), means.data(), num.data(), n_samples,
                        n_features);
            }
            CUDA_CHECK_LAST_ERROR(autocorr_sparse_kernel);
        },
        "adj_row_ptr"_a, "adj_col_ind"_a, "adj_data"_a, nb::kw_only(),
        "data_row_ptr"_a, "data_col_ind"_a, "data_values"_a, "means"_a,
        "perm"_a = nb::none(), "num"_a, "n_samples"_a, "n_features"_a,
        "geary"_a, "stream"_a = 0);
}

template <typename T, typename DataIdxT, typename Device>
void def_autocorr_sparse_stats(nb::module_& m) {
    m.def(
        "autocorr_sparse_stats",
        [](gpu_array_c<const DataIdxT, Device> data_row_ptr,
           gpu_array_c<const DataIdxT, Device> data_col_ind,
           gpu_array_c<const T, Device> data_values,
           gpu_array_c<const double, Device> colsum_w,
           gpu_array_c<double, Device> sum_x,
           gpu_array_c<double, Device> sum_x2, gpu_array_c<double, Device> tail,
           bool geary, std::uintptr_t stream) {
            const int n_samples = static_cast<int>(data_row_ptr.shape(0)) - 1;
            nb_require(static_cast<int>(colsum_w.shape(0)) == n_samples,
                       "autocorr_sparse_stats: colsum_w must have n_samples");
            nb_require(sum_x.shape(0) == sum_x2.shape(0) &&
                           sum_x.shape(0) == tail.shape(0),
                       "autocorr_sparse_stats: output length mismatch");
            nb_require(data_col_ind.shape(0) == data_values.shape(0),
                       "autocorr_sparse_stats: indices/values length mismatch");
            if (n_samples <= 0) return;
            dim3 grid(strided_grid(n_samples, 1));
            dim3 block(AUTOCORR_STATS_BLOCK);
            if (geary) {
                autocorr_sparse_stats_kernel<T, DataIdxT, AUTOCORR_GEARY>
                    <<<grid, block, 0, (cudaStream_t)stream>>>(
                        data_row_ptr.data(), data_col_ind.data(),
                        data_values.data(), colsum_w.data(), sum_x.data(),
                        sum_x2.data(), tail.data(), n_samples);
            } else {
                autocorr_sparse_stats_kernel<T, DataIdxT, AUTOCORR_MORAN>
                    <<<grid, block, 0, (cudaStream_t)stream>>>(
                        data_row_ptr.data(), data_col_ind.data(),
                        data_values.data(), colsum_w.data(), sum_x.data(),
                        sum_x2.data(), tail.data(), n_samples);
            }
            CUDA_CHECK_LAST_ERROR(autocorr_sparse_stats_kernel);
        },
        "data_row_ptr"_a, "data_col_ind"_a, "data_values"_a, nb::kw_only(),
        "colsum_w"_a, "sum_x"_a, "sum_x2"_a, "tail"_a, "geary"_a,
        "stream"_a = 0);
}

template <typename Device>
void register_bindings(nb::module_& m) {
    def_morans_dense<float, Device>(m);
    def_morans_dense<double, Device>(m);

    def_morans_sparse<float, int, int, Device>(m);
    def_morans_sparse<float, int, long long, Device>(m);
    def_morans_sparse<float, long long, int, Device>(m);
    def_morans_sparse<float, long long, long long, Device>(m);
    def_morans_sparse<double, int, int, Device>(m);
    def_morans_sparse<double, int, long long, Device>(m);
    def_morans_sparse<double, long long, int, Device>(m);
    def_morans_sparse<double, long long, long long, Device>(m);

    def_gearys_dense<float, Device>(m);
    def_gearys_dense<double, Device>(m);

    def_gearys_sparse<float, int, int, Device>(m);
    def_gearys_sparse<float, int, long long, Device>(m);
    def_gearys_sparse<float, long long, int, Device>(m);
    def_gearys_sparse<float, long long, long long, Device>(m);
    def_gearys_sparse<double, int, int, Device>(m);
    def_gearys_sparse<double, int, long long, Device>(m);
    def_gearys_sparse<double, long long, int, Device>(m);
    def_gearys_sparse<double, long long, long long, Device>(m);

    def_pre_den_sparse<float, int, Device>(m);
    def_pre_den_sparse<float, long long, Device>(m);
    def_pre_den_sparse<double, int, Device>(m);
    def_pre_den_sparse<double, long long, Device>(m);

    def_autocorr_dense<float, int, Device>(m);
    def_autocorr_dense<float, long long, Device>(m);
    def_autocorr_dense<double, int, Device>(m);
    def_autocorr_dense<double, long long, Device>(m);

    def_autocorr_sparse<float, int, int, Device>(m);
    def_autocorr_sparse<float, int, long long, Device>(m);
    def_autocorr_sparse<float, long long, int, Device>(m);
    def_autocorr_sparse<float, long long, long long, Device>(m);
    def_autocorr_sparse<double, int, int, Device>(m);
    def_autocorr_sparse<double, int, long long, Device>(m);
    def_autocorr_sparse<double, long long, int, Device>(m);
    def_autocorr_sparse<double, long long, long long, Device>(m);

    def_autocorr_sparse_stats<float, int, Device>(m);
    def_autocorr_sparse_stats<float, long long, Device>(m);
    def_autocorr_sparse_stats<double, int, Device>(m);
    def_autocorr_sparse_stats<double, long long, Device>(m);
}

NB_MODULE(_autocorr_cuda, m) {
    REGISTER_GPU_BINDINGS(register_bindings, m);
}
