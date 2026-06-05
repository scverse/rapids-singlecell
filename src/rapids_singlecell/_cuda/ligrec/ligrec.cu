#include <cuda_runtime.h>
#include "../nb_types.h"

#include "kernels_ligrec.cuh"

using namespace nb::literals;

constexpr int SPARSE_BLOCK_SIZE = 32;
constexpr int DENSE_BLOCK_DIM = 32;

template <typename T>
static inline void launch_sum_count_dense(const T* data, const int* clusters,
                                          T* sum, int* count, size_t rows,
                                          size_t cols, size_t ncls,
                                          cudaStream_t stream) {
    dim3 block(DENSE_BLOCK_DIM, DENSE_BLOCK_DIM);
    dim3 grid(strided_grid(static_cast<long long>(rows), DENSE_BLOCK_DIM),
              strided_grid_y(static_cast<long long>(cols), DENSE_BLOCK_DIM));
    sum_and_count_dense_kernel<T><<<grid, block, 0, stream>>>(
        data, clusters, sum, count, rows, cols, ncls);
    CUDA_CHECK_LAST_ERROR(sum_and_count_dense_kernel);
}

template <typename T, typename IdxT>
static inline void launch_sum_count_sparse(const IdxT* indptr,
                                           const IdxT* index, const T* data,
                                           const int* clusters, T* sum,
                                           int* count, int rows, int ncls,
                                           cudaStream_t stream) {
    dim3 block(SPARSE_BLOCK_SIZE);
    dim3 grid((rows + SPARSE_BLOCK_SIZE - 1) / SPARSE_BLOCK_SIZE);
    sum_and_count_sparse_kernel<T, IdxT><<<grid, block, 0, stream>>>(
        indptr, index, data, clusters, sum, count, rows, ncls);
    CUDA_CHECK_LAST_ERROR(sum_and_count_sparse_kernel);
}

template <typename T>
static inline void launch_mean_dense(const T* data, const int* clusters, T* g,
                                     size_t rows, size_t cols, size_t ncls,
                                     cudaStream_t stream) {
    dim3 block(DENSE_BLOCK_DIM, DENSE_BLOCK_DIM);
    dim3 grid(strided_grid(static_cast<long long>(rows), DENSE_BLOCK_DIM),
              strided_grid_y(static_cast<long long>(cols), DENSE_BLOCK_DIM));
    mean_dense_kernel<T>
        <<<grid, block, 0, stream>>>(data, clusters, g, rows, cols, ncls);
    CUDA_CHECK_LAST_ERROR(mean_dense_kernel);
}

template <typename T, typename IdxT>
static inline void launch_mean_sparse(const IdxT* indptr, const IdxT* index,
                                      const T* data, const int* clusters, T* g,
                                      int rows, int ncls, cudaStream_t stream) {
    dim3 block(SPARSE_BLOCK_SIZE);
    dim3 grid((rows + SPARSE_BLOCK_SIZE - 1) / SPARSE_BLOCK_SIZE);
    mean_sparse_kernel<T, IdxT><<<grid, block, 0, stream>>>(
        indptr, index, data, clusters, g, rows, ncls);
    CUDA_CHECK_LAST_ERROR(mean_sparse_kernel);
}

template <typename T>
static inline void launch_elementwise_diff(T* g, const T* total_counts,
                                           size_t n_genes, size_t n_clusters,
                                           cudaStream_t stream) {
    dim3 block(DENSE_BLOCK_DIM, DENSE_BLOCK_DIM);
    dim3 grid(
        strided_grid(static_cast<long long>(n_genes), DENSE_BLOCK_DIM),
        strided_grid_y(static_cast<long long>(n_clusters), DENSE_BLOCK_DIM));
    elementwise_diff_kernel<T>
        <<<grid, block, 0, stream>>>(g, total_counts, n_genes, n_clusters);
    CUDA_CHECK_LAST_ERROR(elementwise_diff_kernel);
}

template <typename T>
static inline void launch_interaction(const int* interactions,
                                      const int* interaction_clusters,
                                      const T* mean, T* res, const bool* mask,
                                      const T* g, size_t n_iter,
                                      size_t n_inter_clust, size_t ncls,
                                      cudaStream_t stream) {
    dim3 block(DENSE_BLOCK_DIM, DENSE_BLOCK_DIM);
    dim3 grid(
        strided_grid(static_cast<long long>(n_iter), DENSE_BLOCK_DIM),
        strided_grid_y(static_cast<long long>(n_inter_clust), DENSE_BLOCK_DIM));
    interaction_kernel<T>
        <<<grid, block, 0, stream>>>(interactions, interaction_clusters, mean,
                                     res, mask, g, n_iter, n_inter_clust, ncls);
    CUDA_CHECK_LAST_ERROR(interaction_kernel);
}

template <typename T>
static inline void launch_res_mean(const int* interactions,
                                   const int* interaction_clusters,
                                   const T* mean, T* res_mean, size_t n_inter,
                                   size_t n_inter_clust, size_t ncls,
                                   cudaStream_t stream) {
    dim3 block(DENSE_BLOCK_DIM, DENSE_BLOCK_DIM);
    dim3 grid(
        strided_grid(static_cast<long long>(n_inter), DENSE_BLOCK_DIM),
        strided_grid_y(static_cast<long long>(n_inter_clust), DENSE_BLOCK_DIM));
    res_mean_kernel<T>
        <<<grid, block, 0, stream>>>(interactions, interaction_clusters, mean,
                                     res_mean, n_inter, n_inter_clust, ncls);
    CUDA_CHECK_LAST_ERROR(res_mean_kernel);
}

template <typename T, typename Device>
void def_sum_count_dense(nb::module_& m) {
    m.def(
        "sum_count_dense",
        [](gpu_array_c<const T, Device> data,
           gpu_array_c<const int, Device> clusters, gpu_array_c<T, Device> sum,
           gpu_array_c<int, Device> count, size_t rows, size_t cols,
           size_t ncls, std::uintptr_t stream) {
            launch_sum_count_dense<T>(data.data(), clusters.data(), sum.data(),
                                      count.data(), rows, cols, ncls,
                                      (cudaStream_t)stream);
        },
        "data"_a, nb::kw_only(), "clusters"_a, "sum"_a, "count"_a, "rows"_a,
        "cols"_a, "ncls"_a, "stream"_a = 0);
}

template <typename T, typename IdxT, typename Device>
void def_sum_count_sparse(nb::module_& m) {
    m.def(
        "sum_count_sparse",
        [](gpu_array_c<const IdxT, Device> indptr,
           gpu_array_c<const IdxT, Device> index,
           gpu_array_c<const T, Device> data,
           gpu_array_c<const int, Device> clusters, gpu_array_c<T, Device> sum,
           gpu_array_c<int, Device> count, int rows, int ncls,
           std::uintptr_t stream) {
            launch_sum_count_sparse<T, IdxT>(
                indptr.data(), index.data(), data.data(), clusters.data(),
                sum.data(), count.data(), rows, ncls, (cudaStream_t)stream);
        },
        "indptr"_a, "index"_a, "data"_a, nb::kw_only(), "clusters"_a, "sum"_a,
        "count"_a, "rows"_a, "ncls"_a, "stream"_a = 0);
}

template <typename T, typename Device>
void def_mean_dense(nb::module_& m) {
    m.def(
        "mean_dense",
        [](gpu_array_c<const T, Device> data,
           gpu_array_c<const int, Device> clusters, gpu_array_c<T, Device> g,
           size_t rows, size_t cols, size_t ncls, std::uintptr_t stream) {
            launch_mean_dense<T>(data.data(), clusters.data(), g.data(), rows,
                                 cols, ncls, (cudaStream_t)stream);
        },
        "data"_a, nb::kw_only(), "clusters"_a, "g"_a, "rows"_a, "cols"_a,
        "ncls"_a, "stream"_a = 0);
}

template <typename T, typename IdxT, typename Device>
void def_mean_sparse(nb::module_& m) {
    m.def(
        "mean_sparse",
        [](gpu_array_c<const IdxT, Device> indptr,
           gpu_array_c<const IdxT, Device> index,
           gpu_array_c<const T, Device> data,
           gpu_array_c<const int, Device> clusters, gpu_array_c<T, Device> g,
           int rows, int ncls, std::uintptr_t stream) {
            launch_mean_sparse<T, IdxT>(indptr.data(), index.data(),
                                        data.data(), clusters.data(), g.data(),
                                        rows, ncls, (cudaStream_t)stream);
        },
        "indptr"_a, "index"_a, "data"_a, nb::kw_only(), "clusters"_a, "g"_a,
        "rows"_a, "ncls"_a, "stream"_a = 0);
}

template <typename T, typename Device>
void def_elementwise_diff(nb::module_& m) {
    m.def(
        "elementwise_diff",
        [](gpu_array_c<T, Device> g, gpu_array_c<const T, Device> total_counts,
           size_t n_genes, size_t n_clusters, std::uintptr_t stream) {
            launch_elementwise_diff<T>(g.data(), total_counts.data(), n_genes,
                                       n_clusters, (cudaStream_t)stream);
        },
        "g"_a, nb::kw_only(), "total_counts"_a, "n_genes"_a, "n_clusters"_a,
        "stream"_a = 0);
}

template <typename T, typename Device>
void def_interaction(nb::module_& m) {
    m.def(
        "interaction",
        [](gpu_array_c<const int, Device> interactions,
           gpu_array_c<const int, Device> interaction_clusters,
           gpu_array_c<const T, Device> mean, gpu_array_c<T, Device> res,
           gpu_array_c<const bool, Device> mask, gpu_array_c<const T, Device> g,
           size_t n_iter, size_t n_inter_clust, size_t ncls,
           std::uintptr_t stream) {
            launch_interaction<T>(interactions.data(),
                                  interaction_clusters.data(), mean.data(),
                                  res.data(), mask.data(), g.data(), n_iter,
                                  n_inter_clust, ncls, (cudaStream_t)stream);
        },
        "interactions"_a, nb::kw_only(), "interaction_clusters"_a, "mean"_a,
        "res"_a, "mask"_a, "g"_a, "n_iter"_a, "n_inter_clust"_a, "ncls"_a,
        "stream"_a = 0);
}

template <typename T, typename Device>
void def_res_mean(nb::module_& m) {
    m.def(
        "res_mean",
        [](gpu_array_c<const int, Device> interactions,
           gpu_array_c<const int, Device> interaction_clusters,
           gpu_array_c<const T, Device> mean, gpu_array_c<T, Device> res_mean,
           size_t n_inter, size_t n_inter_clust, size_t ncls,
           std::uintptr_t stream) {
            launch_res_mean<T>(interactions.data(), interaction_clusters.data(),
                               mean.data(), res_mean.data(), n_inter,
                               n_inter_clust, ncls, (cudaStream_t)stream);
        },
        "interactions"_a, nb::kw_only(), "interaction_clusters"_a, "mean"_a,
        "res_mean"_a, "n_inter"_a, "n_inter_clust"_a, "ncls"_a, "stream"_a = 0);
}

template <typename Device>
void register_bindings(nb::module_& m) {
    def_sum_count_dense<float, Device>(m);
    def_sum_count_dense<double, Device>(m);

    def_sum_count_sparse<float, int, Device>(m);
    def_sum_count_sparse<float, long long, Device>(m);
    def_sum_count_sparse<double, int, Device>(m);
    def_sum_count_sparse<double, long long, Device>(m);

    def_mean_dense<float, Device>(m);
    def_mean_dense<double, Device>(m);

    def_mean_sparse<float, int, Device>(m);
    def_mean_sparse<float, long long, Device>(m);
    def_mean_sparse<double, int, Device>(m);
    def_mean_sparse<double, long long, Device>(m);

    def_elementwise_diff<float, Device>(m);
    def_elementwise_diff<double, Device>(m);

    def_interaction<float, Device>(m);
    def_interaction<double, Device>(m);

    def_res_mean<float, Device>(m);
    def_res_mean<double, Device>(m);
}

NB_MODULE(_ligrec_cuda, m) {
    REGISTER_GPU_BINDINGS(register_bindings, m);
}
