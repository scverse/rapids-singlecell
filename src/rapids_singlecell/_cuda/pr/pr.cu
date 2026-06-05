#include <cuda_runtime.h>
#include "../nb_types.h"

#include "kernels_pr.cuh"
#include "kernels_pr_hvg.cuh"

using namespace nb::literals;

constexpr int SPARSE_BLOCK_SIZE = 32;
constexpr int CSR_NORM_BLOCK_SIZE = 8;
constexpr int DENSE_BLOCK_DIM = 8;

template <typename T, typename IdxT>
static inline void launch_sparse_norm_res_csc(
    const IdxT* indptr, const IdxT* index, const T* data, const T* sums_cells,
    const T* sums_genes, T* residuals, T inv_sum_total, T clip, T inv_theta,
    int n_cells, int n_genes, cudaStream_t stream) {
    dim3 block(SPARSE_BLOCK_SIZE);
    dim3 grid((n_genes + SPARSE_BLOCK_SIZE - 1) / SPARSE_BLOCK_SIZE);
    sparse_norm_res_csc_kernel<T, IdxT><<<grid, block, 0, stream>>>(
        indptr, index, data, sums_cells, sums_genes, residuals, inv_sum_total,
        clip, inv_theta, n_cells, n_genes);
    CUDA_CHECK_LAST_ERROR(sparse_norm_res_csc_kernel);
}

template <typename T, typename IdxT>
static inline void launch_sparse_norm_res_csr(
    const IdxT* indptr, const IdxT* index, const T* data, const T* sums_cells,
    const T* sums_genes, T* residuals, T inv_sum_total, T clip, T inv_theta,
    int n_cells, int n_genes, cudaStream_t stream) {
    dim3 block(CSR_NORM_BLOCK_SIZE);
    dim3 grid((n_cells + CSR_NORM_BLOCK_SIZE - 1) / CSR_NORM_BLOCK_SIZE);
    sparse_norm_res_csr_kernel<T, IdxT><<<grid, block, 0, stream>>>(
        indptr, index, data, sums_cells, sums_genes, residuals, inv_sum_total,
        clip, inv_theta, n_cells, n_genes);
    CUDA_CHECK_LAST_ERROR(sparse_norm_res_csr_kernel);
}

template <typename T>
static inline void launch_dense_norm_res(const T* X, T* residuals,
                                         const T* sums_cells,
                                         const T* sums_genes, T inv_sum_total,
                                         T clip, T inv_theta, int n_cells,
                                         int n_genes, cudaStream_t stream) {
    dim3 block(DENSE_BLOCK_DIM, DENSE_BLOCK_DIM);
    dim3 grid((n_cells + DENSE_BLOCK_DIM - 1) / DENSE_BLOCK_DIM,
              (n_genes + DENSE_BLOCK_DIM - 1) / DENSE_BLOCK_DIM);
    dense_norm_res_kernel<T><<<grid, block, 0, stream>>>(
        X, residuals, sums_cells, sums_genes, inv_sum_total, clip, inv_theta,
        n_cells, n_genes);
    CUDA_CHECK_LAST_ERROR(dense_norm_res_kernel);
}

template <typename T, typename IdxT>
static inline void launch_sparse_sum_csc(const IdxT* indptr, const IdxT* index,
                                         const T* data, T* sums_genes,
                                         T* sums_cells, int n_genes,
                                         cudaStream_t stream) {
    dim3 block(SPARSE_BLOCK_SIZE);
    dim3 grid((n_genes + SPARSE_BLOCK_SIZE - 1) / SPARSE_BLOCK_SIZE);
    sparse_sum_csc_kernel<T, IdxT><<<grid, block, 0, stream>>>(
        indptr, index, data, sums_genes, sums_cells, n_genes);
    CUDA_CHECK_LAST_ERROR(sparse_sum_csc_kernel);
}

template <typename T, typename IdxT>
static inline void launch_csc_hvg_res(const IdxT* indptr, const IdxT* index,
                                      const T* data, const T* sums_genes,
                                      const T* sums_cells, T* residuals,
                                      T inv_sum_total, T clip, T inv_theta,
                                      int n_genes, int n_cells,
                                      cudaStream_t stream) {
    dim3 block(SPARSE_BLOCK_SIZE);
    dim3 grid((n_genes + SPARSE_BLOCK_SIZE - 1) / SPARSE_BLOCK_SIZE);
    csc_hvg_res_kernel<T, IdxT><<<grid, block, 0, stream>>>(
        indptr, index, data, sums_genes, sums_cells, residuals, inv_sum_total,
        clip, inv_theta, n_genes, n_cells);
    CUDA_CHECK_LAST_ERROR(csc_hvg_res_kernel);
}

template <typename T>
static inline void launch_dense_hvg_res(const T* data, const T* sums_genes,
                                        const T* sums_cells, T* residuals,
                                        T inv_sum_total, T clip, T inv_theta,
                                        int n_genes, int n_cells,
                                        cudaStream_t stream) {
    dim3 block(SPARSE_BLOCK_SIZE);
    dim3 grid((n_genes + SPARSE_BLOCK_SIZE - 1) / SPARSE_BLOCK_SIZE);
    dense_hvg_res_kernel<T><<<grid, block, 0, stream>>>(
        data, sums_genes, sums_cells, residuals, inv_sum_total, clip, inv_theta,
        n_genes, n_cells);
    CUDA_CHECK_LAST_ERROR(dense_hvg_res_kernel);
}

// Helper to define sparse_norm_res_csc for a given dtype and index type
template <typename T, typename IdxT, typename Device>
void def_sparse_norm_res_csc(nb::module_& m) {
    m.def(
        "sparse_norm_res_csc",
        [](gpu_array_c<const IdxT, Device> indptr,
           gpu_array_c<const IdxT, Device> index,
           gpu_array_c<const T, Device> data,
           gpu_array_c<const T, Device> sums_cells,
           gpu_array_c<const T, Device> sums_genes,
           gpu_array_c<T, Device> residuals, T inv_sum_total, T clip,
           T inv_theta, int n_cells, int n_genes, std::uintptr_t stream) {
            launch_sparse_norm_res_csc<T, IdxT>(
                indptr.data(), index.data(), data.data(), sums_cells.data(),
                sums_genes.data(), residuals.data(), inv_sum_total, clip,
                inv_theta, n_cells, n_genes, (cudaStream_t)stream);
        },
        "indptr"_a, "index"_a, "data"_a, nb::kw_only(), "sums_cells"_a,
        "sums_genes"_a, "residuals"_a, "inv_sum_total"_a, "clip"_a,
        "inv_theta"_a, "n_cells"_a, "n_genes"_a, "stream"_a = 0);
}

// Helper to define sparse_norm_res_csr for a given dtype and index type
template <typename T, typename IdxT, typename Device>
void def_sparse_norm_res_csr(nb::module_& m) {
    m.def(
        "sparse_norm_res_csr",
        [](gpu_array_c<const IdxT, Device> indptr,
           gpu_array_c<const IdxT, Device> index,
           gpu_array_c<const T, Device> data,
           gpu_array_c<const T, Device> sums_cells,
           gpu_array_c<const T, Device> sums_genes,
           gpu_array_c<T, Device> residuals, T inv_sum_total, T clip,
           T inv_theta, int n_cells, int n_genes, std::uintptr_t stream) {
            launch_sparse_norm_res_csr<T, IdxT>(
                indptr.data(), index.data(), data.data(), sums_cells.data(),
                sums_genes.data(), residuals.data(), inv_sum_total, clip,
                inv_theta, n_cells, n_genes, (cudaStream_t)stream);
        },
        "indptr"_a, "index"_a, "data"_a, nb::kw_only(), "sums_cells"_a,
        "sums_genes"_a, "residuals"_a, "inv_sum_total"_a, "clip"_a,
        "inv_theta"_a, "n_cells"_a, "n_genes"_a, "stream"_a = 0);
}

// Helper to define dense_norm_res for a given dtype
template <typename T, typename Device>
void def_dense_norm_res(nb::module_& m) {
    m.def(
        "dense_norm_res",
        [](gpu_array_c<const T, Device> X, gpu_array_c<T, Device> residuals,
           gpu_array_c<const T, Device> sums_cells,
           gpu_array_c<const T, Device> sums_genes, T inv_sum_total, T clip,
           T inv_theta, int n_cells, int n_genes, std::uintptr_t stream) {
            launch_dense_norm_res<T>(X.data(), residuals.data(),
                                     sums_cells.data(), sums_genes.data(),
                                     inv_sum_total, clip, inv_theta, n_cells,
                                     n_genes, (cudaStream_t)stream);
        },
        "X"_a, nb::kw_only(), "residuals"_a, "sums_cells"_a, "sums_genes"_a,
        "inv_sum_total"_a, "clip"_a, "inv_theta"_a, "n_cells"_a, "n_genes"_a,
        "stream"_a = 0);
}

// Helper to define sparse_sum_csc for a given dtype and index type
template <typename T, typename IdxT, typename Device>
void def_sparse_sum_csc(nb::module_& m) {
    m.def(
        "sparse_sum_csc",
        [](gpu_array_c<const IdxT, Device> indptr,
           gpu_array_c<const IdxT, Device> index,
           gpu_array_c<const T, Device> data, gpu_array_c<T, Device> sums_genes,
           gpu_array_c<T, Device> sums_cells, int n_genes,
           std::uintptr_t stream) {
            launch_sparse_sum_csc<T, IdxT>(
                indptr.data(), index.data(), data.data(), sums_genes.data(),
                sums_cells.data(), n_genes, (cudaStream_t)stream);
        },
        "indptr"_a, "index"_a, "data"_a, nb::kw_only(), "sums_genes"_a,
        "sums_cells"_a, "n_genes"_a, "stream"_a = 0);
}

// Helper to define csc_hvg_res for a given dtype and index type
template <typename T, typename IdxT, typename Device>
void def_csc_hvg_res(nb::module_& m) {
    m.def(
        "csc_hvg_res",
        [](gpu_array_c<const IdxT, Device> indptr,
           gpu_array_c<const IdxT, Device> index,
           gpu_array_c<const T, Device> data,
           gpu_array_c<const T, Device> sums_genes,
           gpu_array_c<const T, Device> sums_cells,
           gpu_array_c<T, Device> residuals, T inv_sum_total, T clip,
           T inv_theta, int n_genes, int n_cells, std::uintptr_t stream) {
            launch_csc_hvg_res<T, IdxT>(
                indptr.data(), index.data(), data.data(), sums_genes.data(),
                sums_cells.data(), residuals.data(), inv_sum_total, clip,
                inv_theta, n_genes, n_cells, (cudaStream_t)stream);
        },
        "indptr"_a, "index"_a, "data"_a, nb::kw_only(), "sums_genes"_a,
        "sums_cells"_a, "residuals"_a, "inv_sum_total"_a, "clip"_a,
        "inv_theta"_a, "n_genes"_a, "n_cells"_a, "stream"_a = 0);
}

// Helper to define dense_hvg_res for a given dtype (always F-contiguous input)
template <typename T, typename Device>
void def_dense_hvg_res(nb::module_& m) {
    m.def(
        "dense_hvg_res",
        [](gpu_array_contig<const T, Device, nb::f_contig> data,
           gpu_array_c<const T, Device> sums_genes,
           gpu_array_c<const T, Device> sums_cells,
           gpu_array_c<T, Device> residuals, T inv_sum_total, T clip,
           T inv_theta, int n_genes, int n_cells, std::uintptr_t stream) {
            launch_dense_hvg_res<T>(data.data(), sums_genes.data(),
                                    sums_cells.data(), residuals.data(),
                                    inv_sum_total, clip, inv_theta, n_genes,
                                    n_cells, (cudaStream_t)stream);
        },
        "data"_a, nb::kw_only(), "sums_genes"_a, "sums_cells"_a, "residuals"_a,
        "inv_sum_total"_a, "clip"_a, "inv_theta"_a, "n_genes"_a, "n_cells"_a,
        "stream"_a = 0);
}

template <typename Device>
void register_bindings(nb::module_& m) {
    // sparse_norm_res_csc
    def_sparse_norm_res_csc<float, int, Device>(m);
    def_sparse_norm_res_csc<double, int, Device>(m);
    def_sparse_norm_res_csc<float, long long, Device>(m);
    def_sparse_norm_res_csc<double, long long, Device>(m);

    // sparse_norm_res_csr
    def_sparse_norm_res_csr<float, int, Device>(m);
    def_sparse_norm_res_csr<double, int, Device>(m);
    def_sparse_norm_res_csr<float, long long, Device>(m);
    def_sparse_norm_res_csr<double, long long, Device>(m);

    // dense_norm_res
    def_dense_norm_res<float, Device>(m);
    def_dense_norm_res<double, Device>(m);

    // sparse_sum_csc
    def_sparse_sum_csc<float, int, Device>(m);
    def_sparse_sum_csc<double, int, Device>(m);
    def_sparse_sum_csc<float, long long, Device>(m);
    def_sparse_sum_csc<double, long long, Device>(m);

    // csc_hvg_res
    def_csc_hvg_res<float, int, Device>(m);
    def_csc_hvg_res<double, int, Device>(m);
    def_csc_hvg_res<float, long long, Device>(m);
    def_csc_hvg_res<double, long long, Device>(m);

    // dense_hvg_res - always F-contiguous (Python calls cp.asfortranarray)
    def_dense_hvg_res<float, Device>(m);
    def_dense_hvg_res<double, Device>(m);
}

NB_MODULE(_pr_cuda, m) {
    REGISTER_GPU_BINDINGS(register_bindings, m);
}
