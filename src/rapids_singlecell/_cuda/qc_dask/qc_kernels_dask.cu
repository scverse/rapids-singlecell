#include <cuda_runtime.h>
#include "../nb_types.h"

#include "kernels_qcd.cuh"

using namespace nb::literals;

constexpr int SPARSE_BLOCK_SIZE = 32;
constexpr int GENES_BLOCK_SIZE = 256;
constexpr int DENSE_BLOCK_DIM = 16;

template <typename T, typename IdxT>
static inline void launch_qc_csr_cells(const IdxT* indptr, const IdxT* index,
                                       const T* data, T* sums_cells,
                                       int* cell_ex, int n_cells,
                                       cudaStream_t stream) {
    dim3 block(SPARSE_BLOCK_SIZE);
    dim3 grid((n_cells + SPARSE_BLOCK_SIZE - 1) / SPARSE_BLOCK_SIZE);
    qc_csr_cells_kernel<T, IdxT><<<grid, block, 0, stream>>>(
        indptr, index, data, sums_cells, cell_ex, n_cells);
    CUDA_CHECK_LAST_ERROR(qc_csr_cells_kernel);
}

template <typename T, typename IdxT>
static inline void launch_qc_csr_genes(const IdxT* index, const T* data,
                                       T* sums_genes, int* gene_ex,
                                       long long nnz, cudaStream_t stream) {
    int block = GENES_BLOCK_SIZE;
    unsigned int grid = strided_grid(nnz, block);
    qc_csr_genes_kernel<T, IdxT>
        <<<grid, block, 0, stream>>>(index, data, sums_genes, gene_ex, nnz);
    CUDA_CHECK_LAST_ERROR(qc_csr_genes_kernel);
}

template <typename T>
static inline void launch_qc_dense_cells(const T* data, T* sums_cells,
                                         int* cell_ex, int n_cells, int n_genes,
                                         cudaStream_t stream) {
    dim3 block(DENSE_BLOCK_DIM, DENSE_BLOCK_DIM);
    dim3 grid((n_cells + DENSE_BLOCK_DIM - 1) / DENSE_BLOCK_DIM,
              (n_genes + DENSE_BLOCK_DIM - 1) / DENSE_BLOCK_DIM);
    qc_dense_cells_kernel<T><<<grid, block, 0, stream>>>(
        data, sums_cells, cell_ex, n_cells, n_genes);
    CUDA_CHECK_LAST_ERROR(qc_dense_cells_kernel);
}

template <typename T>
static inline void launch_qc_dense_genes(const T* data, T* sums_genes,
                                         int* gene_ex, int n_cells, int n_genes,
                                         cudaStream_t stream) {
    dim3 block(DENSE_BLOCK_DIM, DENSE_BLOCK_DIM);
    dim3 grid((n_cells + DENSE_BLOCK_DIM - 1) / DENSE_BLOCK_DIM,
              (n_genes + DENSE_BLOCK_DIM - 1) / DENSE_BLOCK_DIM);
    qc_dense_genes_kernel<T><<<grid, block, 0, stream>>>(
        data, sums_genes, gene_ex, n_cells, n_genes);
    CUDA_CHECK_LAST_ERROR(qc_dense_genes_kernel);
}

template <typename T, typename IdxT, typename Device>
void def_sparse_qc_csr_cells(nb::module_& m) {
    m.def(
        "sparse_qc_csr_cells",
        [](gpu_array_c<const IdxT, Device> indptr,
           gpu_array_c<const IdxT, Device> index,
           gpu_array_c<const T, Device> data, gpu_array_c<T, Device> sums_cells,
           gpu_array_c<int, Device> cell_ex, int n_cells,
           std::uintptr_t stream) {
            launch_qc_csr_cells<T, IdxT>(
                indptr.data(), index.data(), data.data(), sums_cells.data(),
                cell_ex.data(), n_cells, (cudaStream_t)stream);
        },
        "indptr"_a, "index"_a, "data"_a, nb::kw_only(), "sums_cells"_a,
        "cell_ex"_a, "n_cells"_a, "stream"_a = 0);
}

template <typename T, typename IdxT, typename Device>
void def_sparse_qc_csr_genes(nb::module_& m) {
    m.def(
        "sparse_qc_csr_genes",
        [](gpu_array_c<const IdxT, Device> index,
           gpu_array_c<const T, Device> data, gpu_array_c<T, Device> sums_genes,
           gpu_array_c<int, Device> gene_ex, long long nnz,
           std::uintptr_t stream) {
            launch_qc_csr_genes<T, IdxT>(index.data(), data.data(),
                                         sums_genes.data(), gene_ex.data(), nnz,
                                         (cudaStream_t)stream);
        },
        "index"_a, "data"_a, nb::kw_only(), "sums_genes"_a, "gene_ex"_a,
        "nnz"_a, "stream"_a = 0);
}

template <typename T, typename Device>
void def_sparse_qc_dense_cells(nb::module_& m) {
    m.def(
        "sparse_qc_dense_cells",
        [](gpu_array_c<const T, Device> data, gpu_array_c<T, Device> sums_cells,
           gpu_array_c<int, Device> cell_ex, int n_cells, int n_genes,
           std::uintptr_t stream) {
            launch_qc_dense_cells<T>(data.data(), sums_cells.data(),
                                     cell_ex.data(), n_cells, n_genes,
                                     (cudaStream_t)stream);
        },
        "data"_a, nb::kw_only(), "sums_cells"_a, "cell_ex"_a, "n_cells"_a,
        "n_genes"_a, "stream"_a = 0);
}

template <typename T, typename Device>
void def_sparse_qc_dense_genes(nb::module_& m) {
    m.def(
        "sparse_qc_dense_genes",
        [](gpu_array_c<const T, Device> data, gpu_array_c<T, Device> sums_genes,
           gpu_array_c<int, Device> gene_ex, int n_cells, int n_genes,
           std::uintptr_t stream) {
            launch_qc_dense_genes<T>(data.data(), sums_genes.data(),
                                     gene_ex.data(), n_cells, n_genes,
                                     (cudaStream_t)stream);
        },
        "data"_a, nb::kw_only(), "sums_genes"_a, "gene_ex"_a, "n_cells"_a,
        "n_genes"_a, "stream"_a = 0);
}

template <typename Device>
void register_bindings(nb::module_& m) {
    def_sparse_qc_csr_cells<float, int, Device>(m);
    def_sparse_qc_csr_cells<float, long long, Device>(m);
    def_sparse_qc_csr_cells<double, int, Device>(m);
    def_sparse_qc_csr_cells<double, long long, Device>(m);

    def_sparse_qc_csr_genes<float, int, Device>(m);
    def_sparse_qc_csr_genes<float, long long, Device>(m);
    def_sparse_qc_csr_genes<double, int, Device>(m);
    def_sparse_qc_csr_genes<double, long long, Device>(m);

    def_sparse_qc_dense_cells<float, Device>(m);
    def_sparse_qc_dense_cells<double, Device>(m);

    def_sparse_qc_dense_genes<float, Device>(m);
    def_sparse_qc_dense_genes<double, Device>(m);
}

NB_MODULE(_qc_dask_cuda, m) {
    REGISTER_GPU_BINDINGS(register_bindings, m);
}
