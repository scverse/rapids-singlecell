#include <cuda_runtime.h>
#include "../nb_types.h"

#include "kernels_qc.cuh"

using namespace nb::literals;

constexpr int DENSE_BLOCK_DIM = 16;

// Sparse QC = major-axis sums/counts (warp per row) + minor-axis QcOp sweep.
// Returns whether unsorted rows were detected.
template <typename T, typename IdxT>
static inline bool launch_qc_sparse(const IdxT* indptr, const IdxT* index,
                                    const T* data, T* sums_major, int* ex_major,
                                    T* sums_minor, int* ex_minor, int major,
                                    int minor, long long nnz,
                                    bool assume_unsorted, cudaStream_t stream) {
    row_reduce<T, IdxT>(indptr, index, data, nullptr, sums_major, ex_major,
                        major, stream);
    QcOp<T> op{data, sums_minor, ex_minor, 0};
    return minor_reduce<IdxT>(indptr, index, op, major, minor, nnz,
                              assume_unsorted, stream);
}

template <typename T>
static inline void launch_qc_dense(const T* data, T* sums_cells, T* sums_genes,
                                   int* cell_ex, int* gene_ex, int n_cells,
                                   int n_genes, cudaStream_t stream) {
    dim3 block(DENSE_BLOCK_DIM, DENSE_BLOCK_DIM);
    dim3 grid((n_cells + DENSE_BLOCK_DIM - 1) / DENSE_BLOCK_DIM,
              (n_genes + DENSE_BLOCK_DIM - 1) / DENSE_BLOCK_DIM);
    qc_dense_kernel<T><<<grid, block, 0, stream>>>(
        data, sums_cells, sums_genes, cell_ex, gene_ex, n_cells, n_genes);
    CUDA_CHECK_LAST_ERROR(qc_dense_kernel);
}

template <typename T>
static inline void launch_qc_dense_sub(const T* data, T* sums_cells,
                                       const bool* mask, int n_cells,
                                       int n_genes, cudaStream_t stream) {
    dim3 block(DENSE_BLOCK_DIM, DENSE_BLOCK_DIM);
    dim3 grid((n_cells + DENSE_BLOCK_DIM - 1) / DENSE_BLOCK_DIM,
              (n_genes + DENSE_BLOCK_DIM - 1) / DENSE_BLOCK_DIM);
    qc_dense_sub_kernel<T>
        <<<grid, block, 0, stream>>>(data, sums_cells, mask, n_cells, n_genes);
    CUDA_CHECK_LAST_ERROR(qc_dense_sub_kernel);
}

template <typename T, typename IdxT, typename Device>
void def_sparse_qc_csc(nb::module_& m) {
    m.def(
        "sparse_qc_csc",
        [](gpu_array_c<const IdxT, Device> indptr,
           gpu_array_c<const IdxT, Device> index,
           gpu_array_c<const T, Device> data, gpu_array_c<T, Device> sums_cells,
           gpu_array_c<T, Device> sums_genes, gpu_array_c<int, Device> cell_ex,
           gpu_array_c<int, Device> gene_ex, int n_genes, bool assume_unsorted,
           std::uintptr_t stream) {
            return launch_qc_sparse<T, IdxT>(
                indptr.data(), index.data(), data.data(), sums_genes.data(),
                gene_ex.data(), sums_cells.data(), cell_ex.data(), n_genes,
                (int)sums_cells.shape(0), (long long)data.shape(0),
                assume_unsorted, (cudaStream_t)stream);
        },
        "indptr"_a, "index"_a, "data"_a, nb::kw_only(), "sums_cells"_a,
        "sums_genes"_a, "cell_ex"_a, "gene_ex"_a, "n_genes"_a,
        "assume_unsorted"_a = false, "stream"_a = 0);
}

template <typename T, typename IdxT, typename Device>
void def_sparse_qc_csr(nb::module_& m) {
    m.def(
        "sparse_qc_csr",
        [](gpu_array_c<const IdxT, Device> indptr,
           gpu_array_c<const IdxT, Device> index,
           gpu_array_c<const T, Device> data, gpu_array_c<T, Device> sums_cells,
           gpu_array_c<T, Device> sums_genes, gpu_array_c<int, Device> cell_ex,
           gpu_array_c<int, Device> gene_ex, int n_cells, bool assume_unsorted,
           std::uintptr_t stream) {
            return launch_qc_sparse<T, IdxT>(
                indptr.data(), index.data(), data.data(), sums_cells.data(),
                cell_ex.data(), sums_genes.data(), gene_ex.data(), n_cells,
                (int)sums_genes.shape(0), (long long)data.shape(0),
                assume_unsorted, (cudaStream_t)stream);
        },
        "indptr"_a, "index"_a, "data"_a, nb::kw_only(), "sums_cells"_a,
        "sums_genes"_a, "cell_ex"_a, "gene_ex"_a, "n_cells"_a,
        "assume_unsorted"_a = false, "stream"_a = 0);
}

// Minor-axis half of sparse_qc_csr alone (Dask chunks reduce the gene side
// separately from the cell side).
template <typename T, typename IdxT, typename Device>
void def_sparse_qc_genes(nb::module_& m) {
    m.def(
        "sparse_qc_genes",
        [](gpu_array_c<const IdxT, Device> indptr,
           gpu_array_c<const IdxT, Device> index,
           gpu_array_c<const T, Device> data, gpu_array_c<T, Device> sums_genes,
           gpu_array_c<int, Device> gene_ex, bool assume_unsorted,
           std::uintptr_t stream) {
            QcOp<T> op{data.data(), sums_genes.data(), gene_ex.data(), 0};
            return minor_reduce<IdxT>(
                indptr.data(), index.data(), op, (int)indptr.shape(0) - 1,
                (int)sums_genes.shape(0), (long long)data.shape(0),
                assume_unsorted, (cudaStream_t)stream);
        },
        "indptr"_a, "index"_a, "data"_a, nb::kw_only(), "sums_genes"_a,
        "gene_ex"_a, "assume_unsorted"_a = false, "stream"_a = 0);
}

template <typename T, typename Device>
void def_sparse_qc_dense(nb::module_& m) {
    m.def(
        "sparse_qc_dense",
        [](gpu_array_c<const T, Device> data, gpu_array_c<T, Device> sums_cells,
           gpu_array_c<T, Device> sums_genes, gpu_array_c<int, Device> cell_ex,
           gpu_array_c<int, Device> gene_ex, int n_cells, int n_genes,
           std::uintptr_t stream) {
            launch_qc_dense<T>(data.data(), sums_cells.data(),
                               sums_genes.data(), cell_ex.data(),
                               gene_ex.data(), n_cells, n_genes,
                               (cudaStream_t)stream);
        },
        "data"_a, nb::kw_only(), "sums_cells"_a, "sums_genes"_a, "cell_ex"_a,
        "gene_ex"_a, "n_cells"_a, "n_genes"_a, "stream"_a = 0);
}

// Masked genes (rows of the CSC) summed per cell (minor axis).
template <typename T, typename IdxT, typename Device>
void def_sparse_qc_csc_sub(nb::module_& m) {
    m.def(
        "sparse_qc_csc_sub",
        [](gpu_array_c<const IdxT, Device> indptr,
           gpu_array_c<const IdxT, Device> index,
           gpu_array_c<const T, Device> data, gpu_array_c<T, Device> sums_cells,
           gpu_array_c<const bool, Device> mask, int n_genes,
           bool assume_unsorted, std::uintptr_t stream) {
            MinorSumOp<T> op{data.data(), sums_cells.data(), mask.data(), 0};
            return minor_reduce<IdxT>(indptr.data(), index.data(), op, n_genes,
                                      (int)sums_cells.shape(0),
                                      (long long)data.shape(0), assume_unsorted,
                                      (cudaStream_t)stream);
        },
        "indptr"_a, "index"_a, "data"_a, nb::kw_only(), "sums_cells"_a,
        "mask"_a, "n_genes"_a, "assume_unsorted"_a = false, "stream"_a = 0);
}

// Masked genes summed per cell (major axis of the CSR); no atomics.
template <typename T, typename IdxT, typename Device>
void def_sparse_qc_csr_sub(nb::module_& m) {
    m.def(
        "sparse_qc_csr_sub",
        [](gpu_array_c<const IdxT, Device> indptr,
           gpu_array_c<const IdxT, Device> index,
           gpu_array_c<const T, Device> data, gpu_array_c<T, Device> sums_cells,
           gpu_array_c<const bool, Device> mask, int n_cells,
           std::uintptr_t stream) {
            row_reduce<T, IdxT>(indptr.data(), index.data(), data.data(),
                                mask.data(), sums_cells.data(), nullptr,
                                n_cells, (cudaStream_t)stream);
        },
        "indptr"_a, "index"_a, "data"_a, nb::kw_only(), "sums_cells"_a,
        "mask"_a, "n_cells"_a, "stream"_a = 0);
}

template <typename T, typename Device>
void def_sparse_qc_dense_sub(nb::module_& m) {
    m.def(
        "sparse_qc_dense_sub",
        [](gpu_array_c<const T, Device> data, gpu_array_c<T, Device> sums_cells,
           gpu_array_c<const bool, Device> mask, int n_cells, int n_genes,
           std::uintptr_t stream) {
            launch_qc_dense_sub<T>(data.data(), sums_cells.data(), mask.data(),
                                   n_cells, n_genes, (cudaStream_t)stream);
        },
        "data"_a, nb::kw_only(), "sums_cells"_a, "mask"_a, "n_cells"_a,
        "n_genes"_a, "stream"_a = 0);
}

template <typename Device>
void register_bindings(nb::module_& m) {
    def_sparse_qc_csc<float, int, Device>(m);
    def_sparse_qc_csc<float, long long, Device>(m);
    def_sparse_qc_csc<double, int, Device>(m);
    def_sparse_qc_csc<double, long long, Device>(m);

    def_sparse_qc_csr<float, int, Device>(m);
    def_sparse_qc_csr<float, long long, Device>(m);
    def_sparse_qc_csr<double, int, Device>(m);
    def_sparse_qc_csr<double, long long, Device>(m);

    def_sparse_qc_genes<float, int, Device>(m);
    def_sparse_qc_genes<float, long long, Device>(m);
    def_sparse_qc_genes<double, int, Device>(m);
    def_sparse_qc_genes<double, long long, Device>(m);

    def_sparse_qc_csc_sub<float, int, Device>(m);
    def_sparse_qc_csc_sub<float, long long, Device>(m);
    def_sparse_qc_csc_sub<double, int, Device>(m);
    def_sparse_qc_csc_sub<double, long long, Device>(m);

    def_sparse_qc_csr_sub<float, int, Device>(m);
    def_sparse_qc_csr_sub<float, long long, Device>(m);
    def_sparse_qc_csr_sub<double, int, Device>(m);
    def_sparse_qc_csr_sub<double, long long, Device>(m);

    def_sparse_qc_dense<float, Device>(m);
    def_sparse_qc_dense<double, Device>(m);
    def_sparse_qc_dense_sub<float, Device>(m);
    def_sparse_qc_dense_sub<double, Device>(m);
}

NB_MODULE(_qc_cuda, m) {
    REGISTER_GPU_BINDINGS(register_bindings, m);
    register_scratch_allocator(m);
}
