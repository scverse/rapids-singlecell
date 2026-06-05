#include <cuda_runtime.h>
#include "../nb_types.h"

#include "kernels_wilcoxon_binned.cuh"

using namespace nb::literals;

constexpr int BLOCK_SIZE = 256;

template <typename T>
static inline void launch_dense_hist(const T* X, const int* gcodes,
                                     unsigned int* hist, int n_cells,
                                     int n_genes, int n_groups, int n_bins,
                                     double bin_low, double inv_bin_width,
                                     cudaStream_t stream) {
    dim3 block(BLOCK_SIZE);
    dim3 grid(n_genes);
    dense_hist_kernel<T><<<grid, block, 0, stream>>>(X, gcodes, hist, n_cells,
                                                     n_genes, n_groups, n_bins,
                                                     bin_low, inv_bin_width);
    CUDA_CHECK_LAST_ERROR(dense_hist_kernel);
}

template <typename T, typename IdxT>
static inline void launch_csr_hist(const T* data, const IdxT* indices,
                                   const IdxT* indptr, const int* gcodes,
                                   unsigned int* hist, int n_cells, int n_genes,
                                   int n_groups, int n_bins, double bin_low,
                                   double inv_bin_width, int gene_start,
                                   cudaStream_t stream) {
    dim3 block(BLOCK_SIZE);
    dim3 grid(n_cells);
    csr_hist_kernel<T, IdxT><<<grid, block, 0, stream>>>(
        data, indices, indptr, gcodes, hist, n_cells, n_genes, n_groups, n_bins,
        bin_low, inv_bin_width, gene_start);
    CUDA_CHECK_LAST_ERROR(csr_hist_kernel);
}

template <typename T, typename IdxT>
static inline void launch_csc_hist(const T* data, const IdxT* indices,
                                   const IdxT* indptr, const int* gcodes,
                                   unsigned int* hist, int n_cells, int n_genes,
                                   int n_groups, int n_bins, double bin_low,
                                   double inv_bin_width, int gene_start,
                                   cudaStream_t stream) {
    dim3 block(BLOCK_SIZE);
    dim3 grid(n_genes);
    csc_hist_kernel<T, IdxT><<<grid, block, 0, stream>>>(
        data, indices, indptr, gcodes, hist, n_cells, n_genes, n_groups, n_bins,
        bin_low, inv_bin_width, gene_start);
    CUDA_CHECK_LAST_ERROR(csc_hist_kernel);
}

template <typename T, typename Device>
void def_dense_hist(nb::module_& m) {
    m.def(
        "dense_hist",
        [](gpu_array_f<const T, Device> X,
           gpu_array_c<const int, Device> gcodes,
           gpu_array_c<unsigned int, Device> hist, int n_cells, int n_genes,
           int n_groups, int n_bins, double bin_low, double inv_bin_width,
           std::uintptr_t stream) {
            launch_dense_hist<T>(X.data(), gcodes.data(), hist.data(), n_cells,
                                 n_genes, n_groups, n_bins, bin_low,
                                 inv_bin_width, (cudaStream_t)stream);
        },
        "X"_a, "gcodes"_a, "hist"_a, nb::kw_only(), "n_cells"_a, "n_genes"_a,
        "n_groups"_a, "n_bins"_a, "bin_low"_a, "inv_bin_width"_a,
        "stream"_a = 0);
}

template <typename T, typename IdxT, typename Device>
void def_csr_hist(nb::module_& m) {
    m.def(
        "csr_hist",
        [](gpu_array_c<const T, Device> data,
           gpu_array_c<const IdxT, Device> indices,
           gpu_array_c<const IdxT, Device> indptr,
           gpu_array_c<const int, Device> gcodes,
           gpu_array_c<unsigned int, Device> hist, int n_cells, int n_genes,
           int n_groups, int n_bins, double bin_low, double inv_bin_width,
           int gene_start, std::uintptr_t stream) {
            launch_csr_hist<T, IdxT>(
                data.data(), indices.data(), indptr.data(), gcodes.data(),
                hist.data(), n_cells, n_genes, n_groups, n_bins, bin_low,
                inv_bin_width, gene_start, (cudaStream_t)stream);
        },
        "data"_a, "indices"_a, "indptr"_a, "gcodes"_a, "hist"_a, nb::kw_only(),
        "n_cells"_a, "n_genes"_a, "n_groups"_a, "n_bins"_a, "bin_low"_a,
        "inv_bin_width"_a, "gene_start"_a, "stream"_a = 0);
}

template <typename T, typename IdxT, typename Device>
void def_csc_hist(nb::module_& m) {
    m.def(
        "csc_hist",
        [](gpu_array_c<const T, Device> data,
           gpu_array_c<const IdxT, Device> indices,
           gpu_array_c<const IdxT, Device> indptr,
           gpu_array_c<const int, Device> gcodes,
           gpu_array_c<unsigned int, Device> hist, int n_cells, int n_genes,
           int n_groups, int n_bins, double bin_low, double inv_bin_width,
           int gene_start, std::uintptr_t stream) {
            launch_csc_hist<T, IdxT>(
                data.data(), indices.data(), indptr.data(), gcodes.data(),
                hist.data(), n_cells, n_genes, n_groups, n_bins, bin_low,
                inv_bin_width, gene_start, (cudaStream_t)stream);
        },
        "data"_a, "indices"_a, "indptr"_a, "gcodes"_a, "hist"_a, nb::kw_only(),
        "n_cells"_a, "n_genes"_a, "n_groups"_a, "n_bins"_a, "bin_low"_a,
        "inv_bin_width"_a, "gene_start"_a, "stream"_a = 0);
}

template <typename Device>
void register_bindings(nb::module_& m) {
    def_dense_hist<float, Device>(m);
    def_dense_hist<double, Device>(m);

    def_csr_hist<float, int, Device>(m);
    def_csr_hist<float, long long, Device>(m);
    def_csr_hist<double, int, Device>(m);
    def_csr_hist<double, long long, Device>(m);

    def_csc_hist<float, int, Device>(m);
    def_csc_hist<float, long long, Device>(m);
    def_csc_hist<double, int, Device>(m);
    def_csc_hist<double, long long, Device>(m);
}

NB_MODULE(_wilcoxon_binned_cuda, m) {
    REGISTER_GPU_BINDINGS(register_bindings, m);
}
