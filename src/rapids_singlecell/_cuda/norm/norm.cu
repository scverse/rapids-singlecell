#include <cuda_runtime.h>
#include "../nb_types.h"

#include "kernels_norm.cuh"

using namespace nb::literals;

constexpr int BLOCK_SIZE = 256;

template <typename T>
static inline void launch_dense_row_scale(T* data, int nrows, int ncols,
                                          T target_sum, cudaStream_t stream) {
    dim3 block(BLOCK_SIZE);
    dim3 grid(nrows);
    dense_row_scale_kernel<T>
        <<<grid, block, 0, stream>>>(data, nrows, ncols, target_sum);
    CUDA_CHECK_LAST_ERROR(dense_row_scale_kernel);
}

template <typename T, typename IdxT>
static inline void launch_csr_row_scale(const IdxT* indptr, T* data, int nrows,
                                        T target_sum, cudaStream_t stream) {
    dim3 block(BLOCK_SIZE);
    dim3 grid(nrows);
    csr_row_scale_kernel<T, IdxT>
        <<<grid, block, 0, stream>>>(indptr, data, nrows, target_sum);
    CUDA_CHECK_LAST_ERROR(csr_row_scale_kernel);
}

template <typename T, typename IdxT>
static inline void launch_csr_sum_major(const IdxT* indptr, const T* data,
                                        T* sums, int major,
                                        cudaStream_t stream) {
    dim3 block(BLOCK_SIZE);
    dim3 grid(major);
    csr_sum_major_kernel<T, IdxT>
        <<<grid, block, 0, stream>>>(indptr, data, sums, major);
    CUDA_CHECK_LAST_ERROR(csr_sum_major_kernel);
}

template <typename T, typename IdxT>
static inline void launch_find_hi_genes_csr(const IdxT* indptr,
                                            const IdxT* indices, const T* data,
                                            bool* gene_is_hi, T max_fraction,
                                            int nrows, cudaStream_t stream) {
    dim3 block(BLOCK_SIZE);
    dim3 grid(nrows);
    find_hi_genes_csr_kernel<T, IdxT><<<grid, block, 0, stream>>>(
        indptr, indices, data, gene_is_hi, max_fraction, nrows);
    CUDA_CHECK_LAST_ERROR(find_hi_genes_csr_kernel);
}

template <typename T, typename IdxT>
static inline void launch_masked_mul_csr(const IdxT* indptr,
                                         const IdxT* indices, T* data,
                                         const bool* gene_mask, int nrows,
                                         T tsum, cudaStream_t stream) {
    dim3 block(BLOCK_SIZE);
    dim3 grid(nrows);
    masked_mul_csr_kernel<T, IdxT><<<grid, block, 0, stream>>>(
        indptr, indices, data, gene_mask, nrows, tsum);
    CUDA_CHECK_LAST_ERROR(masked_mul_csr_kernel);
}

template <typename T, typename IdxT>
static inline void launch_masked_sum_major(const IdxT* indptr,
                                           const IdxT* indices, const T* data,
                                           const bool* gene_mask, T* sums,
                                           int major, cudaStream_t stream) {
    dim3 block(BLOCK_SIZE);
    dim3 grid(major);
    masked_sum_major_kernel<T, IdxT><<<grid, block, 0, stream>>>(
        indptr, indices, data, gene_mask, sums, major);
    CUDA_CHECK_LAST_ERROR(masked_sum_major_kernel);
}

template <typename T, typename IdxT>
static inline void launch_prescaled_mul_csr(const IdxT* indptr, T* data,
                                            const T* scales, int nrows,
                                            cudaStream_t stream) {
    dim3 block(BLOCK_SIZE);
    dim3 grid(nrows);
    prescaled_mul_csr_kernel<T, IdxT>
        <<<grid, block, 0, stream>>>(indptr, data, scales, nrows);
    CUDA_CHECK_LAST_ERROR(prescaled_mul_csr_kernel);
}

template <typename T>
static inline void launch_prescaled_mul_dense(T* data, const T* scales,
                                              int nrows, int ncols,
                                              cudaStream_t stream) {
    dim3 block(BLOCK_SIZE);
    dim3 grid(nrows);
    prescaled_mul_dense_kernel<T>
        <<<grid, block, 0, stream>>>(data, scales, nrows, ncols);
    CUDA_CHECK_LAST_ERROR(prescaled_mul_dense_kernel);
}

template <typename T, typename IdxT, typename Device>
void def_mul_csr(nb::module_& m) {
    m.def(
        "mul_csr",
        [](gpu_array_c<const IdxT, Device> indptr, gpu_array_c<T, Device> data,
           int nrows, T target_sum, std::uintptr_t stream) {
            launch_csr_row_scale<T, IdxT>(indptr.data(), data.data(), nrows,
                                          target_sum, (cudaStream_t)stream);
        },
        "indptr"_a, "data"_a, nb::kw_only(), "nrows"_a, "target_sum"_a,
        "stream"_a = 0);
}

template <typename T, typename IdxT, typename Device>
void def_sum_major(nb::module_& m) {
    m.def(
        "sum_major",
        [](gpu_array_c<const IdxT, Device> indptr,
           gpu_array_c<const T, Device> data, gpu_array_c<T, Device> sums,
           int major, std::uintptr_t stream) {
            launch_csr_sum_major<T, IdxT>(indptr.data(), data.data(),
                                          sums.data(), major,
                                          (cudaStream_t)stream);
        },
        "indptr"_a, "data"_a, nb::kw_only(), "sums"_a, "major"_a,
        "stream"_a = 0);
}

template <typename T, typename IdxT, typename Device>
void def_find_hi_genes_csr(nb::module_& m) {
    m.def(
        "find_hi_genes_csr",
        [](gpu_array_c<const IdxT, Device> indptr,
           gpu_array_c<const IdxT, Device> indices,
           gpu_array_c<const T, Device> data,
           gpu_array_c<bool, Device> gene_is_hi, T max_fraction, int nrows,
           std::uintptr_t stream) {
            launch_find_hi_genes_csr<T, IdxT>(
                indptr.data(), indices.data(), data.data(), gene_is_hi.data(),
                max_fraction, nrows, (cudaStream_t)stream);
        },
        "indptr"_a, "indices"_a, "data"_a, nb::kw_only(), "gene_is_hi"_a,
        "max_fraction"_a, "nrows"_a, "stream"_a = 0);
}

template <typename T, typename IdxT, typename Device>
void def_masked_mul_csr(nb::module_& m) {
    m.def(
        "masked_mul_csr",
        [](gpu_array_c<const IdxT, Device> indptr,
           gpu_array_c<const IdxT, Device> indices, gpu_array_c<T, Device> data,
           gpu_array_c<const bool, Device> gene_mask, int nrows, T tsum,
           std::uintptr_t stream) {
            launch_masked_mul_csr<T, IdxT>(indptr.data(), indices.data(),
                                           data.data(), gene_mask.data(), nrows,
                                           tsum, (cudaStream_t)stream);
        },
        "indptr"_a, "indices"_a, "data"_a, nb::kw_only(), "gene_mask"_a,
        "nrows"_a, "tsum"_a, "stream"_a = 0);
}

template <typename T, typename IdxT, typename Device>
void def_masked_sum_major(nb::module_& m) {
    m.def(
        "masked_sum_major",
        [](gpu_array_c<const IdxT, Device> indptr,
           gpu_array_c<const IdxT, Device> indices,
           gpu_array_c<const T, Device> data,
           gpu_array_c<const bool, Device> gene_mask,
           gpu_array_c<T, Device> sums, int major, std::uintptr_t stream) {
            launch_masked_sum_major<T, IdxT>(
                indptr.data(), indices.data(), data.data(), gene_mask.data(),
                sums.data(), major, (cudaStream_t)stream);
        },
        "indptr"_a, "indices"_a, "data"_a, nb::kw_only(), "gene_mask"_a,
        "sums"_a, "major"_a, "stream"_a = 0);
}

template <typename T, typename IdxT, typename Device>
void def_prescaled_mul_csr(nb::module_& m) {
    m.def(
        "prescaled_mul_csr",
        [](gpu_array_c<const IdxT, Device> indptr, gpu_array_c<T, Device> data,
           gpu_array_c<const T, Device> scales, int nrows,
           std::uintptr_t stream) {
            launch_prescaled_mul_csr<T, IdxT>(indptr.data(), data.data(),
                                              scales.data(), nrows,
                                              (cudaStream_t)stream);
        },
        "indptr"_a, "data"_a, nb::kw_only(), "scales"_a, "nrows"_a,
        "stream"_a = 0);
}

template <typename T, typename Device>
void def_mul_dense(nb::module_& m) {
    m.def(
        "mul_dense",
        [](gpu_array_c<T, Device> data, int nrows, int ncols, T target_sum,
           std::uintptr_t stream) {
            launch_dense_row_scale<T>(data.data(), nrows, ncols, target_sum,
                                      (cudaStream_t)stream);
        },
        "data"_a, nb::kw_only(), "nrows"_a, "ncols"_a, "target_sum"_a,
        "stream"_a = 0);
}

template <typename T, typename Device>
void def_prescaled_mul_dense(nb::module_& m) {
    m.def(
        "prescaled_mul_dense",
        [](gpu_array_c<T, Device> data, gpu_array_c<const T, Device> scales,
           int nrows, int ncols, std::uintptr_t stream) {
            launch_prescaled_mul_dense<T>(data.data(), scales.data(), nrows,
                                          ncols, (cudaStream_t)stream);
        },
        "data"_a, nb::kw_only(), "scales"_a, "nrows"_a, "ncols"_a,
        "stream"_a = 0);
}

template <typename Device>
void register_bindings(nb::module_& m) {
    def_mul_dense<float, Device>(m);
    def_mul_dense<double, Device>(m);

    def_mul_csr<float, int, Device>(m);
    def_mul_csr<float, long long, Device>(m);
    def_mul_csr<double, int, Device>(m);
    def_mul_csr<double, long long, Device>(m);

    def_sum_major<float, int, Device>(m);
    def_sum_major<float, long long, Device>(m);
    def_sum_major<double, int, Device>(m);
    def_sum_major<double, long long, Device>(m);

    def_find_hi_genes_csr<float, int, Device>(m);
    def_find_hi_genes_csr<float, long long, Device>(m);
    def_find_hi_genes_csr<double, int, Device>(m);
    def_find_hi_genes_csr<double, long long, Device>(m);

    def_masked_mul_csr<float, int, Device>(m);
    def_masked_mul_csr<float, long long, Device>(m);
    def_masked_mul_csr<double, int, Device>(m);
    def_masked_mul_csr<double, long long, Device>(m);

    def_masked_sum_major<float, int, Device>(m);
    def_masked_sum_major<float, long long, Device>(m);
    def_masked_sum_major<double, int, Device>(m);
    def_masked_sum_major<double, long long, Device>(m);

    def_prescaled_mul_csr<float, int, Device>(m);
    def_prescaled_mul_csr<float, long long, Device>(m);
    def_prescaled_mul_csr<double, int, Device>(m);
    def_prescaled_mul_csr<double, long long, Device>(m);

    def_prescaled_mul_dense<float, Device>(m);
    def_prescaled_mul_dense<double, Device>(m);
}

NB_MODULE(_norm_cuda, m) {
    REGISTER_GPU_BINDINGS(register_bindings, m);
}
