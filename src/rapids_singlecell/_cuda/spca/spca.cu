#include <cuda_runtime.h>
#include "../nb_types.h"

#include "kernels_spca.cuh"

using namespace nb::literals;

constexpr int GRAM_BLOCK_SIZE = 128;
constexpr int MATRIX_BLOCK_DIM = 32;
constexpr int ELEMENTWISE_BLOCK_SIZE = 32;

template <typename T, typename IdxT>
static inline void launch_gram_csr_upper(const IdxT* indptr, const IdxT* index,
                                         const T* data, size_t nrows,
                                         size_t ncols, T* out,
                                         cudaStream_t stream) {
    dim3 block(GRAM_BLOCK_SIZE);
    dim3 grid(strided_grid(nrows, 1));
    gram_csr_upper_kernel<T, IdxT>
        <<<grid, block, 0, stream>>>(indptr, index, data, nrows, ncols, out);
    CUDA_CHECK_LAST_ERROR(gram_csr_upper_kernel);
}

template <typename T>
static inline void launch_copy_upper_to_lower(T* out, size_t ncols,
                                              cudaStream_t stream) {
    dim3 block(MATRIX_BLOCK_DIM, MATRIX_BLOCK_DIM);
    dim3 grid(strided_grid(ncols, MATRIX_BLOCK_DIM),
              strided_grid_y(ncols, MATRIX_BLOCK_DIM));
    copy_upper_to_lower_kernel<T><<<grid, block, 0, stream>>>(out, ncols);
    CUDA_CHECK_LAST_ERROR(copy_upper_to_lower_kernel);
}

template <typename T>
static inline void launch_cov_from_gram(T* cov, const T* gram, const T* meanx,
                                        const T* meany, size_t ncols,
                                        cudaStream_t stream) {
    dim3 block(MATRIX_BLOCK_DIM, MATRIX_BLOCK_DIM);
    dim3 grid(strided_grid(ncols, MATRIX_BLOCK_DIM),
              strided_grid_y(ncols, MATRIX_BLOCK_DIM));
    cov_from_gram_kernel<T>
        <<<grid, block, 0, stream>>>(cov, gram, meanx, meany, ncols);
    CUDA_CHECK_LAST_ERROR(cov_from_gram_kernel);
}

template <typename IdxT>
static inline void launch_check_zero_genes(const IdxT* indices, int* genes,
                                           long long nnz, int num_genes,
                                           cudaStream_t stream) {
    if (nnz > 0) {
        dim3 block(ELEMENTWISE_BLOCK_SIZE);
        dim3 grid(strided_grid(nnz, ELEMENTWISE_BLOCK_SIZE));
        check_zero_genes_kernel<IdxT>
            <<<grid, block, 0, stream>>>(indices, genes, nnz, num_genes);
        CUDA_CHECK_LAST_ERROR(check_zero_genes_kernel);
    }
}

template <typename T, typename IdxT, typename Device>
void def_gram_csr_upper(nb::module_& m) {
    m.def(
        "gram_csr_upper",
        [](gpu_array_c<const IdxT, Device> indptr,
           gpu_array_c<const IdxT, Device> index,
           gpu_array_c<const T, Device> data, size_t nrows, size_t ncols,
           gpu_array_c<T, Device> out, std::uintptr_t stream) {
            launch_gram_csr_upper<T, IdxT>(indptr.data(), index.data(),
                                           data.data(), nrows, ncols,
                                           out.data(), (cudaStream_t)stream);
        },
        "indptr"_a, "index"_a, "data"_a, nb::kw_only(), "nrows"_a, "ncols"_a,
        "out"_a, "stream"_a = 0);
}

template <typename T, typename Device>
void def_copy_upper_to_lower(nb::module_& m) {
    m.def(
        "copy_upper_to_lower",
        [](gpu_array_c<T, Device> out, size_t ncols, std::uintptr_t stream) {
            launch_copy_upper_to_lower<T>(out.data(), ncols,
                                          (cudaStream_t)stream);
        },
        nb::kw_only(), "out"_a, "ncols"_a, "stream"_a = 0);
}

template <typename T, typename Device>
void def_cov_from_gram(nb::module_& m) {
    m.def(
        "cov_from_gram",
        [](gpu_array_c<const T, Device> gram,
           gpu_array_c<const T, Device> meanx,
           gpu_array_c<const T, Device> meany, gpu_array_c<T, Device> cov,
           size_t ncols, std::uintptr_t stream) {
            launch_cov_from_gram<T>(cov.data(), gram.data(), meanx.data(),
                                    meany.data(), ncols, (cudaStream_t)stream);
        },
        "gram"_a, "meanx"_a, "meany"_a, nb::kw_only(), "cov"_a, "ncols"_a,
        "stream"_a = 0);
}

template <typename IdxT, typename Device>
void def_check_zero_genes(nb::module_& m) {
    m.def(
        "check_zero_genes",
        [](gpu_array_c<const IdxT, Device> indices,
           gpu_array_c<int, Device> out, long long nnz, int num_genes,
           std::uintptr_t stream) {
            launch_check_zero_genes<IdxT>(indices.data(), out.data(), nnz,
                                          num_genes, (cudaStream_t)stream);
        },
        "indices"_a, nb::kw_only(), "out"_a, "nnz"_a, "num_genes"_a,
        "stream"_a = 0);
}

template <typename Device>
void register_bindings(nb::module_& m) {
    def_gram_csr_upper<float, int, Device>(m);
    def_gram_csr_upper<float, long long, Device>(m);
    def_gram_csr_upper<double, int, Device>(m);
    def_gram_csr_upper<double, long long, Device>(m);

    def_copy_upper_to_lower<float, Device>(m);
    def_copy_upper_to_lower<double, Device>(m);

    def_cov_from_gram<float, Device>(m);
    def_cov_from_gram<double, Device>(m);

    def_check_zero_genes<int, Device>(m);
    def_check_zero_genes<long long, Device>(m);
}

NB_MODULE(_spca_cuda, m) {
    REGISTER_GPU_BINDINGS(register_bindings, m);
}
