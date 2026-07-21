#include <cuda_runtime.h>

#include <stdexcept>

#include "../../nb_types.h"

#include "kernels_scatter.cuh"

using namespace nb::literals;

constexpr int BLOCK_DIM_1D = 256;

template <typename T>
static inline void launch_scatter_add(const T* v, const int* cats,
                                      size_t n_cells, size_t n_pcs,
                                      int n_covariates, int switcher, T* a,
                                      cudaStream_t stream) {
    if (n_covariates < 1)
        throw std::invalid_argument(
            "scatter_add requires at least one covariate");

    dim3 block(BLOCK_DIM_1D);
    size_t N = n_cells * n_pcs;
    dim3 grid(strided_grid((long long)N, BLOCK_DIM_1D));
    if (n_covariates == 1) {
        scatter_add_kernel<T, 1><<<grid, block, 0, stream>>>(
            v, cats, n_cells, n_pcs, n_covariates, switcher, a);
    } else if (n_covariates == 2) {
        scatter_add_kernel<T, 2><<<grid, block, 0, stream>>>(
            v, cats, n_cells, n_pcs, n_covariates, switcher, a);
    } else if (n_covariates == 3) {
        scatter_add_kernel<T, 3><<<grid, block, 0, stream>>>(
            v, cats, n_cells, n_pcs, n_covariates, switcher, a);
    } else {
        scatter_add_kernel<T, 0><<<grid, block, 0, stream>>>(
            v, cats, n_cells, n_pcs, n_covariates, switcher, a);
    }
    CUDA_CHECK_LAST_ERROR(scatter_add_kernel);
}

template <typename T>
static inline void launch_scatter_add_shared(const T* v, const int* cats,
                                             int n_cells, int n_pcs,
                                             int n_batches, int n_covariates,
                                             int switcher, T* a, int n_blocks,
                                             cudaStream_t stream) {
    if (n_covariates < 1)
        throw std::invalid_argument(
            "scatter_add_shared requires at least one covariate");

    dim3 block(BLOCK_DIM_1D);
    dim3 grid(n_blocks);
    size_t shared_mem = (size_t)n_batches * n_pcs * sizeof(T);
    if (n_covariates == 1) {
        scatter_add_shared_kernel<T, 1><<<grid, block, shared_mem, stream>>>(
            v, cats, n_cells, n_pcs, n_batches, n_covariates, switcher, a);
    } else if (n_covariates == 2) {
        scatter_add_shared_kernel<T, 2><<<grid, block, shared_mem, stream>>>(
            v, cats, n_cells, n_pcs, n_batches, n_covariates, switcher, a);
    } else if (n_covariates == 3) {
        scatter_add_shared_kernel<T, 3><<<grid, block, shared_mem, stream>>>(
            v, cats, n_cells, n_pcs, n_batches, n_covariates, switcher, a);
    } else {
        scatter_add_shared_kernel<T, 0><<<grid, block, shared_mem, stream>>>(
            v, cats, n_cells, n_pcs, n_batches, n_covariates, switcher, a);
    }
    CUDA_CHECK_LAST_ERROR(scatter_add_shared_kernel);
}

template <typename T>
static inline void launch_gather_rows(const T* src, const int* idx, T* dst,
                                      int n_rows, int n_cols,
                                      cudaStream_t stream) {
    size_t n = (size_t)n_rows * n_cols;
    gather_rows_kernel<T>
        <<<strided_grid((long long)n, BLOCK_DIM_1D), BLOCK_DIM_1D, 0, stream>>>(
            src, idx, dst, n_rows, n_cols);
    CUDA_CHECK_LAST_ERROR(gather_rows_kernel);
}

template <typename T>
static inline void launch_scatter_rows(const T* src, const int* idx, T* dst,
                                       int n_rows, int n_cols,
                                       cudaStream_t stream) {
    size_t n = (size_t)n_rows * n_cols;
    scatter_rows_kernel<T>
        <<<strided_grid((long long)n, BLOCK_DIM_1D), BLOCK_DIM_1D, 0, stream>>>(
            src, idx, dst, n_rows, n_cols);
    CUDA_CHECK_LAST_ERROR(scatter_rows_kernel);
}

static inline void launch_gather_int(const int* src, const int* idx, int* dst,
                                     int n, cudaStream_t stream) {
    gather_int_kernel<<<strided_grid(n, BLOCK_DIM_1D), BLOCK_DIM_1D, 0,
                        stream>>>(src, idx, dst, n);
    CUDA_CHECK_LAST_ERROR(gather_int_kernel);
}

template <typename T, typename Device>
void def_scatter_add(nb::module_& m) {
    m.def(
        "scatter_add",
        [](gpu_array_c<const T, Device> v, gpu_array_c<const int, Device> cats,
           size_t n_cells, size_t n_pcs, int n_covariates, int switcher,
           gpu_array_c<T, Device> a, std::uintptr_t stream) {
            launch_scatter_add<T>(v.data(), cats.data(), n_cells, n_pcs,
                                  n_covariates, switcher, a.data(),
                                  (cudaStream_t)stream);
        },
        "v"_a, nb::kw_only(), "cats"_a, "n_cells"_a, "n_pcs"_a,
        "n_covariates"_a = 1, "switcher"_a, "a"_a, "stream"_a = 0);
}

template <typename T, typename Device>
void def_scatter_add_shared(nb::module_& m) {
    m.def(
        "scatter_add_shared",
        [](gpu_array_c<const T, Device> v, gpu_array_c<const int, Device> cats,
           int n_cells, int n_pcs, int n_batches, int n_covariates,
           int switcher, gpu_array_c<T, Device> a, int n_blocks,
           std::uintptr_t stream) {
            launch_scatter_add_shared<T>(
                v.data(), cats.data(), n_cells, n_pcs, n_batches, n_covariates,
                switcher, a.data(), n_blocks, (cudaStream_t)stream);
        },
        "v"_a, nb::kw_only(), "cats"_a, "n_cells"_a, "n_pcs"_a, "n_batches"_a,
        "n_covariates"_a = 1, "switcher"_a, "a"_a, "n_blocks"_a,
        "stream"_a = 0);
}

template <typename T, typename Device>
void def_gather_rows(nb::module_& m) {
    m.def(
        "gather_rows",
        [](gpu_array_c<const T, Device> src, gpu_array_c<const int, Device> idx,
           gpu_array_c<T, Device> dst, int n_rows, int n_cols,
           std::uintptr_t stream) {
            launch_gather_rows<T>(src.data(), idx.data(), dst.data(), n_rows,
                                  n_cols, (cudaStream_t)stream);
        },
        "src"_a, nb::kw_only(), "idx"_a, "dst"_a, "n_rows"_a, "n_cols"_a,
        "stream"_a = 0);
}

template <typename T, typename Device>
void def_scatter_rows(nb::module_& m) {
    m.def(
        "scatter_rows",
        [](gpu_array_c<const T, Device> src, gpu_array_c<const int, Device> idx,
           gpu_array_c<T, Device> dst, int n_rows, int n_cols,
           std::uintptr_t stream) {
            launch_scatter_rows<T>(src.data(), idx.data(), dst.data(), n_rows,
                                   n_cols, (cudaStream_t)stream);
        },
        "src"_a, nb::kw_only(), "idx"_a, "dst"_a, "n_rows"_a, "n_cols"_a,
        "stream"_a = 0);
}

template <typename Device>
void register_bindings(nb::module_& m) {
    def_scatter_add<float, Device>(m);
    def_scatter_add<double, Device>(m);
    def_scatter_add_shared<float, Device>(m);
    def_scatter_add_shared<double, Device>(m);
    def_gather_rows<float, Device>(m);
    def_gather_rows<double, Device>(m);
    def_scatter_rows<float, Device>(m);
    def_scatter_rows<double, Device>(m);

    // gather_int is not overloaded (int only)
    m.def(
        "gather_int",
        [](gpu_array_c<const int, Device> src,
           gpu_array_c<const int, Device> idx, gpu_array_c<int, Device> dst,
           int n, std::uintptr_t stream) {
            launch_gather_int(src.data(), idx.data(), dst.data(), n,
                              (cudaStream_t)stream);
        },
        "src"_a, nb::kw_only(), "idx"_a, "dst"_a, "n"_a, "stream"_a = 0);
}

NB_MODULE(_harmony_scatter_cuda, m) {
    REGISTER_GPU_BINDINGS(register_bindings, m);
}
