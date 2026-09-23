#include <cuda_runtime.h>

#include "../../nb_types.h"

#include "kernels_scatter.cuh"

using namespace nb::literals;

constexpr int BLOCK_DIM_1D = 256;

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
