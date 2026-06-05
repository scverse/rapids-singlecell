#include <cuda_runtime.h>
#include "../../nb_types.h"

#include "kernels_normalize.cuh"

using namespace nb::literals;

constexpr unsigned WARP_SIZE = 32;
constexpr unsigned MAX_BLOCK_DIM = 256;

template <typename T>
static inline void launch_normalize(T* X, long long rows, long long cols,
                                    cudaStream_t stream) {
    unsigned block_dim = std::min(
        MAX_BLOCK_DIM, std::max(WARP_SIZE, ((unsigned)cols + WARP_SIZE - 1u) /
                                               WARP_SIZE * WARP_SIZE));
    dim3 block(block_dim);
    dim3 grid(rows);
    normalize_kernel<T><<<grid, block, 0, stream>>>(X, rows, cols);
    CUDA_CHECK_LAST_ERROR(normalize_kernel);
}

template <typename T>
static inline void launch_l2_row_normalize(const T* src, T* dst, int n_rows,
                                           int n_cols, cudaStream_t stream) {
    unsigned block_dim = std::min(
        MAX_BLOCK_DIM, std::max(WARP_SIZE, ((unsigned)n_cols + WARP_SIZE - 1u) /
                                               WARP_SIZE * WARP_SIZE));
    l2_row_normalize_kernel<T>
        <<<n_rows, block_dim, 0, stream>>>(src, dst, n_rows, n_cols);
    CUDA_CHECK_LAST_ERROR(l2_row_normalize_kernel);
}

template <typename T, typename Device>
void def_normalize(nb::module_& m) {
    m.def(
        "normalize",
        [](gpu_array_c<T, Device> X, long long rows, long long cols,
           std::uintptr_t stream) {
            launch_normalize<T>(X.data(), rows, cols, (cudaStream_t)stream);
        },
        "X"_a, nb::kw_only(), "rows"_a, "cols"_a, "stream"_a = 0);
}

template <typename T, typename Device>
void def_l2_row_normalize(nb::module_& m) {
    m.def(
        "l2_row_normalize",
        [](gpu_array_c<const T, Device> src, gpu_array_c<T, Device> dst,
           int n_rows, int n_cols, std::uintptr_t stream) {
            launch_l2_row_normalize<T>(src.data(), dst.data(), n_rows, n_cols,
                                       (cudaStream_t)stream);
        },
        "src"_a, nb::kw_only(), "dst"_a, "n_rows"_a, "n_cols"_a,
        "stream"_a = 0);
}

template <typename Device>
void register_bindings(nb::module_& m) {
    def_normalize<float, Device>(m);
    def_normalize<double, Device>(m);
    def_l2_row_normalize<float, Device>(m);
    def_l2_row_normalize<double, Device>(m);
}

NB_MODULE(_harmony_normalize_cuda, m) {
    REGISTER_GPU_BINDINGS(register_bindings, m);
}
