#include <algorithm>
#include <cuda_runtime.h>
#include "../../nb_types.h"

#include "kernels_colsum.cuh"

using namespace nb::literals;

constexpr unsigned WARP_SIZE = 32;
constexpr int MAX_BLOCK_DIM_1D = 1024;
constexpr int BLOCKS_PER_SM = 8;
constexpr int ATOMIC_BLOCKS_PER_SM = 4;

template <typename T>
static inline void launch_colsum(const T* A, T* out, size_t rows, size_t cols,
                                 cudaStream_t stream) {
    int device;
    cudaGetDevice(&device);
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, device);
    int max_blocks = prop.multiProcessorCount * BLOCKS_PER_SM;

    // Scale thread count with rows, capped at MAX_BLOCK_DIM_1D, minimum
    // WARP_SIZE
    int threads = std::min(
        MAX_BLOCK_DIM_1D,
        std::max((int)WARP_SIZE,
                 (int)((rows + WARP_SIZE - 1) / WARP_SIZE) * (int)WARP_SIZE));
    int blocks = std::min((int)cols, max_blocks);
    colsum_kernel<T><<<blocks, threads, 0, stream>>>(A, out, rows, cols);
    CUDA_CHECK_LAST_ERROR(colsum_kernel);
}

template <typename T>
static inline void launch_colsum_atomic(const T* A, T* out, size_t rows,
                                        size_t cols, cudaStream_t stream) {
    int device;
    cudaGetDevice(&device);
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, device);
    int n_sm = prop.multiProcessorCount;

    int col_tiles = (int)((cols + WARP_SIZE - 1) / WARP_SIZE);
    int target_row_tiles =
        std::max(1, n_sm * ATOMIC_BLOCKS_PER_SM / std::max(1, col_tiles));
    size_t rows_per_tile = std::max(
        (size_t)WARP_SIZE, (rows + target_row_tiles - 1) / target_row_tiles);

    int row_tiles = (int)((rows + rows_per_tile - 1) / rows_per_tile);
    dim3 grid(col_tiles, row_tiles);
    dim3 threads(WARP_SIZE, WARP_SIZE);
    colsum_atomic_kernel<T>
        <<<grid, threads, 0, stream>>>(A, out, rows, cols, rows_per_tile);
    CUDA_CHECK_LAST_ERROR(colsum_atomic_kernel);
}

template <typename T, typename Device>
void def_colsum(nb::module_& m) {
    m.def(
        "colsum",
        [](gpu_array_c<const T, Device> A, gpu_array_c<T, Device> out,
           size_t rows, size_t cols, std::uintptr_t stream) {
            launch_colsum<T>(A.data(), out.data(), rows, cols,
                             (cudaStream_t)stream);
        },
        "A"_a, nb::kw_only(), "out"_a, "rows"_a, "cols"_a, "stream"_a = 0);
}

template <typename T, typename Device>
void def_colsum_atomic(nb::module_& m) {
    m.def(
        "colsum_atomic",
        [](gpu_array_c<const T, Device> A, gpu_array_c<T, Device> out,
           size_t rows, size_t cols, std::uintptr_t stream) {
            launch_colsum_atomic<T>(A.data(), out.data(), rows, cols,
                                    (cudaStream_t)stream);
        },
        "A"_a, nb::kw_only(), "out"_a, "rows"_a, "cols"_a, "stream"_a = 0);
}

template <typename Device>
void register_bindings(nb::module_& m) {
    def_colsum<float, Device>(m);
    def_colsum<double, Device>(m);
    def_colsum<int, Device>(m);
    def_colsum_atomic<float, Device>(m);
    def_colsum_atomic<double, Device>(m);
    def_colsum_atomic<int, Device>(m);
}

NB_MODULE(_harmony_colsum_cuda, m) {
    REGISTER_GPU_BINDINGS(register_bindings, m);
}
