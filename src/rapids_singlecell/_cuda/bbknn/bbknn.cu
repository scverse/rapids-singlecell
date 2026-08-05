#include <cuda_runtime.h>
#include "../nb_types.h"

#include "kernels_bbknn.cuh"

using namespace nb::literals;

constexpr int BLOCK_SIZE = 64;
// Block-cooperative sort kernel: BLOCK_THREADS * ITEMS_PER_THREAD = 2048.
// Rows larger than this must use the per-thread kernel (kernel 1).
constexpr int SORT_BLOCK_THREADS = 128;
constexpr int SORT_ITEMS_PER_THREAD = 16;
constexpr int SORT_TILE_SIZE = SORT_BLOCK_THREADS * SORT_ITEMS_PER_THREAD;

static inline void launch_find_top_k_per_row(const float* data,
                                             const int* indptr, int n_rows,
                                             int trim, float* vals,
                                             cudaStream_t stream) {
    // Each thread keeps its row's top-`trim` values in shared memory, so the
    // per-block shared-mem request is BLOCK_SIZE * trim * sizeof(float).
    // The default per-block shared-mem cap is ~48 KB; halve the block size
    // until the request fits so the launch succeeds for any reasonable trim.
    constexpr size_t SHARED_MEM_BUDGET = 48 * 1024;
    const size_t per_thread_bytes = static_cast<size_t>(trim) * sizeof(float);
    int block_size = BLOCK_SIZE;
    while (block_size > 1 &&
           static_cast<size_t>(block_size) * per_thread_bytes >
               SHARED_MEM_BUDGET) {
        block_size /= 2;
    }
    if (static_cast<size_t>(block_size) * per_thread_bytes >
        SHARED_MEM_BUDGET) {
        throw std::runtime_error(
            "find_top_k_per_row: trim too large for shared-memory budget; "
            "use find_top_k_per_row_sorted instead");
    }
    dim3 block(block_size);
    dim3 grid((n_rows + block_size - 1) / block_size);
    size_t shared_mem_size = static_cast<size_t>(block_size) * per_thread_bytes;
    find_top_k_per_row_kernel<<<grid, block, shared_mem_size, stream>>>(
        data, indptr, n_rows, trim, vals);
    CUDA_CHECK_LAST_ERROR(find_top_k_per_row_kernel);
}

static inline void launch_find_top_k_per_row_sorted(const float* data,
                                                    const int* indptr,
                                                    int n_rows, int trim,
                                                    float* vals,
                                                    cudaStream_t stream) {
    dim3 block(SORT_BLOCK_THREADS);
    dim3 grid(n_rows);
    find_top_k_per_row_sorted_kernel<SORT_BLOCK_THREADS, SORT_ITEMS_PER_THREAD>
        <<<grid, block, 0, stream>>>(data, indptr, n_rows, trim, vals);
    CUDA_CHECK_LAST_ERROR(find_top_k_per_row_sorted_kernel);
}

static inline void launch_cut_smaller(int* indptr, int* index, float* data,
                                      float* vals, int n_rows,
                                      cudaStream_t stream) {
    dim3 grid(n_rows);
    dim3 block(BLOCK_SIZE);
    cut_smaller_kernel<<<grid, block, 0, stream>>>(indptr, index, data, vals,
                                                   n_rows);
    CUDA_CHECK_LAST_ERROR(cut_smaller_kernel);
}

template <typename Device>
void register_bindings(nb::module_& m) {
    m.def(
        "find_top_k_per_row",
        [](gpu_array_c<const float, Device> data,
           gpu_array_c<const int, Device> indptr, int n_rows, int trim,
           gpu_array_c<float, Device> vals, std::uintptr_t stream) {
            launch_find_top_k_per_row(data.data(), indptr.data(), n_rows, trim,
                                      vals.data(), (cudaStream_t)stream);
        },
        "data"_a, "indptr"_a, nb::kw_only(), "n_rows"_a, "trim"_a, "vals"_a,
        "stream"_a = 0);

    m.def(
        "find_top_k_per_row_sorted",
        [](gpu_array_c<const float, Device> data,
           gpu_array_c<const int, Device> indptr, int n_rows, int trim,
           gpu_array_c<float, Device> vals, std::uintptr_t stream) {
            launch_find_top_k_per_row_sorted(data.data(), indptr.data(), n_rows,
                                             trim, vals.data(),
                                             (cudaStream_t)stream);
        },
        "data"_a, "indptr"_a, nb::kw_only(), "n_rows"_a, "trim"_a, "vals"_a,
        "stream"_a = 0);

    m.def("sort_tile_size", []() { return SORT_TILE_SIZE; });

    m.def(
        "cut_smaller",
        [](gpu_array_c<int, Device> indptr, gpu_array_c<int, Device> index,
           gpu_array_c<float, Device> data, gpu_array_c<float, Device> vals,
           int n_rows, std::uintptr_t stream) {
            launch_cut_smaller(indptr.data(), index.data(), data.data(),
                               vals.data(), n_rows, (cudaStream_t)stream);
        },
        "indptr"_a, "index"_a, "data"_a, nb::kw_only(), "vals"_a, "n_rows"_a,
        "stream"_a = 0);
}

NB_MODULE(_bbknn_cuda, m) {
    REGISTER_GPU_BINDINGS(register_bindings, m);
}
