#include <cuda_runtime.h>
#include "../../nb_types.h"

#include "kernels_scatter.cuh"

using namespace nb::literals;

constexpr unsigned WARP_SIZE = 32;
constexpr int BLOCK_DIM_1D = 256;
constexpr int SCATTER_BLOCK_DIM = 1024;
constexpr int PCS_PER_THREAD = 2;  // Each thread handles 2 PCs
constexpr int GRID_Y = 8;          // Y-dimension of grid for scatter_add

template <typename T>
static inline void launch_scatter_add(const T* v, const int* cats,
                                      size_t n_cells, size_t n_pcs,
                                      size_t switcher, T* a,
                                      cudaStream_t stream) {
    dim3 block(BLOCK_DIM_1D);
    size_t N = n_cells * n_pcs;
    dim3 grid(strided_grid((long long)N, BLOCK_DIM_1D));
    scatter_add_kernel<T>
        <<<grid, block, 0, stream>>>(v, cats, n_cells, n_pcs, switcher, a);
    CUDA_CHECK_LAST_ERROR(scatter_add_kernel);
}

template <typename T>
static inline void launch_aggregated_matrix(T* aggregated_matrix, const T* sum,
                                            T top_corner, int n_batches,
                                            cudaStream_t stream) {
    dim3 block(WARP_SIZE);
    dim3 grid((n_batches + 1 + WARP_SIZE - 1) / WARP_SIZE);
    aggregated_matrix_kernel<T><<<grid, block, 0, stream>>>(
        aggregated_matrix, sum, top_corner, n_batches);
    CUDA_CHECK_LAST_ERROR(aggregated_matrix_kernel);
}

template <typename T>
static inline void launch_scatter_add_shared(const T* v, const int* cats,
                                             int n_cells, int n_pcs,
                                             int n_batches, int switcher, T* a,
                                             int n_blocks,
                                             cudaStream_t stream) {
    dim3 block(BLOCK_DIM_1D);
    dim3 grid(n_blocks);
    size_t shared_mem = (size_t)n_batches * n_pcs * sizeof(T);
    scatter_add_shared_kernel<T><<<grid, block, shared_mem, stream>>>(
        v, cats, n_cells, n_pcs, n_batches, switcher, a);
    CUDA_CHECK_LAST_ERROR(scatter_add_shared_kernel);
}

template <typename T>
static inline void launch_scatter_add_cat0(const T* v, int n_cells, int n_pcs,
                                           T* a, const T* bias,
                                           cudaStream_t stream) {
    dim3 block(SCATTER_BLOCK_DIM);
    dim3 grid((n_pcs + PCS_PER_THREAD - 1) / PCS_PER_THREAD, GRID_Y);
    scatter_add_kernel_with_bias_cat0<T>
        <<<grid, block, 0, stream>>>(v, n_cells, n_pcs, a, bias);
    CUDA_CHECK_LAST_ERROR(scatter_add_kernel_with_bias_cat0);
}

template <typename T>
static inline void launch_scatter_add_block(const T* v, const int* cat_offsets,
                                            const int* cell_indices,
                                            int n_cells, int n_pcs,
                                            int n_batches, T* a, const T* bias,
                                            cudaStream_t stream) {
    dim3 block(SCATTER_BLOCK_DIM);
    dim3 grid(n_batches * ((n_pcs + PCS_PER_THREAD - 1) / PCS_PER_THREAD));
    scatter_add_kernel_with_bias_block<T><<<grid, block, 0, stream>>>(
        v, cat_offsets, cell_indices, n_cells, n_pcs, n_batches, a, bias);
    CUDA_CHECK_LAST_ERROR(scatter_add_kernel_with_bias_block);
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
           size_t n_cells, size_t n_pcs, size_t switcher,
           gpu_array_c<T, Device> a, std::uintptr_t stream) {
            launch_scatter_add<T>(v.data(), cats.data(), n_cells, n_pcs,
                                  switcher, a.data(), (cudaStream_t)stream);
        },
        "v"_a, nb::kw_only(), "cats"_a, "n_cells"_a, "n_pcs"_a, "switcher"_a,
        "a"_a, "stream"_a = 0);
}

template <typename T, typename Device>
void def_aggregated_matrix(nb::module_& m) {
    m.def(
        "aggregated_matrix",
        [](gpu_array_c<T, Device> aggregated_matrix,
           gpu_array_c<const T, Device> sum, T top_corner, int n_batches,
           std::uintptr_t stream) {
            launch_aggregated_matrix<T>(aggregated_matrix.data(), sum.data(),
                                        top_corner, n_batches,
                                        (cudaStream_t)stream);
        },
        "aggregated_matrix"_a, nb::kw_only(), "sum"_a, "top_corner"_a,
        "n_batches"_a, "stream"_a = 0);
}

template <typename T, typename Device>
void def_scatter_add_shared(nb::module_& m) {
    m.def(
        "scatter_add_shared",
        [](gpu_array_c<const T, Device> v, gpu_array_c<const int, Device> cats,
           int n_cells, int n_pcs, int n_batches, int switcher,
           gpu_array_c<T, Device> a, int n_blocks, std::uintptr_t stream) {
            launch_scatter_add_shared<T>(v.data(), cats.data(), n_cells, n_pcs,
                                         n_batches, switcher, a.data(),
                                         n_blocks, (cudaStream_t)stream);
        },
        "v"_a, nb::kw_only(), "cats"_a, "n_cells"_a, "n_pcs"_a, "n_batches"_a,
        "switcher"_a, "a"_a, "n_blocks"_a, "stream"_a = 0);
}

template <typename T, typename Device>
void def_scatter_add_cat0(nb::module_& m) {
    m.def(
        "scatter_add_cat0",
        [](gpu_array_c<const T, Device> v, int n_cells, int n_pcs,
           gpu_array_c<T, Device> a, gpu_array_c<const T, Device> bias,
           std::uintptr_t stream) {
            launch_scatter_add_cat0<T>(v.data(), n_cells, n_pcs, a.data(),
                                       bias.data(), (cudaStream_t)stream);
        },
        "v"_a, nb::kw_only(), "n_cells"_a, "n_pcs"_a, "a"_a, "bias"_a,
        "stream"_a = 0);
}

template <typename T, typename Device>
void def_scatter_add_block(nb::module_& m) {
    m.def(
        "scatter_add_block",
        [](gpu_array_c<const T, Device> v,
           gpu_array_c<const int, Device> cat_offsets,
           gpu_array_c<const int, Device> cell_indices, int n_cells, int n_pcs,
           int n_batches, gpu_array_c<T, Device> a,
           gpu_array_c<const T, Device> bias, std::uintptr_t stream) {
            launch_scatter_add_block<T>(
                v.data(), cat_offsets.data(), cell_indices.data(), n_cells,
                n_pcs, n_batches, a.data(), bias.data(), (cudaStream_t)stream);
        },
        "v"_a, nb::kw_only(), "cat_offsets"_a, "cell_indices"_a, "n_cells"_a,
        "n_pcs"_a, "n_batches"_a, "a"_a, "bias"_a, "stream"_a = 0);
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
    def_aggregated_matrix<float, Device>(m);
    def_aggregated_matrix<double, Device>(m);
    def_scatter_add_shared<float, Device>(m);
    def_scatter_add_shared<double, Device>(m);
    def_scatter_add_cat0<float, Device>(m);
    def_scatter_add_cat0<double, Device>(m);
    def_scatter_add_block<float, Device>(m);
    def_scatter_add_block<double, Device>(m);
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
