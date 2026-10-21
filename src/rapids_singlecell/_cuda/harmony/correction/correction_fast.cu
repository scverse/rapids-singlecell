#include <cuda_runtime.h>
#include "../../nb_types.h"

#include "../outer/kernels_outer.cuh"
#include "../scatter/kernels_scatter.cuh"
#include "kernels_correction_fast.cuh"

using namespace nb::literals;

constexpr int BLOCK_DIM_1D = 256;

// One cluster at a time (bounded memory): S rows 1.. are R_b^T X_b per batch
// (float block sums), W the coefficients (ridge_coefficients_kernel), then
// Z -= R_k W[batch]. Both are (n_batches + 1, n_pcs); row 0 is unused.
template <typename T>
static void correction_fast_impl(const T* X, const T* R, const T* O,
                                 const int* cats, const int* cat_offsets,
                                 const int* cell_indices, const T* lambda_kb,
                                 int n_cells, int n_pcs, int n_clusters,
                                 int n_batches, T* Z, T* R_col, T* S, T* W,
                                 cudaStream_t stream) {
    cudaMemcpyAsync(Z, X, (size_t)n_cells * n_pcs * sizeof(T),
                    cudaMemcpyDeviceToDevice, stream);
    for (int k = 0; k < n_clusters; k++) {
        gather_column_kernel<T>
            <<<strided_grid(n_cells, BLOCK_DIM_1D), BLOCK_DIM_1D, 0, stream>>>(
                R, R_col, k, n_cells, n_clusters);
        CUDA_CHECK_LAST_ERROR(gather_column_kernel);
        scatter_add_kernel_with_bias_block<T>
            <<<n_batches*((n_pcs + 1) / 2), 1024, 0, stream>>>(
                X, cat_offsets, cell_indices, n_cells, n_pcs, n_batches, S,
                R_col);
        CUDA_CHECK_LAST_ERROR(scatter_add_kernel_with_bias_block);
        ridge_coefficients_kernel<T>
            <<<strided_grid(n_pcs, BLOCK_DIM_1D), BLOCK_DIM_1D, 0, stream>>>(
                O, lambda_kb, S + n_pcs, W + n_pcs, 0, n_batches, n_clusters,
                n_pcs, k, 1);
        CUDA_CHECK_LAST_ERROR(ridge_coefficients_kernel);
        long long n = (long long)n_cells * n_pcs;
        harmony_correction_kernel<T>
            <<<strided_grid(n, BLOCK_DIM_1D), BLOCK_DIM_1D, 0, stream>>>(
                Z, W, cats, R_col, n_cells, n_pcs);
        CUDA_CHECK_LAST_ERROR(harmony_correction_kernel);
    }
}

template <typename T, typename Device>
static void register_correction_fast(nb::module_& m) {
    m.def(
        "correction_fast",
        [](gpu_array_c<const T, Device> X, gpu_array_c<const T, Device> R,
           gpu_array_c<const T, Device> O, gpu_array_c<const int, Device> cats,
           gpu_array_c<const int, Device> cat_offsets,
           gpu_array_c<const int, Device> cell_indices,
           gpu_array_c<const T, Device> lambda_kb, int n_cells, int n_pcs,
           int n_clusters, int n_batches, gpu_array_c<T, Device> Z,
           gpu_array_c<T, Device> R_col, gpu_array_c<T, Device> S,
           gpu_array_c<T, Device> W, std::uintptr_t stream) {
            correction_fast_impl<T>(
                X.data(), R.data(), O.data(), cats.data(), cat_offsets.data(),
                cell_indices.data(), lambda_kb.data(), n_cells, n_pcs,
                n_clusters, n_batches, Z.data(), R_col.data(), S.data(),
                W.data(), (cudaStream_t)stream);
        },
        "X"_a.noconvert(), nb::kw_only(), "R"_a.noconvert(), "O"_a.noconvert(),
        "cats"_a.noconvert(), "cat_offsets"_a.noconvert(),
        "cell_indices"_a.noconvert(), "lambda_kb"_a.noconvert(), "n_cells"_a,
        "n_pcs"_a, "n_clusters"_a, "n_batches"_a, "Z"_a.noconvert(),
        "R_col"_a.noconvert(), "S"_a.noconvert(), "W"_a.noconvert(),
        "stream"_a = 0);
}

template <typename Device>
void register_bindings(nb::module_& m) {
    register_correction_fast<float, Device>(m);
    register_correction_fast<double, Device>(m);
}

NB_MODULE(_harmony_correction_cuda, m) {
    REGISTER_GPU_BINDINGS(register_bindings, m);
}
