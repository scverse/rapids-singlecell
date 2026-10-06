#include <cuda_runtime.h>
#include "../../nb_types.h"

#include "../outer/kernels_outer.cuh"
#include "../scatter/kernels_scatter.cuh"
#include "../../cublas_helpers.cuh"
#include "kernels_correction_fast.cuh"

using namespace nb::literals;

constexpr int BLOCK_DIM_1D = 256;

// Inverse matrix of cluster k (one block of whole warps over the batches).
template <typename T>
static void compute_inv_mat(const T* O, const T* lambda_kb, T* inv_mat,
                            T* g_factor, T* g_P_row0, int n_batches,
                            int n_clusters, int k, cudaStream_t stream) {
    int bdim = std::min(1024, std::max(32, (n_batches + 31) / 32 * 32));
    compute_inv_mats_kernel<T><<<1, bdim, 0, stream>>>(
        O, lambda_kb, inv_mat, g_factor, g_P_row0, n_batches, n_clusters, k);
    CUDA_CHECK_LAST_ERROR(compute_inv_mats_kernel);
}

template <typename T>
static void correction_fast_impl(const T* X, const T* R, const T* O,
                                 const int* cats, const int* cat_offsets,
                                 const int* cell_indices, const T* lambda_kb,
                                 int n_cells, int n_pcs, int n_clusters,
                                 int n_batches, T* Z, T* inv_mat, T* R_col,
                                 T* Phi_t_diag_R_X, T* W, T* g_factor,
                                 T* g_P_row0, cudaStream_t stream,
                                 cublasHandle_t handle) {
    int nb1 = n_batches + 1;
    cudaMemcpyAsync(Z, X, (size_t)n_cells * n_pcs * sizeof(T),
                    cudaMemcpyDeviceToDevice, stream);
    cublas_check_status(cublasSetStream(handle, stream), "cublasSetStream");
    T one = T(1), zero = T(0);
    for (int k = 0; k < n_clusters; k++) {
        compute_inv_mat(O, lambda_kb, inv_mat, g_factor, g_P_row0, n_batches,
                        n_clusters, k, stream);
        gather_column_kernel<T>
            <<<strided_grid(n_cells, BLOCK_DIM_1D), BLOCK_DIM_1D, 0, stream>>>(
                R, R_col, k, n_cells, n_clusters);
        CUDA_CHECK_LAST_ERROR(gather_column_kernel);
        cudaMemsetAsync(Phi_t_diag_R_X, 0, (size_t)nb1 * n_pcs * sizeof(T),
                        stream);
        // Row 0: X^T R_col over all cells; rows 1..: per batch.
        cublas_check_status(
            cublas_gemv<T>(handle, CUBLAS_OP_N, n_pcs, n_cells, &one, X, n_pcs,
                           R_col, 1, &zero, Phi_t_diag_R_X, 1),
            "cublas_gemv(correction_fast row0)");
        scatter_add_kernel_with_bias_block<T>
            <<<n_batches*((n_pcs + 1) / 2), 1024, 0, stream>>>(
                X, cat_offsets, cell_indices, n_cells, n_pcs, n_batches,
                Phi_t_diag_R_X, R_col);
        CUDA_CHECK_LAST_ERROR(scatter_add_kernel_with_bias_block);
        // W = inv_mat @ Phi_t_diag_R_X (row-major, as column-major B^T A^T).
        cublas_check_status(
            cublas_gemm<T>(handle, CUBLAS_OP_N, CUBLAS_OP_N, n_pcs, nb1, nb1,
                           &one, Phi_t_diag_R_X, n_pcs, inv_mat, nb1, &zero, W,
                           n_pcs),
            "cublas_gemm(correction_fast W)");
        cudaMemsetAsync(W, 0, n_pcs * sizeof(T), stream);  // W[0, :] = 0
        // Z -= R_col[cell] * W[cats[cell] + 1, :]
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
           gpu_array_c<T, Device> inv_mat, gpu_array_c<T, Device> R_col,
           gpu_array_c<T, Device> Phi_t_diag_R_X, gpu_array_c<T, Device> W,
           gpu_array_c<T, Device> g_factor, gpu_array_c<T, Device> g_P_row0,
           std::uintptr_t stream, std::uintptr_t handle) {
            correction_fast_impl<T>(
                X.data(), R.data(), O.data(), cats.data(), cat_offsets.data(),
                cell_indices.data(), lambda_kb.data(), n_cells, n_pcs,
                n_clusters, n_batches, Z.data(), inv_mat.data(), R_col.data(),
                Phi_t_diag_R_X.data(), W.data(), g_factor.data(),
                g_P_row0.data(), (cudaStream_t)stream, (cublasHandle_t)handle);
        },
        "X"_a, nb::kw_only(), "R"_a, "O"_a, "cats"_a, "cat_offsets"_a,
        "cell_indices"_a, "lambda_kb"_a, "n_cells"_a, "n_pcs"_a, "n_clusters"_a,
        "n_batches"_a, "Z"_a, "inv_mat"_a, "R_col"_a, "Phi_t_diag_R_X"_a, "W"_a,
        "g_factor"_a, "g_P_row0"_a, "stream"_a = 0, "handle"_a);
    // Single-cluster inverse matrix, for tests.
    m.def(
        "compute_inv_mat",
        [](gpu_array_c<const T, Device> O,
           gpu_array_c<const T, Device> lambda_kb, int n_batches,
           int n_clusters, int cluster_k, gpu_array_c<T, Device> inv_mat,
           gpu_array_c<T, Device> g_factor, gpu_array_c<T, Device> g_P_row0,
           std::uintptr_t stream) {
            compute_inv_mat(O.data(), lambda_kb.data(), inv_mat.data(),
                            g_factor.data(), g_P_row0.data(), n_batches,
                            n_clusters, cluster_k, (cudaStream_t)stream);
        },
        "O"_a, nb::kw_only(), "lambda_kb"_a, "n_batches"_a, "n_clusters"_a,
        "cluster_k"_a, "inv_mat"_a, "g_factor"_a, "g_P_row0"_a, "stream"_a = 0);
}

template <typename Device>
void register_bindings(nb::module_& m) {
    register_correction_fast<float, Device>(m);
    register_correction_fast<double, Device>(m);
}

NB_MODULE(_harmony_correction_cuda, m) {
    REGISTER_GPU_BINDINGS(register_bindings, m);
}
