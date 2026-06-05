#include <cuda_runtime.h>
#include <vector>

#include "../../nb_types.h"

#include "../outer/kernels_outer.cuh"
#include "../scatter/kernels_scatter.cuh"
#include "../../cublas_helpers.cuh"
#include "kernels_correction_fast.cuh"

using namespace nb::literals;

constexpr int WARP_SIZE = 32;
constexpr int MAX_BLOCK_DIM = 256;
constexpr int BLOCK_DIM_1D = 256;

template <typename T>
static void correction_batched_impl(
    const T* X, const T* R, const T* O, const int* cats, const int* cat_offsets,
    const int* cell_indices, const T* lambda_kb, int n_cells, int n_pcs,
    int n_clusters, int n_batches,
    // workspace
    T* Z, T* inv_mats, T* Phi_t_diag_R_X_all, T* W_all, T* g_factor,
    T* g_P_row0, T* X_sorted, T* R_sorted, cudaStream_t stream,
    cublasHandle_t handle) {
    int nb1 = n_batches + 1;

    // Step 1: Z = X.copy()
    cudaMemcpyAsync(Z, X, (size_t)n_cells * n_pcs * sizeof(T),
                    cudaMemcpyDeviceToDevice, stream);

    // Step 2: Compute all inv_mats at once (n_clusters blocks, cluster_k=-1)
    int bdim = std::min(MAX_BLOCK_DIM,
                        std::max(WARP_SIZE, (n_batches + WARP_SIZE - 1) /
                                                WARP_SIZE * WARP_SIZE));
    compute_inv_mats_kernel<T><<<n_clusters, bdim, 0, stream>>>(
        O, lambda_kb, inv_mats, g_factor, g_P_row0, n_batches, n_clusters,
        /*cluster_k=*/-1);
    CUDA_CHECK_LAST_ERROR(compute_inv_mats_kernel);

    cublas_check_status(cublasSetStream(handle, stream), "cublasSetStream");

    T one = T(1), zero = T(0);

    // Step 3: Compute Phi_t_diag_R_X_all (n_clusters, nb1, n_pcs)
    // Zero the result (needed for empty batches)
    cudaMemsetAsync(Phi_t_diag_R_X_all, 0,
                    (size_t)n_clusters * nb1 * n_pcs * sizeof(T), stream);

    // Row 0: result[:,0,:] = R.T @ X
    // R is (n_cells, n_clusters) row-major → cuBLAS sees (n_clusters, n_cells)
    // X is (n_cells, n_pcs) row-major → cuBLAS sees (n_pcs, n_cells)
    // Want C_cublas(n_pcs, n_clusters) = X_cublas @ R_cublas^T
    // op=N,T  m=n_pcs  n=n_clusters  k=n_cells  ldc=nb1*n_pcs (strided write)
    cublas_check_status(
        cublas_gemm<T>(handle, CUBLAS_OP_N, CUBLAS_OP_T, n_pcs, n_clusters,
                       n_cells, &one, X, n_pcs, R, n_clusters, &zero,
                       Phi_t_diag_R_X_all, nb1 * n_pcs),
        "cublas_gemm(correction_batched row0)");

    // Gather X and R into batch-sorted order
    {
        size_t n_x = (size_t)n_cells * n_pcs;
        gather_rows_kernel<T>
            <<<strided_grid((long long)n_x, BLOCK_DIM_1D), BLOCK_DIM_1D, 0,
               stream>>>(X, cell_indices, X_sorted, n_cells, n_pcs);
        CUDA_CHECK_LAST_ERROR(gather_rows_kernel);

        size_t n_r = (size_t)n_cells * n_clusters;
        gather_rows_kernel<T>
            <<<strided_grid((long long)n_r, BLOCK_DIM_1D), BLOCK_DIM_1D, 0,
               stream>>>(R, cell_indices, R_sorted, n_cells, n_clusters);
        CUDA_CHECK_LAST_ERROR(gather_rows_kernel);
    }

    // Copy cat_offsets to host for batch loop
    std::vector<int> h_offsets(n_batches + 1);
    cudaMemcpyAsync(h_offsets.data(), cat_offsets,
                    (size_t)(n_batches + 1) * sizeof(int),
                    cudaMemcpyDeviceToHost, stream);
    cudaStreamSynchronize(stream);

    // Rows 1..n_batches: per-batch GEMMs on sorted data
    for (int b = 0; b < n_batches; b++) {
        int start = h_offsets[b];
        int end = h_offsets[b + 1];
        int n_batch_cells = end - start;
        if (n_batch_cells == 0) continue;

        const T* X_batch = X_sorted + (size_t)start * n_pcs;
        const T* R_batch = R_sorted + (size_t)start * n_clusters;
        T* C_ptr = Phi_t_diag_R_X_all + (b + 1) * n_pcs;

        // result[:,b+1,:] = R_batch.T @ X_batch (same N,T trick as row 0)
        cublas_check_status(
            cublas_gemm<T>(handle, CUBLAS_OP_N, CUBLAS_OP_T, n_pcs, n_clusters,
                           n_batch_cells, &one, X_batch, n_pcs, R_batch,
                           n_clusters, &zero, C_ptr, nb1 * n_pcs),
            "cublas_gemm(correction_batched per-batch)");
    }

    // Step 4: W_all = inv_mats @ Phi_t_diag_R_X_all (strided batched GEMM)
    // Row-major: C(nb1,n_pcs) = A(nb1,nb1) @ B(nb1,n_pcs) per cluster
    // cuBLAS col-major: C_cm(n_pcs,nb1) = B_cm(n_pcs,nb1) @ A_cm(nb1,nb1)
    {
        long long sA = (long long)nb1 * n_pcs;  // stride for Phi_t_diag_R_X_all
        long long sB = (long long)nb1 * nb1;    // stride for inv_mats
        long long sC = (long long)nb1 * n_pcs;  // stride for W_all

        cublas_check_status(
            cublas_gemm_strided_batched<T>(
                handle, CUBLAS_OP_N, CUBLAS_OP_N, n_pcs, nb1, nb1, &one,
                Phi_t_diag_R_X_all, n_pcs, sA, inv_mats, nb1, sB, &zero, W_all,
                n_pcs, sC, n_clusters),
            "cublas_gemm_strided_batched(correction_batched W_all)");
    }

    // Step 5: W_all[:, 0, :] = 0
    cudaMemset2DAsync(W_all, (size_t)nb1 * n_pcs * sizeof(T), 0,
                      n_pcs * sizeof(T), n_clusters, stream);

    // Step 6: Apply correction
    {
        size_t n_total = (size_t)n_cells * n_pcs;
        batched_correction_kernel<T>
            <<<strided_grid((long long)n_total, BLOCK_DIM_1D), BLOCK_DIM_1D, 0,
               stream>>>(Z, W_all, cats, R, n_cells, n_pcs, n_clusters, nb1);
        CUDA_CHECK_LAST_ERROR(batched_correction_kernel);
    }
}

// ---- nanobind registration ----

template <typename T, typename Device>
static void register_correction_batched(nb::module_& m) {
    m.def(
        "correction_batched",
        [](gpu_array_c<const T, Device> X, gpu_array_c<const T, Device> R,
           gpu_array_c<const T, Device> O, gpu_array_c<const int, Device> cats,
           gpu_array_c<const int, Device> cat_offsets,
           gpu_array_c<const int, Device> cell_indices,
           gpu_array_c<const T, Device> lambda_kb, int n_cells, int n_pcs,
           int n_clusters, int n_batches,
           // workspace
           gpu_array_c<T, Device> Z, gpu_array_c<T, Device> inv_mats,
           gpu_array_c<T, Device> Phi_t_diag_R_X_all,
           gpu_array_c<T, Device> W_all, gpu_array_c<T, Device> g_factor,
           gpu_array_c<T, Device> g_P_row0, gpu_array_c<T, Device> X_sorted,
           gpu_array_c<T, Device> R_sorted, std::uintptr_t stream,
           std::uintptr_t handle) {
            correction_batched_impl<T>(
                X.data(), R.data(), O.data(), cats.data(), cat_offsets.data(),
                cell_indices.data(), lambda_kb.data(), n_cells, n_pcs,
                n_clusters, n_batches, Z.data(), inv_mats.data(),
                Phi_t_diag_R_X_all.data(), W_all.data(), g_factor.data(),
                g_P_row0.data(), X_sorted.data(), R_sorted.data(),
                (cudaStream_t)stream, (cublasHandle_t)handle);
        },
        "X"_a, nb::kw_only(), "R"_a, "O"_a, "cats"_a, "cat_offsets"_a,
        "cell_indices"_a, "lambda_kb"_a, "n_cells"_a, "n_pcs"_a, "n_clusters"_a,
        "n_batches"_a, "Z"_a, "inv_mats"_a, "Phi_t_diag_R_X_all"_a, "W_all"_a,
        "g_factor"_a, "g_P_row0"_a, "X_sorted"_a, "R_sorted"_a, "stream"_a = 0,
        "handle"_a);
}

template <typename Device>
void register_bindings(nb::module_& m) {
    register_correction_batched<float, Device>(m);
    register_correction_batched<double, Device>(m);
}

NB_MODULE(_harmony_correction_batched_cuda, m) {
    REGISTER_GPU_BINDINGS(register_bindings, m);
}
