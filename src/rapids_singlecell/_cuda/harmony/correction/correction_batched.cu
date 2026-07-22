#include <cuda_runtime.h>
#include <algorithm>
#include <stdexcept>
#include <vector>

#include "../../nb_types.h"

#include "../outer/kernels_outer.cuh"
#include "../scatter/kernels_scatter.cuh"
#include "../../cublas_helpers.cuh"
#include "kernels_correction_fast.cuh"
#include "kernels_correction_multi.cuh"

using namespace nb::literals;

constexpr int WARP_SIZE = 32;
constexpr int MAX_BLOCK_DIM = 256;
constexpr int BLOCK_DIM_1D = 256;
constexpr int MULTI_RHS_BLOCK_DIM = 256;

template <typename T>
static void prepare_multi_impl(
    const T* X, const T* R, const T* O, const int* joint_codes,
    const int* joint_cats, const int* joint_offsets,
    const int* joint_cell_indices, const int* marginal_joint_offsets,
    const int* marginal_joint_indices, const T* lambda_kb,
    const uint8_t* active, int n_cells, int n_pcs, int n_clusters,
    int n_batches, int n_covariates, int n_joint_categories,
    // workspace/output
    T* gram, T* rhs, T* joint_O, T* joint_rhs, cudaStream_t stream,
    cublasHandle_t handle) {
    if (n_covariates < 2)
        throw std::invalid_argument(
            "prepare_multi requires at least two covariates");

    int nb1 = n_batches + 1;
    size_t joint_elements = (size_t)n_joint_categories * n_clusters;
    size_t gram_elements = (size_t)n_clusters * nb1 * nb1;
    size_t rhs_elements = (size_t)n_clusters * nb1 * n_pcs;

    cudaMemsetAsync(joint_O, 0, joint_elements * sizeof(T), stream);
    cudaMemsetAsync(gram, 0, gram_elements * sizeof(T), stream);
    cudaMemsetAsync(rhs, 0, rhs_elements * sizeof(T), stream);

    if (n_cells > 0 && n_clusters > 0) {
        size_t work = (size_t)n_cells * n_clusters;
        joint_observed_kernel<T>
            <<<strided_grid((long long)work, BLOCK_DIM_1D), BLOCK_DIM_1D, 0,
               stream>>>(R, joint_codes, joint_O, n_cells, n_clusters);
        CUDA_CHECK_LAST_ERROR(joint_observed_kernel);
    }

    if (n_clusters > 0) {
        initialize_multi_gram_kernel<T>
            <<<n_clusters, MAX_BLOCK_DIM, 0, stream>>>(
                O, lambda_kb, active, joint_O, gram, n_batches, n_clusters,
                n_joint_categories);
        CUDA_CHECK_LAST_ERROR(initialize_multi_gram_kernel);
    }

    if (joint_elements > 0) {
        dim3 grid(strided_grid((long long)joint_elements, BLOCK_DIM_1D));
        dim3 block(BLOCK_DIM_1D);
        if (n_covariates == 2) {
            add_joint_cross_kernel<T, 2><<<grid, block, 0, stream>>>(
                joint_O, joint_cats, active, gram, n_joint_categories,
                n_covariates, n_batches, n_clusters);
        } else if (n_covariates == 3) {
            add_joint_cross_kernel<T, 3><<<grid, block, 0, stream>>>(
                joint_O, joint_cats, active, gram, n_joint_categories,
                n_covariates, n_batches, n_clusters);
        } else {
            add_joint_cross_kernel<T, 0><<<grid, block, 0, stream>>>(
                joint_O, joint_cats, active, gram, n_joint_categories,
                n_covariates, n_batches, n_clusters);
        }
        CUDA_CHECK_LAST_ERROR(add_joint_cross_kernel);
    }

    cublas_check_status(cublasSetStream(handle, stream), "cublasSetStream");
    T one = T(1), zero = T(0);

    // Intercept rows for all clusters: rhs[:, 0, :] = R.T @ X.
    if (n_cells > 0 && n_pcs > 0 && n_clusters > 0) {
        cublas_check_status(
            cublas_gemm<T>(handle, CUBLAS_OP_N, CUBLAS_OP_T, n_pcs, n_clusters,
                           n_cells, &one, X, n_pcs, R, n_clusters, &zero, rhs,
                           nb1 * n_pcs),
            "cublas_gemm(prepare_multi intercept rhs)");
    }

    // Compute each observed joint row once, then deterministically gather the
    // much smaller joint result into all marginal category rows.
    int pc_pairs = (n_pcs + 1) / 2;
    size_t joint_rhs_blocks =
        (size_t)n_joint_categories * n_clusters * pc_pairs;
    if (joint_rhs_blocks > 0) {
        if (joint_rhs_blocks > (size_t)max_grid_dim_x())
            throw std::invalid_argument(
                "prepare_multi joint RHS grid exceeds the CUDA grid-x limit");
        segmented_joint_rhs_kernel<T><<<(unsigned int)joint_rhs_blocks,
                                        MULTI_RHS_BLOCK_DIM, 0, stream>>>(
            X, R, joint_offsets, joint_cell_indices, joint_cats, active,
            joint_rhs, n_pcs, n_clusters, n_joint_categories, n_covariates);
        CUDA_CHECK_LAST_ERROR(segmented_joint_rhs_kernel);
    }

    size_t marginal_rhs_elements = (size_t)n_batches * n_clusters * n_pcs;
    if (marginal_rhs_elements > 0) {
        marginal_from_joint_rhs_kernel<T>
            <<<strided_grid((long long)marginal_rhs_elements, BLOCK_DIM_1D),
               BLOCK_DIM_1D, 0, stream>>>(joint_rhs, marginal_joint_offsets,
                                          marginal_joint_indices, active, rhs,
                                          n_pcs, n_clusters, n_batches);
        CUDA_CHECK_LAST_ERROR(marginal_from_joint_rhs_kernel);
    }
}

template <typename T>
static void apply_multi_impl(const T* X, const T* R, const T* W_all,
                             const int* cats, int n_cells, int n_pcs,
                             int n_clusters, int n_batches, int n_covariates,
                             bool initialize_output, T* Z,
                             cudaStream_t stream) {
    if (n_covariates < 2)
        throw std::invalid_argument(
            "apply_multi requires at least two covariates");

    size_t work = (size_t)n_cells * n_pcs;
    if (work == 0) return;

    dim3 grid(strided_grid((long long)work, BLOCK_DIM_1D));
    dim3 block(BLOCK_DIM_1D);
    if (n_covariates == 2) {
        apply_multi_correction_kernel<T, 2><<<grid, block, 0, stream>>>(
            X, R, W_all, cats, Z, n_cells, n_pcs, n_clusters, n_batches,
            n_covariates, initialize_output);
    } else if (n_covariates == 3) {
        apply_multi_correction_kernel<T, 3><<<grid, block, 0, stream>>>(
            X, R, W_all, cats, Z, n_cells, n_pcs, n_clusters, n_batches,
            n_covariates, initialize_output);
    } else {
        apply_multi_correction_kernel<T, 0><<<grid, block, 0, stream>>>(
            X, R, W_all, cats, Z, n_cells, n_pcs, n_clusters, n_batches,
            n_covariates, initialize_output);
    }
    CUDA_CHECK_LAST_ERROR(apply_multi_correction_kernel);
}

template <typename T>
static void correction_batched_impl(
    const T* X, const T* R, const T* O, const int* cats, const int* cat_offsets,
    const int* cell_indices, const T* lambda_kb, int n_cells, int n_pcs,
    int n_clusters, int n_batches,
    // workspace
    T* Z, T* inv_mats, T* Phi_t_diag_R_X_all, T* W_all, T* g_factor,
    T* g_P_row0, T* X_batch, T* R_batch, int batch_chunk_size,
    cudaStream_t stream, cublasHandle_t handle) {
    if (batch_chunk_size < 1)
        throw std::invalid_argument(
            "correction_batched requires positive batch scratch capacity");

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

    // Copy cat_offsets to host for batch loop
    std::vector<int> h_offsets(n_batches + 1);
    cudaMemcpyAsync(h_offsets.data(), cat_offsets,
                    (size_t)(n_batches + 1) * sizeof(int),
                    cudaMemcpyDeviceToHost, stream);
    cudaStreamSynchronize(stream);

    // Rows 1..n_batches: gather one bounded category chunk at a time, then use
    // the established dense GEMM. This retains the fast arrowhead path without
    // materializing full N-by-D and N-by-K sorted copies.
    for (int b = 0; b < n_batches; b++) {
        int start = h_offsets[b];
        int end = h_offsets[b + 1];
        int n_batch_cells = end - start;
        if (n_batch_cells == 0) continue;

        T* C_ptr = Phi_t_diag_R_X_all + (b + 1) * n_pcs;
        if (n_batch_cells == n_cells) {
            cublas_check_status(
                cublas_gemm<T>(handle, CUBLAS_OP_N, CUBLAS_OP_T, n_pcs,
                               n_clusters, n_cells, &one, X, n_pcs, R,
                               n_clusters, &zero, C_ptr, nb1 * n_pcs),
                "cublas_gemm(correction_batched full category)");
            continue;
        }

        bool first_chunk = true;
        for (int chunk_start = start; chunk_start < end;
             chunk_start += batch_chunk_size) {
            int chunk_cells = std::min(batch_chunk_size, end - chunk_start);
            const int* chunk_indices = cell_indices + chunk_start;

            size_t n_x = (size_t)chunk_cells * n_pcs;
            gather_rows_kernel<T>
                <<<strided_grid((long long)n_x, BLOCK_DIM_1D), BLOCK_DIM_1D, 0,
                   stream>>>(X, chunk_indices, X_batch, chunk_cells, n_pcs);
            CUDA_CHECK_LAST_ERROR(gather_rows_kernel);

            size_t n_r = (size_t)chunk_cells * n_clusters;
            gather_rows_kernel<T><<<strided_grid((long long)n_r, BLOCK_DIM_1D),
                                    BLOCK_DIM_1D, 0, stream>>>(
                R, chunk_indices, R_batch, chunk_cells, n_clusters);
            CUDA_CHECK_LAST_ERROR(gather_rows_kernel);

            // result[:,b+1,:] += R_batch.T @ X_batch. The first chunk
            // overwrites the zeroed row; later chunks accumulate in order.
            const T* beta = first_chunk ? &zero : &one;
            cublas_check_status(
                cublas_gemm<T>(handle, CUBLAS_OP_N, CUBLAS_OP_T, n_pcs,
                               n_clusters, chunk_cells, &one, X_batch, n_pcs,
                               R_batch, n_clusters, beta, C_ptr, nb1 * n_pcs),
                "cublas_gemm(correction_batched per-batch chunk)");
            first_chunk = false;
        }
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
           gpu_array_c<T, Device> g_P_row0, gpu_array_c<T, Device> X_batch,
           gpu_array_c<T, Device> R_batch, int batch_chunk_size,
           std::uintptr_t stream, std::uintptr_t handle) {
            correction_batched_impl<T>(
                X.data(), R.data(), O.data(), cats.data(), cat_offsets.data(),
                cell_indices.data(), lambda_kb.data(), n_cells, n_pcs,
                n_clusters, n_batches, Z.data(), inv_mats.data(),
                Phi_t_diag_R_X_all.data(), W_all.data(), g_factor.data(),
                g_P_row0.data(), X_batch.data(), R_batch.data(),
                batch_chunk_size, (cudaStream_t)stream, (cublasHandle_t)handle);
        },
        "X"_a, nb::kw_only(), "R"_a, "O"_a, "cats"_a, "cat_offsets"_a,
        "cell_indices"_a, "lambda_kb"_a, "n_cells"_a, "n_pcs"_a, "n_clusters"_a,
        "n_batches"_a, "Z"_a, "inv_mats"_a, "Phi_t_diag_R_X_all"_a, "W_all"_a,
        "g_factor"_a, "g_P_row0"_a, "X_batch"_a, "R_batch"_a,
        "batch_chunk_size"_a, "stream"_a = 0, "handle"_a);

    m.def(
        "prepare_multi",
        [](gpu_array_c<const T, Device> X, gpu_array_c<const T, Device> R,
           gpu_array_c<const T, Device> O,
           gpu_array_c<const int, Device> joint_codes,
           gpu_array_c<const int, Device> joint_cats,
           gpu_array_c<const int, Device> joint_offsets,
           gpu_array_c<const int, Device> joint_cell_indices,
           gpu_array_c<const int, Device> marginal_joint_offsets,
           gpu_array_c<const int, Device> marginal_joint_indices,
           gpu_array_c<const T, Device> lambda_kb,
           gpu_array_c<const uint8_t, Device> active, int n_cells, int n_pcs,
           int n_clusters, int n_batches, int n_covariates,
           int n_joint_categories, gpu_array_c<T, Device> gram,
           gpu_array_c<T, Device> rhs, gpu_array_c<T, Device> joint_O,
           gpu_array_c<T, Device> joint_rhs, std::uintptr_t stream,
           std::uintptr_t handle) {
            prepare_multi_impl<T>(
                X.data(), R.data(), O.data(), joint_codes.data(),
                joint_cats.data(), joint_offsets.data(),
                joint_cell_indices.data(), marginal_joint_offsets.data(),
                marginal_joint_indices.data(), lambda_kb.data(), active.data(),
                n_cells, n_pcs, n_clusters, n_batches, n_covariates,
                n_joint_categories, gram.data(), rhs.data(), joint_O.data(),
                joint_rhs.data(), (cudaStream_t)stream, (cublasHandle_t)handle);
        },
        "X"_a, nb::kw_only(), "R"_a, "O"_a, "joint_codes"_a, "joint_cats"_a,
        "joint_offsets"_a, "joint_cell_indices"_a, "marginal_joint_offsets"_a,
        "marginal_joint_indices"_a, "lambda_kb"_a, "active_mask"_a, "n_cells"_a,
        "n_pcs"_a, "n_clusters"_a, "n_batches"_a, "n_covariates"_a,
        "n_joint_categories"_a, "gram"_a, "rhs"_a, "joint_O"_a, "joint_rhs"_a,
        "stream"_a = 0, "handle"_a);

    m.def(
        "apply_multi",
        [](gpu_array_c<const T, Device> X, gpu_array_c<const T, Device> R,
           gpu_array_c<const T, Device> W_all,
           gpu_array_c<const int, Device> cats, int n_cells, int n_pcs,
           int n_clusters, int n_batches, int n_covariates,
           bool initialize_output, gpu_array_c<T, Device> Z,
           std::uintptr_t stream) {
            apply_multi_impl<T>(X.data(), R.data(), W_all.data(), cats.data(),
                                n_cells, n_pcs, n_clusters, n_batches,
                                n_covariates, initialize_output, Z.data(),
                                (cudaStream_t)stream);
        },
        "X"_a, nb::kw_only(), "R"_a, "W_all"_a, "cats"_a, "n_cells"_a,
        "n_pcs"_a, "n_clusters"_a, "n_batches"_a, "n_covariates"_a,
        "initialize_output"_a, "Z"_a, "stream"_a = 0);
}

template <typename Device>
void register_bindings(nb::module_& m) {
    register_correction_batched<float, Device>(m);
    register_correction_batched<double, Device>(m);
}

NB_MODULE(_harmony_correction_batched_cuda, m) {
    REGISTER_GPU_BINDINGS(register_bindings, m);
}
