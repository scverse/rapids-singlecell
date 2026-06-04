#include <cuda_runtime.h>
#include <math_constants.h>

#include <cublas_v2.h>
#include <cusolverDn.h>

#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

#include "../cublas_helpers.cuh"
#include "../nb_types.h"

#include "kernels_gmm.cuh"

using namespace nb::literals;

constexpr int E_STEP_BLOCK = 64;
constexpr int E_STEP_LARGE64_TILE = 64;
constexpr int E_STEP_THREAD64_BLOCK = 512;
constexpr int NORMALIZE_BLOCK = 32;
constexpr int CHOL_FILL_THREADS = 256;
constexpr size_t DEFAULT_DYNAMIC_SMEM_LIMIT = 48 * 1024;

static inline size_t upper_tri_size(size_t d) {
    return (d * (d + 1)) / 2;
}

static inline void cuda_check_runtime(cudaError_t err, const char* what) {
    if (err != cudaSuccess) {
        throw std::runtime_error(std::string(what) +
                                 " failed: " + cudaGetErrorString(err));
    }
}

static inline void cusolver_check(cusolverStatus_t status, const char* what) {
    if (status != CUSOLVER_STATUS_SUCCESS) {
        throw std::runtime_error(std::string(what) +
                                 " failed with cuSOLVER status " +
                                 std::to_string(static_cast<int>(status)));
    }
}

// Typed wrappers for the batched Cholesky (cuSOLVER) and triangular solve
// (cuBLAS) used by the full-covariance precision-Cholesky. Ported from cuML's
// GMM (ML::GMM, Apache-2.0).
template <typename T>
static inline cusolverStatus_t potrf_batched(cusolverDnHandle_t h,
                                             cublasFillMode_t uplo, int nn,
                                             T* Aarray[], int lda,
                                             int* infoArray, int batch);
template <>
inline cusolverStatus_t potrf_batched<float>(cusolverDnHandle_t h,
                                             cublasFillMode_t uplo, int nn,
                                             float* Aarray[], int lda,
                                             int* infoArray, int batch) {
    return cusolverDnSpotrfBatched(h, uplo, nn, Aarray, lda, infoArray, batch);
}
template <>
inline cusolverStatus_t potrf_batched<double>(cusolverDnHandle_t h,
                                              cublasFillMode_t uplo, int nn,
                                              double* Aarray[], int lda,
                                              int* infoArray, int batch) {
    return cusolverDnDpotrfBatched(h, uplo, nn, Aarray, lda, infoArray, batch);
}

template <typename T>
static inline cublasStatus_t trsm_batched(
    cublasHandle_t h, cublasSideMode_t side, cublasFillMode_t uplo,
    cublasOperation_t trans, cublasDiagType_t diag, int m, int n,
    const T* alpha, const T* const Aarray[], int lda, T* const Barray[],
    int ldb, int batch);
template <>
inline cublasStatus_t trsm_batched<float>(
    cublasHandle_t h, cublasSideMode_t side, cublasFillMode_t uplo,
    cublasOperation_t trans, cublasDiagType_t diag, int m, int n,
    const float* alpha, const float* const Aarray[], int lda,
    float* const Barray[], int ldb, int batch) {
    return cublasStrsmBatched(h, side, uplo, trans, diag, m, n, alpha, Aarray,
                              lda, Barray, ldb, batch);
}
template <>
inline cublasStatus_t trsm_batched<double>(
    cublasHandle_t h, cublasSideMode_t side, cublasFillMode_t uplo,
    cublasOperation_t trans, cublasDiagType_t diag, int m, int n,
    const double* alpha, const double* const Aarray[], int lda,
    double* const Barray[], int ldb, int batch) {
    return cublasDtrsmBatched(h, side, uplo, trans, diag, m, n, alpha, Aarray,
                              lda, Barray, ldb, batch);
}

// Full-covariance precision-Cholesky: covariances (K, d, d) -> upper precision
// factor prec_chol (K, d, d) and log_det (K,), in one potrfBatched + one
// trsmBatched (a single host sync for the positive-definite check). The
// column-major L^{-1} from the solve is, read row-major, the upper precision
// factor U with U Uᵀ = Σ⁻¹ -- matching sklearn's _compute_precision_cholesky
// and rsc's previous CuPy ``solve_triangular(L, I).T``. Ported from cuML's
// ML::GMM::precision_cholesky_full_batched (Apache-2.0), using raw cuBLAS /
// cuSOLVER handles instead of raft::handle_t.
template <typename T>
static inline void launch_precision_cholesky_full(
    const T* covariances, T* prec_chol, T* log_det, int d, int K,
    cudaStream_t stream, cublasHandle_t cublas, cusolverDnHandle_t solver) {
    if (d == 0 || K == 0) return;
    const size_t cov_elems = (size_t)K * d * d;

    // Stream-ordered scratch (no implicit device-wide sync).
    T* cov_work = nullptr;
    T** dA = nullptr;
    T** dB = nullptr;
    int* dev_info = nullptr;
    cuda_check_runtime(
        cudaMallocAsync(&cov_work, cov_elems * sizeof(T), stream),
        "cudaMallocAsync(cov_work)");
    cuda_check_runtime(cudaMallocAsync(&dA, sizeof(T*) * K, stream),
                       "cudaMallocAsync(dA)");
    cuda_check_runtime(cudaMallocAsync(&dB, sizeof(T*) * K, stream),
                       "cudaMallocAsync(dB)");
    cuda_check_runtime(cudaMallocAsync(&dev_info, sizeof(int) * K, stream),
                       "cudaMallocAsync(dev_info)");

    // potrf factorizes in place, so leave the caller's covariances intact.
    cuda_check_runtime(
        cudaMemcpyAsync(cov_work, covariances, cov_elems * sizeof(T),
                        cudaMemcpyDeviceToDevice, stream),
        "cudaMemcpyAsync(cov_work)");

    std::vector<T*> hA(K), hB(K);
    for (int k = 0; k < K; ++k) {
        hA[k] = cov_work + (size_t)k * d * d;
        hB[k] = prec_chol + (size_t)k * d * d;
    }
    cuda_check_runtime(cudaMemcpyAsync(dA, hA.data(), sizeof(T*) * K,
                                       cudaMemcpyHostToDevice, stream),
                       "cudaMemcpyAsync(dA)");
    cuda_check_runtime(cudaMemcpyAsync(dB, hB.data(), sizeof(T*) * K,
                                       cudaMemcpyHostToDevice, stream),
                       "cudaMemcpyAsync(dB)");

    cublas_check_status(cublasSetStream(cublas, stream), "cublasSetStream");
    cusolver_check(cusolverDnSetStream(solver, stream), "cusolverDnSetStream");

    // Lower Cholesky of each covariance (in place on cov_work).
    cusolver_check(
        potrf_batched<T>(solver, CUBLAS_FILL_MODE_LOWER, d, dA, d, dev_info, K),
        "potrfBatched");

    set_identity_batched_kernel<T>
        <<<dim3((unsigned int)((cov_elems + CHOL_FILL_THREADS - 1) /
                               CHOL_FILL_THREADS)),
           dim3(CHOL_FILL_THREADS), 0, stream>>>(prec_chol, d, K);
    CUDA_CHECK_LAST_ERROR(set_identity_batched_kernel);

    // Solve L X = I -> X = L^{-1} (written into prec_chol).
    const T one = T(1);
    cublas_check_status(
        trsm_batched<T>(cublas, CUBLAS_SIDE_LEFT, CUBLAS_FILL_MODE_LOWER,
                        CUBLAS_OP_N, CUBLAS_DIAG_NON_UNIT, d, d, &one, dA, d,
                        dB, d, K),
        "trsmBatched");

    log_det_full_kernel<T>
        <<<dim3((unsigned int)K), dim3(FULL_LOGDET_THREADS), 0, stream>>>(
            prec_chol, d, K, log_det);
    CUDA_CHECK_LAST_ERROR(log_det_full_kernel);

    // One host sync to surface a non-positive-definite covariance.
    std::vector<int> h_info(K);
    cuda_check_runtime(cudaMemcpyAsync(h_info.data(), dev_info, sizeof(int) * K,
                                       cudaMemcpyDeviceToHost, stream),
                       "cudaMemcpyAsync(dev_info)");
    cuda_check_runtime(cudaStreamSynchronize(stream),
                       "cudaStreamSynchronize(precision_cholesky)");
    // Best-effort stream-ordered cleanup; free return codes are not checked so
    // a failure here never masks an already-computed result.
    cudaFreeAsync(cov_work, stream);
    cudaFreeAsync(dA, stream);
    cudaFreeAsync(dB, stream);
    cudaFreeAsync(dev_info, stream);
    for (int k = 0; k < K; ++k) {
        if (h_info[k] != 0) {
            throw std::runtime_error(
                "Precision Cholesky failed: covariance component " +
                std::to_string(k) +
                " is not positive definite. Increase reg_covar or scale the "
                "input data.");
        }
    }
}

template <typename T, int D>
static inline void launch_e_step_log_prob_fixed_d_impl(
    const T* X, const T* weights, const T* means, const T* prec_chol,
    const T* log_det_half, int n, int K, T* log_prob, dim3 grid, dim3 block,
    cudaStream_t stream) {
    size_t shmem = (D + upper_tri_size(D)) * sizeof(T);
    if (shmem > DEFAULT_DYNAMIC_SMEM_LIMIT) {
        cuda_check_runtime(
            cudaFuncSetAttribute(e_step_log_prob_small_kernel<T, D>,
                                 cudaFuncAttributeMaxDynamicSharedMemorySize,
                                 (int)shmem),
            "cudaFuncSetAttribute(e_step_log_prob_small_kernel)");
    }
    e_step_log_prob_small_kernel<T, D><<<grid, block, shmem, stream>>>(
        X, weights, means, prec_chol, log_det_half, n, D, K, log_prob);
    CUDA_CHECK_LAST_ERROR(e_step_log_prob_small_kernel);
}

template <typename T>
static inline void launch_e_step(const T* X, const T* weights, const T* means,
                                 const T* prec_chol, const T* log_det_half,
                                 int n, int d, int K, T* log_prob, T* resp,
                                 T* ll_per_cell, cudaStream_t stream) {
    if (n == 0 || d == 0 || K == 0) return;
    if (d <= 64) {
        dim3 block(E_STEP_BLOCK);
        dim3 grid((n + E_STEP_BLOCK - 1) / E_STEP_BLOCK, K);
        if (d == 16) {
            launch_e_step_log_prob_fixed_d_impl<T, 16>(
                X, weights, means, prec_chol, log_det_half, n, K, log_prob,
                grid, block, stream);
        } else if (d == 32) {
            launch_e_step_log_prob_fixed_d_impl<T, 32>(
                X, weights, means, prec_chol, log_det_half, n, K, log_prob,
                grid, block, stream);
        } else if (d == 50) {
            launch_e_step_log_prob_fixed_d_impl<T, 50>(
                X, weights, means, prec_chol, log_det_half, n, K, log_prob,
                grid, block, stream);
        } else if (d == 64) {
            launch_e_step_log_prob_fixed_d_impl<T, 64>(
                X, weights, means, prec_chol, log_det_half, n, K, log_prob,
                grid, block, stream);
        } else {
            size_t shmem = ((size_t)d + upper_tri_size(d)) * sizeof(T);
            e_step_log_prob_small_kernel<T><<<grid, block, shmem, stream>>>(
                X, weights, means, prec_chol, log_det_half, n, d, K, log_prob);
            CUDA_CHECK_LAST_ERROR(e_step_log_prob_small_kernel);
        }
    } else {
        dim3 block(E_STEP_THREAD64_BLOCK);
        dim3 grid((n + E_STEP_THREAD64_BLOCK - 1) / E_STEP_THREAD64_BLOCK, K);
        size_t shmem = ((size_t)E_STEP_LARGE64_TILE +
                        (size_t)E_STEP_LARGE64_TILE * E_STEP_LARGE64_TILE) *
                       sizeof(T);
        e_step_log_prob_large_d_thread64_kernel<T, E_STEP_LARGE64_TILE>
            <<<grid, block, shmem, stream>>>(X, weights, means, prec_chol,
                                             log_det_half, n, d, K, log_prob);
        CUDA_CHECK_LAST_ERROR(e_step_log_prob_large_d_thread64_kernel);
    }
    {
        dim3 block(NORMALIZE_BLOCK);
        dim3 grid(n);
        e_step_normalize_kernel<T>
            <<<grid, block, 0, stream>>>(log_prob, n, K, resp, ll_per_cell);
        CUDA_CHECK_LAST_ERROR(e_step_normalize_kernel);
    }
}

template <typename T>
static inline void launch_e_step_cublas(const T* X, const T* weights,
                                        const T* means, const T* prec_chol,
                                        const T* log_det_half, int n, int d,
                                        int K, T* centered_workspace,
                                        T* y_workspace, T* log_prob, T* resp,
                                        T* ll_per_cell, cudaStream_t stream,
                                        cublasHandle_t handle) {
    if (n == 0 || d == 0 || K == 0) return;

    cublas_check_status(cublasSetStream(handle, stream), "cublasSetStream");

    T one = T(1);
    T zero = T(0);
    int threads = 256;
    int center_blocks = (int)(((size_t)n * d + threads - 1) / threads);
    int row_blocks = (n + threads - 1) / threads;

    for (int k = 0; k < K; ++k) {
        e_step_center_kernel<T><<<center_blocks, threads, 0, stream>>>(
            X, means, n, d, k, centered_workspace);
        CUDA_CHECK_LAST_ERROR(e_step_center_kernel);

        const T* pc_k = prec_chol + (size_t)k * d * d;
        cublas_check_status(
            cublas_gemm<T>(handle, CUBLAS_OP_N, CUBLAS_OP_N, d, n, d, &one,
                           pc_k, d, centered_workspace, d, &zero, y_workspace,
                           d),
            "cublas_gemm(e_step)");

        e_step_log_prob_from_y_kernel<T><<<row_blocks, threads, 0, stream>>>(
            y_workspace, weights, log_det_half, n, d, K, k, log_prob);
        CUDA_CHECK_LAST_ERROR(e_step_log_prob_from_y_kernel);
    }

    {
        dim3 block(NORMALIZE_BLOCK);
        dim3 grid(n);
        e_step_normalize_kernel<T>
            <<<grid, block, 0, stream>>>(log_prob, n, K, resp, ll_per_cell);
        CUDA_CHECK_LAST_ERROR(e_step_normalize_kernel);
    }
}

template <typename T>
static inline void launch_m_step(const T* resp, const T* X, const T* ones,
                                 int n, int d, int K, T reg_covar, T* weights,
                                 T* means, T* covariances, T* workspace_N_k,
                                 T* workspace_num, T* workspace_centered,
                                 cudaStream_t stream, cublasHandle_t handle) {
    if (n == 0 || d == 0 || K == 0) return;

    cublas_check_status(cublasSetStream(handle, stream), "cublasSetStream");

    T one = T(1);
    T zero = T(0);
    T eps = std::numeric_limits<T>::epsilon();

    // Row-major resp(n,K) is cuBLAS column-major (K,n). N_k = resp.T @ 1.
    cublas_check_status(cublas_gemv<T>(handle, CUBLAS_OP_N, K, n, &one, resp, K,
                                       ones, 1, &zero, workspace_N_k, 1),
                        "cublas_gemv(N_k)");

    // Row-major X(n,d) is cuBLAS column-major (d,n). Fill row-major
    // workspace_num(K,d) through its column-major (d,K) view with X.T @ resp.
    cublas_check_status(
        cublas_gemm<T>(handle, CUBLAS_OP_N, CUBLAS_OP_T, d, K, n, &one, X, d,
                       resp, K, &zero, workspace_num, d),
        "cublas_gemm(num)");

    {
        int threads = 256;
        dim3 block(threads);
        dim3 grid(K);
        m_step_finalize_means_kernel<T><<<grid, block, 0, stream>>>(
            workspace_N_k, workspace_num, weights, means, eps, n, d, K);
        CUDA_CHECK_LAST_ERROR(m_step_finalize_means_kernel);
    }

    {
        int threads = 256;
        int blocks = (int)(((size_t)n * d + threads - 1) / threads);
        for (int k = 0; k < K; ++k) {
            weighted_center_kernel<T><<<blocks, threads, 0, stream>>>(
                X, resp, means, n, d, K, k, workspace_centered);
            CUDA_CHECK_LAST_ERROR(weighted_center_kernel);

            T* cov_k = covariances + (size_t)k * d * d;
            cublas_check_status(
                cublas_gemm<T>(handle, CUBLAS_OP_N, CUBLAS_OP_T, d, d, n, &one,
                               workspace_centered, d, workspace_centered, d,
                               &zero, cov_k, d),
                "cublas_gemm(covariance)");
        }
    }

    {
        int threads = 256;
        dim3 block(threads);
        dim3 grid(K);
        m_step_finalize_cov_cublas_kernel<T><<<grid, block, 0, stream>>>(
            workspace_N_k, covariances, reg_covar, eps, d, K);
        CUDA_CHECK_LAST_ERROR(m_step_finalize_cov_cublas_kernel);
    }
}

template <typename T, typename Device>
void def_e_step(nb::module_& m) {
    m.def(
        "e_step",
        [](gpu_array_c<const T, Device> X, gpu_array_c<const T, Device> weights,
           gpu_array_c<const T, Device> means,
           gpu_array_c<const T, Device> prec_chol,
           gpu_array_c<const T, Device> log_det_half,
           gpu_array_c<T, Device> log_prob, gpu_array_c<T, Device> resp,
           gpu_array_c<T, Device> ll_per_cell, int n, int d, int K,
           std::uintptr_t stream) {
            launch_e_step<T>(X.data(), weights.data(), means.data(),
                             prec_chol.data(), log_det_half.data(), n, d, K,
                             log_prob.data(), resp.data(), ll_per_cell.data(),
                             (cudaStream_t)stream);
        },
        "X"_a, "weights"_a, "means"_a, "prec_chol"_a, "log_det_half"_a,
        "log_prob"_a, "resp"_a, "ll_per_cell"_a, nb::kw_only(), "n"_a, "d"_a,
        "K"_a, "stream"_a = 0);
}

template <typename T, typename Device>
void def_e_step_cublas(nb::module_& m) {
    m.def(
        "e_step_cublas",
        [](gpu_array_c<const T, Device> X, gpu_array_c<const T, Device> weights,
           gpu_array_c<const T, Device> means,
           gpu_array_c<const T, Device> prec_chol,
           gpu_array_c<const T, Device> log_det_half,
           gpu_array_c<T, Device> centered_workspace,
           gpu_array_c<T, Device> y_workspace, gpu_array_c<T, Device> log_prob,
           gpu_array_c<T, Device> resp, gpu_array_c<T, Device> ll_per_cell,
           int n, int d, int K, std::uintptr_t stream, std::uintptr_t handle) {
            launch_e_step_cublas<T>(
                X.data(), weights.data(), means.data(), prec_chol.data(),
                log_det_half.data(), n, d, K, centered_workspace.data(),
                y_workspace.data(), log_prob.data(), resp.data(),
                ll_per_cell.data(), (cudaStream_t)stream,
                (cublasHandle_t)handle);
        },
        "X"_a, "weights"_a, "means"_a, "prec_chol"_a, "log_det_half"_a,
        "centered_workspace"_a, "y_workspace"_a, "log_prob"_a, "resp"_a,
        "ll_per_cell"_a, nb::kw_only(), "n"_a, "d"_a, "K"_a, "stream"_a = 0,
        "handle"_a);
}

template <typename T, typename Device>
void def_m_step(nb::module_& m) {
    m.def(
        "m_step",
        [](gpu_array_c<const T, Device> resp, gpu_array_c<const T, Device> X,
           gpu_array_c<const T, Device> ones, gpu_array_c<T, Device> weights,
           gpu_array_c<T, Device> means, gpu_array_c<T, Device> covariances,
           gpu_array_c<T, Device> N_k_workspace,
           gpu_array_c<T, Device> num_workspace,
           gpu_array_c<T, Device> centered_workspace, int n, int d, int K,
           T reg_covar, std::uintptr_t stream, std::uintptr_t handle) {
            launch_m_step<T>(resp.data(), X.data(), ones.data(), n, d, K,
                             reg_covar, weights.data(), means.data(),
                             covariances.data(), N_k_workspace.data(),
                             num_workspace.data(), centered_workspace.data(),
                             (cudaStream_t)stream, (cublasHandle_t)handle);
        },
        "resp"_a, "X"_a, "ones"_a, "weights"_a, "means"_a, "covariances"_a,
        "N_k_workspace"_a, "num_workspace"_a, "centered_workspace"_a,
        nb::kw_only(), "n"_a, "d"_a, "K"_a, "reg_covar"_a, "stream"_a = 0,
        "handle"_a);
}

template <typename T>
static inline void launch_spherical_gmm_fit_batched(
    const T* pvec, const int* offsets, const T* m0, const T* v0,
    const T* m1_init, const T* v1_init, int n_genes, int max_iter, T tol,
    T reg_covar, T* resp1, T* m1_out, T* v1_out, T* w1_out,
    cudaStream_t stream) {
    if (n_genes == 0) return;
    dim3 block(MIXSCAPE_EM_THREADS);
    dim3 grid(n_genes);
    mixscape_em_batched_kernel<T><<<grid, block, 0, stream>>>(
        pvec, offsets, m0, v0, m1_init, v1_init, n_genes, max_iter, tol,
        reg_covar, resp1, m1_out, v1_out, w1_out);
    CUDA_CHECK_LAST_ERROR(mixscape_em_batched_kernel);
}

template <typename T>
static inline void launch_mixscape_project_em(
    const T* dat, const long long* dat_offsets, const int* n_per_gene,
    const int* k_per_gene, const int* cell_offsets, const int* feat_offsets,
    const T* nt_cells_mean, const bool* guide_sel, const bool* nt_in_all,
    const int* active_genes, int n_active, int max_k, int max_iter, T tol,
    T reg_covar, T* pvec_scratch, T* resp1, cudaStream_t stream) {
    if (n_active == 0 || max_k == 0) return;
    dim3 block(MIXSCAPE_EM_THREADS);
    dim3 grid(n_active);
    size_t shmem = (size_t)max_k * sizeof(T);
    if (shmem > DEFAULT_DYNAMIC_SMEM_LIMIT) {
        cuda_check_runtime(
            cudaFuncSetAttribute(mixscape_project_em_kernel<T>,
                                 cudaFuncAttributeMaxDynamicSharedMemorySize,
                                 (int)shmem),
            "cudaFuncSetAttribute(mixscape_project_em_kernel)");
    }
    mixscape_project_em_kernel<T><<<grid, block, shmem, stream>>>(
        dat, dat_offsets, n_per_gene, k_per_gene, cell_offsets, feat_offsets,
        nt_cells_mean, guide_sel, nt_in_all, active_genes, n_active, max_iter,
        tol, reg_covar, pvec_scratch, resp1);
    CUDA_CHECK_LAST_ERROR(mixscape_project_em_kernel);
}

template <typename T, typename Device>
void def_mixscape_project_em(nb::module_& m) {
    m.def(
        "mixscape_project_em",
        [](gpu_array_c<const T, Device> dat,
           gpu_array_c<const long long, Device> dat_offsets,
           gpu_array_c<const int, Device> n_per_gene,
           gpu_array_c<const int, Device> k_per_gene,
           gpu_array_c<const int, Device> cell_offsets,
           gpu_array_c<const int, Device> feat_offsets,
           gpu_array_c<const T, Device> nt_cells_mean,
           gpu_array_c<const bool, Device> guide_sel,
           gpu_array_c<const bool, Device> nt_in_all,
           gpu_array_c<const int, Device> active_genes,
           gpu_array_c<T, Device> pvec_scratch, gpu_array_c<T, Device> resp1,
           int n_active, int max_k, int max_iter, T tol, T reg_covar,
           std::uintptr_t stream) {
            launch_mixscape_project_em<T>(
                dat.data(), dat_offsets.data(), n_per_gene.data(),
                k_per_gene.data(), cell_offsets.data(), feat_offsets.data(),
                nt_cells_mean.data(), guide_sel.data(), nt_in_all.data(),
                active_genes.data(), n_active, max_k, max_iter, tol, reg_covar,
                pvec_scratch.data(), resp1.data(), (cudaStream_t)stream);
        },
        "dat"_a, "dat_offsets"_a, "n_per_gene"_a, "k_per_gene"_a,
        "cell_offsets"_a, "feat_offsets"_a, "nt_cells_mean"_a, "guide_sel"_a,
        "nt_in_all"_a, "active_genes"_a, "pvec_scratch"_a, "resp1"_a,
        nb::kw_only(), "n_active"_a, "max_k"_a, "max_iter"_a, "tol"_a,
        "reg_covar"_a, "stream"_a = 0);
}

template <typename T, typename Device>
void def_spherical_gmm_fit_batched(nb::module_& m) {
    m.def(
        "spherical_gmm_fit_batched",
        [](gpu_array_c<const T, Device> pvec,
           gpu_array_c<const int, Device> offsets,
           gpu_array_c<const T, Device> m0, gpu_array_c<const T, Device> v0,
           gpu_array_c<const T, Device> m1_init,
           gpu_array_c<const T, Device> v1_init, gpu_array_c<T, Device> resp1,
           gpu_array_c<T, Device> m1_out, gpu_array_c<T, Device> v1_out,
           gpu_array_c<T, Device> w1_out, int n_genes, int max_iter, T tol,
           T reg_covar, std::uintptr_t stream) {
            launch_spherical_gmm_fit_batched<T>(
                pvec.data(), offsets.data(), m0.data(), v0.data(),
                m1_init.data(), v1_init.data(), n_genes, max_iter, tol,
                reg_covar, resp1.data(), m1_out.data(), v1_out.data(),
                w1_out.data(), (cudaStream_t)stream);
        },
        "pvec"_a, "offsets"_a, "m0"_a, "v0"_a, "m1_init"_a, "v1_init"_a,
        "resp1"_a, "m1_out"_a, "v1_out"_a, "w1_out"_a, nb::kw_only(),
        "n_genes"_a, "max_iter"_a, "tol"_a, "reg_covar"_a, "stream"_a = 0);
}

template <typename T, typename Device>
void def_precision_cholesky_full(nb::module_& m) {
    m.def(
        "precision_cholesky_full",
        [](gpu_array_c<const T, Device> covariances,
           gpu_array_c<T, Device> prec_chol, gpu_array_c<T, Device> log_det,
           int d, int K, std::uintptr_t stream, std::uintptr_t cublas_handle,
           std::uintptr_t cusolver_handle) {
            launch_precision_cholesky_full<T>(
                covariances.data(), prec_chol.data(), log_det.data(), d, K,
                (cudaStream_t)stream, (cublasHandle_t)cublas_handle,
                (cusolverDnHandle_t)cusolver_handle);
        },
        "covariances"_a, "prec_chol"_a, "log_det"_a, nb::kw_only(), "d"_a,
        "K"_a, "stream"_a = 0, "cublas_handle"_a, "cusolver_handle"_a);
}

template <typename Device>
void register_bindings(nb::module_& m) {
    def_e_step<float, Device>(m);
    def_e_step<double, Device>(m);
    def_e_step_cublas<float, Device>(m);
    def_e_step_cublas<double, Device>(m);
    def_m_step<float, Device>(m);
    def_m_step<double, Device>(m);
    def_precision_cholesky_full<float, Device>(m);
    def_precision_cholesky_full<double, Device>(m);
    def_spherical_gmm_fit_batched<float, Device>(m);
    def_spherical_gmm_fit_batched<double, Device>(m);
    def_mixscape_project_em<float, Device>(m);
    def_mixscape_project_em<double, Device>(m);
}

NB_MODULE(_gmm_cuda, m) {
    REGISTER_GPU_BINDINGS(register_bindings, m);
}
