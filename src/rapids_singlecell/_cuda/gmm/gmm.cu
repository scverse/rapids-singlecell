#include <cuda_runtime.h>
#include <math_constants.h>

#include <cublas_v2.h>

#include <limits>
#include <stdexcept>
#include <string>

#include "../cublas_helpers.cuh"
#include "../nb_types.h"

#include "kernels_gmm.cuh"

using namespace nb::literals;

constexpr int E_STEP_BLOCK = 64;
constexpr int E_STEP_LARGE64_TILE = 64;
constexpr int E_STEP_THREAD64_BLOCK = 512;
constexpr int NORMALIZE_BLOCK = 32;
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

template <typename Device>
void register_bindings(nb::module_& m) {
    def_e_step<float, Device>(m);
    def_e_step<double, Device>(m);
    def_e_step_cublas<float, Device>(m);
    def_e_step_cublas<double, Device>(m);
    def_m_step<float, Device>(m);
    def_m_step<double, Device>(m);
}

NB_MODULE(_gmm_cuda, m) {
    REGISTER_GPU_BINDINGS(register_bindings, m);
}
