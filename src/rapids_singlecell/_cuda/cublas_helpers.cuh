#pragma once

#include <cublas_v2.h>

#include <stdexcept>
#include <string>

// ---------- cublas_check_status ----------

static inline void cublas_check_status(cublasStatus_t status,
                                       const char* what) {
    if (status != CUBLAS_STATUS_SUCCESS) {
        throw std::runtime_error(std::string(what) +
                                 " failed with cuBLAS status " +
                                 std::to_string(static_cast<int>(status)));
    }
}

// ---------- cublas_gemm ----------

template <typename T>
static inline cublasStatus_t cublas_gemm(cublasHandle_t handle,
                                         cublasOperation_t transa,
                                         cublasOperation_t transb, int m, int n,
                                         int k, const T* alpha, const T* A,
                                         int lda, const T* B, int ldb,
                                         const T* beta, T* C, int ldc);

template <>
inline cublasStatus_t cublas_gemm<float>(
    cublasHandle_t handle, cublasOperation_t transa, cublasOperation_t transb,
    int m, int n, int k, const float* alpha, const float* A, int lda,
    const float* B, int ldb, const float* beta, float* C, int ldc) {
    return cublasSgemm(handle, transa, transb, m, n, k, alpha, A, lda, B, ldb,
                       beta, C, ldc);
}

template <>
inline cublasStatus_t cublas_gemm<double>(
    cublasHandle_t handle, cublasOperation_t transa, cublasOperation_t transb,
    int m, int n, int k, const double* alpha, const double* A, int lda,
    const double* B, int ldb, const double* beta, double* C, int ldc) {
    return cublasDgemm(handle, transa, transb, m, n, k, alpha, A, lda, B, ldb,
                       beta, C, ldc);
}

// ---------- cublas_gemv ----------

template <typename T>
static inline cublasStatus_t cublas_gemv(cublasHandle_t handle,
                                         cublasOperation_t trans, int m, int n,
                                         const T* alpha, const T* A, int lda,
                                         const T* x, int incx, const T* beta,
                                         T* y, int incy);

template <>
inline cublasStatus_t cublas_gemv<float>(cublasHandle_t handle,
                                         cublasOperation_t trans, int m, int n,
                                         const float* alpha, const float* A,
                                         int lda, const float* x, int incx,
                                         const float* beta, float* y,
                                         int incy) {
    return cublasSgemv(handle, trans, m, n, alpha, A, lda, x, incx, beta, y,
                       incy);
}

template <>
inline cublasStatus_t cublas_gemv<double>(cublasHandle_t handle,
                                          cublasOperation_t trans, int m, int n,
                                          const double* alpha, const double* A,
                                          int lda, const double* x, int incx,
                                          const double* beta, double* y,
                                          int incy) {
    return cublasDgemv(handle, trans, m, n, alpha, A, lda, x, incx, beta, y,
                       incy);
}

// ---------- cublas_gemm_strided_batched ----------

template <typename T>
static inline cublasStatus_t cublas_gemm_strided_batched(
    cublasHandle_t handle, cublasOperation_t transa, cublasOperation_t transb,
    int m, int n, int k, const T* alpha, const T* A, int lda, long long strideA,
    const T* B, int ldb, long long strideB, const T* beta, T* C, int ldc,
    long long strideC, int batchCount);

template <>
inline cublasStatus_t cublas_gemm_strided_batched<float>(
    cublasHandle_t handle, cublasOperation_t transa, cublasOperation_t transb,
    int m, int n, int k, const float* alpha, const float* A, int lda,
    long long strideA, const float* B, int ldb, long long strideB,
    const float* beta, float* C, int ldc, long long strideC, int batchCount) {
    return cublasSgemmStridedBatched(handle, transa, transb, m, n, k, alpha, A,
                                     lda, strideA, B, ldb, strideB, beta, C,
                                     ldc, strideC, batchCount);
}

template <>
inline cublasStatus_t cublas_gemm_strided_batched<double>(
    cublasHandle_t handle, cublasOperation_t transa, cublasOperation_t transb,
    int m, int n, int k, const double* alpha, const double* A, int lda,
    long long strideA, const double* B, int ldb, long long strideB,
    const double* beta, double* C, int ldc, long long strideC, int batchCount) {
    return cublasDgemmStridedBatched(handle, transa, transb, m, n, k, alpha, A,
                                     lda, strideA, B, ldb, strideB, beta, C,
                                     ldc, strideC, batchCount);
}
