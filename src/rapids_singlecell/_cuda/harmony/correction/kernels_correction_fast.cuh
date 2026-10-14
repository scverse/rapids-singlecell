#pragma once

#include <cuda_runtime.h>

#include "../storage.cuh"

// ---- Ridge coefficients of one batch key ----
// The design [intercept, batch indicators] with ridge lambda_kb on the batch
// terms has the arrowhead Gram [[N_k, O_k^T], [O_k, diag(O_k + lambda_k)]].
// Eliminating the batch terms, with S_b = R_b^T X_b and
// f_b = 1 / (O_kb + lambda_kb):
//   beta0 = sum_b f_b lambda_kb S_b / sum_b f_b lambda_kb O_kb  (0 if empty)
//   W_b   = f_b (S_b - O_kb beta0)
// Unlike N_k - sum_b f_b O_kb^2 (the inverse's pivot), no term cancels at
// large counts, so a fixed lambda stays accurate in float32. One thread per
// (cluster, PC) of clusters k0 .. k0 + n_k - 1; S and W hold rows b of
// cluster k at (k - k0) * cluster_stride + b * n_pcs. A cluster without cells
// (all O_kb = 0) gets no correction.
template <typename T>
__global__ void ridge_coefficients_kernel(
    const T* __restrict__ O, const T* __restrict__ lambda_kb,
    const T* __restrict__ S, T* __restrict__ W, long long cluster_stride,
    int n_batches, int n_clusters, int n_pcs, int k0, int n_k) {
    long long t = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (t >= (long long)n_k * n_pcs) return;
    int k = k0 + (int)(t / n_pcs);
    size_t base = (size_t)(k - k0) * cluster_stride + t % n_pcs;
    double c = 0, num = 0;
    for (int b = 0; b < n_batches; ++b) {
        double o = O[(size_t)b * n_clusters + k];
        double lam = lambda_kb[(size_t)b * n_clusters + k];
        double w = lam / (o + lam);  // f_b lambda_kb
        c += w * o;
        num += w * S[base + (size_t)b * n_pcs];
    }
    double beta0 = c > 0 ? num / c : 0;
    for (int b = 0; b < n_batches; ++b) {
        double o = O[(size_t)b * n_clusters + k];
        double lam = lambda_kb[(size_t)b * n_clusters + k];
        size_t i = base + (size_t)b * n_pcs;
        W[i] = T((S[i] - o * beta0) / (o + lam));
    }
}

// ---- Gather column: dst[i] = src[i * n_cols + col] ----
template <typename T>
__global__ void gather_column_kernel(const T* __restrict__ src,
                                     T* __restrict__ dst, int col, int n_rows,
                                     int n_cols) {
    for (long long i = (long long)blockIdx.x * blockDim.x + threadIdx.x;
         i < n_rows; i += (long long)blockDim.x * gridDim.x) {
        dst[i] = src[(size_t)i * n_cols + col];
    }
}
