#pragma once

#include <cuda_runtime.h>

#include <algorithm>
#include <cstdint>
#include <type_traits>

#include "../../nb_types.h"
#include "../storage.cuh"

// ---- Ridge coefficients of one batch key ----
// The design [intercept, batch indicators] with ridge lambda_kb on the batch
// terms has the arrowhead Gram [[N_k, O_k^T], [O_k, diag(O_k + lambda_k)]].
// Eliminating the batch terms, with S_b = R_b^T X_b and
// f_b = 1 / (O_kb + lambda_kb):
//   beta0 = sum_b f_b lambda_kb S_b / sum_b f_b lambda_kb O_kb  (0 if empty)
//   W_b   = f_b (S_b - O_kb beta0)
// Unlike N_k - sum_b f_b O_kb^2 (the inverse's pivot), no term cancels at
// large counts, so a fixed lambda stays accurate in float32. Clusters k0 ..
// k0 + n_k - 1; S and W hold rows b of cluster k at (k - k0) * cluster_stride
// + b * n_pcs. A cluster without cells (all O_kb = 0) gets no correction.

// beta[(k - k0) * n_pcs + d] of cluster k0 + blockIdx.x: both sums run over
// b in order (fma chains in double), the denominator in warp 0, the numerators
// of 32 PCs in each other warp; f_b lambda_kb once per tile of b.
template <typename T>
__global__ void __launch_bounds__(288)
    ridge_intercept_kernel(const T* __restrict__ O,
                           const T* __restrict__ lambda_kb,
                           const T* __restrict__ S, double* __restrict__ beta,
                           long long cluster_stride, int n_batches,
                           int n_clusters, int n_pcs, int k0) {
    constexpr int TB = 256;
    __shared__ double ws[TB], os[TB], denominator;
    const int k = k0 + blockIdx.x, warp = threadIdx.x / 32;
    const int d =
        (blockIdx.y * (blockDim.x / 32 - 1) + warp - 1) * 32 + threadIdx.x % 32;
    const bool pc = warp > 0 && d < n_pcs;
    const T* s = S + blockIdx.x * cluster_stride + d;
    double c = 0, num = 0;
    for (int b0 = 0; b0 < n_batches; b0 += TB) {
        const int n = min(TB, n_batches - b0);
        __syncthreads();
        for (int t = threadIdx.x; t < n; t += blockDim.x) {
            double o = O[(size_t)(b0 + t) * n_clusters + k];
            double lam = lambda_kb[(size_t)(b0 + t) * n_clusters + k];
            ws[t] = lam / (o + lam);  // f_b lambda_kb
            os[t] = o;
        }
        __syncthreads();
        if (warp == 0) {
#pragma unroll 16
            for (int t = 0; t < n; ++t) c += ws[t] * os[t];
        } else if (pc) {
#pragma unroll 16
            for (int t = 0; t < n; ++t)
                num += ws[t] * s[(size_t)(b0 + t) * n_pcs];
        }
    }
    if (threadIdx.x == 0) denominator = c;
    __syncthreads();
    if (pc)
        beta[(size_t)blockIdx.x * n_pcs + d] =
            denominator > 0 ? num / denominator : 0;
}

// W from beta, one element per thread.
template <typename T>
__global__ void ridge_coefficients_kernel(
    const T* __restrict__ O, const T* __restrict__ lambda_kb,
    const T* __restrict__ S, const double* __restrict__ beta, T* __restrict__ W,
    long long cluster_stride, int n_batches, int n_clusters, int n_pcs, int k0,
    int n_k) {
    const long long n = (long long)n_k * n_batches * n_pcs;
    for (long long t = (long long)blockIdx.x * blockDim.x + threadIdx.x; t < n;
         t += (long long)blockDim.x * gridDim.x) {
        const int d = int(t % n_pcs), b = int(t / n_pcs % n_batches),
                  kk = int(t / n_pcs / n_batches);
        double o = O[(size_t)b * n_clusters + k0 + kk];
        double lam = lambda_kb[(size_t)b * n_clusters + k0 + kk];
        size_t i = kk * cluster_stride + (size_t)b * n_pcs + d;
        W[i] = T((S[i] - o * beta[(size_t)kk * n_pcs + d]) / (o + lam));
    }
}

// W of clusters k0 .. k0 + n_k - 1; beta: n_k x n_pcs doubles of scratch.
template <typename T>
static void ridge_coefficients(const T* O, const T* lambda_kb, const T* S, T* W,
                               double* beta, long long cluster_stride,
                               int n_batches, int n_clusters, int n_pcs, int k0,
                               int n_k, cudaStream_t stream) {
    const int warps = std::min(8, (n_pcs + 31) / 32);  // + warp 0
    ridge_intercept_kernel<T>
        <<<dim3(n_k, (n_pcs + 32 * warps - 1) / (32 * warps)), 32 * (warps + 1),
           0, stream>>>(O, lambda_kb, S, beta, cluster_stride, n_batches,
                        n_clusters, n_pcs, k0);
    CUDA_CHECK_LAST_ERROR(ridge_intercept_kernel);
    ridge_coefficients_kernel<T>
        <<<strided_grid((long long)n_k * n_batches * n_pcs, 256), 256, 0,
           stream>>>(O, lambda_kb, S, beta, W, cluster_stride, n_batches,
                     n_clusters, n_pcs, k0, n_k);
    CUDA_CHECK_LAST_ERROR(ridge_coefficients_kernel);
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

// a[cat + 1, pc0 .. pc0 + 1] = sum over the category's cells (sorted by
// category: cat_offsets) of v * bias; one block per (category, pair of PCs).
template <typename T>
__global__ void scatter_add_kernel_with_bias_block(
    const T* __restrict__ v, const int* __restrict__ cat_offsets, int n_pcs,
    int n_batches, T* __restrict__ a, const T* __restrict__ bias) {
    using VecPC = std::conditional_t<std::is_same_v<T, float>, float2, double2>;
    int pairs = (n_pcs + 1) / 2;
    if ((int)blockIdx.x >= n_batches * pairs) return;
    int cat = blockIdx.x / pairs + 1, pc0 = blockIdx.x % pairs * 2;
    bool has_pc1 = pc0 + 1 < n_pcs;
    T acc0 = T(0), acc1 = T(0);
    for (int cell = cat_offsets[cat - 1] + threadIdx.x; cell < cat_offsets[cat];
         cell += blockDim.x) {
        const T* ptr = v + (size_t)cell * n_pcs + pc0;
        T bb = __ldg(bias + cell);
        if (has_pc1 && ((uintptr_t)ptr & (sizeof(VecPC) - 1)) == 0) {
            VecPC vv = *(const VecPC*)ptr;
            acc0 += vv.x * bb;
            acc1 += vv.y * bb;
        } else {
            acc0 += ptr[0] * bb;
            if (has_pc1) acc1 += ptr[1] * bb;
        }
    }
    acc0 = block_sum(acc0);
    acc1 = block_sum(acc1);
    if (threadIdx.x == 0) {
        a[cat * n_pcs + pc0] = acc0;
        if (has_pc1) a[cat * n_pcs + pc0 + 1] = acc1;
    }
}

// Z -= R W[cat + 1] element by element.
template <typename T>
__global__ void harmony_correction_kernel(T* __restrict__ Z,
                                          const T* __restrict__ W,
                                          const int* __restrict__ cats,
                                          const T* __restrict__ R,
                                          long long n_cells, long long n_pcs) {
    long long N = n_cells * n_pcs;
    for (long long i = (long long)blockIdx.x * blockDim.x + threadIdx.x; i < N;
         i += (long long)blockDim.x * gridDim.x) {
        long long cell_idx = i / n_pcs;
        long long pc_idx = i % n_pcs;
        int cat = cats[cell_idx];
        T correction = W[(cat + 1) * n_pcs + pc_idx] * R[cell_idx];
        Z[i] -= correction;
    }
}
