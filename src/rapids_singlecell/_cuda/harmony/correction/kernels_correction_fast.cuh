#pragma once

#include <cuda_runtime.h>

#include "../storage.cuh"

// ---- Inverse of the (n_batches + 1)^2 correction matrix of a cluster ----
// Closed form instead of an explicit inversion:
//   factor[b] = 1 / (O_k[b] + lambda_kb[b,k])
//   P_row0[b] = -factor[b] * O_k[b]
//   c_inv = 1 / (N_k - sum(factor[b] * O_k[b]^2))
//   inv[0,0] = c_inv
//   inv[0,j] = c_inv * P_row0[j-1], inv[i,0] = P_row0[i-1] * c_inv
//   inv[i,j] = P_row0[i-1]*c_inv*P_row0[j-1] + factor[i-1]*delta(i,j)
// O and lambda_kb are (n_batches, n_clusters) row-major. With cluster_k >= 0
// one block computes that cluster (g_factor/g_P_row0: n_batches); otherwise
// block k computes cluster k (n_clusters x n_batches workspace).
template <typename T>
__global__ void compute_inv_mats_kernel(const T* __restrict__ O,
                                        const T* __restrict__ lambda_kb,
                                        T* __restrict__ inv_mats,
                                        T* __restrict__ g_factor,
                                        T* __restrict__ g_P_row0, int n_batches,
                                        int n_clusters, int cluster_k = -1) {
    int k = (cluster_k >= 0) ? cluster_k : blockIdx.x;
    if (k >= n_clusters) return;
    int nb1 = n_batches + 1;
    T* inv = (cluster_k >= 0) ? inv_mats : inv_mats + (size_t)k * nb1 * nb1;
    size_t ws = cluster_k >= 0 ? 0 : (size_t)k * n_batches;
    T* my_factor = g_factor + ws;
    T* my_P_row0 = g_P_row0 + ws;

    T local_Nk = T(0), local_c_neg = T(0);
    for (int b = threadIdx.x; b < n_batches; b += blockDim.x) {
        T o_val = O[b * n_clusters + k];
        T f = T(1) / (o_val + lambda_kb[b * n_clusters + k]);
        my_factor[b] = f;
        my_P_row0[b] = -f * o_val;
        local_Nk += o_val;
        local_c_neg += f * o_val * o_val;
    }
    T Nk = block_sum(local_Nk);
    T c_inv = T(1) / (Nk - block_sum(local_c_neg));
    for (int b = threadIdx.x; b < n_batches; b += blockDim.x)
        inv[(size_t)(b + 1) * nb1] = my_P_row0[b] * c_inv;
    __syncthreads();

    for (int idx = threadIdx.x; idx < nb1 * nb1; idx += blockDim.x) {
        int i = idx / nb1, j = idx % nb1;
        if (i == 0) {
            inv[idx] = j == 0 ? c_inv : c_inv * my_P_row0[j - 1];
        } else if (j > 0) {
            T val = inv[(size_t)i * nb1] * my_P_row0[j - 1];
            if (i == j) val += my_factor[i - 1];
            inv[idx] = val;
        }
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
