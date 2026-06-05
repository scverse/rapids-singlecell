#pragma once

#include <cuda_runtime.h>
#include <type_traits>

// ---- Compute inverse matrix for one or all clusters (algebraic fast-inverse)
// ---- Uses the algebraic simplification to avoid explicit matrix inversion:
//   factor[b] = 1 / (O_k[b] + lambda_kb[b,k])
//   P_row0[b] = -factor[b] * O_k[b]
//   c = N_k - sum(factor[b] * O_k[b]^2)
//   c_inv = 1 / c
//   inv[0,0] = c_inv
//   inv[0,j] = c_inv * P_row0[j-1]        (j > 0)
//   inv[i,0] = P_row0[i-1] * c_inv        (i > 0)
//   inv[i,j] = P_row0[i-1]*c_inv*P_row0[j-1] + factor[i-1]*delta(i,j)  (i,j >
//   0)
//
// lambda_kb is (n_batches, n_clusters) row-major, same layout as O.
//
// When cluster_k >= 0, computes only that cluster (launch with 1 block).
//   g_factor/g_P_row0 are flat (n_batches,) workspace.
// When cluster_k < 0, computes all clusters (launch with n_clusters blocks).
//   g_factor/g_P_row0 are (n_clusters, n_batches) workspace.
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

    // Choose workspace: single-cluster or batched global memory
    T* my_factor;
    T* my_P_row0;
    if (cluster_k >= 0) {
        my_factor = g_factor;
        my_P_row0 = g_P_row0;
    } else {
        my_factor = g_factor + (size_t)k * n_batches;
        my_P_row0 = g_P_row0 + (size_t)k * n_batches;
    }

    // Phase 1: compute factor, P_row0, and reduce for N_k and c
    T local_Nk = T(0);
    T local_c_neg = T(0);

    for (int b = threadIdx.x; b < n_batches; b += blockDim.x) {
        T o_val =
            O[b * n_clusters + k];  // O is (n_batches, n_clusters) row-major
        T lam = lambda_kb[b * n_clusters + k];  // same layout as O
        T f = T(1) / (o_val + lam);
        T p = -f * o_val;
        my_factor[b] = f;
        my_P_row0[b] = p;
        local_Nk += o_val;
        local_c_neg += f * o_val * o_val;
    }

    // Warp-level reduction
#pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1) {
        local_Nk += __shfl_down_sync(0xffffffff, local_Nk, offset);
        local_c_neg += __shfl_down_sync(0xffffffff, local_c_neg, offset);
    }

    // Block-level reduction
    __shared__ T s_Nk[32];
    __shared__ T s_c_neg[32];
    int warp_id = threadIdx.x >> 5;
    int lane = threadIdx.x & 31;

    if (lane == 0) {
        s_Nk[warp_id] = local_Nk;
        s_c_neg[warp_id] = local_c_neg;
    }
    __syncthreads();

    __shared__ T s_c_inv;
    if (threadIdx.x < 32) {
        int nwarps = (blockDim.x + 31) >> 5;
        T Nk =
            (static_cast<int>(threadIdx.x) < nwarps) ? s_Nk[threadIdx.x] : T(0);
        T c_neg = (static_cast<int>(threadIdx.x) < nwarps)
                      ? s_c_neg[threadIdx.x]
                      : T(0);
#pragma unroll
        for (int offset = 16; offset > 0; offset >>= 1) {
            Nk += __shfl_down_sync(0xffffffff, Nk, offset);
            c_neg += __shfl_down_sync(0xffffffff, c_neg, offset);
        }
        if (threadIdx.x == 0) {
            s_c_inv = T(1) / (Nk - c_neg);
        }
    }
    __syncthreads();

    T c_inv = s_c_inv;

    // Phase 2: fill inv_mat entries (read factor/P_row0 from global memory)
    int total = nb1 * nb1;
    for (int idx = threadIdx.x; idx < total; idx += blockDim.x) {
        int i = idx / nb1;
        int j = idx % nb1;

        T val;
        if (i == 0 && j == 0) {
            val = c_inv;
        } else if (i == 0) {
            val = c_inv * my_P_row0[j - 1];
        } else if (j == 0) {
            val = my_P_row0[i - 1] * c_inv;
        } else {
            val = my_P_row0[i - 1] * c_inv * my_P_row0[j - 1];
            if (i == j) val += my_factor[i - 1];
        }
        inv[idx] = val;
    }
}

// ---- Gather column: dst[i] = src[i * n_cols + col] ----
// Extracts a single column from a row-major matrix.
template <typename T>
__global__ void gather_column_kernel(const T* __restrict__ src,
                                     T* __restrict__ dst, int col, int n_rows,
                                     int n_cols) {
    for (long long i = (long long)blockIdx.x * blockDim.x + threadIdx.x;
         i < n_rows; i += (long long)blockDim.x * gridDim.x) {
        dst[i] = src[(size_t)i * n_cols + col];
    }
}
