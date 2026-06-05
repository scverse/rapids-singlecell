#pragma once

#include <cuda_runtime.h>

// ----------------------------------------------------------------------------
// Per-(n, k) E-step log-probability.
//
// Each block (k, n_chunk) caches means[k] and prec_chol[k] in shared memory,
// then each thread computes mahalanobis for one cell against the cached
// component. Output is row-major log_prob[n, k] with the log-weight already
// folded in:
//
//   y[j]        = Σ_d (X[n, d] − means[k, d]) · prec_chol[k, d, j]
//   mahal[n, k] = Σ_j y[j]²
//   log_prob[n, k] = −0.5·d·log(2π) + log_det_half[k] − 0.5·mahal +
//   log(weights[k])
//
// A separate normalize kernel does the per-row logsumexp.
// ----------------------------------------------------------------------------

constexpr float LOG_2PI_F = 1.8378770664093453f;
constexpr double LOG_2PI_D = 1.8378770664093453;

template <typename T>
__device__ __forceinline__ T log_2pi_const();
template <>
__device__ __forceinline__ float log_2pi_const<float>() {
    return LOG_2PI_F;
}
template <>
__device__ __forceinline__ double log_2pi_const<double>() {
    return LOG_2PI_D;
}

__device__ __forceinline__ int upper_tri_col_offset(int col) {
    return (col * (col + 1)) / 2;
}

template <typename T, int D = 0>
__global__ void e_step_log_prob_small_kernel(
    const T* __restrict__ X,             // (n, d) row-major
    const T* __restrict__ weights,       // (K,)
    const T* __restrict__ means,         // (K, d)
    const T* __restrict__ prec_chol,     // (K, d, d) row-major; upper factor
                                         // with cov_inv = chol·cholᵀ
    const T* __restrict__ log_det_half,  // (K,)
    int n, int d, int K,
    T* __restrict__ log_prob  // (n, K)
) {
    static_assert(D >= 0 && D <= 64,
                  "GMM small E-step supports runtime d or fixed D <= 64");
    constexpr bool fixed_d = D != 0;
    int dim = fixed_d ? D : d;
    int k = blockIdx.y;
    int n_idx = blockIdx.x * blockDim.x + threadIdx.x;
    int tid = threadIdx.x;

    extern __shared__ unsigned char smem_raw[];
    T* sh_mean = reinterpret_cast<T*>(smem_raw);
    T* sh_pc = sh_mean + dim;

    // Cooperatively load means[k] and the used upper triangle of prec_chol[k]
    // into shared memory.
    for (int i = tid; i < dim; i += blockDim.x)
        sh_mean[i] = means[(size_t)k * dim + i];
    int pc_size_dense = dim * dim;
    for (int i = tid; i < pc_size_dense; i += blockDim.x) {
        int row = i / dim;
        int col = i - row * dim;
        if (row <= col) {
            sh_pc[upper_tri_col_offset(col) + row] =
                prec_chol[(size_t)k * pc_size_dense + i];
        }
    }

    __shared__ T sh_const;
    if (tid == 0) {
        sh_const = T(-0.5) * T(dim) * log_2pi_const<T>() + log_det_half[k] +
                   log(weights[k]);
    }

    __syncthreads();

    if (n_idx >= n) return;

    // Compute mahal = || (X[n] - μ_k) · prec_chol[k] ||²
    T centered_vals[fixed_d ? D : 64];
    if constexpr (fixed_d) {
#pragma unroll
        for (int dd = 0; dd < D; ++dd)
            centered_vals[dd] = X[(size_t)n_idx * D + dd] - sh_mean[dd];
    } else {
        for (int dd = 0; dd < dim; ++dd)
            centered_vals[dd] = X[(size_t)n_idx * dim + dd] - sh_mean[dd];
    }

    T mahal = T(0);
    if constexpr (fixed_d) {
#pragma unroll
        for (int j = 0; j < D; ++j) {
            T y = T(0);
            int pc_col = upper_tri_col_offset(j);
#pragma unroll
            for (int dd = 0; dd <= j; ++dd) {
                y += centered_vals[dd] * sh_pc[pc_col + dd];
            }
            mahal += y * y;
        }
    } else {
        for (int j = 0; j < dim; ++j) {
            T y = T(0);
            int pc_col = upper_tri_col_offset(j);
            // prec_chol is the upper triangular precision factor, so entries
            // below the diagonal are zero. Skip that half of the multiply.
            for (int dd = 0; dd <= j; ++dd) {
                y += centered_vals[dd] * sh_pc[pc_col + dd];
            }
            mahal += y * y;
        }
    }
    log_prob[(size_t)n_idx * K + k] = sh_const - T(0.5) * mahal;
}

template <typename T, int TILE_D>
__global__ void e_step_log_prob_large_d_thread64_kernel(
    const T* __restrict__ X,             // (n, d) row-major
    const T* __restrict__ weights,       // (K,)
    const T* __restrict__ means,         // (K, d)
    const T* __restrict__ prec_chol,     // (K, d, d) row-major; upper factor
    const T* __restrict__ log_det_half,  // (K,)
    int n, int d, int K,
    T* __restrict__ log_prob  // (n, K)
) {
    static_assert(TILE_D == 64,
                  "GMM thread64 E-step expects a 64-column precision tile");

    int k = blockIdx.y;
    int row = blockIdx.x * blockDim.x + threadIdx.x;
    int tid = threadIdx.x;

    extern __shared__ unsigned char smem_raw[];
    T* sh_mean = reinterpret_cast<T*>(smem_raw);  // (64,)
    T* sh_pc = sh_mean + TILE_D;                  // (64, 64)

    __shared__ T sh_const;
    if (tid == 0) {
        sh_const = T(-0.5) * T(d) * log_2pi_const<T>() + log_det_half[k] +
                   log(weights[k]);
    }

    T local_mahal = T(0);
    const T* pc = prec_chol + (size_t)k * d * d;

    for (int j_base = 0; j_base < d; j_base += TILE_D) {
        int cols_in_tile = min(TILE_D, d - j_base);
        int dd_limit = min(d, j_base + TILE_D);
        T y[TILE_D];
#pragma unroll
        for (int col = 0; col < TILE_D; ++col) y[col] = T(0);

        for (int dd_base = 0; dd_base < dd_limit; dd_base += TILE_D) {
            int feats_in_tile = min(TILE_D, dd_limit - dd_base);

            for (int idx = tid; idx < TILE_D; idx += blockDim.x) {
                sh_mean[idx] = (idx < feats_in_tile)
                                   ? means[(size_t)k * d + dd_base + idx]
                                   : T(0);
            }

            constexpr int pc_tile_elems = TILE_D * TILE_D;
            for (int idx = tid; idx < pc_tile_elems; idx += blockDim.x) {
                int feat = idx / TILE_D;
                int col_local = idx - feat * TILE_D;
                int dd = dd_base + feat;
                int col = j_base + col_local;
                T val = T(0);
                if (feat < feats_in_tile && col_local < cols_in_tile &&
                    dd <= col) {
                    val = pc[(size_t)dd * d + col];
                }
                sh_pc[feat * TILE_D + col_local] = val;
            }

            __syncthreads();

            if (row < n) {
#pragma unroll
                for (int feat = 0; feat < TILE_D; ++feat) {
                    if (feat >= feats_in_tile) break;
                    T diff =
                        X[(size_t)row * d + dd_base + feat] - sh_mean[feat];
#pragma unroll
                    for (int col = 0; col < TILE_D; ++col) {
                        if (col >= cols_in_tile) break;
                        y[col] += diff * sh_pc[feat * TILE_D + col];
                    }
                }
            }

            __syncthreads();
        }

        if (row < n) {
#pragma unroll
            for (int col = 0; col < TILE_D; ++col) {
                if (col >= cols_in_tile) break;
                local_mahal += y[col] * y[col];
            }
        }
    }

    if (row < n)
        log_prob[(size_t)row * K + k] = sh_const - T(0.5) * local_mahal;
}

template <typename T>
__global__ void e_step_center_kernel(const T* __restrict__ X,
                                     const T* __restrict__ means, int n, int d,
                                     int k, T* __restrict__ centered) {
    size_t idx = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    size_t total = (size_t)n * d;
    if (idx >= total) return;

    int col = idx % d;
    centered[idx] = X[idx] - means[(size_t)k * d + col];
}

template <typename T>
__global__ void e_step_log_prob_from_y_kernel(
    const T* __restrict__ y, const T* __restrict__ weights,
    const T* __restrict__ log_det_half, int n, int d, int K, int k,
    T* __restrict__ log_prob) {
    int row = blockIdx.x * blockDim.x + threadIdx.x;
    if (row >= n) return;

    T mahal = T(0);
    T compensation = T(0);
    for (int col = 0; col < d; ++col) {
        T v = y[(size_t)row * d + col];
        T term = v * v - compensation;
        T next = mahal + term;
        compensation = (next - mahal) - term;
        mahal = next;
    }

    T constant =
        T(-0.5) * T(d) * log_2pi_const<T>() + log_det_half[k] + log(weights[k]);
    log_prob[(size_t)row * K + k] = constant - T(0.5) * mahal;
}

// ----------------------------------------------------------------------------
// Per-cell logsumexp normalize: resp[n, k] = exp(log_prob[n, k] − logΣ_k).
// Also writes per-cell log-likelihood (= logΣ_k) into ll_per_cell for later
// reduction. One block per cell; threads stride across K.
// ----------------------------------------------------------------------------

template <typename T>
__global__ void e_step_normalize_kernel(
    const T* __restrict__ log_prob,  // (n, K)
    int n, int K,
    T* __restrict__ resp,        // (n, K)
    T* __restrict__ ll_per_cell  // (n,)
) {
    int n_idx = blockIdx.x;
    if (n_idx >= n) return;
    int tid = threadIdx.x;

    __shared__ T sh_max;
    __shared__ T sh_sum;

    // pass 1: max over K
    T local_max = -CUDART_INF_F;
    for (int k = tid; k < K; k += blockDim.x) {
        T v = log_prob[n_idx * K + k];
        if (v > local_max) local_max = v;
    }
    // warp + block reduce max
    for (int off = 16; off > 0; off >>= 1) {
        T other = __shfl_down_sync(0xffffffff, local_max, off);
        if (other > local_max) local_max = other;
    }
    if (tid == 0) sh_max = local_max;
    __syncthreads();
    T mx = sh_max;

    // pass 2: sum exp(log_prob - max)
    T local_sum = T(0);
    for (int k = tid; k < K; k += blockDim.x) {
        local_sum += exp(log_prob[n_idx * K + k] - mx);
    }
    for (int off = 16; off > 0; off >>= 1)
        local_sum += __shfl_down_sync(0xffffffff, local_sum, off);
    if (tid == 0) {
        sh_sum = local_sum;
        T log_total = log(local_sum) + mx;
        ll_per_cell[n_idx] = log_total;
    }
    __syncthreads();
    T log_total = log(sh_sum) + mx;

    // pass 3: write normalized responsibilities
    for (int k = tid; k < K; k += blockDim.x) {
        resp[n_idx * K + k] = exp(log_prob[n_idx * K + k] - log_total);
    }
}

template <typename T>
__global__ void m_step_finalize_means_kernel(const T* __restrict__ N_k,
                                             const T* __restrict__ num,
                                             T* __restrict__ weights,
                                             T* __restrict__ means, T eps,
                                             int n, int d, int K) {
    int k = blockIdx.x;
    int tid = threadIdx.x;
    if (k >= K) return;

    T Nk = N_k[k] + T(10) * eps;
    T inv_Nk = T(1) / Nk;
    if (tid == 0) weights[k] = Nk / T(n);

    for (int i = tid; i < d; i += blockDim.x)
        means[k * d + i] = num[k * d + i] * inv_Nk;
}

template <typename T>
__global__ void weighted_center_kernel(const T* __restrict__ X,
                                       const T* __restrict__ resp,
                                       const T* __restrict__ means, int n,
                                       int d, int K, int k,
                                       T* __restrict__ centered) {
    size_t idx = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    size_t total = (size_t)n * d;
    if (idx >= total) return;

    int row = idx / d;
    int col = idx - (size_t)row * d;
    T r = resp[row * K + k];
    centered[idx] = sqrt(r) * (X[idx] - means[k * d + col]);
}

template <typename T>
__global__ void m_step_finalize_cov_cublas_kernel(const T* __restrict__ N_k,
                                                  T* __restrict__ covariances,
                                                  T reg_covar, T eps, int d,
                                                  int K) {
    int k = blockIdx.x;
    int tid = threadIdx.x;
    if (k >= K) return;

    T Nk = N_k[k] + T(10) * eps;
    T inv_Nk = T(1) / Nk;
    int total = d * d;
    T* cov = covariances + (size_t)k * d * d;

    for (int idx = tid; idx < total; idx += blockDim.x) {
        int i = idx / d;
        int j = idx % d;
        if (i > j) continue;

        // cuBLAS wrote the row-major symmetric result through a column-major
        // view. Read the transposed element and write a symmetric row-major
        // covariance.
        T v = cov[j * d + i] * inv_Nk;
        if (i == j) v += reg_covar;
        cov[i * d + j] = v;
        if (i != j) cov[j * d + i] = v;
    }
}
