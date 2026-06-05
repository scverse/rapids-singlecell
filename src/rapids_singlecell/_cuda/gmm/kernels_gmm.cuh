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

// ----------------------------------------------------------------------------
// Full-covariance precision-Cholesky helpers for the batched path that turns
// covariances into the upper precision factor with one ``potrfBatched`` + one
// ``trsmBatched``.
//
//   set_identity_batched_kernel : each component's (d, d) buffer -> identity,
//   so
//       the triangular solve ``L X = I`` yields ``X = L^{-1}``.
//   log_det_full_kernel         : log_det[k] = Σ_i log(prec_chol[k, i, i]); the
//       diagonal of the upper precision factor equals ``1 / L_ii``, so this is
//       ``-Σ_i log(L_ii)`` (the precision-Cholesky log-det term).
// ----------------------------------------------------------------------------
constexpr int FULL_LOGDET_THREADS = 256;

template <typename T>
__global__ void set_identity_batched_kernel(T* __restrict__ A, int d, int K) {
    size_t idx = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    size_t total = (size_t)K * d * d;
    if (idx >= total) return;
    size_t within = idx % ((size_t)d * d);
    int i = (int)(within / d);
    int j = (int)(within % d);
    A[idx] = (i == j) ? T(1) : T(0);
}

template <typename T>
__global__ void log_det_full_kernel(const T* __restrict__ prec_chol, int d,
                                    int K, T* __restrict__ log_det) {
    int k = blockIdx.x;
    int tid = threadIdx.x;
    if (k >= K) return;

    __shared__ T sh[FULL_LOGDET_THREADS];
    T local = T(0);
    const T* pc_k = prec_chol + (size_t)k * d * d;
    for (int i = tid; i < d; i += blockDim.x)
        local += log(pc_k[(size_t)i * d + i]);
    sh[tid] = local;
    __syncthreads();
    for (int off = blockDim.x / 2; off > 0; off >>= 1) {
        if (tid < off) sh[tid] += sh[tid + off];
        __syncthreads();
    }
    if (tid == 0) log_det[k] = sh[0];
}

// ----------------------------------------------------------------------------
// Batched per-segment 2-component spherical-Gaussian EM, component 0 pinned.
//
// One CUDA block per segment ("gene" in Mixscape): segment g owns the values
// pvec[offsets[g] : offsets[g+1]]. Component 0 (control) is fixed at
// (m0[g], v0[g]); component 1 (perturbed) is free, initialized at
// (m1_init[g], v1_init[g]) with uniform mixing weights. The whole EM runs in
// the block (block-reduced sufficient statistics, in-kernel convergence on the
// mean log-likelihood) for the 2-component (K=2, d=1) spherical mixture, and
// writes the component-1 posterior per cell (== Mixscape's KO probability) plus
// the fitted (m1, v1, w1) per segment.
// ----------------------------------------------------------------------------
constexpr int MIXSCAPE_EM_THREADS = 256;

// Reduce four per-thread partials to their block sums, broadcast to every
// thread (used by the fused projection+EM kernel for counts, dot products and
// EM sufficient statistics). One reduction pass over a (4, threads) buffer.
template <typename T>
__device__ inline void block_reduce4(T a, T b, T c, T d,
                                     T red[4][MIXSCAPE_EM_THREADS], T* out) {
    int tid = threadIdx.x;
    red[0][tid] = a;
    red[1][tid] = b;
    red[2][tid] = c;
    red[3][tid] = d;
    __syncthreads();
    for (int off = blockDim.x / 2; off > 0; off >>= 1) {
        if (tid < off) {
            red[0][tid] += red[0][tid + off];
            red[1][tid] += red[1][tid + off];
            red[2][tid] += red[2][tid + off];
            red[3][tid] += red[3][tid + off];
        }
        __syncthreads();
    }
    out[0] = red[0][0];
    out[1] = red[1][0];
    out[2] = red[2][0];
    out[3] = red[3][0];
    __syncthreads();
}

template <typename T>
__global__ void mixscape_em_batched_kernel(
    const T* __restrict__ pvec, const int* __restrict__ offsets,
    const T* __restrict__ m0, const T* __restrict__ v0,
    const T* __restrict__ m1_init, const T* __restrict__ v1_init, int n_genes,
    int max_iter, T tol, T reg_covar, T* __restrict__ resp1,
    T* __restrict__ m1_out, T* __restrict__ v1_out, T* __restrict__ w1_out) {
    int g = blockIdx.x;
    if (g >= n_genes) return;
    int tid = threadIdx.x;
    int start = offsets[g];
    int end = offsets[g + 1];
    int n = end - start;

    const T HALF_LOG2PI = T(0.91893853320467274178);
    const T WEIGHT_FLOOR = T(1e-10);

    __shared__ T red[4][MIXSCAPE_EM_THREADS];
    __shared__ T s_m1, s_v1, s_w1, s_prev;
    __shared__ int s_done;

    if (n <= 0) {
        if (tid == 0) {
            m1_out[g] = m1_init[g];
            v1_out[g] = v1_init[g];
            w1_out[g] = T(0.5);
        }
        return;
    }
    if (tid == 0) {
        s_m1 = m1_init[g];
        s_v1 = fmax(v1_init[g], reg_covar);
        s_w1 = T(0.5);
        s_prev = -T(1e30);
        s_done = 0;
    }
    __syncthreads();

    const T cm0 = m0[g];
    const T cv0 = fmax(v0[g], reg_covar);

    for (int iter = 0; iter < max_iter; ++iter) {
        T m1 = s_m1, v1 = s_v1, w1 = s_w1, w0 = T(1) - w1;
        T c0 = log(fmax(w0, WEIGHT_FLOOR)) - T(0.5) * log(cv0) - HALF_LOG2PI;
        T c1 = log(fmax(w1, WEIGHT_FLOOR)) - T(0.5) * log(v1) - HALF_LOG2PI;
        T inv2v0 = T(0.5) / cv0, inv2v1 = T(0.5) / v1;

        T lN1 = 0, lS1 = 0, lS2 = 0, lLL = 0;
        for (int i = start + tid; i < end; i += blockDim.x) {
            T y = pvec[i];
            T d0 = y - cm0, d1 = y - m1;
            T lp0 = c0 - inv2v0 * d0 * d0;
            T lp1 = c1 - inv2v1 * d1 * d1;
            T mx = fmax(lp0, lp1);
            T se = exp(lp0 - mx) + exp(lp1 - mx);
            T llc = mx + log(se);
            T r1 = exp(lp1 - llc);
            lN1 += r1;
            lS1 += r1 * y;
            lS2 += r1 * y * y;
            lLL += llc;
        }
        red[0][tid] = lN1;
        red[1][tid] = lS1;
        red[2][tid] = lS2;
        red[3][tid] = lLL;
        __syncthreads();
        for (int off = blockDim.x / 2; off > 0; off >>= 1) {
            if (tid < off) {
                red[0][tid] += red[0][tid + off];
                red[1][tid] += red[1][tid + off];
                red[2][tid] += red[2][tid + off];
                red[3][tid] += red[3][tid + off];
            }
            __syncthreads();
        }
        if (tid == 0) {
            T N1 = red[0][0], S1 = red[1][0], S2 = red[2][0], LL = red[3][0];
            T meanll = LL / T(n);
            if (fabs(meanll - s_prev) < tol) {
                s_done = 1;
            } else {
                s_prev = meanll;
                T inv = T(1) / fmax(N1, T(1e-12));
                T nm1 = S1 * inv;
                T nv1 = S2 * inv - nm1 * nm1 + reg_covar;
                s_w1 = N1 / T(n);
                s_m1 = nm1;
                s_v1 = fmax(nv1, reg_covar);
            }
        }
        __syncthreads();
        if (s_done) break;
    }

    // Final E-step with the converged parameters: write the component-1
    // posterior (Mixscape reads probabilities[:, 1] directly as post_prob).
    T m1 = s_m1, v1 = s_v1, w1 = s_w1, w0 = T(1) - w1;
    T c0 = log(fmax(w0, WEIGHT_FLOOR)) - T(0.5) * log(cv0) - HALF_LOG2PI;
    T c1 = log(fmax(w1, WEIGHT_FLOOR)) - T(0.5) * log(v1) - HALF_LOG2PI;
    T inv2v0 = T(0.5) / cv0, inv2v1 = T(0.5) / v1;
    for (int i = start + tid; i < end; i += blockDim.x) {
        T y = pvec[i];
        T d0 = y - cm0, d1 = y - m1;
        T lp0 = c0 - inv2v0 * d0 * d0;
        T lp1 = c1 - inv2v1 * d1 * d1;
        resp1[i] = T(1) / (T(1) + exp(lp0 - lp1));
    }
    if (tid == 0) {
        m1_out[g] = m1;
        v1_out[g] = v1;
        w1_out[g] = w1;
    }
}

// ----------------------------------------------------------------------------
// Fused projection + statistics + EM, one block per (active) gene.
//
// For each gene in ``active_genes`` this computes, entirely in-block, one
// Mixscape outer-iteration step on the gene's cells:
//   1. guide_mean[j] over the currently-perturbed cells, vec = guide_mean -
//   nt_mean
//   2. pvec[cell] = (dat[cell] . vec) / (vec . vec)
//   3. control/guide statistics (m0, v0, m1, v1; ddof=1) from pvec
//   4. the 2-component spherical EM (component 0 pinned to control)
//   5. the component-1 posterior per cell.
// The Python outer loop only updates labels/guide_sel and re-launches, so the
// per-gene projection + GMM that dominated the host loop now run on the GPU.
//
// Ragged layout: gene g owns dat[dat_offsets[g] : ...] as an (n_g, k_g)
// row-major block, cells [cell_offsets[g] : cell_offsets[g+1]] in the per-cell
// arrays, and features [feat_offsets[g] : ...] in nt_cells_mean. ``vec`` lives
// in dynamic shared memory sized to max_k by the launcher.
// ----------------------------------------------------------------------------
template <typename T>
__global__ void mixscape_project_em_kernel(
    const T* __restrict__ dat, const long long* __restrict__ dat_offsets,
    const int* __restrict__ n_per_gene, const int* __restrict__ k_per_gene,
    const int* __restrict__ cell_offsets, const int* __restrict__ feat_offsets,
    const T* __restrict__ nt_cells_mean, const bool* __restrict__ guide_sel,
    const bool* __restrict__ nt_in_all, const int* __restrict__ active_genes,
    int n_active, int max_iter, T tol, T reg_covar,
    T* __restrict__ pvec_scratch, T* __restrict__ resp1) {
    int a = blockIdx.x;
    if (a >= n_active) return;
    int g = active_genes[a];
    int tid = threadIdx.x;
    int n = n_per_gene[g];
    int k = k_per_gene[g];
    if (n <= 0 || k <= 0) return;

    const T* dat_g = dat + dat_offsets[g];
    const T* ntm_g = nt_cells_mean + feat_offsets[g];
    int cell0 = cell_offsets[g];
    const bool* guide_g = guide_sel + cell0;
    const bool* nt_g = nt_in_all + cell0;
    T* pvec_g = pvec_scratch + cell0;
    T* resp_g = resp1 + cell0;

    extern __shared__ char smem[];
    T* vec = reinterpret_cast<T*>(smem);  // (k,)

    const T HALF_LOG2PI = T(0.91893853320467274178);
    const T WEIGHT_FLOOR = T(1e-10);
    const T MIN_VAR = T(1e-12);

    __shared__ T red[4][MIXSCAPE_EM_THREADS];
    __shared__ T s_ng, s_nnt, s_dotvv;
    __shared__ T s_m0, s_v0, s_m1, s_v1, s_w1, s_prev;
    __shared__ int s_done;
    T out[4];

    // 1. guide / control counts.
    {
        T lg = 0, lnt = 0;
        for (int cell = tid; cell < n; cell += blockDim.x) {
            lg += guide_g[cell] ? T(1) : T(0);
            lnt += nt_g[cell] ? T(1) : T(0);
        }
        block_reduce4(lg, lnt, T(0), T(0), red, out);
        if (tid == 0) {
            s_ng = out[0];
            s_nnt = out[1];
        }
    }
    __syncthreads();
    T n_guide = s_ng, n_nt = s_nnt;
    T inv_ng = T(1) / fmax(n_guide, T(1));

    // 2. vec[j] = mean over perturbed cells of dat[:, j] - nt_cells_mean[j].
    for (int j = tid; j < k; j += blockDim.x) {
        T s = 0;
        for (int cell = 0; cell < n; ++cell)
            if (guide_g[cell]) s += dat_g[(size_t)cell * k + j];
        vec[j] = s * inv_ng - ntm_g[j];
    }
    __syncthreads();

    // 3. dotvv = vec . vec.
    {
        T ld = 0;
        for (int j = tid; j < k; j += blockDim.x) ld += vec[j] * vec[j];
        block_reduce4(ld, T(0), T(0), T(0), red, out);
        if (tid == 0) s_dotvv = out[0];
    }
    __syncthreads();
    // Guard the degenerate case vec . vec == 0 (guide mean == control mean
    // across all features) so pvec stays finite instead of NaN.
    T inv_dot = T(1) / fmax(s_dotvv, MIN_VAR);

    // 4. pvec[cell] = (dat[cell] . vec) / dotvv.
    for (int cell = tid; cell < n; cell += blockDim.x) {
        const T* row = dat_g + (size_t)cell * k;
        T s = 0;
        for (int j = 0; j < k; ++j) s += row[j] * vec[j];
        pvec_g[cell] = s * inv_dot;
    }
    __syncthreads();

    // 5. control/guide statistics (ddof=1) -> init mixture parameters.
    {
        T snt = 0, snt2 = 0, sg = 0, sg2 = 0;
        for (int cell = tid; cell < n; cell += blockDim.x) {
            T y = pvec_g[cell];
            if (nt_g[cell]) {
                snt += y;
                snt2 += y * y;
            }
            if (guide_g[cell]) {
                sg += y;
                sg2 += y * y;
            }
        }
        block_reduce4(snt, snt2, sg, sg2, red, out);
        if (tid == 0) {
            T Snt = out[0], Snt2 = out[1], Sg = out[2], Sg2 = out[3];
            s_m0 = Snt / fmax(n_nt, T(1));
            s_m1 = Sg / fmax(n_guide, T(1));
            T v0 = (n_nt > T(1)) ? (Snt2 - Snt * Snt / n_nt) / (n_nt - T(1))
                                 : MIN_VAR;
            T v1 = (n_guide > T(1))
                       ? (Sg2 - Sg * Sg / n_guide) / (n_guide - T(1))
                       : MIN_VAR;
            s_v0 = fmax(v0, MIN_VAR);
            s_v1 = fmax(v1, MIN_VAR);
            s_w1 = T(0.5);
            s_prev = -T(1e30);
            s_done = 0;
        }
    }
    __syncthreads();

    // 6. EM with component 0 pinned to (m0, v0).
    T cm0 = s_m0, cv0 = fmax(s_v0, reg_covar);
    for (int iter = 0; iter < max_iter; ++iter) {
        T m1 = s_m1, v1 = s_v1, w1 = s_w1, w0 = T(1) - w1;
        T c0 = log(fmax(w0, WEIGHT_FLOOR)) - T(0.5) * log(cv0) - HALF_LOG2PI;
        T c1 = log(fmax(w1, WEIGHT_FLOOR)) - T(0.5) * log(v1) - HALF_LOG2PI;
        T inv2v0 = T(0.5) / cv0, inv2v1 = T(0.5) / v1;
        T lN1 = 0, lS1 = 0, lS2 = 0, lLL = 0;
        for (int cell = tid; cell < n; cell += blockDim.x) {
            T y = pvec_g[cell];
            T d0 = y - cm0, d1 = y - m1;
            T lp0 = c0 - inv2v0 * d0 * d0;
            T lp1 = c1 - inv2v1 * d1 * d1;
            T mx = fmax(lp0, lp1);
            T se = exp(lp0 - mx) + exp(lp1 - mx);
            T llc = mx + log(se);
            T r1 = exp(lp1 - llc);
            lN1 += r1;
            lS1 += r1 * y;
            lS2 += r1 * y * y;
            lLL += llc;
        }
        block_reduce4(lN1, lS1, lS2, lLL, red, out);
        if (tid == 0) {
            T N1 = out[0], S1 = out[1], S2 = out[2], LL = out[3];
            T meanll = LL / T(n);
            if (fabs(meanll - s_prev) < tol) {
                s_done = 1;
            } else {
                s_prev = meanll;
                T inv = T(1) / fmax(N1, T(1e-12));
                T nm1 = S1 * inv;
                T nv1 = S2 * inv - nm1 * nm1 + reg_covar;
                s_w1 = N1 / T(n);
                s_m1 = nm1;
                s_v1 = fmax(nv1, reg_covar);
            }
        }
        __syncthreads();
        if (s_done) break;
    }

    // 7. final E-step: component-1 posterior per cell.
    T m1 = s_m1, v1 = s_v1, w1 = s_w1, w0 = T(1) - w1;
    T c0 = log(fmax(w0, WEIGHT_FLOOR)) - T(0.5) * log(cv0) - HALF_LOG2PI;
    T c1 = log(fmax(w1, WEIGHT_FLOOR)) - T(0.5) * log(v1) - HALF_LOG2PI;
    T inv2v0 = T(0.5) / cv0, inv2v1 = T(0.5) / v1;
    for (int cell = tid; cell < n; cell += blockDim.x) {
        T y = pvec_g[cell];
        T d0 = y - cm0, d1 = y - m1;
        T lp0 = c0 - inv2v0 * d0 * d0;
        T lp1 = c1 - inv2v1 * d1 * d1;
        resp_g[cell] = T(1) / (T(1) + exp(lp0 - lp1));
    }
}
