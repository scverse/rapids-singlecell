#pragma once

#include <cuda_runtime.h>

// One block per gene; 256 threads stride over a gene's cells / DE-features.
constexpr int MIXSCALE_THREADS = 256;

// Block-wide sum of four partials (every thread must call it).
template <typename T>
__device__ inline void block_reduce4(T a, T b, T c, T d,
                                     T red[4][MIXSCALE_THREADS], T* out) {
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

// Mixscale continuous score, one block per gene. Reads the gene's (rows x cols)
// block from dense X, scales in-kernel (z-score, ddof=1; do_scale off => raw),
// projects onto vec = guide mean - control mean (pvec = scaled.vec /
// ||vec||^2), then z-scores each guide cell's pvec against control and scatters
// it to scores_out by obs index (race-free: each guide cell belongs to one
// gene).
template <typename T>
__global__ void mixscale_project_score_kernel(
    const T* __restrict__ X, long long n_vars, const int* __restrict__ row_ids,
    const int* __restrict__ col_ids, const int* __restrict__ n_per_gene,
    const int* __restrict__ k_per_gene, const int* __restrict__ cell_offsets,
    const int* __restrict__ feat_offsets, const bool* __restrict__ is_guide,
    const bool* __restrict__ nt_in_all, int n_genes, bool do_scale,
    T* __restrict__ pvec_scratch, T* __restrict__ scores_out) {
    int g = blockIdx.x;
    if (g >= n_genes) return;
    int tid = threadIdx.x;
    int n = n_per_gene[g];
    int k = k_per_gene[g];
    if (n <= 0 || k <= 0) return;

    const int* rows = row_ids + cell_offsets[g];
    const int* cols = col_ids + feat_offsets[g];
    const bool* guide_g = is_guide + cell_offsets[g];
    const bool* nt_g = nt_in_all + cell_offsets[g];
    T* pvec_g = pvec_scratch + cell_offsets[g];

    extern __shared__ char smem[];
    T* vec = reinterpret_cast<T*>(smem);  // (k,)
    T* col_mean = vec + k;                // (k,)
    T* col_std = col_mean + k;            // (k,)

    const T MIN_VAR = T(1e-12);
    __shared__ T red[4][MIXSCALE_THREADS];
    __shared__ T s_ng, s_nnt, s_dotvv, s_ntmean, s_ntstd;
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
    T n_guide = fmax(s_ng, T(1)), n_nt = fmax(s_nnt, T(1));

    // 2. per-column scaling stats and direction vec[j] = (guideMean - ctrlMean)
    //    / std (column mean cancels). do_scale off => mean=0, std=1.
    for (int j = tid; j < k; j += blockDim.x) {
        long long col = cols[j];
        T sum = 0, gsum = 0, ntsum = 0;
        T sumsq = 0;
        for (int cell = 0; cell < n; ++cell) {
            T v = X[(size_t)rows[cell] * n_vars + col];
            sum += v;
            if (do_scale) sumsq += v * v;
            if (guide_g[cell]) gsum += v;
            if (nt_g[cell]) ntsum += v;
        }
        T cmean = T(0), sd = T(1);
        if (do_scale) {
            cmean = sum / T(n);
            T var = (n > 1) ? (sumsq - sum * sum / T(n)) / T(n - 1) : T(0);
            sd = sqrt(fmax(var, T(0)));
            if (sd == T(0)) sd = T(1);
        }
        col_mean[j] = cmean;
        col_std[j] = sd;
        vec[j] = (gsum / n_guide - ntsum / n_nt) / sd;
    }
    __syncthreads();

    // 3. dotvv = vec · vec. vec == 0 (guide mean == control mean) makes every
    //    projection 0 via the floored denom, matching pertpy's skip.
    {
        T ld = 0;
        for (int j = tid; j < k; j += blockDim.x) ld += vec[j] * vec[j];
        block_reduce4(ld, T(0), T(0), T(0), red, out);
        if (tid == 0) s_dotvv = out[0];
    }
    __syncthreads();
    T inv_dot = T(1) / fmax(s_dotvv, MIN_VAR);

    // 4. project every cell: pvec = scaled_row · vec / ‖vec‖² (read X again).
    for (int cell = tid; cell < n; cell += blockDim.x) {
        size_t base = (size_t)rows[cell] * n_vars;
        T s = 0;
        for (int j = 0; j < k; ++j) {
            T v = (X[base + cols[j]] - col_mean[j]) / col_std[j];
            s += v * vec[j];
        }
        pvec_g[cell] = s * inv_dot;
    }
    __syncthreads();

    // 5. control-cell projection stats: mean, population std (ddof=0).
    {
        T snt = 0, snt2 = 0;
        for (int cell = tid; cell < n; cell += blockDim.x) {
            if (nt_g[cell]) {
                T y = pvec_g[cell];
                snt += y;
                snt2 += y * y;
            }
        }
        block_reduce4(snt, snt2, T(0), T(0), red, out);
        if (tid == 0) {
            T mean = out[0] / n_nt;
            T var = out[1] / n_nt - mean * mean;
            T sd = sqrt(fmax(var, T(0)));
            s_ntmean = mean;
            s_ntstd = (sd == T(0)) ? T(1) : sd;  // pertpy: nt_std == 0 -> 1
        }
    }
    __syncthreads();
    T nt_mean = s_ntmean, inv_std = T(1) / s_ntstd;

    // 6. guide cells: standardized projection -> global scatter.
    for (int cell = tid; cell < n; cell += blockDim.x) {
        if (guide_g[cell])
            scores_out[rows[cell]] = (pvec_g[cell] - nt_mean) * inv_std;
    }
}
