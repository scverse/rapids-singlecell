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
// gene). All reductions/stats accumulate in double regardless of T (so float32
// inputs keep full precision, matching mean_var); only X reads and the final
// score write use T.
template <typename T>
__global__ void mixscale_project_score_kernel(
    const T* __restrict__ X, long long n_vars, const int* __restrict__ row_ids,
    const int* __restrict__ col_ids, const int* __restrict__ n_per_gene,
    const int* __restrict__ k_per_gene, const int* __restrict__ cell_offsets,
    const int* __restrict__ feat_offsets, const bool* __restrict__ is_guide,
    const bool* __restrict__ nt_in_all, int n_genes, bool do_scale,
    double* __restrict__ pvec_scratch, T* __restrict__ scores_out) {
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
    double* pvec_g = pvec_scratch + cell_offsets[g];

    extern __shared__ double smem_d[];
    double* vec = smem_d;            // (k,)
    double* col_mean = vec + k;      // (k,)
    double* col_std = col_mean + k;  // (k,)

    const double MIN_VAR = 1e-12;
    __shared__ double red[4][MIXSCALE_THREADS];
    __shared__ double s_ng, s_nnt, s_dotvv, s_ntmean, s_ntstd;
    double out[4];

    // 1. guide / control counts.
    {
        double lg = 0, lnt = 0;
        for (int cell = tid; cell < n; cell += blockDim.x) {
            lg += guide_g[cell] ? 1.0 : 0.0;
            lnt += nt_g[cell] ? 1.0 : 0.0;
        }
        block_reduce4(lg, lnt, 0.0, 0.0, red, out);
        if (tid == 0) {
            s_ng = out[0];
            s_nnt = out[1];
        }
    }
    __syncthreads();
    double n_guide = fmax(s_ng, 1.0), n_nt = fmax(s_nnt, 1.0);

    // 2. per-column scaling stats and direction vec[j] = (guideMean - ctrlMean)
    //    / std (column mean cancels). do_scale off => mean=0, std=1.
    for (int j = tid; j < k; j += blockDim.x) {
        long long col = cols[j];
        double sum = 0, gsum = 0, ntsum = 0, sumsq = 0;
        for (int cell = 0; cell < n; ++cell) {
            double v = (double)X[(size_t)rows[cell] * n_vars + col];
            sum += v;
            if (do_scale) sumsq += v * v;
            if (guide_g[cell]) gsum += v;
            if (nt_g[cell]) ntsum += v;
        }
        double cmean = 0.0, sd = 1.0;
        if (do_scale) {
            cmean = sum / (double)n;
            double var = (n > 1)
                             ? (sumsq - sum * sum / (double)n) / (double)(n - 1)
                             : 0.0;
            sd = sqrt(fmax(var, 0.0));
            if (sd == 0.0) sd = 1.0;
        }
        col_mean[j] = cmean;
        col_std[j] = sd;
        vec[j] = (gsum / n_guide - ntsum / n_nt) / sd;
    }
    __syncthreads();

    // 3. dotvv = vec . vec. vec == 0 (guide mean == control mean) makes every
    //    projection 0 via the floored denom, matching pertpy's skip.
    {
        double ld = 0;
        for (int j = tid; j < k; j += blockDim.x) ld += vec[j] * vec[j];
        block_reduce4(ld, 0.0, 0.0, 0.0, red, out);
        if (tid == 0) s_dotvv = out[0];
    }
    __syncthreads();
    double inv_dot = 1.0 / fmax(s_dotvv, MIN_VAR);

    // 4. project every cell: pvec = scaled_row . vec / ||vec||^2 (read X
    // again).
    for (int cell = tid; cell < n; cell += blockDim.x) {
        size_t base = (size_t)rows[cell] * n_vars;
        double s = 0;
        for (int j = 0; j < k; ++j) {
            double v = ((double)X[base + cols[j]] - col_mean[j]) / col_std[j];
            s += v * vec[j];
        }
        pvec_g[cell] = s * inv_dot;
    }
    __syncthreads();

    // 5. control-cell projection stats: mean, sample std (ddof=1).
    {
        double snt = 0, snt2 = 0;
        for (int cell = tid; cell < n; cell += blockDim.x) {
            if (nt_g[cell]) {
                double y = pvec_g[cell];
                snt += y;
                snt2 += y * y;
            }
        }
        block_reduce4(snt, snt2, 0.0, 0.0, red, out);
        if (tid == 0) {
            double mean = out[0] / n_nt;
            double var = out[1] / n_nt - mean * mean;
            var = (n_nt > 1.0) ? var * n_nt / (n_nt - 1.0) : 0.0;
            double sd = sqrt(fmax(var, 0.0));
            s_ntmean = mean;
            s_ntstd = (sd == 0.0) ? 1.0 : sd;  // pertpy: nt_std == 0 -> 1
        }
    }
    __syncthreads();
    double nt_mean = s_ntmean, inv_std = 1.0 / s_ntstd;

    // 6. guide cells: standardized projection -> global scatter.
    for (int cell = tid; cell < n; cell += blockDim.x) {
        if (guide_g[cell])
            scores_out[rows[cell]] = (T)((pvec_g[cell] - nt_mean) * inv_std);
    }
}
