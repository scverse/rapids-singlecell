#pragma once

#include <cuda_runtime.h>

// Compute column sums (sums_genes) and row sums (sums_cells) from CSC sparse
// matrix One thread per column (gene), atomicAdd for row sums
template <typename T, typename IdxT>
__global__ void sparse_sum_csc_kernel(const IdxT* __restrict__ indptr,
                                      const IdxT* __restrict__ index,
                                      const T* __restrict__ data,
                                      T* __restrict__ sums_genes,
                                      T* __restrict__ sums_cells, int n_genes) {
    int gene = blockDim.x * blockIdx.x + threadIdx.x;
    if (gene >= n_genes) {
        return;
    }
    IdxT start = indptr[gene];
    IdxT stop = indptr[gene + 1];
    T col_sum = (T)0;
    for (IdxT i = start; i < stop; ++i) {
        T val = data[i];
        col_sum += val;
        atomicAdd(&sums_cells[index[i]], val);
    }
    sums_genes[gene] = col_sum;
}

// Welford's single-pass algorithm for variance of clipped Pearson residuals
// (CSC sparse)
template <typename T, typename IdxT>
__global__ void csc_hvg_res_kernel(
    const IdxT* __restrict__ indptr, const IdxT* __restrict__ index,
    const T* __restrict__ data, const T* __restrict__ sums_genes,
    const T* __restrict__ sums_cells, T* __restrict__ residuals,
    const T inv_sum_total, const T clip, const T inv_theta, int n_genes,
    int n_cells) {
    int gene = blockDim.x * blockIdx.x + threadIdx.x;
    if (gene >= n_genes) return;

    IdxT start = indptr[gene];
    IdxT stop = indptr[gene + 1];

    T gene_sum = sums_genes[gene];

    // Welford's online algorithm: single pass for mean and variance
    T mean = (T)0;
    T M2 = (T)0;
    IdxT sparse_idx = start;

    for (int cell = 0; cell < n_cells; ++cell) {
        T mu = gene_sum * sums_cells[cell] * inv_sum_total;
        T value = (T)0;
        if (sparse_idx < stop && index[sparse_idx] == cell) {
            value = data[sparse_idx];
            ++sparse_idx;
        }
        T diff = value - mu;
        T x = fmin(fmax(diff * rsqrt(mu + mu * mu * inv_theta), -clip), clip);

        // Welford update
        T delta = x - mean;
        mean += delta / (T)(cell + 1);
        T delta2 = x - mean;
        M2 = fma(delta, delta2, M2);
    }
    residuals[gene] = M2 / (T)n_cells;
}

// Welford's single-pass algorithm for variance of clipped Pearson residuals
// (dense, column-major)
template <typename T>
__global__ void dense_hvg_res_kernel(const T* __restrict__ data,
                                     const T* __restrict__ sums_genes,
                                     const T* __restrict__ sums_cells,
                                     T* __restrict__ residuals,
                                     const T inv_sum_total, const T clip,
                                     const T inv_theta, int n_genes,
                                     int n_cells) {
    int gene = blockDim.x * blockIdx.x + threadIdx.x;
    if (gene >= n_genes) return;

    T gene_sum = sums_genes[gene];

    // Welford's online algorithm: single pass for mean and variance
    T mean = (T)0;
    T M2 = (T)0;

    for (int cell = 0; cell < n_cells; ++cell) {
        long long res_index = static_cast<long long>(gene) * n_cells + cell;
        T mu = gene_sum * sums_cells[cell] * inv_sum_total;
        T value = data[res_index];
        T diff = value - mu;
        T x = fmin(fmax(diff * rsqrt(mu + mu * mu * inv_theta), -clip), clip);

        // Welford update
        T delta = x - mean;
        mean += delta / (T)(cell + 1);
        T delta2 = x - mean;
        M2 = fma(delta, delta2, M2);
    }
    residuals[gene] = M2 / (T)n_cells;
}
