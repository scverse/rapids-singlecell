#pragma once

#include <cuda_runtime.h>

template <typename T, typename IdxT>
__global__ void sparse_norm_res_csc_kernel(
    const IdxT* __restrict__ indptr, const IdxT* __restrict__ index,
    const T* __restrict__ data, const T* __restrict__ sums_cells,
    const T* __restrict__ sums_genes, T* __restrict__ residuals,
    const T inv_sum_total, const T clip, const T inv_theta, int n_cells,
    int n_genes) {
    int gene = blockDim.x * blockIdx.x + threadIdx.x;
    if (gene >= n_genes) {
        return;
    }
    IdxT start = indptr[gene];
    IdxT stop = indptr[gene + 1];
    IdxT sparse_idx = start;
    for (int cell = 0; cell < n_cells; ++cell) {
        T mu = sums_genes[gene] * sums_cells[cell] * inv_sum_total;
        long long res_index = static_cast<long long>(cell) * n_genes + gene;
        if (sparse_idx < stop && index[sparse_idx] == cell) {
            residuals[res_index] += data[sparse_idx];
            ++sparse_idx;
        }
        residuals[res_index] -= mu;
        residuals[res_index] *= rsqrt(mu + mu * mu * inv_theta);
        // clamp to [-clip, clip]
        if (residuals[res_index] < -clip) residuals[res_index] = -clip;
        if (residuals[res_index] > clip) residuals[res_index] = clip;
    }
}

template <typename T, typename IdxT>
__global__ void sparse_norm_res_csr_kernel(
    const IdxT* __restrict__ indptr, const IdxT* __restrict__ index,
    const T* __restrict__ data, const T* __restrict__ sums_cells,
    const T* __restrict__ sums_genes, T* __restrict__ residuals,
    const T inv_sum_total, const T clip, const T inv_theta, int n_cells,
    int n_genes) {
    int cell = blockDim.x * blockIdx.x + threadIdx.x;
    if (cell >= n_cells) {
        return;
    }
    IdxT start = indptr[cell];
    IdxT stop = indptr[cell + 1];
    IdxT sparse_idx = start;
    for (int gene = 0; gene < n_genes; ++gene) {
        long long res_index = static_cast<long long>(cell) * n_genes + gene;
        T mu = sums_genes[gene] * sums_cells[cell] * inv_sum_total;
        if (sparse_idx < stop && index[sparse_idx] == gene) {
            residuals[res_index] += data[sparse_idx];
            ++sparse_idx;
        }
        residuals[res_index] -= mu;
        residuals[res_index] *= rsqrt(mu + mu * mu * inv_theta);

        if (residuals[res_index] < -clip) residuals[res_index] = -clip;
        if (residuals[res_index] > clip) residuals[res_index] = clip;
    }
}

template <typename T>
__global__ void dense_norm_res_kernel(const T* __restrict__ X,
                                      T* __restrict__ residuals,
                                      const T* __restrict__ sums_cells,
                                      const T* __restrict__ sums_genes,
                                      const T inv_inv_sum_total, const T clip,
                                      const T inv_theta, int n_cells,
                                      int n_genes) {
    int cell = blockDim.x * blockIdx.x + threadIdx.x;
    int gene = blockDim.y * blockIdx.y + threadIdx.y;
    if (cell >= n_cells || gene >= n_genes) {
        return;
    }
    T mu = sums_genes[gene] * sums_cells[cell] * inv_inv_sum_total;
    long long res_index = static_cast<long long>(cell) * n_genes + gene;
    T r = X[res_index] - mu;
    r *= rsqrt(mu + mu * mu * inv_theta);
    if (r < -clip) r = -clip;
    if (r > clip) r = clip;
    residuals[res_index] = r;
}
