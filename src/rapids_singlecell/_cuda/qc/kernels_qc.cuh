#pragma once

#include <cuda_runtime.h>

template <typename T, typename IdxT>
__global__ void qc_csc_kernel(const IdxT* __restrict__ indptr,
                              const IdxT* __restrict__ index,
                              const T* __restrict__ data,
                              T* __restrict__ sums_cells,
                              T* __restrict__ sums_genes,
                              int* __restrict__ cell_ex,
                              int* __restrict__ gene_ex, int n_genes) {
    int gene = blockDim.x * blockIdx.x + threadIdx.x;
    if (gene >= n_genes) return;
    IdxT start_idx = indptr[gene];
    IdxT stop_idx = indptr[gene + 1];
    T sums_genes_i = T(0);
    int gene_ex_i = 0;
    for (IdxT p = start_idx; p < stop_idx; ++p) {
        T v = data[p];
        IdxT cell = index[p];
        sums_genes_i += v;
        atomicAdd(&sums_cells[cell], v);
        ++gene_ex_i;
        atomicAdd(&cell_ex[cell], 1);
    }
    sums_genes[gene] = sums_genes_i;
    gene_ex[gene] = gene_ex_i;
}

template <typename T, typename IdxT>
__global__ void qc_csr_kernel(const IdxT* __restrict__ indptr,
                              const IdxT* __restrict__ index,
                              const T* __restrict__ data,
                              T* __restrict__ sums_cells,
                              T* __restrict__ sums_genes,
                              int* __restrict__ cell_ex,
                              int* __restrict__ gene_ex, int n_cells) {
    int cell = blockDim.x * blockIdx.x + threadIdx.x;
    if (cell >= n_cells) return;
    IdxT start_idx = indptr[cell];
    IdxT stop_idx = indptr[cell + 1];
    T sums_cells_i = T(0);
    int cell_ex_i = 0;
    for (IdxT p = start_idx; p < stop_idx; ++p) {
        T v = data[p];
        IdxT gene = index[p];
        atomicAdd(&sums_genes[gene], v);
        sums_cells_i += v;
        atomicAdd(&gene_ex[gene], 1);
        ++cell_ex_i;
    }
    sums_cells[cell] = sums_cells_i;
    cell_ex[cell] = cell_ex_i;
}

template <typename T>
__global__ void qc_dense_kernel(const T* __restrict__ data,
                                T* __restrict__ sums_cells,
                                T* __restrict__ sums_genes,
                                int* __restrict__ cell_ex,
                                int* __restrict__ gene_ex, int n_cells,
                                int n_genes) {
    int cell = blockDim.x * blockIdx.x + threadIdx.x;
    int gene = blockDim.y * blockIdx.y + threadIdx.y;
    if (cell >= n_cells || gene >= n_genes) return;
    long long idx = (long long)cell * n_genes + gene;
    T v = data[idx];
    if (v > T(0)) {
        atomicAdd(&sums_genes[gene], v);
        atomicAdd(&sums_cells[cell], v);
        atomicAdd(&gene_ex[gene], 1);
        atomicAdd(&cell_ex[cell], 1);
    }
}

template <typename T, typename IdxT>
__global__ void qc_csc_sub_kernel(const IdxT* __restrict__ indptr,
                                  const IdxT* __restrict__ index,
                                  const T* __restrict__ data,
                                  T* __restrict__ sums_cells,
                                  const bool* __restrict__ mask, int n_genes) {
    int gene = blockDim.x * blockIdx.x + threadIdx.x;
    if (gene >= n_genes) return;
    if (!mask[gene]) return;
    IdxT start_idx = indptr[gene];
    IdxT stop_idx = indptr[gene + 1];
    for (IdxT p = start_idx; p < stop_idx; ++p) {
        IdxT cell = index[p];
        atomicAdd(&sums_cells[cell], data[p]);
    }
}

template <typename T, typename IdxT>
__global__ void qc_csr_sub_kernel(const IdxT* __restrict__ indptr,
                                  const IdxT* __restrict__ index,
                                  const T* __restrict__ data,
                                  T* __restrict__ sums_cells,
                                  const bool* __restrict__ mask, int n_cells) {
    int cell = blockDim.x * blockIdx.x + threadIdx.x;
    if (cell >= n_cells) return;
    IdxT start_idx = indptr[cell];
    IdxT stop_idx = indptr[cell + 1];
    T sums_cells_i = T(0);
    for (IdxT p = start_idx; p < stop_idx; ++p) {
        IdxT gene = index[p];
        if (mask[gene]) sums_cells_i += data[p];
    }
    sums_cells[cell] = sums_cells_i;
}

template <typename T>
__global__ void qc_dense_sub_kernel(const T* __restrict__ data,
                                    T* __restrict__ sums_cells,
                                    const bool* __restrict__ mask, int n_cells,
                                    int n_genes) {
    int cell = blockDim.x * blockIdx.x + threadIdx.x;
    int gene = blockDim.y * blockIdx.y + threadIdx.y;
    if (cell >= n_cells || gene >= n_genes) return;
    if (!mask[gene]) return;
    long long idx = (long long)cell * n_genes + gene;
    atomicAdd(&sums_cells[cell], data[idx]);
}
