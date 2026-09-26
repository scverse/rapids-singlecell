#pragma once

#include <cuda_runtime.h>
#include "../minor_tiles.cuh"
#include "../qc/kernels_qc.cuh"

template <typename T>
__global__ void qc_dense_cells_kernel(const T* __restrict__ data,
                                      T* __restrict__ sums_cells,
                                      int* __restrict__ cell_ex, int n_cells,
                                      int n_genes) {
    int cell = blockDim.x * blockIdx.x + threadIdx.x;
    int gene = blockDim.y * blockIdx.y + threadIdx.y;
    if (cell >= n_cells || gene >= n_genes) return;
    long long idx = (long long)cell * n_genes + gene;
    T v = data[idx];
    if (v > T(0)) {
        atomicAdd(&sums_cells[cell], v);
        atomicAdd(&cell_ex[cell], 1);
    }
}

template <typename T>
__global__ void qc_dense_genes_kernel(const T* __restrict__ data,
                                      T* __restrict__ sums_genes,
                                      int* __restrict__ gene_ex, int n_cells,
                                      int n_genes) {
    int cell = blockDim.x * blockIdx.x + threadIdx.x;
    int gene = blockDim.y * blockIdx.y + threadIdx.y;
    if (cell >= n_cells || gene >= n_genes) return;
    long long idx = (long long)cell * n_genes + gene;
    T v = data[idx];
    if (v > T(0)) {
        atomicAdd(&sums_genes[gene], v);
        atomicAdd(&gene_ex[gene], 1);
    }
}
