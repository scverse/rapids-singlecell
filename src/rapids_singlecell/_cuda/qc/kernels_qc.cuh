#pragma once

#include <cuda_runtime.h>
#include "../minor_tiles.cuh"

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

/// Minor-axis sum (as T) and stored-entry count per column (see
/// minor_tiles.cuh). Layout: double sums, int counts.
template <typename T>
struct QcOp {
    const T* data;
    T* sums;
    int* counts;
    int tile_size;
    static constexpr size_t bytes_per_col = sizeof(double) + sizeof(int);
    static constexpr bool needs_rows = false;
    __device__ double* s_sum(char* acc) const {
        return reinterpret_cast<double*>(acc);
    }
    __device__ int* s_cnt(char* acc) const {
        return reinterpret_cast<int*>(acc + (size_t)tile_size * sizeof(double));
    }
    __device__ bool row_active(int) const {
        return true;
    }
    __device__ void zero_col(char* acc, int g, int) const {
        s_sum(acc)[g] = 0.0;
        s_cnt(acc)[g] = 0;
    }
    __device__ void add(char* acc, long long q, int g) const {
        atomicAdd(&s_sum(acc)[g], static_cast<double>(data[q]));
        atomicAdd(&s_cnt(acc)[g], 1);
    }
    __device__ void flush_col(const char* acc, int, int col, int g) const {
        char* a = const_cast<char*>(acc);
        const int c = s_cnt(a)[g];
        if (c != 0) {
            atomicAdd(&sums[col], static_cast<T>(s_sum(a)[g]));
            atomicAdd(&counts[col], c);
        }
    }
    __device__ void add_global(long long q, int col, int) const {
        atomicAdd(&sums[col], data[q]);
        atomicAdd(&counts[col], 1);
    }
    void zero_outputs(int minor, int, cudaStream_t stream) const {
        cuda_check(cudaMemsetAsync(sums, 0, (size_t)minor * sizeof(T), stream),
                   "cudaMemsetAsync(QcOp outputs)");
        cuda_check(
            cudaMemsetAsync(counts, 0, (size_t)minor * sizeof(int), stream),
            "cudaMemsetAsync(QcOp outputs)");
    }
};
