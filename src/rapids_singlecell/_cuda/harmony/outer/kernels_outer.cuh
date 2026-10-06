#pragma once

#include <cuda_runtime.h>

template <typename T>
__global__ void harmony_correction_kernel(T* __restrict__ Z,
                                          const T* __restrict__ W,
                                          const int* __restrict__ cats,
                                          const T* __restrict__ R,
                                          long long n_cells, long long n_pcs) {
    long long N = n_cells * n_pcs;
    for (long long i = (long long)blockIdx.x * blockDim.x + threadIdx.x; i < N;
         i += (long long)blockDim.x * gridDim.x) {
        long long cell_idx = i / n_pcs;
        long long pc_idx = i % n_pcs;
        int cat = cats[cell_idx];
        T correction = W[(cat + 1) * n_pcs + pc_idx] * R[cell_idx];
        Z[i] -= correction;
    }
}

// Row 0 of each cluster's right-hand side (R^T X over all cells) as the sum
// of its per-batch rows. Layout (n_clusters, n_batches + 1, n_pcs).
template <typename T>
__global__ void sum_batch_rows_kernel(T* __restrict__ rhs, int n_batches,
                                      int n_pcs, int n_clusters) {
    int nb1 = n_batches + 1;
    for (long long i = (long long)blockIdx.x * blockDim.x + threadIdx.x;
         i < (long long)n_clusters * n_pcs;
         i += (long long)blockDim.x * gridDim.x) {
        long long k = i / n_pcs, d = i % n_pcs;
        T sum = T(0);
        for (int b = 1; b < nb1; ++b) sum += rhs[(k * nb1 + b) * n_pcs + d];
        rhs[k * nb1 * n_pcs + d] = sum;
    }
}
