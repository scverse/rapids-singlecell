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
