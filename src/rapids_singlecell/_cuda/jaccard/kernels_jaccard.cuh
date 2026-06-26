#pragma once

#include <cuda_runtime.h>

// One thread per KNN entry (i, slot): edge = i*k + slot, j = knn[edge].
// Writes |N(i) & N(j)| (shared-neighbor count) for each directed KNN edge.
// The self entry (j == i) and any out-of-range index write 0 and are dropped
// downstream. Inner loops skip column 0 so the count matches the exclude-self
// definition. All row offsets are computed in 64-bit (overflow-proof for any
// n_obs); the grid-stride loop covers n_edges beyond one launch.
__global__ void jaccard_shared_counts_kernel(const int* __restrict__ knn,
                                             int n_obs, int k,
                                             float* __restrict__ jaccard_vals) {
    const long long n_edges = (long long)n_obs * k;
    const long long stride = (long long)blockDim.x * gridDim.x;
    for (long long edge = (long long)blockIdx.x * blockDim.x + threadIdx.x;
         edge < n_edges; edge += stride) {
        const long long i = edge / k;
        const int j = knn[edge];
        if (j == (int)i || j < 0 || j >= n_obs) {
            jaccard_vals[edge] = 0.0f;
            continue;
        }
        const int* Ni = knn + i * (long long)k;
        const int* Nj = knn + (long long)j * k;
        int c = 0;
        for (int a = 1; a < k; ++a) {
            const int va = Ni[a];
            for (int b = 1; b < k; ++b) {
                c += (va == Nj[b]);
            }
        }
        jaccard_vals[edge] = (float)c / (2 * (k - 1) - c);
        ;
    }
}
