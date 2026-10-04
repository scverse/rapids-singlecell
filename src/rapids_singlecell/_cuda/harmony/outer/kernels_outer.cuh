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

// ---------- batched_correction ----------
// Each thread handles one (cell, pc) pair, accumulating corrections from all
// clusters. W_all layout: (n_clusters, n_batches+1, n_pcs) row-major
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

// One warp per cell: Z = X + Z, where Z holds -sum_k R_k W[k, batch] from
// the per-batch GEMMs; optionally L2-normalized as l2_row_normalize_kernel
// does.
template <typename T>
__global__ void add_rows_normalize_kernel(const T* __restrict__ X,
                                          T* __restrict__ Z, int n_cells,
                                          int n_pcs, bool normalize) {
    int lane = threadIdx.x & 31;
    long long warps = ((long long)gridDim.x * blockDim.x) >> 5;
    for (long long i = ((long long)blockIdx.x * blockDim.x + threadIdx.x) >> 5;
         i < n_cells; i += warps) {
        const T* x = X + (size_t)i * n_pcs;
        T* z = Z + (size_t)i * n_pcs;
        T sq = T(0);
        for (int d = lane; d < n_pcs; d += 32) {
            T v = x[d] + z[d];
            z[d] = v;
            sq += v * v;
        }
        if (!normalize) continue;
#pragma unroll
        for (int offset = 16; offset > 0; offset >>= 1)
            sq += __shfl_xor_sync(0xffffffff, sq, offset);
        T scale = rsqrt(sq);
        if (scale > T(1e12)) scale = T(1e12);
        for (int d = lane; d < n_pcs; d += 32) z[d] *= scale;
    }
}
