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
// does. With SLOTS > 0 (n_pcs <= 32 * SLOTS) each lane keeps its columns
// lane + 32 c in registers, so Z is written once.
template <typename T, int SLOTS>
__global__ void add_rows_normalize_kernel(const T* __restrict__ X,
                                          T* __restrict__ Z, int n_cells,
                                          int n_pcs, bool normalize) {
    int lane = threadIdx.x & 31;
    long long warps = ((long long)gridDim.x * blockDim.x) >> 5;
    for (long long i = ((long long)blockIdx.x * blockDim.x + threadIdx.x) >> 5;
         i < n_cells; i += warps) {
        const T* x = X + (size_t)i * n_pcs;
        T* z = Z + (size_t)i * n_pcs;
        T v[SLOTS > 0 ? SLOTS : 1], sq = T(0);
        if constexpr (SLOTS > 0) {
#pragma unroll
            for (int c = 0; c < SLOTS; ++c) {
                int d = lane + 32 * c;
                v[c] = d < n_pcs ? x[d] + z[d] : T(0);
                sq += v[c] * v[c];
            }
        } else {
            for (int d = lane; d < n_pcs; d += 32) {
                T value = x[d] + z[d];
                z[d] = value;
                sq += value * value;
            }
        }
        T scale = T(1);
        if (normalize) {
#pragma unroll
            for (int offset = 16; offset > 0; offset >>= 1)
                sq += __shfl_xor_sync(0xffffffff, sq, offset);
            scale = rsqrt(sq);
            if (scale > T(1e12)) scale = T(1e12);
        }
        if constexpr (SLOTS > 0) {
#pragma unroll
            for (int c = 0; c < SLOTS; ++c)
                if (lane + 32 * c < n_pcs) z[lane + 32 * c] = v[c] * scale;
        } else if (normalize) {
            for (int d = lane; d < n_pcs; d += 32) z[d] *= scale;
        }
    }
}

template <typename T>
static void add_rows_normalize(const T* X, T* Z, int n_cells, int n_pcs,
                               bool normalize, int grid, cudaStream_t stream) {
    auto kernel = n_pcs <= 64    ? add_rows_normalize_kernel<T, 2>
                  : n_pcs <= 128 ? add_rows_normalize_kernel<T, 4>
                                 : add_rows_normalize_kernel<T, 0>;
    kernel<<<grid, 256, 0, stream>>>(X, Z, n_cells, n_pcs, normalize);
}
