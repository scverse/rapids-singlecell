#pragma once

#include <cuda_runtime.h>

// Count cluster-pair edges for a batch of label vectors.
// blockIdx.y selects the label vector (one permutation); blockIdx.x strides
// over the edge list. `out` has shape (n_batch, k, k).
// With USE_SHARED, each block accumulates a private k*k histogram in shared
// memory and flushes it once; otherwise it adds straight into `out`.
template <bool USE_SHARED>
__global__ void nhood_count_kernel(const int* __restrict__ rows,
                                   const int* __restrict__ cols, long long nnz,
                                   const int* __restrict__ labels,
                                   long long n_cells, int k,
                                   unsigned long long* __restrict__ out) {
    extern __shared__ unsigned int hist[];
    const int kk = k * k;
    const int* lab = labels + (long long)blockIdx.y * n_cells;
    unsigned long long* o = out + (long long)blockIdx.y * kk;

    if constexpr (USE_SHARED) {
        for (int i = threadIdx.x; i < kk; i += blockDim.x) {
            hist[i] = 0u;
        }
        __syncthreads();
    }

    const long long stride = (long long)blockDim.x * gridDim.x;
    for (long long e = (long long)blockIdx.x * blockDim.x + threadIdx.x;
         e < nnz; e += stride) {
        const int bin =
            __ldg(lab + __ldg(rows + e)) * k + __ldg(lab + __ldg(cols + e));
        if constexpr (USE_SHARED) {
            atomicAdd(&hist[bin], 1u);
        } else {
            atomicAdd(&o[bin], 1ull);
        }
    }

    if constexpr (USE_SHARED) {
        __syncthreads();
        for (int i = threadIdx.x; i < kk; i += blockDim.x) {
            const unsigned int v = hist[i];
            if (v) {
                atomicAdd(&o[i], (unsigned long long)v);
            }
        }
    }
}

// Uniform random permutations within groups: sorting random keys inside each
// (permutation, group) segment orders that group uniformly at random. The
// random bits are the SplitMix64 stream of the permutation's seed.
__device__ __forceinline__ unsigned long long splitmix64(unsigned long long x) {
    x += 0x9E3779B97F4A7C15ull;
    x = (x ^ (x >> 30)) * 0xBF58476D1CE4E5B9ull;
    x = (x ^ (x >> 27)) * 0x94D049BB133111EBull;
    return x ^ (x >> 31);
}

// Group of sorted index s: the last g with group_off[g] <= s.
__device__ __forceinline__ int find_group(const int* group_off, int n_groups,
                                          int s) {
    int lo = 0, hi = n_groups;
    while (hi - lo > 1) {
        const int mid = (lo + hi) / 2;
        if (group_off[mid] <= s) {
            lo = mid;
        } else {
            hi = mid;
        }
    }
    return lo;
}

// Key of sorted index s in permutation b: segment (b, group) above 32 random
// bits. Ties occur with probability 2^-32 per pair and keep index order.
__global__ void shuffle_keys_kernel(
    const int* __restrict__ group_off, int n_groups,
    const unsigned long long* __restrict__ seeds, int n_cells,
    unsigned long long* __restrict__ keys, int* __restrict__ vals) {
    const int b = blockIdx.y;
    const unsigned long long seed = seeds[b];
    const int stride = blockDim.x * gridDim.x;
    for (int s = blockIdx.x * blockDim.x + threadIdx.x; s < n_cells;
         s += stride) {
        const unsigned long long seg = (unsigned long long)b * n_groups +
                                       find_group(group_off, n_groups, s);
        const unsigned long long rnd =
            splitmix64(seed + (unsigned long long)s * 0x9E3779B97F4A7C15ull);
        const long long i = (long long)b * n_cells + s;
        keys[i] = (seg << 32) | (rnd >> 32);
        vals[i] = s;
    }
}

// out[b, pos[s]] = labels[pos[order[b, s]]]: sorted index s of group g takes
// the label of the cell the shuffle moved there from the same group.
__global__ void scatter_labels_kernel(const int* __restrict__ labels,
                                      const int* __restrict__ pos,
                                      const int* __restrict__ order,
                                      long long total, int n_cells,
                                      int* __restrict__ out) {
    const long long stride = (long long)blockDim.x * gridDim.x;
    for (long long i = (long long)blockIdx.x * blockDim.x + threadIdx.x;
         i < total; i += stride) {
        const long long row = i / n_cells * n_cells;
        out[row + pos[i - row]] = labels[pos[order[i]]];
    }
}
