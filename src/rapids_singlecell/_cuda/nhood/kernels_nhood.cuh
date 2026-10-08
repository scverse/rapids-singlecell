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

// Random permutations within groups via a keyed Feistel bijection with cycle
// walking (the cipher of thrust::shuffle, applied per group).
static constexpr int FEISTEL_ROUNDS = 24;

__device__ __forceinline__ unsigned int mix32(unsigned int x) {
    x ^= x >> 16;
    x *= 0x85ebca6bu;
    x ^= x >> 13;
    x *= 0xc2b2ae35u;
    x ^= x >> 16;
    return x;
}

__device__ __forceinline__ unsigned int feistel(unsigned int val,
                                                const unsigned int* key,
                                                unsigned int salt,
                                                int left_bits, int right_bits) {
    const unsigned int left_mask = (1u << left_bits) - 1u;
    const unsigned int right_mask = (1u << right_bits) - 1u;
    unsigned int l = val >> right_bits;
    unsigned int r = val & right_mask;
    for (int i = 0; i < FEISTEL_ROUNDS; ++i) {
        const unsigned long long p = 0xD2B74407B1CE6E93ull * l;
        const unsigned int hi = (unsigned int)(p >> 32);
        unsigned int lo = (unsigned int)p;
        lo = (lo << (right_bits - left_bits)) | (r >> left_bits);
        l = ((hi ^ mix32(key[i] ^ salt)) ^ r) & left_mask;
        r = lo & right_mask;
    }
    return (l << right_bits) | r;
}

// out[b, pos[s]] = labels[pos[off[g] + perm_{b,g}(s - off[g])]] for every
// sorted index s in group g. `pos` lists cells grouped by library and `keys`
// holds FEISTEL_ROUNDS round keys per permutation.
__global__ void permute_labels_kernel(const int* __restrict__ labels,
                                      const int* __restrict__ pos,
                                      const int* __restrict__ group_off,
                                      int n_groups,
                                      const unsigned int* __restrict__ keys,
                                      int n_cells, int* __restrict__ out) {
    __shared__ unsigned int key[FEISTEL_ROUNDS];
    const int b = blockIdx.y;
    if (threadIdx.x < FEISTEL_ROUNDS) {
        key[threadIdx.x] = keys[b * FEISTEL_ROUNDS + threadIdx.x];
    }
    __syncthreads();

    int* o = out + (long long)b * n_cells;
    const int stride = blockDim.x * gridDim.x;
    for (int s = blockIdx.x * blockDim.x + threadIdx.x; s < n_cells;
         s += stride) {
        int lo = 0, hi = n_groups;  // last g with group_off[g] <= s
        while (hi - lo > 1) {
            const int mid = (lo + hi) / 2;
            if (group_off[mid] <= s)
                lo = mid;
            else
                hi = mid;
        }
        const int g = lo;
        const int start = group_off[g];
        const unsigned int m = (unsigned int)(group_off[g + 1] - start);
        unsigned int v = (unsigned int)(s - start);
        if (m > 1) {
            const int bits = m <= 16u ? 4 : 32 - __clz(m - 1u);
            const int left_bits = bits / 2;
            const int right_bits = bits - left_bits;
            const unsigned int salt = (unsigned int)g * 0x9E3779B9u;
            do {
                v = feistel(v, key, salt, left_bits, right_bits);
            } while (v >= m);
        }
        o[pos[s]] = labels[pos[start + (int)v]];
    }
}
