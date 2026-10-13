#pragma once

#include <cuda_runtime.h>

// Spatial autocorrelation of a block of up to 32 genes (one per lane) under a
// batch of permutations. A permutation pi moves the rows of the weights W, so
// with z = W x and r = W 1 each permuted statistic needs only
//     A = sum_i x_i z_pi(i)    and    B = sum_i x_i^e r_pi(i)
// (e = 1 for Moran's I, 2 for Geary's C); the rest does not depend on pi.

constexpr int AUTOCORR_ROUNDS = 6;
constexpr int AUTOCORR_BATCH = 16;

__device__ __forceinline__ unsigned int autocorr_fmix32(unsigned int h) {
    h ^= h >> 16;
    h *= 0x85EBCA6Bu;
    h ^= h >> 13;
    h *= 0xC2B2AE35u;
    return h ^ (h >> 16);
}

// A bijection of [0, size), size >= 2: a Feistel network on the bits covering
// size (halves of unequal width when odd; widths swap every round, an even
// number of rounds restores them), cycle-walked into range.
__device__ __forceinline__ unsigned int autocorr_permute(
    unsigned int x, unsigned int size, const unsigned int* keys) {
    const int bits = 32 - __clz(size - 1);
    const int w_right = bits / 2;
    do {
        int wl = bits - w_right;
        int wr = w_right;
        unsigned int l = x >> wr;
        unsigned int r = x & ((1u << wr) - 1u);
        for (int round = 0; round < AUTOCORR_ROUNDS; ++round) {
            const unsigned int f =
                autocorr_fmix32(r ^ keys[round]) & ((1u << wl) - 1u);
            const unsigned int next = l ^ f;
            l = r;
            r = next;
            const int w = wl;
            wl = wr;
            wr = w;
        }
        x = (l << wr) | r;
    } while (x >= size);
    return x;
}

__global__ void autocorr_permutation_kernel(const unsigned int* keys, int n,
                                            int* out) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) out[i] = autocorr_permute(i, n, keys);
}

// out[p, 0, g] += A, out[p, 1, g] += B for the permutations [0, n_perms) with
// round keys keys[p * AUTOCORR_ROUNDS ...]; n_perms == 0 means the identity.
// x and z are (n, n_genes) row-major. A warp handles one cell at a time; rows
// without expression in the block are skipped.
// With UNIFORM row sums, B is left to the caller (it does not depend on pi).
template <typename T, bool GEARY, bool UNIFORM>
__global__ void autocorr_sums_kernel(const T* __restrict__ x,
                                     const T* __restrict__ z,
                                     const T* __restrict__ r, int n,
                                     int n_genes, const unsigned int* keys,
                                     int n_perms, double* __restrict__ out) {
    __shared__ double block_sums[AUTOCORR_BATCH * 2 * 32];
    for (int t = threadIdx.x; t < AUTOCORR_BATCH * 2 * 32; t += blockDim.x)
        block_sums[t] = 0.0;
    __syncthreads();
    const int lane = threadIdx.x & 31;
    const int batch = n_perms ? n_perms : 1;
    double a[AUTOCORR_BATCH], b[AUTOCORR_BATCH];
#pragma unroll
    for (int p = 0; p < AUTOCORR_BATCH; ++p) a[p] = b[p] = 0.0;
    const int warps = (gridDim.x * blockDim.x) >> 5;
    for (int i = (blockIdx.x * blockDim.x + threadIdx.x) >> 5; i < n;
         i += warps) {
        const T xv = lane < n_genes ? x[(long long)i * n_genes + lane] : T(0);
        if (!__any_sync(0xFFFFFFFFu, xv != T(0))) continue;
        // lane p finds permutation p's source row, shared by every lane
        const int mine =
            !n_perms ? i
            : lane < n_perms
                ? (int)autocorr_permute(i, n, keys + lane * AUTOCORR_ROUNDS)
                : 0;
#pragma unroll
        for (int p = 0; p < AUTOCORR_BATCH; ++p) {
            const int j = __shfl_sync(0xFFFFFFFFu, mine, p);
            if (p < batch && xv != T(0)) {
                a[p] += (double)xv * (double)z[(long long)j * n_genes + lane];
                if constexpr (!UNIFORM) {
                    b[p] += (double)(GEARY ? xv * xv : xv) * (double)r[j];
                }
            }
        }
    }
#pragma unroll
    for (int p = 0; p < AUTOCORR_BATCH; ++p) {
        if (p < batch && lane < n_genes) {
            atomicAdd(&block_sums[(2 * p) * 32 + lane], a[p]);
            atomicAdd(&block_sums[(2 * p + 1) * 32 + lane], b[p]);
        }
    }
    __syncthreads();
    for (int t = threadIdx.x; t < batch * 2 * 32; t += blockDim.x) {
        const int lane_t = t % 32;
        if (lane_t < n_genes)
            atomicAdd(&out[(long long)(t / 32) * n_genes + lane_t],
                      block_sums[t]);
    }
}

// z = W x for row-major x and z with up to 32 columns: a warp per row, lane g
// accumulating column g over the row's neighbours (coalesced row reads).
template <typename T, typename P>
__global__ void autocorr_lag_kernel(const P* __restrict__ indptr,
                                    const int* __restrict__ indices,
                                    const T* __restrict__ weights,
                                    const T* __restrict__ x, int n, int n_genes,
                                    T* __restrict__ z) {
    const int lane = threadIdx.x & 31;
    const int warps = (gridDim.x * blockDim.x) >> 5;
    for (int i = (blockIdx.x * blockDim.x + threadIdx.x) >> 5; i < n;
         i += warps) {
        T acc = 0;
        if (lane < n_genes)
            for (P e = indptr[i]; e < indptr[i + 1]; ++e)
                acc += weights[e] * x[(long long)indices[e] * n_genes + lane];
        if (lane < n_genes) z[(long long)i * n_genes + lane] = acc;
    }
}

// Per gene: sum x, sum x^2, sum c x, sum c x^2 (double), for row-major x with
// up to 32 columns; out is (4, n_genes).
template <typename T>
__global__ void autocorr_colstats_kernel(const T* __restrict__ x,
                                         const double* __restrict__ c, int n,
                                         int n_genes,
                                         double* __restrict__ out) {
    __shared__ double block_sums[4 * 32];
    for (int t = threadIdx.x; t < 4 * 32; t += blockDim.x) block_sums[t] = 0.0;
    __syncthreads();
    const int lane = threadIdx.x & 31;
    double s[4] = {0.0, 0.0, 0.0, 0.0};
    const int warps = (gridDim.x * blockDim.x) >> 5;
    if (lane < n_genes)
        for (int i = (blockIdx.x * blockDim.x + threadIdx.x) >> 5; i < n;
             i += warps) {
            const double v = x[(long long)i * n_genes + lane];
            s[0] += v;
            s[1] += v * v;
            s[2] += c[i] * v;
            s[3] += c[i] * v * v;
        }
    if (lane < n_genes)
        for (int t = 0; t < 4; ++t) atomicAdd(&block_sums[t * 32 + lane], s[t]);
    __syncthreads();
    for (int t = threadIdx.x; t < 4 * 32; t += blockDim.x)
        if (t % 32 < n_genes)
            atomicAdd(&out[(t / 32) * n_genes + t % 32], block_sums[t]);
}

// sum w, sum w^2 and sum_ij w_ij w_ji of a canonical CSR matrix (sorted,
// unique columns per row): each edge looks for its mirror in the target row.
template <typename T, typename P>
__global__ void autocorr_moments_kernel(const P* __restrict__ indptr,
                                        const int* __restrict__ indices,
                                        const T* __restrict__ data, int n,
                                        double* __restrict__ out) {
    double s[3] = {0.0, 0.0, 0.0};
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < n;
         i += gridDim.x * blockDim.x) {
        for (P e = indptr[i]; e < indptr[i + 1]; ++e) {
            const double w = data[e];
            const int j = indices[e];
            s[0] += w;
            s[1] += w * w;
            P lo = indptr[j], hi = indptr[j + 1];
            while (lo < hi) {  // first column >= i in row j
                const P mid = lo + (hi - lo) / 2;
                if (indices[mid] < i)
                    lo = mid + 1;
                else
                    hi = mid;
            }
            if (lo < indptr[j + 1] && indices[lo] == i) s[2] += w * data[lo];
        }
    }
    for (int t = 0; t < 3; ++t) {
        double v = s[t];
        for (int o = 16; o > 0; o >>= 1)
            v += __shfl_down_sync(0xFFFFFFFFu, v, o);
        if ((threadIdx.x & 31) == 0) atomicAdd(&out[t], v);
    }
}
