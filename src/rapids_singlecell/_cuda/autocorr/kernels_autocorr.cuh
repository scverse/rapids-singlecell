#pragma once

#include <cuda_runtime.h>
#include <cub/block/block_reduce.cuh>

// Permuting the rows of W permutes its spatial lag z = W x and row sums r.
// With x translated by a per-gene offset, the change from the observed score
// needs only sum_i x_i (z_pi(i) - z_i) and sum_i x_i^e (r_pi(i) - r_i),
// where e is one for Moran and two for Geary.

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

// Each grid row scores up to BATCH permutations with their own round keys.
template <typename T, bool GEARY, bool UNIFORM>
__global__ void autocorr_sums_kernel(const T* __restrict__ x,
                                     const T* __restrict__ z,
                                     const double* __restrict__ r, int n,
                                     int n_genes, const unsigned int* keys,
                                     int n_scores, int start,
                                     double* __restrict__ out) {
    constexpr int terms = UNIFORM ? 1 : 2;
    constexpr int stride = terms * 32;
    __shared__ double block_sums[AUTOCORR_BATCH * stride];
    const int first = start + blockIdx.y * AUTOCORR_BATCH;
    const int batch = min(n_scores - first, AUTOCORR_BATCH);
    for (int t = threadIdx.x; t < batch * stride; t += blockDim.x)
        block_sums[t] = 0.0;
    __syncthreads();
    const int lane = threadIdx.x & 31;
    const double offset = lane < n_genes ? x[lane] : 0;
    double a[AUTOCORR_BATCH] = {}, b[AUTOCORR_BATCH] = {};
    const int warps = (gridDim.x * blockDim.x) >> 5;
    for (long long i = (blockIdx.x * blockDim.x + threadIdx.x) >> 5; i < n;
         i += warps) {
        const double xv = lane < n_genes ? x[i * n_genes + lane] - offset : 0;
        if (!__any_sync(0xFFFFFFFFu, xv != 0)) continue;
        const double zi = xv != 0 ? z[i * n_genes + lane] : 0;
        const double ri = UNIFORM ? 0 : r[i];
        // Lane p computes a source row shared by every gene in the warp.
        const long long score = (long long)first + lane;
        int mine = i;
        if (lane < batch)
            mine = autocorr_permute(i, n, keys + score * AUTOCORR_ROUNDS);
#pragma unroll
        for (int p = 0; p < AUTOCORR_BATCH; ++p) {
            const int j = __shfl_sync(0xFFFFFFFFu, mine, p);
            if (p < batch && xv != 0) {
                a[p] += xv * (z[(long long)j * n_genes + lane] - zi);
                if constexpr (!UNIFORM)
                    b[p] += (GEARY ? xv * xv : xv) * (r[j] - ri);
            }
        }
    }
#pragma unroll
    for (int p = 0; p < AUTOCORR_BATCH; ++p) {
        if (p < batch && lane < n_genes) {
            atomicAdd(&block_sums[p * stride + lane], a[p]);
            if constexpr (!UNIFORM)
                atomicAdd(&block_sums[p * stride + 32 + lane], b[p]);
        }
    }
    __syncthreads();
    for (int t = threadIdx.x; t < batch * stride; t += blockDim.x) {
        const int g = t % 32;
        if (g < n_genes)
            atomicAdd(&out[((long long)first * terms + t / 32) * n_genes + g],
                      block_sums[t]);
    }
}

template <typename T, typename P>
__global__ void autocorr_normalize_kernel(const P* indptr, const T* weights,
                                          int n, T* normalized) {
    for (long long i = blockIdx.x * blockDim.x + threadIdx.x; i < n;
         i += gridDim.x * blockDim.x) {
        const P begin = indptr[i], end = indptr[i + 1];
        double sum = 0;
        for (P e = begin; e < end; ++e) sum += weights[e];
        const double scale = sum == 0 ? 0 : 1.0 / sum;
        for (P e = begin; e < end; ++e) normalized[e] = weights[e] * scale;
    }
}

// One warp per cell, one lane per gene. Compute the observed numerator
// directly from the edges. Only permutation tests need a stored spatial lag.
template <typename T, typename P, bool GEARY, bool SAVE_LAG>
__global__ void autocorr_observed_kernel(
    const P* __restrict__ indptr, const int* __restrict__ indices,
    const T* __restrict__ weights, const T* __restrict__ x,
    const double* __restrict__ row, const double* __restrict__ col, int n,
    int n_genes, T* __restrict__ z, double* __restrict__ stats) {
    const int lane = threadIdx.x & 31;
    const int warps = (gridDim.x * blockDim.x) >> 5;
    double s[4] = {};
    if (lane < n_genes) {
        const double offset = x[lane];
        for (long long i = (blockIdx.x * blockDim.x + threadIdx.x) >> 5; i < n;
             i += warps) {
            const double v = x[i * n_genes + lane] - offset;
            double lag = 0;
            for (P e = indptr[i]; e < indptr[i + 1]; ++e) {
                const double other =
                    x[(long long)indices[e] * n_genes + lane] - offset;
                const double w = weights[e];
                if constexpr (SAVE_LAG || !GEARY) lag += w * other;
                if constexpr (GEARY) s[3] += w * ((v - other) * (v - other));
            }
            if constexpr (SAVE_LAG) z[i * n_genes + lane] = lag;
            s[0] += v;
            s[1] += v * v;
            if constexpr (!GEARY) {
                s[2] += (row[i] + col[i]) * v;
                s[3] += v * lag;
            }
        }
    }
    __shared__ double block_sums[4 * 32];
    if (threadIdx.x < 4 * 32) block_sums[threadIdx.x] = 0.0;
    __syncthreads();
    if (lane < n_genes)
        for (int t = 0; t < 4; ++t) atomicAdd(&block_sums[t * 32 + lane], s[t]);
    __syncthreads();
    if (threadIdx.x < 4 * 32 && lane < n_genes)
        atomicAdd(&stats[(threadIdx.x / 32) * n_genes + lane],
                  block_sums[threadIdx.x]);
}

// Both observed statistics depend only on W + W^T. Visit the edges incident
// to this gene's stored cells: a stored-stored edge is visited from both ends,
// a stored-zero edge once. Zero-zero edges contribute no squared difference.
constexpr int AUTOCORR_SPARSE_BLOCK = 128;

template <typename T>
__global__ void autocorr_sparse_observed_kernel(
    const int* colptr, const int* rows, const T* values, const int* rowptr,
    const int* neighbors, const double* weights, const double* degree, int n,
    double total, bool geary, double* out) {
    using Reduce = cub::BlockReduce<double, AUTOCORR_SPARSE_BLOCK>;
    __shared__ typename Reduce::TempStorage scratch;
    __shared__ double mean;
    auto sum = [&](double value) {
        __syncthreads();
        return Reduce(scratch).Sum(value);
    };
    const int g = blockIdx.x, begin = colptr[g], end = colptr[g + 1];
    const int missing = n - (end - begin);
    const double offset = missing ? 0.0 : values[begin];
    double sx = 0.0;
    for (long long k = (long long)begin + threadIdx.x; k < end; k += blockDim.x)
        sx += values[k] - offset;
    sx = sum(sx);
    if (threadIdx.x == 0) mean = sx / n;
    __syncthreads();

    double ss = 0.0, weighted_ss = 0.0, stored_degree = 0.0, diff2 = 0.0;
    for (long long k = (long long)begin + threadIdx.x; k < end;
         k += blockDim.x) {
        const int i = rows[k];
        const double v = values[k], centered = (v - offset) - mean;
        ss += centered * centered;
        if (!geary) {
            weighted_ss += degree[i] * centered * centered;
            stored_degree += degree[i];
        }
        for (int edge = rowptr[i]; edge < rowptr[i + 1]; ++edge) {
            const int j = neighbors[edge];
            int lo = begin, hi = end;
            while (lo < hi) {
                const int mid = lo + (hi - lo) / 2;
                if (rows[mid] < j)
                    lo = mid + 1;
                else
                    hi = mid;
            }
            const bool stored = lo < end && rows[lo] == j;
            const double d = stored ? v - values[lo] : v;
            diff2 += weights[edge] * (stored ? 0.5 : 1.0) * (d * d);
        }
    }
    ss = sum(ss) + missing * mean * mean;
    diff2 = sum(diff2);
    if (!geary) {
        weighted_ss = sum(weighted_ss);
        stored_degree = sum(stored_degree);
        if (missing) weighted_ss += (2 * total - stored_degree) * mean * mean;
    }
    if (threadIdx.x == 0)
        out[g] = geary ? (n - 1) * diff2 / (2 * total * ss)
                       : (weighted_ss - diff2) / (2 * ss);
}

// sum w, sum w^2 and sum_ij w_ij w_ji of a canonical CSR matrix (sorted,
// unique columns per row): each edge looks for its mirror in the target row.
template <typename T, typename P>
__global__ void autocorr_moments_kernel(const P* __restrict__ indptr,
                                        const int* __restrict__ indices,
                                        const T* __restrict__ data, int n,
                                        double* __restrict__ row,
                                        double* __restrict__ col,
                                        double* __restrict__ out) {
    double s[3] = {0.0, 0.0, 0.0};
    for (long long i = blockIdx.x * blockDim.x + threadIdx.x; i < n;
         i += gridDim.x * blockDim.x) {
        double sum = 0.0;
        for (P e = indptr[i]; e < indptr[i + 1]; ++e) {
            const double w = data[e];
            const int j = indices[e];
            sum += w;
            atomicAdd(&col[j], w);
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
        row[i] = sum;
        s[0] += sum;
    }
    for (int t = 0; t < 3; ++t) {
        double v = s[t];
        for (int o = 16; o > 0; o >>= 1)
            v += __shfl_down_sync(0xFFFFFFFFu, v, o);
        if ((threadIdx.x & 31) == 0) atomicAdd(&out[t], v);
    }
}
