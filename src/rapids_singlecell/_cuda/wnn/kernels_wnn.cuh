#pragma once

#include <cuda_runtime.h>
#include <math_constants.h>
#include <cub/block/block_scan.cuh>
#include <cuda/std/utility>
#include <climits>

// Kernels for Seurat's weighted nearest neighbor (WNN) analysis
// (FindModalityWeights + MultiModalNN). All kNN arrays are int32, row-major
// (n_obs, k_max) with the cell itself in column 0.

__device__ __forceinline__ float warp_sum(float v) {
#pragma unroll
    for (int off = 16; off > 0; off >>= 1) {
        v += __shfl_xor_sync(0xffffffff, v, off);
    }
    return v;
}

// Warp per cell: distance between the cell's embedding and the mean embedding
// of its neighbors 1..k-1 (Seurat's PredictAssay, which drops the self column),
// minus the distance to the nearest neighbor, clipped at zero (impute_dist).
// `emb` is (n_obs, d); `knn` is the neighbor list of any modality.
__global__ void wnn_impute_dist_kernel(const float* __restrict__ emb, int d,
                                       const int* __restrict__ knn, int k_max,
                                       int k, const float* __restrict__ nearest,
                                       float* __restrict__ out, int n_obs) {
    const int lane = threadIdx.x & 31;
    const long long warp =
        ((long long)blockIdx.x * blockDim.x + threadIdx.x) >> 5;
    const long long n_warps = ((long long)gridDim.x * blockDim.x) >> 5;
    const float inv = 1.0f / (float)(k - 1);
    for (long long i = warp; i < n_obs; i += n_warps) {
        const int* nn = knn + i * k_max;
        const float* xi = emb + i * d;
        float acc = 0.0f;
        for (int f = lane; f < d; f += 32) {
            float s = 0.0f;
            for (int c = 1; c < k; ++c) s += emb[(long long)nn[c] * d + f];
            const float diff = xi[f] - s * inv;
            acc += diff * diff;
        }
        acc = warp_sum(acc);
        if (lane == 0) out[i] = fmaxf(sqrtf(acc) - nearest[i], 0.0f);
    }
}

// Shared nearest neighbor (SNN) kernels, block of WNN_SNN_THREADS per row.
// Row i's SNN partners are the reverse neighbor lists (postings) of its first
// `s` neighbors; a partner sharing c of them appears c times, so once sorted,
// equal partners form runs whose length is Seurat's ComputeSNN overlap. Rows
// whose partners fit are gathered and sorted in shared memory; the rest come
// presorted from a global sort.
constexpr int WNN_SNN_THREADS = 128;
using SnnScan = cub::BlockScan<int, WNN_SNN_THREADS>;

// Thread per (cell, neighbor slot) of the first `s` neighbors: counts the
// cell in cursor[neighbor] or, with `postings`, appends it to the neighbor's
// list at cursor[neighbor]. The order within a list is arbitrary.
__global__ void wnn_postings_kernel(const int* __restrict__ knn, int k_max,
                                    int s, long long n_obs,
                                    unsigned long long* __restrict__ cursor,
                                    int* __restrict__ postings) {
    const long long stride = (long long)gridDim.x * blockDim.x;
    for (long long p = (long long)blockIdx.x * blockDim.x + threadIdx.x;
         p < n_obs * s; p += stride) {
        const long long j = p / s;
        const unsigned long long at =
            atomicAdd(&cursor[knn[j * k_max + (p - j * s)]], 1ull);
        if (postings) postings[at] = (int)j;
    }
}

// Ascending bitonic sort of P (power of two) entries of the arrays `a`, all
// permuted the same way; `less(x, y)` compares positions x and y.
template <typename Less, typename... T>
__device__ __forceinline__ void block_bitonic_sort(int P, Less less, T*... a) {
    for (int size = 2; size <= P; size <<= 1) {
        for (int stride = size >> 1; stride > 0; stride >>= 1) {
            for (int t = threadIdx.x; t < (P >> 1); t += blockDim.x) {
                const int lo = 2 * t - (t & (stride - 1));
                const int hi = lo + stride;
                const bool up = (lo & size) == 0;
                if (less(hi, lo) == up) (cuda::std::swap(a[lo], a[hi]), ...);
            }
            __syncthreads();
        }
    }
}

// Writes row i's L SNN partners, OR'ed with `tag`, to out[0, L) in arbitrary
// order. Returns L.
template <typename T>
__device__ __forceinline__ int snn_gather(
    const int* __restrict__ knn, int k_max, int s, long long i,
    const long long* __restrict__ post_off, const int* __restrict__ postings,
    T tag, T* out) {
    __shared__ int s_len;
    if (threadIdx.x == 0) s_len = 0;
    __syncthreads();
    const int* nn = knn + i * k_max;
    const int lane = threadIdx.x & 31;
    for (int t = threadIdx.x >> 5; t < s; t += WNN_SNN_THREADS / 32) {
        const long long beg = post_off[nn[t]];
        const int len = (int)(post_off[nn[t] + 1] - beg);
        int at = 0;
        if (lane == 0) at = atomicAdd(&s_len, len);
        at = __shfl_sync(0xffffffffu, at, 0);
        for (int q = lane; q < len; q += 32)
            out[at + q] = tag | (T)(unsigned int)postings[beg + q];
    }
    __syncthreads();
    return s_len;
}

// Block per listed row r: its partners, tagged r << 32, to keys[seg[r], ...)
// for a global sort (rows too long for shared memory).
__global__ void wnn_snn_emit_kernel(
    const int* __restrict__ rows, const int* __restrict__ knn, int k_max, int s,
    const long long* __restrict__ post_off, const int* __restrict__ postings,
    const long long* __restrict__ seg, unsigned long long* __restrict__ keys) {
    snn_gather(knn, k_max, s, (long long)rows[blockIdx.x], post_off, postings,
               (unsigned long long)blockIdx.x << 32, keys + seg[blockIdx.x]);
}

// Gathers row i's SNN partners into keys[0, L), pads them with UINT_MAX to
// the next power of two (keys must have room) and sorts them. Returns L.
__device__ __forceinline__ int snn_gather_sorted(
    const int* __restrict__ knn, int k_max, int s, long long i,
    const long long* __restrict__ post_off, const int* __restrict__ postings,
    unsigned int* keys) {
    const int L = snn_gather(knn, k_max, s, i, post_off, postings, 0u, keys);
    int P = 1;
    while (P < L) P <<= 1;
    for (int u = L + threadIdx.x; u < P; u += WNN_SNN_THREADS)
        keys[u] = UINT_MAX;
    __syncthreads();
    block_bitonic_sort(
        P, [&](int x, int y) { return keys[x] < keys[y]; }, keys);
    return L;
}

// Length of the run of equal keys starting at p < L, 0 inside a run.
__device__ __forceinline__ int run_length(const unsigned int* keys, int L,
                                          int p) {
    if (p > 0 && keys[p - 1] == keys[p]) return 0;
    int q = p + 1;
    while (q < L && keys[q] == keys[p]) ++q;
    return q - p;
}

// Adds one to hist[bin] for every lane with bin >= 0, one atomic per
// distinct bin of the warp (most lanes of a warp share a bin here).
template <typename T>
__device__ __forceinline__ void warp_hist_add(T* hist, int bin) {
    const unsigned int peers = __match_any_sync(0xffffffffu, bin);
    if (bin >= 0 && (threadIdx.x & 31) == __ffs(peers) - 1)
        atomicAdd(&hist[bin], (T)__popc(peers));
}

// Block per listed row i: Seurat's ComputeSNNwidth (SNN_SmallestNonzero_Dist).
// Among the partners whose overlap is <= the n-th smallest (n = min(#partners,
// s), ties included), sigma[i] averages the n largest (nearest neighbor
// corrected) distances. Without SORTED, the row's L partners are gathered and
// sorted in shared memory (P keys, then P distances); with SORTED, row r's
// sorted partners are sorted[seg[r], seg[r + 1]) and `dist` is scratch of the
// same size. Thread t handles positions t, t + WNN_SNN_THREADS, ... and the
// final sum is reduced in a fixed order, so bandwidths are deterministic.
// Dynamic shared memory: s + 1 ints (histogram), plus 2P words without SORTED.
template <bool SORTED>
__global__ void wnn_snn_bandwidth_kernel(
    const int* __restrict__ rows, const int* __restrict__ knn, int k_max, int s,
    const long long* __restrict__ post_off, const int* __restrict__ postings,
    int P, unsigned int* sorted, const long long* __restrict__ seg, float* dist,
    const float* __restrict__ emb, int d, const float* __restrict__ nearest,
    float* __restrict__ sigma) {
    extern __shared__ __align__(16) unsigned int snn_smem[];
    __shared__ SnnScan::TempStorage scan;
    static_assert(WNN_SNN_THREADS == 128, "2 radix digits per thread");
    __shared__ unsigned int digit_hist[256];
    __shared__ int s_thresh, s_krem;
    __shared__ unsigned int s_prefix;
    __shared__ float warp_sums[WNN_SNN_THREADS / 32];
    const long long i = rows[blockIdx.x];
    int* hist = reinterpret_cast<int*>(snn_smem);
    unsigned int* keys = SORTED ? sorted + seg[blockIdx.x] : snn_smem + s + 1;
    const int L =
        SORTED ? (int)(seg[blockIdx.x + 1] - seg[blockIdx.x])
               : snn_gather_sorted(knn, k_max, s, i, post_off, postings, keys);
    dist = SORTED ? dist + seg[blockIdx.x] : reinterpret_cast<float*>(keys + P);

    for (int t = threadIdx.x; t <= s; t += WNN_SNN_THREADS) hist[t] = 0;
    __syncthreads();
    // Pass 1: histogram of the overlaps (clipped at s) of unique partners.
    // The loops over positions run equally often in every lane of a warp.
    for (int base = 0; base < L; base += WNN_SNN_THREADS) {
        const int p = base + threadIdx.x;
        const int run = p < L ? run_length(keys, L, p) : 0;
        warp_hist_add(hist, run ? min(run, s) : -1);
    }
    __syncthreads();
    if (threadIdx.x == 0) {
        int n = 0;
        for (int c = 1; c <= s; ++c) n += hist[c];
        n = min(n, s);
        // The smallest overlap whose cumulative count reaches n (<= s).
        int thresh = 1, cum = hist[1];
        while (cum < n) cum += hist[++thresh];
        s_thresh = thresh;
        s_krem = n;
        s_prefix = 0u;
    }
    __syncthreads();
    const int thresh = s_thresh;
    const int n = s_krem;  // rewritten only after the radix loop's first sync

    // Pass 2: distances to partners with overlap <= threshold (-1 elsewhere).
    const float* xi = emb + i * d;
    const float near = nearest[i];
    for (int p = threadIdx.x; p < L; p += WNN_SNN_THREADS) {
        const int run = run_length(keys, L, p);
        float out = -1.0f;
        if (run > 0 && run <= thresh) {
            const float* xj = emb + (long long)keys[p] * d;
            float acc = 0.0f;
            for (int f = 0; f < d; ++f) {
                const float diff = xi[f] - xj[f];
                acc += diff * diff;
            }
            out = sqrtf(acc);
            if (near > 0.0f) out = fmaxf(out - near, 0.0f);
        }
        dist[p] = out;
    }

    // Radix select the n-th largest distance (non-negative float bits sort
    // like unsigned ints), 8 bits per pass.
    unsigned int mask = 0u;
    for (int shift = 24; shift >= 0; shift -= 8) {
        for (int t = threadIdx.x; t < 256; t += WNN_SNN_THREADS)
            digit_hist[t] = 0u;
        __syncthreads();
        const unsigned int prefix = s_prefix;
        const int krem = s_krem;
        for (int base = 0; base < L; base += WNN_SNN_THREADS) {
            const int p = base + threadIdx.x;
            int dg = -1;
            if (p < L) {
                const float v = dist[p];
                const unsigned int u = __float_as_uint(v);
                if (v >= 0.0f && (u & mask) == prefix)
                    dg = (int)((u >> shift) & 255u);
            }
            warp_hist_add(digit_hist, dg);
        }
        __syncthreads();
        // The largest digit whose values, with those of larger digits, reach
        // the remaining count (digit 0 if none of 255..1 does). Thread t owns
        // digits 255 - 2t and 254 - 2t.
        const int hi = 255 - 2 * (int)threadIdx.x;
        const int c_hi = (int)digit_hist[hi];
        const int c_lo = (int)digit_hist[hi - 1];
        int above;
        SnnScan(scan).ExclusiveSum(c_hi + c_lo, above);
        if (above < krem && (krem <= above + c_hi + c_lo || hi == 1)) {
            const bool top = krem <= above + c_hi;
            s_prefix = prefix | ((unsigned int)(top ? hi : hi - 1) << shift);
            s_krem = top ? krem - above : krem - above - c_hi;
        }
        mask |= 255u << shift;
        __syncthreads();
    }
    const unsigned int t_bits = s_prefix;

    // Sum of the distances strictly above the n-th largest one.
    float local = 0.0f;
    for (int p = threadIdx.x; p < L; p += WNN_SNN_THREADS) {
        const float v = dist[p];
        if (v >= 0.0f && __float_as_uint(v) > t_bits) local += v;
    }
    local = warp_sum(local);
    if ((threadIdx.x & 31) == 0) warp_sums[threadIdx.x >> 5] = local;
    __syncthreads();
    if (threadIdx.x == 0) {
        float total = (float)s_krem * __uint_as_float(t_bits);
        for (int w = 0; w < WNN_SNN_THREADS / 32; ++w) total += warp_sums[w];
        sigma[i] = n > 0 ? total / (float)n : 0.0f;
    }
}

// Block per listed row i: Seurat's ComputeSNN, partners as in the bandwidth
// kernel. The Jaccard index c / (2s - c) of every partner with overlap c,
// pruned below `prune`. Without `fill`, writes the number of kept partners to
// indptr[i + 1]; with `fill`, writes them in partner order to
// cols / vals[indptr[i], ...). Dynamic shared memory: P words without SORTED.
template <bool SORTED>
__global__ void wnn_snn_graph_kernel(
    const int* __restrict__ rows, const int* __restrict__ knn, int k_max, int s,
    const long long* __restrict__ post_off, const int* __restrict__ postings,
    unsigned int* sorted, const long long* __restrict__ seg, float prune,
    bool fill, long long* __restrict__ indptr, int* __restrict__ cols,
    float* __restrict__ vals) {
    extern __shared__ __align__(16) unsigned int snn_smem[];
    __shared__ SnnScan::TempStorage scan;
    const long long i = rows[blockIdx.x];
    const unsigned int* keys = SORTED ? sorted + seg[blockIdx.x] : snn_smem;
    const int L = SORTED ? (int)(seg[blockIdx.x + 1] - seg[blockIdx.x])
                         : snn_gather_sorted(knn, k_max, s, i, post_off,
                                             postings, snn_smem);
    const int per = (L + WNN_SNN_THREADS - 1) / WNN_SNN_THREADS;
    const int lo = (int)threadIdx.x * per;
    const int hi = min(lo + per, L);
    // Jaccard index of the partner whose run starts at p, NaN elsewhere.
    auto jaccard = [&](int p) {
        const float shared = (float)run_length(keys, L, p);
        return shared > 0.0f ? shared / ((float)(2 * s) - shared)
                             : CUDART_NAN_F;
    };
    int n_keep = 0;
    for (int p = lo; p < hi; ++p) n_keep += jaccard(p) >= prune;
    int off, total;
    SnnScan(scan).ExclusiveSum(n_keep, off, total);
    if (!fill) {
        if (threadIdx.x == 0) indptr[i + 1] = total;
        return;
    }
    long long out = indptr[i] + off;
    for (int p = lo; p < hi; ++p) {
        const float val = jaccard(p);
        if (val >= prune) {
            cols[out] = (int)keys[p];
            vals[out++] = val;
        }
    }
}

// Block per cell: Seurat's MultiModalNN. Candidates are the union of the
// cell's `knn_range - 1` non-self neighbors in every modality (first
// occurrence wins, as in R's union). Seurat scores each candidate by the
// modality-weighted sum s of exp(-relu(dist - nearest) / sigma); as the
// weights sum to one, we compute 1 - s = sum(w * -expm1(-x)) directly, which
// avoids float32 cancellation for close neighbors. The `k_out` best are
// written in increasing 1 - s order (ties by union order).
// `emb` concatenates the L2-normalized embeddings (n_obs, dim_off[M]); `knn`,
// `weight`, `sigma` and `nearest` are stacked per modality.
// Dynamic shared memory: P ints (candidates) + P ints (positions) + P floats.
__global__ void wnn_multimodal_knn_kernel(
    const float* __restrict__ emb, int D, const int* __restrict__ dim_off,
    const int* __restrict__ knn, int k_max, int knn_range, int n_mod,
    const float* __restrict__ weight, const float* __restrict__ sigma,
    const float* __restrict__ nearest, int n_obs, int P, int k_out,
    int* __restrict__ out_idx, float* __restrict__ out_dissim) {
    extern __shared__ unsigned char smem[];
    int* cand = reinterpret_cast<int*>(smem);
    int* pos = cand + P;
    float* dissim = reinterpret_cast<float*>(pos + P);

    const long long i = blockIdx.x;
    const int per = knn_range - 1;
    const int L = n_mod * per;

    for (int t = threadIdx.x; t < P; t += blockDim.x) {
        const int m = t / per, c = t - m * per + 1;  // column c of modality m
        cand[t] = t < L ? knn[((long long)m * n_obs + i) * k_max + c] : INT_MAX;
        pos[t] = t;
    }
    __syncthreads();

    // Group duplicates; the first occurrence (smallest position) leads.
    // dissim is not permuted here: the scoring loop below sets all P.
    block_bitonic_sort(
        P,
        [&](int x, int y) {
            return cand[x] < cand[y] || (cand[x] == cand[y] && pos[x] < pos[y]);
        },
        cand, pos);

    // Warp per unique candidate: weighted kernel similarity over modalities.
    const int lane = threadIdx.x & 31;
    const int warp = threadIdx.x >> 5;
    const int n_warps = blockDim.x >> 5;
    const float* xi = emb + i * D;
    for (int t = warp; t < P; t += n_warps) {
        const int j = cand[t];
        const bool lead = j != INT_MAX && (t == 0 || cand[t - 1] != j);
        if (!lead) {
            if (lane == 0) dissim[t] = CUDART_INF_F;
            continue;
        }
        const float* xj = emb + (long long)j * D;
        float s = 0.0f;
        for (int m = 0; m < n_mod; ++m) {
            float acc = 0.0f;
            for (int f = dim_off[m] + lane; f < dim_off[m + 1]; f += 32) {
                const float diff = xi[f] - xj[f];
                acc += diff * diff;
            }
            acc = warp_sum(acc);
            const long long mi = (long long)m * n_obs + i;
            const float dist = fmaxf(sqrtf(acc) - nearest[mi], 0.0f);
            s -= expm1f(-dist / sigma[mi]) * weight[mi];
        }
        if (lane == 0) dissim[t] = s;
    }
    __syncthreads();

    // Increasing 1 - s, ties broken by union order (R's stable order()).
    block_bitonic_sort(
        P,
        [&](int x, int y) {
            return dissim[x] < dissim[y] ||
                   (dissim[x] == dissim[y] && pos[x] < pos[y]);
        },
        cand, pos, dissim);

    for (int t = threadIdx.x; t < k_out; t += blockDim.x) {
        out_idx[i * k_out + t] = cand[t];
        out_dissim[i * k_out + t] = dissim[t];
    }
}
