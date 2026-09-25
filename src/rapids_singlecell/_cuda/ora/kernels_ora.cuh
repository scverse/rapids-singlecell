#pragma once

#include <cuda_runtime.h>
#include <cmath>

namespace {
constexpr unsigned MASK = 0xffffffff;
constexpr int BLOCK_SIZE = 128;
constexpr int WARPS_PER_BLOCK = BLOCK_SIZE / 32;

// Packed CPU factors preserve decoupler's probability ties. Other tests
// evaluate log-gamma on the GPU. Wide offsets avoid overflowing packed indices.
template <bool compat>
struct LogFactorial {
    const double* gamma;
    double background;
    long long tail_offset;
    __device__ double operator()(long long x, bool tail = false) const {
        if constexpr (compat) {
            return gamma[tail ? (long long)floor(background) - x + tail_offset
                              : x];
        }
        return lgamma((tail ? background - x : (double)x) + 1);
    }
    __device__ double denominator(int a, int size, int count) const {
        // Explicit rounding retains the CPU's left-associative additions.
        return __dadd_rn(__dadd_rn(__dadd_rn((*this)(a), (*this)(size - a)),
                                   (*this)(count - a)),
                         (*this)((long long)size + count - a, true));
    }
};

// Every lane participates; only lane zero receives the complete FP64 tail.
template <bool compat>
__device__ double fisher_warp(int a, int count, int size, int alternative,
                              LogFactorial<compat> g) {
    const int lane = threadIdx.x % 32;
    const double lower = max(0.0, (double)size + count - g.background);
    // Numba truncates the descending range's stop for fractional backgrounds.
    int lo = lower > 0 ? (int)(lower - 1) + 1 : 0;
    int hi = min(size, count);
    if (g.background == floor(g.background) &&
        (lo == hi || (alternative == 1 && a <= lo) ||
         (alternative == 2 && a >= hi)))
        return 1.0;
    if (alternative == 1) lo = max(lo, a);
    if (alternative == 2) hi = min(hi, a);
    double observed = 0, normalizer = 0;
    if (lane == 0) {
        observed = g.denominator(a, size, count);
        normalizer = __dsub_rn(
            __dadd_rn(__dadd_rn(__dadd_rn(g(size), g(count)), g(count, true)),
                      g(size, true)),
            g(0, true));
    }
    observed = __shfl_sync(MASK, observed, 0);
    normalizer = __shfl_sync(MASK, normalizer, 0);
    double total = 0;
    // Unsigned loop positions allow the final increment past INT_MAX.
    for (unsigned k = (unsigned)lo + lane; k <= (unsigned)hi; k += 32) {
        const double denominator = g.denominator((int)k, size, count);
        if (alternative || denominator >= observed - (compat ? 0.0 : 1e-7)) {
            total +=
                exp(__dsub_rn(compat ? observed : normalizer, denominator));
        }
    }
    for (int delta = 16; delta > 0; delta /= 2) {
        total += __shfl_down_sync(MASK, total, delta);
    }
    if constexpr (!compat) return min(1.0, total);
    return exp(
        -max(0.0, __dsub_rn(__dsub_rn(observed, normalizer), log(total))));
}

__global__ void fisher_kernel(const int* a, const int* counts, const int* sizes,
                              size_t n, size_t ncols, bool shared_count,
                              bool shared_size, double background,
                              int alternative, const double* gamma,
                              long long tail_offset, double* pv) {
    const size_t stride = (size_t)gridDim.x * WARPS_PER_BLOCK;
    for (size_t i = (size_t)blockIdx.x * WARPS_PER_BLOCK + threadIdx.x / 32;
         i < n; i += stride) {
        const int count = counts[shared_count ? 0 : i % ncols];
        const int size = sizes[shared_size ? 0 : i / ncols];
        double p = 0;
        if (a[i] >= 0 && a[i] <= min(count, size) &&
            background - size - (count - a[i]) >= 0) {
            p = gamma ? fisher_warp<true>(a[i], count, size, 0,
                                          {gamma, background, tail_offset})
                      : fisher_warp<false>(a[i], count, size, alternative,
                                           {nullptr, background, 0});
        }
        if (threadIdx.x % 32 == 0) pv[i] = p;
    }
}

// IEEE float ordering followed by original feature index gives ordinal ranks.
// Canonicalize signed zero, which must be tied under numeric comparison.
__device__ unsigned long long rank_key(float x, unsigned index) {
    unsigned bits = x == 0.0f ? 0U : __float_as_uint(x);
    bits ^= (bits & 0x80000000U) ? 0xffffffffU : 0x80000000U;
    return ((unsigned long long)bits << 32) | index;
}

// Find an inclusive upper key for the first k features without sorting a row.
// Refine radix buckets until the selected bucket can be included in full.
__device__ unsigned long long rank_cutoff(const float* row, int nvar, int k) {
    if (k == 0) return 0;
    if (k == nvar) return ~0ULL;
    __shared__ unsigned histogram[256];
    __shared__ unsigned long long prefix, mask, result;
    __shared__ int remaining, done;
    if (threadIdx.x == 0) {
        prefix = mask = result = 0;
        remaining = k;
        done = 0;
    }
    __syncthreads();
    for (int shift = 56; shift >= 0; shift -= 8) {
        histogram[threadIdx.x] = 0;
        __syncthreads();
        for (size_t base = 0; base < (size_t)nvar; base += blockDim.x) {
            const size_t j = base + threadIdx.x;
            const auto key =
                j < (size_t)nvar ? rank_key(row[j], (unsigned)j) : 0;
            const bool active = j < (size_t)nvar && (key & mask) == prefix;
            const unsigned participants = __ballot_sync(MASK, active);
            if (active) {
                const unsigned bucket = (key >> shift) & 255;
                // Aggregate equal buckets within a warp to handle large ties.
                const unsigned peers = __match_any_sync(participants, bucket);
                if ((int)(threadIdx.x % 32) == __ffs(peers) - 1) {
                    atomicAdd(&histogram[bucket], (unsigned)__popc(peers));
                }
            }
        }
        __syncthreads();
        if (threadIdx.x == 0) {
            for (unsigned bucket = 0; bucket < 256; ++bucket) {
                const int size = histogram[bucket];
                if (remaining <= size) {
                    prefix |= (unsigned long long)bucket << shift;
                    mask |= 255ULL << shift;
                    if (remaining == size) {
                        result = prefix | ((1ULL << shift) - 1);
                        done = 1;
                    }
                    break;
                }
                remaining -= size;
            }
        }
        __syncthreads();
        if (done) break;
    }
    return result;
}

__global__ void select_kernel(const float* mat, size_t nobs, int nvar, int n_up,
                              int n_bm, unsigned char* selected) {
    for (size_t row = blockIdx.x; row < nobs; row += gridDim.x) {
        const float* values = mat + row * nvar;
        const auto top = rank_cutoff(values, nvar, nvar - n_up);
        // All threads must finish reading shared cutoff state before reusing
        // it.
        __syncthreads();
        const auto bottom = rank_cutoff(values, nvar, n_bm);
        for (size_t j = threadIdx.x; j < (size_t)nvar; j += blockDim.x) {
            const auto key = rank_key(values[j], (unsigned)j);
            selected[j * nobs + row] = key > top || key <= bottom;
        }
        __syncthreads();
    }
}

struct Lookup {
    const int* groups;
    const double *scores, *pvalues;
    int width, count;
    double background;
    double *es, *pv;
    int* invalid;
    __device__ void store(size_t index, int source, int size, int a) const {
        if (background - size - count + a < 0) {
            atomicExch(invalid, 1);
            return;
        }
        const size_t entry = (size_t)groups[source] * width + a;
        es[index] = scores[entry];
        pv[index] = pvalues[entry];
    }
};

// Threads in a block process adjacent cells of the same source. The selected
// mask and output are feature/source-major, so global memory accesses coalesce.
template <bool lookup = false>
__global__ void overlap_kernel(const unsigned char* __restrict__ selected,
                               const int* __restrict__ cnct,
                               const int* __restrict__ starts,
                               const int* __restrict__ offsets, size_t nobs,
                               int nsrc, int* __restrict__ overlaps,
                               Lookup tables = {}) {
    const int lanes = nobs < 32 ? 32 : 1;
    for (int source = blockIdx.y; source < nsrc; source += gridDim.y) {
        const int start = starts[source];
        const int size = offsets[source];
        for (size_t row =
                 ((size_t)blockIdx.x * blockDim.x + threadIdx.x) / lanes;
             row < nobs; row += (size_t)blockDim.x * gridDim.x / lanes) {
            int a = 0;
            for (unsigned j = threadIdx.x % lanes; j < (unsigned)size;
                 j += lanes) {
                a += selected[(size_t)cnct[start + j] * nobs + row];
            }
            for (int delta = lanes / 2; delta > 0; delta /= 2) {
                a += __shfl_down_sync(MASK, a, delta);
            }
            if (threadIdx.x % lanes == 0) {
                const size_t index = (size_t)source * nobs + row;
                if constexpr (lookup)
                    tables.store(index, source, size, a);
                else
                    overlaps[index] = a;
            }
        }
    }
}
}  // namespace
