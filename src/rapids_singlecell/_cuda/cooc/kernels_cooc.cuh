#pragma once

#include <cuda_runtime.h>

// CUDA's 64-bit integer atomicAdd overload is for unsigned long long.
using cooc_count_t = unsigned long long;

// One warp per radius, with the same FP32 reduction order for every category
// count.
__global__ void cooc_normalize(const cooc_count_t* counts, const int64_t* order,
                               float* marginals, float* out, int k, int bins) {
    int r = blockIdx.x, lane = threadIdx.x;
    size_t cells = static_cast<size_t>(k) * k;
    float* y = out + order[r];
    marginals += static_cast<size_t>(r) * k;

    float total = 0;
    for (size_t idx = lane; idx < cells; idx += 32) {
        int i = idx / k, j = idx % k;
        size_t pair = static_cast<size_t>(min(i, j)) * k + max(i, j);
        float count = static_cast<float>(counts[pair * bins + r]);
        if (i == j) count += count;  // Both directions of a same-category pair.
        y[idx * bins] = count;
        total += count;
    }
    for (int offset = 16; offset; offset >>= 1)
        total += __shfl_down_sync(0xffffffff, total, offset);
    total = __shfl_sync(0xffffffff, total, 0);
    if (total == 0) return;
    __syncwarp();

    for (int i = lane; i < k; i += 32) {
        float* row = y + static_cast<size_t>(i) * k * bins;
        float sum = 0;
        for (int j = 0; j < k; ++j) {
            float& value = row[static_cast<size_t>(j) * bins];
            value /= total;
            sum += value;
        }
        marginals[i] = sum;
    }
    __syncwarp();
    for (size_t idx = lane; idx < cells; idx += 32) {
        float a = marginals[idx / k], b = marginals[idx % k];
        y[idx * bins] = a && b ? (y[idx * bins] / a) / b : 0.0f;
    }
}

__device__ inline int cooc_distance_bin(const float* thresholds, float d,
                                        int lo, int hi) {
    while (lo < hi) {
        int mid = lo + (hi - lo) / 2;
        if (d <= thresholds[mid])
            hi = mid;
        else
            lo = mid + 1;
    }
    return lo;
}

// Interleave two 16-bit coordinates to keep nearby cells together in each
// category.
__device__ inline unsigned int cooc_spread_bits(unsigned int x) {
    x = (x | x << 8) & 0x00ff00ff;
    x = (x | x << 4) & 0x0f0f0f0f;
    x = (x | x << 2) & 0x33333333;
    return (x | x << 1) & 0x55555555;
}

__global__ void cooc_spatial_keys(const float* points, const int* labels,
                                  const float* lower, const float* upper,
                                  unsigned long long* keys, int n, int dims) {
    size_t i = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (i >= n) return;
    const float* p = points + i * dims;
    int y_dim = min(1, dims - 1);
    unsigned int x = __float2uint_rz(
        __saturatef((p[0] - lower[0]) / (upper[0] - lower[0])) * 65535);
    unsigned int y = __float2uint_rz(
        __saturatef((p[y_dim] - lower[y_dim]) / (upper[y_dim] - lower[y_dim])) *
        65535);
    keys[i] = (static_cast<unsigned long long>(labels[i]) << 32) |
              cooc_spread_bits(x) | (cooc_spread_bits(y) << 1);
}

struct CoocSpatialTile {
    int begin, size, category;
    bool finite;
};

// Each warp describes up to 32 consecutive, spatially sorted cells of one
// category. Bounds hold a (minimum, maximum) pair per coordinate.
__global__ void cooc_build_tiles(const float* points, int dims,
                                 const int* offsets, const int* tile_offsets,
                                 int k, CoocSpatialTile* tiles,
                                 float2* bounds) {
    int tile = blockIdx.x, lane = threadIdx.x;
    int lo = 0, hi = k;
    while (lo + 1 < hi) {
        int mid = lo + (hi - lo) / 2;
        if (tile < tile_offsets[mid])
            hi = mid;
        else
            lo = mid;
    }
    int begin = offsets[lo] + (tile - tile_offsets[lo]) * 32;
    int size = min(32, offsets[lo + 1] - begin);
    size_t cell = begin + min(lane, size - 1);
    bool finite = true;
    for (int d = 0; d < dims; ++d) {
        float p = points[cell * dims + d];
        finite = finite && isfinite(p);
        float lower = p, upper = p;
        for (int offset = 16; offset; offset >>= 1) {
            lower = fminf(lower, __shfl_down_sync(0xffffffff, lower, offset));
            upper = fmaxf(upper, __shfl_down_sync(0xffffffff, upper, offset));
        }
        if (lane == 0)
            bounds[static_cast<size_t>(tile) * dims + d] =
                make_float2(lower, upper);
    }
    bool all_finite = __all_sync(0xffffffff, finite);
    if (lane == 0) tiles[tile] = {begin, size, lo, all_finite};
}

template <int FixedDims>
__global__ void cooc_count_tiles(const float* __restrict__ points, int n_dims,
                                 const CoocSpatialTile* __restrict__ tiles,
                                 const float2* __restrict__ bounds, int n_tiles,
                                 const float* __restrict__ thresholds, int bins,
                                 int k, bool within_categories,
                                 cooc_count_t* __restrict__ counts, int rank,
                                 int n_devices) {
    int row = rank + blockIdx.x * n_devices;
    if (row >= n_tiles) return;
    CoocSpatialTile a = tiles[row];
    int lane = threadIdx.x % 32, warp = threadIdx.x / 32;
    int dims = FixedDims ? FixedDims : n_dims;
    size_t i = a.begin + min(lane, a.size - 1);
    const float* pa = points + i * dims;
    constexpr int hist_size = 64;
    __shared__ unsigned int histograms[4][hist_size];
    unsigned int* hist = histograms[warp];
    for (int col = row + blockIdx.y * 4 + warp; col < n_tiles;
         col += gridDim.y * 4) {
        CoocSpatialTile b = tiles[col];
        if (within_categories && b.category != a.category) break;
        int lo = 0, hi = bins;
        // Infinite or NaN coordinates require individual distance comparisons.
        if (a.finite && b.finite) {
            float dmin = 0, dmax = 0;
            for (int d = dims - 1; d >= 0; --d) {
                float2 ab = bounds[static_cast<size_t>(row) * dims + d];
                float2 bb = bounds[static_cast<size_t>(col) * dims + d];
                float near = fmaxf(0, fmaxf(ab.x - bb.y, bb.x - ab.y));
                float far = fmaxf(ab.y - bb.x, bb.y - ab.x);
                dmin = fmaf(near, near, dmin);
                dmax = fmaf(far, far, dmax);
            }
            if (dmin > thresholds[bins - 1]) continue;
            lo = cooc_distance_bin(thresholds, dmin, 0, bins - 1);
            hi = cooc_distance_bin(thresholds, dmax, lo, bins);
        }
        size_t pair = within_categories
                          ? a.category
                          : static_cast<size_t>(a.category) * k + b.category;
        cooc_count_t* out = counts + pair * bins;
        if (lo == hi) {
            if (lane == 0) {
                int pairs =
                    row == col ? a.size * (a.size - 1) / 2 : a.size * b.size;
                atomicAdd(out + lo, static_cast<cooc_count_t>(pairs));
            }
            continue;
        }
        // Cache the first bins touched by this tile pair in shared memory.
        int width = min(hist_size, min(hi, bins - 1) - lo + 1);
        for (int bin = lane; bin < width; bin += 32) hist[bin] = 0;
        __syncwarp();

        const float* pb =
            points +
            static_cast<size_t>(b.begin + min(lane, b.size - 1)) * dims;
#pragma unroll
        for (int j = 0; j < 32; ++j) {
            float distance = 0;
            for (int d = dims - 1; d >= 0; --d) {
                float delta = pa[d] - __shfl_sync(0xffffffff, pb[d], j);
                distance = fmaf(delta, delta, distance);
            }
            if (lane < a.size && j < b.size && (row != col || lane < j)) {
                int bin = cooc_distance_bin(thresholds, distance, lo, hi);
                if (bin - lo < width)
                    atomicAdd(hist + (bin - lo), 1u);
                else if (bin < bins)
                    atomicAdd(out + bin, 1ULL);
            }
        }
        __syncwarp();
        for (int bin = lane; bin < width; bin += 32) {
            if (hist[bin])
                atomicAdd(out + lo + bin, static_cast<cooc_count_t>(hist[bin]));
        }
        __syncwarp();
    }
}
