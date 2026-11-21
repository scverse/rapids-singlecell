#pragma once

#include <cuda_runtime.h>

// CUDA's 64-bit integer atomicAdd overload is for unsigned long long.
using cooc_count_t = unsigned long long;

// Use identical FP32 rounding for tile bounds and individual radius tests.
__device__ inline float cooc_dist_sq(float dx, float dy) {
    return fmaf(dx, dx, dy * dy);
}

// Fallback pairwise kernel (for when shared memory is insufficient for CSR
// kernel)
__global__ void occur_count_kernel_pairwise(
    const float* __restrict__ spatial, const float* __restrict__ thresholds,
    const int* __restrict__ label_idx, cooc_count_t* __restrict__ result, int n,
    int k, int l_val) {
    int i = blockIdx.x;
    int s = i % 2;
    if (i >= n) return;
    int offset = (i % 4 < 2) ? 0 : l_val;
    float spx = spatial[i * 2];
    float spy = spatial[i * 2 + 1];
    int label_i = label_idx[i];

    for (int j = i + 1; j < n; j++) {
        float dx = spx - spatial[j * 2];
        float dy = spy - spatial[j * 2 + 1];
        float dist_sq = cooc_dist_sq(dx, dy);

        int low = label_i;
        int high = label_idx[j];
        if (high < low) {
            int tmp = low;
            low = high;
            high = tmp;
        }
        if (s == 0) {
            int tmp = low;
            low = high;
            high = tmp;
        }
        for (int r = threadIdx.x; r < l_val; r += blockDim.x) {
            if (dist_sq <= thresholds[r]) {
                int index =
                    low * (k * l_val * 2) + high * l_val * 2 + r + offset;
                atomicAdd(&result[index], 1ULL);
            }
        }
    }
}

// Reduction kernel using shared memory (for small k)
__global__ void occur_reduction_kernel_shared(
    const cooc_count_t* __restrict__ result, float* __restrict__ out, int k,
    int l_val, int format) {
    // Each block handles one threshold index.
    int r_th = blockIdx.x;  // threshold index

    // Shared memory allocation
    extern __shared__ float shared[];
    float* Y = shared;
    float* col_sum = shared + (k * k);

    int total_elements = k * k;

    // Initialize shared memory
    for (int i = threadIdx.x; i < total_elements; i += blockDim.x) {
        Y[i] = 0.0f;
    }
    __syncthreads();

    // --- Load counts for this threshold and convert to float---
    if (format == 0) {
        for (int i = threadIdx.x; i < k; i += blockDim.x) {
            for (int j = 0; j < k; j++) {
                Y[i * k + j] +=
                    float(result[i * (k * l_val * 2) + j * l_val * 2 + r_th]);
                Y[j * k + i] +=
                    float(result[i * (k * l_val * 2) + j * l_val * 2 + r_th]);
                Y[i * k + j] += float(
                    result[i * (k * l_val * 2) + j * l_val * 2 + r_th + l_val]);
                Y[j * k + i] += float(
                    result[i * (k * l_val * 2) + j * l_val * 2 + r_th + l_val]);
            }
        }
    } else {
        for (int i = threadIdx.x; i < k; i += blockDim.x) {
            for (int j = 0; j < k; j++) {
                float v = float(result[i * (k * l_val) + j * l_val + r_th]);
                Y[i * k + j] += v;
                Y[j * k + i] += v;
            }
        }
    }
    __syncthreads();

    // Compute total sum of the counts
    __shared__ float total;
    float sum_val = 0.0f;
    for (int idx = threadIdx.x; idx < k * k; idx += blockDim.x) {
        sum_val += Y[idx];
    }

    // Warp-level reduction
    unsigned int mask = 0xFFFFFFFF;  // full warp mask
    for (int offset = warpSize / 2; offset > 0; offset /= 2) {
        sum_val += __shfl_down_sync(mask, sum_val, offset);
    }

    if (threadIdx.x == 0) {
        total = sum_val;
    }
    __syncthreads();

    // Normalize the matrix Y = Y / total (if total > 0)
    if (total > 0.0f) {
        for (int idx = threadIdx.x; idx < total_elements; idx += blockDim.x) {
            Y[idx] = Y[idx] / total;
        }
    } else {
        for (int i = threadIdx.x; i < k; i += blockDim.x) {
            for (int j = 0; j < k; j++) {
                out[i * (k * l_val) + j * l_val + r_th] = 0.0f;
            }
        }
        return;
    }
    __syncthreads();

    // Compute column sums of the normalized matrix
    for (int j = threadIdx.x; j < k; j += blockDim.x) {
        float sum_col = 0.0f;
        for (int i = 0; i < k; i++) {
            sum_col += Y[i * k + j];
        }
        col_sum[j] = sum_col;
    }
    __syncthreads();

    // Compute conditional probabilities
    for (int i = threadIdx.x; i < k; i += blockDim.x) {
        float row_sum = 0.0f;
        for (int j = 0; j < k; j++) {
            row_sum += Y[i * k + j];
        }

        for (int j = 0; j < k; j++) {
            float cond = 0.0f;
            if (row_sum != 0.0f) {
                cond = Y[i * k + j] / row_sum;
            }

            float final_val = 0.0f;
            if (col_sum[j] != 0.0f) {
                final_val = cond / col_sum[j];
            }

            // Write to output with (row, column, threshold) ordering
            out[i * (k * l_val) + j * l_val + r_th] = final_val;
        }
    }
    __syncthreads();
}

// Reduction kernel using global memory (for large k)
__global__ void occur_reduction_kernel_global(
    const cooc_count_t* __restrict__ result, float* __restrict__ inter_out,
    float* __restrict__ out, int k, int l_val, int format) {
    // Each block handles one threshold index.
    int r_th = blockIdx.x;  // threshold index
    if (r_th >= l_val) return;
    // Shared memory allocation
    extern __shared__ float shared[];
    float* Y = inter_out + r_th * k * k;
    float* col_sum = shared;

    int total_elements = k * k;

    // --- Load counts for this threshold and convert to float---
    if (format == 0) {
        for (int i = threadIdx.x; i < k; i += blockDim.x) {
            for (int j = 0; j < k; j++) {
                Y[i * k + j] +=
                    float(result[i * (k * l_val * 2) + j * l_val * 2 + r_th]);
                Y[j * k + i] +=
                    float(result[i * (k * l_val * 2) + j * l_val * 2 + r_th]);
                Y[i * k + j] += float(
                    result[i * (k * l_val * 2) + j * l_val * 2 + r_th + l_val]);
                Y[j * k + i] += float(
                    result[i * (k * l_val * 2) + j * l_val * 2 + r_th + l_val]);
            }
        }
    } else {
        for (int i = threadIdx.x; i < k; i += blockDim.x) {
            for (int j = 0; j < k; j++) {
                float v = float(result[i * (k * l_val) + j * l_val + r_th]);
                Y[i * k + j] += v;
                Y[j * k + i] += v;
            }
        }
    }
    __syncthreads();

    // Compute total sum of the counts
    __shared__ float total;
    float sum_val = 0.0f;
    for (int idx = threadIdx.x; idx < total_elements; idx += blockDim.x) {
        sum_val += Y[idx];
    }
    __syncthreads();
    // Warp-level reduction
    unsigned int mask = 0xFFFFFFFF;  // full warp mask
    for (int offset = warpSize / 2; offset > 0; offset /= 2) {
        sum_val += __shfl_down_sync(mask, sum_val, offset);
    }
    __syncthreads();
    if (threadIdx.x == 0) {
        total = sum_val;
    }
    __syncthreads();

    // Normalize the matrix Y = Y / total (if total > 0)
    if (total > 0.0f) {
        for (int idx = threadIdx.x; idx < total_elements; idx += blockDim.x) {
            Y[idx] = Y[idx] / total;
        }
    } else {
        for (int i = threadIdx.x; i < k; i += blockDim.x) {
            for (int j = 0; j < k; j++) {
                out[i * (k * l_val) + j * l_val + r_th] = 0.0f;
            }
        }
        return;
    }
    __syncthreads();

    // Compute column sums of the normalized matrix
    for (int j = threadIdx.x; j < k; j += blockDim.x) {
        float sum_col = 0.0f;
        for (int i = 0; i < k; i++) {
            sum_col += Y[i * k + j];
        }
        col_sum[j] = sum_col;
    }
    __syncthreads();

    // Compute conditional probabilities
    for (int i = threadIdx.x; i < k; i += blockDim.x) {
        float row_sum = 0.0f;
        for (int j = 0; j < k; j++) {
            row_sum += Y[i * k + j];
        }

        for (int j = 0; j < k; j++) {
            float cond = 0.0f;
            if (row_sum != 0.0f) {
                cond = Y[i * k + j] / row_sum;
            }

            float final_val = 0.0f;
            if (col_sum[j] != 0.0f) {
                final_val = cond / col_sum[j];
            }

            // Write to output with (row, column, threshold) ordering
            out[i * (k * l_val) + j * l_val + r_th] = final_val;
        }
    }
    __syncthreads();
}

__device__ inline int cooc_distance_bin(const float* thresholds, float d,
                                        int lo, int hi) {
    while (lo < hi) {
        int mid = (lo + hi) / 2;
        if (d <= thresholds[mid])
            hi = mid;
        else
            lo = mid + 1;
    }
    return lo;
}

// Templated CSR catpairs kernel with cell tile caching
// CELL_TILE: number of B cells to cache in shared memory
template <int CELL_TILE>
__global__ void occur_count_kernel_csr_catpairs_tiled(
    const float* __restrict__ spatial, const float* __restrict__ thresholds,
    const int* __restrict__ cat_offsets, const int* __restrict__ cell_indices,
    const int* __restrict__ pair_left, const int* __restrict__ pair_right,
    cooc_count_t* __restrict__ counts, int k, int l_val, int blocks_per_pair,
    int l_pad) {
    // Shared memory layout:
    // - B-tile coords: CELL_TILE * 2 floats
    // - Warp histograms: warps_per_block * l_pad counters
    extern __shared__ char smem[];
    float* smem_b = reinterpret_cast<float*>(smem);
    cooc_count_t* shared_hist =
        reinterpret_cast<cooc_count_t*>(smem + CELL_TILE * 2 * sizeof(float));

    const int lane = threadIdx.x & 31;
    const int warp_id = threadIdx.x >> 5;
    const int warps_per_block = blockDim.x >> 5;
    cooc_count_t* warp_hist = shared_hist + warp_id * l_pad;

    // Zero per-warp histograms
    for (int r = lane; r < l_pad; r += 32) {
        warp_hist[r] = 0;
    }
    __syncthreads();

    const int pair_id = blockIdx.x;
    const int block_in_pair = blockIdx.y;

    const int a = pair_left[pair_id];
    const int b = pair_right[pair_id];

    const int start_oa = cat_offsets[a];
    const int end_oa = cat_offsets[a + 1];
    const int start_ob = cat_offsets[b];
    const int end_ob = cat_offsets[b + 1];

    // Always iterate over the larger group (A) and tile the smaller group (B)
    // into shared memory. Small B stays hot in L2 across many A iterations.
    const bool do_swap = (end_oa - start_oa) < (end_ob - start_ob);
    const int start_a = do_swap ? start_ob : start_oa;
    const int end_a = do_swap ? end_ob : end_oa;
    const int start_b = do_swap ? start_oa : start_ob;
    const int end_b = do_swap ? end_oa : end_ob;

    const int n_a = end_a - start_a;
    const int n_b = end_b - start_b;

    // Distribute A cells across blocks_per_pair
    const int total_threads_for_pair = blocks_per_pair * blockDim.x;
    const int global_thread_in_pair = block_in_pair * blockDim.x + threadIdx.x;

    const int is_diagonal = (a == b);

    // Tile over B cells with shared memory caching
    for (int jb_base = 0; jb_base < n_b; jb_base += CELL_TILE) {
        const int cells_in_tile = min(CELL_TILE, n_b - jb_base);

        // Cooperatively load B tile coords into shared memory
        for (int i = threadIdx.x; i < CELL_TILE * 2; i += blockDim.x) {
            int cell = i / 2;
            int coord = i % 2;
            if (cell < cells_in_tile) {
                int idx = cell_indices[start_b + jb_base + cell];
                smem_b[cell * 2 + coord] = spatial[idx * 2 + coord];
            }
        }
        __syncthreads();

        // Each thread processes a subset of A cells
        for (int ia = global_thread_in_pair; ia < n_a;
             ia += total_threads_for_pair) {
            const int idx_i = cell_indices[start_a + ia];
            const float xi = spatial[idx_i * 2];
            const float yi = spatial[idx_i * 2 + 1];

            for (int c = 0; c < cells_in_tile; ++c) {
                // For diagonal blocks (a == b), skip lower triangle + diagonal
                if (is_diagonal && ia >= jb_base + c) continue;

                const float dx = xi - smem_b[c * 2];
                const float dy = yi - smem_b[c * 2 + 1];
                const float dist_sq = cooc_dist_sq(dx, dy);

                int lo = cooc_distance_bin(thresholds, dist_sq, 0, l_val);
                if (lo < l_val) {
                    atomicAdd(&warp_hist[lo], 1ULL);
                }
            }
        }
        __syncthreads();
    }

    // Reduce warp histograms into block result
    if (warp_id == 0) {
        // Sum each bin across warps into warp0's histogram
        for (int r = lane; r < l_pad; r += 32) {
            cooc_count_t sum = 0;
            for (int w = 0; w < warps_per_block; ++w) {
                sum += shared_hist[w * l_pad + r];
            }
            shared_hist[r] = sum;
        }
        __syncwarp();

        // Inclusive scan (cumulative) along thresholds
        if (threadIdx.x == 0) {
            cooc_count_t acc = 0;
            for (int r = 0; r < l_val; ++r) {
                acc += shared_hist[r];
                shared_hist[r] = acc;
            }
        }
        __syncwarp();

        // Atomic add to global counts (partial results from each block)
        for (int r = lane; r < l_val; r += 32) {
            atomicAdd(&counts[a * (k * l_val) + b * l_val + r], shared_hist[r]);
        }
    }
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
                                  const float2* lower, const float2* upper,
                                  unsigned long long* keys, int n) {
    size_t i = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (i >= n) return;
    float2 p = make_float2(points[2 * i], points[2 * i + 1]);
    float2 lo = *lower, hi = *upper;
    unsigned int x =
        __float2uint_rz(__saturatef((p.x - lo.x) / (hi.x - lo.x)) * 65535);
    unsigned int y =
        __float2uint_rz(__saturatef((p.y - lo.y) / (hi.y - lo.y)) * 65535);
    keys[i] = (static_cast<unsigned long long>(labels[i]) << 32) |
              cooc_spread_bits(x) | (cooc_spread_bits(y) << 1);
}

struct CoocSpatialTile {
    float xmin, xmax, ymin, ymax;
    int begin, size, category;
};

// Each warp describes up to 32 consecutive, spatially sorted cells of one
// category.
__global__ void cooc_build_tiles(const float2* points, const int* offsets,
                                 const int* tile_offsets, int k,
                                 CoocSpatialTile* tiles) {
    int tile = blockIdx.x, lane = threadIdx.x;
    if (tile >= tile_offsets[k]) {
        if (lane == 0) tiles[tile].size = 0;
        return;
    }
    int lo = 0, hi = k;
    while (lo + 1 < hi) {
        int mid = (lo + hi) / 2;
        if (tile < tile_offsets[mid])
            hi = mid;
        else
            lo = mid;
    }
    int begin = offsets[lo] + (tile - tile_offsets[lo]) * 32;
    int size = min(32, offsets[lo + 1] - begin);
    float2 p = points[begin + min(lane, size - 1)];
    float xmin = p.x, xmax = p.x, ymin = p.y, ymax = p.y;
    for (int offset = 16; offset; offset >>= 1) {
        xmin = fminf(xmin, __shfl_down_sync(0xffffffff, xmin, offset));
        xmax = fmaxf(xmax, __shfl_down_sync(0xffffffff, xmax, offset));
        ymin = fminf(ymin, __shfl_down_sync(0xffffffff, ymin, offset));
        ymax = fmaxf(ymax, __shfl_down_sync(0xffffffff, ymax, offset));
    }
    if (lane == 0) tiles[tile] = {xmin, xmax, ymin, ymax, begin, size, lo};
}

__global__ void cooc_count_tiles(const float2* points,
                                 const CoocSpatialTile* tiles, int n_tiles,
                                 const float* thresholds, int bins, int k,
                                 cooc_count_t* counts, int rank,
                                 int n_devices) {
    int row = rank + blockIdx.x * n_devices;
    if (row >= n_tiles) return;
    CoocSpatialTile a = tiles[row];
    if (a.size == 0) return;
    int lane = threadIdx.x % 32, warp = threadIdx.x / 32;
    float2 pa = points[a.begin + min(lane, a.size - 1)];
    extern __shared__ unsigned int histograms[];
    unsigned int* hist = histograms + warp * bins;
    for (int col = row + blockIdx.y * 4 + warp; col < n_tiles;
         col += gridDim.y * 4) {
        CoocSpatialTile b = tiles[col];
        if (b.size == 0) break;
        float dx_min = fmaxf(0, fmaxf(a.xmin - b.xmax, b.xmin - a.xmax));
        float dy_min = fmaxf(0, fmaxf(a.ymin - b.ymax, b.ymin - a.ymax));
        float dmin = cooc_dist_sq(dx_min, dy_min);
        if (dmin > thresholds[bins - 1]) continue;
        float dx_max = fmaxf(a.xmax - b.xmin, b.xmax - a.xmin);
        float dy_max = fmaxf(a.ymax - b.ymin, b.ymax - a.ymin);
        float dmax = cooc_dist_sq(dx_max, dy_max);
        int lo = cooc_distance_bin(thresholds, dmin, 0, bins);
        int hi = cooc_distance_bin(thresholds, dmax, lo, bins);
        cooc_count_t* out =
            counts + (static_cast<size_t>(a.category) * k + b.category) * bins;
        if (lo == hi) {
            if (lane == 0) {
                int pairs =
                    row == col ? a.size * (a.size - 1) / 2 : a.size * b.size;
                atomicAdd(out + lo, static_cast<cooc_count_t>(pairs));
            }
            continue;
        }
        for (int bin = lo + lane; bin <= min(hi, bins - 1); bin += 32)
            hist[bin] = 0;
        __syncwarp();
        float2 pb = points[b.begin + min(lane, b.size - 1)];
#pragma unroll
        for (int j = 0; j < 32; ++j) {
            float dx = pa.x - __shfl_sync(0xffffffff, pb.x, j);
            float dy = pa.y - __shfl_sync(0xffffffff, pb.y, j);
            if (lane < a.size && j < b.size && (row != col || lane < j)) {
                int bin =
                    cooc_distance_bin(thresholds, cooc_dist_sq(dx, dy), lo, hi);
                if (bin < bins) atomicAdd(hist + bin, 1u);
            }
        }
        __syncwarp();
        for (int bin = lo + lane; bin <= min(hi, bins - 1); bin += 32) {
            if (hist[bin])
                atomicAdd(out + bin, static_cast<cooc_count_t>(hist[bin]));
        }
        __syncwarp();
    }
}
