#pragma once

#include <cuda_runtime.h>

// CUDA's 64-bit integer atomicAdd overload is for unsigned long long.
using cooc_count_t = unsigned long long;

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
        float dist_sq = dx * dx + dy * dy;

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
                const float dist_sq = dx * dx + dy * dy;

                // Binary search for threshold bin
                int lo = 0, hi = l_val;
                while (lo < hi) {
                    int mid = (lo + hi) >> 1;
                    if (dist_sq <= thresholds[mid])
                        hi = mid;
                    else
                        lo = mid + 1;
                }
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
