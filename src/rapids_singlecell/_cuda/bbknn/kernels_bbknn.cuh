#pragma once

#include <cub/block/block_radix_sort.cuh>
#include <cuda_runtime.h>
#include <math_constants.h>

__global__ void find_top_k_per_row_kernel(const float* __restrict__ data,
                                          const int* __restrict__ indptr,
                                          const int n_rows, const int trim,
                                          float* __restrict__ vals) {
    int row = blockIdx.x * blockDim.x + threadIdx.x;
    if (row >= n_rows) {
        return;
    }

    int start = indptr[row];
    int end = indptr[row + 1];
    int length = end - start;

    if (length <= trim) {
        vals[row] = 0.0f;  // insufficient elements
        return;
    }

    extern __shared__ float shared_memory[];
    int shared_offset = threadIdx.x * trim;
    float* top_k = &shared_memory[shared_offset];

    // Initialize top_k with zeros
    for (int i = 0; i < trim; ++i) {
        top_k[i] = 0.0f;
    }

    int min_index = 0;
    // Process each element in the row
    for (int idx = start; idx < end; ++idx) {
        float v = data[idx];
        if (v <= top_k[min_index]) {
            continue;
        }
        // Replace the current minimum in top_k
        top_k[min_index] = v;
        // Find new smallest element index in top_k
        for (int i = 0; i < trim; ++i) {
            if (top_k[i] < top_k[min_index]) {
                min_index = i;
            }
        }
    }

    vals[row] = top_k[min_index];
}

// Block-cooperative variant: one CUDA block per row, sorts the row with
// BlockRadixSort, returns the `trim`-th largest as the cut value. Shared
// memory is the CUB sort temp storage only, independent of `trim`, so it
// scales to large `trim` values where the per-thread top-k kernel runs out
// of shared memory. Requires every row to fit in BLOCK_THREADS *
// ITEMS_PER_THREAD.
template <int BLOCK_THREADS, int ITEMS_PER_THREAD>
__global__ void find_top_k_per_row_sorted_kernel(const float* __restrict__ data,
                                                 const int* __restrict__ indptr,
                                                 const int n_rows,
                                                 const int trim,
                                                 float* __restrict__ vals) {
    int row = blockIdx.x;
    if (row >= n_rows) {
        return;
    }

    int start = indptr[row];
    int end = indptr[row + 1];
    int length = end - start;

    if (length <= trim) {
        if (threadIdx.x == 0) {
            vals[row] = 0.0f;  // insufficient elements
        }
        return;
    }

    using BlockRadixSort =
        cub::BlockRadixSort<float, BLOCK_THREADS, ITEMS_PER_THREAD>;
    __shared__ typename BlockRadixSort::TempStorage temp_storage;

    float thread_keys[ITEMS_PER_THREAD];
#pragma unroll
    for (int i = 0; i < ITEMS_PER_THREAD; ++i) {
        int idx = threadIdx.x * ITEMS_PER_THREAD + i;
        // Pad out-of-range with -inf so they sort to the bottom of a
        // descending sort and never appear among the trim largest.
        thread_keys[i] = (idx < length) ? data[start + idx] : -CUDART_INF_F;
    }

    BlockRadixSort(temp_storage).SortDescending(thread_keys);

    // After SortDescending with blocked arrangement, sorted index i lives at
    // thread (i / ITEMS_PER_THREAD), local slot (i % ITEMS_PER_THREAD).
    int target_idx = trim - 1;
    int target_thread = target_idx / ITEMS_PER_THREAD;
    int target_item = target_idx % ITEMS_PER_THREAD;
    if (threadIdx.x == target_thread) {
        vals[row] = thread_keys[target_item];
    }
}

__global__ void cut_smaller_kernel(const int* __restrict__ indptr,
                                   const int* __restrict__ index,
                                   float* __restrict__ data,
                                   const float* __restrict__ vals,
                                   const int n_rows) {
    int row_id = blockIdx.x;
    if (row_id >= n_rows) {
        return;
    }

    int start_idx = indptr[row_id];
    int stop_idx = indptr[row_id + 1];
    float cut_row = vals[row_id];

    for (int i = start_idx + threadIdx.x; i < stop_idx; i += blockDim.x) {
        float neighbor_cut = vals[index[i]];
        float cut = fmaxf(neighbor_cut, cut_row);
        if (data[i] < cut) {
            data[i] = 0.0f;
        }
    }
}
