#pragma once

#include <cuda_runtime.h>

static constexpr int COLSUM_ATOMIC_TILE_COLS = 32;

template <typename T>
__global__ void colsum_kernel(const T* __restrict__ A, T* __restrict__ out,
                              size_t rows, size_t cols) {
    size_t tid = threadIdx.x;
    for (size_t col = blockIdx.x; col < cols; col += gridDim.x) {
        T acc = (T)0;
        for (size_t i = tid; i < rows; i += blockDim.x) {
            acc += A[i * cols + col];
        }
#pragma unroll
        for (int offset = 16; offset > 0; offset >>= 1)
            acc += __shfl_down_sync(0xffffffff, acc, offset);
        __shared__ T s[32];
        if ((threadIdx.x & 31) == 0) s[threadIdx.x >> 5] = acc;
        __syncthreads();
        if (threadIdx.x < 32) {
            T val = (threadIdx.x < (blockDim.x >> 5)) ? s[threadIdx.x] : (T)0;
#pragma unroll
            for (int off = 16; off > 0; off >>= 1)
                val += __shfl_down_sync(0xffffffff, val, off);
            if (threadIdx.x == 0) out[col] = val;
        }
        __syncthreads();
    }
}

// Block handles one column-tile, processes rows_per_tile rows.
// blockIdx.x = column tile, blockIdx.y = row tile.
// Uses shared memory to reduce atomics: one per column per block.
template <typename T>
__global__ void colsum_atomic_kernel(const T* __restrict__ A,
                                     T* __restrict__ out, size_t rows,
                                     size_t cols, size_t rows_per_tile) {
    __shared__ T col_sums[COLSUM_ATOMIC_TILE_COLS];

    size_t col = blockIdx.x * COLSUM_ATOMIC_TILE_COLS + threadIdx.y;
    size_t start_row = blockIdx.y * rows_per_tile;
    size_t end_row = start_row + rows_per_tile;
    if (end_row > rows) end_row = rows;

    // Initialize shared memory
    if (threadIdx.y == 0 && threadIdx.x < COLSUM_ATOMIC_TILE_COLS)
        col_sums[threadIdx.x] = (T)0;
    __syncthreads();

    // Each thread accumulates multiple rows
    T acc = (T)0;
    if (col < cols) {
#pragma unroll 4
        for (size_t row = start_row + threadIdx.x; row < end_row; row += 32) {
            acc += A[row * cols + col];
        }
    }

// Warp-level reduction across 32 threads (different rows)
#pragma unroll
    for (int off = 16; off > 0; off >>= 1)
        acc += __shfl_down_sync(0xffffffff, acc, off);

    // Lane 0 of each warp accumulates into shared memory
    if (threadIdx.x == 0 && threadIdx.y < COLSUM_ATOMIC_TILE_COLS)
        atomicAdd(&col_sums[threadIdx.y], acc);
    __syncthreads();

    // First warp writes to global memory (one atomic per column)
    if (threadIdx.x == 0 && threadIdx.y < COLSUM_ATOMIC_TILE_COLS) {
        size_t out_col = blockIdx.x * COLSUM_ATOMIC_TILE_COLS + threadIdx.y;
        if (out_col < cols && col_sums[threadIdx.y] != (T)0)
            atomicAdd(&out[out_col], col_sums[threadIdx.y]);
    }
}
