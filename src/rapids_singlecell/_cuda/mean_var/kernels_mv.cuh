#pragma once

#include <cuda_runtime.h>
#include "../minor_tiles.cuh"

constexpr int BLOCK_SIZE_MAJOR = 64;

template <typename T, typename IdxT>
__global__ void mean_var_major_kernel(const IdxT* __restrict__ indptr,
                                      const IdxT* __restrict__ indices,
                                      const T* __restrict__ data,
                                      double* __restrict__ means,
                                      double* __restrict__ vars, int major,
                                      int /*minor*/) {
    int major_idx = blockIdx.x;
    if (major_idx >= major) return;

    IdxT start_idx = indptr[major_idx];
    IdxT stop_idx = indptr[major_idx + 1];

    __shared__ double mean_place[BLOCK_SIZE_MAJOR];
    __shared__ double var_place[BLOCK_SIZE_MAJOR];

    mean_place[threadIdx.x] = 0.0;
    var_place[threadIdx.x] = 0.0;
    __syncthreads();

    for (IdxT minor_idx = start_idx + threadIdx.x; minor_idx < stop_idx;
         minor_idx += blockDim.x) {
        double value = static_cast<double>(data[minor_idx]);
        mean_place[threadIdx.x] += value;
        var_place[threadIdx.x] += value * value;
    }
    __syncthreads();

    for (unsigned int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (threadIdx.x < s) {
            mean_place[threadIdx.x] += mean_place[threadIdx.x + s];
            var_place[threadIdx.x] += var_place[threadIdx.x + s];
        }
        __syncthreads();
    }
    if (threadIdx.x == 0) {
        means[major_idx] = mean_place[0];
        vars[major_idx] = var_place[0];
    }
}

/// Minor-axis sum / sum-of-squares per column (see minor_tiles.cuh).
template <typename T>
struct MeanVarOp {
    const T* data;
    double* means;
    double* vars;
    int tile_size;
    static constexpr size_t bytes_per_col = 2 * sizeof(double);
    static constexpr bool needs_rows = false;
    __device__ bool row_active(int) const {
        return true;
    }
    __device__ void zero_col(char* acc, int g, int) const {
        double* s = reinterpret_cast<double*>(acc);
        s[g] = 0.0;
        s[tile_size + g] = 0.0;
    }
    __device__ void add(char* acc, long long q, int g) const {
        double* s = reinterpret_cast<double*>(acc);
        const double v = static_cast<double>(data[q]);
        atomicAdd(&s[g], v);
        atomicAdd(&s[tile_size + g], v * v);
    }
    __device__ void flush_col(const char* acc, int, int col, int g) const {
        const double* s = reinterpret_cast<const double*>(acc);
        const double sq = s[tile_size + g];
        // Zero only when no nonzero of this column landed in the block.
        if (sq != 0.0) {
            atomicAdd(&means[col], s[g]);
            atomicAdd(&vars[col], sq);
        }
    }
    __device__ void add_global(long long q, int col, int) const {
        const double v = static_cast<double>(data[q]);
        atomicAdd(&means[col], v);
        atomicAdd(&vars[col], v * v);
    }
    void zero_outputs(int minor, int, cudaStream_t stream) const {
        cuda_check(
            cudaMemsetAsync(means, 0, (size_t)minor * sizeof(double), stream),
            "cudaMemsetAsync(MeanVarOp outputs)");
        cuda_check(
            cudaMemsetAsync(vars, 0, (size_t)minor * sizeof(double), stream),
            "cudaMemsetAsync(MeanVarOp outputs)");
    }
};
