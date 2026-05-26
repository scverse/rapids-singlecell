#pragma once

#include <cuda_runtime.h>
#include <cmath>
#include <cstdint>

static constexpr int WARP_SIZE = 32;

enum class PseudobulkOp { Squared, AbsMean };

template <PseudobulkOp Op>
__device__ __forceinline__ double pseudobulk_elem(double x, double y) {
    double diff = x - y;
    if constexpr (Op == PseudobulkOp::Squared) {
        return diff * diff;
    } else {
        return fabs(diff);
    }
}

// Post-reduction step applied to the summed pseudobulk_elem<Op> over features.
//   Squared -> identity: callers consume the raw sum (Euclidean takes sqrt; MSE
//   divides by n). AbsMean -> divide by n_features to produce the mean absolute
//   difference (MAE).
template <PseudobulkOp Op>
__device__ __forceinline__ double pseudobulk_finalize(double acc,
                                                      int64_t n_features) {
    if constexpr (Op == PseudobulkOp::Squared) {
        return acc;
    } else {
        return acc / static_cast<double>(n_features);
    }
}

template <PseudobulkOp Op>
__global__ void paired_kernel(const double* __restrict__ X,
                              const double* __restrict__ Y,
                              double* __restrict__ out, int64_t n_pairs,
                              int64_t n_features) {
    int64_t pair = blockIdx.x;
    if (pair >= n_pairs) return;

    double acc = 0.0;
    for (int64_t f = threadIdx.x; f < n_features; f += blockDim.x) {
        acc += pseudobulk_elem<Op>(X[pair * n_features + f],
                                   Y[pair * n_features + f]);
    }

#pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1)
        acc += __shfl_down_sync(0xffffffff, acc, offset);

    __shared__ double s[32];
    if ((threadIdx.x & 31) == 0) s[threadIdx.x >> 5] = acc;
    __syncthreads();

    if (threadIdx.x < 32) {
        double val = (threadIdx.x < (blockDim.x >> 5)) ? s[threadIdx.x] : 0.0;
#pragma unroll
        for (int offset = 16; offset > 0; offset >>= 1)
            val += __shfl_down_sync(0xffffffff, val, offset);
        if (threadIdx.x == 0) {
            out[pair] = pseudobulk_finalize<Op>(val, n_features);
        }
    }
}

template <PseudobulkOp Op>
__global__ void pairwise_kernel(const double* __restrict__ X,
                                const double* __restrict__ Y,
                                double* __restrict__ out, int64_t n_x,
                                int64_t n_y, int64_t n_features) {
    int64_t pair = blockIdx.x;
    int64_t x_idx = pair / n_y;
    int64_t y_idx = pair - x_idx * n_y;
    if (x_idx >= n_x || y_idx >= n_y) return;

    double acc = 0.0;
    for (int64_t f = threadIdx.x; f < n_features; f += blockDim.x) {
        acc += pseudobulk_elem<Op>(X[x_idx * n_features + f],
                                   Y[y_idx * n_features + f]);
    }

#pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1)
        acc += __shfl_down_sync(0xffffffff, acc, offset);

    __shared__ double s[32];
    if ((threadIdx.x & 31) == 0) s[threadIdx.x >> 5] = acc;
    __syncthreads();

    if (threadIdx.x < 32) {
        double val = (threadIdx.x < (blockDim.x >> 5)) ? s[threadIdx.x] : 0.0;
#pragma unroll
        for (int offset = 16; offset > 0; offset >>= 1)
            val += __shfl_down_sync(0xffffffff, val, offset);
        if (threadIdx.x == 0) {
            out[pair] = pseudobulk_finalize<Op>(val, n_features);
        }
    }
}
