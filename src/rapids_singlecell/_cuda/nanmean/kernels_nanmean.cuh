#pragma once

#include <cuda_runtime.h>
#include "../minor_tiles.cuh"

template <typename T, typename IdxT>
__global__ void nan_mean_major_kernel(const IdxT* __restrict__ indptr,
                                      const IdxT* __restrict__ index,
                                      const T* __restrict__ data,
                                      double* __restrict__ means,
                                      int* __restrict__ nans,
                                      const bool* __restrict__ mask, int major,
                                      int minor) {
    int major_idx = blockIdx.x;
    if (major_idx >= major) {
        return;
    }
    IdxT start_idx = indptr[major_idx];
    IdxT stop_idx = indptr[major_idx + 1];

    __shared__ double mean_place[64];
    __shared__ int nan_place[64];

    mean_place[threadIdx.x] = 0.0;
    nan_place[threadIdx.x] = 0;
    __syncthreads();

    for (IdxT minor_idx = start_idx + threadIdx.x; minor_idx < stop_idx;
         minor_idx += blockDim.x) {
        IdxT gene_number = index[minor_idx];
        if (mask[gene_number]) {
            T v = data[minor_idx];
            if (isnan((double)v)) {
                nan_place[threadIdx.x] += 1;
            } else {
                mean_place[threadIdx.x] += (double)v;
            }
        }
    }
    __syncthreads();

    for (unsigned int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (threadIdx.x < s) {
            mean_place[threadIdx.x] += mean_place[threadIdx.x + s];
            nan_place[threadIdx.x] += nan_place[threadIdx.x + s];
        }
        __syncthreads();
    }
    if (threadIdx.x == 0) {
        means[major_idx] = mean_place[0];
        nans[major_idx] = nan_place[0];
    }
}

/// Minor-axis NaN-aware sum and NaN count per masked column (see
/// minor_tiles.cuh). Layout: double sums, int NaN counts, bool mask.
template <typename T>
struct NanMeanOp {
    const T* data;
    double* means;
    int* nans;
    const bool* mask;
    int tile_size;
    static constexpr size_t bytes_per_col =
        sizeof(double) + sizeof(int) + sizeof(bool);
    static constexpr bool needs_rows = false;
    __device__ double* s_sum(char* acc) const {
        return reinterpret_cast<double*>(acc);
    }
    __device__ int* s_nan(char* acc) const {
        return reinterpret_cast<int*>(acc + (size_t)tile_size * sizeof(double));
    }
    __device__ bool* s_mask(char* acc) const {
        return reinterpret_cast<bool*>(
            acc + (size_t)tile_size * (sizeof(double) + sizeof(int)));
    }
    __device__ bool row_active(int) const {
        return true;
    }
    __device__ void zero_col(char* acc, int g, int col) const {
        s_sum(acc)[g] = 0.0;
        s_nan(acc)[g] = 0;
        s_mask(acc)[g] = mask[col];
    }
    __device__ void add(char* acc, long long q, int g) const {
        if (!s_mask(acc)[g]) return;
        const double v = static_cast<double>(data[q]);
        if (isnan(v)) {
            atomicAdd(&s_nan(acc)[g], 1);
        } else {
            atomicAdd(&s_sum(acc)[g], v);
        }
    }
    __device__ void flush_col(const char* acc, int, int col, int g) const {
        char* a = const_cast<char*>(acc);
        const int n = s_nan(a)[g];
        const double s = s_sum(a)[g];
        if (n != 0) atomicAdd(&nans[col], n);
        if (s != 0.0) atomicAdd(&means[col], s);
    }
    __device__ void add_global(long long q, int col, int) const {
        if (!mask[col]) return;
        const double v = static_cast<double>(data[q]);
        if (isnan(v)) {
            atomicAdd(&nans[col], 1);
        } else {
            atomicAdd(&means[col], v);
        }
    }
    void zero_outputs(int minor, int, cudaStream_t stream) const {
        cuda_check(
            cudaMemsetAsync(means, 0, (size_t)minor * sizeof(double), stream),
            "cudaMemsetAsync(NanMeanOp outputs)");
        cuda_check(
            cudaMemsetAsync(nans, 0, (size_t)minor * sizeof(int), stream),
            "cudaMemsetAsync(NanMeanOp outputs)");
    }
};
