#pragma once

#include <cuda_runtime.h>
#include <stdint.h>
#include <type_traits>

// Materialize marginal observed counts from the persistent joint-code table.
// `marginal_joint_offsets` / `marginal_joint_indices` are a CSR mapping from
// each marginal category to the joint categories that contain it. One thread
// owns one (marginal category, cluster) output and reduces its joint list in a
// fixed order, so this stage needs neither atomics nor a preceding memset.
template <typename T>
__global__ void materialize_marginal_from_joint_kernel(
    const T* __restrict__ joint_values,
    const int* __restrict__ marginal_joint_offsets,
    const int* __restrict__ marginal_joint_indices, T* __restrict__ marginal,
    int n_batches, int n_clusters) {
    size_t total = (size_t)n_batches * n_clusters;
    for (size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x; i < total;
         i += (size_t)blockDim.x * gridDim.x) {
        int batch = (int)(i / n_clusters);
        int cluster = (int)(i % n_clusters);
        int begin = marginal_joint_offsets[batch];
        int end = marginal_joint_offsets[batch + 1];

        T sum = T(0);
        for (int position = begin; position < end; ++position) {
            int joint = marginal_joint_indices[position];
            sum += joint_values[(size_t)joint * n_clusters + cluster];
        }
        marginal[i] = sum;
    }
}

template <typename T>
__global__ void scatter_add_kernel_with_bias_block(
    const T* __restrict__ v, const int* __restrict__ cat_offsets,
    const int* __restrict__ cell_indices, int n_cells, int n_pcs, int n_batches,
    T* __restrict__ a, const T* __restrict__ bias) {
    using VecPC = typename std::conditional<std::is_same<T, float>::value,
                                            float2, double2>::type;
    int pairs = (n_pcs + 1) / 2;
    int block_idx = blockIdx.x;
    if (block_idx >= n_batches * pairs) return;

    int cat = block_idx / pairs + 1;
    int pc_pair = block_idx % pairs;

    int pc0 = pc_pair * 2;
    int pc1 = pc0 + 1;
    bool has_pc1 = (pc1 < n_pcs);

    T acc0 = T(0);
    T acc1 = T(0);

    int start_idx = cat_offsets[cat - 1];
    int end_idx = cat_offsets[cat];

    for (int i = start_idx + threadIdx.x; i < end_idx; i += blockDim.x) {
        int cell_idx = cell_indices[i];
        size_t in_index = static_cast<size_t>(cell_idx) * n_pcs + pc0;
        const T* ptr = v + in_index;
        T bb = __ldg(bias + cell_idx);
        if (has_pc1 && (((uintptr_t)ptr & (sizeof(VecPC) - 1)) == 0)) {
            VecPC vv = *(const VecPC*)ptr;
            acc0 += (T)vv.x * bb;
            acc1 += (T)vv.y * bb;
        } else {
            acc0 += ptr[0] * bb;
            if (has_pc1) acc1 += ptr[1] * bb;
        }
    }

#pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1) {
        acc0 += __shfl_down_sync(0xffffffff, acc0, offset);
        if (has_pc1) acc1 += __shfl_down_sync(0xffffffff, acc1, offset);
    }

    __shared__ float2 s_f[32];
    __shared__ double2 s_d[32];
    if (std::is_same<T, float>::value) {
        if ((threadIdx.x & 31) == 0)
            s_f[threadIdx.x >> 5] = make_float2((float)acc0, (float)acc1);
        __syncthreads();
        if (threadIdx.x < 32) {
            float2 val = (threadIdx.x < (blockDim.x >> 5))
                             ? s_f[threadIdx.x]
                             : make_float2(0.f, 0.f);
#pragma unroll
            for (int off = 16; off > 0; off >>= 1) {
                val.x += __shfl_down_sync(0xffffffff, val.x, off);
                val.y += __shfl_down_sync(0xffffffff, val.y, off);
            }
            if (threadIdx.x == 0) {
                int out_base = cat * n_pcs + pc0;
                a[out_base] = (T)val.x;
                if (has_pc1) a[out_base + 1] = (T)val.y;
            }
        }
    } else {
        if ((threadIdx.x & 31) == 0)
            s_d[threadIdx.x >> 5] = make_double2((double)acc0, (double)acc1);
        __syncthreads();
        if (threadIdx.x < 32) {
            double2 val = (threadIdx.x < (blockDim.x >> 5))
                              ? s_d[threadIdx.x]
                              : make_double2(0.0, 0.0);
#pragma unroll
            for (int off = 16; off > 0; off >>= 1) {
                val.x += __shfl_down_sync(0xffffffff, val.x, off);
                val.y += __shfl_down_sync(0xffffffff, val.y, off);
            }
            if (threadIdx.x == 0) {
                int out_base = cat * n_pcs + pc0;
                a[out_base] = (T)val.x;
                if (has_pc1) a[out_base + 1] = (T)val.y;
            }
        }
    }
}

// ---- Gather rows: dst[i,:] = src[idx[i],:] ----
template <typename T>
__global__ void gather_rows_kernel(const T* __restrict__ src,
                                   const int* __restrict__ idx,
                                   T* __restrict__ dst, int n_rows,
                                   int n_cols) {
    size_t N = (size_t)n_rows * n_cols;
    for (size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x; i < N;
         i += (size_t)blockDim.x * gridDim.x) {
        int row = (int)(i / n_cols);
        int col = (int)(i % n_cols);
        dst[i] = src[(size_t)idx[row] * n_cols + col];
    }
}

// ---- Scatter rows: dst[idx[i],:] = src[i,:] ----
template <typename T>
__global__ void scatter_rows_kernel(const T* __restrict__ src,
                                    const int* __restrict__ idx,
                                    T* __restrict__ dst, int n_rows,
                                    int n_cols) {
    size_t N = (size_t)n_rows * n_cols;
    for (size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x; i < N;
         i += (size_t)blockDim.x * gridDim.x) {
        int row = (int)(i / n_cols);
        int col = (int)(i % n_cols);
        dst[(size_t)idx[row] * n_cols + col] = src[i];
    }
}

// ---- Gather int: dst[i] = src[idx[i]] ----
__global__ void gather_int_kernel(const int* __restrict__ src,
                                  const int* __restrict__ idx,
                                  int* __restrict__ dst, int n) {
    for (long long i = (long long)blockIdx.x * blockDim.x + threadIdx.x; i < n;
         i += (long long)blockDim.x * gridDim.x)
        dst[i] = src[idx[i]];
}
