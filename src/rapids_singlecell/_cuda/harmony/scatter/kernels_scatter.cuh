#pragma once

#include <cuda_runtime.h>
#include <stdint.h>
#include <type_traits>

template <typename T>
__global__ void scatter_add_kernel(const T* __restrict__ v,
                                   const int* __restrict__ cats, size_t n_cells,
                                   size_t n_pcs, size_t switcher,
                                   T* __restrict__ a) {
    size_t N = n_cells * n_pcs;
    for (size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x; i < N;
         i += (size_t)blockDim.x * gridDim.x) {
        size_t row = i / n_pcs;
        size_t col = i % n_pcs;

        size_t cat = static_cast<size_t>(cats[row]);
        size_t out_index = cat * n_pcs + col;

        if (switcher == 0)
            atomicAdd(&a[out_index], -v[i]);
        else
            atomicAdd(&a[out_index], v[i]);
    }
}

template <typename T>
__global__ void aggregated_matrix_kernel(T* __restrict__ aggregated_matrix,
                                         const T* __restrict__ sum,
                                         T top_corner, int n_batches) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n_batches + 1) return;

    if (i == 0) {
        aggregated_matrix[0] = top_corner;
    } else {
        aggregated_matrix[i] = sum[i - 1];
        aggregated_matrix[(n_batches + 1) * i] = sum[i - 1];
        aggregated_matrix[(n_batches + 1) * i + i] = sum[i - 1];
    }
}

template <typename T>
__global__ void scatter_add_kernel_with_bias_cat0(const T* __restrict__ v,
                                                  int n_cells, int n_pcs,
                                                  T* __restrict__ a,
                                                  const T* __restrict__ bias) {
    using VecPC = typename std::conditional<std::is_same<T, float>::value,
                                            float2, double2>::type;
    int pairs = (n_pcs + 1) / 2;
    int pc_pair = blockIdx.x;
    int eighth = blockIdx.y;
    if (pc_pair >= pairs) return;

    int pc0 = pc_pair * 2;
    int pc1 = pc0 + 1;
    bool has_pc1 = (pc1 < n_pcs);

    T acc0 = T(0);
    T acc1 = T(0);

    int cells_per_eighth = (n_cells + 7) / 8;
    int start_cell = eighth * cells_per_eighth;
    int end_cell = min(start_cell + cells_per_eighth, n_cells);

    for (int i = start_cell + threadIdx.x; i < end_cell; i += blockDim.x) {
        size_t base = static_cast<size_t>(i) * n_pcs + pc0;
        const T* ptr = v + base;
        T bb = __ldg(bias + i);
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
                int out_base = 0 * n_pcs + pc0;
                atomicAdd(&a[out_base], (T)val.x);
                if (has_pc1) atomicAdd(&a[out_base + 1], (T)val.y);
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
                int out_base = 0 * n_pcs + pc0;
                atomicAdd(&a[out_base], (T)val.x);
                if (has_pc1) atomicAdd(&a[out_base + 1], (T)val.y);
            }
        }
    }
}

// ---------- scatter_add_shared ----------
// Uses dynamic shared memory for local accumulation to reduce global atomic
// contention.
template <typename T>
__global__ void scatter_add_shared_kernel(const T* __restrict__ v,
                                          const int* __restrict__ cats,
                                          int n_cells, int n_pcs, int n_batches,
                                          int switcher, T* __restrict__ a) {
    // Dynamic shared memory: [n_batches * n_pcs] for local accumulation
    extern __shared__ unsigned char smem_raw[];
    T* shared_acc = reinterpret_cast<T*>(smem_raw);

    // Initialize shared memory to zero
    for (int i = threadIdx.x; i < n_batches * n_pcs; i += blockDim.x)
        shared_acc[i] = T(0);
    __syncthreads();

    // Calculate cell range for this block
    int cells_per_block = (n_cells + gridDim.x - 1) / gridDim.x;
    int start_cell = blockIdx.x * cells_per_block;
    int end_cell = min(start_cell + cells_per_block, n_cells);

    // Each thread processes cells with stride, accumulating into shared memory
    for (int cell = start_cell + threadIdx.x; cell < end_cell;
         cell += blockDim.x) {
        int cat = cats[cell];
        size_t v_base = (size_t)cell * n_pcs;
        int shared_base = cat * n_pcs;

        for (int pc = 0; pc < n_pcs; pc++) {
            T val = v[v_base + pc];
            atomicAdd(&shared_acc[shared_base + pc], val);
        }
    }
    __syncthreads();

    // Write shared memory results to global memory
    for (int i = threadIdx.x; i < n_batches * n_pcs; i += blockDim.x) {
        T val = shared_acc[i];
        if (val != T(0)) {
            if (switcher == 0)
                atomicAdd(&a[i], -val);
            else
                atomicAdd(&a[i], val);
        }
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
