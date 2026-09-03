#pragma once

#include <cuda_runtime.h>

// Moran's I - dense numerator
template <typename T>
__global__ void morans_I_num_dense_kernel(const T* __restrict__ data_centered,
                                          const int* __restrict__ adj_row_ptr,
                                          const int* __restrict__ adj_col_ind,
                                          const T* __restrict__ adj_data,
                                          T* __restrict__ num, size_t n_samples,
                                          size_t n_features) {
    const size_t feature_stride = static_cast<size_t>(gridDim.x) * blockDim.x;
    const size_t sample_stride = static_cast<size_t>(gridDim.y) * blockDim.y;
    for (size_t f = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
         f < n_features; f += feature_stride) {
        for (size_t i =
                 static_cast<size_t>(blockIdx.y) * blockDim.y + threadIdx.y;
             i < n_samples; i += sample_stride) {
            int k_start = adj_row_ptr[i];
            int k_end = adj_row_ptr[i + 1];
            for (int k = k_start; k < k_end; ++k) {
                int j = adj_col_ind[k];
                T w = adj_data[k];
                T prod = data_centered[i * n_features + f] *
                         data_centered[static_cast<size_t>(j) * n_features + f];
                atomicAdd(&num[f], w * prod);
            }
        }
    }
}

// Moran's I - sparse numerator
template <typename T, typename AdjIdxT, typename DataIdxT>
__global__ void morans_I_num_sparse_kernel(
    const AdjIdxT* __restrict__ adj_row_ptr,
    const AdjIdxT* __restrict__ adj_col_ind, const T* __restrict__ adj_data,
    const DataIdxT* __restrict__ data_row_ptr,
    const DataIdxT* __restrict__ data_col_ind,
    const T* __restrict__ data_values, int n_samples, int n_features,
    const T* __restrict__ mean_array, T* __restrict__ num) {
    int i = blockIdx.x;
    if (i >= n_samples) {
        return;
    }
    int numThreads = blockDim.x;
    int threadid = threadIdx.x;

    __shared__ T cell1[3072];
    __shared__ T cell2[3072];
    int numruns = (n_features + 3072 - 1) / 3072;
    AdjIdxT k_start = adj_row_ptr[i];
    AdjIdxT k_end = adj_row_ptr[i + 1];
    for (AdjIdxT k = k_start; k < k_end; ++k) {
        AdjIdxT raw_j = adj_col_ind[k];
        if (raw_j < 0 || raw_j >= static_cast<AdjIdxT>(n_samples)) continue;
        int j = static_cast<int>(raw_j);
        T w = adj_data[k];
        DataIdxT cell1_start = data_row_ptr[i];
        DataIdxT cell1_stop = data_row_ptr[i + 1];
        DataIdxT cell2_start = data_row_ptr[j];
        DataIdxT cell2_stop = data_row_ptr[j + 1];
        for (int run = 0; run < numruns; ++run) {
            for (int idx = threadid; idx < 3072; idx += numThreads) {
                cell1[idx] = T(0);
                cell2[idx] = T(0);
            }
            __syncthreads();
            int batch_start = 3072 * run;
            int batch_end = 3072 * (run + 1);
            for (DataIdxT a = cell1_start + threadid; a < cell1_stop;
                 a += numThreads) {
                DataIdxT g = data_col_ind[a];
                if (g >= static_cast<DataIdxT>(batch_start) &&
                    g < static_cast<DataIdxT>(batch_end)) {
                    cell1[static_cast<int>(g - batch_start)] = data_values[a];
                }
            }
            __syncthreads();
            for (DataIdxT b = cell2_start + threadid; b < cell2_stop;
                 b += numThreads) {
                DataIdxT g = data_col_ind[b];
                if (g >= static_cast<DataIdxT>(batch_start) &&
                    g < static_cast<DataIdxT>(batch_end)) {
                    cell2[static_cast<int>(g - batch_start)] = data_values[b];
                }
            }
            __syncthreads();
            for (int gene = threadid; gene < 3072; gene += numThreads) {
                int global_gene = batch_start + gene;
                if (global_gene < n_features) {
                    T prod = (cell1[gene] - mean_array[global_gene]) *
                             (cell2[gene] - mean_array[global_gene]);
                    atomicAdd(&num[global_gene], w * prod);
                }
            }
        }
    }
}

// Geary's C - dense numerator
template <typename T>
__global__ void gearys_C_num_dense_kernel(const T* __restrict__ data,
                                          const int* __restrict__ adj_row_ptr,
                                          const int* __restrict__ adj_col_ind,
                                          const T* __restrict__ adj_data,
                                          T* __restrict__ num, size_t n_samples,
                                          size_t n_features) {
    const size_t feature_stride = static_cast<size_t>(gridDim.x) * blockDim.x;
    const size_t sample_stride = static_cast<size_t>(gridDim.y) * blockDim.y;
    for (size_t f = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
         f < n_features; f += feature_stride) {
        for (size_t i =
                 static_cast<size_t>(blockIdx.y) * blockDim.y + threadIdx.y;
             i < n_samples; i += sample_stride) {
            int k_start = adj_row_ptr[i];
            int k_end = adj_row_ptr[i + 1];
            for (int k = k_start; k < k_end; ++k) {
                int j = adj_col_ind[k];
                T w = adj_data[k];
                T diff = data[i * n_features + f] -
                         data[static_cast<size_t>(j) * n_features + f];
                atomicAdd(&num[f], w * diff * diff);
            }
        }
    }
}

// Geary's C - sparse numerator
template <typename T, typename AdjIdxT, typename DataIdxT>
__global__ void gearys_C_num_sparse_kernel(
    const AdjIdxT* __restrict__ adj_row_ptr,
    const AdjIdxT* __restrict__ adj_col_ind, const T* __restrict__ adj_data,
    const DataIdxT* __restrict__ data_row_ptr,
    const DataIdxT* __restrict__ data_col_ind,
    const T* __restrict__ data_values, int n_samples, int n_features,
    T* __restrict__ num) {
    int i = blockIdx.x;
    int numThreads = blockDim.x;
    int threadid = threadIdx.x;
    __shared__ T cell1[3072];
    __shared__ T cell2[3072];
    int numruns = (n_features + 3072 - 1) / 3072;
    if (i >= n_samples) {
        return;
    }
    AdjIdxT k_start = adj_row_ptr[i];
    AdjIdxT k_end = adj_row_ptr[i + 1];
    for (AdjIdxT k = k_start; k < k_end; ++k) {
        AdjIdxT raw_j = adj_col_ind[k];
        if (raw_j < 0 || raw_j >= static_cast<AdjIdxT>(n_samples)) continue;
        int j = static_cast<int>(raw_j);
        T w = adj_data[k];
        DataIdxT cell1_start = data_row_ptr[i];
        DataIdxT cell1_stop = data_row_ptr[i + 1];
        DataIdxT cell2_start = data_row_ptr[j];
        DataIdxT cell2_stop = data_row_ptr[j + 1];
        for (int run = 0; run < numruns; ++run) {
            for (int idx = threadid; idx < 3072; idx += numThreads) {
                cell1[idx] = T(0);
                cell2[idx] = T(0);
            }
            __syncthreads();
            int batch_start = 3072 * run;
            int batch_end = 3072 * (run + 1);
            for (DataIdxT a = cell1_start + threadid; a < cell1_stop;
                 a += numThreads) {
                DataIdxT g = data_col_ind[a];
                if (g >= static_cast<DataIdxT>(batch_start) &&
                    g < static_cast<DataIdxT>(batch_end)) {
                    cell1[static_cast<int>(g - batch_start)] = data_values[a];
                }
            }
            __syncthreads();
            for (DataIdxT b = cell2_start + threadid; b < cell2_stop;
                 b += numThreads) {
                DataIdxT g = data_col_ind[b];
                if (g >= static_cast<DataIdxT>(batch_start) &&
                    g < static_cast<DataIdxT>(batch_end)) {
                    cell2[static_cast<int>(g - batch_start)] = data_values[b];
                }
            }
            __syncthreads();
            for (int gene = threadid; gene < 3072; gene += numThreads) {
                int global_gene = batch_start + gene;
                if (global_gene < n_features) {
                    T diff = cell1[gene] - cell2[gene];
                    atomicAdd(&num[global_gene], w * diff * diff);
                }
            }
        }
    }
}

// Pre-denominator for sparse paths
template <typename T, typename IdxT>
__global__ void pre_den_sparse_kernel(const IdxT* __restrict__ data_col_ind,
                                      const T* __restrict__ data_values,
                                      long long nnz,
                                      const T* __restrict__ mean_array,
                                      T* __restrict__ den,
                                      int* __restrict__ counter) {
    const long long stride = (long long)blockDim.x * gridDim.x;
    for (long long i = (long long)blockIdx.x * blockDim.x + threadIdx.x;
         i < nnz; i += stride) {
        IdxT geneidx = data_col_ind[i];
        T value = data_values[i] - mean_array[geneidx];
        atomicAdd(&counter[geneidx], 1);
        atomicAdd(&den[geneidx], value * value);
    }
}

// ---------------------------------------------------------------------------
// Fused numerator kernels (Moran's I / Geary's C) with double accumulation.
//
// Both statistics share the cross term  cross_g = sum_i x_ig * sum_j w_ij x_jg.
// With r_i = sum_j w_ij (row sum) and c_j = sum_i w_ij (column sum):
//   Moran:  num_g = sum_i (x_ig - m_g)(acc_ig - m_g r_i)
//                 = sum_i x_ig (acc_ig - m_g r_i) - m_g * sum_j c_j x_jg +
//                 m_g^2 S0
//   Geary:  num_g = sum_ij w_ij (x_ig - x_jg)^2
//                 = sum_i x_ig (r_i x_ig - 2 acc_ig) + sum_j c_j x_jg^2
// The kernels below produce the first (permutation-dependent) sum over the
// stored nonzeros of row i only; the invariant tail terms come from
// autocorr_sparse_stats_kernel and are added on the Python side. A row
// permutation of W (permutation test) is applied via `perm` without
// materialising the permuted matrix.
// ---------------------------------------------------------------------------

constexpr int AUTOCORR_MORAN = 0;
constexpr int AUTOCORR_GEARY = 1;
constexpr int AUTOCORR_SMEM_BYTES = 32768;

template <typename IdxT>
__device__ __forceinline__ IdxT autocorr_lower_bound(const IdxT* __restrict__ a,
                                                     IdxT lo, IdxT hi,
                                                     long long key) {
    while (lo < hi) {
        IdxT mid = lo + (hi - lo) / 2;
        if (static_cast<long long>(a[mid]) < key) {
            lo = mid + 1;
        } else {
            hi = mid;
        }
    }
    return lo;
}

// Narrow [*lo, *hi) to the entries whose sorted column lies in [b0, b1).
// Rows fully inside the window (the common case) skip the binary searches.
template <typename IdxT>
__device__ __forceinline__ void autocorr_window(const IdxT* __restrict__ cols,
                                                long long b0, long long b1,
                                                IdxT* lo, IdxT* hi) {
    if (*hi <= *lo) return;
    const long long first = static_cast<long long>(cols[*lo]);
    const long long last = static_cast<long long>(cols[*hi - 1]);
    if (first < b0) *lo = autocorr_lower_bound(cols, *lo, *hi, b0);
    if (last >= b1) *hi = autocorr_lower_bound(cols, *lo, *hi, b1);
}

// Dense: one thread per feature (coalesced row reads), grid.y slabs over
// cells, register accumulation, one double atomic per (thread, slab).
// `den` (optional) receives sum_i (x_ig - m_g)^2 for the un-permuted call.
template <typename T, typename AdjIdxT, int MODE>
__global__ void autocorr_dense_kernel(
    const T* __restrict__ x, const double* __restrict__ means,
    const AdjIdxT* __restrict__ adj_row_ptr,
    const AdjIdxT* __restrict__ adj_col_ind, const T* __restrict__ adj_data,
    const int* __restrict__ perm, double* __restrict__ num,
    double* __restrict__ den, int n_samples, int n_features) {
    const int f = blockIdx.x * blockDim.x + threadIdx.x;
    if (f >= n_features) return;
    // Per-cell math stays in T (the fp64 pipe is the bottleneck otherwise);
    // only the cross-cell sums are fp64.
    const T m = static_cast<T>(means[f]);
    const size_t nf = static_cast<size_t>(n_features);
    double out = 0.0;
    double sq = 0.0;
    for (int i = blockIdx.y; i < n_samples; i += gridDim.y) {
        const int row = perm ? perm[i] : i;
        const AdjIdxT k_start = adj_row_ptr[row];
        const AdjIdxT k_end = adj_row_ptr[row + 1];
        const T xi = x[static_cast<size_t>(i) * nf + f] - m;
        if (den) sq += static_cast<double>(xi) * static_cast<double>(xi);
        T acc = T(0);
        for (AdjIdxT k = k_start; k < k_end; ++k) {
            const AdjIdxT j = adj_col_ind[k];
            if (j < 0 || j >= static_cast<AdjIdxT>(n_samples)) continue;
            const T xj = x[static_cast<size_t>(j) * nf + f] - m;
            if constexpr (MODE == AUTOCORR_MORAN) {
                acc += adj_data[k] * xj;
            } else {
                const T d = xi - xj;
                acc += adj_data[k] * d * d;
            }
        }
        if constexpr (MODE == AUTOCORR_MORAN) {
            out += static_cast<double>(xi) * static_cast<double>(acc);
        } else {
            out += static_cast<double>(acc);
        }
    }
    atomicAdd(&num[f], out);
    if (den) atomicAdd(&den[f], sq);
}

// Sparse (CSR data, sorted indices): one block per cell i. Neighbour rows are
// scatter-added into a shared gene-window accumulator (warp per neighbour,
// binary search to the window), then only the stored nonzeros of row i emit
// one double atomic each. Work is O(deg * nnz), not O(deg * n * n_features).
template <typename T, typename AdjIdxT, typename DataIdxT, int MODE>
__global__ void autocorr_sparse_kernel(
    const AdjIdxT* __restrict__ adj_row_ptr,
    const AdjIdxT* __restrict__ adj_col_ind, const T* __restrict__ adj_data,
    const int* __restrict__ perm, const DataIdxT* __restrict__ data_row_ptr,
    const DataIdxT* __restrict__ data_col_ind,
    const T* __restrict__ data_values, const double* __restrict__ means,
    double* __restrict__ num, int n_samples, int n_features) {
    constexpr int BATCH = AUTOCORR_SMEM_BYTES / static_cast<int>(sizeof(T));
    __shared__ T acc[BATCH];
    const int tid = threadIdx.x;
    const int lane = tid & 31;
    const int warp = tid >> 5;
    const int n_warps = blockDim.x >> 5;

    for (int i = blockIdx.x; i < n_samples; i += gridDim.x) {
        const DataIdxT i_start = data_row_ptr[i];
        const DataIdxT i_end = data_row_ptr[i + 1];
        if (i_end <= i_start) continue;  // block-uniform
        const int row = perm ? perm[i] : i;
        const AdjIdxT k_start = adj_row_ptr[row];
        const AdjIdxT k_end = adj_row_ptr[row + 1];
        double r = 0.0;
        for (AdjIdxT k = k_start; k < k_end; ++k) {
            const AdjIdxT j = adj_col_ind[k];
            if (j < 0 || j >= static_cast<AdjIdxT>(n_samples)) continue;
            r += static_cast<double>(adj_data[k]);
        }
        const long long g_lo = static_cast<long long>(data_col_ind[i_start]);
        const long long g_hi = static_cast<long long>(data_col_ind[i_end - 1]);
        const int run_lo = static_cast<int>(g_lo / BATCH);
        const int run_hi = static_cast<int>(g_hi / BATCH);
        for (int run = run_lo; run <= run_hi; ++run) {
            const long long b0 = static_cast<long long>(run) * BATCH;
            const long long b1 =
                min(b0 + BATCH, static_cast<long long>(n_features));
            const int width = static_cast<int>(b1 - b0);
            for (int t = tid; t < width; t += blockDim.x) acc[t] = T(0);
            __syncthreads();
            for (AdjIdxT k = k_start + warp; k < k_end; k += n_warps) {
                const AdjIdxT j = adj_col_ind[k];
                if (j < 0 || j >= static_cast<AdjIdxT>(n_samples)) continue;
                const T w = adj_data[k];
                const DataIdxT j_start = data_row_ptr[j];
                const DataIdxT j_end = data_row_ptr[j + 1];
                DataIdxT lo = j_start;
                DataIdxT hi = j_end;
                autocorr_window(data_col_ind, b0, b1, &lo, &hi);
                for (DataIdxT a = lo + lane; a < hi; a += 32) {
                    atomicAdd(&acc[static_cast<int>(data_col_ind[a] - b0)],
                              w * data_values[a]);
                }
            }
            __syncthreads();
            DataIdxT lo = i_start;
            DataIdxT hi = i_end;
            autocorr_window(data_col_ind, b0, b1, &lo, &hi);
            for (DataIdxT a = lo + tid; a < hi; a += blockDim.x) {
                const long long g = static_cast<long long>(data_col_ind[a]);
                const double xv = static_cast<double>(data_values[a]);
                const double av =
                    static_cast<double>(acc[static_cast<int>(g - b0)]);
                double v;
                if constexpr (MODE == AUTOCORR_MORAN) {
                    v = xv * (av - means[g] * r);
                } else {
                    v = xv * (r * xv - 2.0 * av);
                }
                atomicAdd(&num[g], v);
            }
            __syncthreads();
        }
    }
}

// Per-gene invariants for the sparse path, one pass over nnz:
// sum_x, sum_x2 (mean / denominator) and the W-column-sum weighted tail
// t_g = sum_j c_j x_jg (Moran) or sum_j c_j x_jg^2 (Geary).
template <typename T, typename DataIdxT, int MODE>
__global__ void autocorr_sparse_stats_kernel(
    const DataIdxT* __restrict__ data_row_ptr,
    const DataIdxT* __restrict__ data_col_ind,
    const T* __restrict__ data_values, const double* __restrict__ colsum_w,
    double* __restrict__ sum_x, double* __restrict__ sum_x2,
    double* __restrict__ tail, int n_samples) {
    for (int i = blockIdx.x; i < n_samples; i += gridDim.x) {
        const double c = colsum_w[i];
        const DataIdxT start = data_row_ptr[i];
        const DataIdxT end = data_row_ptr[i + 1];
        for (DataIdxT a = start + threadIdx.x; a < end; a += blockDim.x) {
            const DataIdxT g = data_col_ind[a];
            const double xv = static_cast<double>(data_values[a]);
            atomicAdd(&sum_x[g], xv);
            atomicAdd(&sum_x2[g], xv * xv);
            if constexpr (MODE == AUTOCORR_MORAN) {
                atomicAdd(&tail[g], c * xv);
            } else {
                atomicAdd(&tail[g], c * xv * xv);
            }
        }
    }
}
