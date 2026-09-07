#pragma once

#include <cuda_runtime.h>
#include "../minor_tiles.cuh"

template <typename T>
__global__ void sum_and_count_dense_kernel(const T* __restrict__ data,
                                           const int* __restrict__ clusters,
                                           T* __restrict__ sum_gt0,
                                           int* __restrict__ count_gt0,
                                           size_t num_rows, size_t num_cols,
                                           size_t n_cls) {
    const size_t row_stride = static_cast<size_t>(blockDim.x) * gridDim.x;
    const size_t col_stride = static_cast<size_t>(blockDim.y) * gridDim.y;
    for (size_t i = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
         i < num_rows; i += row_stride) {
        int cluster = clusters[i];
        for (size_t j =
                 static_cast<size_t>(blockIdx.y) * blockDim.y + threadIdx.y;
             j < num_cols; j += col_stride) {
            T value = data[i * num_cols + j];
            if (value > (T)0) {
                const size_t out_idx = j * n_cls + static_cast<size_t>(cluster);
                atomicAdd(&sum_gt0[out_idx], value);
                atomicAdd(&count_gt0[out_idx], 1);
            }
        }
    }
}

template <typename T>
__global__ void mean_dense_kernel(const T* __restrict__ data,
                                  const int* __restrict__ clusters,
                                  T* __restrict__ g_cluster, size_t num_rows,
                                  size_t num_cols, size_t n_cls) {
    const size_t row_stride = static_cast<size_t>(blockDim.x) * gridDim.x;
    const size_t col_stride = static_cast<size_t>(blockDim.y) * gridDim.y;
    for (size_t i = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
         i < num_rows; i += row_stride) {
        int cluster = clusters[i];
        for (size_t j =
                 static_cast<size_t>(blockIdx.y) * blockDim.y + threadIdx.y;
             j < num_cols; j += col_stride) {
            const size_t out_idx = j * n_cls + static_cast<size_t>(cluster);
            atomicAdd(&g_cluster[out_idx], data[i * num_cols + j]);
        }
    }
}

template <typename T>
__global__ void elementwise_diff_kernel(T* __restrict__ g_cluster,
                                        const T* __restrict__ total_counts,
                                        size_t num_genes, size_t num_clusters) {
    const size_t gene_stride = static_cast<size_t>(blockDim.x) * gridDim.x;
    const size_t cluster_stride = static_cast<size_t>(blockDim.y) * gridDim.y;
    for (size_t i = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
         i < num_genes; i += gene_stride) {
        for (size_t j =
                 static_cast<size_t>(blockIdx.y) * blockDim.y + threadIdx.y;
             j < num_clusters; j += cluster_stride) {
            const size_t idx = i * num_clusters + j;
            g_cluster[idx] = g_cluster[idx] / total_counts[j];
        }
    }
}

template <typename T>
__global__ void interaction_kernel(const int* __restrict__ interactions,
                                   const int* __restrict__ interaction_clusters,
                                   const T* __restrict__ mean,
                                   T* __restrict__ res,
                                   const bool* __restrict__ mask,
                                   const T* __restrict__ g, size_t n_iter,
                                   size_t n_inter_clust, size_t n_cls) {
    const size_t iter_stride = static_cast<size_t>(blockDim.x) * gridDim.x;
    const size_t cluster_stride = static_cast<size_t>(blockDim.y) * gridDim.y;
    for (size_t i = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
         i < n_iter; i += iter_stride) {
        int rec = interactions[i * 2];
        int lig = interactions[i * 2 + 1];
        for (size_t j =
                 static_cast<size_t>(blockIdx.y) * blockDim.y + threadIdx.y;
             j < n_inter_clust; j += cluster_stride) {
            int c1 = interaction_clusters[j * 2];
            int c2 = interaction_clusters[j * 2 + 1];
            const size_t rec_idx =
                static_cast<size_t>(rec) * n_cls + static_cast<size_t>(c1);
            const size_t lig_idx =
                static_cast<size_t>(lig) * n_cls + static_cast<size_t>(c2);
            const size_t res_idx = i * n_inter_clust + j;
            T m1 = mean[rec_idx];
            T m2 = mean[lig_idx];
            if (!isnan(res[res_idx])) {
                if (m1 > (T)0 && m2 > (T)0) {
                    if (mask[rec_idx] && mask[lig_idx]) {
                        T g_sum = g[rec_idx] + g[lig_idx];
                        res[res_idx] += (g_sum > (m1 + m2));
                    } else {
                        res[res_idx] = nan("");
                    }
                } else {
                    res[res_idx] = nan("");
                }
            }
        }
    }
}

template <typename T>
__global__ void res_mean_kernel(const int* __restrict__ interactions,
                                const int* __restrict__ interaction_clusters,
                                const T* __restrict__ mean,
                                T* __restrict__ res_mean, size_t n_inter,
                                size_t n_inter_clust, size_t n_cls) {
    const size_t inter_stride = static_cast<size_t>(blockDim.x) * gridDim.x;
    const size_t cluster_stride = static_cast<size_t>(blockDim.y) * gridDim.y;
    for (size_t i = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
         i < n_inter; i += inter_stride) {
        int rec = interactions[i * 2];
        int lig = interactions[i * 2 + 1];
        for (size_t j =
                 static_cast<size_t>(blockIdx.y) * blockDim.y + threadIdx.y;
             j < n_inter_clust; j += cluster_stride) {
            int c1 = interaction_clusters[j * 2];
            int c2 = interaction_clusters[j * 2 + 1];
            const size_t rec_idx =
                static_cast<size_t>(rec) * n_cls + static_cast<size_t>(c1);
            const size_t lig_idx =
                static_cast<size_t>(lig) * n_cls + static_cast<size_t>(c2);
            T m1 = mean[rec_idx];
            T m2 = mean[lig_idx];
            if (m1 > (T)0 && m2 > (T)0) {
                res_mean[i * n_inter_clust + j] = (m1 + m2) / (T)2;
            }
        }
    }
}

/// Per (gene, cluster) sum and optionally count of positive values via the
/// grouped tile sweep (see minor_tiles.cuh); output is gene-major with the
/// cluster fastest. Layout: double sums, then int counts.
template <typename T, bool WITH_COUNT>
struct LigrecOp {
    const T* data;
    T* sum;
    int* count;  // unused when !WITH_COUNT
    int n_cls;
    int tile_size;
    static constexpr size_t bytes_per_col =
        sizeof(double) + (WITH_COUNT ? sizeof(int) : 0);
    static constexpr bool needs_rows = true;
    __device__ double* s_sum(char* acc) const {
        return reinterpret_cast<double*>(acc);
    }
    __device__ int* s_cnt(char* acc) const {
        return reinterpret_cast<int*>(acc + (size_t)tile_size * sizeof(double));
    }
    __device__ bool row_active(int) const {
        return true;
    }
    __device__ void zero_col(char* acc, int g, int) const {
        s_sum(acc)[g] = 0.0;
        if constexpr (WITH_COUNT) s_cnt(acc)[g] = 0;
    }
    __device__ void add(char* acc, long long q, int g) const {
        const T v = data[q];
        if (v > (T)0) {
            atomicAdd(&s_sum(acc)[g], static_cast<double>(v));
            if constexpr (WITH_COUNT) atomicAdd(&s_cnt(acc)[g], 1);
        }
    }
    __device__ void flush_col(const char* acc, int group, int col,
                              int g) const {
        char* a = const_cast<char*>(acc);
        const double s = s_sum(a)[g];
        const long long idx = (long long)col * n_cls + group;
        if (s != 0.0) atomicAdd(&sum[idx], static_cast<T>(s));
        if constexpr (WITH_COUNT) {
            const int c = s_cnt(a)[g];
            if (c != 0) atomicAdd(&count[idx], c);
        }
    }
    __device__ void add_global(long long q, int col, int group) const {
        const T v = data[q];
        if (v > (T)0) {
            const long long idx = (long long)col * n_cls + group;
            atomicAdd(&sum[idx], v);
            if constexpr (WITH_COUNT) atomicAdd(&count[idx], 1);
        }
    }
    void zero_outputs(int minor, int n_groups, cudaStream_t stream) const {
        cudaMemsetAsync(sum, 0, (size_t)minor * n_groups * sizeof(T), stream);
        if constexpr (WITH_COUNT)
            cudaMemsetAsync(count, 0, (size_t)minor * n_groups * sizeof(int),
                            stream);
    }
};
