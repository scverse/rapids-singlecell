#pragma once

#include <cuda_runtime.h>

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

template <typename T, typename IdxT>
__global__ void sum_and_count_sparse_kernel(const IdxT* __restrict__ indptr,
                                            const IdxT* __restrict__ index,
                                            const T* __restrict__ data,
                                            const int* __restrict__ clusters,
                                            T* __restrict__ sum_gt0,
                                            int* __restrict__ count_gt0,
                                            int nrows, int n_cls) {
    int cell = blockDim.x * blockIdx.x + threadIdx.x;
    if (cell >= nrows) return;
    IdxT start_idx = indptr[cell];
    IdxT stop_idx = indptr[cell + 1];
    int cluster = clusters[cell];
    for (IdxT gene = start_idx; gene < stop_idx; gene++) {
        T value = data[gene];
        IdxT gene_number = index[gene];
        if (value > (T)0) {
            long long out_idx = (long long)gene_number * n_cls + cluster;
            atomicAdd(&sum_gt0[out_idx], value);
            atomicAdd(&count_gt0[out_idx], 1);
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

template <typename T, typename IdxT>
__global__ void mean_sparse_kernel(const IdxT* __restrict__ indptr,
                                   const IdxT* __restrict__ index,
                                   const T* __restrict__ data,
                                   const int* __restrict__ clusters,
                                   T* __restrict__ sum_gt0, int nrows,
                                   int n_cls) {
    int cell = blockDim.x * blockIdx.x + threadIdx.x;
    if (cell >= nrows) return;
    IdxT start_idx = indptr[cell];
    IdxT stop_idx = indptr[cell + 1];
    int cluster = clusters[cell];
    for (IdxT gene = start_idx; gene < stop_idx; gene++) {
        T value = data[gene];
        IdxT gene_number = index[gene];
        if (value > (T)0) {
            long long out_idx = (long long)gene_number * n_cls + cluster;
            atomicAdd(&sum_gt0[out_idx], value);
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
