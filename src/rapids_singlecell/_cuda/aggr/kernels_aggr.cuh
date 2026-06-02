#pragma once

#include <cuda_runtime.h>

// Compile-time selector for which raw accumulators a kernel writes. Combined as
// a bitmask so each kernel instantiation emits only the atomicAdds (and only
// touches the output buffers) that the caller actually requested. A null buffer
// pointer is never dereferenced because its bit is absent from MASK.
constexpr int AGGR_SUM = 1;    // sum of values
constexpr int AGGR_COUNT = 2;  // count of nonzero entries
constexpr int AGGR_SQSUM = 4;  // sum of squared values

// sparse -> dense aggregate (CSR by cells), mask per cell, cats per cell
template <typename T, typename IdxT, int MASK>
__global__ void csr_aggr_kernel(
    const IdxT* __restrict__ indptr, const IdxT* __restrict__ index,
    const T* __restrict__ data, double* __restrict__ out_sum,
    double* __restrict__ out_count, double* __restrict__ out_sqsum,
    const int* __restrict__ cats, const bool* __restrict__ mask, size_t n_cells,
    size_t n_genes) {
    size_t cell = blockIdx.x;
    if (cell >= n_cells || !mask[cell]) return;
    IdxT cell_start = indptr[cell];
    IdxT cell_end = indptr[cell + 1];
    size_t group = static_cast<size_t>(cats[cell]);
    for (IdxT p = cell_start + threadIdx.x; p < cell_end; p += blockDim.x) {
        size_t idx = group * n_genes + static_cast<size_t>(index[p]);
        double v = static_cast<double>(data[p]);
        if constexpr (MASK & AGGR_SUM) atomicAdd(&out_sum[idx], v);
        if constexpr (MASK & AGGR_COUNT) atomicAdd(&out_count[idx], 1.0);
        if constexpr (MASK & AGGR_SQSUM) atomicAdd(&out_sqsum[idx], v * v);
    }
}

// sparse -> dense aggregate (CSC by genes), mask per cell, cats per cell
template <typename T, typename IdxT, int MASK>
__global__ void csc_aggr_kernel(
    const IdxT* __restrict__ indptr, const IdxT* __restrict__ index,
    const T* __restrict__ data, double* __restrict__ out_sum,
    double* __restrict__ out_count, double* __restrict__ out_sqsum,
    const int* __restrict__ cats, const bool* __restrict__ mask, size_t n_cells,
    size_t n_genes) {
    size_t gene = blockIdx.x;
    if (gene >= n_genes) return;
    IdxT gene_start = indptr[gene];
    IdxT gene_end = indptr[gene + 1];
    for (IdxT p = gene_start + threadIdx.x; p < gene_end; p += blockDim.x) {
        size_t cell = static_cast<size_t>(index[p]);
        if (!mask[cell]) continue;
        size_t group = static_cast<size_t>(cats[cell]);
        size_t idx = group * n_genes + gene;
        double v = static_cast<double>(data[p]);
        if constexpr (MASK & AGGR_SUM) atomicAdd(&out_sum[idx], v);
        if constexpr (MASK & AGGR_COUNT) atomicAdd(&out_count[idx], 1.0);
        if constexpr (MASK & AGGR_SQSUM) atomicAdd(&out_sqsum[idx], v * v);
    }
}

// sparse -> sparse copy (CSR by cells) row/col/value from one to another by
// cats/mask
template <typename T, typename IdxT, typename OutIdxT>
__global__ void csr_to_coo_kernel(
    const IdxT* __restrict__ indptr, const IdxT* __restrict__ index,
    const T* __restrict__ data, OutIdxT* __restrict__ row,
    OutIdxT* __restrict__ col, double* __restrict__ ndata,
    const int* __restrict__ cats, const bool* __restrict__ mask, int n_cells) {
    int cell = blockIdx.x;
    if (cell >= n_cells || !mask[cell]) return;
    IdxT start = indptr[cell];
    IdxT end = indptr[cell + 1];
    int group = cats[cell];
    for (IdxT p = start + threadIdx.x; p < end; p += blockDim.x) {
        ndata[p] = static_cast<double>(data[p]);
        row[p] = static_cast<OutIdxT>(group);
        col[p] = static_cast<OutIdxT>(index[p]);
    }
}

// variance adjust per group (CSR-like segment)
template <typename IdxT>
__global__ void sparse_var_kernel(const IdxT* __restrict__ indptr,
                                  const IdxT* __restrict__ index,
                                  double* __restrict__ data,
                                  const double* __restrict__ mean_data,
                                  double* __restrict__ n_cells, int dof,
                                  int n_groups) {
    int group = blockIdx.x;
    if (group >= n_groups) return;
    IdxT start = indptr[group];
    IdxT end = indptr[group + 1];
    double doffer =
        n_cells[group] / (n_cells[group] - static_cast<double>(dof));
    for (IdxT p = start + threadIdx.x; p < end; p += blockDim.x) {
        double var = data[p];
        double mean_sq = mean_data[p] * mean_data[p];
        var = var - mean_sq;
        data[p] = var * doffer;
    }
}

// dense C-order aggregator
template <typename T, int MASK>
__global__ void dense_aggr_kernel_C(const T* __restrict__ data,
                                    double* __restrict__ out_sum,
                                    double* __restrict__ out_count,
                                    double* __restrict__ out_sqsum,
                                    const int* __restrict__ cats,
                                    const bool* __restrict__ mask,
                                    size_t n_cells, size_t n_genes) {
    size_t i = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    size_t stride = static_cast<size_t>(gridDim.x) * blockDim.x;
    size_t N = n_cells * n_genes;
    while (i < N) {
        size_t cell = i / n_genes;
        size_t gene = i % n_genes;
        if (mask[cell]) {
            size_t group = static_cast<size_t>(cats[cell]);
            double v = static_cast<double>(data[cell * n_genes + gene]);
            if (v != 0.0) {
                size_t idx = group * n_genes + gene;
                if constexpr (MASK & AGGR_SUM) atomicAdd(&out_sum[idx], v);
                if constexpr (MASK & AGGR_COUNT)
                    atomicAdd(&out_count[idx], 1.0);
                if constexpr (MASK & AGGR_SQSUM)
                    atomicAdd(&out_sqsum[idx], v * v);
            }
        }
        i += stride;
    }
}

// dense F-order aggregator
template <typename T, int MASK>
__global__ void dense_aggr_kernel_F(const T* __restrict__ data,
                                    double* __restrict__ out_sum,
                                    double* __restrict__ out_count,
                                    double* __restrict__ out_sqsum,
                                    const int* __restrict__ cats,
                                    const bool* __restrict__ mask,
                                    size_t n_cells, size_t n_genes) {
    size_t i = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    size_t stride = static_cast<size_t>(gridDim.x) * blockDim.x;
    size_t N = n_cells * n_genes;
    while (i < N) {
        size_t cell = i % n_cells;
        size_t gene = i / n_cells;
        if (mask[cell]) {
            size_t group = static_cast<size_t>(cats[cell]);
            double v = static_cast<double>(data[gene * n_cells + cell]);
            if (v != 0.0) {
                size_t idx = group * n_genes + gene;
                if constexpr (MASK & AGGR_SUM) atomicAdd(&out_sum[idx], v);
                if constexpr (MASK & AGGR_COUNT)
                    atomicAdd(&out_count[idx], 1.0);
                if constexpr (MASK & AGGR_SQSUM)
                    atomicAdd(&out_sqsum[idx], v * v);
            }
        }
        i += stride;
    }
}
