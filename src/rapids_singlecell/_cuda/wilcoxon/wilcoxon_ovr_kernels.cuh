#pragma once

#include "../sparse_extract/sparse_extract.cuh"

/** Count nonzeros per column from CSR. One thread per row. */
template <typename IndexT, typename IndptrT>
__global__ void csr_col_histogram_kernel(const IndexT* __restrict__ indices,
                                         const IndptrT* __restrict__ indptr,
                                         unsigned int* __restrict__ col_counts,
                                         int n_rows, int n_cols) {
    int row = blockIdx.x * blockDim.x + threadIdx.x;
    if (row >= n_rows) return;
    IndptrT rs = indptr[row];
    IndptrT re = indptr[row + 1];
    for (IndptrT p = rs; p < re; ++p) {
        int c = (int)indices[p];
        if (c >= 0 && c < n_cols) atomicAdd(&col_counts[c], 1u);
    }
}

// CRITICAL: dense OVR gmem fallback is load-bearing for large n_groups.
// Shared-memory thresholds are device-queried; oversized smem would not launch.
static size_t ovr_smem_config(int n_groups, bool& use_gmem) {
    size_t need = (size_t)(n_groups + 32) * sizeof(double);
    if (need <= wilcoxon_max_smem_per_block()) {
        use_gmem = false;
        return need;
    }
    // Fall back to global memory accumulators; only need warp buf in smem
    use_gmem = true;
    return 32 * sizeof(double);
}

/** CRITICAL: sparse OVR gmem fallback is required for Perturb-seq-scale groups.
 *  Shared-memory thresholds are device-queried; oversized smem cannot launch.
 */
static size_t sparse_ovr_smem_config(int n_groups, bool& use_gmem) {
    size_t need = (size_t)(2 * n_groups + 32) * sizeof(double);
    if (need <= wilcoxon_max_smem_per_block()) {
        use_gmem = false;
        return need;
    }
    use_gmem = true;
    return 32 * sizeof(double);
}

/** Fill sort values with row indices [0, 1, ..., n_rows-1] per column.
 *  Grid: (n_cols,), block: 256 threads. */
__global__ void fill_row_indices_kernel(int* __restrict__ vals, int n_rows,
                                        int n_cols) {
    int col = blockIdx.x;
    if (col >= n_cols) return;
    int* out = vals + (long long)col * n_rows;
    for (int i = threadIdx.x; i < n_rows; i += blockDim.x) {
        out[i] = i;
    }
}

/** Read one dense column batch into f32 F-order for segmented sort.
 *  F-order is identity cast; C-order reads into F-order while casting. */
template <typename T>
__global__ void dense_block_to_f32_kernel(const T* __restrict__ stg,
                                          float* __restrict__ out, int n_rows,
                                          int sb_cols, bool f_order) {
    const long long total = (long long)n_rows * sb_cols;
    const long long stride = (long long)gridDim.x * blockDim.x;
    for (long long idx = (long long)blockIdx.x * blockDim.x + threadIdx.x;
         idx < total; idx += stride) {
        if (f_order) {
            out[idx] = (float)stg[idx];
        } else {
            int col = (int)(idx / n_rows);
            int row = (int)(idx % n_rows);
            out[idx] = (float)stg[(long long)row * sb_cols + col];
        }
    }
}

/** Accumulate dense batch per-group sums and optional nnz in f64.
 *  Reads native staging so means match Aggregate; ranking cast is separate. */
template <typename T>
__global__ void dense_group_accumulate_kernel(
    const T* __restrict__ stg, const int* __restrict__ group_codes,
    double* __restrict__ group_sums, double* __restrict__ group_nnz,
    double* __restrict__ total_sums, double* __restrict__ total_nnz, int n_rows,
    int sb_cols, int n_groups, bool f_order, bool compute_nnz,
    bool compute_totals) {
    int col = blockIdx.x;
    if (col >= sb_cols) return;
    for (int row = threadIdx.x; row < n_rows; row += blockDim.x) {
        double v = f_order ? (double)stg[(long long)col * n_rows + row]
                           : (double)stg[(long long)row * sb_cols + col];
        if (compute_totals) {
            atomicAdd(&total_sums[col], v);
            if (compute_nnz && v != 0.0) {
                atomicAdd(&total_nnz[col], 1.0);
            }
        }
        int g = group_codes[row];
        if (g < 0 || g >= n_groups) continue;
        atomicAdd(&group_sums[(long long)g * sb_cols + col], v);
        if (compute_nnz && v != 0.0) {
            atomicAdd(&group_nnz[(long long)g * sb_cols + col], 1.0);
        }
    }
}
