#pragma once

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
        if (c < n_cols) atomicAdd(&col_counts[c], 1u);
    }
}

/**
 * Scatter CSR nonzeros into CSC layout for columns [col_start, col_stop).
 * write_pos[c - col_start] must be initialized to the prefix-sum offset
 * for column c.  Each thread atomically claims a unique destination slot.
 */
template <typename InT, typename IndexT, typename IndptrT>
__global__ void csr_scatter_to_csc_kernel(
    const InT* __restrict__ data, const IndexT* __restrict__ indices,
    const IndptrT* __restrict__ indptr, int* __restrict__ write_pos,
    InT* __restrict__ csc_vals, int* __restrict__ csc_row_idx, int n_rows,
    int col_start, int col_stop) {
    int row = blockIdx.x * blockDim.x + threadIdx.x;
    if (row >= n_rows) return;
    IndptrT rs = indptr[row];
    IndptrT re = indptr[row + 1];
    // Binary search for col_start (overflow-safe midpoint)
    IndptrT lo = rs, hi = re;
    while (lo < hi) {
        IndptrT m = lo + ((hi - lo) >> 1);
        if (indices[m] < col_start)
            lo = m + 1;
        else
            hi = m;
    }
    for (IndptrT p = lo; p < re; ++p) {
        int c = (int)indices[p];
        if (c >= col_stop) break;
        int dest = atomicAdd(&write_pos[c - col_start], 1);
        csc_vals[dest] = data[p];
        csc_row_idx[dest] = row;
    }
}

// CRITICAL — DO NOT REMOVE the gmem branch (large n_groups / perturbation DE).
//
// Decide smem-vs-gmem for the DENSE OVR rank kernel
// (rank_sums_from_sorted_kernel). Per-block accumulator is one double per group
// plus a 32-slot warp buffer, i.e. (n_groups + 32) doubles. When that exceeds
// the per-block smem limit (~48 KB) the kernel must fall back to a
// global-memory accumulator (use_gmem=true). With a 48 KB limit this flips at
// roughly n_groups > 6112. Not dead: a kernel launched in smem mode with an
// oversized request simply fails to launch. Limit is device-queried via
// wilcoxon_max_smem_per_block(), so it auto-scales.
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

/**
 * CRITICAL — DO NOT REMOVE the gmem branch. This is the load-bearing path for
 * Perturb-seq / pooled-CRISPR DE, where n_groups is in the thousands.
 *
 * Decide smem-vs-gmem for the sparse OVR rank kernel. The per-block accumulator
 * is two double arrays of size n_groups (grp_sums + grp_nz_count) plus a
 * 32-slot warp buffer, i.e. (2*n_groups + 32) doubles. When that exceeds the
 * per-block shared-memory limit (~48 KB) the kernel CANNOT launch in smem mode,
 * so we set use_gmem=true and rank_sums_sparse_ovr_kernel accumulates in a
 * caller-provided global-memory buffer instead. With a 48 KB limit this flips
 * at roughly n_groups > 3056. Reviewers/static analysis have twice mistaken
 * this fallback for dead code; it is the ONLY path that works at large
 * n_groups. The limit is queried per device via wilcoxon_max_smem_per_block(),
 * so the threshold auto-scales with the GPU.
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

/**
 * Fill sort values with row indices [0,1,...,n_rows-1] per column.
 * Grid: (n_cols,), block: 256 threads.
 */
__global__ void fill_row_indices_kernel(int* __restrict__ vals, int n_rows,
                                        int n_cols) {
    int col = blockIdx.x;
    if (col >= n_cols) return;
    int* out = vals + (long long)col * n_rows;
    for (int i = threadIdx.x; i < n_rows; i += blockDim.x) {
        out[i] = i;
    }
}
