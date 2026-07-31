#pragma once

#include <cuda_runtime.h>

// ---------- dense_hist ----------
// One block per gene. X is column-major (Fortran order).
// Bins values into histogram with atomicAdd.
template <typename T>
__global__ void dense_hist_kernel(const T* __restrict__ X,
                                  const int* __restrict__ gcodes,
                                  unsigned int* __restrict__ hist, int n_cells,
                                  int n_genes, int n_groups, int n_bins,
                                  double bin_low, double inv_bin_width) {
    int gene = blockIdx.x;
    int nbt = n_bins + 1;
    unsigned int* dst = hist + (long long)gene * n_groups * nbt;

    const T* col = X + (long long)gene * n_cells;
    for (int c = threadIdx.x; c < n_cells; c += blockDim.x) {
        double val = (double)col[c];
        int grp = gcodes[c];

        if (grp < 0 || grp >= n_groups) continue;  // skip unselected cells

        int raw = (int)((val - bin_low) * inv_bin_width);
        int bin = min(max(raw, 0), n_bins - 1) + 1;
        atomicAdd(&dst[grp * nbt + bin], 1u);
    }
}

// ---------- csr_hist ----------
// One block per cell row. Threads stride over nonzeros.
template <typename T, typename IdxT>
__global__ void csr_hist_kernel(
    const T* __restrict__ data, const IdxT* __restrict__ indices,
    const IdxT* __restrict__ indptr, const int* __restrict__ gcodes,
    unsigned int* __restrict__ hist, int n_cells, int n_genes, int n_groups,
    int n_bins, double bin_low, double inv_bin_width, int gene_start) {
    int row = blockIdx.x;
    if (row >= n_cells) return;

    int grp = gcodes[row];
    if (grp < 0 || grp >= n_groups) return;  // skip unselected cells

    int nbt = n_bins + 1;
    IdxT row_start = indptr[row];
    IdxT row_end = indptr[row + 1];
    long long gene_stop = (long long)gene_start + n_genes;

    for (IdxT i = row_start + threadIdx.x; i < row_end; i += blockDim.x) {
        long long col = static_cast<long long>(indices[i]);
        if (col < gene_start || col >= gene_stop) continue;

        double val = (double)data[i];
        if (val == 0.0) continue;  // explicit zero skipped

        long long gene = col - gene_start;
        int raw = (int)((val - bin_low) * inv_bin_width);
        int bin = min(max(raw, 0), n_bins - 1) + 1;

        atomicAdd(&hist[gene * n_groups * nbt + grp * nbt + bin], 1u);
    }
}

// ---------- csc_hist ----------
// One block per gene column. Threads stride over nonzeros.
template <typename T, typename IdxT>
__global__ void csc_hist_kernel(
    const T* __restrict__ data, const IdxT* __restrict__ indices,
    const IdxT* __restrict__ indptr, const int* __restrict__ gcodes,
    unsigned int* __restrict__ hist, int n_cells, int n_genes, int n_groups,
    int n_bins, double bin_low, double inv_bin_width, int gene_start) {
    int gene = blockIdx.x;
    if (gene >= n_genes) return;

    int nbt = n_bins + 1;
    unsigned int* dst = hist + (long long)gene * n_groups * nbt;

    long long col = (long long)gene_start + gene;
    IdxT col_start = indptr[col];
    IdxT col_end = indptr[col + 1];

    for (IdxT i = col_start + threadIdx.x; i < col_end; i += blockDim.x) {
        double val = (double)data[i];
        if (val == 0.0) continue;

        long long row = static_cast<long long>(indices[i]);
        if (row < 0 || row >= n_cells) continue;
        int grp = gcodes[row];
        if (grp < 0 || grp >= n_groups) continue;  // skip unselected cells

        int raw = (int)((val - bin_low) * inv_bin_width);
        int bin = min(max(raw, 0), n_bins - 1) + 1;

        atomicAdd(&dst[grp * nbt + bin], 1u);
    }
}
