#pragma once

#include <cuda_runtime.h>

// Shared CSR/CSC extraction kernels for compact CSC and dense F-order tiles.
// Callers canonicalize/sort before kernels that binary-search row indices.

// Scatter CSR nonzeros into compact CSC for columns [col_start, col_stop).
// `row_offset` rebases local row blocks; write_pos is atomically claimed.
template <typename InT, typename IndexT, typename IndptrT>
__global__ void csr_scatter_to_csc_kernel(
    const InT* __restrict__ data, const IndexT* __restrict__ indices,
    const IndptrT* __restrict__ indptr, int* __restrict__ write_pos,
    InT* __restrict__ csc_vals, int* __restrict__ csc_row_idx, int n_rows,
    int col_start, int col_stop, int row_offset = 0) {
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
        csc_row_idx[dest] = row_offset + row;
    }
}

// CSR column window [col_lb, col_ub) -> pre-zeroed dense F-order tile.
// atomicAdd preserves summed duplicate semantics for canonicalized CSR.
template <typename TData, typename IndptrT, typename IndexT,
          typename OutT = double>
__global__ void csr_tile_to_dense_kernel(const IndptrT* __restrict__ indptr,
                                         const IndexT* __restrict__ indices,
                                         const TData* __restrict__ data,
                                         OutT* __restrict__ out, int col_lb,
                                         int col_ub, int n_cells) {
    const long long row =
        static_cast<long long>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (row >= n_cells) {
        return;
    }
    const long long row_start = static_cast<long long>(indptr[row]);
    const long long row_end = static_cast<long long>(indptr[row + 1]);
    // Keep column ids in IndexT: narrowing a 64-bit IndexT to int would
    // truncate large column ids and misplace writes.
    const IndexT lb = static_cast<IndexT>(col_lb);
    const IndexT ub = static_cast<IndexT>(col_ub);
    for (long long k = row_start; k < row_end; ++k) {
        const IndexT col = indices[k];
        if (col >= lb && col < ub) {
            atomicAdd(&out[static_cast<long long>(col - lb) * n_cells + row],
                      static_cast<OutT>(data[k]));
        }
    }
}

// CSC column window [col_lb, col_ub) -> pre-zeroed dense F-order tile.
// No atomics: canonical CSC has one stored value per (col, row).
template <typename TData, typename IndptrT, typename IndexT,
          typename OutT = double>
__global__ void csc_tile_to_dense_kernel(const IndptrT* __restrict__ indptr,
                                         const IndexT* __restrict__ indices,
                                         const TData* __restrict__ data,
                                         OutT* __restrict__ out, int col_lb,
                                         int col_ub, int n_cells) {
    const int col = col_lb + static_cast<int>(blockIdx.x);
    if (col >= col_ub) return;
    const long long col_local = blockIdx.x;
    const IndptrT s = indptr[col];
    const IndptrT e = indptr[col + 1];
    for (IndptrT p = s + threadIdx.x; p < e; p += blockDim.x) {
        const long long row = static_cast<long long>(indices[p]);
        out[col_local * n_cells + row] = static_cast<OutT>(data[p]);
    }
}

// CSR selected rows -> pre-zeroed dense F-order tile.
// Requires sorted row indices for binary-search + col_stop break.
template <typename T, typename IndexT = int, typename IndptrT = int>
__global__ void csr_extract_dense_kernel(const T* __restrict__ data,
                                         const IndexT* __restrict__ indices,
                                         const IndptrT* __restrict__ indptr,
                                         const int* __restrict__ row_ids,
                                         T* __restrict__ out, int n_target,
                                         int col_start, int col_stop) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= n_target) return;

    int row = row_ids[tid];
    IndptrT rs = indptr[row];
    IndptrT re = indptr[row + 1];

    IndptrT lo = rs, hi = re;
    while (lo < hi) {
        IndptrT m = lo + ((hi - lo) >> 1);
        if (indices[m] < col_start)
            lo = m + 1;
        else
            hi = m;
    }

    for (IndptrT p = lo; p < re; ++p) {
        int c = indices[p];
        if (c >= col_stop) break;
        out[(long long)(c - col_start) * n_target + tid] = data[p];
    }
}

// Compact CSR rows -> pre-zeroed dense F-order tile. The compact row order is
// already the output order, and sorted indices let each thread inspect only
// the requested column interval instead of rescanning the full row per tile.
template <typename T, typename IndexT = int>
__global__ void csr_extract_dense_identity_rows_kernel(
    const T* __restrict__ data, const IndexT* __restrict__ indices,
    const int* __restrict__ indptr, T* __restrict__ out, int n_target,
    int col_start, int col_stop) {
    int row = blockIdx.x * blockDim.x + threadIdx.x;
    if (row >= n_target) return;

    int lo = indptr[row];
    int hi = indptr[row + 1];
    while (lo < hi) {
        int mid = lo + ((hi - lo) >> 1);
        if (indices[mid] < col_start)
            lo = mid + 1;
        else
            hi = mid;
    }

    int row_end = indptr[row + 1];
    for (int p = lo; p < row_end; p++) {
        int col = (int)indices[p];
        if (col >= col_stop) break;
        out[(long long)(col - col_start) * n_target + row] = data[p];
    }
}

// CSC selected rows -> pre-zeroed dense F-order tile.
// row_map[original_row] gives output row, or -1 to skip.
template <typename IndexT = int, typename IndptrT = int>
__global__ void csc_extract_mapped_kernel(const float* __restrict__ data,
                                          const IndexT* __restrict__ indices,
                                          const IndptrT* __restrict__ indptr,
                                          const int* __restrict__ row_map,
                                          float* __restrict__ out, int n_target,
                                          int col_start) {
    int col_local = blockIdx.x;
    int col = col_start + col_local;

    IndptrT start = indptr[col];
    IndptrT end = indptr[col + 1];

    for (IndptrT p = start + threadIdx.x; p < end; p += blockDim.x) {
        int out_row = row_map[(int)indices[p]];
        if (out_row >= 0) {
            out[(long long)col_local * n_target + out_row] = data[p];
        }
    }
}

// Narrowing element-wise cast, used only when input index width exceeds int32.
// Caller guarantees row/column positions fit the destination type.
template <typename SrcT, typename DstT>
__global__ void cast_array_kernel(const SrcT* __restrict__ src,
                                  DstT* __restrict__ dst, size_t n) {
    size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) dst[i] = (DstT)src[i];
}
