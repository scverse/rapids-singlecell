#pragma once

#include "wilcoxon_common.cuh"

// Walk this thread's chunk of the sorted run [lo, hi), adding each value's
// tie-averaged rank plus rank_offset to its group; runs crossing chunk borders
// are bounded by search. Returns the tie sum of runs starting in the chunk.
template <typename IndexT>
__device__ __forceinline__ double ovr_walk(const float* sv, const IndexT* si,
                                           const int* group_codes,
                                           double* col_sums, int ld,
                                           int n_groups, int lo, int hi,
                                           double rank_offset,
                                           bool compute_tie_corr) {
    int chunk = (hi - lo + blockDim.x - 1) / blockDim.x;
    int my_start = lo + threadIdx.x * chunk;
    int my_end = min(my_start + chunk, hi);
    double local_tie = 0.0;
    for (int i = my_start; i < my_end;) {
        float val = sv[i];
        int run_end = i + 1;
        while (run_end < my_end && sv[run_end] == val) ++run_end;
        int start = (i == my_start && i > lo && sv[i - 1] == val)
                        ? sorted_lower_bound(sv, lo, i, val)
                        : i;
        int end = (run_end == my_end && run_end < hi && sv[run_end] == val)
                      ? sorted_upper_bound(sv, run_end, hi, val)
                      : run_end;
        double rank = rank_offset + ((double)start + (double)end + 1.0) / 2.0;
        for (int j = i; j < run_end; ++j) {
            int g = group_codes[si[j]];
            if (g >= 0 && g < n_groups)
                atomicAdd(&col_sums[(size_t)g * ld], rank);
        }
        if (compute_tie_corr && start >= my_start)
            local_tie += tie3(end - start);
        i = run_end;
    }
    return local_tie;
}

// OVR rank sums of sorted column segments, one block per column, written to
// rank_sums[g * ld + col]. Stored and implicit zeros rank as one tie block
// between negatives and positives: each group starts at size * zero_rank and
// every nonzero adds rank - zero_rank, so dense and sparse columns share one
// walk. Global FP64 atomics are native; shared-memory ones are CAS loops.
template <typename IndexT>
__global__ void ovr_rank_kernel(
    const float* __restrict__ sorted_vals,
    const IndexT* __restrict__ sorted_rows, const int* __restrict__ seg_offsets,
    const int* __restrict__ group_codes, const double* __restrict__ group_sizes,
    double* __restrict__ rank_sums, double* __restrict__ tie_corr, int ld,
    int n_rows, int n_groups, bool compute_tie_corr) {
    int col = blockIdx.x;
    int seg_start = seg_offsets[col];
    int nnz = seg_offsets[col + 1] - seg_start;
    const float* sv = sorted_vals + seg_start;
    const IndexT* si = sorted_rows + seg_start;

    __shared__ int sh_neg_end, sh_pos_start;
    __shared__ double warp_buf[WARP_REDUCE_BUF];
    if (threadIdx.x == 0) {
        sh_neg_end = sorted_lower_bound(sv, 0, nnz, 0.0f);
        sh_pos_start = sorted_upper_bound(sv, sh_neg_end, nnz, 0.0f);
    }
    __syncthreads();
    int neg_end = sh_neg_end;
    int pos_start = sh_pos_start;
    int n_implicit = n_rows - nnz;
    int total_zero = n_implicit + pos_start - neg_end;
    double zero_rank = (double)neg_end + ((double)total_zero + 1.0) / 2.0;

    double* col_sums = rank_sums + col;
    for (int g = threadIdx.x; g < n_groups; g += blockDim.x)
        col_sums[(size_t)g * ld] = group_sizes[g] * zero_rank;
    __syncthreads();
    double local_tie =
        ovr_walk(sv, si, group_codes, col_sums, ld, n_groups, 0, neg_end,
                 -zero_rank, compute_tie_corr) +
        ovr_walk(sv, si, group_codes, col_sums, ld, n_groups, pos_start, nnz,
                 (double)n_implicit - zero_rank, compute_tie_corr);
    if (!compute_tie_corr) return;
    if (threadIdx.x == 0) local_tie += tie3(total_zero);
    double tie = wilcoxon_block_sum(local_tie, warp_buf);
    if (threadIdx.x == 0) tie_corr[col] = finalize_tie_corr(n_rows, tie);
}

template <typename IndexT>
static inline void launch_ovr_rank(
    const float* sorted_vals, const IndexT* sorted_rows, const int* seg_offsets,
    const int* group_codes, const double* group_sizes, double* rank_sums,
    double* tie_corr, int ld, int n_rows, int sb_cols, int n_groups,
    bool compute_tie_corr, cudaStream_t stream) {
    ovr_rank_kernel<IndexT><<<sb_cols, UTIL_BLOCK_SIZE, 0, stream>>>(
        sorted_vals, sorted_rows, seg_offsets, group_codes, group_sizes,
        rank_sums, tie_corr, ld, n_rows, n_groups, compute_tie_corr);
    CUDA_CHECK_LAST_ERROR(ovr_rank_kernel);
}

// Cast stored values to f32 sort keys and add f64 group sums / nnz (and
// per-column totals) for column segments, one block per column; outputs are
// [group * ld + col] and must start zeroed. data_in may alias keys_out.
template <typename InT, typename IndexT>
__global__ void ovr_cast_and_accumulate_kernel(
    const InT* data_in, float* keys_out, const IndexT* __restrict__ rows,
    const int* __restrict__ col_offsets, const int* __restrict__ group_codes,
    double* __restrict__ group_sums, double* __restrict__ group_nnz,
    double* __restrict__ total_sums, double* __restrict__ total_nnz, int ld,
    int n_groups, bool compute_nnz, bool compute_totals, double expm1_scale) {
    int col = blockIdx.x;
    double local_sum = 0.0;
    double local_nnz = 0.0;
    for (int i = col_offsets[col] + threadIdx.x; i < col_offsets[col + 1];
         i += blockDim.x) {
        InT raw = data_in[i];
        double v = (double)raw;
        keys_out[i] = (float)raw;
        if (v == 0.0) continue;
        v = stat_value(v, expm1_scale);
        local_sum += v;
        local_nnz += 1.0;
        int g = group_codes[rows[i]];
        if (g < 0 || g >= n_groups) continue;
        atomicAdd(&group_sums[(size_t)g * ld + col], v);
        if (compute_nnz) atomicAdd(&group_nnz[(size_t)g * ld + col], 1.0);
    }
    if (!compute_totals) return;
    __shared__ double warp_buf[WARP_REDUCE_BUF];
    double total = wilcoxon_block_sum(local_sum, warp_buf);
    if (threadIdx.x == 0) total_sums[col] = total;
    if (!compute_nnz) return;
    __syncthreads();
    total = wilcoxon_block_sum(local_nnz, warp_buf);
    if (threadIdx.x == 0) total_nnz[col] = total;
}

template <typename InT, typename IndexT>
static void launch_ovr_cast_and_accumulate(
    const InT* data_in, float* keys_out, const IndexT* rows,
    const int* col_offsets, const int* group_codes, double* group_sums,
    double* group_nnz, double* total_sums, double* total_nnz, int ld,
    int sb_cols, int n_groups, bool compute_nnz, bool compute_totals,
    double expm1_scale, cudaStream_t stream) {
    ovr_cast_and_accumulate_kernel<InT, IndexT>
        <<<sb_cols, UTIL_BLOCK_SIZE, 0, stream>>>(
            data_in, keys_out, rows, col_offsets, group_codes, group_sums,
            group_nnz, total_sums, total_nnz, ld, n_groups, compute_nnz,
            compute_totals, expm1_scale);
    CUDA_CHECK_LAST_ERROR(ovr_cast_and_accumulate_kernel);
}

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

// Histogram compacted shard-local column ids. Each block accumulates into a
// small shared histogram, then merges one count per column to global memory.
// This avoids one contended global atomic for every staged nonzero.
template <typename IndexT>
__global__ void compact_col_histogram_kernel(const IndexT* __restrict__ indices,
                                             int* __restrict__ col_counts,
                                             int n_items, int col_start,
                                             int n_cols) {
    extern __shared__ int block_counts[];
    for (int c = threadIdx.x; c < n_cols; c += blockDim.x) block_counts[c] = 0;
    __syncthreads();

    long long stride = (long long)gridDim.x * blockDim.x;
    for (long long item = (long long)blockIdx.x * blockDim.x + threadIdx.x;
         item < n_items; item += stride) {
        int c = (int)indices[item] - col_start;
        if (c >= 0 && c < n_cols) atomicAdd(&block_counts[c], 1);
    }
    __syncthreads();

    for (int c = threadIdx.x; c < n_cols; c += blockDim.x) {
        int count = block_counts[c];
        if (count != 0) atomicAdd(&col_counts[c], count);
    }
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

/** Gather rows (all when row_ids is null) of one dense column batch, F-order
 *  with n_rows per column or C-order rows row_stride apart, into an F-order
 *  float slab [n_out, sb_cols]. A shared 32x32 tile keeps both the reads and
 *  the writes coalesced for either input order. Block (32, 8). */
template <typename T>
__global__ void dense_gather_kernel(const T* __restrict__ src,
                                    const int* __restrict__ row_ids,
                                    float* __restrict__ out, int n_out,
                                    int n_rows, int sb_cols, bool f_order,
                                    int row_stride) {
    __shared__ float tile[32][33];  // [row][col]
    long long i0 = (long long)blockIdx.x * 32;
    int c0 = blockIdx.y * 32;
    int tx = threadIdx.x;
    for (int j = threadIdx.y; j < 32; j += blockDim.y) {
        // F-order reads walk rows (tx), C-order reads walk columns (tx).
        long long i = i0 + (f_order ? tx : j);
        int c = c0 + (f_order ? j : tx);
        if (i < n_out && c < sb_cols) {
            long long row = row_ids ? row_ids[i] : i;
            float v = (float)(f_order ? src[(long long)c * n_rows + row]
                                      : src[row * row_stride + c]);
            tile[i - i0][c - c0] = v;
        }
    }
    __syncthreads();
    for (int j = threadIdx.y; j < 32; j += blockDim.y) {
        long long i = i0 + tx;
        int c = c0 + j;
        if (i < n_out && c < sb_cols)
            out[(long long)c * n_out + i] = tile[tx][j];
    }
}

template <typename T>
static void launch_dense_gather(const T* src, const int* row_ids, float* out,
                                int n_out, int n_rows, int sb_cols,
                                bool f_order, int row_stride,
                                cudaStream_t stream) {
    dim3 grid((n_out + 31) / 32, (sb_cols + 31) / 32);
    dense_gather_kernel<T><<<grid, dim3(32, 8), 0, stream>>>(
        src, row_ids, out, n_out, n_rows, sb_cols, f_order, row_stride);
    CUDA_CHECK_LAST_ERROR(dense_gather_kernel);
}

/** Accumulate f64 group sums / nnz (and per-column totals) of a dense column
 *  batch (layout as in dense_gather_kernel) into zeroed [code * ld + col]
 *  outputs. 32x32 tiles keep the reads coalesced for either order. */
template <typename T>
__global__ void dense_group_accumulate_kernel(
    const T* __restrict__ src, const int* __restrict__ group_codes,
    double* __restrict__ group_sums, double* __restrict__ group_nnz,
    double* __restrict__ total_sums, double* __restrict__ total_nnz, int ld,
    int n_rows, int sb_cols, int n_groups, bool f_order, int row_stride,
    bool compute_nnz, bool compute_totals, double expm1_scale) {
    long long r0 = (long long)blockIdx.x * 32;
    int c0 = blockIdx.y * 32;
    for (int j = threadIdx.y; j < 32; j += blockDim.y) {
        long long row = r0 + (f_order ? threadIdx.x : j);
        int col = c0 + (f_order ? j : threadIdx.x);
        if (row >= n_rows || col >= sb_cols) continue;
        double v = (double)(f_order ? src[(long long)col * n_rows + row]
                                    : src[row * row_stride + col]);
        if (v == 0.0) continue;
        v = stat_value(v, expm1_scale);
        if (compute_totals) {
            atomicAdd(&total_sums[col], v);
            if (compute_nnz) atomicAdd(&total_nnz[col], 1.0);
        }
        int g = group_codes[row];
        if (g < 0 || g >= n_groups) continue;
        atomicAdd(&group_sums[(long long)g * ld + col], v);
        if (compute_nnz) atomicAdd(&group_nnz[(long long)g * ld + col], 1.0);
    }
}
