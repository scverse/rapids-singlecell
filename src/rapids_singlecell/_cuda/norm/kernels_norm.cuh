#pragma once

#include <cuda_runtime.h>

// One block per row. Parallel row sum + scale in one pass.
template <typename T>
__global__ void dense_row_scale_kernel(T* __restrict__ data, int nrows,
                                       int ncols, T target_sum) {
    int row = blockIdx.x;
    if (row >= nrows) return;

    int row_offset = row * ncols;

    // Parallel row sum
    T val = (T)0;
    for (int i = threadIdx.x; i < ncols; i += blockDim.x)
        val += data[row_offset + i];

    // Warp-level reduction
    for (int offset = warpSize / 2; offset > 0; offset >>= 1)
        val += __shfl_down_sync(0xffffffff, val, offset);

    // Cross-warp reduction
    int nWarps = blockDim.x >> 5;
    __shared__ T warp_sums[8];
    int laneIdx = threadIdx.x & 31;
    int warpIdx = threadIdx.x >> 5;
    if (laneIdx == 0) warp_sums[warpIdx] = val;
    __syncthreads();

    val = (threadIdx.x < nWarps) ? warp_sums[threadIdx.x] : (T)0;
    if (warpIdx == 0) {
        for (int offset = warpSize / 2; offset > 0; offset >>= 1)
            val += __shfl_down_sync(0xffffffff, val, offset);
    }

    // Broadcast scale
    if (threadIdx.x == 0)
        warp_sums[0] = (val > (T)0) ? (target_sum / val) : (T)0;
    __syncthreads();

    // Parallel multiply
    T scale = warp_sums[0];
    if (scale > (T)0) {
        for (int i = threadIdx.x; i < ncols; i += blockDim.x)
            data[row_offset + i] *= scale;
    }
}

// One block per row. Parallel row sum + scale in one pass.
template <typename T, typename IdxT>
__global__ void csr_row_scale_kernel(const IdxT* __restrict__ indptr,
                                     T* __restrict__ data, int nrows,
                                     T target_sum) {
    int row = blockIdx.x;
    if (row >= nrows) return;

    IdxT start = indptr[row];
    IdxT end = indptr[row + 1];

    // Parallel row sum
    T val = (T)0;
    for (IdxT i = start + threadIdx.x; i < end; i += blockDim.x) val += data[i];

    // Warp-level reduction
    for (int offset = warpSize / 2; offset > 0; offset >>= 1)
        val += __shfl_down_sync(0xffffffff, val, offset);

    // Cross-warp reduction
    int nWarps = blockDim.x >> 5;
    __shared__ T warp_sums[8];
    int laneIdx = threadIdx.x & 31;
    int warpIdx = threadIdx.x >> 5;
    if (laneIdx == 0) warp_sums[warpIdx] = val;
    __syncthreads();

    val = (threadIdx.x < nWarps) ? warp_sums[threadIdx.x] : (T)0;
    if (warpIdx == 0) {
        for (int offset = warpSize / 2; offset > 0; offset >>= 1)
            val += __shfl_down_sync(0xffffffff, val, offset);
    }

    // Broadcast scale
    if (threadIdx.x == 0)
        warp_sums[0] = (val > (T)0) ? (target_sum / val) : (T)0;
    __syncthreads();

    // Parallel multiply
    T scale = warp_sums[0];
    if (scale > (T)0) {
        for (IdxT i = start + threadIdx.x; i < end; i += blockDim.x)
            data[i] *= scale;
    }
}

// One block per row. Warp-shuffle reduction for row sum.
template <typename T, typename IdxT>
__global__ void csr_sum_major_kernel(const IdxT* __restrict__ indptr,
                                     const T* __restrict__ data,
                                     T* __restrict__ sums, int major) {
    int major_idx = blockIdx.x;
    if (major_idx >= major) return;

    IdxT start_idx = indptr[major_idx];
    IdxT stop_idx = indptr[major_idx + 1];

    T val = (T)0;
    for (IdxT i = start_idx + threadIdx.x; i < stop_idx; i += blockDim.x)
        val += data[i];

    // Warp-level reduction
    for (int offset = warpSize / 2; offset > 0; offset >>= 1)
        val += __shfl_down_sync(0xffffffff, val, offset);

    // Cross-warp reduction
    int nWarps = blockDim.x >> 5;
    __shared__ T warp_sums[8];
    int laneIdx = threadIdx.x & 31;
    int warpIdx = threadIdx.x >> 5;
    if (laneIdx == 0) warp_sums[warpIdx] = val;
    __syncthreads();

    val = (threadIdx.x < nWarps) ? warp_sums[threadIdx.x] : (T)0;
    if (warpIdx == 0) {
        for (int offset = warpSize / 2; offset > 0; offset >>= 1)
            val += __shfl_down_sync(0xffffffff, val, offset);
    }

    if (threadIdx.x == 0) sums[major_idx] = val;
}

// ---------- find_hi_genes_csr ----------
// One block per row. Computes row sum, then marks genes whose value >
// max_fraction * row_sum.
template <typename T, typename IdxT>
__global__ void find_hi_genes_csr_kernel(const IdxT* __restrict__ indptr,
                                         const IdxT* __restrict__ indices,
                                         const T* __restrict__ data,
                                         bool* __restrict__ gene_is_hi,
                                         T max_fraction, int nrows) {
    int row = blockIdx.x;
    if (row >= nrows) return;

    IdxT start = indptr[row];
    IdxT end = indptr[row + 1];

    // Phase 1: partial row sum
    T val = (T)0;
    for (IdxT i = start + threadIdx.x; i < end; i += blockDim.x) val += data[i];

    // Warp-level reduction
    for (int offset = warpSize / 2; offset > 0; offset >>= 1)
        val += __shfl_down_sync(0xffffffff, val, offset);

    // Cross-warp reduction
    int nWarps = blockDim.x >> 5;
    __shared__ T warp_sums[8];
    int laneIdx = threadIdx.x & 31;
    int warpIdx = threadIdx.x >> 5;
    if (laneIdx == 0) warp_sums[warpIdx] = val;
    __syncthreads();

    val = (threadIdx.x < nWarps) ? warp_sums[threadIdx.x] : (T)0;
    if (warpIdx == 0) {
        for (int offset = warpSize / 2; offset > 0; offset >>= 1)
            val += __shfl_down_sync(0xffffffff, val, offset);
    }

    // Broadcast threshold
    if (threadIdx.x == 0) warp_sums[0] = max_fraction * val;
    __syncthreads();

    // Phase 2: check elements against threshold
    T threshold = warp_sums[0];
    for (IdxT i = start + threadIdx.x; i < end; i += blockDim.x) {
        if (data[i] > threshold) gene_is_hi[indices[i]] = true;
    }
}

// ---------- masked_mul_csr ----------
// One block per row. Computes masked row sum (skipping masked genes), then
// scales all data.
template <typename T, typename IdxT>
__global__ void masked_mul_csr_kernel(const IdxT* __restrict__ indptr,
                                      const IdxT* __restrict__ indices,
                                      T* __restrict__ data,
                                      const bool* __restrict__ gene_mask,
                                      int nrows, T tsum) {
    int row = blockIdx.x;
    if (row >= nrows) return;

    IdxT start = indptr[row];
    IdxT end = indptr[row + 1];

    // Phase 1: masked row sum
    T val = (T)0;
    for (IdxT i = start + threadIdx.x; i < end; i += blockDim.x) {
        if (!gene_mask[indices[i]]) val += data[i];
    }

    // Warp-level reduction
    for (int offset = warpSize / 2; offset > 0; offset >>= 1)
        val += __shfl_down_sync(0xffffffff, val, offset);

    // Cross-warp reduction
    int nWarps = blockDim.x >> 5;
    __shared__ T warp_sums[8];
    int laneIdx = threadIdx.x & 31;
    int warpIdx = threadIdx.x >> 5;
    if (laneIdx == 0) warp_sums[warpIdx] = val;
    __syncthreads();

    val = (threadIdx.x < nWarps) ? warp_sums[threadIdx.x] : (T)0;
    if (warpIdx == 0) {
        for (int offset = warpSize / 2; offset > 0; offset >>= 1)
            val += __shfl_down_sync(0xffffffff, val, offset);
    }

    // Broadcast scale
    if (threadIdx.x == 0) warp_sums[0] = (val > (T)0) ? (tsum / val) : (T)0;
    __syncthreads();

    // Phase 2: multiply
    T scale = warp_sums[0];
    if (scale > (T)0) {
        for (IdxT i = start + threadIdx.x; i < end; i += blockDim.x)
            data[i] *= scale;
    }
}

// ---------- masked_sum_major ----------
// One block per row. Computes sum skipping masked genes.
template <typename T, typename IdxT>
__global__ void masked_sum_major_kernel(const IdxT* __restrict__ indptr,
                                        const IdxT* __restrict__ indices,
                                        const T* __restrict__ data,
                                        const bool* __restrict__ gene_mask,
                                        T* __restrict__ sums, int major) {
    int major_idx = blockIdx.x;
    if (major_idx >= major) return;

    IdxT start_idx = indptr[major_idx];
    IdxT stop_idx = indptr[major_idx + 1];

    T val = (T)0;
    for (IdxT i = start_idx + threadIdx.x; i < stop_idx; i += blockDim.x) {
        if (!gene_mask[indices[i]]) val += data[i];
    }

    // Warp-level reduction
    for (int offset = warpSize / 2; offset > 0; offset >>= 1)
        val += __shfl_down_sync(0xffffffff, val, offset);

    // Cross-warp reduction
    int nWarps = blockDim.x >> 5;
    __shared__ T warp_sums[8];
    int laneIdx = threadIdx.x & 31;
    int warpIdx = threadIdx.x >> 5;
    if (laneIdx == 0) warp_sums[warpIdx] = val;
    __syncthreads();

    val = (threadIdx.x < nWarps) ? warp_sums[threadIdx.x] : (T)0;
    if (warpIdx == 0) {
        for (int offset = warpSize / 2; offset > 0; offset >>= 1)
            val += __shfl_down_sync(0xffffffff, val, offset);
    }

    if (threadIdx.x == 0) sums[major_idx] = val;
}

// ---------- prescaled_mul_csr ----------
// One block per row. Multiplies each element by a pre-computed per-row scale.
template <typename T, typename IdxT>
__global__ void prescaled_mul_csr_kernel(const IdxT* __restrict__ indptr,
                                         T* __restrict__ data,
                                         const T* __restrict__ scales,
                                         int nrows) {
    int row = blockIdx.x;
    if (row >= nrows) return;

    T scale = scales[row];
    IdxT start = indptr[row];
    IdxT end = indptr[row + 1];

    for (IdxT i = start + threadIdx.x; i < end; i += blockDim.x)
        data[i] *= scale;
}

// ---------- prescaled_mul_dense ----------
// One block per row. Multiplies each element by a pre-computed per-row scale.
template <typename T>
__global__ void prescaled_mul_dense_kernel(T* __restrict__ data,
                                           const T* __restrict__ scales,
                                           int nrows, int ncols) {
    int row = blockIdx.x;
    if (row >= nrows) return;

    T scale = scales[row];
    int row_offset = row * ncols;
    for (int i = threadIdx.x; i < ncols; i += blockDim.x)
        data[row_offset + i] *= scale;
}
