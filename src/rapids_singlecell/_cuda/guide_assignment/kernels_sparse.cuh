#pragma once

#include <cuda_runtime.h>

#include <climits>

namespace guide_assignment {

// Column-wise access to CSC input: entries j in [begin, end) of column g.
template <typename T, typename IdxT>
struct CscColumns {
    const IdxT* indptr;
    const IdxT* indices;
    const T* data;
    __device__ long long begin(int g) const {
        return indptr[g];
    }
    __device__ long long end(int g) const {
        return indptr[g + 1];
    }
    __device__ double value(int, long long j) const {
        return data[j];
    }
};

// Column-wise access to dense input of any memory order.
template <typename T>
struct DenseColumns {
    const T* x;
    long long n_rows;
    long long row_stride;
    long long col_stride;
    __device__ long long begin(int) const {
        return 0;
    }
    __device__ long long end(int) const {
        return n_rows;
    }
    __device__ double value(int g, long long j) const {
        return x[j * row_stride + g * col_stride];
    }
};

// Entries of column blockIdx.x handled by this block: chunk blockIdx.y.
template <typename Columns>
__device__ inline void column_chunk(const Columns& columns, long long chunk,
                                    long long& first, long long& last) {
    const int g = blockIdx.x;
    first = columns.begin(g) + blockIdx.y * chunk;
    last = min(columns.end(g), first + chunk);
}

// Counts a positive integer value below `cap` in `bins`; other non-zero values
// go to the tail as guide << 32 | float bits, which sorts by guide, then value.
__device__ inline void count_value(double v, int guide, int cap,
                                   unsigned int* bins,
                                   unsigned long long* n_tail,
                                   unsigned long long* tail,
                                   unsigned long long capacity) {
    if (v == 0.0) {
        return;
    }
    if (v >= 1.0 && v < cap && v == trunc(v)) {
        atomicAdd(&bins[static_cast<int>(v)], 1u);
    } else {
        const unsigned long long pos = atomicAdd(n_tail, 1ull);
        if (pos < capacity) {
            tail[pos] = (static_cast<unsigned long long>(guide) << 32) |
                        __float_as_uint(static_cast<float>(v));
        }
    }
}

// CSR: one thread per stored entry, the guide is the column index.
template <typename T, typename IdxT>
__global__ void value_histogram_csr_kernel(
    const IdxT* __restrict__ indices, const T* __restrict__ data, long long nnz,
    int cap, unsigned int* __restrict__ hist, unsigned long long* n_tail,
    unsigned long long* tail, unsigned long long capacity) {
    for (long long j =
             blockIdx.x * static_cast<long long>(blockDim.x) + threadIdx.x;
         j < nnz; j += static_cast<long long>(gridDim.x) * blockDim.x) {
        const int guide = static_cast<int>(indices[j]);
        count_value(data[j], guide, cap,
                    hist + static_cast<long long>(guide) * cap, n_tail, tail,
                    capacity);
    }
}

// CSC or dense: one block per chunk of a guide column, counted in shared
// memory and added to the guide's histogram.
template <typename Columns>
__global__ void value_histogram_columns_kernel(Columns columns, long long chunk,
                                               int cap,
                                               unsigned int* __restrict__ hist,
                                               unsigned long long* n_tail,
                                               unsigned long long* tail,
                                               unsigned long long capacity) {
    extern __shared__ unsigned int bins[];
    const int guide = blockIdx.x;
    long long first, last;
    column_chunk(columns, chunk, first, last);
    if (first >= last) {
        return;
    }
    for (int i = threadIdx.x; i < cap; i += blockDim.x) {
        bins[i] = 0u;
    }
    __syncthreads();
    for (long long j = first + threadIdx.x; j < last; j += blockDim.x) {
        count_value(columns.value(guide, j), guide, cap, bins, n_tail, tail,
                    capacity);
    }
    __syncthreads();
    for (int i = threadIdx.x; i < cap; i += blockDim.x) {
        if (bins[i]) {
            atomicAdd(&hist[static_cast<long long>(guide) * cap + i], bins[i]);
        }
    }
}

// CSR: one warp per row. Without FILL, counts the entries at or above their
// guide's threshold; with FILL, writes their column indices in order from
// offsets[row]. A NaN threshold assigns nothing.
template <typename T, typename IdxT, bool FILL>
__global__ void assigned_csr_kernel(const IdxT* __restrict__ indptr,
                                    const IdxT* __restrict__ indices,
                                    const T* __restrict__ data,
                                    const double* __restrict__ thresholds,
                                    long long n_rows,
                                    IdxT* __restrict__ offsets,
                                    IdxT* __restrict__ out_indices) {
    const long long row =
        (blockIdx.x * static_cast<long long>(blockDim.x) + threadIdx.x) / 32;
    if (row >= n_rows) {
        return;
    }
    const int lane = threadIdx.x & 31;
    const long long end = indptr[row + 1];
    long long written = FILL ? offsets[row] : 0;
    for (long long base = indptr[row]; base < end; base += 32) {
        const long long j = base + lane;
        const bool hit = j < end && data[j] >= thresholds[indices[j]];
        const unsigned int mask = __ballot_sync(0xffffffff, hit);
        if (FILL && hit) {
            out_indices[written + __popc(mask & ((1u << lane) - 1u))] =
                indices[j];
        }
        written += __popc(mask);
    }
    if (!FILL && lane == 0) {
        offsets[row] = static_cast<IdxT>(written);
    }
}

// CSC: one block per chunk of a guide column, so few long columns still fill
// the GPU. Without FILL, counts the entries at or above the threshold into
// offsets[guide * n_chunks + chunk]; with FILL, writes their row indices in
// order from there.
template <typename T, typename IdxT, bool FILL>
__global__ void assigned_csc_kernel(CscColumns<T, IdxT> columns,
                                    const double* __restrict__ thresholds,
                                    long long chunk, IdxT* __restrict__ offsets,
                                    IdxT* __restrict__ out_indices) {
    __shared__ int warp_hits[32];
    const int guide = blockIdx.x;
    long long first, last;
    column_chunk(columns, chunk, first, last);
    if (first >= last) {
        return;
    }
    const long long slot =
        static_cast<long long>(guide) * gridDim.y + blockIdx.y;
    const double threshold = thresholds[guide];
    const int lane = threadIdx.x & 31;
    const int warp = threadIdx.x >> 5;
    long long written = FILL ? offsets[slot] : 0;
    for (long long base = first; base < last; base += blockDim.x) {
        const long long j = base + threadIdx.x;
        const bool hit = j < last && columns.value(guide, j) >= threshold;
        const unsigned int mask = __ballot_sync(0xffffffff, hit);
        if (lane == 0) {
            warp_hits[warp] = __popc(mask);
        }
        __syncthreads();
        int before = 0;
        int total = 0;
        for (int w = 0; w < static_cast<int>(blockDim.x >> 5); ++w) {
            before += w < warp ? warp_hits[w] : 0;
            total += warp_hits[w];
        }
        if (FILL && hit) {
            out_indices[written + before + __popc(mask & ((1u << lane) - 1u))] =
                columns.indices[j];
        }
        written += total;
        __syncthreads();
    }
    if (!FILL && threadIdx.x == 0) {
        offsets[slot] = static_cast<IdxT>(written);
    }
}

// Order-preserving key of a value: larger values have larger keys, -0.0 and
// 0.0 are equal and every NaN is largest, as for np.argmax. Key 0 is below
// every value.
__device__ inline unsigned long long order_key(double v) {
    if (isnan(v)) {
        return ~0ull;
    }
    const unsigned long long bits = __double_as_longlong(v == 0.0 ? 0.0 : v);
    return (bits >> 63) ? ~bits : (bits | 0x8000000000000000ull);
}

// Inverse of order_key.
__device__ inline double key_value(unsigned long long key) {
    return __longlong_as_double((key >> 63) ? (key & 0x7FFFFFFFFFFFFFFFull)
                                            : ~key);
}

// CSR: one warp per row; the largest stored value and its first guide.
// guides[row] is -1 for rows without stored values.
template <typename T, typename IdxT>
__global__ void max_guide_csr_kernel(const IdxT* __restrict__ indptr,
                                     const IdxT* __restrict__ indices,
                                     const T* __restrict__ data,
                                     long long n_rows,
                                     double* __restrict__ values,
                                     int* __restrict__ guides) {
    const long long row =
        (blockIdx.x * static_cast<long long>(blockDim.x) + threadIdx.x) / 32;
    if (row >= n_rows) {
        return;
    }
    const int lane = threadIdx.x & 31;
    unsigned long long best_key = 0;
    int best_guide = INT_MAX;
    for (long long j = indptr[row] + lane; j < indptr[row + 1]; j += 32) {
        const unsigned long long key = order_key(data[j]);
        const int guide = static_cast<int>(indices[j]);
        if (key > best_key || (key == best_key && guide < best_guide)) {
            best_key = key;
            best_guide = guide;
        }
    }
    for (int offset = 16; offset > 0; offset >>= 1) {
        const unsigned long long key =
            __shfl_down_sync(0xffffffff, best_key, offset);
        const int guide = __shfl_down_sync(0xffffffff, best_guide, offset);
        if (key > best_key || (key == best_key && guide < best_guide)) {
            best_key = key;
            best_guide = guide;
        }
    }
    if (lane == 0) {
        values[row] = best_key ? key_value(best_key) : 0.0;
        guides[row] = best_key ? best_guide : -1;
    }
}

// CSC, first pass: the largest key of every row's stored values.
template <typename T, typename IdxT>
__global__ void max_key_csc_kernel(CscColumns<T, IdxT> columns, long long chunk,
                                   unsigned long long* __restrict__ keys) {
    long long first, last;
    column_chunk(columns, chunk, first, last);
    for (long long j = first + threadIdx.x; j < last; j += blockDim.x) {
        atomicMax(&keys[columns.indices[j]], order_key(columns.data[j]));
    }
}

// CSC, second pass: the first guide holding each row's largest value.
// guides must start at INT_MAX.
template <typename T, typename IdxT>
__global__ void max_guide_csc_kernel(
    CscColumns<T, IdxT> columns, long long chunk,
    const unsigned long long* __restrict__ keys, int* __restrict__ guides) {
    const int guide = blockIdx.x;
    long long first, last;
    column_chunk(columns, chunk, first, last);
    for (long long j = first + threadIdx.x; j < last; j += blockDim.x) {
        const IdxT row = columns.indices[j];
        if (order_key(columns.data[j]) == keys[row]) {
            atomicMin(&guides[row], guide);
        }
    }
}

// CSC, last pass: decode the keys; -1 marks rows without stored values.
__global__ void max_guide_finish_kernel(const unsigned long long* keys,
                                        long long n_rows, double* values,
                                        int* guides) {
    for (long long row =
             blockIdx.x * static_cast<long long>(blockDim.x) + threadIdx.x;
         row < n_rows; row += static_cast<long long>(gridDim.x) * blockDim.x) {
        const unsigned long long key = keys[row];
        values[row] = key ? key_value(key) : 0.0;
        if (!key) {
            guides[row] = -1;
        }
    }
}

}  // namespace guide_assignment
