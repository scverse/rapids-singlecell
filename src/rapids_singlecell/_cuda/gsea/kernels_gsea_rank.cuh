#pragma once

// Adapted from Numba's numba/misc/quicksort.py to preserve its tie ordering.
#include <cuda_runtime.h>
#include <cub/block/block_reduce.cuh>
#include <cub/block/block_scan.cuh>
#include <cstddef>
#include <cstdint>

namespace gsea_rank {

constexpr int THREADS = 128;
constexpr int INSERTION_CUTOFF = 15;
constexpr int ZERO_CHAIN_CUTOFF = 256;
using Scan = cub::BlockScan<int, THREADS>;
using Reduce = cub::BlockReduce<int, THREADS>;

struct Work {
    union {
        Scan::TempStorage scan;
        Reduce::TempStorage reduce;
    } temp;
    int2 ranges[THREADS];  // Inclusive lower/upper bounds of disjoint ranges.
    int2 metadata;         // Partition result, or temporary row reduction.
};

template <typename Value>
__device__ void exchange(Value& left, Value& right) {
    const Value temporary = left;
    left = right;
    right = temporary;
}

template <typename Index>
__device__ float prepare_pivot(const float* values, Index* order, int low,
                               int high) {
    const int middle = low + (high - low) / 2;
    if (values[order[middle]] > values[order[low]]) {
        exchange(order[low], order[middle]);
    }
    if (values[order[high]] > values[order[middle]]) {
        exchange(order[high], order[middle]);
    }
    if (values[order[middle]] > values[order[low]]) {
        exchange(order[low], order[middle]);
    }
    exchange(order[high], order[middle]);
    return values[order[high]];
}

template <typename Index>
__device__ int partition(const float* values, Index* order, int low, int high,
                         float pivot) {
    int left = low;
    int right = high - 1;
    while (true) {
        while (left < high && values[order[left]] > pivot) {
            ++left;
        }
        while (right >= low && pivot > values[order[right]]) {
            --right;
        }
        if (left >= right) {
            break;
        }
        exchange(order[left], order[right]);
        ++left;
        --right;
    }
    exchange(order[left], order[high]);
    return left;
}

template <typename Index>
__device__ void quicksort(const float* values, Index* order, int2 range) {
    // Processing smaller children first bounds the int32-sized stack at 32.
    int2 stack[32];
    stack[0] = range;
    int pending_ranges = 1;
    while (pending_ranges) {
        range = stack[--pending_ranges];
        while (range.y - range.x >= INSERTION_CUTOFF) {
            const int pivot =
                partition(values, order, range.x, range.y,
                          prepare_pivot(values, order, range.x, range.y));
            int2 left = make_int2(range.x, pivot - 1);
            int2 right = make_int2(pivot + 1, range.y);
            if (left.y - left.x > right.y - right.x) {
                exchange(left, right);
            }
            stack[pending_ranges++] = right;
            range = left;
        }
        for (int position = range.x + 1; position <= range.y; ++position) {
            const Index gene = order[position];
            int insertion = position;
            while (insertion > range.x &&
                   values[gene] > values[order[insertion - 1]]) {
                order[insertion] = order[insertion - 1];
                --insertion;
            }
            order[insertion] = gene;
        }
    }
}

template <typename Index>
__device__ void sort_order(const float* values, Index* order, int n_columns,
                           Work& work) {
    if (threadIdx.x == 0) {
        work.ranges[0] = make_int2(0, n_columns - 1);
    }
    __syncthreads();
    // Disjoint children preserve every swap regardless of scheduling order.
    for (int n_ranges = 1; n_ranges < THREADS; n_ranges *= 2) {
        if (threadIdx.x < n_ranges) {
            const int2 range = work.ranges[threadIdx.x];
            int2 right = make_int2(0, -1);
            if (range.y - range.x >= INSERTION_CUTOFF) {
                const int pivot =
                    partition(values, order, range.x, range.y,
                              prepare_pivot(values, order, range.x, range.y));
                work.ranges[threadIdx.x].y = pivot - 1;
                right = make_int2(pivot + 1, range.y);
            }
            work.ranges[threadIdx.x + n_ranges] = right;
        }
        __syncthreads();
    }
    quicksort(values, order, work.ranges[threadIdx.x]);
    __syncthreads();
}

// Return the pivot position and zero count excluding the pivot (-1 for a
// positive pivot). The row must be nonnegative for cooperative zero swaps.
template <typename Index>
__device__ int2 zero_partition(const float* values, Index* order, int n_columns,
                               int* scratch, Work& work) {
    if (threadIdx.x == 0) {
        const float pivot = prepare_pivot(values, order, 0, n_columns - 1);
        work.metadata = make_int2(
            pivot == 0 ? n_columns - 1
                       : partition(values, order, 0, n_columns - 1, pivot),
            -1);
        work.ranges[0].x = pivot == 0;  // The range queue is unused here.
    }
    __syncthreads();
    if (work.ranges[0].x) {
        const int lane = threadIdx.x % 32;
        const int warp = threadIdx.x / 32;
        const int n_words = (n_columns - 1 + 31) / 32;
        auto* masks = reinterpret_cast<unsigned*>(scratch);
        int* prefixes = scratch + n_words;
        // Capture original zero positions before swaps change right endpoints.
        for (int word = warp; word < n_words; word += THREADS / 32) {
            const int position = word * 32 + lane;
            const unsigned mask =
                __ballot_sync(0xffffffffu, position < n_columns - 1 &&
                                               values[order[position]] == 0);
            if (lane == 0) {
                masks[word] = mask;
            }
        }
        __syncthreads();
        const int words_per_thread = (n_words + THREADS - 1) / THREADS;
        const int begin = threadIdx.x * words_per_thread;
        const int end = min(n_words, begin + words_per_thread);
        int local_zeros = 0, zero_prefix, total_zeros;
        for (int word = begin; word < end; ++word) {
            local_zeros += __popc(masks[word]);
        }
        Scan(work.temp.scan)
            .ExclusiveSum(local_zeros, zero_prefix, total_zeros);
        __syncthreads();
        for (int word = begin; word < end; ++word) {
            prefixes[word] = zero_prefix;
            zero_prefix += __popc(masks[word]);
        }
        __syncthreads();
        int pivot_stop = n_columns - 1;
        // The kth zero swaps with high-1-k until the scans cross. The earliest
        // original zero or newly zero right endpoint determines the pivot.
        for (int word = warp; word < n_words; word += THREADS / 32) {
            const int position = word * 32 + lane;
            const unsigned mask = masks[word];
            const unsigned lower_lanes = (1u << lane) - 1u;
            if (mask & (1u << lane)) {
                const int right =
                    n_columns - 2 - prefixes[word] - __popc(mask & lower_lanes);
                pivot_stop = min(pivot_stop, max(position, right));
                if (position < right) {
                    exchange(order[position], order[right]);
                }
            }
        }
        const int pivot = Reduce(work.temp.reduce)
                              .Reduce(pivot_stop, [](int left, int right) {
                                  return min(left, right);
                              });
        __syncthreads();
        if (threadIdx.x == 0) {
            exchange(order[pivot], order[n_columns - 1]);
            work.metadata = make_int2(pivot, total_zeros);
        }
    }
    __syncthreads();
    const int2 result = work.metadata;
    __syncthreads();
    return result;
}

template <typename Index>
__device__ void sort_equal(Index* order, int low, int high, int* scratch) {
    if (high - low < INSERTION_CUTOFF) {
        return;
    }
    // An equal partition swaps middle/end, reverses [low,end), then swaps
    // middle/end again. Follow each position through this balanced tree;
    // the middle stays fixed and insertion sorting leaves equal ties alone.
    for (int position = low + threadIdx.x; position <= high;
         position += THREADS) {
        int destination = position, range_low = low, range_high = high;
        while (range_high - range_low >= INSERTION_CUTOFF) {
            const int middle = range_low + (range_high - range_low) / 2;
            const int reflected_middle = range_low + range_high - 1 - middle;
            if (destination == middle) {
                break;
            }
            if (destination == range_high) {
                destination =
                    reflected_middle == middle ? range_high : reflected_middle;
            } else if (destination == reflected_middle) {
                destination = range_high;
            } else {
                destination = range_low + range_high - 1 - destination;
            }
            if (destination < middle) {
                range_high = middle - 1;
            } else {
                range_low = middle + 1;
            }
        }
        // Scatter through unused rank output without overwriting unread order.
        scratch[destination] = order[position];
    }
    __syncthreads();
    for (int position = low + threadIdx.x; position <= high;
         position += THREADS) {
        order[position] = scratch[position];
    }
    __syncthreads();
}

template <typename Index>
__device__ void sort_zero_chain(const float* values, Index* order,
                                int n_columns, int* scratch, Work& work) {
    int low = 0, high = n_columns - 1;
    while (high - low + 1 >= ZERO_CHAIN_CUTOFF) {
        const int2 split =
            zero_partition(values, order + low, high - low + 1, scratch, work);
        // The pivot position is relative to this slice.
        // Entries in order still refer to original genes.
        const int pivot = low + split.x;
        if (split.y >= 0) {
            // Sort the all-zero right child, then continue on the left.
            sort_equal(order, pivot + 1, high, scratch);
            if (split.y == high - low) {
                sort_equal(order, low, pivot - 1, scratch);
                return;
            }
            high = pivot - 1;
        } else {
            // A positive pivot leaves every zero on the right.
            sort_order(values, order + low, split.x, work);
            low = pivot + 1;
        }
    }
    sort_order(values, order + low, high - low + 1, work);
}

template <typename Index>
__global__ void rank_kernel(const float* values, size_t n_rows, int n_columns,
                            int* order, int* ranks) {
    extern __shared__ int shared_data[];
    __shared__ Work work;
    for (size_t row = blockIdx.x; row < n_rows; row += gridDim.x) {
        Index* row_order;
        if constexpr (sizeof(Index) == 2) {
            row_order = reinterpret_cast<Index*>(shared_data);
        } else {
            row_order = order + row * n_columns;
        }
        const float* row_values = values + row * n_columns;
        bool finite_nonnegative = true;
        int zero_count = 0;
        for (size_t gene = threadIdx.x; gene < size_t(n_columns);
             gene += THREADS) {
            row_order[gene] = Index(gene);
            if constexpr (sizeof(Index) == 2) {
                const unsigned bits = __float_as_uint(row_values[gene]);
                const bool zero = (bits & 0x7fffffffu) == 0;
                finite_nonnegative =
                    finite_nonnegative &&
                    (bits < 0x7f800000u || bits == 0x80000000u);
                zero_count += zero;
            }
        }
        bool sparse = false;
        if constexpr (sizeof(Index) == 2) {
            const bool nonnegative = __syncthreads_and(finite_nonnegative);
            const int total_zeros = Reduce(work.temp.reduce).Sum(zero_count);
            if (threadIdx.x == 0) {
                work.metadata.x = total_zeros;
            }
            __syncthreads();
            sparse = nonnegative && work.metadata.x > n_columns / 2;
            __syncthreads();
        }
        if (sparse) {
            sort_zero_chain(row_values, row_order, n_columns,
                            ranks + row * n_columns, work);
        } else {
            sort_order(row_values, row_order, n_columns, work);
        }
        for (size_t position = threadIdx.x; position < size_t(n_columns);
             position += THREADS) {
            const int gene = row_order[position];
            if constexpr (sizeof(Index) == 2) {
                order[row * n_columns + position] = gene;
            }
            ranks[row * n_columns + gene] = position;
        }
        __syncthreads();
    }
}

}  // namespace gsea_rank
