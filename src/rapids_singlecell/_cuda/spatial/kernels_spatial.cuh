#pragma once

#include <math_constants.h>

#include <climits>
#include <cuda/std/utility>
#include <type_traits>

// Left-balanced subtree bounds, adapted from CuPy's KDTree (MIT license,
// Copyright (c) 2015 Preferred Infrastructure, Inc. and Preferred Networks,
// Inc.).
__device__ long long kd_subtree_begin(int level, long long n, int n_levels,
                                      long long node) {
    const long long offset = node - ((1ll << level) - 1);
    const long long to_left_last = offset << (n_levels - level - 1);
    const long long total_last = n - ((1ll << (n_levels - 1)) - 1);
    return (1ll << level) - 1 + offset * ((1ll << (n_levels - level)) - 1) -
           to_left_last + min(total_last, to_left_last);
}

__device__ long long kd_subtree_size(long long n, int n_levels,
                                     long long node) {
    if (node >= n) return 0;
    const long long half = 1ll << (n_levels - (64 - __clzll(node + 1)));
    return 2 * half - 1 - max(min((node + 2) * half - 1 - n, half), 0ll);
}

// One level of CuPy's KDTree build for all libraries at once. Node tags are
// offset by the library start, so a stable radix sort of (tag, coordinate
// rank) keys per level orders every tree like CuPy's two stable argsorts.
// Level -1 seeds the keys; shorter libraries keep their settled tags.
__global__ void kd_keys(const int* ranks, const int* codes,
                        const long long* segments, int n, int dims, int bits,
                        int level, unsigned long long* keys, int* ids) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    if (level < 0) ids[i] = i;
    const int id = ids[i], library = codes ? codes[id] : 0;
    const long long start = segments[library], local = i - start;
    long long tag = level < 0 ? 0 : (long long)(keys[i] >> bits) - start;
    if (level >= 0 && local >= (1ll << level) - 1) {
        const long long length = segments[library + 1] - start;
        const int n_levels = 64 - __clzll(length);
        const long long left = 2 * tag + 1;
        const long long pivot = kd_subtree_begin(level, length, n_levels, tag) +
                                kd_subtree_size(length, n_levels, left);
        tag = local < pivot ? left : local > pivot ? left + 1 : tag;
    }
    keys[i] = (unsigned long long)(start + tag) << bits |
              (unsigned)ranks[(long long)id * dims + (level + 1) % dims];
}

// Bounding boxes [lower..., upper...] of the subtrees rooted at one level,
// built from the deepest level up.
template <typename T>
__global__ void kd_boxes(const T* points, const int* index, const int* codes,
                         const long long* segments, int n, int dims, int level,
                         T* boxes) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    const int library = codes ? codes[index[i]] : 0;
    const long long start = segments[library], local = i - start;
    if (local < (1ll << level) - 1 || local >= (2ll << level) - 1) return;
    T* box = boxes + 2ll * dims * i;
    for (int d = 0; d < dims; ++d)
        box[d] = box[dims + d] = points[(long long)index[i] * dims + d];
    const long long end = min(start + 2 * local + 3, segments[library + 1]);
    for (long long child = start + 2 * local + 1; child < end; ++child)
        for (int d = 0; d < 2 * dims; ++d) {
            const T other = boxes[2 * dims * child + d];
            box[d] = d < dims ? min(box[d], other) : max(box[d], other);
        }
}

template <typename D, typename T>
__device__ D kd_distance(const T* a, const T* b, int dims) {
    D distance = fabs((D)a[0] - (D)b[0]);
    for (int d = 1; d < dims; ++d)
        distance = hypot(distance, (D)a[d] - (D)b[d]);
    return distance;
}

// Rounding is monotone, so this never exceeds the distance to a point inside
// the box; the slack also covers hypot's final ulp.
template <typename T>
__device__ double kd_box_distance(const T* point, const T* box, int dims) {
    T distance = 0;
    for (int d = 0; d < dims; ++d) {
        const T gap =
            max(max(box[d] - point[d], point[d] - box[dims + d]), T(0));
        distance = d ? hypot(distance, gap) : gap;
    }
    return distance * (1 - 0x1p-20);
}

__device__ bool kd_before(double distance, long long column, double other,
                          long long other_column) {
    return distance < other || (distance == other && column < other_column);
}

// Exact search of each point's library tree. Splits cycle through the axes by
// depth; tracking the previous node gives a stackless depth-first traversal,
// even for coincident points. Split planes and boxes prune far subtrees. kNN
// (MAXK >= 0) keeps the k best (distance, column) pairs sorted, in registers
// behind MAXK - k fixed pads or, for MAXK == 0, in the outputs. Radius search
// (MAXK < 0) counts into offsets[row + 1], then fills from offsets[row]. Edges
// within radii beyond FLT_MAX whose float distance overflows stay infinite
// for callers to reject.
template <typename T, int MAXK, bool fill = false>
__global__ void kd_search(const T* points, const T* tree, const int* index,
                          const T* boxes, const long long* segments,
                          const int* codes, int n_rows, int dims, int k,
                          double radius, long long* offsets, long long* rows,
                          long long* columns, T* distances) {
    const int row = blockIdx.x * blockDim.x + threadIdx.x;
    if (row >= n_rows) return;
    const int library = codes ? codes[row] : 0;
    const long long start = segments[library];
    const long long n = segments[library + 1] - start;
    const T* point = points + (long long)row * dims;
    tree += start * dims;
    index += start;
    boxes += start * 2 * dims;
    constexpr bool knn = MAXK >= 0;
    const bool overflow =
        !knn && sizeof(T) == 4 && radius > CUDART_MAX_NORMAL_F;
    using C = std::conditional_t<(MAXK > 0), int, long long>;
    const int size = MAXK > 0 ? MAXK : k;
    T local_distances[MAXK > 0 ? MAXK : 1];
    C local_columns[MAXK > 0 ? MAXK : 1];
    T* best = MAXK > 0 ? local_distances : distances + (long long)row * k;
    C* best_columns =
        MAXK > 0 ? local_columns : (C*)columns + (long long)row * k;
#pragma unroll
    for (int j = 0; j < (knn ? size : 0); ++j) {
        best[j] = j < size - k ? -CUDART_INF : CUDART_INF;
        best_columns[j] = j < size - k ? -1 : INT_MAX;
    }
    double limit = knn ? CUDART_INF : radius;
    long long count = 0, node = 0, previous = -1;
    for (int depth = 0; node >= 0;) {
        const long long parent = node ? (node - 1) / 2 : -1;
        const T* other = tree + node * dims;
        const int axis = depth % dims;
        const T delta = point[axis] - other[axis];
        const double split = overflow && !isfinite(delta)
                                 ? fabs((double)point[axis] - other[axis])
                                 : fabs(delta);
        const long long near = 2 * node + 1 + (delta > 0);
        const long long far = 2 * node + 1 + (delta <= 0);
        if (previous == parent && index[node] != row) {
            const int column = index[node];
            const T distance = kd_distance<T>(point, other, dims);
            if constexpr (knn) {
                if (kd_before(distance, column, best[size - 1],
                              best_columns[size - 1])) {
                    int slot = size - 1;
                    // Memory shifts, loading columns only on ties; registers
                    // need fixed indices, so they bubble below.
                    if constexpr (MAXK == 0)
                        for (; slot > 0 && (distance < best[slot - 1] ||
                                            (distance == best[slot - 1] &&
                                             column < best_columns[slot - 1]));
                             --slot) {
                            best[slot] = best[slot - 1];
                            best_columns[slot] = best_columns[slot - 1];
                        }
                    best[slot] = distance;
                    best_columns[slot] = column;
#pragma unroll
                    for (int j = MAXK - 1; j > 0; --j)
                        if (kd_before(best[j], best_columns[j], best[j - 1],
                                      best_columns[j - 1])) {
                            cuda::std::swap(best[j], best[j - 1]);
                            cuda::std::swap(best_columns[j],
                                            best_columns[j - 1]);
                        }
                    limit = best[size - 1];
                }
            } else if (distance <= radius ||
                       (overflow && !isfinite(distance) &&
                        kd_distance<double>(point, other, dims) <= radius)) {
                if (fill) {
                    const long long at = offsets[row] + count;
                    rows[at] = row;
                    columns[at] = column;
                    distances[at] = distance;
                }
                ++count;
            }
        }
        long long next = parent;  // Near child first, then far if in reach.
        if (previous == parent && near < n)
            next = near;
        else if (previous != far && far < n && split <= limit &&
                 (overflow || kd_box_distance(point, boxes + far * 2 * dims,
                                              dims) <= limit))
            next = far;
        depth += next == parent ? -1 : 1;
        previous = node;
        node = next;
    }
    if constexpr (knn) {
#pragma unroll
        for (int j = 0; j < size; ++j)
            if (j >= size - k) {
                const long long at = (long long)row * k + j - (size - k);
                rows[at] = row;
                columns[at] = best_columns[j];
                distances[at] = best[j];
            }
    } else if (!fill) {
        offsets[row + 1] = count;
    }
}

template <typename T, typename R, typename C>
__global__ void kd_edge_distances(const T* points, const R* rows,
                                  const C* columns, long long n, int dims,
                                  T* distances) {
    const long long i = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n)
        distances[i] =
            kd_distance<T>(points + (long long)rows[i] * dims,
                           points + (long long)columns[i] * dims, dims);
}
