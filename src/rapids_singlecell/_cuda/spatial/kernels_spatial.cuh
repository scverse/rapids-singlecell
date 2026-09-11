#pragma once

#include <math_constants.h>

template <typename T>
__device__ T distance_hypot(T a, T b);
template <>
__device__ float distance_hypot(float a, float b) {
    return hypotf(a, b);
}
template <>
__device__ double distance_hypot(double a, double b) {
    return hypot(a, b);
}

// The tree is a left-balanced binary heap. Splits cycle through coordinate
// axes by depth. Remembering the previously visited node permits a depth-first
// traversal without a per-thread stack, even for coincident points.
template <typename T, bool radius_search, bool fill>
__global__ void spatial_tree_search(const T* points, const T* tree,
                                    const long long* index, const long long n,
                                    const int dimensions, const int k,
                                    const double radius,
                                    const long long* offsets, long long* counts,
                                    long long* rows, long long* columns,
                                    T* distances) {
    const long long row = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (row >= n) return;

    long long count = 0;
    const bool check_overflow = radius_search && sizeof(T) == sizeof(float) &&
                                radius > CUDART_MAX_NORMAL_F;
    // Keep small 2D candidate lists local instead of repeatedly writing
    // outputs.
    const bool local_topk = !radius_search && dimensions == 2 && k <= 8;
    long long* output_columns = columns;
    T* output_distances = distances;
    long long local_columns[8];
    T local_distances[8];
    if (!radius_search) {
        columns += row * k;
        distances += row * k;
        if (local_topk) {
            columns = local_columns;
            distances = local_distances;
        }
        for (int j = 0; j < k; ++j) {
            columns[j] = -1;
            distances[j] = CUDART_INF;
        }
    }

    long long node = 0, previous = -1;
    int depth = 0;
    while (node >= 0) {
        const long long parent = node == 0 ? -1 : (node - 1) / 2;
        const int axis = depth % dimensions;
        const T delta =
            points[row * dimensions + axis] - tree[node * dimensions + axis];
        double split_distance = fabs(delta);
        if (check_overflow && !isfinite(delta))
            split_distance = fabs((double)points[row * dimensions + axis] -
                                  (double)tree[node * dimensions + axis]);
        const long long near = 2 * node + 1 + (delta > 0);
        const long long far = 2 * node + 1 + (delta <= 0);
        long long next = parent;

        if (previous == parent) {
            const long long column = index[node];
            if (column != row) {
                T distance =
                    fabs(points[row * dimensions] - tree[node * dimensions]);
                for (int d = 1; d < dimensions; ++d) {
                    distance = distance_hypot(distance,
                                              points[row * dimensions + d] -
                                                  tree[node * dimensions + d]);
                }
                if constexpr (radius_search && !fill) {
                    if (check_overflow && !isfinite(distance)) {
                        double precise_distance = 0;
                        for (int d = 0; d < dimensions; ++d)
                            precise_distance =
                                hypot(precise_distance,
                                      (double)points[row * dimensions + d] -
                                          (double)tree[node * dimensions + d]);
                        if (precise_distance <= radius) {
                            // Reject unrepresentable edges before allocating
                            // outputs.
                            counts[row] = -1;
                            return;
                        }
                    }
                }
                if (radius_search) {
                    if (distance <= radius) {
                        if (fill) {
                            const long long position = offsets[row] + count;
                            rows[position] = row;
                            columns[position] = column;
                            distances[position] = distance;
                        }
                        ++count;
                    }
                } else if (columns[k - 1] < 0 || distance < distances[k - 1] ||
                           (distance == distances[k - 1] &&
                            column < columns[k - 1])) {
                    int j = k - 1;
                    while (j > 0 &&
                           (columns[j - 1] < 0 || distance < distances[j - 1] ||
                            (distance == distances[j - 1] &&
                             column < columns[j - 1]))) {
                        columns[j] = columns[j - 1];
                        distances[j] = distances[j - 1];
                        --j;
                    }
                    columns[j] = column;
                    distances[j] = distance;
                }
            }
            if (near < n)
                next = near;
            else if (far < n && split_distance <=
                                    (radius_search ? radius : distances[k - 1]))
                next = far;
        } else if (previous == near && far < n &&
                   split_distance <=
                       (radius_search ? radius : distances[k - 1]))
            next = far;

        depth += next == parent ? -1 : 1;
        previous = node;
        node = next;
    }
    if (radius_search && !fill) counts[row] = count;
    if (local_topk) {
        for (int j = 0; j < k; ++j) {
            output_columns[row * k + j] = columns[j];
            output_distances[row * k + j] = distances[j];
        }
    }
}

template <typename T, typename R, typename C>
__global__ void spatial_edge_distances(const T* coords, const R* rows,
                                       const C* columns, T* distances,
                                       long long n_edges, int dimensions) {
    const long long i = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n_edges) return;
    const long long row = (long long)rows[i] * dimensions;
    const long long column = (long long)columns[i] * dimensions;
    T distance = fabs(coords[row] - coords[column]);
    for (int d = 1; d < dimensions; ++d)
        distance =
            distance_hypot(distance, coords[row + d] - coords[column + d]);
    distances[i] = distance;
}
