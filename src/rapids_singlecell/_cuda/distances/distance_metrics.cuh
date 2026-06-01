#pragma once

#include <cuda_runtime.h>

// Pluggable per-feature distance policies for the tiled pairwise-distance
// kernels. Each policy accumulates one feature pair's contribution into a
// running scalar, then finalizes it into the distance:
//
//   T s = Dist::init();
//   for (d) Dist::acc(s, a[d], b[d]);
//   dist = Dist::finalize(s);
//
// This fits both the materialize path (write the cost matrix) and the reduce
// path (edistance's sum), and the per-feature shape works inside a
// feature-tiled loop. Single-accumulator metrics only; multi-accumulator
// metrics (cosine, correlation) would need a wider policy interface.
//
// The integer ``Metric`` ids are the ABI between the Python callers and the
// kernel-launch dispatch; keep them in sync with the host side.

namespace distances {

enum Metric : int {
    SQEUCLIDEAN = 0,
    EUCLIDEAN = 1,
    MANHATTAN = 2,
};

template <typename T>
struct SqEuclidean {
    __device__ __forceinline__ static T init() {
        return T(0);
    }
    __device__ __forceinline__ static void acc(T& s, T a, T b) {
        T d = a - b;
        s += d * d;
    }
    __device__ __forceinline__ static T finalize(T s) {
        return s;
    }
};

template <typename T>
struct Euclidean {
    __device__ __forceinline__ static T init() {
        return T(0);
    }
    __device__ __forceinline__ static void acc(T& s, T a, T b) {
        T d = a - b;
        s += d * d;
    }
    __device__ __forceinline__ static T finalize(T s) {
        return sqrt(s);
    }
};

template <typename T>
struct Manhattan {
    __device__ __forceinline__ static T init() {
        return T(0);
    }
    __device__ __forceinline__ static void acc(T& s, T a, T b) {
        T d = a - b;
        s += d < T(0) ? -d : d;
    }
    __device__ __forceinline__ static T finalize(T s) {
        return s;
    }
};

}  // namespace distances
