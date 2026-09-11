#pragma once

#include <cuda_runtime.h>
#include <math_constants.h>
#include "kernels_predicates.cuh"

constexpr double PREDICATE_ERROR_FACTOR = 64 * 0x1p-52;

// Explicit rounding keeps the predicates independent of compiler FMA settings.
__device__ double orientation(const double* p, int a, int b, int c) {
    double x = (p[2 * a] - p[2 * c]) * (p[2 * b + 1] - p[2 * c + 1]);
    double y = (p[2 * a + 1] - p[2 * c + 1]) * (p[2 * b] - p[2 * c]);
    double area = __dsub_rn(x, y);
    if (fabs(area) > PREDICATE_ERROR_FACTOR * (fabs(x) + fabs(y))) return area;
    return spatial_predicates::orient2dExact(spatial_predicates::constants,
                                             p + 2 * a, p + 2 * b, p + 2 * c);
}

__device__ double incircle(const double* points, int a, int b, int c, int d) {
    double ax = points[2 * a] - points[2 * d];
    double ay = points[2 * a + 1] - points[2 * d + 1];
    double bx = points[2 * b] - points[2 * d];
    double by = points[2 * b + 1] - points[2 * d + 1];
    double cx = points[2 * c] - points[2 * d];
    double cy = points[2 * c + 1] - points[2 * d + 1];
    double alift = __dadd_rn(ax * ax, ay * ay);
    double blift = __dadd_rn(bx * bx, by * by);
    double clift = __dadd_rn(cx * cx, cy * cy);
    double det = __dadd_rn(__dadd_rn(alift * __dsub_rn(bx * cy, by * cx),
                                     blift * __dsub_rn(cx * ay, cy * ax)),
                           clift * __dsub_rn(ax * by, ay * bx));
    double permanent =
        __dadd_rn(__dadd_rn(alift * __dadd_rn(fabs(bx * cy), fabs(by * cx)),
                            blift * __dadd_rn(fabs(cx * ay), fabs(cy * ax))),
                  clift * __dadd_rn(fabs(ax * by), fabs(ay * bx)));
    if (fabs(det) <= PREDICATE_ERROR_FACTOR * permanent) {
        // A tiny cluster can underflow degree-four arithmetic after
        // global scaling. Local binary scaling preserves its sign.
        double local[8] = {
            points[2 * a], points[2 * a + 1], points[2 * b], points[2 * b + 1],
            points[2 * c], points[2 * c + 1], points[2 * d], points[2 * d + 1]};
        double magnitude = 0;
        for (int i = 0; i < 8; ++i) magnitude = fmax(magnitude, fabs(local[i]));
        int exponent;
        frexp(magnitude, &exponent);
        for (int i = 0; i < 8; ++i) local[i] = ldexp(local[i], -exponent);
        det = spatial_predicates::incircleExact(spatial_predicates::constants,
                                                local, local + 2, local + 4,
                                                local + 6);
    }
    return det;
}

extern "C" __global__ void delaunay_orientations(const double* points, int n,
                                                 int a, int b, double* output) {
    int c = blockIdx.x * blockDim.x + threadIdx.x;
    if (c < n) output[c] = orientation(points, a, b, c);
}

extern "C" __global__ void delaunay_triangle_orientations(
    const double* points, int n, const int* triangles, int nt, double* output) {
    int t = blockIdx.x * blockDim.x + threadIdx.x;
    if (t >= nt) return;
    int a = triangles[3 * t], b = triangles[3 * t + 1],
        c = triangles[3 * t + 2];
    output[t] = a < 0 || a >= n || b < 0 || b >= n || c < 0 || c >= n
                    ? CUDART_NAN
                    : orientation(points, a, b, c);
}

extern "C" __global__ void delaunay_normalize(const float* points,
                                              long long size, int exponent,
                                              double* output) {
    long long i = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= size) return;
    unsigned int bits = __float_as_uint(points[i]);
    unsigned int fraction = bits & 0x7fffff;
    int biased = (bits >> 23) & 0xff;
    if (biased == 0xff) {
        output[i] = static_cast<double>(points[i]);
        return;
    }
    // Decode the significand as an integer so float32 subnormals cannot be
    // flushed before conversion to double, including under a CuPy FTZ build.
    unsigned int significand = biased ? fraction | 0x800000 : fraction;
    int power = biased ? biased - 150 : -149;
    double value = ldexp(static_cast<double>(significand), power - exponent);
    output[i] = bits >> 31 ? -value : value;
}

extern "C" __global__ void validate_triangles(const double* points,
                                              const int* tri, const int* opp,
                                              int n, int nt, int* invalid,
                                              int* seen, int* next_boundary,
                                              int* prev_boundary,
                                              int* boundary_count) {
    int t = blockIdx.x * blockDim.x + threadIdx.x;
    if (t >= nt) return;
    int a = tri[3 * t], b = tri[3 * t + 1], c = tri[3 * t + 2];
    if (a < 0 || a >= n || b < 0 || b >= n || c < 0 || c >= n) {
        atomicOr(invalid, 1);
        return;
    }
    if (!(orientation(points, a, b, c) > 0)) {
        atomicOr(invalid, 1);
        return;
    }
    for (int vi = 0; vi < 3; ++vi) {
        int encoded = opp[3 * t + vi];
        int other = encoded >> 4;
        int ov = encoded & 3;
        int u = tri[3 * t + (vi + 1) % 3], v = tri[3 * t + (vi + 2) % 3];
        if (encoded >= 0) {
            if (other < 0 || other >= nt || ov > 2 || other == t) {
                atomicOr(invalid, 1);
                continue;
            }
            int reverse = opp[3 * other + ov];
            if ((reverse >> 4) != t || (reverse & 3) != vi ||
                tri[3 * other + (ov + 1) % 3] != v ||
                tri[3 * other + (ov + 2) % 3] != u) {
                atomicOr(invalid, 1);
                continue;
            }
            if (t > other) continue;
            int d = tri[3 * other + ov];
            if (d < 0 || d >= n) {
                atomicOr(invalid, 1);
                continue;
            }
            double det = incircle(points, a, b, c, d);
            // Either diagonal is valid for an exactly cocircular quadrilateral.
            if (!isfinite(det))
                atomicOr(invalid, 1);
            else if (det > 0)
                atomicOr(invalid, 2);
        } else {
            if (encoded != -1) {
                atomicOr(invalid, 1);
                continue;
            }
            if (atomicCAS(next_boundary + u, -1, v) != -1 ||
                atomicCAS(prev_boundary + v, -1, u) != -1)
                atomicOr(invalid, 1);
            atomicAdd(boundary_count, 1);
        }
        atomicAdd(seen + u, 1);
        atomicAdd(seen + v, 1);
    }
}

extern "C" __global__ void validate_boundary(const double* points, int n,
                                             int nt, const int* seen,
                                             const int* next_boundary,
                                             const int* prev_boundary,
                                             int* boundary_count,
                                             int* invalid) {
    int v = blockIdx.x * blockDim.x + threadIdx.x;
    if (v >= n) return;
    int nb = *boundary_count;
    if (nb < 3 || nt != 2 * n - nb - 2) {
        atomicOr(invalid, 1);
        return;
    }
    if (!seen[v]) atomicOr(invalid, 1);
    int before = prev_boundary[v], after = next_boundary[v];
    if (before < 0 && after < 0) return;
    if (before < 0 || after < 0 || before >= n || after >= n ||
        next_boundary[before] != v || prev_boundary[after] != v) {
        atomicOr(invalid, 1);
        return;
    }
    double turn = orientation(points, before, v, after);
    bool between_x = (points[2 * before] < points[2 * v] &&
                      points[2 * v] < points[2 * after]) ||
                     (points[2 * before] > points[2 * v] &&
                      points[2 * v] > points[2 * after]);
    bool between_y = (points[2 * before + 1] < points[2 * v + 1] &&
                      points[2 * v + 1] < points[2 * after + 1]) ||
                     (points[2 * before + 1] > points[2 * v + 1] &&
                      points[2 * v + 1] > points[2 * after + 1]);
    if (!(turn >= 0) || (turn == 0 && !between_x && !between_y)) {
        atomicOr(invalid, 1);
        return;
    }
    // Nonnegative turns and exactly one full rotation certify one convex
    // boundary cycle, including straight runs, without traversing every edge.
    bool incoming_lower = points[2 * v + 1] < points[2 * before + 1] ||
                          (points[2 * v + 1] == points[2 * before + 1] &&
                           points[2 * v] < points[2 * before]);
    bool outgoing_lower = points[2 * after + 1] < points[2 * v + 1] ||
                          (points[2 * after + 1] == points[2 * v + 1] &&
                           points[2 * after] < points[2 * v]);
    if (incoming_lower && !outgoing_lower) atomicAdd(boundary_count + 1, 1);
}

extern "C" __global__ void validate_boundary_winding(const int* boundary_count,
                                                     int* invalid) {
    if (boundary_count[1] != 1) atomicOr(invalid, 1);
}

// Called only after validation reports a valid mesh with illegal interior
// edges.
extern "C" __global__ void exact_flip_votes(const double* points,
                                            const int* tri, const int* opp,
                                            int nt, int* votes) {
    int t = blockIdx.x * blockDim.x + threadIdx.x;
    if (t >= nt) return;
    int a = tri[3 * t], b = tri[3 * t + 1], c = tri[3 * t + 2];
    for (int vi = 0; vi < 3; ++vi) {
        int encoded = opp[3 * t + vi];
        int other = encoded >> 4;
        if (encoded < 0 || t >= other) continue;
        int d = tri[3 * other + (encoded & 3)];
        if (!(incircle(points, a, b, c, d) > 0)) continue;
        int u = tri[3 * t + (vi + 1) % 3], v = tri[3 * t + (vi + 2) % 3];
        int apex = tri[3 * t + vi];
        if (!(orientation(points, apex, u, d) > 0) ||
            !(orientation(points, apex, d, v) > 0))
            continue;
        int vote = (t << 2) | vi;
        atomicMin(votes + t, vote);
        atomicMin(votes + other, vote);
    }
}

extern "C" __global__ void fill_edges(const int* tri, const int* opp, int nt,
                                      int* cursor, int* rows, int* cols) {
    int t = blockIdx.x * blockDim.x + threadIdx.x;
    if (t >= nt) return;
    for (int vi = 0; vi < 3; ++vi) {
        int other = opp[3 * t + vi] >> 4;
        if (other >= 0 && t > other) continue;
        int u = tri[3 * t + (vi + 1) % 3], v = tri[3 * t + (vi + 2) % 3];
        int forward = atomicAdd(cursor + u, 1);
        int reverse = atomicAdd(cursor + v, 1);
        rows[forward] = u;
        cols[forward] = v;
        rows[reverse] = v;
        cols[reverse] = u;
    }
}
