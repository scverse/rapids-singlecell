#pragma once

#include <math_constants.h>

#include "kernels_predicates.cuh"

// Float filters with explicit rounding (independent of FMA contraction), then
// Shewchuk's exact expansions.
__device__ double orientation(const double* p, int a, int b, int c) {
    double x = (p[2 * a] - p[2 * c]) * (p[2 * b + 1] - p[2 * c + 1]);
    double y = (p[2 * a + 1] - p[2 * c + 1]) * (p[2 * b] - p[2 * c]);
    double area = __dsub_rn(x, y);
    if (fabs(area) > 0x1p-46 * (fabs(x) + fabs(y))) return area;
    return spatial_predicates::orient2dExact(spatial_predicates::constants,
                                             p + 2 * a, p + 2 * b, p + 2 * c);
}

__device__ double incircle(const double* p, int a, int b, int c, int d) {
    double ax = p[2 * a] - p[2 * d], ay = p[2 * a + 1] - p[2 * d + 1];
    double bx = p[2 * b] - p[2 * d], by = p[2 * b + 1] - p[2 * d + 1];
    double cx = p[2 * c] - p[2 * d], cy = p[2 * c + 1] - p[2 * d + 1];
    double al = __dadd_rn(ax * ax, ay * ay), bl = __dadd_rn(bx * bx, by * by);
    double cl = __dadd_rn(cx * cx, cy * cy);
    double det = __dadd_rn(__dadd_rn(al * __dsub_rn(bx * cy, by * cx),
                                     bl * __dsub_rn(cx * ay, cy * ax)),
                           cl * __dsub_rn(ax * by, ay * bx));
    double bound =
        __dadd_rn(__dadd_rn(al * __dadd_rn(fabs(bx * cy), fabs(by * cx)),
                            bl * __dadd_rn(fabs(cx * ay), fabs(cy * ax))),
                  cl * __dadd_rn(fabs(ax * by), fabs(ay * bx)));
    if (fabs(det) > 0x1p-46 * bound) return det;
    // Local binary scaling keeps tiny clusters from underflowing.
    double q[8] = {p[2 * a], p[2 * a + 1], p[2 * b], p[2 * b + 1],
                   p[2 * c], p[2 * c + 1], p[2 * d], p[2 * d + 1]};
    double m = 0;
    for (int i = 0; i < 8; ++i) m = fmax(m, fabs(q[i]));
    int e;
    frexp(m, &e);
    for (int i = 0; i < 8; ++i) q[i] = ldexp(q[i], -e);
    return spatial_predicates::incircleExact(spatial_predicates::constants, q,
                                             q + 2, q + 4, q + 6);
}

// nvcc keeps float32 subnormals, which CuPy's casts flush to zero.
__global__ void delaunay_widen(const float* in, long long n, int shift,
                               double* out) {
    long long i = blockIdx.x * (long long)blockDim.x + threadIdx.x;
    if (i < n) out[i] = ldexp((double)in[i], shift);
}

__global__ void delaunay_orientations(const double* p, const int* tri,
                                      long long nt, long long n, double* out) {
    int t = blockIdx.x * blockDim.x + threadIdx.x;
    if (t >= nt) return;
    int a = tri[3 * t], b = tri[3 * t + 1], c = tri[3 * t + 2];
    out[t] = min(a, min(b, c)) < 0 || max(a, max(b, c)) >= n
                 ? CUDART_NAN
                 : orientation(p, a, b, c);
}

// state: failure bits (1 invalid mesh, 2 illegal edge), boundary edges,
// lowest boundary vertices. Illegal edges vote for CuPy's flip kernels.
__global__ void delaunay_validate_triangles(const double* p, const int* tri,
                                            const int* opp, long long n,
                                            long long nt, int* state, int* seen,
                                            int* after, int* before,
                                            int* votes) {
    int t = blockIdx.x * blockDim.x + threadIdx.x;
    if (t >= nt) return;
    int a = tri[3 * t], b = tri[3 * t + 1], c = tri[3 * t + 2];
    if (min(a, min(b, c)) < 0 || max(a, max(b, c)) >= n ||
        !(orientation(p, a, b, c) > 0)) {
        atomicOr(state, 1);
        return;
    }
    for (int vi = 0; vi < 3; ++vi) {
        int code = opp[3 * t + vi], other = code >> 4, ov = code & 3;
        int u = tri[3 * t + (vi + 1) % 3], v = tri[3 * t + (vi + 2) % 3];
        if (code >= 0) {
            // The neighbor must share the reversed edge and point back.
            if (other >= nt || ov > 2 || other == t ||
                opp[3 * other + ov] >> 4 != t ||
                (opp[3 * other + ov] & 3) != vi ||
                tri[3 * other + (ov + 1) % 3] != v ||
                tri[3 * other + (ov + 2) % 3] != u) {
                atomicOr(state, 1);
                continue;
            }
            if (t > other) continue;
            int d = tri[3 * other + ov];
            if (d < 0 || d >= n) {
                atomicOr(state, 1);
            } else if (incircle(p, a, b, c, d) > 0) {
                // Cocircular quads accept either diagonal.
                atomicOr(state, 2);
                atomicMin(votes + t, t << 2 | vi);
                atomicMin(votes + other, t << 2 | vi);
            }
        } else if (code != -1 || atomicCAS(after + u, -1, v) != -1 ||
                   atomicCAS(before + v, -1, u) != -1) {
            atomicOr(state, 1);
        } else {
            atomicAdd(state + 1, 1);
        }
        atomicAdd(seen + u, 1);
        atomicAdd(seen + v, 1);
    }
}

__device__ bool delaunay_between(double a, double b, double c) {
    return (a < b && b < c) || (a > b && b > c);
}

__device__ bool delaunay_below(const double* a, const double* b) {
    return a[1] < b[1] || (a[1] == b[1] && a[0] < b[0]);
}

__global__ void delaunay_validate_boundary(const double* p, long long n,
                                           const int* seen, const int* after,
                                           const int* before, int* state) {
    int v = blockIdx.x * blockDim.x + threadIdx.x;
    if (v >= n) return;
    int u = before[v], w = after[v];
    if (!seen[v] || (u < 0) != (w < 0)) atomicOr(state, 1);
    if (u < 0 || w < 0) return;
    // Nonnegative turns, straight runs that pass through, and one lowest
    // vertex certify a convex boundary traversed once.
    double turn = orientation(p, u, v, w);
    if (!(turn >= 0) ||
        (turn == 0 && !delaunay_between(p[2 * u], p[2 * v], p[2 * w]) &&
         !delaunay_between(p[2 * u + 1], p[2 * v + 1], p[2 * w + 1])))
        atomicOr(state, 1);
    if (delaunay_below(p + 2 * v, p + 2 * u) &&
        !delaunay_below(p + 2 * w, p + 2 * v))
        atomicAdd(state + 2, 1);
}

__global__ void delaunay_fill_edges(const int* tri, const int* opp,
                                    long long nt, int* cursor, int* rows,
                                    int* cols) {
    int t = blockIdx.x * blockDim.x + threadIdx.x;
    if (t >= nt) return;
    for (int vi = 0; vi < 3; ++vi) {
        int other = opp[3 * t + vi] >> 4;
        if (other >= 0 && t > other) continue;
        int u = tri[3 * t + (vi + 1) % 3], v = tri[3 * t + (vi + 2) % 3];
        int i = atomicAdd(cursor + u, 1), j = atomicAdd(cursor + v, 1);
        rows[i] = cols[j] = u;
        cols[i] = rows[j] = v;
    }
}
