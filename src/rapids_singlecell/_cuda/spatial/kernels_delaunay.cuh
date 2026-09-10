#pragma once

#include <cuda_runtime.h>

// Explicit rounding keeps the predicates independent of compiler FMA settings.
__device__ bool positive_orientation(
    const double* p, int a, int b, int c, double* area
) {
    double x = (p[2*a] - p[2*c]) * (p[2*b+1] - p[2*c+1]);
    double y = (p[2*a+1] - p[2*c+1]) * (p[2*b] - p[2*c]);
    *area = __dsub_rn(x, y);
    return *area > 1.4210854715202004e-14 * (fabs(x) + fabs(y));
}

extern "C" __global__ void validate_triangles(
    const double* points, const int* tri, const int* opp,
    int n, int nt, double tolerance, int* invalid, int* seen,
    int* next_boundary, int* prev_boundary, int* boundary_count
) {
    int t = blockIdx.x * blockDim.x + threadIdx.x;
    if (t >= nt) return;
    int a = tri[3*t], b = tri[3*t+1], c = tri[3*t+2];
    if (a < 0 || a >= n || b < 0 || b >= n || c < 0 || c >= n) {
        atomicExch(invalid, 1);
        return;
    }
    double area;
    if (!positive_orientation(points, a, b, c, &area)) {
        atomicExch(invalid, 1);
        return;
    }
    for (int vi = 0; vi < 3; ++vi) {
        int encoded = opp[3*t+vi];
        int other = encoded >> 4;
        int ov = encoded & 3;
        int u = tri[3*t+(vi+1)%3], v = tri[3*t+(vi+2)%3];
        double dx = points[2*u] - points[2*v];
        double dy = points[2*u+1] - points[2*v+1];
        if (__dadd_rn(dx*dx, dy*dy) <= 3.552713678800501e-15)
            atomicExch(invalid, 1);
        if (encoded >= 0) {
            if (other < 0 || other >= nt || ov > 2 || other == t) {
                atomicExch(invalid, 1);
                continue;
            }
            int reverse = opp[3*other+ov];
            if ((reverse >> 4) != t || (reverse & 3) != vi
                || tri[3*other+(ov+1)%3] != v
                || tri[3*other+(ov+2)%3] != u) {
                atomicExch(invalid, 1);
                continue;
            }
            if (t > other) continue;
            int d = tri[3*other+ov];
            if (d < 0 || d >= n) {
                atomicExch(invalid, 1);
                continue;
            }
            double ax = points[2*a] - points[2*d];
            double ay = points[2*a+1] - points[2*d+1];
            double bx = points[2*b] - points[2*d];
            double by = points[2*b+1] - points[2*d+1];
            double cx = points[2*c] - points[2*d];
            double cy = points[2*c+1] - points[2*d+1];
            double alift = __dadd_rn(ax*ax, ay*ay);
            double blift = __dadd_rn(bx*bx, by*by);
            double clift = __dadd_rn(cx*cx, cy*cy);
            double det = __dadd_rn(
                __dadd_rn(alift * __dsub_rn(bx*cy, by*cx),
                          blift * __dsub_rn(cx*ay, cy*ax)),
                clift * __dsub_rn(ax*by, ay*bx));
            double permanent = __dadd_rn(
                __dadd_rn(alift * __dadd_rn(fabs(bx*cy), fabs(by*cx)),
                          blift * __dadd_rn(fabs(cx*ay), fabs(cy*ax))),
                clift * __dadd_rn(fabs(ax*by), fabs(ay*bx)));
            // The relative term bounds local arithmetic error; the global
            // term screens near-coplanar lifted faces that Qhull may merge.
            double bound = __dadd_rn(
                1.4210854715202004e-14 * permanent, tolerance*area);
            if (!(det < -bound)) atomicExch(invalid, 1);
        } else {
            if (atomicCAS(next_boundary+u, -1, v) != -1
                || atomicCAS(prev_boundary+v, -1, u) != -1)
                atomicExch(invalid, 1);
            atomicAdd(boundary_count, 1);
        }
        atomicAdd(seen+u, 1);
        atomicAdd(seen+v, 1);
    }
}

extern "C" __global__ void validate_boundary(
    const double* points, int n, int nt, const int* seen,
    const int* next_boundary, const int* prev_boundary,
    const int* boundary_count, int* invalid
) {
    int v = blockIdx.x * blockDim.x + threadIdx.x;
    if (v >= n) return;
    int nb = *boundary_count;
    if (nb < 3 || nb > 4096 || nt != 2*n - nb - 2) {
        atomicExch(invalid, 1);
        return;
    }
    if (!seen[v]) atomicExch(invalid, 1);
    int before = prev_boundary[v], after = next_boundary[v];
    if (before < 0 && after < 0) return;
    double area;
    if (before < 0 || after < 0 || before >= n || after >= n
        || next_boundary[before] != v || prev_boundary[after] != v
        || !positive_orientation(points, before, v, after, &area)) {
        atomicExch(invalid, 1);
        return;
    }
    // Every boundary vertex must belong to one complete convex cycle.
    // Checking all boundary vertices against each edge rules out star-shaped
    // self intersections that positive local turns alone would not detect.
    int cursor = after, steps = 0;
    while (cursor != v && steps < nb) {
        if (cursor < 0 || cursor >= n
            || (cursor != after
                && !positive_orientation(points, v, after, cursor, &area))) {
            atomicExch(invalid, 1);
            return;
        }
        cursor = next_boundary[cursor];
        ++steps;
    }
    if (cursor != v || steps != nb - 1) atomicExch(invalid, 1);
}

extern "C" __global__ void fill_edges(
    const int* tri, const int* opp, int nt, int* cursor, int* rows, int* cols
) {
    int t = blockIdx.x * blockDim.x + threadIdx.x;
    if (t >= nt) return;
    for (int vi = 0; vi < 3; ++vi) {
        int other = opp[3*t+vi] >> 4;
        if (other >= 0 && t > other) continue;
        int u = tri[3*t+(vi+1)%3], v = tri[3*t+(vi+2)%3];
        int forward = atomicAdd(cursor+u, 1);
        int reverse = atomicAdd(cursor+v, 1);
        rows[forward] = u;
        cols[forward] = v;
        rows[reverse] = v;
        cols[reverse] = u;
    }
}
