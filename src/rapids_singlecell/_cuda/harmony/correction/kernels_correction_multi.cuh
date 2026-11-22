#pragma once

#include <cuda_runtime.h>

#include <stdint.h>
#include <type_traits>

#include "../storage.cuh"

// The regression systems (solved by solve_systems, correction_batched.cu) are
// assembled in double from exact sums:
// the Gram matrices from the clustering's int64 joint counts (every entry,
// the intercept and the diagonal included, from the same counts, so the rows
// stay consistent and a small ridge lambda does not lose the solution to
// rounding), the right-hand sides from exact per-level sums of R^T X.

// The ridge penalties of a chunk's clusters (rows of stride ld, those of
// lambda_kb): a category is active unless its penalty suppresses it (a pruned
// batch).
template <typename T>
struct Penalties {
    const T* lambda;
    int ld;
    T suppress;
    __device__ T operator()(int batch, int cluster) const {
        return lambda[(size_t)batch * ld + cluster];
    }
    __device__ bool active(int batch, int cluster) const {
        return (*this)(batch, cluster) < suppress;
    }
};

// Exact joint counts of a chunk's clusters (counts: J x k int64, the
// clustering's fixed point) summed into the Gram matrices, as int64 in the
// matrices' own storage: the diagonal of every level and, above it, the
// cross-key entries of every pair of levels of a joint category. Integer
// atomics: the same bits in any order. finish_gram_kernel converts them.
__global__ void count_blocks_kernel(
    const unsigned long long* __restrict__ counts,
    const int* __restrict__ joint_cats, int n_covariates, int J, int k, int nb1,
    unsigned long long* __restrict__ gram) {
    long long total = (long long)J * k;
    for (long long e = blockIdx.x * (long long)blockDim.x + threadIdx.x;
         e < total; e += (long long)gridDim.x * blockDim.x) {
        unsigned long long v = counts[e];
        if (!v) continue;
        const int* levels = joint_cats + (e / k) * n_covariates;
        unsigned long long* g = gram + (e % k) * (size_t)nb1 * nb1;
        for (int c = 0; c < n_covariates; ++c) {
            int b = levels[c] + 1;
            atomicAdd(g + (size_t)b * nb1 + b, v);
            for (int d = c + 1; d < n_covariates; ++d)  // levels[d] > levels[c]
                atomicAdd(g + (size_t)b * nb1 + levels[d] + 1, v);
        }
    }
}

// One block per cluster: the counted Gram matrix as double (count_inv: the
// fixed point's unit), symmetric, with the ridge on the diagonal, the
// intercept row and column from the level counts (the first key's levels
// hold every cell once) and an isolated unit diagonal for an inactive
// (pruned) level, which omits it from the solve. A cluster without cells
// gets a unit pivot (W = 0).
template <typename T>
__global__ void finish_gram_kernel(Penalties<T> lambda, double count_inv,
                                   int n_first, int n_batches, int k,
                                   double* __restrict__ gram) {
    int nb1 = n_batches + 1, cluster = blockIdx.x;
    if (cluster >= k) return;
    double* g = gram + (size_t)cluster * nb1 * nb1;
    auto u = reinterpret_cast<unsigned long long*>(g);
    unsigned long long total = 0;
    for (int b = 1 + threadIdx.x; b <= n_first; b += blockDim.x)
        total += u[(size_t)b * nb1 + b];
    total = block_sum(total);
    __syncthreads();  // every count read before any entry becomes double
    for (size_t e = threadIdx.x; e < (size_t)nb1 * nb1; e += blockDim.x) {
        int r = (int)(e / nb1), c = (int)(e % nb1);
        if (r == 0 && c == 0) {
            g[0] = total ? (double)total * count_inv : 1.0;
        } else if (r == 0 || c == 0) {
            continue;  // written with the diagonal
        } else if (r == c) {
            bool active = lambda.active(r - 1, cluster);
            double count = (double)u[e] * count_inv;
            g[e] = active ? count + lambda(r - 1, cluster) : 1.0;
            g[r] = g[(size_t)r * nb1] = active ? count : 0.0;
        } else if (r < c) {
            double v =
                lambda.active(r - 1, cluster) && lambda.active(c - 1, cluster)
                    ? (double)u[e] * count_inv
                    : 0.0;
            g[e] = g[(size_t)c * nb1 + r] = v;
        }
    }
}

// The right-hand sides' intercept row (all cells) as the sum of the first
// key's level rows, in order; then zeros in the rows of inactive levels.
template <typename T>
__global__ void finish_rhs_kernel(Penalties<T> lambda, int n_first,
                                  int n_batches, int n_pcs, int k,
                                  double* __restrict__ rhs) {
    const int nb1 = n_batches + 1;
    for (size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
         i < (size_t)k * n_pcs; i += (size_t)gridDim.x * blockDim.x) {
        const int cluster = (int)(i / n_pcs);
        double* r = rhs + (size_t)cluster * nb1 * n_pcs + i % n_pcs;
        double total = 0.0;
        for (int b = 0; b < n_first; ++b) total += r[(size_t)(b + 1) * n_pcs];
        r[0] = total;
        for (int b = 0; b < n_batches; ++b)
            if (!lambda.active(b, cluster)) r[(size_t)(b + 1) * n_pcs] = 0.0;
    }
}

// The coefficients applied per level: table[b][k][:] = W[k][b + 1][:]; the
// intercept W[k][0][:] stays in the embedding, as in Harmony.
template <typename T>
__global__ void level_tables_kernel(const double* __restrict__ W, int k,
                                    int nb1, int n_pcs, T* __restrict__ table) {
    int n_batches = nb1 - 1;
    size_t total = (size_t)n_batches * k * n_pcs;
    for (size_t idx = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
         idx < total; idx += (size_t)blockDim.x * gridDim.x) {
        int pc = (int)(idx % n_pcs), cluster = (int)(idx / n_pcs % k);
        int b = (int)(idx / ((size_t)n_pcs * k));
        table[idx] = (T)W[((size_t)cluster * nb1 + b + 1) * n_pcs + pc];
    }
}
