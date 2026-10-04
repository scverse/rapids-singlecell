#pragma once

#include <cuda_runtime.h>

#include <stdint.h>
#include <type_traits>

template <typename T>
__device__ __forceinline__ T warp_sum_multi_correction(T value) {
#pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1)
        value += __shfl_down_sync(0xffffffff, value, offset);
    return value;
}

// Initialize the intercept, intercept/category, and within-covariate diagonal
// entries of every regression Gram matrix.  All other entries are already
// zero.  An inactive category is represented by an isolated unit diagonal,
// which is algebraically equivalent to omitting that category from the solve
// and then restoring a zero coefficient.
template <typename T>
__global__ void initialize_multi_gram_kernel(const T* __restrict__ O,
                                             const T* __restrict__ lambda_kb,
                                             const uint8_t* __restrict__ active,
                                             const T* __restrict__ joint_O,
                                             T* __restrict__ gram,
                                             int n_batches, int n_clusters,
                                             int n_joint_categories) {
    int cluster = blockIdx.x;
    if (cluster >= n_clusters) return;

    int nb1 = n_batches + 1;
    T* cluster_gram = gram + (size_t)cluster * nb1 * nb1;

    T cluster_sum = T(0);
    for (int joint = threadIdx.x; joint < n_joint_categories;
         joint += blockDim.x) {
        cluster_sum += joint_O[(size_t)joint * n_clusters + cluster];
    }
    cluster_sum = warp_sum_multi_correction(cluster_sum);

    __shared__ T warp_sums[32];
    int lane = threadIdx.x & 31;
    int warp = threadIdx.x >> 5;
    if (lane == 0) warp_sums[warp] = cluster_sum;
    __syncthreads();

    if (warp == 0) {
        int n_warps = (blockDim.x + 31) >> 5;
        T block_sum = lane < n_warps ? warp_sums[lane] : T(0);
        block_sum = warp_sum_multi_correction(block_sum);
        if (lane == 0) cluster_gram[0] = block_sum;
    }

    for (int batch = threadIdx.x; batch < n_batches; batch += blockDim.x) {
        int row = batch + 1;
        size_t bk = (size_t)batch * n_clusters + cluster;
        if (active[bk] != 0) {
            T observed = O[bk];
            cluster_gram[row] = observed;
            cluster_gram[(size_t)row * nb1] = observed;
            cluster_gram[(size_t)row * nb1 + row] = observed + lambda_kb[bk];
        } else {
            cluster_gram[(size_t)row * nb1 + row] = T(1);
        }
    }
}

// Add the cross-covariate blocks A diag(q_k) A^T, where each row of
// joint_cats contains the F active marginal levels for one observed joint
// category.  Same-covariate blocks are diagonal and were initialized from O.
template <typename T, int N_COVARIATES>
__global__ void add_joint_cross_kernel(
    const T* __restrict__ joint_O, const int* __restrict__ joint_cats,
    const int* __restrict__ marginal_joint_offsets,
    const int* __restrict__ marginal_joint_indices,
    const uint8_t* __restrict__ active, T* __restrict__ gram, int n_covariates,
    int n_batches, int n_clusters) {
    size_t total = (size_t)n_batches * n_clusters;
    int nb1 = n_batches + 1;

    for (size_t idx = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
         idx < total; idx += (size_t)blockDim.x * gridDim.x) {
        int batch = (int)(idx / n_clusters);
        int cluster = (int)(idx % n_clusters);
        if (active[idx] == 0) continue;
        int row = batch + 1;
        T* cluster_gram = gram + (size_t)cluster * nb1 * nb1;
        // The lower category owns each symmetric pair and visits its joint
        // categories in CSR order. No other thread writes either entry.
        for (int position = marginal_joint_offsets[batch];
             position < marginal_joint_offsets[batch + 1]; ++position) {
            int joint = marginal_joint_indices[position];
            T value = joint_O[(size_t)joint * n_clusters + cluster];
            if (value == T(0)) continue;
            const int* levels =
                joint_cats + (size_t)joint * (N_COVARIATES > 0 ? N_COVARIATES
                                                               : n_covariates);
            int count = N_COVARIATES > 0 ? N_COVARIATES : n_covariates;
#pragma unroll
            for (int other = 0; other < count; ++other) {
                int other_batch = levels[other];
                if (other_batch <= batch ||
                    active[(size_t)other_batch * n_clusters + cluster] == 0)
                    continue;
                int col = other_batch + 1;
                T updated = cluster_gram[(size_t)row * nb1 + col] + value;
                cluster_gram[(size_t)row * nb1 + col] = updated;
                cluster_gram[(size_t)col * nb1 + row] = updated;
            }
        }
    }
}

// Expand joint-category cross-products R_j^T X_j into the regression
// right-hand sides: row 0 (intercept) sums every joint, row b + 1 the joints
// containing active category b (in ascending CSR order, deterministic).
template <typename T>
__global__ void marginal_from_joint_rhs_kernel(
    const T* __restrict__ joint_rhs,
    const int* __restrict__ marginal_joint_offsets,
    const int* __restrict__ marginal_joint_indices,
    const uint8_t* __restrict__ active, T* __restrict__ rhs, int n_pcs,
    int n_clusters, int n_batches, int n_joint_categories) {
    int nb1 = n_batches + 1;
    size_t total = (size_t)n_clusters * nb1 * n_pcs;
    for (size_t idx = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
         idx < total; idx += (size_t)blockDim.x * gridDim.x) {
        int pc = (int)(idx % n_pcs);
        int row = (int)(idx / n_pcs % nb1);
        int cluster = (int)(idx / ((size_t)n_pcs * nb1));
        auto at = [&](int joint) {
            return joint_rhs[((size_t)joint * n_clusters + cluster) * n_pcs +
                             pc];
        };
        T value = T(0);
        if (row == 0) {
            for (int joint = 0; joint < n_joint_categories; ++joint)
                value += at(joint);
        } else if (active[(size_t)(row - 1) * n_clusters + cluster] != 0) {
            for (int position = marginal_joint_offsets[row - 1];
                 position < marginal_joint_offsets[row]; ++position)
                value += at(marginal_joint_indices[position]);
        }
        rhs[idx] = value;
    }
}

// Coefficients per joint category: W_joint[j][k][:] is the sum of the
// marginal rows W_all[k][b + 1][:] over the categories b of joint j. The
// intercept is deliberately retained in the embedding, as in Harmony.
template <typename T>
__global__ void joint_coefficients_kernel(const T* __restrict__ W_all,
                                          const int* __restrict__ joint_cats,
                                          T* __restrict__ W_joint, int n_pcs,
                                          int n_clusters, int n_batches,
                                          int n_covariates,
                                          int n_joint_categories) {
    int nb1 = n_batches + 1;
    size_t total = (size_t)n_joint_categories * n_clusters * n_pcs;
    for (size_t idx = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
         idx < total; idx += (size_t)blockDim.x * gridDim.x) {
        int pc = (int)(idx % n_pcs);
        int cluster = (int)(idx / n_pcs % n_clusters);
        int joint = (int)(idx / ((size_t)n_pcs * n_clusters));
        const int* levels = joint_cats + (size_t)joint * n_covariates;
        T value = T(0);
        for (int c = 0; c < n_covariates; ++c)
            value +=
                W_all[((size_t)cluster * nb1 + levels[c] + 1) * n_pcs + pc];
        W_joint[idx] = value;
    }
}
