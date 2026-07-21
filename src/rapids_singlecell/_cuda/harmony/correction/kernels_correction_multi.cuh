#pragma once

#include <cuda_runtime.h>

#include <stdint.h>
#include <type_traits>

// Accumulate the probability assigned to each observed joint category.  The
// joint categories are used only as a compact factorization of the marginal
// design matrix; the regression coefficients remain marginal.
template <typename T>
__global__ void joint_observed_kernel(const T* __restrict__ R,
                                      const int* __restrict__ joint_codes,
                                      T* __restrict__ joint_O, int n_cells,
                                      int n_clusters) {
    size_t total = (size_t)n_cells * n_clusters;
    for (size_t idx = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
         idx < total; idx += (size_t)blockDim.x * gridDim.x) {
        int cell = (int)(idx / n_clusters);
        int cluster = (int)(idx % n_clusters);
        int joint = joint_codes[cell];
        atomicAdd(&joint_O[(size_t)joint * n_clusters + cluster], R[idx]);
    }
}

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
__global__ void add_joint_cross_kernel(const T* __restrict__ joint_O,
                                       const int* __restrict__ joint_cats,
                                       const uint8_t* __restrict__ active,
                                       T* __restrict__ gram,
                                       int n_joint_categories, int n_covariates,
                                       int n_batches, int n_clusters) {
    size_t total = (size_t)n_joint_categories * n_clusters;
    int nb1 = n_batches + 1;

    for (size_t idx = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
         idx < total; idx += (size_t)blockDim.x * gridDim.x) {
        int joint = (int)(idx / n_clusters);
        int cluster = (int)(idx % n_clusters);
        T value = joint_O[idx];
        if (value == T(0)) continue;

        const int* levels =
            joint_cats +
            (size_t)joint * (N_COVARIATES > 0 ? N_COVARIATES : n_covariates);
        T* cluster_gram = gram + (size_t)cluster * nb1 * nb1;

        if constexpr (N_COVARIATES > 0) {
#pragma unroll
            for (int left = 0; left < N_COVARIATES; ++left) {
                int left_batch = levels[left];
                if (active[(size_t)left_batch * n_clusters + cluster] == 0)
                    continue;
#pragma unroll
                for (int right = left + 1; right < N_COVARIATES; ++right) {
                    int right_batch = levels[right];
                    if (active[(size_t)right_batch * n_clusters + cluster] == 0)
                        continue;
                    int row = left_batch + 1;
                    int col = right_batch + 1;
                    atomicAdd(&cluster_gram[(size_t)row * nb1 + col], value);
                    atomicAdd(&cluster_gram[(size_t)col * nb1 + row], value);
                }
            }
        } else {
            for (int left = 0; left < n_covariates; ++left) {
                int left_batch = levels[left];
                if (active[(size_t)left_batch * n_clusters + cluster] == 0)
                    continue;
                for (int right = left + 1; right < n_covariates; ++right) {
                    int right_batch = levels[right];
                    if (active[(size_t)right_batch * n_clusters + cluster] == 0)
                        continue;
                    int row = left_batch + 1;
                    int col = right_batch + 1;
                    atomicAdd(&cluster_gram[(size_t)row * nb1 + col], value);
                    atomicAdd(&cluster_gram[(size_t)col * nb1 + row], value);
                }
            }
        }
    }
}

// Compute all non-intercept rows of
//   Phi* diag(R[:, k]) X
// in one launch. cat_offsets/cell_indices form a CSR-like marginal category
// index. Every cell occurs once per covariate in cell_indices. A block owns one
// (marginal level, cluster, PC pair), so the reduced result needs no atomics.
template <typename T>
__global__ void segmented_multi_rhs_kernel(const T* __restrict__ X,
                                           const T* __restrict__ R,
                                           const int* __restrict__ cat_offsets,
                                           const int* __restrict__ cell_indices,
                                           const uint8_t* __restrict__ active,
                                           T* __restrict__ rhs, int n_pcs,
                                           int n_clusters, int n_batches) {
    int pc_pairs = (n_pcs + 1) / 2;
    size_t blocks_per_batch = (size_t)n_clusters * pc_pairs;
    size_t linear_block = blockIdx.x;
    int batch = (int)(linear_block / blocks_per_batch);
    if (batch >= n_batches) return;

    size_t remainder = linear_block % blocks_per_batch;
    int cluster = (int)(remainder / pc_pairs);
    int pc0 = (int)(remainder % pc_pairs) * 2;
    int pc1 = pc0 + 1;
    bool has_pc1 = pc1 < n_pcs;

    size_t bk = (size_t)batch * n_clusters + cluster;
    if (active[bk] == 0) return;

    T sum0 = T(0);
    T sum1 = T(0);
    int begin = cat_offsets[batch];
    int end = cat_offsets[batch + 1];

    using Vec = typename std::conditional<std::is_same<T, float>::value, float2,
                                          double2>::type;
    for (int position = begin + threadIdx.x; position < end;
         position += blockDim.x) {
        int cell = cell_indices[position];
        T weight = __ldg(R + (size_t)cell * n_clusters + cluster);
        const T* x_ptr = X + (size_t)cell * n_pcs + pc0;
        if (has_pc1 && (((uintptr_t)x_ptr & (sizeof(Vec) - 1)) == 0)) {
            Vec values = *reinterpret_cast<const Vec*>(x_ptr);
            sum0 += (T)values.x * weight;
            sum1 += (T)values.y * weight;
        } else {
            sum0 += x_ptr[0] * weight;
            if (has_pc1) sum1 += x_ptr[1] * weight;
        }
    }

    sum0 = warp_sum_multi_correction(sum0);
    sum1 = warp_sum_multi_correction(sum1);

    __shared__ T shared0[32];
    __shared__ T shared1[32];
    int lane = threadIdx.x & 31;
    int warp = threadIdx.x >> 5;
    if (lane == 0) {
        shared0[warp] = sum0;
        shared1[warp] = sum1;
    }
    __syncthreads();

    if (warp == 0) {
        int n_warps = (blockDim.x + 31) >> 5;
        T block_sum0 = lane < n_warps ? shared0[lane] : T(0);
        T block_sum1 = lane < n_warps ? shared1[lane] : T(0);
        block_sum0 = warp_sum_multi_correction(block_sum0);
        block_sum1 = warp_sum_multi_correction(block_sum1);
        if (lane == 0) {
            int nb1 = n_batches + 1;
            size_t out = ((size_t)cluster * nb1 + (batch + 1)) * n_pcs + pc0;
            rhs[out] = block_sum0;
            if (has_pc1) rhs[out + 1] = block_sum1;
        }
    }
}

// Compute one weighted X cross-product per observed joint category.  Unlike
// the marginal-category reduction above, every cell is scanned only once:
// the much smaller joint result is expanded into the F marginal rows by
// marginal_from_joint_rhs_kernel below.
template <typename T>
__global__ void segmented_joint_rhs_kernel(
    const T* __restrict__ X, const T* __restrict__ R,
    const int* __restrict__ joint_offsets,
    const int* __restrict__ joint_cell_indices,
    const int* __restrict__ joint_cats, const uint8_t* __restrict__ active,
    T* __restrict__ joint_rhs, int n_pcs, int n_clusters,
    int n_joint_categories, int n_covariates) {
    int pc_pairs = (n_pcs + 1) / 2;
    size_t blocks_per_joint = (size_t)n_clusters * pc_pairs;
    size_t linear_block = blockIdx.x;
    int joint = (int)(linear_block / blocks_per_joint);
    if (joint >= n_joint_categories) return;

    size_t remainder = linear_block % blocks_per_joint;
    int cluster = (int)(remainder / pc_pairs);
    int pc0 = (int)(remainder % pc_pairs) * 2;
    int pc1 = pc0 + 1;
    bool has_pc1 = pc1 < n_pcs;

    // No active marginal row can consume this joint/cluster result.
    bool any_active = false;
    const int* levels = joint_cats + (size_t)joint * n_covariates;
    for (int covariate = 0; covariate < n_covariates; ++covariate) {
        int batch = levels[covariate];
        any_active |= active[(size_t)batch * n_clusters + cluster] != 0;
    }
    if (!any_active) return;

    T sum0 = T(0);
    T sum1 = T(0);
    int begin = joint_offsets[joint];
    int end = joint_offsets[joint + 1];

    using Vec = typename std::conditional<std::is_same<T, float>::value, float2,
                                          double2>::type;
    for (int position = begin + threadIdx.x; position < end;
         position += blockDim.x) {
        int cell = joint_cell_indices[position];
        T weight = __ldg(R + (size_t)cell * n_clusters + cluster);
        const T* x_ptr = X + (size_t)cell * n_pcs + pc0;
        if (has_pc1 && (((uintptr_t)x_ptr & (sizeof(Vec) - 1)) == 0)) {
            Vec values = *reinterpret_cast<const Vec*>(x_ptr);
            sum0 += (T)values.x * weight;
            sum1 += (T)values.y * weight;
        } else {
            sum0 += x_ptr[0] * weight;
            if (has_pc1) sum1 += x_ptr[1] * weight;
        }
    }

    sum0 = warp_sum_multi_correction(sum0);
    sum1 = warp_sum_multi_correction(sum1);

    __shared__ T shared0[32];
    __shared__ T shared1[32];
    int lane = threadIdx.x & 31;
    int warp = threadIdx.x >> 5;
    if (lane == 0) {
        shared0[warp] = sum0;
        shared1[warp] = sum1;
    }
    __syncthreads();

    if (warp == 0) {
        int n_warps = (blockDim.x + 31) >> 5;
        T block_sum0 = lane < n_warps ? shared0[lane] : T(0);
        T block_sum1 = lane < n_warps ? shared1[lane] : T(0);
        block_sum0 = warp_sum_multi_correction(block_sum0);
        block_sum1 = warp_sum_multi_correction(block_sum1);
        if (lane == 0) {
            size_t out = ((size_t)joint * n_clusters + cluster) * n_pcs + pc0;
            joint_rhs[out] = block_sum0;
            if (has_pc1) joint_rhs[out + 1] = block_sum1;
        }
    }
}

// Expand joint-category cross-products into marginal-category rows.  The
// category-to-joint CSR lists joint ids in ascending order, so this second
// reduction is deterministic and requires no atomics.
template <typename T>
__global__ void marginal_from_joint_rhs_kernel(
    const T* __restrict__ joint_rhs,
    const int* __restrict__ marginal_joint_offsets,
    const int* __restrict__ marginal_joint_indices,
    const uint8_t* __restrict__ active, T* __restrict__ rhs, int n_pcs,
    int n_clusters, int n_batches) {
    size_t total = (size_t)n_batches * n_clusters * n_pcs;
    int nb1 = n_batches + 1;
    for (size_t idx = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
         idx < total; idx += (size_t)blockDim.x * gridDim.x) {
        int pc = (int)(idx % n_pcs);
        size_t remainder = idx / n_pcs;
        int batch = (int)(remainder % n_batches);
        int cluster = (int)(remainder / n_batches);

        T value = T(0);
        if (active[(size_t)batch * n_clusters + cluster] != 0) {
            int begin = marginal_joint_offsets[batch];
            int end = marginal_joint_offsets[batch + 1];
            for (int position = begin; position < end; ++position) {
                int joint = marginal_joint_indices[position];
                value +=
                    joint_rhs[((size_t)joint * n_clusters + cluster) * n_pcs +
                              pc];
            }
        }
        rhs[((size_t)cluster * nb1 + (batch + 1)) * n_pcs + pc] = value;
    }
}

// Apply only the marginal regression terms. The intercept participates in the
// solve but, as in Harmony's correction equation, is deliberately retained in
// the embedding.
template <typename T, int N_COVARIATES>
__global__ void apply_multi_correction_kernel(
    const T* __restrict__ X, const T* __restrict__ R,
    const T* __restrict__ W_all, const int* __restrict__ cats,
    T* __restrict__ Z, int n_cells, int n_pcs, int n_clusters, int n_batches,
    int n_covariates, bool initialize_output) {
    size_t total = (size_t)n_cells * n_pcs;
    int nb1 = n_batches + 1;

    for (size_t idx = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
         idx < total; idx += (size_t)blockDim.x * gridDim.x) {
        int cell = (int)(idx / n_pcs);
        int pc = (int)(idx % n_pcs);
        const int* cell_cats =
            cats +
            (size_t)cell * (N_COVARIATES > 0 ? N_COVARIATES : n_covariates);

        T correction = T(0);
        for (int cluster = 0; cluster < n_clusters; ++cluster) {
            T coefficient = T(0);
            if constexpr (N_COVARIATES > 0) {
#pragma unroll
                for (int covariate = 0; covariate < N_COVARIATES; ++covariate) {
                    int row = cell_cats[covariate] + 1;
                    coefficient +=
                        W_all[((size_t)cluster * nb1 + row) * n_pcs + pc];
                }
            } else {
                for (int covariate = 0; covariate < n_covariates; ++covariate) {
                    int row = cell_cats[covariate] + 1;
                    coefficient +=
                        W_all[((size_t)cluster * nb1 + row) * n_pcs + pc];
                }
            }
            correction += R[(size_t)cell * n_clusters + cluster] * coefficient;
        }
        T base = initialize_output ? X[idx] : Z[idx];
        Z[idx] = base - correction;
    }
}
