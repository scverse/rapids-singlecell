#pragma once

#include <cuda_runtime.h>

#include <stdint.h>
#include <type_traits>

// Coalesced cluster lanes reduce fixed row partitions within each joint
// category. The existing joint RHS buffer holds partials until RHS assembly.
template <typename T>
__global__ void joint_observed_kernel(const T* __restrict__ R,
                                      const int* __restrict__ joint_offsets,
                                      const int* __restrict__ cell_indices,
                                      T* __restrict__ partial, int n_clusters,
                                      int n_joint_categories, int n_parts) {
    constexpr int CLUSTER_LANES = 32;
    constexpr int ROW_LANES = 8;
    __shared__ T sums[ROW_LANES * CLUSTER_LANES];
    int cluster_tiles = (n_clusters + CLUSTER_LANES - 1) / CLUSTER_LANES;
    size_t total = (size_t)n_joint_categories * cluster_tiles * n_parts;
    int lane = threadIdx.x % CLUSTER_LANES;
    int row_lane = threadIdx.x / CLUSTER_LANES;

    for (size_t block = blockIdx.x; block < total; block += gridDim.x) {
        int part = (int)(block % n_parts);
        size_t tile = block / n_parts;
        int cluster = (int)(tile % cluster_tiles) * CLUSTER_LANES + lane;
        int joint = (int)(tile / cluster_tiles);
        T sum = T(0);
        if (cluster < n_clusters) {
            for (long long position = (long long)joint_offsets[joint] +
                                      part * ROW_LANES + row_lane;
                 position < joint_offsets[joint + 1];
                 position += n_parts * ROW_LANES) {
                int cell = cell_indices[position];
                sum += R[(size_t)cell * n_clusters + cluster];
            }
        }
        sums[threadIdx.x] = sum;
        __syncthreads();
        if (row_lane == 0 && cluster < n_clusters) {
            T value = sums[lane];
#pragma unroll
            for (int row = 1; row < ROW_LANES; ++row)
                value += sums[row * CLUSTER_LANES + lane];
            partial[((size_t)joint * n_clusters + cluster) * n_parts + part] =
                value;
        }
        __syncthreads();
    }
}

template <typename T>
__global__ void finish_joint_observed_kernel(const T* __restrict__ partial,
                                             T* __restrict__ joint_O,
                                             size_t total, int n_parts) {
    for (size_t idx = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
         idx < total; idx += (size_t)blockDim.x * gridDim.x) {
        T value = partial[idx * n_parts];
        for (int part = 1; part < n_parts; ++part)
            value += partial[idx * n_parts + part];
        joint_O[idx] = value;
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

// Compute one weighted X cross-product per observed joint category.  Unlike
// a marginal-category reduction, every cell is scanned only once. The much
// smaller joint result is expanded into the F marginal rows by
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

constexpr int JOINT_RHS_TILE_ROWS = 4096;
constexpr int JOINT_RHS_CLUSTER_LANES = 8;
constexpr int JOINT_RHS_PC_LANES = 32;
constexpr int JOINT_RHS_ROW_LANES = 8;

template <typename T>
__device__ __forceinline__ void accumulate_multi_rhs(T value, T& sum,
                                                     T& correction) {
    if constexpr (std::is_same<T, float>::value) {
        T adjusted = value - correction;
        T next = sum + adjusted;
        correction = (next - sum) - adjusted;
        sum = next;
    } else {
        sum += value;
    }
}

// Reuse coalesced PC loads across clusters. Fixed row tiles bound rounding
// error and keep the reduction order independent of workspace capacity.
template <typename T>
__global__ void joint_rhs_partials_kernel(
    const T* __restrict__ X, const T* __restrict__ R,
    const int* __restrict__ joint_offsets,
    const int* __restrict__ joint_cell_indices,
    const int* __restrict__ joint_cats, const uint8_t* __restrict__ active,
    const int* __restrict__ tile_offsets, T* __restrict__ partials, int n_pcs,
    int n_clusters, int n_joint_categories, int n_covariates,
    size_t max_tiles) {
    constexpr int CK = JOINT_RHS_CLUSTER_LANES;
    constexpr int PC = JOINT_RHS_PC_LANES;
    constexpr int ROWS = JOINT_RHS_ROW_LANES;
    __shared__ T reduced[CK][ROWS][PC];
    int pc_tiles = (n_pcs + PC - 1) / PC;
    int cluster_tiles = (n_clusters + CK - 1) / CK;
    size_t total = max_tiles * cluster_tiles * pc_tiles;

    for (size_t block = blockIdx.x; block < total; block += gridDim.x) {
        int pc = (int)(block % pc_tiles) * PC + threadIdx.x;
        size_t remainder = block / pc_tiles;
        int cluster = (int)(remainder % cluster_tiles) * CK;
        size_t tile = remainder / cluster_tiles;
        if (tile >= (size_t)tile_offsets[n_joint_categories]) continue;

        int joint = 0;
        if (threadIdx.x == 0) {
            int high = n_joint_categories;
            while (joint < high) {
                int mid = joint + (high - joint) / 2;
                if ((size_t)tile_offsets[mid + 1] <= tile)
                    joint = mid + 1;
                else
                    high = mid;
            }
        }
        joint = __shfl_sync(0xffffffff, joint, 0);
        bool enabled[CK];
        T sums[CK], corrections[CK];
        bool any_active = false;
#pragma unroll
        for (int c = 0; c < CK; ++c) {
            enabled[c] = false;
            sums[c] = corrections[c] = T(0);
            if (cluster + c < n_clusters) {
                for (int cov = 0; cov < n_covariates; ++cov) {
                    int batch = joint_cats[(size_t)joint * n_covariates + cov];
                    enabled[c] |=
                        active[(size_t)batch * n_clusters + cluster + c] != 0;
                }
            }
            any_active |= enabled[c];
        }

        if (any_active) {
            long long begin =
                joint_offsets[joint] +
                (long long)(tile - tile_offsets[joint]) * JOINT_RHS_TILE_ROWS;
            long long end = min(begin + JOINT_RHS_TILE_ROWS,
                                (long long)joint_offsets[joint + 1]);
            for (long long pos = begin + threadIdx.y; pos < end; pos += ROWS) {
                int cell = threadIdx.x == 0 ? joint_cell_indices[pos] : 0;
                cell = __shfl_sync(0xffffffff, cell, 0);
                T loaded =
                    threadIdx.x < CK && cluster + threadIdx.x < n_clusters
                        ? R[(size_t)cell * n_clusters + cluster + threadIdx.x]
                        : T(0);
                T value = pc < n_pcs ? X[(size_t)cell * n_pcs + pc] : T(0);
#pragma unroll
                for (int c = 0; c < CK; ++c) {
                    T weight = __shfl_sync(0xffffffff, loaded, c);
                    if (enabled[c]) {
                        if constexpr (std::is_same<T, float>::value)
                            accumulate_multi_rhs(__fmul_rn(value, weight),
                                                 sums[c], corrections[c]);
                        else
                            sums[c] += value * weight;
                    }
                }
            }
        }
#pragma unroll
        for (int c = 0; c < CK; ++c)
            reduced[c][threadIdx.y][threadIdx.x] = sums[c];
        __syncthreads();
        if (threadIdx.y == 0 && pc < n_pcs) {
#pragma unroll
            for (int c = 0; c < CK; ++c) {
                if (cluster + c >= n_clusters) continue;
                T value = reduced[c][0][threadIdx.x], correction = T(0);
                for (int row = 1; row < ROWS; ++row)
                    accumulate_multi_rhs(reduced[c][row][threadIdx.x], value,
                                         correction);
                partials[(tile * n_clusters + cluster + c) * n_pcs + pc] =
                    value;
            }
        }
        __syncthreads();
    }
}

template <typename T>
__global__ void finish_joint_rhs_kernel(const T* __restrict__ partials,
                                        const int* __restrict__ tile_offsets,
                                        T* __restrict__ joint_rhs, int n_pcs,
                                        int n_clusters, size_t total) {
    for (size_t idx = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
         idx < total; idx += (size_t)blockDim.x * gridDim.x) {
        size_t joint_stride = (size_t)n_clusters * n_pcs;
        int joint = (int)(idx / joint_stride);
        size_t offset = idx % joint_stride;
        T value = T(0), correction = T(0);
        for (int tile = tile_offsets[joint]; tile < tile_offsets[joint + 1];
             ++tile)
            accumulate_multi_rhs(partials[(size_t)tile * joint_stride + offset],
                                 value, correction);
        joint_rhs[idx] = value;
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
