#pragma once

#include <cuda_runtime.h>

// Warp size (threads per warp on NVIDIA GPUs)
static constexpr int WARP_SIZE = 32;

// ---------------------------------------------------------------------------
// Shared helpers used by BOTH the dense and sparse kernels. These cover the
// parts that are genuinely identical and NOT on the hot path: the group-pair
// geometry and the final warp/block reduction. The hot load+compute loop is
// kept straight-line inside each kernel on purpose — wrapping it behind a
// policy/abstraction (e.g. a loader functor) measurably regresses the sparse
// kernel ~1.5x even though the instruction mix is identical, so the inner loop
// is deliberately duplicated.
// ---------------------------------------------------------------------------

// Larger group becomes A (strided across threads); smaller becomes B (tiled
// into shared memory and kept hot in L2 across A iterations).
struct EDistancePairTiles {
    int start_a, end_a, start_b, end_b;
};

__device__ inline EDistancePairTiles edistance_pair_tiles(
    int pair_a, int pair_b, const int* __restrict__ cat_offsets) {
    const int start_pa = cat_offsets[pair_a];
    const int end_pa = cat_offsets[pair_a + 1];
    const int start_pb = cat_offsets[pair_b];
    const int end_pb = cat_offsets[pair_b + 1];
    const bool swap = (end_pa - start_pa) < (end_pb - start_pb);
    EDistancePairTiles t;
    t.start_a = swap ? start_pb : start_pa;
    t.end_a = swap ? end_pb : end_pa;
    t.start_b = swap ? start_pa : start_pb;
    t.end_b = swap ? end_pa : end_pb;
    return t;
}

// Warp-shuffle + block reduction of local_sum, then atomicAdd into
// pairwise_sums[pair_id]. Runs once per block (not hot).
template <typename T>
__device__ inline void edistance_block_reduce_add(T local_sum,
                                                  T* __restrict__ pairwise_sums,
                                                  int pair_id, int thread_id,
                                                  int block_size) {
#pragma unroll
    for (int offset = WARP_SIZE / 2; offset > 0; offset >>= 1)
        local_sum += __shfl_down_sync(0xffffffff, local_sum, offset);

    static __shared__ T warp_sums[WARP_SIZE];
    if ((thread_id & (WARP_SIZE - 1)) == 0)
        warp_sums[thread_id / WARP_SIZE] = local_sum;
    __syncthreads();

    if (thread_id < WARP_SIZE) {
        T val = (thread_id < (block_size / WARP_SIZE)) ? warp_sums[thread_id]
                                                       : T(0.0);
#pragma unroll
        for (int offset = WARP_SIZE / 2; offset > 0; offset >>= 1)
            val += __shfl_down_sync(0xffffffff, val, offset);

        if (thread_id == 0) {
            atomicAdd(&pairwise_sums[pair_id], val);
        }
    }
}

// ---------------------------------------------------------------------------
// Dense kernel: cells come from a row-major (n_cells x n_features) embedding.
// Output is flat: one sum per pair, indexed by pair_id (blockIdx.x).
// ---------------------------------------------------------------------------
template <typename T, int CELL_TILE, int FEAT_TILE>
__global__ void edistance_kernel(const T* __restrict__ embedding,
                                 const int* __restrict__ cat_offsets,
                                 const int* __restrict__ cell_indices,
                                 const int* __restrict__ pair_left,
                                 const int* __restrict__ pair_right,
                                 T* __restrict__ pairwise_sums, int n_features,
                                 int blocks_per_pair) {
    extern __shared__ char smem_raw[];
    T* smem_b = reinterpret_cast<T*>(smem_raw);

    const int thread_id = threadIdx.x;
    const int pair_id = blockIdx.x;
    const int block_in_pair = blockIdx.y;
    const int block_size = blockDim.x;

    T local_sum = T(0.0);

    const int pair_a = pair_left[pair_id];
    const int pair_b = pair_right[pair_id];
    const EDistancePairTiles t =
        edistance_pair_tiles(pair_a, pair_b, cat_offsets);
    const int start_a = t.start_a, end_a = t.end_a;
    const int start_b = t.start_b, end_b = t.end_b;

    const int n_a = end_a - start_a;
    const int n_b = end_b - start_b;

    const int total_threads_for_pair = blocks_per_pair * block_size;
    const int global_thread_in_pair = block_in_pair * block_size + thread_id;
    const int n_iters_a =
        (n_a + total_threads_for_pair - 1) / total_threads_for_pair;

    for (int iter_a = 0; iter_a < n_iters_a; ++iter_a) {
        const int ia =
            start_a + iter_a * total_threads_for_pair + global_thread_in_pair;
        const bool valid_a = (ia < end_a);
        const int idx_i = valid_a ? cell_indices[ia] : 0;
        const int i_local = ia - start_a;

        for (int jb_base = 0; jb_base < n_b; jb_base += CELL_TILE) {
            const int cells_in_tile = min(CELL_TILE, n_b - jb_base);

            T dist_sq[CELL_TILE];
#pragma unroll
            for (int c = 0; c < CELL_TILE; ++c) dist_sq[c] = T(0.0);

            for (int feat_base = 0; feat_base < n_features;
                 feat_base += FEAT_TILE) {
                const int feats_in_tile =
                    min(FEAT_TILE, n_features - feat_base);

                // Cooperatively load B tile into shared memory
                const int total_elems = FEAT_TILE * CELL_TILE;
                for (int i = thread_id; i < total_elems; i += block_size) {
                    int cell_idx = i / FEAT_TILE;
                    int feat_idx = i % FEAT_TILE;
                    T val = T(0.0);
                    if (cell_idx < cells_in_tile && feat_idx < feats_in_tile) {
                        int global_b_idx =
                            cell_indices[start_b + jb_base + cell_idx];
                        val = embedding[static_cast<size_t>(global_b_idx) *
                                            n_features +
                                        feat_base + feat_idx];
                    }
                    smem_b[feat_idx * CELL_TILE + cell_idx] = val;
                }

                __syncthreads();

                if (valid_a) {
                    for (int f = 0; f < feats_in_tile; ++f) {
                        T val_a =
                            embedding[static_cast<size_t>(idx_i) * n_features +
                                      feat_base + f];
#pragma unroll
                        for (int c = 0; c < CELL_TILE; ++c) {
                            T val_b = smem_b[f * CELL_TILE + c];
                            T diff = val_a - val_b;
                            dist_sq[c] += diff * diff;
                        }
                    }
                }

                __syncthreads();
            }

            if (valid_a) {
#pragma unroll
                for (int c = 0; c < CELL_TILE; ++c) {
                    if (c >= cells_in_tile) break;
                    int j_local = jb_base + c;
                    if (pair_a == pair_b && i_local >= j_local) continue;
                    local_sum += sqrt(dist_sq[c]);
                }
            }
        }
    }

    edistance_block_reduce_add(local_sum, pairwise_sums, pair_id, thread_id,
                               block_size);
}

// ---------------------------------------------------------------------------
// Sparse kernel: cells come from CSR (data, indices, indptr) and each feature
// window is densified on the fly — B into the shared tile, A into a per-thread
// register/local window a_win. The inner squared-difference math is identical
// to the dense kernel, so results match bit-for-bit. Requires canonical
// (column-sorted) CSR so a single forward cursor per row walks the ascending
// feature windows, touching each nonzero once. IndptrT is int (int32) or
// int64_t so nnz > 2^31-1 is addressable; column indices stay int32.
// ---------------------------------------------------------------------------
template <typename T, int CELL_TILE, int FEAT_TILE, typename IndptrT>
__global__ void edistance_sparse_kernel(
    const T* __restrict__ data, const int* __restrict__ indices,
    const IndptrT* __restrict__ indptr, const int* __restrict__ cat_offsets,
    const int* __restrict__ cell_indices, const int* __restrict__ pair_left,
    const int* __restrict__ pair_right, T* __restrict__ pairwise_sums,
    int n_features, int blocks_per_pair) {
    extern __shared__ char smem_raw[];
    T* smem_b = reinterpret_cast<T*>(smem_raw);

    const int thread_id = threadIdx.x;
    const int pair_id = blockIdx.x;
    const int block_in_pair = blockIdx.y;
    const int block_size = blockDim.x;

    T local_sum = T(0.0);

    const int pair_a = pair_left[pair_id];
    const int pair_b = pair_right[pair_id];
    const EDistancePairTiles t =
        edistance_pair_tiles(pair_a, pair_b, cat_offsets);
    const int start_a = t.start_a, end_a = t.end_a;
    const int start_b = t.start_b, end_b = t.end_b;

    const int n_a = end_a - start_a;
    const int n_b = end_b - start_b;

    const int total_threads_for_pair = blocks_per_pair * block_size;
    const int global_thread_in_pair = block_in_pair * block_size + thread_id;
    const int n_iters_a =
        (n_a + total_threads_for_pair - 1) / total_threads_for_pair;

    for (int iter_a = 0; iter_a < n_iters_a; ++iter_a) {
        const int ia =
            start_a + iter_a * total_threads_for_pair + global_thread_in_pair;
        const bool valid_a = (ia < end_a);
        const int idx_i = valid_a ? cell_indices[ia] : 0;
        const int i_local = ia - start_a;

        for (int jb_base = 0; jb_base < n_b; jb_base += CELL_TILE) {
            const int cells_in_tile = min(CELL_TILE, n_b - jb_base);

            // CSR cursors for this B tile / A cell. Thread `t` owns B cell `t`
            // for the whole tile, so its cursor lives in registers. A is
            // re-walked from its start against every B tile.
            IndptrT b_cur = 0, b_end = 0;
            if (thread_id < cells_in_tile) {
                const int cellB = cell_indices[start_b + jb_base + thread_id];
                b_cur = indptr[cellB];
                b_end = indptr[cellB + 1];
            }
            IndptrT a_cur = 0, a_end = 0;
            if (valid_a) {
                a_cur = indptr[idx_i];
                a_end = indptr[idx_i + 1];
            }

            T dist_sq[CELL_TILE];
#pragma unroll
            for (int c = 0; c < CELL_TILE; ++c) dist_sq[c] = T(0.0);

            for (int feat_base = 0; feat_base < n_features;
                 feat_base += FEAT_TILE) {
                const int feats_in_tile =
                    min(FEAT_TILE, n_features - feat_base);
                const int feat_hi = feat_base + feats_in_tile;

                // Zero the shared B tile, then densify B cells into it: one
                // thread per B cell scatters its nonzeros in [feat_base,
                // feat_hi) into smem_b[feat][cell].
                const int total_elems = FEAT_TILE * CELL_TILE;
                for (int i = thread_id; i < total_elems; i += block_size) {
                    smem_b[i] = T(0.0);
                }
                __syncthreads();

                if (thread_id < cells_in_tile) {
                    while (b_cur < b_end) {
                        const int col = indices[b_cur];
                        if (col >= feat_hi) break;
                        // col >= feat_base is guaranteed: the cursor stopped at
                        // the previous window's feat_hi == this feat_base.
                        smem_b[(col - feat_base) * CELL_TILE + thread_id] =
                            data[b_cur];
                        ++b_cur;
                    }
                }

                // Densify the A cell into a per-thread window.
                T a_win[FEAT_TILE];
                if (valid_a) {
#pragma unroll
                    for (int f = 0; f < FEAT_TILE; ++f) a_win[f] = T(0.0);
                    while (a_cur < a_end) {
                        const int col = indices[a_cur];
                        if (col >= feat_hi) break;
                        a_win[col - feat_base] = data[a_cur];
                        ++a_cur;
                    }
                }

                __syncthreads();

                if (valid_a) {
                    for (int f = 0; f < feats_in_tile; ++f) {
                        const T val_a = a_win[f];
#pragma unroll
                        for (int c = 0; c < CELL_TILE; ++c) {
                            const T val_b = smem_b[f * CELL_TILE + c];
                            const T diff = val_a - val_b;
                            dist_sq[c] += diff * diff;
                        }
                    }
                }

                __syncthreads();
            }

            if (valid_a) {
#pragma unroll
                for (int c = 0; c < CELL_TILE; ++c) {
                    if (c >= cells_in_tile) break;
                    int j_local = jb_base + c;
                    if (pair_a == pair_b && i_local >= j_local) continue;
                    local_sum += sqrt(dist_sq[c]);
                }
            }
        }
    }

    edistance_block_reduce_add(local_sum, pairwise_sums, pair_id, thread_id,
                               block_size);
}
