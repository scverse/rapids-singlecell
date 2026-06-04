#pragma once

#include <cuda_runtime.h>

// Warp size (threads per warp on NVIDIA GPUs)
static constexpr int WARP_SIZE = 32;

// ---------------------------------------------------------------------------
// edistance_kernel_impl: a single pairwise-distance kernel parameterized by a
// compile-time "loader" policy. The loop structure, squared-difference math,
// diagonal (within-group) upper-triangle skip, and the warp/block reduction are
// shared; the ONLY things that differ between the dense and sparse paths are
// how the B tile is brought into shared memory and how the A cell's value for a
// given feature is obtained. Those two operations live in the loader, so there
// is exactly one copy of the actual algorithm.
//
// The kernel processes one group pair per block (blockIdx.x), tiling the
// smaller group (B) into shared memory (smem_b, [FEAT_TILE][CELL_TILE]) and
// striding the larger group (A) across threads. Output is one sum per pair.
//
// Loader contract (all methods __device__, called by every thread):
//   init_a(ia, valid):        prepare the A cell for row `ia`.
//   init_b_tile(start_b, jb_base, cit, tid): prepare the B tile of cit cells.
//   fill_b(smem_b, feat_base, fit, tid, bs): fill smem_b for this feature
//       window; leaves no trailing sync (the kernel syncs after).
//   fill_a(a_win, feat_base, fit): produce A's window (no-op if a_value reads
//       directly from global).
//   a_value(a_win, f, feat_base) -> T: A's value for window-feature f.
//
// Per-thread mutable loader state (cursors, idx_i, ...) lives in the loader,
// which is passed by value (each thread mutates its own copy).
// ---------------------------------------------------------------------------
template <typename T, int CELL_TILE, int FEAT_TILE, typename Loader>
__global__ void edistance_kernel_impl(Loader ld,
                                      const int* __restrict__ cat_offsets,
                                      const int* __restrict__ pair_left,
                                      const int* __restrict__ pair_right,
                                      T* __restrict__ pairwise_sums,
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

    const int start_pa = cat_offsets[pair_a];
    const int end_pa = cat_offsets[pair_a + 1];
    const int start_pb = cat_offsets[pair_b];
    const int end_pb = cat_offsets[pair_b + 1];

    // Always iterate over the larger group (A) and tile the smaller group (B)
    // into shared memory. Small B stays hot in L2 across many A iterations.
    const bool swap = (end_pa - start_pa) < (end_pb - start_pb);
    const int start_a = swap ? start_pb : start_pa;
    const int end_a = swap ? end_pb : end_pa;
    const int start_b = swap ? start_pa : start_pb;
    const int end_b = swap ? end_pa : end_pb;

    const int n_a = end_a - start_a;
    const int n_b = end_b - start_b;

    // Distribute A cells across blocks_per_pair
    const int total_threads_for_pair = blocks_per_pair * block_size;
    const int global_thread_in_pair = block_in_pair * block_size + thread_id;
    const int n_iters_a =
        (n_a + total_threads_for_pair - 1) / total_threads_for_pair;

    for (int iter_a = 0; iter_a < n_iters_a; ++iter_a) {
        const int ia =
            start_a + iter_a * total_threads_for_pair + global_thread_in_pair;
        const bool valid_a = (ia < end_a);
        const int i_local = ia - start_a;

        // Tile over B cells
        for (int jb_base = 0; jb_base < n_b; jb_base += CELL_TILE) {
            const int cells_in_tile = min(CELL_TILE, n_b - jb_base);
            // Reset the A and B data cursors for this B tile: A is re-walked
            // from its start against every B tile (sparse), and dense just
            // re-stores idx_i.
            ld.init_a(ia, valid_a);
            ld.init_b_tile(start_b, jb_base, cells_in_tile, thread_id);

            // Accumulate squared distances for this cell tile
            T dist_sq[CELL_TILE];
#pragma unroll
            for (int c = 0; c < CELL_TILE; ++c) dist_sq[c] = T(0.0);

            // Tile over features
            for (int feat_base = 0; feat_base < n_features;
                 feat_base += FEAT_TILE) {
                const int feats_in_tile =
                    min(FEAT_TILE, n_features - feat_base);

                // Dense kernels read A directly from global per feature, so
                // a_win stays unused and is optimized away. Sparse kernels fill
                // it in fill_a and read it in a_value.
                T a_win[FEAT_TILE];

                ld.fill_b(smem_b, feat_base, feats_in_tile, thread_id,
                          block_size);
                ld.fill_a(a_win, feat_base, feats_in_tile);

                __syncthreads();

                // Compute partial squared differences for this feature chunk
                if (valid_a) {
                    for (int f = 0; f < feats_in_tile; ++f) {
                        const T val_a = ld.a_value(a_win, f, feat_base);
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

            // dist_sq[c] contains full squared distance for cell c
            if (valid_a) {
#pragma unroll
                for (int c = 0; c < CELL_TILE; ++c) {
                    if (c >= cells_in_tile) break;
                    int j_local = jb_base + c;

                    // Skip lower triangle for diagonal blocks
                    if (pair_a == pair_b && i_local >= j_local) continue;

                    local_sum += sqrt(dist_sq[c]);
                }
            }
        }
    }

    // Warp shuffle reduction
#pragma unroll
    for (int offset = WARP_SIZE / 2; offset > 0; offset >>= 1)
        local_sum += __shfl_down_sync(0xffffffff, local_sum, offset);

    // Block reduction via shared memory
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
// Dense loader: reads cells from a row-major (n_cells x n_features) embedding.
// A is read directly from global per feature (1 register), so fill_a is a no-op
// and a_win is unused (optimized away) — this preserves the tuned dense
// kernel's register footprint exactly.
// ---------------------------------------------------------------------------
template <typename T, int CELL_TILE, int FEAT_TILE>
struct DenseLoader {
    const T* __restrict__ embedding;
    const int* __restrict__ cell_indices;
    int n_features;
    // per-thread / per-tile state
    int idx_i;
    int sb, jb, cit;

    __device__ void init_a(int ia, bool valid) {
        idx_i = valid ? cell_indices[ia] : 0;
    }

    __device__ void init_b_tile(int start_b, int jb_base, int cells_in_tile,
                                int /*thread_id*/) {
        sb = start_b;
        jb = jb_base;
        cit = cells_in_tile;
    }

    __device__ void fill_b(T* smem_b, int feat_base, int feats_in_tile,
                           int thread_id, int block_size) {
        const int total_elems = FEAT_TILE * CELL_TILE;
        for (int i = thread_id; i < total_elems; i += block_size) {
            int cell_idx = i / FEAT_TILE;
            int feat_idx = i % FEAT_TILE;
            T val = T(0.0);
            if (cell_idx < cit && feat_idx < feats_in_tile) {
                int global_b_idx = cell_indices[sb + jb + cell_idx];
                val = embedding[static_cast<size_t>(global_b_idx) * n_features +
                                feat_base + feat_idx];
            }
            // Store as smem_b[feat][cell] for sequential access
            smem_b[feat_idx * CELL_TILE + cell_idx] = val;
        }
    }

    __device__ void fill_a(T* /*a_win*/, int /*feat_base*/,
                           int /*feats_in_tile*/) {
    }

    __device__ T a_value(const T* /*a_win*/, int f, int feat_base) const {
        return embedding[static_cast<size_t>(idx_i) * n_features + feat_base +
                         f];
    }
};

// ---------------------------------------------------------------------------
// Sparse loader: reads cells from CSR (data, indices, indptr) and densifies
// each feature window on the fly — B into shared, A into the per-thread a_win
// window. Requires canonical (column-sorted) CSR so a single forward cursor per
// row walks the ascending feature windows, touching each nonzero once. IndptrT
// is int (int32) or int64_t so nnz > 2^31-1 is addressable.
// ---------------------------------------------------------------------------
template <typename T, int CELL_TILE, int FEAT_TILE, typename IndptrT>
struct SparseLoader {
    const T* __restrict__ data;
    const int* __restrict__ indices;
    const IndptrT* __restrict__ indptr;
    const int* __restrict__ cell_indices;
    // per-thread state
    IndptrT a_cur, a_end;
    IndptrT b_cur, b_end;
    bool b_active;

    __device__ void init_a(int ia, bool valid) {
        if (valid) {
            const int idx_i = cell_indices[ia];
            a_cur = indptr[idx_i];
            a_end = indptr[idx_i + 1];
        } else {
            a_cur = 0;
            a_end = 0;
        }
    }

    __device__ void init_b_tile(int start_b, int jb_base, int cells_in_tile,
                                int thread_id) {
        b_active = thread_id < cells_in_tile;
        if (b_active) {
            const int cellB = cell_indices[start_b + jb_base + thread_id];
            b_cur = indptr[cellB];
            b_end = indptr[cellB + 1];
        } else {
            b_cur = 0;
            b_end = 0;
        }
    }

    __device__ void fill_b(T* smem_b, int feat_base, int feats_in_tile,
                           int thread_id, int block_size) {
        const int total_elems = FEAT_TILE * CELL_TILE;
        for (int i = thread_id; i < total_elems; i += block_size) {
            smem_b[i] = T(0.0);
        }
        __syncthreads();

        const int feat_hi = feat_base + feats_in_tile;
        if (b_active) {
            while (b_cur < b_end) {
                const int col = indices[b_cur];
                if (col >= feat_hi) break;
                // col >= feat_base is guaranteed: the cursor stopped at the
                // previous window's feat_hi == this feat_base.
                smem_b[(col - feat_base) * CELL_TILE + thread_id] = data[b_cur];
                ++b_cur;
            }
        }
    }

    __device__ void fill_a(T* a_win, int feat_base, int feats_in_tile) {
#pragma unroll
        for (int f = 0; f < FEAT_TILE; ++f) a_win[f] = T(0.0);
        const int feat_hi = feat_base + feats_in_tile;
        while (a_cur < a_end) {
            const int col = indices[a_cur];
            if (col >= feat_hi) break;
            a_win[col - feat_base] = data[a_cur];
            ++a_cur;
        }
    }

    __device__ T a_value(const T* a_win, int f, int /*feat_base*/) const {
        return a_win[f];
    }
};
