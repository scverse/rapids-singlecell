#pragma once

#include <cuda_runtime.h>
#include <cub/device/device_radix_sort.cuh>

#include <algorithm>
#include <stdexcept>
#include <string>
#include <vector>

#include "nb_types.h"
#include "rmm_scratch.h"

// =============================================================================
// Shared-memory tile sweep for minor-axis reductions over CSR/CSC matrices.
// =============================================================================
//
// The job: for every column of a compressed matrix, accumulate something over
// the nonzeros in that column (a sum, a count, a sum of squares, ...). Rows are
// contiguous in memory but columns are scattered, so the naive kernel issues a
// global atomic per nonzero and is bound by the L2 atomic units.
//
// The design, in plain language:
//   * A block owns a range of rows and a private scratch pad in shared memory.
//     It adds into the pad with cheap on-chip atomics and writes each column's
//     subtotal to global memory once, at the end.
//   * The pad only holds a stripe of columns, a TILE. The block works tile by
//     tile.
//   * Each row is split into one SLICE per tile: the run of its nonzeros whose
//     columns fall inside the tile. In a row with sorted columns the slices sit
//     back to back, so a BOOKMARK per row (where this tile's slice ended) tells
//     the next tile where its slice starts. Every nonzero is read exactly once.
//   * Sorted rows are not required. On a later tile, a column below the tile's
//     lower bound can only be one that an earlier tile skipped, which never
//     happens in a sorted row, so the kernel raises a flag. The host then
//     redoes the work order-agnostically: RESCAN (each tile reads the whole
//     row) when there are few tiles, the per-nonzero atomic kernel when there
//     are many.
//   * The tile width comes from the device's shared-memory budget.
//
// Vocabulary used in the code: row, column, tile [tile_begin, tile_end),
// slice, bookmark, tile_col (a column's slot inside the tile), nnz_pos (an
// index into indices/data).
//
// An Op describes one reduction: a POD passed by value that holds the device
// pointers and defines
//   static constexpr size_t bytes_per_col;  // scratch-pad bytes per column
//   static constexpr bool needs_rows;       // fallback must know the row
//   int tile_size;                          // filled in by minor_reduce()
//   __device__ bool row_active(int row) const;
//   __device__ void zero_col(char* pad, int tile_col, int col) const;
//   __device__ void add(char* pad, long long nnz_pos, int tile_col) const;
//   __device__ void flush_col(const char* pad, int group, int col,
//                             int tile_col) const;
//   __device__ void add_global(long long nnz_pos, int col, int group) const;
//   void zero_outputs(int n_cols, int n_groups, cudaStream_t stream) const;
// The pad is 16-byte aligned; lay sub-arrays out in decreasing alignment
// (doubles, then 4-byte types, then bytes). Grouped reductions (aggregate,
// ligrec) first sort rows by group so a block only ever accumulates for one
// group; see build_grouped_rows().

// ---- argument validation ---------------------------------------------------

/// Reject a malformed binding argument with a clear Python error instead of an
/// out-of-bounds kernel read.
inline void require_arg(bool cond, const char* what) {
    if (!cond) throw std::invalid_argument(what);
}

/// The host-side shape checks every compressed-matrix binding can afford:
/// a 1-D indptr and equally long indices/data. Offsets are not read back from
/// the device here; cupyx guarantees their structure.
template <typename IndptrArr, typename IdxArr, typename DataArr>
inline void require_csr_arrays(const char* what, const IndptrArr& indptr,
                               const IdxArr& indices, const DataArr& data) {
    require_arg(
        indptr.ndim() == 1 && indptr.shape(0) >= 1,
        (std::string(what) + ": indptr must be 1-D and non-empty").c_str());
    require_arg(
        indices.shape(0) == data.shape(0),
        (std::string(what) + ": indices and data must have equal length")
            .c_str());
}

// ---- tunables --------------------------------------------------------------

constexpr int WARP = 32;
// 16 warps: one row slice per warp, coalesced 32-wide reads.
constexpr int SWEEP_BLOCK_THREADS = 512;
// Occupancy target; the per-block shared-memory budget follows from it.
constexpr int SWEEP_BLOCKS_PER_SM = 2;
// Rows per block are chosen for this many blocks in flight per SM: more blocks
// balance better, fewer blocks flush fewer subtotals.
constexpr int SWEEP_BLOCKS_IN_FLIGHT_PER_SM = 12;
constexpr int SWEEP_MIN_ROWS_PER_BLOCK = SWEEP_BLOCK_THREADS / WARP;
constexpr int SWEEP_MAX_ROWS_PER_BLOCK = 1024;  // bookmark array size
// Below this many nonzeros per slice the per-tile bookkeeping dominates and the
// atomic kernel is faster (measured crossover between 10 and 20).
constexpr long long SWEEP_MIN_NNZ_PER_SLICE = 16;
// Rescan re-reads 4 B per nonzero per extra tile; it beats the atomic kernel up
// to this many tiles (measured: 0.43x to 0.75x its time at 1 to 4 tiles,
// break-even at 5).
constexpr int SWEEP_MAX_RESCAN_TILES = 4;
// Above this, dynamic shared memory needs an explicit per-kernel opt-in.
constexpr size_t SMEM_DEFAULT_LIMIT = 48 * 1024;
constexpr int ATOMIC_BLOCK_THREADS = 256;

// ---- plan ------------------------------------------------------------------

struct TilePlan {
    bool use_tiled;  // false: run the atomic fallback instead
    int tile_size;   // columns per tile
    int n_tiles;
    int rows_per_block;
    size_t smem_bytes;  // scratch pad + bookmarks
};

/// Scratch-pad bytes for a tile, padded so the bookmarks behind it stay
/// aligned.
__host__ __device__ inline size_t pad_bytes(int tile_size,
                                            size_t bytes_per_col) {
    return ((size_t)tile_size * bytes_per_col + 7) / 8 * 8;
}

/// Size the tile from the per-SM shared memory at the target occupancy, capped
/// by the per-block opt-in limit, so it adapts from a T4 (64 KB per SM) to
/// datacenter parts without hardcoding either.
inline TilePlan plan_tiles(long long nnz, int n_rows, int n_cols,
                           size_t bytes_per_col) {
    const DeviceSmemLimits& lim = device_smem_limits();
    size_t budget = lim.per_sm / SWEEP_BLOCKS_PER_SM;
    budget =
        budget > lim.reserved_per_block ? budget - lim.reserved_per_block : 0;
    budget = std::min(budget, lim.per_block_optin);
    budget = budget > 8 ? budget - 8 : 0;  // alignment padding

    int rows_per_block = n_rows / (lim.n_sms * SWEEP_BLOCKS_IN_FLIGHT_PER_SM);
    rows_per_block = std::clamp(rows_per_block, SWEEP_MIN_ROWS_PER_BLOCK,
                                SWEEP_MAX_ROWS_PER_BLOCK);

    const size_t bookmark_bytes = (size_t)rows_per_block * sizeof(int);
    TilePlan plan{false, 0, 0, rows_per_block, 0};
    if (budget < bookmark_bytes + bytes_per_col) return plan;

    const size_t max_tile_cols = (budget - bookmark_bytes) / bytes_per_col;
    plan.tile_size = (int)std::min<size_t>(max_tile_cols, (size_t)n_cols);
    plan.n_tiles = (n_cols + plan.tile_size - 1) / plan.tile_size;
    const long long nnz_per_slice = nnz / ((long long)n_rows * plan.n_tiles);
    if (plan.n_tiles > 1 && nnz_per_slice < SWEEP_MIN_NNZ_PER_SLICE)
        return plan;

    plan.smem_bytes = pad_bytes(plan.tile_size, bytes_per_col) + bookmark_bytes;
    plan.use_tiled = true;
    return plan;
}

/// Opt a kernel into dynamic shared memory above the 48 KB default.
template <typename Kernel>
inline void opt_in_dynamic_smem(Kernel kernel, size_t bytes, const char* what) {
    if (bytes > SMEM_DEFAULT_LIMIT) {
        cuda_check(cudaFuncSetAttribute(
                       kernel, cudaFuncAttributeMaxDynamicSharedMemorySize,
                       (int)bytes),
                   what);
    }
}

// ---- device: one block, one tile -------------------------------------------

/// Rows [first, last) of the row list that one block owns, all in `group`.
struct BlockRows {
    int first;
    int last;
    int group;
};

__device__ inline BlockRows block_rows(const BlockRows* __restrict__ per_block,
                                       int rows_per_block, int n_rows) {
    if (per_block != nullptr) return per_block[blockIdx.x];
    const int first = blockIdx.x * rows_per_block;
    return {first, min(first + rows_per_block, n_rows), 0};
}

/// Sweep one row's slice of the current tile, resuming at `bookmark` (an offset
/// into the row). Returns where the slice ended, which is the bookmark for the
/// next tile. Sets `out_of_order` when a column below `tile_begin` shows up:
/// that column was skipped by an earlier tile, so the row is not sorted.
template <typename IdxT, typename Op>
__device__ inline int warp_sweep_slice(const IdxT* __restrict__ indices,
                                       IdxT row_begin, IdxT row_end,
                                       int bookmark, IdxT tile_begin,
                                       IdxT tile_end, const Op& op, char* pad,
                                       bool& out_of_order) {
    const int lane = threadIdx.x & (WARP - 1);
    IdxT pos = row_begin + bookmark;
    while (pos < row_end) {
        const IdxT nnz_pos = pos + lane;
        // Lanes past the row read a sentinel that ends the slice exactly like a
        // column past the tile would.
        const IdxT col = (nnz_pos < row_end) ? indices[nnz_pos] : tile_end;
        const bool in_tile = col < tile_end;
        if (in_tile) {
            if (col < tile_begin) {
                out_of_order = true;
            } else {
                op.add(pad, (long long)nnz_pos,
                       static_cast<int>(col - tile_begin));
            }
        }
        const unsigned past_tile = __ballot_sync(0xffffffffu, !in_tile);
        if (past_tile) {
            // The slice ends at the first lane whose column is past the tile.
            return static_cast<int>(pos + (__ffs(past_tile) - 1) - row_begin);
        }
        pos += WARP;
    }
    return static_cast<int>(row_end - row_begin);
}

/// Order-agnostic alternative: read the whole row, keep what falls in the tile.
template <typename IdxT, typename Op>
__device__ inline void warp_rescan_row(const IdxT* __restrict__ indices,
                                       IdxT row_begin, IdxT row_end,
                                       IdxT tile_begin, IdxT tile_end,
                                       const Op& op, char* pad) {
    const int lane = threadIdx.x & (WARP - 1);
    for (IdxT nnz_pos = row_begin + lane; nnz_pos < row_end; nnz_pos += WARP) {
        const IdxT col = indices[nnz_pos];
        if (col >= tile_begin && col < tile_end) {
            op.add(pad, (long long)nnz_pos, static_cast<int>(col - tile_begin));
        }
    }
}

template <typename Op>
__device__ inline void zero_tile(const Op& op, char* pad, int tile_begin,
                                 int width) {
    for (int tile_col = threadIdx.x; tile_col < width; tile_col += blockDim.x) {
        op.zero_col(pad, tile_col, tile_begin + tile_col);
    }
}

template <typename Op>
__device__ inline void flush_tile(const Op& op, const char* pad, int tile_begin,
                                  int width, int group) {
    for (int tile_col = threadIdx.x; tile_col < width; tile_col += blockDim.x) {
        op.flush_col(pad, group, tile_begin + tile_col, tile_col);
    }
}

/// Every warp sweeps its rows' slices of the current tile. Rows are
/// `row_order[i]` when a reordering is given, else `i`.
template <typename IdxT, typename Op>
__device__ inline void sweep_tile(const IdxT* __restrict__ indptr,
                                  const IdxT* __restrict__ indices,
                                  const int* __restrict__ row_order,
                                  int* bookmarks, BlockRows rows,
                                  int tile_begin, int tile_size, bool rescan,
                                  int* __restrict__ out_of_order_flag,
                                  const Op& op, char* pad) {
    const int lane = threadIdx.x & (WARP - 1);
    const int warp = threadIdx.x / WARP;
    const int warps_per_block = blockDim.x / WARP;
    const IdxT tile_lo = static_cast<IdxT>(tile_begin);
    const IdxT tile_hi = tile_lo + tile_size;
    bool out_of_order = false;

    for (int i = rows.first + warp; i < rows.last; i += warps_per_block) {
        const int row = row_order != nullptr ? row_order[i] : i;
        if (!op.row_active(row)) continue;
        const IdxT row_begin = indptr[row];
        const IdxT row_end = indptr[row + 1];
        if (rescan) {
            warp_rescan_row(indices, row_begin, row_end, tile_lo, tile_hi, op,
                            pad);
            continue;
        }
        const int bookmark = warp_sweep_slice(
            indices, row_begin, row_end, bookmarks[i - rows.first], tile_lo,
            tile_hi, op, pad, out_of_order);
        if (lane == 0) bookmarks[i - rows.first] = bookmark;
    }
    if (out_of_order && out_of_order_flag != nullptr) *out_of_order_flag = 1;
}

/// The tile sweep: zero the pad, sweep every row's slice, flush the subtotals,
/// tile after tile. grid = number of row blocks.
template <typename IdxT, typename Op>
__global__ void __launch_bounds__(SWEEP_BLOCK_THREADS, SWEEP_BLOCKS_PER_SM)
    tile_sweep_kernel(const IdxT* __restrict__ indptr,
                      const IdxT* __restrict__ indices,
                      const int* __restrict__ row_order,
                      const BlockRows* __restrict__ per_block_rows, Op op,
                      int* __restrict__ out_of_order_flag, int n_rows,
                      int n_cols, int n_tiles, int rows_per_block,
                      bool rescan) {
    extern __shared__ __align__(16) char smem[];
    char* pad = smem;  // the Op's accumulators for the current tile
    int* bookmarks = reinterpret_cast<int*>(
        smem + pad_bytes(op.tile_size, Op::bytes_per_col));

    const BlockRows rows = block_rows(per_block_rows, rows_per_block, n_rows);
    for (int r = threadIdx.x; r < rows.last - rows.first; r += blockDim.x) {
        bookmarks[r] = 0;  // every row starts at its beginning
    }

    for (int tile = 0; tile < n_tiles; ++tile) {
        const int tile_begin = tile * op.tile_size;
        const int width = min(op.tile_size, n_cols - tile_begin);
        zero_tile(op, pad, tile_begin, width);
        __syncthreads();
        sweep_tile(indptr, indices, row_order, bookmarks, rows, tile_begin,
                   op.tile_size, rescan, out_of_order_flag, op, pad);
        __syncthreads();
        flush_tile(op, pad, tile_begin, width, rows.group);
        __syncthreads();
    }
}

// ---- device: order-agnostic fallbacks --------------------------------------

/// One global atomic per nonzero; for Ops without groups or row filters.
template <typename IdxT, typename Op>
__global__ void atomic_per_nonzero_kernel(const IdxT* __restrict__ indices,
                                          Op op, long long nnz) {
    const long long stride = (long long)blockDim.x * gridDim.x;
    for (long long nnz_pos = (long long)blockIdx.x * blockDim.x + threadIdx.x;
         nnz_pos < nnz; nnz_pos += stride) {
        op.add_global(nnz_pos, static_cast<int>(indices[nnz_pos]), 0);
    }
}

/// Warp per row, lanes stride the nonzeros; knows the row, so it serves grouped
/// Ops and Ops with a row filter.
template <typename IdxT, typename Op>
__global__ void atomic_per_row_kernel(
    const IdxT* __restrict__ indptr, const IdxT* __restrict__ indices,
    const int* __restrict__ row_order,
    const BlockRows* __restrict__ per_block_rows, Op op, int n_rows,
    int rows_per_block) {
    const BlockRows rows = block_rows(per_block_rows, rows_per_block, n_rows);
    const int lane = threadIdx.x & (WARP - 1);
    const int warp = threadIdx.x / WARP;
    const int warps_per_block = blockDim.x / WARP;
    for (int i = rows.first + warp; i < rows.last; i += warps_per_block) {
        const int row = row_order != nullptr ? row_order[i] : i;
        if (!op.row_active(row)) continue;
        const IdxT row_end = indptr[row + 1];
        for (IdxT nnz_pos = indptr[row] + lane; nnz_pos < row_end;
             nnz_pos += WARP) {
            op.add_global((long long)nnz_pos,
                          static_cast<int>(indices[nnz_pos]), rows.group);
        }
    }
}

// ---- grouped rows -----------------------------------------------------------

/// Rows sorted by group: `row_order` (device, owned by the caller's scratch
/// pool) plus host group offsets into it.
struct GroupedRows {
    const int* row_order = nullptr;
    std::vector<int> offsets;  // n_groups + 1
    int n_active = 0;
};

__global__ inline void group_keys_kernel(const int* __restrict__ cats,
                                         const bool* __restrict__ mask,
                                         int n_rows, int n_groups,
                                         int* __restrict__ keys,
                                         int* __restrict__ vals,
                                         int* __restrict__ counts) {
    const int stride = blockDim.x * gridDim.x;
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < n_rows;
         i += stride) {
        int k = (mask == nullptr || mask[i]) ? cats[i] : n_groups;
        if (k < 0 || k > n_groups) k = n_groups;  // sorts last, excluded
        keys[i] = k;
        vals[i] = i;
        if (k < n_groups) atomicAdd(&counts[k], 1);
    }
}

/// Sort rows by category (masked-out or invalid rows are dropped) so a grouped
/// reduction can hand each block rows of a single group. All device scratch,
/// including the returned row order, lives in `pool`.
inline GroupedRows build_grouped_rows(RmmScratchPool& pool, const int* cats,
                                      const bool* mask, int n_rows,
                                      int n_groups, cudaStream_t stream) {
    GroupedRows g;
    g.offsets.assign((size_t)n_groups + 1, 0);
    if (n_rows <= 0 || n_groups <= 0) return g;
    int* keys = pool.alloc<int>((size_t)n_rows);
    int* keys_sorted = pool.alloc<int>((size_t)n_rows);
    int* vals = pool.alloc<int>((size_t)n_rows);
    int* counts = pool.alloc<int>((size_t)n_groups);
    int* row_order = pool.alloc<int>((size_t)n_rows);
    g.row_order = row_order;

    cuda_check(
        cudaMemsetAsync(counts, 0, (size_t)n_groups * sizeof(int), stream),
        "cudaMemsetAsync(group counts)");
    group_keys_kernel<<<strided_grid(n_rows, ATOMIC_BLOCK_THREADS),
                        ATOMIC_BLOCK_THREADS, 0, stream>>>(
        cats, mask, n_rows, n_groups, keys, vals, counts);
    CUDA_CHECK_LAST_ERROR(group_keys_kernel);

    int end_bit = 1;
    while ((1 << end_bit) <= n_groups) ++end_bit;
    size_t temp_bytes = 0;
    cuda_check(cub::DeviceRadixSort::SortPairs(nullptr, temp_bytes, keys,
                                               keys_sorted, vals, row_order,
                                               n_rows, 0, end_bit, stream),
               "cub::DeviceRadixSort::SortPairs(size)");
    void* temp = pool.alloc<char>(temp_bytes);
    cuda_check(cub::DeviceRadixSort::SortPairs(temp, temp_bytes, keys,
                                               keys_sorted, vals, row_order,
                                               n_rows, 0, end_bit, stream),
               "cub::DeviceRadixSort::SortPairs");

    std::vector<int> h_counts((size_t)n_groups);
    cuda_check(
        cudaMemcpyAsync(h_counts.data(), counts, (size_t)n_groups * sizeof(int),
                        cudaMemcpyDeviceToHost, stream),
        "cudaMemcpyAsync(group counts)");
    cuda_check(cudaStreamSynchronize(stream), "cudaStreamSynchronize");
    for (int k = 0; k < n_groups; ++k) {
        g.offsets[(size_t)k + 1] = g.offsets[(size_t)k] + h_counts[(size_t)k];
    }
    g.n_active = g.offsets[(size_t)n_groups];
    return g;
}

/// Split each group's run of rows into blocks of at most `rows_per_block`.
inline std::vector<BlockRows> block_rows_by_group(
    const std::vector<int>& offsets, int rows_per_block) {
    std::vector<BlockRows> blocks;
    for (size_t k = 0; k + 1 < offsets.size(); ++k) {
        for (int first = offsets[k]; first < offsets[k + 1];
             first += rows_per_block) {
            blocks.push_back({first,
                              std::min(first + rows_per_block, offsets[k + 1]),
                              (int)k});
        }
    }
    return blocks;
}

// ---- host entry points
// -------------------------------------------------------

/// Run an Op over the minor axis. Returns whether unsorted rows were detected,
/// so the caller can remember it and skip the bookmark sweep next time.
template <typename IdxT, typename Op>
bool minor_reduce(const IdxT* indptr, const IdxT* indices, Op op, int n_rows,
                  int n_cols, long long nnz, bool assume_unsorted,
                  cudaStream_t stream, const GroupedRows* groups = nullptr) {
    if (n_rows <= 0 || n_cols <= 0 || nnz <= 0) return false;
    if (groups != nullptr && groups->n_active == 0) return false;
    const int n_groups =
        groups != nullptr ? (int)groups->offsets.size() - 1 : 1;
    const TilePlan plan =
        plan_tiles(nnz, groups != nullptr ? groups->n_active : n_rows, n_cols,
                   Op::bytes_per_col);
    op.tile_size = plan.tile_size;

    // Which rows each block owns: contiguous chunks, or per-group chunks of the
    // sorted row order when grouped.
    RmmScratchPool pool;  // flag + block table; released after the launches
    const int* row_order = groups != nullptr ? groups->row_order : nullptr;
    const BlockRows* per_block_rows = nullptr;
    unsigned n_blocks =
        (unsigned)((n_rows + plan.rows_per_block - 1) / plan.rows_per_block);
    if (groups != nullptr) {
        const std::vector<BlockRows> h_blocks =
            block_rows_by_group(groups->offsets, plan.rows_per_block);
        BlockRows* d_blocks = pool.alloc<BlockRows>(h_blocks.size());
        cuda_check(cudaMemcpyAsync(d_blocks, h_blocks.data(),
                                   h_blocks.size() * sizeof(BlockRows),
                                   cudaMemcpyHostToDevice, stream),
                   "cudaMemcpyAsync(block rows)");
        per_block_rows = d_blocks;
        n_blocks = (unsigned)h_blocks.size();
    }

    auto launch_atomic = [&]() {
        if (groups != nullptr || Op::needs_rows) {
            atomic_per_row_kernel<IdxT, Op>
                <<<n_blocks, ATOMIC_BLOCK_THREADS, 0, stream>>>(
                    indptr, indices, row_order, per_block_rows, op, n_rows,
                    plan.rows_per_block);
            CUDA_CHECK_LAST_ERROR(atomic_per_row_kernel);
        } else {
            atomic_per_nonzero_kernel<IdxT, Op>
                <<<strided_grid(nnz, ATOMIC_BLOCK_THREADS),
                   ATOMIC_BLOCK_THREADS, 0, stream>>>(indices, op, nnz);
            CUDA_CHECK_LAST_ERROR(atomic_per_nonzero_kernel);
        }
    };
    auto launch_tiled = [&](bool rescan, int* out_of_order_flag) {
        opt_in_dynamic_smem(tile_sweep_kernel<IdxT, Op>, plan.smem_bytes,
                            "cudaFuncSetAttribute(tile_sweep_kernel)");
        tile_sweep_kernel<IdxT, Op>
            <<<n_blocks, SWEEP_BLOCK_THREADS, plan.smem_bytes, stream>>>(
                indptr, indices, row_order, per_block_rows, op,
                out_of_order_flag, n_rows, n_cols, plan.n_tiles,
                plan.rows_per_block, rescan);
        CUDA_CHECK_LAST_ERROR(tile_sweep_kernel);
    };
    const bool few_tiles = plan.n_tiles <= SWEEP_MAX_RESCAN_TILES;

    // 1. The planner may have decided tiles do not pay off here.
    if (!plan.use_tiled) {
        launch_atomic();
        return false;
    }
    // 2. Rows already known to be unsorted: skip the bookmark sweep.
    if (assume_unsorted) {
        if (few_tiles) {
            launch_tiled(true, nullptr);
        } else {
            launch_atomic();
        }
        return false;
    }
    // 3. The bookmark sweep, with out-of-order detection.
    int* flag = pool.alloc<int>(1);
    cuda_check(cudaMemsetAsync(flag, 0, sizeof(int), stream),
               "cudaMemsetAsync(out-of-order flag)");
    launch_tiled(false, flag);
    if (plan.n_tiles == 1) return false;  // one tile cannot skip anything
    // 4. Did a row turn out to be unsorted? Then redo it order-agnostically.
    int h_flag = 0;
    cuda_check(cudaMemcpyAsync(&h_flag, flag, sizeof(int),
                               cudaMemcpyDeviceToHost, stream),
               "cudaMemcpyAsync(out-of-order flag)");
    cuda_check(cudaStreamSynchronize(stream), "cudaStreamSynchronize");
    if (h_flag == 0) return false;
    op.zero_outputs(n_cols, n_groups, stream);
    if (few_tiles) {
        launch_tiled(true, nullptr);
    } else {
        launch_atomic();
    }
    return true;
}

/// Order-agnostic minor-axis reduction without indptr (one atomic per nonzero).
template <typename IdxT, typename Op>
void minor_reduce_flat(const IdxT* indices, Op op, long long nnz,
                       cudaStream_t stream) {
    if (nnz <= 0) return;
    atomic_per_nonzero_kernel<IdxT, Op>
        <<<strided_grid(nnz, ATOMIC_BLOCK_THREADS), ATOMIC_BLOCK_THREADS, 0,
           stream>>>(indices, op, nnz);
    CUDA_CHECK_LAST_ERROR(atomic_per_nonzero_kernel);
}

// ---- major-axis companion
// -----------------------------------------------------

/// Per compressed row: the sum (optionally over masked columns only) and the
/// number of stored entries. Warp per row, no atomics.
template <typename T, typename IdxT>
__global__ void row_reduce_kernel(const IdxT* __restrict__ indptr,
                                  const IdxT* __restrict__ indices,
                                  const T* __restrict__ data,
                                  const bool* __restrict__ col_mask,
                                  T* __restrict__ sums,
                                  int* __restrict__ counts, int n_rows) {
    const int lane = threadIdx.x & (WARP - 1);
    const int warps_total = (gridDim.x * blockDim.x) / WARP;
    for (int row = (blockIdx.x * blockDim.x + threadIdx.x) / WARP; row < n_rows;
         row += warps_total) {
        const IdxT row_end = indptr[row + 1];
        double acc = 0.0;
        int cnt = 0;
        for (IdxT nnz_pos = indptr[row] + lane; nnz_pos < row_end;
             nnz_pos += WARP) {
            if (col_mask != nullptr && !col_mask[indices[nnz_pos]]) continue;
            acc += static_cast<double>(data[nnz_pos]);
            ++cnt;
        }
#pragma unroll
        for (int o = WARP / 2; o > 0; o >>= 1) {
            acc += __shfl_down_sync(0xffffffffu, acc, o);
            cnt += __shfl_down_sync(0xffffffffu, cnt, o);
        }
        if (lane == 0) {
            if (sums != nullptr) sums[row] = static_cast<T>(acc);
            if (counts != nullptr) counts[row] = cnt;
        }
    }
}

template <typename T, typename IdxT>
inline void row_reduce(const IdxT* indptr, const IdxT* indices, const T* data,
                       const bool* col_mask, T* sums, int* counts, int n_rows,
                       cudaStream_t stream) {
    if (n_rows <= 0) return;
    row_reduce_kernel<T, IdxT>
        <<<strided_grid((long long)n_rows * WARP, ATOMIC_BLOCK_THREADS),
           ATOMIC_BLOCK_THREADS, 0, stream>>>(indptr, indices, data, col_mask,
                                              sums, counts, n_rows);
    CUDA_CHECK_LAST_ERROR(row_reduce_kernel);
}

// ---- Ops shared by several modules -----------------------------------------

/// Sum of values per column, optionally restricted to active rows.
template <typename T>
struct MinorSumOp {
    const T* data;
    T* out;
    const bool* row_mask;  // nullable
    int tile_size;
    static constexpr size_t bytes_per_col = sizeof(double);
    static constexpr bool needs_rows = true;
    __device__ bool row_active(int row) const {
        return row_mask == nullptr || row_mask[row];
    }
    __device__ void zero_col(char* pad, int tile_col, int) const {
        reinterpret_cast<double*>(pad)[tile_col] = 0.0;
    }
    __device__ void add(char* pad, long long nnz_pos, int tile_col) const {
        atomicAdd(&reinterpret_cast<double*>(pad)[tile_col],
                  static_cast<double>(data[nnz_pos]));
    }
    __device__ void flush_col(const char* pad, int, int col,
                              int tile_col) const {
        const double s = reinterpret_cast<const double*>(pad)[tile_col];
        if (s != 0.0) atomicAdd(&out[col], static_cast<T>(s));
    }
    __device__ void add_global(long long nnz_pos, int col, int) const {
        atomicAdd(&out[col], data[nnz_pos]);
    }
    void zero_outputs(int n_cols, int, cudaStream_t stream) const {
        cuda_check(cudaMemsetAsync(out, 0, (size_t)n_cols * sizeof(T), stream),
                   "cudaMemsetAsync(MinorSumOp outputs)");
    }
};

/// Number of stored entries per column.
struct MinorCountOp {
    int* out;
    int n_cols;
    int tile_size;
    static constexpr size_t bytes_per_col = sizeof(int);
    static constexpr bool needs_rows = false;
    __device__ bool row_active(int) const {
        return true;
    }
    __device__ void zero_col(char* pad, int tile_col, int) const {
        reinterpret_cast<int*>(pad)[tile_col] = 0;
    }
    __device__ void add(char* pad, long long, int tile_col) const {
        atomicAdd(&reinterpret_cast<int*>(pad)[tile_col], 1);
    }
    __device__ void flush_col(const char* pad, int, int col,
                              int tile_col) const {
        const int c = reinterpret_cast<const int*>(pad)[tile_col];
        if (c != 0) atomicAdd(&out[col], c);
    }
    __device__ void add_global(long long, int col, int) const {
        if (col >= 0 && col < n_cols) atomicAdd(&out[col], 1);
    }
    void zero_outputs(int n_cols_, int, cudaStream_t stream) const {
        cuda_check(
            cudaMemsetAsync(out, 0, (size_t)n_cols_ * sizeof(int), stream),
            "cudaMemsetAsync(MinorCountOp outputs)");
    }
};
