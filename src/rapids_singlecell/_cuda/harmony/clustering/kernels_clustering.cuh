#pragma once

#include <cub/block/block_scan.cuh>
#include <algorithm>
#include <cmath>
#include <cstring>
#include <type_traits>
#include <cuda_runtime.h>

#include "../fixed_point.cuh"
#include "../scatter/kernels_scatter_reduce.cuh"
#include "../storage.cuh"

// ---- Update blocks (see draw_blocks) ----
// PCG hash of each run's global index (`first` + local run), local run index.
__global__ void pcg_hash_kernel(unsigned int* __restrict__ out,
                                int* __restrict__ indices, int n,
                                unsigned int first, unsigned int seed) {
    long long stride = (long long)blockDim.x * gridDim.x;
    for (long long i = (long long)blockIdx.x * blockDim.x + threadIdx.x; i < n;
         i += stride) {
        unsigned int state = (first + (unsigned int)i) ^ seed;
        state = state * 747796405u + 2891336453u;
        unsigned int word =
            ((state >> ((state >> 28u) + 4u)) ^ state) * 277803737u;
        out[i] = (word >> 22u) ^ word;
        indices[i] = (int)i;
    }
}

// keys[j] = unit of run runs[j].
__global__ void run_unit_kernel(const int* __restrict__ runs,
                                unsigned int* __restrict__ keys, int n,
                                int unit_runs) {
    long long stride = (long long)blockDim.x * gridDim.x;
    for (long long j = (long long)blockIdx.x * blockDim.x + threadIdx.x; j < n;
         j += stride)
        keys[j] = runs[j] / unit_runs;
}

// Runs sorted by (unit, hash) are dealt to the blocks in turn within each
// unit: run_block[runs[j]] = (rank in its unit) % n_blocks.
__global__ void deal_runs_kernel(const int* __restrict__ runs,
                                 unsigned int* __restrict__ run_block, int n,
                                 int unit_runs, int n_blocks) {
    long long stride = (long long)blockDim.x * gridDim.x;
    for (long long j = (long long)blockIdx.x * blockDim.x + threadIdx.x; j < n;
         j += stride)
        run_block[runs[j]] = (j % unit_runs) % n_blocks;
}

// keys[i] = block * n_groups + group of cell i; cells[i] = i.
__global__ void block_category_keys_kernel(const unsigned int* run_block,
                                           const int* groups,
                                           unsigned int* keys, int* cells,
                                           int n_cells, int chunk,
                                           int n_groups) {
    long long stride = (long long)blockDim.x * gridDim.x;
    for (long long i = (long long)blockIdx.x * blockDim.x + threadIdx.x;
         i < n_cells; i += stride) {
        keys[i] = run_block[i / chunk] * n_groups + groups[i];
        cells[i] = (int)i;
    }
}

constexpr int KMEANS_WEIGHT_TILE_ROWS = 1024;
constexpr int KMEANS_WEIGHT_THREADS = 256;

template <typename T>
__global__ void kmeans_weight_tiles_kernel(const T* values, double* totals,
                                           int n_rows) {
    size_t start = (size_t)blockIdx.x * KMEANS_WEIGHT_TILE_ROWS;
    double sum = 0;
    for (int j = threadIdx.x; j < KMEANS_WEIGHT_TILE_ROWS; j += blockDim.x)
        if (start + j < n_rows) sum += (double)values[start + j];
    sum = block_sum(sum);
    if (threadIdx.x == 0) totals[blockIdx.x] = sum;
}

// First index whose running sum of `values` exceeds `target`, scanning
// blocks of KMEANS_WEIGHT_THREADS in order with a block-wide inclusive scan.
// Returns n when no prefix exceeds the target; `*before` receives the sum of
// the values preceding the returned index.
template <typename V>
__device__ int kmeans_first_exceeding(const V* values, int n, double target,
                                      double* before) {
    using Scan = cub::BlockScan<double, KMEANS_WEIGHT_THREADS>;
    __shared__ typename Scan::TempStorage scan;
    __shared__ int found;
    __shared__ double found_before, carry;
    __syncthreads();  // a previous call may still read the shared results
    if (threadIdx.x == 0) found = n, carry = 0, found_before = 0;
    __syncthreads();
    for (int begin = 0; begin < n; begin += KMEANS_WEIGHT_THREADS) {
        int i = begin + threadIdx.x;
        double value = i < n ? (double)values[i] : 0.0;
        double inclusive;
        Scan(scan).InclusiveSum(value, inclusive);
        inclusive += carry;
        if (i < n && inclusive > target) atomicMin(&found, i);
        __syncthreads();
        if (found == i) found_before = inclusive - value;
        if (threadIdx.x == KMEANS_WEIGHT_THREADS - 1) carry = inclusive;
        __syncthreads();
        if (found < n) break;
    }
    *before = found_before;
    return found;
}

template <typename T>
__global__ void kmeans_select_center_kernel(const T* X, const T* weights,
                                            const double* totals,
                                            const double* uniforms, T* centers,
                                            int* n_draws, int n_rows,
                                            int n_cols, int n_tiles,
                                            int cluster) {
    __shared__ double total_s;
    __shared__ int choice;
    double total = 0;
    for (int tile = threadIdx.x; tile < n_tiles; tile += blockDim.x)
        total += totals[tile];
    total = block_sum(total);
    if (threadIdx.x == 0) total_s = total, choice = cluster % n_rows;
    __syncthreads();
    total = total_s;
    if (total > 0) {
        double draw = fmin(uniforms[*n_draws] * total, nextafter(total, 0.));
        __syncthreads();
        if (threadIdx.x == 0) ++*n_draws;
        double previous;
        int tile = kmeans_first_exceeding(totals, n_tiles, draw, &previous);
        if (tile == n_tiles) tile = n_tiles - 1;
        int begin = tile * KMEANS_WEIGHT_TILE_ROWS;
        int length = min(KMEANS_WEIGHT_TILE_ROWS, n_rows - begin);
        double unused;
        int row = kmeans_first_exceeding(weights + begin, length,
                                         draw - previous, &unused);
        if (row == length) {
            // Tree and sequential leaf sums can differ by a final ulp: take
            // the tile's last positive weight.
            __shared__ int last_positive;
            if (threadIdx.x == 0) last_positive = 0;
            __syncthreads();
            for (int i = threadIdx.x; i < length; i += blockDim.x)
                if (weights[begin + i] > 0) atomicMax(&last_positive, i);
            __syncthreads();
            row = last_positive;
        }
        if (threadIdx.x == 0) choice = begin + row;
    }
    __syncthreads();
    for (int col = threadIdx.x; col < n_cols; col += blockDim.x)
        centers[(size_t)cluster * n_cols + col] =
            X[(size_t)choice * n_cols + col];
}

// ---- Exact fixed-point sums (see fixed_point.cuh) ----

// acc += fixed-point sum of the per-cell objective terms.
template <typename T>
__global__ void objective_fixed_kernel(const T* __restrict__ partial,
                                       long long n_rows, T scale,
                                       unsigned long long* acc) {
    long long sum = 0;
    for (long long i = (long long)blockIdx.x * blockDim.x + threadIdx.x;
         i < n_rows; i += (long long)gridDim.x * blockDim.x)
        sum += to_fixed(partial[i], scale);
    sum = warp_sum(sum);
    if ((threadIdx.x & 31) == 0) atomicAdd(acc, (unsigned long long)sum);
}

// total = sum of the n_blocks block counts (n values each).
__global__ void sum_blocks_kernel(const long long* __restrict__ blocks,
                                  int n_blocks, long long* __restrict__ total,
                                  int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    long long t = 0;
    for (int b = 0; b < n_blocks; ++b) t += blocks[(size_t)b * n + i];
    total[i] = t;
}

// counts[level K + k] += the fixed-point counts of column k of R, as the
// assignment counts them, over the cells idx[begin .. end): to the level of
// each cell's group (`groups`), or with several keys to each level of its
// joint category (`joint_cats`). A thread sums one column over
// RECOUNT_CELLS positions, loaded before they are summed.
constexpr int RECOUNT_CELLS = 8;
template <typename T, typename RT>
__global__ void recount_kernel(const RT* __restrict__ R, int K,
                               const int* __restrict__ idx, int begin, int end,
                               const int* __restrict__ groups,
                               const int* __restrict__ joint_cats, int n_cov,
                               T scale, unsigned long long* counts) {
    long long n =
        (long long)(end - begin + RECOUNT_CELLS - 1) / RECOUNT_CELLS * K;
    for (long long t = blockIdx.x * (long long)blockDim.x + threadIdx.x; t < n;
         t += (long long)gridDim.x * blockDim.x) {
        int k = (int)(t % K), p = begin + (int)(t / K) * RECOUNT_CELLS;
        int group[RECOUNT_CELLS];
        long long value[RECOUNT_CELLS];
#pragma unroll
        for (int j = 0; j < RECOUNT_CELLS; ++j) {
            int cell = p + j < end ? idx[p + j] : -1;
            group[j] = cell < 0 ? -1 : groups[cell];
            value[j] =
                cell < 0 ? 0
                         : count_to_fixed(
                               from_storage<T>(R[(size_t)cell * K + k]), scale);
        }
        long long sum = 0;
        auto flush = [&](int g) {
            for (int c = 0; sum && c < (joint_cats ? n_cov : 1); ++c) {
                int level = joint_cats ? joint_cats[(size_t)g * n_cov + c] : g;
                atomicAdd(counts + (size_t)level * K + k,
                          (unsigned long long)sum);
            }
            sum = 0;
        };
#pragma unroll
        for (int j = 0; j < RECOUNT_CELLS; ++j) {
            if (j && group[j] != group[j - 1]) flush(group[j - 1]);
            sum += value[j];
        }
        flush(group[RECOUNT_CELLS - 1]);
    }
}

// ---- Assignment (no similarity matrix) ----
constexpr int FUSED_MAX_CLUSTER_SLOTS = 4;  // K <= 128
constexpr int FUSED_THREADS = 256;

// E and the log penalty from the held-out counts; R_sum is the column sum of O.
// One CTA per cluster reduces its column in a fixed tree, then each thread
// fills its batches. `penalty` may be null to update E only.
template <typename T>
__global__ void penalty_from_counts_kernel(const T* O, const T* Pr_b,
                                           const T* theta, T* E, T* penalty,
                                           int n_batches, int n_clusters,
                                           int n_first, bool stabilized) {
    // Each cell counts once per batch key: the cluster total sums the levels
    // of the first key only.
    int k = blockIdx.x;
    T r_sum = T(0);
    for (int b = threadIdx.x; b < n_first; b += blockDim.x)
        r_sum += O[(size_t)b * n_clusters + k];
    r_sum = block_sum(r_sum);
    for (int b = threadIdx.x; b < n_batches; b += blockDim.x) {
        size_t i = (size_t)b * n_clusters + k;
        T e = Pr_b[b] * r_sum;
        E[i] = e;
        if (penalty) {
            T denom = stabilized ? (O[i] + e + T(1)) : (O[i] + T(1));
            penalty[i] = theta[b] * log((e + T(1)) / denom);  // log penalty
        }
    }
}

// One CTA per assignment tile (cells of one batch within the block; several
// keys: any cells of the block, see JOINT below). Each warp holds FUSED_ROWS
// cells of a group and each lane four consecutive clusters: similarities to
// the normalized centroids, penalized soft assignment written to R in place,
// and the cell's distance + entropy objective term. The new column sums
// (fixed point) are added to the counts of the batch (several keys: of each
// level of a joint run); integer atomics are exact, so the order does not
// matter.
constexpr int FUSED_ROWS = 8;
constexpr int FUSED_CLUSTER_STRIDE = 32 * FUSED_MAX_CLUSTER_SLOTS;
constexpr int FUSED_GROUP = FUSED_THREADS / 32 * FUSED_ROWS;
// A group's cells and the transposed centroids reach shared memory
// STREAM_BYTES of each row at a time, double buffered; the dot products stay
// in registers across chunks. Any number of PCs fits in 48 KB (no opt-in).
constexpr int STREAM_BYTES = 128;
constexpr int STREAM_SMEM =
    2 * (FUSED_GROUP + FUSED_CLUSTER_STRIDE) * STREAM_BYTES;
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ < 800
constexpr int FUSED_CTAS = 1;  // 64 KB of shared memory hold one CTA
#else
constexpr int FUSED_CTAS = 2;  // float: the register cap pays for itself
#endif

__host__ __device__ inline int fused_padded_pcs(int n_pcs) {
    return (n_pcs + 3) / 4 * 4;
}

// Whether the centroids (padded PCs) fit in shared memory next to two row
// stages: up to 64 float or 32 double PCs.
template <typename T>
__host__ __device__ inline bool fused_resident(int pcs) {
    return (size_t)pcs * FUSED_CLUSTER_STRIDE +
               2 * FUSED_GROUP * (STREAM_BYTES / sizeof(T)) <=
           STREAM_SMEM / sizeof(T);
}

// V-byte word, and dst = V bytes at src (or zeros unless ok) by cp.async
// (sm_80 on), which copy_wait waits for.
template <int V>
using Piece =
    std::conditional_t<V == 16, int4, std::conditional_t<V == 8, int2, int> >;
template <int V>
__device__ __forceinline__ void copy_async(void* dst, const void* src,
                                           bool ok) {
#if __CUDA_ARCH__ >= 800
    unsigned s = (unsigned)__cvta_generic_to_shared(dst), n = ok ? V : 0;
    if constexpr (V == 16)  // 16 bytes bypass L1
        asm volatile("cp.async.cg.shared.global [%0], [%1], 16, %2;" ::"r"(s),
                     "l"(src), "r"(n)
                     : "memory");
    else
        asm volatile("cp.async.ca.shared.global [%0], [%1], %2, %3;" ::"r"(s),
                     "l"(src), "n"(V), "r"(n)
                     : "memory");
#endif
}

__device__ __forceinline__ void copy_wait() {
#if __CUDA_ARCH__ >= 800
    asm volatile("cp.async.wait_all;" ::: "memory");
#endif
}

template <typename T>
__device__ inline void load4(const T* p, T (&v)[4]) {
    if constexpr (sizeof(T) == 4) {
        float4 f = *reinterpret_cast<const float4*>(p);
        v[0] = f.x, v[1] = f.y, v[2] = f.z, v[3] = f.w;
    } else {
        double2 a = reinterpret_cast<const double2*>(p)[0];
        double2 b = reinterpret_cast<const double2*>(p)[1];
        v[0] = a.x, v[1] = a.y, v[2] = b.x, v[3] = b.y;
    }
}

// p[0..3] = v in one store (bfloat16 or float); p 4-element aligned.
template <typename T>
__device__ inline void store4(T* p, const T (&v)[4]) {
    using V = std::conditional_t<sizeof(T) == 2, uint2, float4>;
    V u;
    memcpy(&u, v, sizeof(u));
    *reinterpret_cast<V*>(p) = u;
}

// dst (pcs x stride) = src (n x d)^T, zero padded; thread t0 handles
// elements t0, t0 + step, ...
template <typename T>
__device__ __forceinline__ void transpose_padded(const T* src, T* dst, int n,
                                                 int d, int pcs, int stride,
                                                 long long t0, long long step) {
    for (long long t = t0; t < (long long)pcs * stride; t += step) {
        int j = int(t / stride), k = int(t % stride);
        dst[t] = j < d && k < n ? src[(size_t)k * d + j] : T(0);
    }
}

// acc[r][q] = f(acc[r][q], x, c) over the columns d < n (a multiple of four),
// x = rows[r * pitch + d] and c = cols_t[d][k0 + q].
template <typename T, typename F>
__device__ __forceinline__ void tile_products(
    const T* rows, int pitch, const T* cols_t, int n, int k0,
    T (&acc)[FUSED_ROWS][FUSED_MAX_CLUSTER_SLOTS], F f) {
    for (int d = 0; d < n; d += 4) {
        T c[4][FUSED_MAX_CLUSTER_SLOTS];
#pragma unroll
        for (int j = 0; j < 4; ++j)
            load4(cols_t + (size_t)(d + j) * FUSED_CLUSTER_STRIDE + k0, c[j]);
#pragma unroll
        for (int r = 0; r < FUSED_ROWS; ++r) {
            T x[4];
            load4(rows + r * pitch + d, x);
#pragma unroll
            for (int j = 0; j < 4; ++j)
#pragma unroll
                for (int q = 0; q < FUSED_MAX_CLUSTER_SLOTS; ++q)
                    acc[r][q] = f(acc[r][q], x[j], c[j][q]);
        }
    }
}

// Z pieces are V bytes; y_t holds the centroids transposed (padded PCs x
// k_stride, see transpose_centroids). Every dot product is one chain over the
// PCs in order, as in tile_products, whatever the tiling. `penalty` holds log
// penalties (null: none). The log weights are shifted by the tile's largest
// log penalty, so none overflows and the row's best cluster under that
// penalty keeps at least exp(2 term); below sigma = 1/15 each row is shifted
// by its own maximum instead.
// BLOCKS (more than 128 clusters): each group takes the clusters in blocks of
// FUSED_CLUSTER_STRIDE, its rows and the block's centroids streamed as above,
// and stores the dot products in `dots_ws` (CTAs x FUSED_GROUP rows x
// k_stride); then each row is normalized from them. Column sums go through
// per-warp slices of `col_ws` (CTAs x warps x K). One block gives the bits of
// the single-block path.
// JOINT (several keys): tiles take any rows of the block, in joint category
// order (`joints` per cell). A warp follows its rows' joint categories: a
// run's log penalty sums its levels' (`joint_cats`, n_cov per category) in
// key order, and its counts go to each of its levels.
template <typename T, typename RT, int V, bool BLOCKS, bool JOINT>
__device__ __forceinline__ void fused_assign_tile(
    int tile, const T* __restrict__ Z, const T* __restrict__ y_t, int k_stride,
    const T* __restrict__ penalty, const int* __restrict__ idx,
    const int* __restrict__ offsets, const int* __restrict__ tiles,
    int n_categories, int tile_rows, RT* __restrict__ R,
    T* __restrict__ partial, unsigned long long* __restrict__ counts,
    T* __restrict__ dots_ws, long long* __restrict__ col_ws, T term, T sigma,
    T count_scale, int n_pcs, int n_clusters, const int* __restrict__ joints,
    const int* __restrict__ joint_cats, int n_cov) {
    int start, end;
    int category = scatter_tile_rows(tile, n_categories, offsets, tiles,
                                     tile_rows, start, end);

    constexpr int S = FUSED_MAX_CLUSTER_SLOTS, TM = FUSED_GROUP;
    constexpr int DC = STREAM_BYTES / sizeof(T), ZV = V / sizeof(T);
    constexpr int STAGE = (TM + FUSED_CLUSTER_STRIDE) * DC;
    int pcs = fused_padded_pcs(n_pcs);
    int lane = threadIdx.x & 31, warp = threadIdx.x >> 5;
    int n_warps = blockDim.x >> 5;
    extern __shared__ __align__(16) unsigned char smem_raw[];
    T* stages = reinterpret_cast<T*>(smem_raw);
    long long* col_sums = reinterpret_cast<long long*>(smem_raw);  // warps x K
    int k0 = 4 * lane;  // this lane's clusters k0 .. k0 + 3 of each block
    const T* pen_row =
        penalty ? penalty + (size_t)category * n_clusters : nullptr;
    const int* levels = joint_cats;  // JOINT: the warp's run's levels
    int joint = -1;                  // JOINT: the warp's run (-1: none yet)
    auto lp = [&](int k) {
        if constexpr (JOINT) {
            T sum = T(0);
            for (int c = 0; penalty && c < n_cov; ++c)
                sum += penalty[(size_t)levels[c] * n_clusters + k];
            return sum;
        }
        return pen_row ? pen_row[k] : T(0);
    };
    T log_pen[S], top = -INFINITY;
    long long col[S] = {};
    long long* cols = nullptr;  // BLOCKS: this warp's column sums
    auto refresh = [&] {        // the log penalties and their top
        top = -INFINITY;
        if constexpr (BLOCKS) {
            for (int k = lane; k < n_clusters; k += 32) top = max(top, lp(k));
        } else {
#pragma unroll
            for (int q = 0; q < S; ++q) {
                log_pen[q] = k0 + q < n_clusters ? lp(k0 + q) : T(0);
                if (k0 + q < n_clusters) top = max(top, log_pen[q]);
            }
        }
        top = warp_max(top);
    };
    // JOINT: the warp's counts of its run go to each of the run's levels.
    auto flush = [&] {
        if (joint < 0) return;
        for (int k = k0; k < n_clusters; k += FUSED_CLUSTER_STRIDE)
#pragma unroll
            for (int q = 0; q < S; ++q) {
                if (k + q >= n_clusters) break;
                long long v;
                if constexpr (BLOCKS)
                    v = cols[k + q], cols[k + q] = 0;
                else
                    v = col[q], col[q] = 0;
                for (int c = 0; v && c < n_cov; ++c)
                    atomicAdd(counts + (size_t)levels[c] * n_clusters + k + q,
                              (unsigned long long)v);
            }
    };
    if constexpr (BLOCKS) {  // this CTA's slot of the scratch
        col_ws += (size_t)blockIdx.x * n_warps * n_clusters;
        dots_ws += (size_t)blockIdx.x * TM * k_stride;
        cols = col_ws + (size_t)warp * n_clusters;
        for (int k = lane; k < n_clusters; k += 32) cols[k] = 0;
        __syncwarp();  // other lanes accumulate these columns
    }
    if constexpr (!JOINT) refresh();
    bool row_shift = term < T(-30);
    // Rows of four-aligned clusters: one store per lane and row (bfloat16
    // and float; for double the two stores cost more than they save).
    bool vec_store = sizeof(RT) <= 4 && n_clusters % 4 == 0 &&
                     reinterpret_cast<size_t>(R) % (4 * sizeof(RT)) == 0;

    // Groups of TM cells take one step per chunk of DC PCs; a step's rows
    // and centroids are staged while the previous step's products run. Up to
    // 64 float (32 double) PCs the centroids stay in shared memory for the
    // whole tile, and each step only waits for the warp's own rows. A lane
    // copies the Z pieces at column zd of its warp's rows zr + j ZR.
    int n_chunks = (pcs + DC - 1) / DC, n_groups = (end - start + TM - 1) / TM;
    int n_blocks = BLOCKS ? k_stride / FUSED_CLUSTER_STRIDE : 1;
    bool resident = !BLOCKS && fused_resident<T>(pcs);
    T* y_s = stages + 2 * TM * DC;  // resident centroids after the row stages
    auto rows_of = [&](int st) {
        return stages + st * (resident ? TM * DC : STAGE);
    };
    auto cols_of = [&](int st, int c) -> const T* {
        return resident ? y_s + (size_t)c * DC * FUSED_CLUSTER_STRIDE
                        : rows_of(st) + TM * DC;
    };
    if (resident)  // the first step's barrier waits for these copies too
        for (int e = threadIdx.x * 16 / sizeof(T);
             e < pcs * FUSED_CLUSTER_STRIDE;
             e += FUSED_THREADS * 16 / sizeof(T)) {
#if __CUDA_ARCH__ >= 800
            copy_async<16>(y_s + e, y_t + e, true);
#else
            *reinterpret_cast<Piece<16>*>(y_s + e) =
                *reinterpret_cast<const Piece<16>*>(y_t + e);
#endif
        }
    auto cell = [&](int p) { return p >= end ? -1 : idx ? idx[p] : p; };
    constexpr int ZP = FUSED_ROWS * DC / (32 * ZV), ZR = 32 * ZV / DC;
    int zr = warp * FUSED_ROWS + lane * ZV / DC, zd = lane * ZV % DC;
    int z_cell[ZP];                      // the loading group's cells (-1: none)
    int lg = 0, lb = 0, lc = 0, ls = 0;  // the next load: group, cluster
                                         // block, chunk, buffer
    // Before sm_80 a step's pieces wait in registers while the previous
    // step's products run (fetch), then go to shared memory (store).
#if __CUDA_ARCH__ >= 800
    constexpr bool async = true;
#else
    constexpr bool async = false;
#endif
    constexpr int YP = STREAM_BYTES * FUSED_CLUSTER_STRIDE / FUSED_THREADS / 16;
    Piece<16> y_reg[YP];
    Piece<V> z_reg[ZP];
    auto load = [&](bool fetch) {
        if (lg == n_groups) return;
        int d0 = lc * DC, d = d0 + zd;
        T* z_s = rows_of(ls);
#pragma unroll
        for (int i = 0; i < YP; ++i) {
            int e = (threadIdx.x + i * FUSED_THREADS) * 16 / sizeof(T);
            T* dst = z_s + TM * DC + e;
            const T* src = y_t + (size_t)d0 * FUSED_CLUSTER_STRIDE + e;
            if constexpr (BLOCKS)  // rows of cluster block lb in y_t
                src = y_t + (size_t)(d0 + e / FUSED_CLUSTER_STRIDE) * k_stride +
                      lb * FUSED_CLUSTER_STRIDE + e % FUSED_CLUSTER_STRIDE;
            if (resident || e >= min(DC, pcs - d0) * FUSED_CLUSTER_STRIDE)
                break;
            if (async)
                copy_async<16>(dst, src, true);
            else if (fetch)
                y_reg[i] = *reinterpret_cast<const Piece<16>*>(src);
            else
                *reinterpret_cast<Piece<16>*>(dst) = y_reg[i];
        }
#pragma unroll
        for (int j = 0; j < ZP; ++j) {
            if ((lc | lb) == 0 && fetch)
                z_cell[j] = cell(start + lg * TM + zr + j * ZR);
            const T* src = Z + (size_t)max(z_cell[j], 0) * n_pcs + d;
            T* dst = z_s + (zr + j * ZR) * DC + zd;
            bool ok = z_cell[j] >= 0 && d < n_pcs;  // never straddles n_pcs
            if (d >= pcs) continue;
            if (async)
                copy_async<V>(dst, src, ok);
            else if (fetch)
                z_reg[j] =
                    ok ? *reinterpret_cast<const Piece<V>*>(src) : Piece<V>{};
            else
                *reinterpret_cast<Piece<V>*>(dst) = z_reg[j];
        }
        if (async || !fetch) {
            ls ^= 1;
            if (++lc == n_chunks) {
                lc = 0;
                if (!BLOCKS || ++lb == n_blocks) lb = 0, ++lg;
            }
        }
    };
    load(true);
    if (!async) load(false);
    T dots[FUSED_ROWS][S];
    for (int g = 0, stage = 0; g < n_groups; ++g) {
        int base = start + g * TM + warp * FUSED_ROWS;
        bool active = base < end;  // warp-uniform
        int lane_cell = lane < FUSED_ROWS ? cell(base + lane) : -1;
        int lane_joint = JOINT && lane_cell >= 0 ? joints[lane_cell] : -1;
        // JOINT: row r of a new run first flushes the warp's counts.
        auto run = [&](int r) {
            if constexpr (JOINT) {
                int j = __shfl_sync(0xffffffff, lane_joint, r);
                if (j != joint) {  // warp-uniform
                    flush();
                    joint = j, levels = joint_cats + (size_t)j * n_cov;
                    refresh();
                }
            }
        };
        for (int b = 0; b < n_blocks; ++b) {
#pragma unroll
            for (int r = 0; r < FUSED_ROWS; ++r)
#pragma unroll
                for (int q = 0; q < S; ++q) dots[r][q] = T(0);
            for (int c = 0; c < n_chunks; ++c, stage ^= 1) {
                copy_wait();
                if (resident && (g | c))
                    __syncwarp();  // the warp's own rows landed
                else
                    __syncthreads();  // this step landed; the other buffer
                                      // is free
                load(true);
                if (active)
                    tile_products(rows_of(stage) + warp * FUSED_ROWS * DC, DC,
                                  cols_of(stage, c), min(DC, pcs - c * DC), k0,
                                  dots,
                                  [](T acc, T z, T y) { return acc + z * y; });
                if (!async) load(false);
            }
            if constexpr (BLOCKS) {  // braced: nvcc 12.2 rejects a bare for
                for (int r = 0; r < FUSED_ROWS && active; ++r) {
                    if (base + r >= end) break;  // warp-uniform
                    T* row = dots_ws +
                             (size_t)(warp * FUSED_ROWS + r) * k_stride +
                             b * FUSED_CLUSTER_STRIDE + k0;
#pragma unroll
                    for (int q = 0; q < S; ++q) row[q] = dots[r][q];
                }
            }
        }
        if (!active) continue;
        if constexpr (BLOCKS) {
            for (int r = 0; r < FUSED_ROWS; ++r) {
                int c = __shfl_sync(0xffffffff, lane_cell, r);
                if (c < 0) break;  // warp-uniform
                run(r);
                const T* drow =
                    dots_ws + (size_t)(warp * FUSED_ROWS + r) * k_stride;
                auto logit = [&](int k) {
                    return term * (T(1) - drow[k]) + lp(k);
                };
                T shift = top;
                if (row_shift) {  // warp-uniform
                    shift = -INFINITY;
                    for (int k = k0; k < n_clusters; k += FUSED_CLUSTER_STRIDE)
#pragma unroll
                        for (int q = 0; q < S; ++q)
                            if (k + q < n_clusters)
                                shift = max(shift, logit(k + q));
                    shift = warp_max(shift);
                }
                T sum = T(0);
                for (int k = k0; k < n_clusters; k += FUSED_CLUSTER_STRIDE)
#pragma unroll
                    for (int q = 0; q < S; ++q)
                        if (k + q < n_clusters)
                            sum += exp(logit(k + q) - shift);
                sum = warp_sum(sum);
                T inv = T(1) / sum, log_sum = log(sum);
                T distance = T(0), entropy = T(0);
                RT* row = R + (size_t)c * n_clusters;
                for (int k = k0; k < n_clusters; k += FUSED_CLUSTER_STRIDE) {
                    RT stored[S];
#pragma unroll
                    for (int q = 0; q < S; ++q)
                        stored[q] = to_storage<RT>(
                            k + q < n_clusters ? exp(logit(k + q) - shift) * inv
                                               : T(0));
                    if constexpr (sizeof(RT) <= 4)
                        if (vec_store) store4(row + k, stored);
#pragma unroll
                    for (int q = 0; q < S; ++q) {
                        if (k + q >= n_clusters) break;
                        if (!vec_store) row[k + q] = stored[q];
                        T value = from_storage<T>(stored[q]);
                        cols[k + q] += count_to_fixed(value, count_scale);
                        T om = T(1) - drow[k + q];
                        distance += value * T(2) * om;
                        entropy +=
                            value * (term * om + lp(k + q) - shift - log_sum);
                    }
                }
                T total = warp_sum(distance + sigma * entropy);
                if (lane == 0) partial[c] = total;
            }
        } else {
#pragma unroll
            for (int r = 0; r < FUSED_ROWS; ++r) {
                int c = __shfl_sync(0xffffffff, lane_cell, r);
                if (c < 0) break;  // warp-uniform
                run(r);
                T vals[S];
                T sum = T(0);
                T shift = top;
                if (row_shift) {  // warp-uniform
                    shift = -INFINITY;
#pragma unroll
                    for (int q = 0; q < S; ++q)
                        if (k0 + q < n_clusters)
                            shift = max(
                                shift, term * (T(1) - dots[r][q]) + log_pen[q]);
                    shift = warp_max(shift);
                }
#pragma unroll
                for (int q = 0; q < S; ++q) {
                    vals[q] = k0 + q < n_clusters
                                  ? exp(term * (T(1) - dots[r][q]) +
                                        log_pen[q] - shift)
                                  : T(0);
                    sum += vals[q];
                }
                sum = warp_sum(sum);
                T inv = T(1) / sum, log_sum = log(sum);
                T distance = T(0), entropy = T(0);
                RT* row = R + (size_t)c * n_clusters;
                RT stored[S];
#pragma unroll
                for (int q = 0; q < S; ++q)
                    stored[q] = to_storage<RT>(vals[q] * inv);
                if constexpr (sizeof(RT) <= 4)
                    if (vec_store && k0 < n_clusters) store4(row + k0, stored);
#pragma unroll
                for (int q = 0; q < S; ++q) {
                    int k = k0 + q;
                    if (k < n_clusters) {
                        if (!vec_store) row[k] = stored[q];
                        // Count the stored (possibly bfloat16) value.
                        T value = from_storage<T>(stored[q]);
                        col[q] += count_to_fixed(value, count_scale);
                        distance += value * T(2) * (T(1) - dots[r][q]);
                        // log(value) = log of the unnormalized weight -
                        // log(sum)
                        entropy += value * (term * (T(1) - dots[r][q]) +
                                            log_pen[q] - shift - log_sum);
                    }
                }
                T total = warp_sum(distance + sigma * entropy);
                if (lane == 0) partial[c] = total;
            }
        }
    }
    if constexpr (JOINT) {  // what is left belongs to the tile's last run
        int last = joints[cell(end - 1)];
        if (joint != last) flush();
        levels = joint_cats + (size_t)last * n_cov;
    }
    __syncthreads();  // col_sums overlays the buffers
    if constexpr (BLOCKS) {
        col_sums = col_ws;
    } else {
#pragma unroll
        for (int q = 0; q < S; ++q)
            if (k0 + q < n_clusters)
                col_sums[warp * n_clusters + k0 + q] = col[q];
        __syncthreads();
    }
    for (int k = threadIdx.x; k < n_clusters; k += blockDim.x) {
        long long sum = 0;
        for (int w = 0; w < n_warps; ++w) sum += col_sums[w * n_clusters + k];
        if constexpr (JOINT) {
            for (int c = 0; sum && c < n_cov; ++c)
                atomicAdd(counts + (size_t)levels[c] * n_clusters + k,
                          (unsigned long long)sum);
        } else {
            atomicAdd(counts + (size_t)category * n_clusters + k,
                      (unsigned long long)sum);
        }
    }
}

// One CTA per tile; with cluster blocks one per slot of the scratch (the
// resident CTAs). Each takes tile blockIdx.x, then blockIdx.x + gridDim.x (in
// launch order: one more per SM before any SM takes two), then claims tiles
// from `next_tile` (zeroed before the pass) as it finishes one, which
// evens out uneven tiles.
template <typename T, typename RT, int V, bool BLOCKS, bool JOINT>
__global__ void __launch_bounds__(FUSED_THREADS,
                                  sizeof(T) == 4 ? FUSED_CTAS : 1)
    fused_assign_kernel(
        const T* __restrict__ Z, const T* __restrict__ y_t, int k_stride,
        const T* __restrict__ penalty, const int* __restrict__ idx,
        const int* __restrict__ offsets, const int* __restrict__ tiles,
        int n_categories, int tile_rows, RT* __restrict__ R,
        T* __restrict__ partial, unsigned long long* __restrict__ counts,
        T* __restrict__ dots_ws, long long* __restrict__ col_ws, T term,
        T sigma, T count_scale, int n_pcs, int n_clusters,
        const int* __restrict__ joints, const int* __restrict__ joint_cats,
        int n_cov, int* __restrict__ next_tile) {
    // The claim waits in the last word of the second row stage: free after a
    // tile's last barrier, and filled again only after the next tile's first.
    extern __shared__ __align__(16) unsigned char smem_raw[];
    int* claimed = reinterpret_cast<int*>(smem_raw + STREAM_SMEM) - 1;
    for (int tile = blockIdx.x, n = 0; tile < tiles[n_categories]; ++n) {
        fused_assign_tile<T, RT, V, BLOCKS, JOINT>(
            tile, Z, y_t, k_stride, penalty, idx, offsets, tiles, n_categories,
            tile_rows, R, partial, counts, dots_ws, col_ws, term, sigma,
            count_scale, n_pcs, n_clusters, joints, joint_cats, n_cov);
        if constexpr (!BLOCKS) {
            break;
        } else {
            if (threadIdx.x == 0 && n > 0)
                *claimed = 2 * gridDim.x + atomicAdd(next_tile, 1);
            __syncthreads();  // the next tile reuses the scratch
            tile = n == 0 ? tile + gridDim.x : *claimed;
        }
    }
}

// y_t (padded PCs x k_stride) = Y_norm^T, zero padded.
template <typename T>
__global__ void transpose_centroids_kernel(const T* Y_norm, T* y_t, int n_pcs,
                                           int pcs, int n_clusters,
                                           int k_stride) {
    transpose_padded(Y_norm, y_t, n_clusters, n_pcs, pcs, k_stride,
                     (long long)blockIdx.x * blockDim.x + threadIdx.x,
                     (long long)blockDim.x * gridDim.x);
}

// Diversity term over O/E, added by the final objective reduction.
template <typename T>
__global__ void objective_diversity_kernel(const T* O, const T* E,
                                           const T* theta, T sigma,
                                           int n_batches, int n_clusters,
                                           bool stabilized, T* out) {
    T diversity = T(0);
    for (size_t i = threadIdx.x; i < (size_t)n_batches * n_clusters;
         i += blockDim.x) {
        T numerator = stabilized ? O[i] + E[i] + T(1) : O[i] + T(1);
        diversity += sigma * theta[i / n_clusters] * O[i] *
                     log(numerator / (E[i] + T(1)));
    }
    diversity = block_sum(diversity);
    if (threadIdx.x == 0) *out = diversity;
}

// ---- k-means initialization ----
// Exact squared distances of FUSED_ROWS cells per warp to all centers
// (lane owns four consecutive centers), with the first minimum as label.
template <typename T>
__global__ void kmeans_assign_kernel(const T* __restrict__ X,
                                     const T* __restrict__ centers,
                                     int* __restrict__ labels,
                                     T* __restrict__ minimum, int n_rows,
                                     int n_cols, int n_clusters) {
    constexpr int S = FUSED_MAX_CLUSTER_SLOTS;
    int cols = fused_padded_pcs(n_cols);
    int lane = threadIdx.x & 31, warp = threadIdx.x >> 5;
    int n_warps = blockDim.x >> 5;
    extern __shared__ __align__(16) unsigned char smem_raw[];
    T* c_t = reinterpret_cast<T*>(smem_raw);  // cols x FUSED_CLUSTER_STRIDE
    T* x_w = c_t + (size_t)cols * FUSED_CLUSTER_STRIDE +
             (size_t)warp * FUSED_ROWS * cols;
    transpose_padded(centers, c_t, n_clusters, n_cols, cols,
                     FUSED_CLUSTER_STRIDE, threadIdx.x, blockDim.x);
    __syncthreads();
    int k0 = 4 * lane;
    long long stride = (long long)gridDim.x * n_warps * FUSED_ROWS;
    for (long long base = ((long long)blockIdx.x * n_warps + warp) * FUSED_ROWS;
         base < n_rows; base += stride) {
        int rows = (int)min((long long)FUSED_ROWS, n_rows - base);
        __syncwarp();
        for (int r = 0; r < FUSED_ROWS; ++r)
            for (int d = lane; d < cols; d += 32)
                x_w[r * cols + d] = r < rows && d < n_cols
                                        ? X[(size_t)(base + r) * n_cols + d]
                                        : T(0);
        __syncwarp();
        T dist[FUSED_ROWS][S] = {};
        tile_products(x_w, cols, c_t, cols, k0, dist, [](T acc, T x, T c) {
            T delta = x - c;
            return acc + delta * delta;
        });
#pragma unroll
        for (int r = 0; r < FUSED_ROWS; ++r) {
            if (r >= rows) break;  // warp-uniform
            T best = T(0);
            int label = INT_MAX;
#pragma unroll
            for (int q = 0; q < S; ++q) {
                int k = k0 + q;
                if (k < n_clusters && (label == INT_MAX || dist[r][q] < best)) {
                    best = dist[r][q];
                    label = k;
                }
            }
#pragma unroll
            for (int offset = 16; offset > 0; offset >>= 1) {
                T other = __shfl_xor_sync(0xffffffff, best, offset);
                int other_label = __shfl_xor_sync(0xffffffff, label, offset);
                if (other_label != INT_MAX &&
                    (label == INT_MAX || other < best ||
                     (other == best && other_label < label))) {
                    best = other;
                    label = other_label;
                }
            }
            if (lane == 0) {
                labels[base + r] = label;
                minimum[base + r] = best;
            }
        }
    }
}

// closest = min(closest, |x - center|^2), one warp per row.
template <typename T>
__global__ void kmeans_closest_kernel(const T* __restrict__ X,
                                      const T* __restrict__ center,
                                      T* __restrict__ closest, int n_rows,
                                      int n_cols) {
    int lane = threadIdx.x & 31;
    long long warps = ((long long)gridDim.x * blockDim.x) >> 5;
    for (long long i = ((long long)blockIdx.x * blockDim.x + threadIdx.x) >> 5;
         i < n_rows; i += warps) {
        T sum = T(0);
        for (int d = lane; d < n_cols; d += 32) {
            T delta = X[(size_t)i * n_cols + d] - center[d];
            sum += delta * delta;
        }
        sum = warp_sum(sum);
        if (lane == 0 && sum < closest[i]) closest[i] = sum;
    }
}

// dst = src with rows scaled to unit L2 norm (inv_norm_scale); one block of
// whole warps per row, src == dst allowed.
template <typename T>
__global__ void l2_row_normalize_kernel(const T* src, T* dst, int n_cols) {
    const T* s = src + (size_t)blockIdx.x * n_cols;
    T* d = dst + (size_t)blockIdx.x * n_cols;
    T acc = T(0);
    for (int col = threadIdx.x; col < n_cols; col += blockDim.x)
        acc += s[col] * s[col];
    T scale = inv_norm_scale(block_sum(acc));
    for (int col = threadIdx.x; col < n_cols; col += blockDim.x)
        d[col] = s[col] * scale;
}
