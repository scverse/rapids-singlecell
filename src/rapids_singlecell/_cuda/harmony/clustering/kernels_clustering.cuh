#pragma once

#include <cub/block/block_scan.cuh>
#include <cmath>
#include <cuda_runtime.h>

#include "../storage.cuh"

// ---- PCG hash for random shuffle keys ----
__global__ void pcg_hash_kernel(unsigned int* __restrict__ out,
                                int* __restrict__ indices, int n,
                                unsigned int seed, int chunk = 1) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;
    for (int i = idx; i < n; i += stride) {
        unsigned int state = static_cast<unsigned int>(i / chunk) ^ seed;
        state = state * 747796405u + 2891336453u;
        unsigned int word =
            ((state >> ((state >> 28u) + 4u)) ^ state) * 277803737u;
        out[i] = (word >> 22u) ^ word;
        indices[i] = i;
    }
}

constexpr int KMEANS_WEIGHT_TILE_ROWS = 1024;
constexpr int KMEANS_WEIGHT_THREADS = 256;

template <typename T>
__global__ void kmeans_weight_tiles_kernel(const T* values, double* totals,
                                           int n_rows) {
    __shared__ double partial[KMEANS_WEIGHT_THREADS];
    int tid = threadIdx.x;
    size_t start = (size_t)blockIdx.x * KMEANS_WEIGHT_TILE_ROWS;
    double sum = 0;
    for (int j = tid; j < KMEANS_WEIGHT_TILE_ROWS; j += KMEANS_WEIGHT_THREADS)
        if (start + j < n_rows) sum += (double)values[start + j];
    partial[tid] = sum;
    __syncthreads();
    for (int step = KMEANS_WEIGHT_THREADS / 2; step; step >>= 1) {
        if (tid < step) partial[tid] += partial[tid + step];
        __syncthreads();
    }
    if (tid == 0) totals[blockIdx.x] = partial[0];
}

template <typename T>
__device__ T objective_block_sum(T value, bool broadcast = true) {
    __shared__ T warp_sums[32];
    int lane = threadIdx.x & 31;
    int warp = threadIdx.x >> 5;
    for (int offset = 16; offset > 0; offset >>= 1)
        value += __shfl_down_sync(0xffffffff, value, offset);
    if (lane == 0) warp_sums[warp] = value;
    __syncthreads();
    value = threadIdx.x < (blockDim.x >> 5) ? warp_sums[lane] : T(0);
    if (warp == 0)
        for (int offset = 16; offset > 0; offset >>= 1)
            value += __shfl_down_sync(0xffffffff, value, offset);
    if (!broadcast) return value;
    if (threadIdx.x == 0) warp_sums[0] = value;
    __syncthreads();
    value = warp_sums[0];
    __syncthreads();
    return value;
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
    total = objective_block_sum<double>(total);
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

template <typename T>
__global__ void objective_reduce_kernel(const T* partial, int n_rows, T* out) {
    T sum = T(0);
    size_t i = threadIdx.x;
    for (; i + 3 * blockDim.x < n_rows; i += 4 * blockDim.x) {
        T first = partial[i], second = partial[i + blockDim.x],
          third = partial[i + 2 * blockDim.x],
          fourth = partial[i + 3 * blockDim.x];
        sum = (((sum + first) + second) + third) + fourth;
    }
    for (; i < n_rows; i += blockDim.x) sum += partial[i];
    sum = objective_block_sum(sum, false);
    if (threadIdx.x == 0) *out = sum;
}

constexpr int OBJECTIVE_REDUCE_BLOCKS = 1024;

// Fixed-shape two-stage sum: deterministic for a given n_rows.
template <typename T>
__global__ void objective_stage_kernel(const T* partial, long long n_rows,
                                       T* stage) {
    long long chunk = (n_rows + gridDim.x - 1) / gridDim.x;
    long long begin = blockIdx.x * chunk;
    long long end = min(n_rows, begin + chunk);
    T sum = T(0);
    for (long long i = begin + threadIdx.x; i < end; i += blockDim.x)
        sum += partial[i];
    sum = objective_block_sum(sum, false);
    if (threadIdx.x == 0) stage[blockIdx.x] = sum;
}

// ---- Fused single-covariate clustering (no similarity matrix) ----
constexpr int FUSED_MAX_CLUSTER_SLOTS = 4;  // K <= 128
constexpr int FUSED_THREADS = 256;

// Block slot of position j and the cell's category, packed into one sort key.
// Sorting by it keeps the exact block sizes of the shuffle and groups each
// block's cells by category for the tiled scatter.
__global__ void block_category_keys_kernel(const int* perm, const int* cats,
                                           unsigned int* keys, int n_cells,
                                           int block_size, int n_batches) {
    int stride = blockDim.x * gridDim.x;
    for (int j = blockIdx.x * blockDim.x + threadIdx.x; j < n_cells;
         j += stride)
        keys[j] = (unsigned int)(j / block_size) * n_batches + cats[perm[j]];
}

// E and the penalty from the held-out counts; R_sum is the column sum of O.
// One CTA per cluster reduces its column in a fixed tree, then each thread
// fills its batches. `penalty` may be null to update E only.
template <typename T, bool Stabilized>
__global__ void penalty_from_counts_kernel(const T* O, const T* Pr_b,
                                           const T* theta, T* E, T* penalty,
                                           int n_batches, int n_clusters,
                                           int n_first) {
    // Each cell counts once per batch key: the cluster total sums the levels
    // of the first key only.
    int k = blockIdx.x;
    T r_sum = T(0);
    for (int b = threadIdx.x; b < n_first; b += blockDim.x)
        r_sum += O[(size_t)b * n_clusters + k];
    r_sum = objective_block_sum(r_sum);
    for (int b = threadIdx.x; b < n_batches; b += blockDim.x) {
        size_t i = (size_t)b * n_clusters + k;
        T e = Pr_b[b] * r_sum;
        E[i] = e;
        if (penalty) {
            T denom = Stabilized ? (O[i] + e + T(1)) : (O[i] + T(1));
            penalty[i] = pow((e + T(1)) / denom, theta[b]);
        }
    }
}

// One CTA per assignment tile (cells of one category within the block). Each
// warp takes FUSED_ROWS cells at a time and each lane owns four consecutive
// clusters: similarities to the normalized centroids (staged in shared
// memory, PCs padded to a multiple of four), penalized soft assignment
// written to R in place, and the cell's distance + entropy objective term.
// The tile's new column sums are reduced in a fixed order into
// `tile_partial`, laid out as scatter_tiles_kernel's partials, so
// scatter_finish_kernel adds them to O.
constexpr int FUSED_ROWS = 8;
constexpr int FUSED_CLUSTER_STRIDE = 32 * FUSED_MAX_CLUSTER_SLOTS;

__host__ __device__ inline int fused_padded_pcs(int n_pcs) {
    return (n_pcs + 3) / 4 * 4;
}

template <typename T>
size_t fused_assign_smem_bytes(int n_pcs, int n_clusters) {
    int warps = FUSED_THREADS / 32, pcs = fused_padded_pcs(n_pcs);
    return ((size_t)pcs * FUSED_CLUSTER_STRIDE +
            (size_t)warps * FUSED_ROWS * pcs + (size_t)warps * n_clusters) *
           sizeof(T);
}

// The fused kernel holds at most 128 clusters, prefetches at most 128 PCs and
// keeps its centroids in shared memory; other shapes use
// fused_assign_general_kernel.
inline bool fused_assign_fits(int n_pcs, int n_clusters, size_t smem) {
    int device = 0, limit = 0;
    cudaGetDevice(&device);
    cudaDeviceGetAttribute(&limit, cudaDevAttrMaxSharedMemoryPerBlockOptin,
                           device);
    return n_clusters <= FUSED_CLUSTER_STRIDE && n_pcs <= 128 &&
           smem <= (size_t)limit;
}

// Padded cluster count of the general kernel's centroid buffer.
__host__ __device__ inline int fused_cluster_stride(int n_clusters) {
    return (n_clusters + FUSED_CLUSTER_STRIDE - 1) / FUSED_CLUSTER_STRIDE *
           FUSED_CLUSTER_STRIDE;
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

// ZQ = ceil(padded PCs / 32): Z values per lane and cell held in registers
// while the next cells are prefetched.
// Two CTAs per SM: the register cap pays for itself in latency hiding.
// LOG_PEN: `penalty` holds log penalties (several batch keys, where the
// product of penalties can overflow); rows are shifted by their maximum.
template <typename T, typename RT, int ZQ, bool LOG_PEN>
__global__ void __launch_bounds__(FUSED_THREADS, 2)
    fused_assign_kernel(const T* __restrict__ Z, const T* __restrict__ Y_norm,
                        const T* __restrict__ penalty,
                        const int* __restrict__ idx,
                        const int* __restrict__ offsets,
                        const int* __restrict__ tiles, int n_categories,
                        int tile_rows, RT* __restrict__ R,
                        T* __restrict__ partial, T* __restrict__ tile_partial,
                        T term, T sigma, int n_pcs, int n_clusters) {
    int tile = blockIdx.x;
    if (tile >= tiles[n_categories]) return;
    int category = scatter_tile_category(tile, n_categories, tiles);
    int start = offsets[category] + (tile - tiles[category]) * tile_rows;
    int end = start + min(tile_rows, offsets[category + 1] - start);

    constexpr int S = FUSED_MAX_CLUSTER_SLOTS;
    int pcs = fused_padded_pcs(n_pcs);
    int lane = threadIdx.x & 31, warp = threadIdx.x >> 5;
    int n_warps = blockDim.x >> 5;
    extern __shared__ __align__(16) unsigned char smem_raw[];
    T* y_t = reinterpret_cast<T*>(smem_raw);  // pcs x FUSED_CLUSTER_STRIDE
    T* z_s = y_t + (size_t)pcs * FUSED_CLUSTER_STRIDE;  // warps x ROWS x pcs
    T* col_sums = z_s + (size_t)n_warps * FUSED_ROWS * pcs;  // warps x K
    for (int t = threadIdx.x; t < pcs * FUSED_CLUSTER_STRIDE; t += blockDim.x) {
        int d = t / FUSED_CLUSTER_STRIDE, k = t % FUSED_CLUSTER_STRIDE;
        y_t[t] =
            d < n_pcs && k < n_clusters ? Y_norm[(size_t)k * n_pcs + d] : T(0);
    }
    T* z_w = z_s + (size_t)warp * FUSED_ROWS * pcs;
    int k0 = 4 * lane;  // this lane's clusters k0 .. k0 + 3
    T pen[S], log_pen[S], col[S] = {};
#pragma unroll
    for (int q = 0; q < S; ++q) {
        int k = k0 + q;
        // No penalty table (initialization): unit penalty.
        T table = penalty && k < n_clusters
                      ? penalty[(size_t)category * n_clusters + k]
                      : T(LOG_PEN ? 0 : 1);
        if constexpr (LOG_PEN) {
            log_pen[q] = k < n_clusters ? table : T(0);
        } else {
            pen[q] = k < n_clusters ? table : T(0);
            log_pen[q] = k < n_clusters ? log(pen[q]) : T(0);
        }
    }
    __syncthreads();

    // The next cells' indices and Z values are loaded into registers while
    // the current cells are assigned, hiding the gather latency.
    int next[FUSED_ROWS];
    T z_next[FUSED_ROWS][ZQ];
    auto prefetch = [&](int b) {
#pragma unroll
        for (int r = 0; r < FUSED_ROWS; ++r) {
            next[r] = b + r >= end ? -1 : idx ? idx[b + r] : b + r;
#pragma unroll
            for (int q = 0; q < ZQ; ++q) {
                int d = lane + 32 * q;
                z_next[r][q] = next[r] >= 0 && d < n_pcs
                                   ? Z[(size_t)next[r] * n_pcs + d]
                                   : T(0);
            }
        }
    };
    int stride = n_warps * FUSED_ROWS;
    if (start + warp * FUSED_ROWS < end) prefetch(start + warp * FUSED_ROWS);
    for (int base = start + warp * FUSED_ROWS; base < end; base += stride) {
        int rows = min(FUSED_ROWS, end - base);
        int cell[FUSED_ROWS];
        __syncwarp();
#pragma unroll
        for (int r = 0; r < FUSED_ROWS; ++r) {
            cell[r] = next[r];
#pragma unroll
            for (int q = 0; q < ZQ; ++q)
                if (lane + 32 * q < pcs)
                    z_w[r * pcs + lane + 32 * q] = z_next[r][q];
        }
        __syncwarp();
        if (base + stride < end) prefetch(base + stride);
        T dots[FUSED_ROWS][S] = {};
        for (int d = 0; d < pcs; d += 4) {
            T y[4][S];
#pragma unroll
            for (int j = 0; j < 4; ++j)
                load4(y_t + (size_t)(d + j) * FUSED_CLUSTER_STRIDE + k0, y[j]);
#pragma unroll
            for (int r = 0; r < FUSED_ROWS; ++r) {
                T z[4];
                load4(z_w + r * pcs + d, z);
#pragma unroll
                for (int j = 0; j < 4; ++j)
#pragma unroll
                    for (int q = 0; q < S; ++q) dots[r][q] += z[j] * y[j][q];
            }
        }
#pragma unroll
        for (int r = 0; r < FUSED_ROWS; ++r) {
            if (r >= rows) break;  // warp-uniform
            T vals[S];
            T sum = T(0);
            T shift = T(0);
            if constexpr (LOG_PEN) {
                shift = -INFINITY;
#pragma unroll
                for (int q = 0; q < S; ++q)
                    if (k0 + q < n_clusters)
                        shift =
                            max(shift, term * (T(1) - dots[r][q]) + log_pen[q]);
#pragma unroll
                for (int offset = 16; offset > 0; offset >>= 1)
                    shift =
                        max(shift, __shfl_xor_sync(0xffffffff, shift, offset));
            }
#pragma unroll
            for (int q = 0; q < S; ++q) {
                if constexpr (LOG_PEN)
                    vals[q] = k0 + q < n_clusters
                                  ? exp(term * (T(1) - dots[r][q]) +
                                        log_pen[q] - shift)
                                  : T(0);
                else
                    vals[q] = k0 + q < n_clusters
                                  ? exp(term * (T(1) - dots[r][q])) * pen[q]
                                  : T(0);
                sum += vals[q];
            }
#pragma unroll
            for (int offset = 16; offset > 0; offset >>= 1)
                sum += __shfl_xor_sync(0xffffffff, sum, offset);
            T inv = T(1) / sum, log_sum = log(sum);
            T distance = T(0), entropy = T(0);
            RT* row = R + (size_t)cell[r] * n_clusters;
#pragma unroll
            for (int q = 0; q < S; ++q) {
                int k = k0 + q;
                if (k < n_clusters) {
                    T value = vals[q] * inv;
                    // Count the stored (possibly bfloat16) value.
                    RT stored = to_storage<RT>(value);
                    row[k] = stored;
                    value = from_storage<T>(stored);
                    col[q] += value;
                    distance += value * T(2) * (T(1) - dots[r][q]);
                    // log(value) = log of the unnormalized weight - log(sum)
                    if constexpr (LOG_PEN)
                        entropy += value * (term * (T(1) - dots[r][q]) +
                                            log_pen[q] - shift - log_sum);
                    else
                        entropy += value * (term * (T(1) - dots[r][q]) +
                                            log_pen[q] - log_sum);
                }
            }
            T total = distance + sigma * entropy;
#pragma unroll
            for (int offset = 16; offset > 0; offset >>= 1)
                total += __shfl_xor_sync(0xffffffff, total, offset);
            if (lane == 0) partial[cell[r]] = total;
        }
    }
#pragma unroll
    for (int q = 0; q < S; ++q)
        if (k0 + q < n_clusters) col_sums[warp * n_clusters + k0 + q] = col[q];
    __syncthreads();
    for (int k = threadIdx.x; k < n_clusters; k += blockDim.x) {
        T sum = T(0);
        for (int w = 0; w < n_warps; ++w) sum += col_sums[w * n_clusters + k];
        tile_partial[(size_t)tile * n_clusters + k] = sum;
    }
}

// General assignment for shapes the fused kernel does not fit (more than 128
// clusters, or centroids beyond shared memory): centroids are read from the
// transposed global buffer `y_t` (padded PCs x `k_stride`), Z directly from
// global memory, and clusters in chunks of FUSED_CLUSTER_STRIDE. With one
// chunk a single sweep normalizes; otherwise a first sweep tracks each row's
// maximum and sum online and a second sweep writes the normalized values.
// Column sums go through per-warp scratch `col_ws` (tiles x warps x K).
template <typename T, typename RT, bool LOG_PEN>
__global__ void fused_assign_general_kernel(
    const T* __restrict__ Z, const T* __restrict__ y_t, int k_stride,
    const T* __restrict__ penalty, const int* __restrict__ idx,
    const int* __restrict__ offsets, const int* __restrict__ tiles,
    int n_categories, int tile_rows, RT* __restrict__ R,
    T* __restrict__ partial, T* __restrict__ tile_partial,
    T* __restrict__ col_ws, T term, T sigma, int n_pcs, int n_clusters) {
    int tile = blockIdx.x;
    if (tile >= tiles[n_categories]) return;
    int category = scatter_tile_category(tile, n_categories, tiles);
    int start = offsets[category] + (tile - tiles[category]) * tile_rows;
    int end = start + min(tile_rows, offsets[category + 1] - start);
    constexpr int S = FUSED_MAX_CLUSTER_SLOTS;
    int lane = threadIdx.x & 31, warp = threadIdx.x >> 5;
    int n_warps = blockDim.x >> 5;
    int n_chunks =
        (n_clusters + FUSED_CLUSTER_STRIDE - 1) / FUSED_CLUSTER_STRIDE;
    T* cols = col_ws + ((size_t)tile * n_warps + warp) * n_clusters;
    for (int k = lane; k < n_clusters; k += 32) cols[k] = T(0);
    const T* pen_row =
        penalty ? penalty + (size_t)category * n_clusters : nullptr;

    // Log weights of the rows' clusters in chunk c (lane owns 4 per chunk).
    auto logits = [&](const int (&cell)[FUSED_ROWS], int c,
                      T(&lw)[FUSED_ROWS][S], T(&om)[FUSED_ROWS][S]) {
        int k0 = c * FUSED_CLUSTER_STRIDE + 4 * lane;
        T dots[FUSED_ROWS][S] = {};
        for (int d = 0; d < n_pcs; ++d) {
            T y[4];
            load4(y_t + (size_t)d * k_stride + k0, y);
#pragma unroll
            for (int r = 0; r < FUSED_ROWS; ++r) {
                T z = cell[r] >= 0 ? Z[(size_t)cell[r] * n_pcs + d] : T(0);
#pragma unroll
                for (int q = 0; q < S; ++q) dots[r][q] += z * y[q];
            }
        }
#pragma unroll
        for (int q = 0; q < S; ++q) {
            int k = k0 + q;
            T lp = !pen_row || k >= n_clusters ? T(0)
                   : LOG_PEN                   ? pen_row[k]
                                               : log(pen_row[k]);
#pragma unroll
            for (int r = 0; r < FUSED_ROWS; ++r) {
                om[r][q] = T(1) - dots[r][q];
                lw[r][q] = k < n_clusters ? term * om[r][q] + lp : -INFINITY;
            }
        }
    };
    auto warp_max = [](T v) {
        for (int offset = 16; offset > 0; offset >>= 1)
            v = max(v, __shfl_xor_sync(0xffffffff, v, offset));
        return v;
    };
    auto warp_sum = [](T v) {
        for (int offset = 16; offset > 0; offset >>= 1)
            v += __shfl_xor_sync(0xffffffff, v, offset);
        return v;
    };

    for (int base = start + warp * FUSED_ROWS; base < end;
         base += n_warps * FUSED_ROWS) {
        int cell[FUSED_ROWS];
#pragma unroll
        for (int r = 0; r < FUSED_ROWS; ++r)
            cell[r] = base + r >= end ? -1 : idx ? idx[base + r] : base + r;
        T lw[FUSED_ROWS][S], om[FUSED_ROWS][S];
        T row_max[FUSED_ROWS], row_sum[FUSED_ROWS];
#pragma unroll
        for (int r = 0; r < FUSED_ROWS; ++r)
            row_max[r] = -INFINITY, row_sum[r] = T(0);
        for (int c = 0; c < n_chunks; ++c) {  // online maximum and sum
            logits(cell, c, lw, om);
#pragma unroll
            for (int r = 0; r < FUSED_ROWS; ++r) {
                T m = row_max[r];
#pragma unroll
                for (int q = 0; q < S; ++q) m = max(m, lw[r][q]);
                m = warp_max(m);
                T s = T(0);
#pragma unroll
                for (int q = 0; q < S; ++q) s += exp(lw[r][q] - m);
                row_sum[r] = row_sum[r] * exp(row_max[r] - m) + warp_sum(s);
                row_max[r] = m;
            }
        }
        T distance[FUSED_ROWS] = {}, entropy[FUSED_ROWS] = {};
        for (int c = 0; c < n_chunks; ++c) {
            if (n_chunks > 1)
                logits(cell, c, lw, om);  // one chunk: still loaded
            int k0 = c * FUSED_CLUSTER_STRIDE + 4 * lane;
#pragma unroll
            for (int r = 0; r < FUSED_ROWS; ++r) {
                if (cell[r] < 0) continue;
                T inv = T(1) / row_sum[r], log_sum = log(row_sum[r]);
                RT* row = R + (size_t)cell[r] * n_clusters;
#pragma unroll
                for (int q = 0; q < S; ++q) {
                    int k = k0 + q;
                    if (k >= n_clusters) continue;
                    T value = exp(lw[r][q] - row_max[r]) * inv;
                    RT stored = to_storage<RT>(value);
                    row[k] = stored;
                    value = from_storage<T>(stored);
                    cols[k] += value;
                    distance[r] += value * T(2) * om[r][q];
                    entropy[r] += value * (lw[r][q] - row_max[r] - log_sum);
                }
            }
        }
#pragma unroll
        for (int r = 0; r < FUSED_ROWS; ++r) {
            T total = warp_sum(distance[r] + sigma * entropy[r]);
            if (lane == 0 && cell[r] >= 0) partial[cell[r]] = total;
        }
    }
    __syncthreads();
    for (int k = threadIdx.x; k < n_clusters; k += blockDim.x) {
        T sum = T(0);
        for (int w = 0; w < n_warps; ++w)
            sum += col_ws[((size_t)tile * n_warps + w) * n_clusters + k];
        tile_partial[(size_t)tile * n_clusters + k] = sum;
    }
}

// y_t (padded PCs x k_stride) = Y_norm^T, zero padded.
template <typename T>
__global__ void transpose_centroids_kernel(const T* Y_norm, T* y_t, int n_pcs,
                                           int pcs, int n_clusters,
                                           int k_stride) {
    for (long long t = blockIdx.x * (long long)blockDim.x + threadIdx.x;
         t < (long long)pcs * k_stride;
         t += (long long)blockDim.x * gridDim.x) {
        int d = (int)(t / k_stride), k = (int)(t % k_stride);
        y_t[t] =
            d < n_pcs && k < n_clusters ? Y_norm[(size_t)k * n_pcs + d] : T(0);
    }
}

// Log penalty of every joint category: the sum of its levels' log penalties.
template <typename T>
__global__ void joint_log_penalty_kernel(const T* penalty,
                                         const int* joint_cats,
                                         int n_covariates, int n_joint,
                                         int n_clusters, T* out) {
    for (long long i = blockIdx.x * (long long)blockDim.x + threadIdx.x;
         i < (long long)n_joint * n_clusters;
         i += (long long)blockDim.x * gridDim.x) {
        long long j = i / n_clusters, k = i % n_clusters;
        T sum = T(0);
        for (int c = 0; c < n_covariates; ++c)
            sum += log(
                penalty[(size_t)joint_cats[j * n_covariates + c] * n_clusters +
                        k]);
        out[i] = sum;
    }
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
    diversity = objective_block_sum(diversity, false);
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
    for (int t = threadIdx.x; t < cols * FUSED_CLUSTER_STRIDE;
         t += blockDim.x) {
        int d = t / FUSED_CLUSTER_STRIDE, k = t % FUSED_CLUSTER_STRIDE;
        c_t[t] = d < n_cols && k < n_clusters ? centers[(size_t)k * n_cols + d]
                                              : T(0);
    }
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
        for (int d = 0; d < cols; d += 4) {
            T c[4][S];
#pragma unroll
            for (int j = 0; j < 4; ++j)
                load4(c_t + (size_t)(d + j) * FUSED_CLUSTER_STRIDE + k0, c[j]);
#pragma unroll
            for (int r = 0; r < FUSED_ROWS; ++r) {
                T x[4];
                load4(x_w + r * cols + d, x);
#pragma unroll
                for (int j = 0; j < 4; ++j)
#pragma unroll
                    for (int q = 0; q < S; ++q) {
                        T delta = x[j] - c[j][q];
                        dist[r][q] += delta * delta;
                    }
            }
        }
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
#pragma unroll
        for (int offset = 16; offset > 0; offset >>= 1)
            sum += __shfl_xor_sync(0xffffffff, sum, offset);
        if (lane == 0 && sum < closest[i]) closest[i] = sum;
    }
}

template <typename T>
__global__ void subtract_kernel(T* __restrict__ out,
                                const T* __restrict__ values, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) out[i] -= values[i];
}
