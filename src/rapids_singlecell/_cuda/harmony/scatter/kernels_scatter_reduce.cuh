#pragma once

#include <cub/block/block_scan.cuh>
#include <cub/device/device_radix_sort.cuh>

constexpr int SCATTER_TILE_ROWS = 1024;
constexpr int SCATTER_SCAN_THREADS = 256;

__global__ void scatter_source_rows_kernel(int* rows, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) rows[i] = i;
}

__global__ void scatter_category_offsets_kernel(const int* sorted_categories,
                                                int n_rows, int n_categories,
                                                int* offsets) {
    int category = blockIdx.x * blockDim.x + threadIdx.x;
    if (category < n_categories) {
        int lo = 0, hi = n_rows;
        while (lo < hi) {
            int mid = lo + (hi - lo) / 2;
            if (sorted_categories[mid] < category)
                lo = mid + 1;
            else
                hi = mid;
        }
        offsets[category] = lo;
    }
    if (category == 0) offsets[n_categories] = n_rows;
}

__global__ void scatter_tile_offsets_kernel(const int* offsets,
                                            int n_categories, int* tiles,
                                            int tile_rows = SCATTER_TILE_ROWS) {
    using Scan = cub::BlockScan<int, SCATTER_SCAN_THREADS>;
    __shared__ typename Scan::TempStorage scratch;
    int total = 0;
    for (long long begin = 0; begin < n_categories;
         begin += SCATTER_SCAN_THREADS) {
        long long category = begin + threadIdx.x;
        int length = category < n_categories
                         ? offsets[category + 1] - offsets[category]
                         : 0;
        int count = length / tile_rows + (length % tile_rows != 0);
        int prefix, aggregate;
        Scan(scratch).ExclusiveSum(count, prefix, aggregate);
        if (category < n_categories) tiles[category] = total + prefix;
        total += aggregate;
        __syncthreads();
    }
    if (threadIdx.x == 0) tiles[n_categories] = total;
}

// Category of `tile` and its rows [start, end) (tiles of `tile_rows` rows cut
// per category, see scatter_tile_offsets_kernel).
__device__ __forceinline__ int scatter_tile_rows(int tile, int n_categories,
                                                 const int* offsets,
                                                 const int* tiles,
                                                 int tile_rows, int& start,
                                                 int& end) {
    int lo = 0, hi = n_categories;
    while (lo < hi) {
        int mid = lo + (hi - lo) / 2;
        if (tiles[mid + 1] <= tile)
            lo = mid + 1;
        else
            hi = mid;
    }
    start = offsets[lo] + (tile - tiles[lo]) * tile_rows;
    end = start + min(tile_rows, offsets[lo + 1] - start);
    return lo;
}

// Sum over threadIdx.y of a (32, 8) block, in order.
template <typename T>
__device__ T scatter_sum_y(T sum) {
    __shared__ T sums[8][32];
    sums[threadIdx.y][threadIdx.x] = sum;
    __syncthreads();
    sum = sums[0][threadIdx.x];
    for (int i = 1; i < 8; ++i) sum += sums[i][threadIdx.x];
    return sum;
}

template <typename T>
__global__ void scatter_tiles_kernel(const T* values, int n_cols,
                                     int n_categories, const int* indices,
                                     const int* offsets, const int* tiles,
                                     T* partial) {
    int col_tiles = (n_cols + 31) / 32;
    int tile = blockIdx.x / col_tiles, start, end;
    if (tile >= tiles[n_categories]) return;
    int col = (blockIdx.x % col_tiles) * 32 + threadIdx.x;
    scatter_tile_rows(tile, n_categories, offsets, tiles, SCATTER_TILE_ROWS,
                      start, end);
    T sum = T(0);
    if (col < n_cols)
        for (int pos = start + threadIdx.y; pos < end; pos += 8)
            sum += values[(size_t)indices[pos] * n_cols + col];
    sum = scatter_sum_y(sum);
    if (threadIdx.y == 0 && col < n_cols)
        partial[(size_t)tile * n_cols + col] = sum;
}

// out[category] += the sum of the category's tile partials.
template <typename T>
__global__ void scatter_finish_kernel(const T* partial, int n_cols,
                                      const int* tiles, T* out) {
    int col_tiles = (n_cols + 31) / 32;
    int category = blockIdx.x / col_tiles;
    int col = (blockIdx.x % col_tiles) * 32 + threadIdx.x;
    if (tiles[category] == tiles[category + 1]) return;
    T sum = T(0);
    if (col < n_cols)
        for (int tile = tiles[category] + threadIdx.y;
             tile < tiles[category + 1]; tile += 8)
            sum += partial[(size_t)tile * n_cols + col];
    sum = scatter_sum_y(sum);
    if (threadIdx.y == 0 && col < n_cols)
        out[(size_t)category * n_cols + col] += sum;
}

static inline int scatter_category_bits(int n_categories) {
    int bits = 1;
    while ((1u << bits) < (unsigned)n_categories) ++bits;
    return bits;
}

static size_t sort_temp_bytes(int n, int bits) {
    size_t bytes = 0;
    auto* keys = reinterpret_cast<unsigned int*>(1);
    cuda_check(
        cub::DeviceRadixSort::SortPairs(nullptr, bytes, keys, keys, (int*)keys,
                                        (int*)keys, n, 0, bits),
        "sort temp query");
    return bytes;
}

static size_t scatter_max_tiles(int n_rows, int n_categories) {
    return std::min((size_t)n_rows, ((size_t)n_rows + SCATTER_TILE_ROWS - 1) /
                                            SCATTER_TILE_ROWS +
                                        n_categories);
}

// Workspace of scatter_reduce: ints (row order, category offsets and tiles,
// sorted categories, source rows), the tile partial sums, then the sort
// scratch. Returns the offsets of the latter two.
static std::pair<size_t, size_t> scatter_layout(int n_rows, int n_cols,
                                                int n_categories,
                                                int itemsize) {
    auto align = [](size_t bytes) { return (bytes + 255) & ~size_t(255); };
    size_t ints = align((3 * (size_t)n_rows + 2 * (size_t)n_categories + 2) *
                        sizeof(int));
    return {ints, ints + align(scatter_max_tiles(n_rows, n_categories) *
                               n_cols * itemsize)};
}

static size_t scatter_temp_bytes(int n_rows, int n_cols, int n_categories,
                                 int itemsize) {
    if (n_rows < 1 || n_cols < 1 || n_categories < 1)
        throw std::invalid_argument("scatter dimensions must be positive");
    return scatter_layout(n_rows, n_cols, n_categories, itemsize).second +
           sort_temp_bytes(n_rows, scatter_category_bits(n_categories));
}

// out[c] += the sum of the rows of `values` in category c, in a fixed order:
// rows sorted by category, summed in tiles, then the tiles in order.
template <typename T>
static void scatter_reduce(const T* values, const int* categories, int n_rows,
                           int n_cols, int n_categories, T* out,
                           uint8_t* workspace, cudaStream_t stream) {
    int* indices = reinterpret_cast<int*>(workspace);
    int* offsets = indices + n_rows;
    int* tiles = offsets + n_categories + 1;
    int* sorted_categories = tiles + n_categories + 1;
    int* rows = sorted_categories + n_rows;
    auto [partial_at, sort_at] =
        scatter_layout(n_rows, n_cols, n_categories, sizeof(T));
    T* partial = reinterpret_cast<T*>(workspace + partial_at);
    int bits = scatter_category_bits(n_categories);
    size_t sort_bytes = sort_temp_bytes(n_rows, bits);
    scatter_source_rows_kernel<<<(n_rows + 255) / 256, 256, 0, stream>>>(
        rows, n_rows);
    CUDA_CHECK_LAST_ERROR(scatter_source_rows_kernel);
    cuda_check(cub::DeviceRadixSort::SortPairs(
                   workspace + sort_at, sort_bytes, categories,
                   sorted_categories, rows, indices, n_rows, 0, bits, stream),
               "category sort");
    scatter_category_offsets_kernel<<<(n_categories + 127) / 128, 128, 0,
                                      stream>>>(sorted_categories, n_rows,
                                                n_categories, offsets);
    CUDA_CHECK_LAST_ERROR(scatter_category_offsets_kernel);
    scatter_tile_offsets_kernel<<<1, SCATTER_SCAN_THREADS, 0, stream>>>(
        offsets, n_categories, tiles);
    CUDA_CHECK_LAST_ERROR(scatter_tile_offsets_kernel);
    int col_tiles = (n_cols + 31) / 32;
    scatter_tiles_kernel<T>
        <<<scatter_max_tiles(n_rows, n_categories) * col_tiles, dim3(32, 8), 0,
           stream>>>(values, n_cols, n_categories, indices, offsets, tiles,
                     partial);
    CUDA_CHECK_LAST_ERROR(scatter_tiles_kernel);
    scatter_finish_kernel<T>
        <<<n_categories * col_tiles, dim3(32, 8), 0, stream>>>(partial, n_cols,
                                                               tiles, out);
    CUDA_CHECK_LAST_ERROR(scatter_finish_kernel);
}
