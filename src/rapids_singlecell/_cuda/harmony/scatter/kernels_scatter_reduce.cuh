#pragma once

#include <cub/block/block_scan.cuh>
#include <cub/device/device_radix_sort.cuh>

#include <climits>

constexpr int SCATTER_TILE_ROWS = 1024;
constexpr int SCATTER_SCAN_THREADS = 256;
constexpr int SCATTER_MAX_ENTRIES = 1 << 26;

__global__ void scatter_source_rows_kernel(int* rows, int n_entries,
                                           int n_covariates) {
    int entry = blockIdx.x * blockDim.x + threadIdx.x;
    if (entry < n_entries) rows[entry] = entry / n_covariates;
}

__global__ void scatter_pack_categories_kernel(const int* categories,
                                               int* packed, int n_entries,
                                               int n_covariates,
                                               int covariate_start,
                                               int group_width) {
    int entry = blockIdx.x * blockDim.x + threadIdx.x;
    if (entry < n_entries)
        packed[entry] =
            categories[(size_t)(entry / group_width) * n_covariates +
                       covariate_start + entry % group_width];
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
                                            int n_categories, int* tiles) {
    using Scan = cub::BlockScan<int, SCATTER_SCAN_THREADS>;
    __shared__ typename Scan::TempStorage scratch;
    int total = 0;
    for (long long begin = 0; begin < n_categories;
         begin += SCATTER_SCAN_THREADS) {
        long long category = begin + threadIdx.x;
        int length = category < n_categories
                         ? offsets[category + 1] - offsets[category]
                         : 0;
        int count =
            length / SCATTER_TILE_ROWS + (length % SCATTER_TILE_ROWS != 0);
        int prefix, aggregate;
        Scan(scratch).ExclusiveSum(count, prefix, aggregate);
        if (category < n_categories) tiles[category] = total + prefix;
        total += aggregate;
        __syncthreads();
    }
    if (threadIdx.x == 0) tiles[n_categories] = total;
}

__device__ int scatter_tile_category(int tile, int n_categories,
                                     const int* tiles) {
    int lo = 0, hi = n_categories;
    while (lo < hi) {
        int mid = lo + (hi - lo) / 2;
        if (tiles[mid + 1] <= tile)
            lo = mid + 1;
        else
            hi = mid;
    }
    return lo;
}

template <typename T>
__global__ void scatter_tiles_kernel(const T* values, int n_cols,
                                     int n_categories, const int* indices,
                                     const int* offsets, const int* tiles,
                                     T* partial) {
    int col_tiles = (n_cols + 31) / 32;
    int tile = blockIdx.x / col_tiles;
    if (tile >= tiles[n_categories]) return;
    int col = (blockIdx.x % col_tiles) * 32 + threadIdx.x;
    int category = scatter_tile_category(tile, n_categories, tiles);
    int start =
        offsets[category] + (tile - tiles[category]) * SCATTER_TILE_ROWS;
    int end = start + min(SCATTER_TILE_ROWS, offsets[category + 1] - start);
    __shared__ T sums[8][32];
    T sum = T(0);
    if (col < n_cols)
        for (long long pos = (long long)start + threadIdx.y; pos < end;
             pos += 8)
            sum += values[(size_t)indices[pos] * n_cols + col];
    sums[threadIdx.y][threadIdx.x] = sum;
    __syncthreads();
    if (threadIdx.y == 0 && col < n_cols) {
        sum = sums[0][threadIdx.x];
        for (int i = 1; i < 8; ++i) sum += sums[i][threadIdx.x];
        partial[(size_t)tile * n_cols + col] = sum;
    }
}

template <typename T>
__global__ void scatter_finish_kernel(const T* partial, int n_cols,
                                      int n_categories, const int* tiles,
                                      int switcher, T* out) {
    int col_tiles = (n_cols + 31) / 32;
    int category = blockIdx.x / col_tiles;
    if (gridDim.x / col_tiles < n_categories) {
        int tile = category;
        if (tile >= tiles[n_categories]) return;
        category = scatter_tile_category(tile, n_categories, tiles);
        if (tile != tiles[category]) return;
    }
    int col = (blockIdx.x % col_tiles) * 32 + threadIdx.x;
    if (tiles[category] == tiles[category + 1]) return;
    __shared__ T sums[8][32];
    T sum = T(0);
    if (col < n_cols)
        for (int tile = tiles[category] + threadIdx.y;
             tile < tiles[category + 1]; tile += 8)
            sum += partial[(size_t)tile * n_cols + col];
    sums[threadIdx.y][threadIdx.x] = sum;
    __syncthreads();
    if (threadIdx.y == 0 && col < n_cols) {
        sum = sums[0][threadIdx.x];
        for (int i = 1; i < 8; ++i) sum += sums[i][threadIdx.x];
        if (switcher)
            out[(size_t)category * n_cols + col] += sum;
        else
            out[(size_t)category * n_cols + col] -= sum;
    }
}

static inline int scatter_category_bits(int n_categories) {
    int bits = 1;
    while ((1u << bits) < (unsigned)n_categories) ++bits;
    return bits;
}

static size_t scatter_sort_temp_bytes(int n_rows, int n_categories) {
    size_t bytes = 0;
    auto* keys = reinterpret_cast<int*>(1);
    cuda_check(cub::DeviceRadixSort::SortPairs(
                   nullptr, bytes, keys, keys, keys, keys, n_rows, 0,
                   scatter_category_bits(n_categories)),
               "category sort temp query");
    return bytes;
}

static int scatter_entries(int n_rows, int n_covariates) {
    long long entries = (long long)n_rows * n_covariates;
    if (n_rows < 1 || n_covariates < 1 || entries > INT_MAX)
        throw std::invalid_argument("scatter entry count exceeds int32 range");
    return (int)entries;
}

static int scatter_group_width(int n_rows, int n_covariates) {
    if (n_rows < 1 || n_covariates < 1)
        throw std::invalid_argument("scatter dimensions must be positive");
    return std::min(n_covariates, std::max(1, SCATTER_MAX_ENTRIES / n_rows));
}

static size_t scatter_int_bytes(int n_entries, int n_categories) {
    return (((size_t)3 * n_entries + (size_t)2 * n_categories + 2) *
                sizeof(int) +
            255) &
           ~size_t(255);
}

static size_t scatter_max_tiles(int n_entries, int n_categories) {
    return std::min((size_t)n_entries, ((size_t)n_entries + SCATTER_TILE_ROWS -
                                        1) / SCATTER_TILE_ROWS +
                                           n_categories);
}

static size_t scatter_grouped_int_bytes(int n_categories) {
    return (((size_t)n_categories + 1) * sizeof(int) + 255) & ~size_t(255);
}

static size_t scatter_base_bytes(int n_entries, int n_cols, int n_categories,
                                 int itemsize, bool pack_categories = false) {
    size_t int_bytes = scatter_int_bytes(n_entries, n_categories);
    size_t partial_bytes =
        scatter_max_tiles(n_entries, n_categories) * n_cols * itemsize;
    if (pack_categories)
        partial_bytes =
            std::max(partial_bytes, (size_t)n_entries * sizeof(int));
    return (int_bytes + partial_bytes + 255) & ~size_t(255);
}

static size_t scatter_temp_bytes(int n_rows, int n_cols, int n_categories,
                                 int n_covariates, int itemsize,
                                 bool grouped = false) {
    if (grouped &&
        (n_covariates != 1 || n_rows < 1 || n_cols < 1 || n_categories < 1))
        throw std::invalid_argument(
            "grouped scatter requires one covariate and positive dimensions");
    if (itemsize != 4 && itemsize != 8)
        throw std::invalid_argument("scatter itemsize must be 4 or 8");
    if (grouped)
        return scatter_grouped_int_bytes(n_categories) +
               scatter_max_tiles(n_rows, n_categories) * n_cols * itemsize;
    int group_width = scatter_group_width(n_rows, n_covariates);
    int n_entries = scatter_entries(n_rows, group_width);
    return scatter_base_bytes(n_entries, n_cols, n_categories, itemsize,
                              group_width < n_covariates) +
           scatter_sort_temp_bytes(n_entries, n_categories);
}

template <typename T>
static void scatter_reduce_tiles(const T* values, int n_entries, int n_cols,
                                 int n_categories, const int* indices,
                                 const int* offsets, const int* tiles,
                                 T* partial, int switcher, T* out,
                                 cudaStream_t stream) {
    size_t max_tiles = scatter_max_tiles(n_entries, n_categories);
    scatter_tiles_kernel<T>
        <<<max_tiles*((n_cols + 31) / 32), dim3(32, 8), 0, stream>>>(
            values, n_cols, n_categories, indices, offsets, tiles, partial);
    CUDA_CHECK_LAST_ERROR(scatter_tiles_kernel);
    scatter_finish_kernel<T>
        <<<std::min((size_t)n_categories, max_tiles) * ((n_cols + 31) / 32),
           dim3(32, 8), 0, stream>>>(partial, n_cols, n_categories, tiles,
                                     switcher, out);
    CUDA_CHECK_LAST_ERROR(scatter_finish_kernel);
}

template <typename T>
static void scatter_reduce(const T* values, const int* categories, int n_rows,
                           int n_cols, int n_categories, int n_covariates,
                           int switcher, T* out, uint8_t* workspace,
                           cudaStream_t stream, bool reuse_grouping = false) {
    int group_width = scatter_group_width(n_rows, n_covariates);
    if (workspace == nullptr)
        throw std::invalid_argument("invalid scatter workspace");

    bool split = group_width < n_covariates;
    int n_entries = scatter_entries(n_rows, group_width);
    int* indices = reinterpret_cast<int*>(workspace);
    int* offsets = indices + n_entries;
    int* tiles = offsets + n_categories + 1;
    int* sorted_categories = tiles + n_categories + 1;
    int* rows = sorted_categories + n_entries;
    size_t int_bytes = scatter_int_bytes(n_entries, n_categories);
    T* partial = reinterpret_cast<T*>(workspace + int_bytes);
    size_t base_bytes =
        scatter_base_bytes(n_entries, n_cols, n_categories, sizeof(T), split);
    void* sort_temp = workspace + base_bytes;
    size_t sort_bytes = scatter_sort_temp_bytes(n_entries, n_categories);

    for (int covariate_start = 0; covariate_start < n_covariates;) {
        int width = std::min(group_width, n_covariates - covariate_start);
        int entries = scatter_entries(n_rows, width);
        const int* keys = categories;
        if (split) {
            // The partial region is unused until sorting has finished.
            auto* packed = reinterpret_cast<int*>(partial);
            scatter_pack_categories_kernel<<<((size_t)entries + 255) / 256, 256,
                                             0, stream>>>(
                categories, packed, entries, n_covariates, covariate_start,
                width);
            CUDA_CHECK_LAST_ERROR(scatter_pack_categories_kernel);
            keys = packed;
        }
        // A split call retains only the final group's ordering in this
        // workspace.
        if (!reuse_grouping || split) {
            scatter_source_rows_kernel<<<((size_t)entries + 255) / 256, 256, 0,
                                         stream>>>(rows, entries, width);
            CUDA_CHECK_LAST_ERROR(scatter_source_rows_kernel);
            cuda_check(cub::DeviceRadixSort::SortPairs(
                           sort_temp, sort_bytes, keys, sorted_categories, rows,
                           indices, entries, 0,
                           scatter_category_bits(n_categories), stream),
                       "category sort");
            scatter_category_offsets_kernel<<<
                ((size_t)n_categories + 127) / 128, 128, 0, stream>>>(
                sorted_categories, entries, n_categories, offsets);
            CUDA_CHECK_LAST_ERROR(scatter_category_offsets_kernel);
            scatter_tile_offsets_kernel<<<1, SCATTER_SCAN_THREADS, 0, stream>>>(
                offsets, n_categories, tiles);
            CUDA_CHECK_LAST_ERROR(scatter_tile_offsets_kernel);
        }
        scatter_reduce_tiles(values, n_entries, n_cols, n_categories, indices,
                             offsets, tiles, partial, switcher, out, stream);
        covariate_start += width;
    }
}
