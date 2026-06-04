#include <cuda_runtime.h>
#include <cstdint>
#include "../nb_types.h"
#include <nanobind/stl/tuple.h>

#include "kernels_edistance.cuh"

using namespace nb::literals;

// Tile sizes for feature dimension (CELL_TILE=16 or 32)
static constexpr int TILE_SIZES[] = {32, 50, 64};
static constexpr int NUM_TILE_SIZES = 3;

// Feature tile sizes for CELL_TILE=64 configuration (Ampere+)
// 25 is optimal for common PC counts (50, 100), 16 for better coalescing
// otherwise
static constexpr int FEAT_TILE_64_PREFERRED = 25;
static constexpr int FEAT_TILE_64_COALESCED = 16;

// Choose feat_tile for CELL_TILE=64 configuration
static int choose_feat_tile_64(int n_features) {
    if (n_features % FEAT_TILE_64_PREFERRED == 0) {
        return FEAT_TILE_64_PREFERRED;
    }
    return FEAT_TILE_64_COALESCED;
}

// Choose optimal feat_tile based on n_features and shared memory limits
static int choose_feat_tile(int n_features, size_t max_shared_bytes,
                            int cell_tile, int dtype_size) {
    // Shared memory: cell_tile * feat_tile * dtype_size + warp_sums overhead
    size_t warp_sums_overhead = WARP_SIZE * dtype_size;
    size_t available_shared = max_shared_bytes - warp_sums_overhead;

    int best_tile = 32;  // default minimum

    // Check exact divisibility - prefer larger tiles
    for (int i = NUM_TILE_SIZES - 1; i >= 0; --i) {
        int tile = TILE_SIZES[i];
        size_t required = static_cast<size_t>(cell_tile) * tile * dtype_size;
        if (required <= available_shared) {
            if (n_features % tile == 0) {
                return tile;
            }
            if (best_tile == 32 || tile > best_tile) {
                best_tile = tile;
            }
        }
    }

    return best_tile;
}

// Get kernel configuration for given parameters
// Returns (cell_tile, feat_tile, block_size, shared_mem_bytes) or None if
// insufficient memory
static nb::object get_kernel_config(int n_features, bool is_double) {
    int device = 0;
    cudaGetDevice(&device);
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, device);

    int dtype_size = is_double ? 8 : 4;
    bool is_ampere_plus = prop.major >= 8;
    int cell_tile;
    int block_size;
    int feat_tile;

    if (is_double) {
        // float64: CELL_TILE=16
        // double kernels use 74 regs/thread; cap at 512 to stay within 64K
        // register limit
        cell_tile = 16;
        block_size = is_ampere_plus ? 512 : 256;
        feat_tile = choose_feat_tile(n_features, prop.sharedMemPerBlock,
                                     cell_tile, dtype_size);
    } else {
        // float32: CELL_TILE=64 with block_size=512
        // Same register pressure as CELL_TILE=32 with block_size=1024, but
        // faster
        cell_tile = 64;
        block_size = 512;
        feat_tile = choose_feat_tile_64(n_features);
    }

    // Shared memory: smem_b (cell_tile * feat_tile)
    size_t shared_mem_bytes =
        static_cast<size_t>(cell_tile) * feat_tile * dtype_size;

    if (shared_mem_bytes > prop.sharedMemPerBlock) {
        return nb::none();
    }

    return nb::make_tuple(cell_tile, feat_tile, block_size,
                          static_cast<int>(shared_mem_bytes));
}

// Launch kernel with specific tile sizes
template <typename T, int CELL_TILE, int FEAT_TILE>
static void launch_edistance_kernel(const T* embedding, const int* cat_offsets,
                                    const int* cell_indices,
                                    const int* pair_left, const int* pair_right,
                                    T* pairwise_sums, int num_pairs,
                                    int n_features, int blocks_per_pair,
                                    int block_size, size_t shared_mem,
                                    cudaStream_t stream) {
    dim3 grid(num_pairs, blocks_per_pair);
    dim3 block(block_size);
    DenseLoader<T, CELL_TILE, FEAT_TILE> loader{embedding, cell_indices,
                                                n_features};
    edistance_kernel_impl<T, CELL_TILE, FEAT_TILE,
                          DenseLoader<T, CELL_TILE, FEAT_TILE>>
        <<<grid, block, shared_mem, stream>>>(loader, cat_offsets, pair_left,
                                              pair_right, pairwise_sums,
                                              n_features, blocks_per_pair);
    CUDA_CHECK_LAST_ERROR(edistance_kernel_impl);
}

// Dispatch to correct tile size specialization for float32
// Supports CELL_TILE=64 with FEAT_TILE=16 or 25, and legacy CELL_TILE=32
static void dispatch_f32(const float* embedding, const int* cat_offsets,
                         const int* cell_indices, const int* pair_left,
                         const int* pair_right, float* pairwise_sums,
                         int num_pairs, int n_features, int blocks_per_pair,
                         int cell_tile, int feat_tile, int block_size,
                         size_t shared_mem, cudaStream_t stream) {
    if (cell_tile == 64) {
        // CELL_TILE=64 configuration (float32 default)
        if (feat_tile == 25) {
            launch_edistance_kernel<float, 64, 25>(
                embedding, cat_offsets, cell_indices, pair_left, pair_right,
                pairwise_sums, num_pairs, n_features, blocks_per_pair,
                block_size, shared_mem, stream);
        } else {
            // feat_tile == 16
            launch_edistance_kernel<float, 64, 16>(
                embedding, cat_offsets, cell_indices, pair_left, pair_right,
                pairwise_sums, num_pairs, n_features, blocks_per_pair,
                block_size, shared_mem, stream);
        }
    } else {
        // Legacy CELL_TILE=32 configuration (fallback)
        if (feat_tile == 64) {
            launch_edistance_kernel<float, 32, 64>(
                embedding, cat_offsets, cell_indices, pair_left, pair_right,
                pairwise_sums, num_pairs, n_features, blocks_per_pair,
                block_size, shared_mem, stream);
        } else if (feat_tile == 50) {
            launch_edistance_kernel<float, 32, 50>(
                embedding, cat_offsets, cell_indices, pair_left, pair_right,
                pairwise_sums, num_pairs, n_features, blocks_per_pair,
                block_size, shared_mem, stream);
        } else {
            launch_edistance_kernel<float, 32, 32>(
                embedding, cat_offsets, cell_indices, pair_left, pair_right,
                pairwise_sums, num_pairs, n_features, blocks_per_pair,
                block_size, shared_mem, stream);
        }
    }
}

// Dispatch to correct tile size specialization for float64
// cell_tile is always 16 for float64
static void dispatch_f64(const double* embedding, const int* cat_offsets,
                         const int* cell_indices, const int* pair_left,
                         const int* pair_right, double* pairwise_sums,
                         int num_pairs, int n_features, int blocks_per_pair,
                         int cell_tile, int feat_tile, int block_size,
                         size_t shared_mem, cudaStream_t stream) {
    // cell_tile parameter is ignored for f64 (always 16), but kept for API
    // consistency
    (void)cell_tile;
    if (feat_tile == 64) {
        launch_edistance_kernel<double, 16, 64>(
            embedding, cat_offsets, cell_indices, pair_left, pair_right,
            pairwise_sums, num_pairs, n_features, blocks_per_pair, block_size,
            shared_mem, stream);
    } else if (feat_tile == 50) {
        launch_edistance_kernel<double, 16, 50>(
            embedding, cat_offsets, cell_indices, pair_left, pair_right,
            pairwise_sums, num_pairs, n_features, blocks_per_pair, block_size,
            shared_mem, stream);
    } else {
        launch_edistance_kernel<double, 16, 32>(
            embedding, cat_offsets, cell_indices, pair_left, pair_right,
            pairwise_sums, num_pairs, n_features, blocks_per_pair, block_size,
            shared_mem, stream);
    }
}

// ---------------------------------------------------------------------------
// Sparse (CSR) variants: same tiling/config as the dense kernel, but cells are
// read from CSR (data, indices, indptr) and densified into shared on the fly.
// IndptrT is int (int32) or int64_t so nnz > 2^31-1 is addressable.
// ---------------------------------------------------------------------------

template <typename T, int CELL_TILE, int FEAT_TILE, typename IndptrT>
static void launch_edistance_sparse_kernel(
    const T* data, const int* indices, const IndptrT* indptr,
    const int* cat_offsets, const int* cell_indices, const int* pair_left,
    const int* pair_right, T* pairwise_sums, int num_pairs, int n_features,
    int blocks_per_pair, int block_size, size_t shared_mem,
    cudaStream_t stream) {
    dim3 grid(num_pairs, blocks_per_pair);
    dim3 block(block_size);
    SparseLoader<T, CELL_TILE, FEAT_TILE, IndptrT> loader{data, indices, indptr,
                                                          cell_indices};
    edistance_kernel_impl<T, CELL_TILE, FEAT_TILE,
                          SparseLoader<T, CELL_TILE, FEAT_TILE, IndptrT>>
        <<<grid, block, shared_mem, stream>>>(loader, cat_offsets, pair_left,
                                              pair_right, pairwise_sums,
                                              n_features, blocks_per_pair);
    CUDA_CHECK_LAST_ERROR(edistance_kernel_impl);
}

// Dispatch to correct tile size specialization for sparse float32
template <typename IndptrT>
static void dispatch_f32_sparse(const float* data, const int* indices,
                                const IndptrT* indptr, const int* cat_offsets,
                                const int* cell_indices, const int* pair_left,
                                const int* pair_right, float* pairwise_sums,
                                int num_pairs, int n_features,
                                int blocks_per_pair, int cell_tile,
                                int feat_tile, int block_size,
                                size_t shared_mem, cudaStream_t stream) {
    if (cell_tile == 64) {
        if (feat_tile == 25) {
            launch_edistance_sparse_kernel<float, 64, 25, IndptrT>(
                data, indices, indptr, cat_offsets, cell_indices, pair_left,
                pair_right, pairwise_sums, num_pairs, n_features,
                blocks_per_pair, block_size, shared_mem, stream);
        } else {
            launch_edistance_sparse_kernel<float, 64, 16, IndptrT>(
                data, indices, indptr, cat_offsets, cell_indices, pair_left,
                pair_right, pairwise_sums, num_pairs, n_features,
                blocks_per_pair, block_size, shared_mem, stream);
        }
    } else {
        if (feat_tile == 64) {
            launch_edistance_sparse_kernel<float, 32, 64, IndptrT>(
                data, indices, indptr, cat_offsets, cell_indices, pair_left,
                pair_right, pairwise_sums, num_pairs, n_features,
                blocks_per_pair, block_size, shared_mem, stream);
        } else if (feat_tile == 50) {
            launch_edistance_sparse_kernel<float, 32, 50, IndptrT>(
                data, indices, indptr, cat_offsets, cell_indices, pair_left,
                pair_right, pairwise_sums, num_pairs, n_features,
                blocks_per_pair, block_size, shared_mem, stream);
        } else {
            launch_edistance_sparse_kernel<float, 32, 32, IndptrT>(
                data, indices, indptr, cat_offsets, cell_indices, pair_left,
                pair_right, pairwise_sums, num_pairs, n_features,
                blocks_per_pair, block_size, shared_mem, stream);
        }
    }
}

// Dispatch to correct tile size specialization for sparse float64
template <typename IndptrT>
static void dispatch_f64_sparse(const double* data, const int* indices,
                                const IndptrT* indptr, const int* cat_offsets,
                                const int* cell_indices, const int* pair_left,
                                const int* pair_right, double* pairwise_sums,
                                int num_pairs, int n_features,
                                int blocks_per_pair, int cell_tile,
                                int feat_tile, int block_size,
                                size_t shared_mem, cudaStream_t stream) {
    (void)cell_tile;  // always 16 for f64
    if (feat_tile == 64) {
        launch_edistance_sparse_kernel<double, 16, 64, IndptrT>(
            data, indices, indptr, cat_offsets, cell_indices, pair_left,
            pair_right, pairwise_sums, num_pairs, n_features, blocks_per_pair,
            block_size, shared_mem, stream);
    } else if (feat_tile == 50) {
        launch_edistance_sparse_kernel<double, 16, 50, IndptrT>(
            data, indices, indptr, cat_offsets, cell_indices, pair_left,
            pair_right, pairwise_sums, num_pairs, n_features, blocks_per_pair,
            block_size, shared_mem, stream);
    } else {
        launch_edistance_sparse_kernel<double, 16, 32, IndptrT>(
            data, indices, indptr, cat_offsets, cell_indices, pair_left,
            pair_right, pairwise_sums, num_pairs, n_features, blocks_per_pair,
            block_size, shared_mem, stream);
    }
}

template <typename T, typename Device, typename IndptrT>
void def_compute_distances_sparse(nb::module_& m, const char* name) {
    m.def(
        name,
        [](gpu_array_c<const T, Device> data,
           gpu_array_c<const int, Device> indices,
           gpu_array_c<const IndptrT, Device> indptr,
           gpu_array_c<const int, Device> cat_offsets,
           gpu_array_c<const int, Device> cell_indices,
           gpu_array_c<const int, Device> pair_left,
           gpu_array_c<const int, Device> pair_right,
           gpu_array_c<T, Device> pairwise_sums, int num_pairs, int n_features,
           int blocks_per_pair, int cell_tile, int feat_tile, int block_size,
           int shared_mem, std::uintptr_t stream) {
            if constexpr (std::is_same_v<T, double>) {
                dispatch_f64_sparse<IndptrT>(
                    data.data(), indices.data(), indptr.data(),
                    cat_offsets.data(), cell_indices.data(), pair_left.data(),
                    pair_right.data(), pairwise_sums.data(), num_pairs,
                    n_features, blocks_per_pair, cell_tile, feat_tile,
                    block_size, static_cast<size_t>(shared_mem),
                    reinterpret_cast<cudaStream_t>(stream));
            } else {
                dispatch_f32_sparse<IndptrT>(
                    data.data(), indices.data(), indptr.data(),
                    cat_offsets.data(), cell_indices.data(), pair_left.data(),
                    pair_right.data(), pairwise_sums.data(), num_pairs,
                    n_features, blocks_per_pair, cell_tile, feat_tile,
                    block_size, static_cast<size_t>(shared_mem),
                    reinterpret_cast<cudaStream_t>(stream));
            }
        },
        "data"_a, "indices"_a, "indptr"_a, "cat_offsets"_a, "cell_indices"_a,
        "pair_left"_a, "pair_right"_a, "pairwise_sums"_a, "num_pairs"_a,
        "n_features"_a, "blocks_per_pair"_a, "cell_tile"_a, "feat_tile"_a,
        "block_size"_a, "shared_mem"_a, "stream"_a = 0);
}

template <typename T, typename Device>
void def_compute_distances(nb::module_& m) {
    m.def(
        "compute_distances",
        [](gpu_array_c<const T, Device> embedding,
           gpu_array_c<const int, Device> cat_offsets,
           gpu_array_c<const int, Device> cell_indices,
           gpu_array_c<const int, Device> pair_left,
           gpu_array_c<const int, Device> pair_right,
           gpu_array_c<T, Device> pairwise_sums, int num_pairs, int n_features,
           int blocks_per_pair, int cell_tile, int feat_tile, int block_size,
           int shared_mem, std::uintptr_t stream) {
            if constexpr (std::is_same_v<T, double>) {
                dispatch_f64(embedding.data(), cat_offsets.data(),
                             cell_indices.data(), pair_left.data(),
                             pair_right.data(), pairwise_sums.data(), num_pairs,
                             n_features, blocks_per_pair, cell_tile, feat_tile,
                             block_size, static_cast<size_t>(shared_mem),
                             reinterpret_cast<cudaStream_t>(stream));
            } else {
                dispatch_f32(embedding.data(), cat_offsets.data(),
                             cell_indices.data(), pair_left.data(),
                             pair_right.data(), pairwise_sums.data(), num_pairs,
                             n_features, blocks_per_pair, cell_tile, feat_tile,
                             block_size, static_cast<size_t>(shared_mem),
                             reinterpret_cast<cudaStream_t>(stream));
            }
        },
        "embedding"_a, "cat_offsets"_a, "cell_indices"_a, "pair_left"_a,
        "pair_right"_a, "pairwise_sums"_a, "num_pairs"_a, "n_features"_a,
        "blocks_per_pair"_a, "cell_tile"_a, "feat_tile"_a, "block_size"_a,
        "shared_mem"_a, "stream"_a = 0);
}

template <typename Device>
void register_bindings(nb::module_& m) {
    // IMPORTANT: f64 must be defined before f32 for proper overload dispatch.
    def_compute_distances<double, Device>(m);
    def_compute_distances<float, Device>(m);

    // Sparse (CSR) variants. int32 indptr -> compute_distances_sparse;
    // int64 indptr -> compute_distances_sparse_i64. f64 before f32.
    def_compute_distances_sparse<double, Device, int>(
        m, "compute_distances_sparse");
    def_compute_distances_sparse<float, Device, int>(
        m, "compute_distances_sparse");
    def_compute_distances_sparse<double, Device, int64_t>(
        m, "compute_distances_sparse_i64");
    def_compute_distances_sparse<float, Device, int64_t>(
        m, "compute_distances_sparse_i64");
}

NB_MODULE(_edistance_cuda, m) {
    m.def("get_kernel_config", &get_kernel_config, "n_features"_a,
          "is_double"_a,
          "Get kernel configuration (cell_tile, feat_tile, block_size, "
          "shared_mem) for given "
          "parameters. Returns None if insufficient shared memory.");

    REGISTER_GPU_BINDINGS(register_bindings, m);
}
