#include <cuda_runtime.h>
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

// ---------------------------------------------------------------------------
// Launch policies: each builds the appropriate loader and launches the shared
// edistance_kernel_impl for compile-time tile sizes <CELL_TILE, FEAT_TILE>.
// The dense and sparse paths differ ONLY here (and in their loaders); the tile
// selection below is shared.
// ---------------------------------------------------------------------------
template <typename T>
struct DenseLaunch {
    const T* embedding;
    const int* cat_offsets;
    const int* cell_indices;
    const int* pair_left;
    const int* pair_right;
    T* pairwise_sums;
    int num_pairs, n_features, blocks_per_pair, block_size;
    size_t shared_mem;
    cudaStream_t stream;

    template <int CELL_TILE, int FEAT_TILE>
    void run() const {
        dim3 grid(num_pairs, blocks_per_pair);
        dim3 block(block_size);
        edistance_kernel<T, CELL_TILE, FEAT_TILE>
            <<<grid, block, shared_mem, stream>>>(
                embedding, cat_offsets, cell_indices, pair_left, pair_right,
                pairwise_sums, n_features, blocks_per_pair);
        CUDA_CHECK_LAST_ERROR(edistance_kernel);
    }
};

template <typename T, typename IndptrT>
struct SparseLaunch {
    const T* data;
    const int* indices;
    const IndptrT* indptr;
    const int* cat_offsets;
    const int* cell_indices;
    const int* pair_left;
    const int* pair_right;
    T* pairwise_sums;
    int num_pairs, n_features, blocks_per_pair, block_size;
    size_t shared_mem;
    cudaStream_t stream;

    template <int CELL_TILE, int FEAT_TILE>
    void run() const {
        dim3 grid(num_pairs, blocks_per_pair);
        dim3 block(block_size);
        edistance_sparse_kernel<T, CELL_TILE, FEAT_TILE, IndptrT>
            <<<grid, block, shared_mem, stream>>>(
                data, indices, indptr, cat_offsets, cell_indices, pair_left,
                pair_right, pairwise_sums, n_features, blocks_per_pair);
        CUDA_CHECK_LAST_ERROR(edistance_sparse_kernel);
    }
};

// Pick the (CELL_TILE, FEAT_TILE) specialization at compile time and invoke
// launcher.run<CELL_TILE, FEAT_TILE>(). f32 uses CELL_TILE=64 (FEAT_TILE 25/16)
// with a legacy CELL_TILE=32 fallback; f64 uses CELL_TILE=16. Shared by both
// the dense and sparse bindings via the launch policy.
template <typename T, typename Launcher>
static void dispatch_tiles(const Launcher& launcher, int cell_tile,
                           int feat_tile) {
    (void)cell_tile;  // unused for f64 (always 16)
    if constexpr (std::is_same_v<T, double>) {
        if (feat_tile == 64) {
            launcher.template run<16, 64>();
        } else if (feat_tile == 50) {
            launcher.template run<16, 50>();
        } else {
            launcher.template run<16, 32>();
        }
    } else {
        if (cell_tile == 64) {
            if (feat_tile == 25) {
                launcher.template run<64, 25>();
            } else {  // feat_tile == 16
                launcher.template run<64, 16>();
            }
        } else {  // legacy CELL_TILE=32 fallback
            if (feat_tile == 64) {
                launcher.template run<32, 64>();
            } else if (feat_tile == 50) {
                launcher.template run<32, 50>();
            } else {
                launcher.template run<32, 32>();
            }
        }
    }
}

template <typename T, typename Device, typename IndptrT>
void def_compute_distances_sparse(nb::module_& m) {
    m.def(
        "compute_distances_sparse",
        [](gpu_array_c<const IndptrT, Device> indptr,
           gpu_array_c<const int, Device> indices,
           gpu_array_c<const T, Device> data,
           gpu_array_c<const int, Device> cat_offsets,
           gpu_array_c<const int, Device> cell_indices,
           gpu_array_c<const int, Device> pair_left,
           gpu_array_c<const int, Device> pair_right,
           gpu_array_c<T, Device> pairwise_sums, int num_pairs, int n_features,
           int blocks_per_pair, int cell_tile, int feat_tile, int block_size,
           int shared_mem, std::uintptr_t stream) {
            SparseLaunch<T, IndptrT> launcher{
                data.data(),
                indices.data(),
                indptr.data(),
                cat_offsets.data(),
                cell_indices.data(),
                pair_left.data(),
                pair_right.data(),
                pairwise_sums.data(),
                num_pairs,
                n_features,
                blocks_per_pair,
                block_size,
                static_cast<size_t>(shared_mem),
                reinterpret_cast<cudaStream_t>(stream)};
            dispatch_tiles<T>(launcher, cell_tile, feat_tile);
        },
        "indptr"_a, "indices"_a, "data"_a, "cat_offsets"_a, "cell_indices"_a,
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
            DenseLaunch<T> launcher{embedding.data(),
                                    cat_offsets.data(),
                                    cell_indices.data(),
                                    pair_left.data(),
                                    pair_right.data(),
                                    pairwise_sums.data(),
                                    num_pairs,
                                    n_features,
                                    blocks_per_pair,
                                    block_size,
                                    static_cast<size_t>(shared_mem),
                                    reinterpret_cast<cudaStream_t>(stream)};
            dispatch_tiles<T>(launcher, cell_tile, feat_tile);
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

    // Sparse (CSR) variants. Nanobind dispatches on data and indptr dtype.
    // Keep f64 before f32 for proper overload dispatch.
    def_compute_distances_sparse<double, Device, int>(m);
    def_compute_distances_sparse<float, Device, int>(m);
    def_compute_distances_sparse<double, Device, long long>(m);
    def_compute_distances_sparse<float, Device, long long>(m);
}

NB_MODULE(_edistance_cuda, m) {
    m.def("get_kernel_config", &get_kernel_config, "n_features"_a,
          "is_double"_a,
          "Get kernel configuration (cell_tile, feat_tile, block_size, "
          "shared_mem) for given "
          "parameters. Returns None if insufficient shared memory.");

    REGISTER_GPU_BINDINGS(register_bindings, m);
}
