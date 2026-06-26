#include <cuda_runtime.h>
#include "../nb_types.h"
#include <nanobind/stl/tuple.h>

#include "kernels_cooc.cuh"

using namespace nb::literals;

// Size constants
static constexpr int WARP_SIZE = 32;
static constexpr int FLOAT32_SIZE = 4;

// Cell tile sizes for B-tile caching (2 floats per cell = 8 bytes)
// Larger tiles reduce iterations but increase shared memory usage
// Ordered largest to smallest - we prefer larger tiles when memory permits
static constexpr int CELL_TILE_SIZES[] = {1024, 512, 256, 128, 64, 32, 16};
static constexpr int NUM_CELL_TILE_SIZES = 7;

// Block sizes to try, ordered by general preference based on benchmarking
// Smaller block sizes generally perform better (more grid parallelism)
// 1024 threads is rarely optimal due to register pressure
static constexpr int BLOCK_SIZES[] = {128, 256, 512, 1024};
static constexpr int NUM_BLOCK_SIZES = 4;

// Block size thresholds based on cells per category
// Larger cells/cat benefits from more threads per block for better parallelism
// Smaller cells/cat benefits from smaller blocks for more grid parallelism
static constexpr int CELLS_PER_CAT_THRESHOLDS[][2] = {
    {10000, 1024},  // >= 10k cells/cat: use block_size=1024
    {5000, 512},    // >= 5k cells/cat: use block_size=512
    {2500, 256},    // >= 2.5k cells/cat: use block_size=256
    {0, 128},       // < 2.5k cells/cat: use block_size=128
};
static constexpr int NUM_THRESHOLDS = 4;

// Compute total shared memory needed for given configuration
static size_t compute_shared_mem(int cell_tile, int l_pad,
                                 int warps_per_block) {
    size_t b_tile_bytes = static_cast<size_t>(cell_tile) * 2 * FLOAT32_SIZE;
    size_t warp_hist_bytes =
        static_cast<size_t>(warps_per_block) * l_pad * sizeof(cooc_count_t);
    return b_tile_bytes + warp_hist_bytes;
}

// Choose optimal block size based on cells per category
static int choose_block_size(int n_cells, int k) {
    int cells_per_cat = n_cells / k;
    for (int i = 0; i < NUM_THRESHOLDS; ++i) {
        if (cells_per_cat >= CELLS_PER_CAT_THRESHOLDS[i][0]) {
            return CELLS_PER_CAT_THRESHOLDS[i][1];
        }
    }
    return 128;  // fallback
}

// Get kernel configuration for given parameters
// Returns (cell_tile, l_pad, block_size, shared_mem_bytes) or None if
// insufficient memory Strategy: Find the best block_size first, then find the
// largest cell_tile that fits
static nb::object get_kernel_config(int l_val, int n_cells, int k) {
    int device = 0;
    cudaGetDevice(&device);
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, device);

    size_t max_shared = prop.sharedMemPerBlock;
    int l_pad = ((l_val + WARP_SIZE - 1) / WARP_SIZE) * WARP_SIZE;

    // Choose optimal block size based on cells per category heuristic
    int target_block_size = choose_block_size(n_cells, k);

    // Try target block size first, then fall back to other sizes in preference
    // order Preference order: target, then other sizes from BLOCK_SIZES array
    int block_sizes_to_try[NUM_BLOCK_SIZES + 1];
    int num_to_try = 0;

    // Add target first
    block_sizes_to_try[num_to_try++] = target_block_size;

    // Add remaining block sizes in preference order (skip if same as target)
    for (int i = 0; i < NUM_BLOCK_SIZES; ++i) {
        if (BLOCK_SIZES[i] != target_block_size) {
            block_sizes_to_try[num_to_try++] = BLOCK_SIZES[i];
        }
    }

    // Try each block size, finding the largest cell_tile that fits
    for (int bi = 0; bi < num_to_try; ++bi) {
        int block_size = block_sizes_to_try[bi];
        int warps_per_block = block_size / WARP_SIZE;

        // Try cell tiles from largest to smallest
        for (int ti = 0; ti < NUM_CELL_TILE_SIZES; ++ti) {
            int cell_tile = CELL_TILE_SIZES[ti];
            size_t total_shared =
                compute_shared_mem(cell_tile, l_pad, warps_per_block);

            if (total_shared <= max_shared) {
                // Found a valid configuration
                return nb::make_tuple(cell_tile, l_pad, block_size,
                                      static_cast<int>(total_shared));
            }
        }
        // No cell_tile fits with this block_size, try smaller block_size
    }

    // No valid configuration found
    return nb::none();
}

// Launch templated kernel with specific cell_tile
template <int CELL_TILE>
static void launch_csr_catpairs_kernel(
    const float* spatial, const float* thresholds, const int* cat_offsets,
    const int* cell_indices, const int* pair_left, const int* pair_right,
    cooc_count_t* counts, int num_pairs, int k, int l_val, int blocks_per_pair,
    int l_pad, int block_size, size_t shared_mem, cudaStream_t stream) {
    dim3 grid(num_pairs, blocks_per_pair);
    dim3 block(block_size);
    occur_count_kernel_csr_catpairs_tiled<CELL_TILE>
        <<<grid, block, shared_mem, stream>>>(
            spatial, thresholds, cat_offsets, cell_indices, pair_left,
            pair_right, counts, k, l_val, blocks_per_pair, l_pad);
    CUDA_CHECK_LAST_ERROR(occur_count_kernel_csr_catpairs_tiled);
}

// Dispatch to correct template specialization based on cell_tile
// l_pad is passed to the kernel as a runtime parameter
static void dispatch_csr_catpairs(
    const float* spatial, const float* thresholds, const int* cat_offsets,
    const int* cell_indices, const int* pair_left, const int* pair_right,
    cooc_count_t* counts, int num_pairs, int k, int l_val, int blocks_per_pair,
    int cell_tile, int block_size, size_t shared_mem, cudaStream_t stream) {
    // Compute l_pad (padded to multiple of 32)
    int l_pad = ((l_val + WARP_SIZE - 1) / WARP_SIZE) * WARP_SIZE;

    // Dispatch based on cell_tile only - l_pad is a runtime parameter
    switch (cell_tile) {
        case 1024:
            launch_csr_catpairs_kernel<1024>(
                spatial, thresholds, cat_offsets, cell_indices, pair_left,
                pair_right, counts, num_pairs, k, l_val, blocks_per_pair, l_pad,
                block_size, shared_mem, stream);
            break;
        case 512:
            launch_csr_catpairs_kernel<512>(
                spatial, thresholds, cat_offsets, cell_indices, pair_left,
                pair_right, counts, num_pairs, k, l_val, blocks_per_pair, l_pad,
                block_size, shared_mem, stream);
            break;
        case 256:
            launch_csr_catpairs_kernel<256>(
                spatial, thresholds, cat_offsets, cell_indices, pair_left,
                pair_right, counts, num_pairs, k, l_val, blocks_per_pair, l_pad,
                block_size, shared_mem, stream);
            break;
        case 128:
            launch_csr_catpairs_kernel<128>(
                spatial, thresholds, cat_offsets, cell_indices, pair_left,
                pair_right, counts, num_pairs, k, l_val, blocks_per_pair, l_pad,
                block_size, shared_mem, stream);
            break;
        case 64:
            launch_csr_catpairs_kernel<64>(
                spatial, thresholds, cat_offsets, cell_indices, pair_left,
                pair_right, counts, num_pairs, k, l_val, blocks_per_pair, l_pad,
                block_size, shared_mem, stream);
            break;
        case 32:
            launch_csr_catpairs_kernel<32>(
                spatial, thresholds, cat_offsets, cell_indices, pair_left,
                pair_right, counts, num_pairs, k, l_val, blocks_per_pair, l_pad,
                block_size, shared_mem, stream);
            break;
        case 16:
        default:
            launch_csr_catpairs_kernel<16>(
                spatial, thresholds, cat_offsets, cell_indices, pair_left,
                pair_right, counts, num_pairs, k, l_val, blocks_per_pair, l_pad,
                block_size, shared_mem, stream);
            break;
    }
}

// Pairwise kernel launch
static inline void launch_count_pairwise(const float* spatial,
                                         const float* thresholds,
                                         const int* labels,
                                         cooc_count_t* result, int n, int k,
                                         int l_val, cudaStream_t stream) {
    dim3 grid(n);
    dim3 block(32);
    occur_count_kernel_pairwise<<<grid, block, 0, stream>>>(
        spatial, thresholds, labels, result, n, k, l_val);
    CUDA_CHECK_LAST_ERROR(occur_count_kernel_pairwise);
}

// Shared memory reduction launch
static inline bool launch_reduce_shared(const cooc_count_t* result, float* out,
                                        int k, int l_val, int format,
                                        cudaStream_t stream) {
    int device = 0;
    cudaGetDevice(&device);
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, device);
    if (prop.sharedMemPerBlock <
        static_cast<size_t>(k) * static_cast<size_t>(k + 1) * sizeof(float)) {
        return false;
    }

    dim3 grid(l_val);
    dim3 block(32);
    size_t smem =
        static_cast<size_t>(k) * static_cast<size_t>(k + 1) * sizeof(float);
    occur_reduction_kernel_shared<<<grid, block, smem, stream>>>(result, out, k,
                                                                 l_val, format);
    CUDA_CHECK_LAST_ERROR(occur_reduction_kernel_shared);
    return true;
}

// Global memory reduction launch
static inline void launch_reduce_global(const cooc_count_t* result,
                                        float* inter_out, float* out, int k,
                                        int l_val, int format,
                                        cudaStream_t stream) {
    dim3 grid(l_val);
    dim3 block(32);
    size_t smem = static_cast<size_t>(k) * sizeof(float);
    occur_reduction_kernel_global<<<grid, block, smem, stream>>>(
        result, inter_out, out, k, l_val, format);
    CUDA_CHECK_LAST_ERROR(occur_reduction_kernel_global);
}

template <typename Device>
void register_bindings(nb::module_& m) {
    m.def(
        "count_csr_catpairs",
        [](gpu_array_c<const float, Device> spatial,
           gpu_array_c<const float, Device> thresholds,
           gpu_array_c<const int, Device> cat_offsets,
           gpu_array_c<const int, Device> cell_indices,
           gpu_array_c<const int, Device> pair_left,
           gpu_array_c<const int, Device> pair_right,
           gpu_array_c<cooc_count_t, Device> counts, int num_pairs, int k,
           int l_val, int blocks_per_pair, int cell_tile, int block_size,
           int shared_mem, std::uintptr_t stream) {
            dispatch_csr_catpairs(
                spatial.data(), thresholds.data(), cat_offsets.data(),
                cell_indices.data(), pair_left.data(), pair_right.data(),
                counts.data(), num_pairs, k, l_val, blocks_per_pair, cell_tile,
                block_size, static_cast<size_t>(shared_mem),
                reinterpret_cast<cudaStream_t>(stream));
        },
        "spatial"_a, nb::kw_only(), "thresholds"_a, "cat_offsets"_a,
        "cell_indices"_a, "pair_left"_a, "pair_right"_a, "counts"_a,
        "num_pairs"_a, "k"_a, "l_val"_a, "blocks_per_pair"_a, "cell_tile"_a,
        "block_size"_a, "shared_mem"_a, "stream"_a = 0);

    m.def(
        "count_pairwise",
        [](gpu_array_c<const float, Device> spatial,
           gpu_array_c<const float, Device> thresholds,
           gpu_array_c<const int, Device> labels,
           gpu_array_c<cooc_count_t, Device> result, int n, int k, int l_val,
           std::uintptr_t stream) {
            launch_count_pairwise(spatial.data(), thresholds.data(),
                                  labels.data(), result.data(), n, k, l_val,
                                  (cudaStream_t)stream);
        },
        "spatial"_a, nb::kw_only(), "thresholds"_a, "labels"_a, "result"_a,
        "n"_a, "k"_a, "l_val"_a, "stream"_a = 0);

    m.def(
        "reduce_shared",
        [](gpu_array_c<const cooc_count_t, Device> result,
           gpu_array_c<float, Device> out, int k, int l_val, int format,
           std::uintptr_t stream) {
            return launch_reduce_shared(result.data(), out.data(), k, l_val,
                                        format, (cudaStream_t)stream);
        },
        "result"_a, nb::kw_only(), "out"_a, "k"_a, "l_val"_a, "format"_a,
        "stream"_a = 0);

    m.def(
        "reduce_global",
        [](gpu_array_c<const cooc_count_t, Device> result,
           gpu_array_c<float, Device> inter_out, gpu_array_c<float, Device> out,
           int k, int l_val, int format, std::uintptr_t stream) {
            launch_reduce_global(result.data(), inter_out.data(), out.data(), k,
                                 l_val, format, (cudaStream_t)stream);
        },
        "result"_a, nb::kw_only(), "inter_out"_a, "out"_a, "k"_a, "l_val"_a,
        "format"_a, "stream"_a = 0);
}

NB_MODULE(_cooc_cuda, m) {
    m.def("get_kernel_config", &get_kernel_config, "l_val"_a, "n_cells"_a,
          "k"_a,
          "Get kernel configuration (cell_tile, l_pad, block_size, shared_mem) "
          "for given "
          "parameters. Returns None if insufficient shared memory.");

    REGISTER_GPU_BINDINGS(register_bindings, m);
}
