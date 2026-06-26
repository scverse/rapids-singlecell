#pragma once

#include <cuda_runtime.h>
#include <nanobind/nanobind.h>
#include <nanobind/ndarray.h>
#include <stdexcept>
#include <string>

namespace nb = nanobind;

/// Check cudaGetLastError after a <<<...>>> launch (invalid grid/block,
/// shared memory overflow, etc.).
inline void cuda_check_last_error(const char* kernel_name) {
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        throw std::runtime_error(std::string(kernel_name) +
                                 " launch failed: " + cudaGetErrorString(err));
    }
}

#define CUDA_CHECK_LAST_ERROR(kernel_name) cuda_check_last_error(#kernel_name)

/// Check a cudaError_t returned directly by a CUDA/CUB API call.
/// Failed calls surface with a clear label instead of corrupted output later.
inline void cuda_check(cudaError_t err, const char* what) {
    if (err != cudaSuccess) {
        throw std::runtime_error(std::string(what) +
                                 " failed: " + cudaGetErrorString(err));
    }
}

/// Validate a binding-argument precondition (array dims vs. scalar shapes).
/// Mismatches become clean Python errors, not out-of-bounds launches.
inline void nb_require(bool cond, const char* what) {
    if (!cond) {
        throw std::invalid_argument(
            std::string("rank_genes_groups CUDA binding: ") + what);
    }
}

/// Per-axis cached cap on `gridDim.{x,y,z}`; y/z are often only 65535.
/// Runtime per-device cache keeps this multi-GPU safe.
inline const int* max_grid_dims() {
    static thread_local int cached_dev = -1;
    static thread_local int cached[3] = {65535, 65535, 65535};  // safe fallback
    int device = 0;
    cudaGetDevice(&device);
    if (device != cached_dev) {
        cudaDeviceProp prop;
        cudaGetDeviceProperties(&prop, device);
        cached[0] = prop.maxGridSize[0];
        cached[1] = prop.maxGridSize[1];
        cached[2] = prop.maxGridSize[2];
        cached_dev = device;
    }
    return cached;
}

inline int max_grid_dim_x() {
    return max_grid_dims()[0];
}
inline int max_grid_dim_y() {
    return max_grid_dims()[1];
}
inline int max_grid_dim_z() {
    return max_grid_dims()[2];
}

/// Grid-stride cap for kernels whose total work exceeds one grid launch.
/// Pair with a grid-strided loop; use `strided_grid_y` for y-axis launches.
inline unsigned int strided_grid(long long nwork, int block_size) {
    const long long max_grid = max_grid_dim_x();
    long long ideal = (nwork + block_size - 1) / block_size;
    long long capped = ideal < max_grid ? ideal : max_grid;
    return (unsigned int)(capped < 1 ? 1 : capped);
}

/// Like `strided_grid` but for the y-axis (much lower cap, typically 65535).
inline unsigned int strided_grid_y(long long nwork, int block_size) {
    const long long max_grid = max_grid_dim_y();
    long long ideal = (nwork + block_size - 1) / block_size;
    long long capped = ideal < max_grid ? ideal : max_grid;
    return (unsigned int)(capped < 1 ? 1 : capped);
}

// GPU array aliases for nanobind bindings, parameterized on device type.
// CUDA and managed-memory variants both preserve CPU/GPU type safety.

// C-contiguous (row-major)
template <typename T, typename Device>
using gpu_array_c = nb::ndarray<T, Device, nb::c_contig>;

// F-contiguous (column-major)
template <typename T, typename Device>
using gpu_array_f = nb::ndarray<T, Device, nb::f_contig>;

// No contiguity constraint
template <typename T, typename Device>
using gpu_array = nb::ndarray<T, Device>;

// Parameterized contiguity (kernels handling both C and F order)
template <typename T, typename Device, typename Contig>
using gpu_array_contig = nb::ndarray<T, Device, Contig>;

// Host (NumPy) array aliases
template <typename T>
using host_array = nb::ndarray<T, nb::numpy, nb::ndim<1>>;
template <typename T>
using host_array_c2 = nb::ndarray<T, nb::numpy, nb::ndim<2>, nb::c_contig>;
template <typename T>
using host_array_f2 = nb::ndarray<T, nb::numpy, nb::ndim<2>, nb::f_contig>;

// Register bindings for both regular CUDA and managed-memory arrays.
// Each registration function must be templated on `Device`.
#define REGISTER_GPU_BINDINGS(func, module) \
    func<nb::device::cuda>(module);         \
    func<nb::device::cuda_managed>(module)
