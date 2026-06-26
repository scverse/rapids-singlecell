#include <cuda_runtime.h>
#include "../nb_types.h"

#include "kernels_nanmean.cuh"

using namespace nb::literals;

constexpr int BLOCK_SIZE_MINOR = 32;
constexpr int BLOCK_SIZE_MAJOR = 64;

template <typename T, typename IdxT>
static inline void launch_nan_mean_minor(const IdxT* index, const T* data,
                                         double* means, int* nans,
                                         const bool* mask, long long nnz,
                                         cudaStream_t stream) {
    dim3 block(BLOCK_SIZE_MINOR);
    dim3 grid(strided_grid(nnz, BLOCK_SIZE_MINOR));
    nan_mean_minor_kernel<T, IdxT>
        <<<grid, block, 0, stream>>>(index, data, means, nans, mask, nnz);
    CUDA_CHECK_LAST_ERROR(nan_mean_minor_kernel);
}

template <typename T, typename IdxT>
static inline void launch_nan_mean_major(const IdxT* indptr, const IdxT* index,
                                         const T* data, double* means,
                                         int* nans, const bool* mask, int major,
                                         int minor, cudaStream_t stream) {
    dim3 block(BLOCK_SIZE_MAJOR);
    dim3 grid(major);
    nan_mean_major_kernel<T, IdxT><<<grid, block, 0, stream>>>(
        indptr, index, data, means, nans, mask, major, minor);
    CUDA_CHECK_LAST_ERROR(nan_mean_major_kernel);
}

template <typename T, typename IdxT, typename Device>
void def_nan_mean_minor(nb::module_& m) {
    m.def(
        "nan_mean_minor",
        [](gpu_array_c<const IdxT, Device> index,
           gpu_array_c<const T, Device> data, gpu_array_c<double, Device> means,
           gpu_array_c<int, Device> nans, gpu_array_c<const bool, Device> mask,
           long long nnz, std::uintptr_t stream) {
            launch_nan_mean_minor<T, IdxT>(
                index.data(), data.data(), means.data(), nans.data(),
                mask.data(), nnz, (cudaStream_t)stream);
        },
        "index"_a, "data"_a, nb::kw_only(), "means"_a, "nans"_a, "mask"_a,
        "nnz"_a, "stream"_a = 0);
}

template <typename T, typename IdxT, typename Device>
void def_nan_mean_major(nb::module_& m) {
    m.def(
        "nan_mean_major",
        [](gpu_array_c<const IdxT, Device> indptr,
           gpu_array_c<const IdxT, Device> index,
           gpu_array_c<const T, Device> data, gpu_array_c<double, Device> means,
           gpu_array_c<int, Device> nans, gpu_array_c<const bool, Device> mask,
           int major, int minor, std::uintptr_t stream) {
            launch_nan_mean_major<T, IdxT>(
                indptr.data(), index.data(), data.data(), means.data(),
                nans.data(), mask.data(), major, minor, (cudaStream_t)stream);
        },
        "indptr"_a, "index"_a, "data"_a, nb::kw_only(), "means"_a, "nans"_a,
        "mask"_a, "major"_a, "minor"_a, "stream"_a = 0);
}

template <typename Device>
void register_bindings(nb::module_& m) {
    def_nan_mean_minor<float, int, Device>(m);
    def_nan_mean_minor<double, int, Device>(m);
    def_nan_mean_minor<float, long long, Device>(m);
    def_nan_mean_minor<double, long long, Device>(m);

    def_nan_mean_major<float, int, Device>(m);
    def_nan_mean_major<double, int, Device>(m);
    def_nan_mean_major<float, long long, Device>(m);
    def_nan_mean_major<double, long long, Device>(m);
}

NB_MODULE(_nanmean_cuda, m) {
    REGISTER_GPU_BINDINGS(register_bindings, m);
}
