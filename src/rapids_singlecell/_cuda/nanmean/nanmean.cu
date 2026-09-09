#include <cuda_runtime.h>
#include "../nb_types.h"

#include "kernels_nanmean.cuh"

using namespace nb::literals;

constexpr int BLOCK_SIZE_MAJOR = 64;

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

// Order-agnostic minor-axis sums (one atomic per nonzero); no indptr needed.
template <typename T, typename IdxT, typename Device>
void def_nan_mean_minor(nb::module_& m) {
    m.def(
        "nan_mean_minor",
        [](gpu_array_c<const IdxT, Device> index,
           gpu_array_c<const T, Device> data, gpu_array_c<double, Device> means,
           gpu_array_c<int, Device> nans, gpu_array_c<const bool, Device> mask,
           long long nnz, std::uintptr_t stream) {
            NanMeanOp<T> op{data.data(), means.data(), nans.data(), mask.data(),
                            0};
            minor_reduce_flat<IdxT>(index.data(), op, nnz,
                                    (cudaStream_t)stream);
        },
        "index"_a, "data"_a, nb::kw_only(), "means"_a, "nans"_a, "mask"_a,
        "nnz"_a, "stream"_a = 0);
}

// Shared-memory tile sweep with atomic fallback. Returns whether unsorted rows
// were detected, so the caller can skip the attempt next time.
template <typename T, typename IdxT, typename Device>
void def_nan_mean_minor_tiled(nb::module_& m) {
    m.def(
        "nan_mean_minor_tiled",
        [](gpu_array_c<const IdxT, Device> indptr,
           gpu_array_c<const IdxT, Device> index,
           gpu_array_c<const T, Device> data, gpu_array_c<double, Device> means,
           gpu_array_c<int, Device> nans, gpu_array_c<const bool, Device> mask,
           int major, int minor, long long nnz, bool assume_unsorted,
           std::uintptr_t stream) {
            NanMeanOp<T> op{data.data(), means.data(), nans.data(), mask.data(),
                            0};
            return minor_reduce<IdxT>(indptr.data(), index.data(), op, major,
                                      minor, nnz, assume_unsorted,
                                      (cudaStream_t)stream);
        },
        "indptr"_a, "index"_a, "data"_a, nb::kw_only(), "means"_a, "nans"_a,
        "mask"_a, "major"_a, "minor"_a, "nnz"_a, "assume_unsorted"_a = false,
        "stream"_a = 0);
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

    def_nan_mean_minor_tiled<float, int, Device>(m);
    def_nan_mean_minor_tiled<double, int, Device>(m);
    def_nan_mean_minor_tiled<float, long long, Device>(m);
    def_nan_mean_minor_tiled<double, long long, Device>(m);

    def_nan_mean_major<float, int, Device>(m);
    def_nan_mean_major<double, int, Device>(m);
    def_nan_mean_major<float, long long, Device>(m);
    def_nan_mean_major<double, long long, Device>(m);
}

NB_MODULE(_nanmean_cuda, m) {
    REGISTER_GPU_BINDINGS(register_bindings, m);
    register_scratch_allocator(m);
}
