#include <cuda_runtime.h>
#include "../nb_types.h"

#include "kernels_mv.cuh"

using namespace nb::literals;

constexpr int BLOCK_SIZE_MINOR = 256;

template <typename T, typename IdxT>
static inline void launch_mean_var_major(const IdxT* indptr,
                                         const IdxT* indices, const T* data,
                                         double* means, double* vars, int major,
                                         int minor, cudaStream_t stream) {
    dim3 block(BLOCK_SIZE_MAJOR);
    dim3 grid(major);
    mean_var_major_kernel<T, IdxT><<<grid, block, 0, stream>>>(
        indptr, indices, data, means, vars, major, minor);
    CUDA_CHECK_LAST_ERROR(mean_var_major_kernel);
}

template <typename T, typename IdxT>
static inline void launch_mean_var_minor(const IdxT* indices, const T* data,
                                         double* means, double* vars,
                                         long long nnz, cudaStream_t stream) {
    int block = BLOCK_SIZE_MINOR;
    unsigned int grid = strided_grid(nnz, block);
    mean_var_minor_kernel<T, IdxT>
        <<<grid, block, 0, stream>>>(indices, data, means, vars, nnz);
    CUDA_CHECK_LAST_ERROR(mean_var_minor_kernel);
}

template <typename T, typename IdxT, typename Device>
void def_mean_var_major(nb::module_& m) {
    m.def(
        "mean_var_major",
        [](gpu_array_c<const IdxT, Device> indptr,
           gpu_array_c<const IdxT, Device> indices,
           gpu_array_c<const T, Device> data, gpu_array_c<double, Device> means,
           gpu_array_c<double, Device> vars, int major, int minor,
           std::uintptr_t stream) {
            launch_mean_var_major<T, IdxT>(
                indptr.data(), indices.data(), data.data(), means.data(),
                vars.data(), major, minor, (cudaStream_t)stream);
        },
        "indptr"_a, "indices"_a, "data"_a, "means"_a, "vars"_a, nb::kw_only(),
        "major"_a, "minor"_a, "stream"_a = 0);
}

template <typename T, typename IdxT, typename Device>
void def_mean_var_minor(nb::module_& m) {
    m.def(
        "mean_var_minor",
        [](gpu_array_c<const IdxT, Device> indices,
           gpu_array_c<const T, Device> data, gpu_array_c<double, Device> means,
           gpu_array_c<double, Device> vars, long long nnz,
           std::uintptr_t stream) {
            launch_mean_var_minor<T, IdxT>(indices.data(), data.data(),
                                           means.data(), vars.data(), nnz,
                                           (cudaStream_t)stream);
        },
        "indices"_a, "data"_a, "means"_a, "vars"_a, nb::kw_only(), "nnz"_a,
        "stream"_a = 0);
}

template <typename Device>
void register_bindings(nb::module_& m) {
    def_mean_var_major<float, int, Device>(m);
    def_mean_var_major<float, long long, Device>(m);
    def_mean_var_major<double, int, Device>(m);
    def_mean_var_major<double, long long, Device>(m);

    def_mean_var_minor<float, int, Device>(m);
    def_mean_var_minor<float, long long, Device>(m);
    def_mean_var_minor<double, int, Device>(m);
    def_mean_var_minor<double, long long, Device>(m);
}

NB_MODULE(_mean_var_cuda, m) {
    REGISTER_GPU_BINDINGS(register_bindings, m);
}
