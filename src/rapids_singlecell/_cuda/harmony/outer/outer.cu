#include <cuda_runtime.h>
#include "../../nb_types.h"

#include "kernels_outer.cuh"

using namespace nb::literals;

constexpr int BLOCK_SIZE = 256;

template <typename T>
static inline void launch_outer(T* E, const T* Pr_b, const T* R_sum,
                                long long n_cats, long long n_pcs,
                                long long switcher, cudaStream_t stream) {
    dim3 block(BLOCK_SIZE);
    long long N = n_cats * n_pcs;
    dim3 grid(strided_grid(N, BLOCK_SIZE));
    outer_kernel<T>
        <<<grid, block, 0, stream>>>(E, Pr_b, R_sum, n_cats, n_pcs, switcher);
    CUDA_CHECK_LAST_ERROR(outer_kernel);
}

template <typename T>
static inline void launch_harmony_corr(T* Z, const T* W, const int* cats,
                                       const T* R, long long n_cells,
                                       long long n_pcs, cudaStream_t stream) {
    dim3 block(BLOCK_SIZE);
    long long N = n_cells * n_pcs;
    dim3 grid(strided_grid(N, BLOCK_SIZE));
    harmony_correction_kernel<T>
        <<<grid, block, 0, stream>>>(Z, W, cats, R, n_cells, n_pcs);
    CUDA_CHECK_LAST_ERROR(harmony_correction_kernel);
}

template <typename T>
static inline void launch_batched_correction(T* Z, const T* W_all,
                                             const int* cats, const T* R,
                                             int n_cells, int n_pcs,
                                             int n_clusters, int n_batches_p1,
                                             cudaStream_t stream) {
    dim3 block(BLOCK_SIZE);
    long long N = (long long)n_cells * n_pcs;
    dim3 grid(strided_grid(N, BLOCK_SIZE));
    batched_correction_kernel<T><<<grid, block, 0, stream>>>(
        Z, W_all, cats, R, n_cells, n_pcs, n_clusters, n_batches_p1);
    CUDA_CHECK_LAST_ERROR(batched_correction_kernel);
}

template <typename T, typename Device>
void def_outer(nb::module_& m) {
    m.def(
        "outer",
        [](gpu_array_c<T, Device> E, gpu_array_c<const T, Device> Pr_b,
           gpu_array_c<const T, Device> R_sum, long long n_cats,
           long long n_pcs, long long switcher, std::uintptr_t stream) {
            launch_outer<T>(E.data(), Pr_b.data(), R_sum.data(), n_cats, n_pcs,
                            switcher, (cudaStream_t)stream);
        },
        "E"_a, nb::kw_only(), "Pr_b"_a, "R_sum"_a, "n_cats"_a, "n_pcs"_a,
        "switcher"_a, "stream"_a = 0);
}

template <typename T, typename Device>
void def_harmony_corr(nb::module_& m) {
    m.def(
        "harmony_corr",
        [](gpu_array_c<T, Device> Z, gpu_array_c<const T, Device> W,
           gpu_array_c<const int, Device> cats, gpu_array_c<const T, Device> R,
           long long n_cells, long long n_pcs, std::uintptr_t stream) {
            launch_harmony_corr<T>(Z.data(), W.data(), cats.data(), R.data(),
                                   n_cells, n_pcs, (cudaStream_t)stream);
        },
        "Z"_a, nb::kw_only(), "W"_a, "cats"_a, "R"_a, "n_cells"_a, "n_pcs"_a,
        "stream"_a = 0);
}

template <typename T, typename Device>
void def_batched_correction(nb::module_& m) {
    m.def(
        "batched_correction",
        [](gpu_array_c<T, Device> Z, gpu_array_c<const T, Device> W_all,
           gpu_array_c<const int, Device> cats, gpu_array_c<const T, Device> R,
           int n_cells, int n_pcs, int n_clusters, int n_batches_p1,
           std::uintptr_t stream) {
            launch_batched_correction<T>(Z.data(), W_all.data(), cats.data(),
                                         R.data(), n_cells, n_pcs, n_clusters,
                                         n_batches_p1, (cudaStream_t)stream);
        },
        "Z"_a, nb::kw_only(), "W_all"_a, "cats"_a, "R"_a, "n_cells"_a,
        "n_pcs"_a, "n_clusters"_a, "n_batches_p1"_a, "stream"_a = 0);
}

template <typename Device>
void register_bindings(nb::module_& m) {
    def_outer<float, Device>(m);
    def_outer<double, Device>(m);
    def_harmony_corr<float, Device>(m);
    def_harmony_corr<double, Device>(m);
    def_batched_correction<float, Device>(m);
    def_batched_correction<double, Device>(m);
}

NB_MODULE(_harmony_outer_cuda, m) {
    REGISTER_GPU_BINDINGS(register_bindings, m);
}
