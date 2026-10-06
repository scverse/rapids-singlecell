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

template <typename Device>
void register_bindings(nb::module_& m) {
    def_outer<float, Device>(m);
    def_outer<double, Device>(m);
}

NB_MODULE(_harmony_outer_cuda, m) {
    REGISTER_GPU_BINDINGS(register_bindings, m);
}
