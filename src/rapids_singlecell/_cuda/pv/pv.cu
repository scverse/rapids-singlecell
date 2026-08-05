#include <cuda_runtime.h>
#include "../nb_types.h"

#include "kernels_pv.cuh"

using namespace nb::literals;

static inline void launch_rev_cummin64(const double* x, double* y, int n_rows,
                                       int m, cudaStream_t stream) {
    constexpr int BLOCK_SIZE = 256;
    dim3 block(BLOCK_SIZE);
    dim3 grid((unsigned)((n_rows + BLOCK_SIZE - 1) / BLOCK_SIZE));
    rev_cummin64_kernel<<<grid, block, 0, stream>>>(x, y, n_rows, m);
    CUDA_CHECK_LAST_ERROR(rev_cummin64_kernel);
}

template <typename Device>
void register_bindings(nb::module_& m) {
    m.def(
        "rev_cummin64",
        [](gpu_array_c<const double, Device> x, gpu_array_c<double, Device> out,
           int n_rows, int m, std::uintptr_t stream) {
            launch_rev_cummin64(x.data(), out.data(), n_rows, m,
                                (cudaStream_t)stream);
        },
        "x"_a, nb::kw_only(), "out"_a, "n_rows"_a, "m"_a, "stream"_a = 0);
}

NB_MODULE(_pv_cuda, m) {
    REGISTER_GPU_BINDINGS(register_bindings, m);
}
