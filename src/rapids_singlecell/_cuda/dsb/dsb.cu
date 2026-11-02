#include <cuda_runtime.h>

#include <cstdint>

#include "../nb_types.h"
#include "kernels_dsb.cuh"

using namespace nb::literals;

constexpr int WARPS_PER_BLOCK = 8;

template <typename T, typename Device>
static void def_mix2(nb::module_& m) {
    // Empty W / init_in / init_out mean "not given" (see mix2_kernel).
    m.def(
        "mix2",
        [](gpu_array_c<const T, Device> X, gpu_array_c<const T, Device> W,
           gpu_array_c<const T, Device> init_in,
           gpu_array_c<T, Device> init_out, gpu_array_c<T, Device> params,
           gpu_array_c<std::uint8_t, Device> flags, bool subset, int max_iter,
           double margin, std::uintptr_t stream) {
            nb_require(X.ndim() == 2, "dsb: X must be 2-D");
            const long long n_fits = X.shape(0);
            const size_t n_vals = n_fits * X.shape(1);
            nb_require(W.size() == 0 || W.size() == n_vals,
                       "dsb: W must match X");
            nb_require(init_in.size() == 0 || init_in.size() == n_fits * 12,
                       "dsb: init_in must be (n_fits, 2, 6)");
            nb_require(init_out.size() == 0 || init_out.size() == n_fits * 12,
                       "dsb: init_out must be (n_fits, 2, 6)");
            nb_require(init_out.size() > 0 || (params.size() == n_fits * 6 &&
                                               flags.size() == n_fits),
                       "dsb: params must be (n_fits, 6), flags (n_fits,)");
            nb_require(flags.size() == 0 || flags.size() == n_fits,
                       "dsb: flags must be (n_fits,)");
            if (n_fits == 0) return;
            const unsigned int grid = strided_grid(n_fits, WARPS_PER_BLOCK);
            dsb::mix2_kernel<T><<<grid, WARPS_PER_BLOCK * dsb::kWarp, 0,
                                  (cudaStream_t)stream>>>(
                X.data(), W.size() ? W.data() : nullptr, n_fits,
                (int)X.shape(1), init_in.size() ? init_in.data() : nullptr,
                init_out.size() ? init_out.data() : nullptr, subset, max_iter,
                margin, params.data(), flags.size() ? flags.data() : nullptr);
            CUDA_CHECK_LAST_ERROR(mix2_kernel);
        },
        "X"_a, "W"_a, "init_in"_a, "init_out"_a.noconvert(),
        "params"_a.noconvert(), "flags"_a.noconvert(), nb::kw_only(),
        "subset"_a, "max_iter"_a, "margin"_a = 0.0, "stream"_a = 0);
}

template <typename Device>
void register_bindings(nb::module_& m) {
    def_mix2<float, Device>(m);
    def_mix2<double, Device>(m);
}

NB_MODULE(_dsb_cuda, m) {
    m.doc() = "Batched two-component univariate Gaussian mixtures for dsb.";
    REGISTER_GPU_BINDINGS(register_bindings, m);
}
