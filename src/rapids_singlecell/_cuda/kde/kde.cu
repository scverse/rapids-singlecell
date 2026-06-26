#include "kernels_kde.cuh"
#include "../nb_types.h"

using namespace nb::literals;

template <typename T>
inline void launch_gaussian_kde_2d(const T* xy, T* out, int n, T a, T b, T c,
                                   cudaStream_t stream) {
    constexpr int threads = 256;
    const int blocks = (n + threads - 1) / threads;
    gaussian_kde_2d_kernel<<<blocks, threads, 0, stream>>>(xy, out, n, a, b, c);
    CUDA_CHECK_LAST_ERROR(gaussian_kde_2d_kernel);
}

template <typename Device>
void register_bindings(nb::module_& m) {
    m.def(
        "gaussian_kde_2d",
        [](gpu_array_c<const float, Device> xy, gpu_array_c<float, Device> out,
           int n, float a, float b, float c, std::uintptr_t stream) {
            launch_gaussian_kde_2d(xy.data(), out.data(), n, a, b, c,
                                   (cudaStream_t)stream);
        },
        "xy"_a, nb::kw_only(), "out"_a, "n"_a, "a"_a, "b"_a, "c"_a,
        "stream"_a = 0);

    m.def(
        "gaussian_kde_2d",
        [](gpu_array_c<const double, Device> xy,
           gpu_array_c<double, Device> out, int n, double a, double b, double c,
           std::uintptr_t stream) {
            launch_gaussian_kde_2d(xy.data(), out.data(), n, a, b, c,
                                   (cudaStream_t)stream);
        },
        "xy"_a, nb::kw_only(), "out"_a, "n"_a, "a"_a, "b"_a, "c"_a,
        "stream"_a = 0);
}

NB_MODULE(_kde_cuda, m) {
    REGISTER_GPU_BINDINGS(register_bindings, m);
}
