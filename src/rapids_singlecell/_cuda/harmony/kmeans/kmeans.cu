#include <cuda_runtime.h>
#include "../../nb_types.h"

#include "kernels_kmeans.cuh"

using namespace nb::literals;

template <typename T>
static inline void launch_kmeans_err(const T* r, const T* dot, size_t n, T* out,
                                     cudaStream_t stream) {
    int device;
    cudaGetDevice(&device);
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, device);

    constexpr int BLOCK_SIZE = 256;
    constexpr int BLOCKS_PER_SM = 8;
    int blocks =
        std::max(1, std::min((int)((n + BLOCK_SIZE - 1) / BLOCK_SIZE),
                             prop.multiProcessorCount * BLOCKS_PER_SM));
    kmeans_err_kernel<T><<<blocks, BLOCK_SIZE, 0, stream>>>(r, dot, n, out);
    CUDA_CHECK_LAST_ERROR(kmeans_err_kernel);
}

template <typename T, typename Device>
void def_kmeans_err(nb::module_& m) {
    m.def(
        "kmeans_err",
        [](gpu_array_c<const T, Device> r, gpu_array_c<const T, Device> dot,
           size_t n, gpu_array_c<T, Device> out, std::uintptr_t stream) {
            launch_kmeans_err<T>(r.data(), dot.data(), n, out.data(),
                                 (cudaStream_t)stream);
        },
        "r"_a, nb::kw_only(), "dot"_a, "n"_a, "out"_a, "stream"_a = 0);
}

template <typename Device>
void register_bindings(nb::module_& m) {
    // -- Test-only bindings below --
    // kmeans_err is used internally by compute_objective in the C++ clustering
    // loop. The binding exists solely for unit testing.
    def_kmeans_err<float, Device>(m);
    def_kmeans_err<double, Device>(m);
}

NB_MODULE(_harmony_kmeans_cuda, m) {
    REGISTER_GPU_BINDINGS(register_bindings, m);
}
