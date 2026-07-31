#include <cuda_runtime.h>
#include "../nb_types.h"

#include "kernels_jaccard.cuh"

using namespace nb::literals;

constexpr int JACCARD_BLOCK_SIZE = 256;

static inline void launch_jaccard_shared_counts(const int* knn, int n_obs,
                                                int k, float* jaccard_vals,
                                                cudaStream_t stream) {
    const long long n_edges = (long long)n_obs * k;
    if (n_edges == 0) return;
    unsigned grid = strided_grid(n_edges, JACCARD_BLOCK_SIZE);
    jaccard_shared_counts_kernel<<<grid, JACCARD_BLOCK_SIZE, 0, stream>>>(
        knn, n_obs, k, jaccard_vals);
    CUDA_CHECK_LAST_ERROR(jaccard_shared_counts_kernel);
}

template <typename Device>
void register_bindings(nb::module_& m) {
    m.def(
        "jaccard_shared_counts",
        [](gpu_array_c<const int, Device> knn, int n_obs, int k,
           gpu_array_c<float, Device> jaccard_vals, std::uintptr_t stream) {
            launch_jaccard_shared_counts(knn.data(), n_obs, k,
                                         jaccard_vals.data(),
                                         (cudaStream_t)stream);
        },
        "knn"_a, nb::kw_only(), "n_obs"_a, "k"_a, "jaccard_vals"_a,
        "stream"_a = 0);
}

NB_MODULE(_jaccard_cuda, m) {
    REGISTER_GPU_BINDINGS(register_bindings, m);
}
