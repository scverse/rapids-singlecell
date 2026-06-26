#include <cuda_runtime.h>
#include "../nb_types.h"

#include "kernels_dist.cuh"

using namespace nb::literals;

constexpr int BLOCK_SIZE = 32;

static inline void launch_sqeuclidean(const float* data, float* out,
                                      const unsigned int* pairs,
                                      long long n_samples, long long n_features,
                                      long long n_neighbors,
                                      cudaStream_t stream) {
    dim3 block(BLOCK_SIZE);
    dim3 grid((unsigned)((n_samples + BLOCK_SIZE - 1) / BLOCK_SIZE));
    compute_distances_sqeuclidean_kernel<<<grid, block, 0, stream>>>(
        data, out, pairs, n_samples, n_features, n_neighbors);
    CUDA_CHECK_LAST_ERROR(compute_distances_sqeuclidean_kernel);
}

static inline void launch_cosine(const float* data, float* out,
                                 const unsigned int* pairs, long long n_samples,
                                 long long n_features, long long n_neighbors,
                                 cudaStream_t stream) {
    dim3 block(BLOCK_SIZE);
    dim3 grid((unsigned)((n_samples + BLOCK_SIZE - 1) / BLOCK_SIZE));
    compute_distances_cosine_kernel<<<grid, block, 0, stream>>>(
        data, out, pairs, n_samples, n_features, n_neighbors);
    CUDA_CHECK_LAST_ERROR(compute_distances_cosine_kernel);
}

static inline void launch_inner(const float* data, float* out,
                                const unsigned int* pairs, long long n_samples,
                                long long n_features, long long n_neighbors,
                                cudaStream_t stream) {
    dim3 block(BLOCK_SIZE);
    dim3 grid((unsigned)((n_samples + BLOCK_SIZE - 1) / BLOCK_SIZE));
    compute_distances_inner_kernel<<<grid, block, 0, stream>>>(
        data, out, pairs, n_samples, n_features, n_neighbors);
    CUDA_CHECK_LAST_ERROR(compute_distances_inner_kernel);
}

template <typename Device>
void register_bindings(nb::module_& m) {
    m.def(
        "sqeuclidean",
        [](gpu_array_c<const float, Device> data,
           gpu_array_c<float, Device> out,
           gpu_array_c<const unsigned int, Device> pairs, long long n_samples,
           long long n_features, long long n_neighbors, std::uintptr_t stream) {
            launch_sqeuclidean(data.data(), out.data(), pairs.data(), n_samples,
                               n_features, n_neighbors, (cudaStream_t)stream);
        },
        "data"_a, nb::kw_only(), "out"_a, "pairs"_a, "n_samples"_a,
        "n_features"_a, "n_neighbors"_a, "stream"_a = 0);

    m.def(
        "cosine",
        [](gpu_array_c<const float, Device> data,
           gpu_array_c<float, Device> out,
           gpu_array_c<const unsigned int, Device> pairs, long long n_samples,
           long long n_features, long long n_neighbors, std::uintptr_t stream) {
            launch_cosine(data.data(), out.data(), pairs.data(), n_samples,
                          n_features, n_neighbors, (cudaStream_t)stream);
        },
        "data"_a, nb::kw_only(), "out"_a, "pairs"_a, "n_samples"_a,
        "n_features"_a, "n_neighbors"_a, "stream"_a = 0);

    m.def(
        "inner",
        [](gpu_array_c<const float, Device> data,
           gpu_array_c<float, Device> out,
           gpu_array_c<const unsigned int, Device> pairs, long long n_samples,
           long long n_features, long long n_neighbors, std::uintptr_t stream) {
            launch_inner(data.data(), out.data(), pairs.data(), n_samples,
                         n_features, n_neighbors, (cudaStream_t)stream);
        },
        "data"_a, nb::kw_only(), "out"_a, "pairs"_a, "n_samples"_a,
        "n_features"_a, "n_neighbors"_a, "stream"_a = 0);
}

NB_MODULE(_nn_descent_cuda, m) {
    REGISTER_GPU_BINDINGS(register_bindings, m);
}
