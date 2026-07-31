#include <cuda_runtime.h>
#include "../nb_types.h"

using namespace nb::literals;

template <typename T>
__global__ void expected_zeros_kernel(const T* __restrict__ scaled_means,
                                      const T* __restrict__ total_counts,
                                      T* __restrict__ expected, int n_genes,
                                      int n_cells) {
    int gene = blockDim.x * blockIdx.x + threadIdx.x;
    if (gene >= n_genes) return;

    T sm = scaled_means[gene];
    T sum = T(0.0);

    for (int c = 0; c < n_cells; c++) {
        sum += exp(-sm * total_counts[c]);
    }

    expected[gene] = sum / T(n_cells);
}

template <typename T>
static void launch_expected_zeros(const T* scaled_means, const T* total_counts,
                                  T* expected, int n_genes, int n_cells,
                                  cudaStream_t stream) {
    constexpr int BLOCK_SIZE = 256;
    int grid_size = (n_genes + BLOCK_SIZE - 1) / BLOCK_SIZE;
    expected_zeros_kernel<T><<<grid_size, BLOCK_SIZE, 0, stream>>>(
        scaled_means, total_counts, expected, n_genes, n_cells);
    CUDA_CHECK_LAST_ERROR(expected_zeros_kernel);
}

template <typename T, typename Device>
void def_expected_zeros(nb::module_& m) {
    m.def(
        "expected_zeros",
        [](gpu_array_c<const T, Device> scaled_means,
           gpu_array_c<const T, Device> total_counts,
           gpu_array_c<T, Device> expected, int n_genes, int n_cells,
           std::uintptr_t stream) {
            launch_expected_zeros<T>(scaled_means.data(), total_counts.data(),
                                     expected.data(), n_genes, n_cells,
                                     reinterpret_cast<cudaStream_t>(stream));
        },
        "scaled_means"_a, "total_counts"_a, "expected"_a, "n_genes"_a,
        "n_cells"_a, "stream"_a = 0);
}

template <typename Device>
void register_bindings(nb::module_& m) {
    def_expected_zeros<float, Device>(m);
    def_expected_zeros<double, Device>(m);
}

NB_MODULE(_hvg_cuda, m) {
    REGISTER_GPU_BINDINGS(register_bindings, m);
}
