#include <cuda_runtime.h>
#include "../nb_types.h"

#include "kernels_pseudobulk.cuh"

using namespace nb::literals;

static constexpr int PSEUDOBULK_MAX_BLOCK_SIZE = 256;

// Round n_features up to the next power of two, floored at WARP_SIZE so the
// block always has a full warp for warp-shuffle reductions, and capped at
// PSEUDOBULK_MAX_BLOCK_SIZE.
static int block_size_for(int64_t n_features) {
    int bs = WARP_SIZE;
    while (bs < n_features && bs < PSEUDOBULK_MAX_BLOCK_SIZE) {
        bs <<= 1;
    }
    return bs;
}

template <PseudobulkOp Op>
static void launch_paired(const double* X, const double* Y, double* out,
                          int64_t n_pairs, int64_t n_features,
                          cudaStream_t stream) {
    if (n_pairs == 0) return;
    int block_size = block_size_for(n_features);
    dim3 grid(static_cast<unsigned int>(n_pairs));
    dim3 block(block_size);
    paired_kernel<Op>
        <<<grid, block, 0, stream>>>(X, Y, out, n_pairs, n_features);
    CUDA_CHECK_LAST_ERROR(paired_kernel);
}

template <PseudobulkOp Op>
static void launch_pairwise(const double* X, const double* Y, double* out,
                            int64_t n_x, int64_t n_y, int64_t n_features,
                            cudaStream_t stream) {
    if (n_x == 0 || n_y == 0) return;
    int block_size = block_size_for(n_features);
    int64_t n_pairs = n_x * n_y;
    dim3 grid(static_cast<unsigned int>(n_pairs));
    dim3 block(block_size);
    pairwise_kernel<Op>
        <<<grid, block, 0, stream>>>(X, Y, out, n_x, n_y, n_features);
    CUDA_CHECK_LAST_ERROR(pairwise_kernel);
}

template <typename Device>
void register_bindings(nb::module_& m) {
    m.def(
        "paired_squared",
        [](gpu_array_c<const double, Device> X,
           gpu_array_c<const double, Device> Y, gpu_array_c<double, Device> out,
           int64_t n_pairs, int64_t n_features, std::uintptr_t stream) {
            launch_paired<PseudobulkOp::Squared>(
                X.data(), Y.data(), out.data(), n_pairs, n_features,
                reinterpret_cast<cudaStream_t>(stream));
        },
        "X"_a, "Y"_a, nb::kw_only(), "out"_a, "n_pairs"_a, "n_features"_a,
        "stream"_a = 0);

    m.def(
        "paired_abs_mean",
        [](gpu_array_c<const double, Device> X,
           gpu_array_c<const double, Device> Y, gpu_array_c<double, Device> out,
           int64_t n_pairs, int64_t n_features, std::uintptr_t stream) {
            launch_paired<PseudobulkOp::AbsMean>(
                X.data(), Y.data(), out.data(), n_pairs, n_features,
                reinterpret_cast<cudaStream_t>(stream));
        },
        "X"_a, "Y"_a, nb::kw_only(), "out"_a, "n_pairs"_a, "n_features"_a,
        "stream"_a = 0);

    m.def(
        "pairwise_squared",
        [](gpu_array_c<const double, Device> X,
           gpu_array_c<const double, Device> Y, gpu_array_c<double, Device> out,
           int64_t n_x, int64_t n_y, int64_t n_features,
           std::uintptr_t stream) {
            launch_pairwise<PseudobulkOp::Squared>(
                X.data(), Y.data(), out.data(), n_x, n_y, n_features,
                reinterpret_cast<cudaStream_t>(stream));
        },
        "X"_a, "Y"_a, nb::kw_only(), "out"_a, "n_x"_a, "n_y"_a, "n_features"_a,
        "stream"_a = 0);

    m.def(
        "pairwise_abs_mean",
        [](gpu_array_c<const double, Device> X,
           gpu_array_c<const double, Device> Y, gpu_array_c<double, Device> out,
           int64_t n_x, int64_t n_y, int64_t n_features,
           std::uintptr_t stream) {
            launch_pairwise<PseudobulkOp::AbsMean>(
                X.data(), Y.data(), out.data(), n_x, n_y, n_features,
                reinterpret_cast<cudaStream_t>(stream));
        },
        "X"_a, "Y"_a, nb::kw_only(), "out"_a, "n_x"_a, "n_y"_a, "n_features"_a,
        "stream"_a = 0);
}

NB_MODULE(_pseudobulk_cuda, m) {
    REGISTER_GPU_BINDINGS(register_bindings, m);
}
