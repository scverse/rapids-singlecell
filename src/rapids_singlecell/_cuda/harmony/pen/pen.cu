#include <cuda_runtime.h>
#include "../../nb_types.h"

#include "kernels_pen.cuh"

using namespace nb::literals;

constexpr unsigned WARP_SIZE = 32;
constexpr unsigned MAX_BLOCK_DIM = 256;
constexpr int BLOCK_DIM_1D = 256;

template <typename T, typename IdxT>
static inline void launch_fused_pen_norm(const T* similarities,
                                         const T* penalty, const int* cats,
                                         const IdxT* idx_in, T* R_out, T term,
                                         int n_rows, int n_cols,
                                         cudaStream_t stream) {
    unsigned block_dim = std::min(
        MAX_BLOCK_DIM, std::max(WARP_SIZE, ((unsigned)n_cols + WARP_SIZE - 1u) /
                                               WARP_SIZE * WARP_SIZE));
    fused_pen_norm_kernel<T, IdxT><<<n_rows, block_dim, 0, stream>>>(
        similarities, penalty, cats, idx_in, R_out, term, n_rows, n_cols);
    CUDA_CHECK_LAST_ERROR(fused_pen_norm_kernel);
}

template <typename T>
static inline void launch_penalty(const T* E, const T* O, const T* theta,
                                  T* penalty, int n_batches, int n_clusters,
                                  bool stabilized, cudaStream_t stream) {
    int total = n_batches * n_clusters;
    if (stabilized)
        penalty_kernel<T, true>
            <<<(total + BLOCK_DIM_1D - 1) / BLOCK_DIM_1D, BLOCK_DIM_1D, 0,
               stream>>>(E, O, theta, penalty, n_batches, n_clusters);
    else
        penalty_kernel<T, false>
            <<<(total + BLOCK_DIM_1D - 1) / BLOCK_DIM_1D, BLOCK_DIM_1D, 0,
               stream>>>(E, O, theta, penalty, n_batches, n_clusters);
    CUDA_CHECK_LAST_ERROR(penalty_kernel);
}

template <typename T, typename Device>
static void register_fused_pen_norm(nb::module_& m) {
    m.def(
        "fused_pen_norm",
        [](gpu_array_c<const T, Device> similarities,
           gpu_array_c<const T, Device> penalty,
           gpu_array_c<const int, Device> cats,
           gpu_array_c<const size_t, Device> idx_in,
           gpu_array_c<T, Device> R_out, double term, int n_rows, int n_cols,
           std::uintptr_t stream) {
            launch_fused_pen_norm<T, size_t>(
                similarities.data(), penalty.data(), cats.data(), idx_in.data(),
                R_out.data(), static_cast<T>(term), n_rows, n_cols,
                (cudaStream_t)stream);
        },
        "similarities"_a, nb::kw_only(), "penalty"_a, "cats"_a, "idx_in"_a,
        "R_out"_a, "term"_a, "n_rows"_a, "n_cols"_a, "stream"_a = 0);
}

template <typename T, typename Device>
static void register_penalty(nb::module_& m) {
    m.def(
        "penalty",
        [](gpu_array_c<const T, Device> E, gpu_array_c<const T, Device> O,
           gpu_array_c<const T, Device> theta, gpu_array_c<T, Device> penalty,
           int n_batches, int n_clusters, bool stabilized,
           std::uintptr_t stream) {
            launch_penalty<T>(E.data(), O.data(), theta.data(), penalty.data(),
                              n_batches, n_clusters, stabilized,
                              (cudaStream_t)stream);
        },
        "E"_a, nb::kw_only(), "O"_a, "theta"_a, "penalty"_a, "n_batches"_a,
        "n_clusters"_a, "stabilized"_a, "stream"_a = 0);
}

template <typename T, typename Device>
static void register_fused_pen_norm_int(nb::module_& m) {
    m.def(
        "fused_pen_norm_int",
        [](gpu_array_c<const T, Device> similarities,
           gpu_array_c<const T, Device> penalty,
           gpu_array_c<const int, Device> cats,
           gpu_array_c<const int, Device> idx_in, gpu_array_c<T, Device> R_out,
           double term, int n_rows, int n_cols, std::uintptr_t stream) {
            launch_fused_pen_norm<T, int>(similarities.data(), penalty.data(),
                                          cats.data(), idx_in.data(),
                                          R_out.data(), static_cast<T>(term),
                                          n_rows, n_cols, (cudaStream_t)stream);
        },
        "similarities"_a, nb::kw_only(), "penalty"_a, "cats"_a, "idx_in"_a,
        "R_out"_a, "term"_a, "n_rows"_a, "n_cols"_a, "stream"_a = 0);
}

template <typename Device>
void register_bindings(nb::module_& m) {
    register_fused_pen_norm<float, Device>(m);
    register_fused_pen_norm<double, Device>(m);
    register_penalty<float, Device>(m);
    register_penalty<double, Device>(m);

    // -- Test-only bindings below --
    // fused_pen_norm_int uses int32 indices (used internally by the C++
    // clustering loop). The binding exists solely for unit testing.
    register_fused_pen_norm_int<float, Device>(m);
    register_fused_pen_norm_int<double, Device>(m);
}

NB_MODULE(_harmony_pen_cuda, m) {
    REGISTER_GPU_BINDINGS(register_bindings, m);
}
