#include <cuda_runtime.h>

#include "../nb_types.h"

#include "kernels_guide_assignment.cuh"

using namespace nb::literals;

template <bool F_ORDER>
static inline void launch_assign_threshold_dense(
    const float* X, const int* valid_guides, const float* lam, const float* mu,
    const float* sigma, const float* pi0, bool* assignments, float* thresholds,
    int n_cells, int n_guides, int n_valid_guides, float posterior_threshold,
    cudaStream_t stream) {
    if (n_valid_guides == 0) return;

    dim3 block(BLOCK_SIZE);
    dim3 grid(n_valid_guides);
    assign_threshold_dense_kernel<F_ORDER><<<grid, block, 0, stream>>>(
        X, valid_guides, lam, mu, sigma, pi0, assignments, thresholds, n_cells,
        n_guides, posterior_threshold);
    CUDA_CHECK_LAST_ERROR(assign_threshold_dense_kernel);
}

template <bool F_ORDER>
static inline void launch_fit_assign_dense(
    const float* X, bool* assignments, float* thresholds, float* lam, float* mu,
    float* sigma, float* pi0, bool* valid_mask, int* nonzero_counts,
    int* max_counts, int n_cells, int n_guides, int max_iter, float tol,
    float posterior_threshold, cudaStream_t stream) {
    if (n_guides == 0) return;

    dim3 block(BLOCK_SIZE);
    dim3 grid(n_guides);
    fit_assign_dense_kernel<F_ORDER><<<grid, block, 0, stream>>>(
        X, assignments, thresholds, lam, mu, sigma, pi0, valid_mask,
        nonzero_counts, max_counts, n_cells, n_guides, max_iter, tol,
        posterior_threshold);
    CUDA_CHECK_LAST_ERROR(fit_assign_dense_kernel);
}

template <typename Device, typename InContig, bool F_ORDER>
void def_assign_threshold_dense(nb::module_& m) {
    m.def(
        "assign_threshold_dense",
        [](gpu_array_contig<const float, Device, InContig> X,
           gpu_array_c<const int, Device> valid_guides,
           gpu_array_c<const float, Device> lam,
           gpu_array_c<const float, Device> mu,
           gpu_array_c<const float, Device> sigma,
           gpu_array_c<const float, Device> pi0,
           gpu_array_c<bool, Device> assignments,
           gpu_array_c<float, Device> thresholds, int n_cells, int n_guides,
           int n_valid_guides, float posterior_threshold,
           std::uintptr_t stream) {
            launch_assign_threshold_dense<F_ORDER>(
                X.data(), valid_guides.data(), lam.data(), mu.data(),
                sigma.data(), pi0.data(), assignments.data(), thresholds.data(),
                n_cells, n_guides, n_valid_guides, posterior_threshold,
                (cudaStream_t)stream);
        },
        "X"_a, "valid_guides"_a, "lam"_a, "mu"_a, "sigma"_a, "pi0"_a,
        "assignments"_a, "thresholds"_a, nb::kw_only(), "n_cells"_a,
        "n_guides"_a, "n_valid_guides"_a, "posterior_threshold"_a,
        "stream"_a = 0);
}

template <typename Device, typename InContig, bool F_ORDER>
void def_fit_assign_dense(nb::module_& m) {
    m.def(
        "fit_assign_dense",
        [](gpu_array_contig<const float, Device, InContig> X,
           gpu_array_c<bool, Device> assignments,
           gpu_array_c<float, Device> thresholds,
           gpu_array_c<float, Device> lam, gpu_array_c<float, Device> mu,
           gpu_array_c<float, Device> sigma, gpu_array_c<float, Device> pi0,
           gpu_array_c<bool, Device> valid_mask,
           gpu_array_c<int, Device> nonzero_counts,
           gpu_array_c<int, Device> max_counts, int n_cells, int n_guides,
           int max_iter, float tol, float posterior_threshold,
           std::uintptr_t stream) {
            launch_fit_assign_dense<F_ORDER>(
                X.data(), assignments.data(), thresholds.data(), lam.data(),
                mu.data(), sigma.data(), pi0.data(), valid_mask.data(),
                nonzero_counts.data(), max_counts.data(), n_cells, n_guides,
                max_iter, tol, posterior_threshold, (cudaStream_t)stream);
        },
        "X"_a, "assignments"_a, "thresholds"_a, "lam"_a, "mu"_a, "sigma"_a,
        "pi0"_a, "valid_mask"_a, "nonzero_counts"_a, "max_counts"_a,
        nb::kw_only(), "n_cells"_a, "n_guides"_a, "max_iter"_a, "tol"_a,
        "posterior_threshold"_a, "stream"_a = 0);
}

template <typename Device>
void register_bindings(nb::module_& m) {
    def_assign_threshold_dense<Device, nb::f_contig, true>(m);
    def_assign_threshold_dense<Device, nb::c_contig, false>(m);
    def_fit_assign_dense<Device, nb::f_contig, true>(m);
    def_fit_assign_dense<Device, nb::c_contig, false>(m);
}

NB_MODULE(_guide_assignment_cuda, m) {
    REGISTER_GPU_BINDINGS(register_bindings, m);
}
