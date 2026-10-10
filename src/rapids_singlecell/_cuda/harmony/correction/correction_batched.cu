#include <cuda_runtime.h>
#include <nanobind/stl/optional.h>
#include <nanobind/stl/vector.h>
#include <algorithm>
#include <optional>
#include <stdexcept>
#include <vector>

#include "../../nb_types.h"

#include "../outer/kernels_outer.cuh"
#include "../../cublas_helpers.cuh"
#include "../segment_gemm.cuh"
#include "apply_rows.cuh"
#include "kernels_correction_fast.cuh"
#include "kernels_correction_multi.cuh"

using namespace nb::literals;

constexpr int WARP_SIZE = 32;
constexpr int MAX_BLOCK_DIM = 256;
constexpr int BLOCK_DIM_1D = 256;

// out[g * group_stride + k * ld_out + d] = sum over the cells of group g of
// R_ik X_id (R row stride ldr), exact over the segments (segment_gemm.cuh) in
// 1 (float32) or 2 limbs of 30 bits; bounds[g] >= n_g max|X|.
template <typename T, typename RT>
static void segment_sums(const T* X, const RT* R, int ldr, int n_pcs,
                         int n_clusters, const int* seg_start,
                         const int* seg_group, int n_seg,
                         const std::vector<double>& bounds, double* scales,
                         long long* acc, T* out, long long group_stride,
                         int ld_out, cudaStream_t stream) {
    int limbs = sizeof(T) == 4 ? 1 : 2, n_groups = (int)bounds.size();
    std::vector<double> h(n_groups);
    for (int g = 0; g < n_groups; g++)
        h[g] = harmony_segments::scale_for_bound(bounds[g], limbs);
    cudaMemcpyAsync(scales, h.data(), n_groups * sizeof(double),
                    cudaMemcpyHostToDevice, stream);
    cudaMemsetAsync(
        acc, 0,
        (size_t)n_groups * limbs * n_clusters * n_pcs * sizeof(long long),
        stream);
    cuda_check(harmony_segments::segment_rtz(X, R, n_pcs, n_clusters, ldr,
                                             seg_start, seg_group, n_seg,
                                             scales, acc, limbs, stream),
               "segment sums");
    cuda_check(harmony_segments::finalize_rtz(acc, n_groups, n_pcs, n_clusters,
                                              limbs, scales, out, group_stride,
                                              ld_out, stream),
               "segment sums");
}

// Multi-key regression systems for the clusters of R (a column slice with
// row stride ldr). Cells are sorted by joint category: the joint right-hand
// sides R_j^T X_j are exact segment sums, the marginal and intercept rows sums
// of those.
template <typename T>
static void prepare_multi_impl(
    const T* X, const T* R, int ldr, const T* O, const T* joint_O,
    const int* joint_cats, const int* marginal_joint_offsets,
    const int* marginal_joint_indices, const T* lambda_kb,
    const uint8_t* active, int n_pcs, int n_clusters, int n_batches,
    int n_covariates, const int* seg_start, const int* seg_group, int n_seg,
    const std::vector<double>& bounds, T* gram, T* rhs, T* joint_rhs,
    double* scales, long long* acc, cudaStream_t stream) {
    int nb1 = n_batches + 1, n_joint_categories = (int)bounds.size();
    cudaMemsetAsync(gram, 0, (size_t)n_clusters * nb1 * nb1 * sizeof(T),
                    stream);
    initialize_multi_gram_kernel<T><<<n_clusters, MAX_BLOCK_DIM, 0, stream>>>(
        O, lambda_kb, active, joint_O, gram, n_batches, n_clusters,
        n_joint_categories);
    CUDA_CHECK_LAST_ERROR(initialize_multi_gram_kernel);
    dim3 grid(strided_grid((long long)n_batches * n_clusters, BLOCK_DIM_1D));
    if (n_covariates == 2)
        add_joint_cross_kernel<T, 2><<<grid, BLOCK_DIM_1D, 0, stream>>>(
            joint_O, joint_cats, marginal_joint_offsets, marginal_joint_indices,
            active, gram, n_covariates, n_batches, n_clusters);
    else if (n_covariates == 3)
        add_joint_cross_kernel<T, 3><<<grid, BLOCK_DIM_1D, 0, stream>>>(
            joint_O, joint_cats, marginal_joint_offsets, marginal_joint_indices,
            active, gram, n_covariates, n_batches, n_clusters);
    else
        add_joint_cross_kernel<T, 0><<<grid, BLOCK_DIM_1D, 0, stream>>>(
            joint_O, joint_cats, marginal_joint_offsets, marginal_joint_indices,
            active, gram, n_covariates, n_batches, n_clusters);
    CUDA_CHECK_LAST_ERROR(add_joint_cross_kernel);

    segment_sums(X, R, ldr, n_pcs, n_clusters, seg_start, seg_group, n_seg,
                 bounds, scales, acc, joint_rhs, (long long)n_clusters * n_pcs,
                 n_pcs, stream);
    marginal_from_joint_rhs_kernel<T>
        <<<strided_grid((long long)n_clusters * nb1 * n_pcs, BLOCK_DIM_1D),
           BLOCK_DIM_1D, 0, stream>>>(
            joint_rhs, marginal_joint_offsets, marginal_joint_indices, active,
            rhs, n_pcs, n_clusters, n_batches, n_joint_categories);
    CUDA_CHECK_LAST_ERROR(marginal_from_joint_rhs_kernel);
}

// Z = X - R_i W_joint[joint of i] (normalized on request) for the clusters of
// R (column slice, row stride ldr); W_joint[j] sums the marginal rows of W_all
// over the categories of joint j. resume / finish: see apply_rows.
template <typename T>
static void apply_multi_impl(const T* X, const T* R, int ldr, const T* W_all,
                             const int* joint_cats, int n_pcs, int n_clusters,
                             int n_batches, int n_covariates,
                             int n_joint_categories, const int* seg_start,
                             const int* seg_group, int n_seg, bool normalize,
                             bool resume, bool finish, T* W_joint, T* Z,
                             cudaStream_t stream) {
    long long joint_size = (long long)n_clusters * n_pcs;
    joint_coefficients_kernel<T>
        <<<strided_grid(n_joint_categories * joint_size, BLOCK_DIM_1D),
           BLOCK_DIM_1D, 0, stream>>>(W_all, joint_cats, W_joint, n_pcs,
                                      n_clusters, n_batches, n_covariates,
                                      n_joint_categories);
    CUDA_CHECK_LAST_ERROR(joint_coefficients_kernel);
    cuda_check(
        harmony_apply::apply_rows(X, R, ldr, W_joint, n_pcs, joint_size, n_pcs,
                                  n_clusters, seg_start, seg_group, n_seg,
                                  normalize, Z, stream, resume, finish),
        "multi-key apply");
}

// Batched correction of cells sorted by batch, all clusters at once. The
// right-hand sides R_b^T X_b are exact sums over segments (segment_gemm.cuh;
// `bounds[b]` >= n_b max|X|), row 0 their sum; Z = X - R_i W[batch] is
// applied row by row and normalized on request (apply_rows.cuh), so results
// do not depend on how the cells are split.
template <typename T, typename RT>
static void correction_batched_impl(
    const T* X, const RT* R, const T* O, const T* lambda_kb, int n_pcs,
    int n_clusters, const int* seg_start, const int* seg_group, int n_seg,
    const std::vector<double>& bounds,
    // workspace
    T* Z, T* inv_mats, T* Phi_t_diag_R_X_all, T* W_all, T* g_factor,
    T* g_P_row0, double* scales, long long* rhs_acc, bool normalize,
    cudaStream_t stream, cublasHandle_t handle) {
    int n_batches = (int)bounds.size(), nb1 = n_batches + 1;

    // inv_mats for all clusters at once (cluster_k = -1)
    int bdim = std::min(MAX_BLOCK_DIM,
                        std::max(WARP_SIZE, (n_batches + WARP_SIZE - 1) /
                                                WARP_SIZE * WARP_SIZE));
    compute_inv_mats_kernel<T><<<n_clusters, bdim, 0, stream>>>(
        O, lambda_kb, inv_mats, g_factor, g_P_row0, n_batches, n_clusters,
        /*cluster_k=*/-1);
    CUDA_CHECK_LAST_ERROR(compute_inv_mats_kernel);

    // Phi_t_diag_R_X_all (n_clusters, nb1, n_pcs): rows b + 1 are R_b^T X_b.
    segment_sums(X, R, n_clusters, n_pcs, n_clusters, seg_start, seg_group,
                 n_seg, bounds, scales, rhs_acc, Phi_t_diag_R_X_all + n_pcs,
                 n_pcs, nb1 * n_pcs, stream);
    sum_batch_rows_kernel<T>
        <<<strided_grid((long long)n_clusters * n_pcs, BLOCK_DIM_1D),
           BLOCK_DIM_1D, 0, stream>>>(Phi_t_diag_R_X_all, n_batches, n_pcs,
                                      n_clusters);
    CUDA_CHECK_LAST_ERROR(sum_batch_rows_kernel);

    // W_all = inv_mats @ Phi_t_diag_R_X_all per cluster (column-major view:
    // C(n_pcs, nb1) = B(n_pcs, nb1) @ A(nb1, nb1)), then W_all[:, 0, :] = 0.
    cublas_check_status(cublasSetStream(handle, stream), "cublasSetStream");
    T one = T(1), zero = T(0);
    long long s_rhs = (long long)nb1 * n_pcs, s_inv = (long long)nb1 * nb1;
    cublas_check_status(cublas_gemm_strided_batched<T>(
                            handle, CUBLAS_OP_N, CUBLAS_OP_N, n_pcs, nb1, nb1,
                            &one, Phi_t_diag_R_X_all, n_pcs, s_rhs, inv_mats,
                            nb1, s_inv, &zero, W_all, n_pcs, s_rhs, n_clusters),
                        "cublas_gemm_strided_batched(W_all)");
    cudaMemset2DAsync(W_all, (size_t)nb1 * n_pcs * sizeof(T), 0,
                      n_pcs * sizeof(T), n_clusters, stream);

    cuda_check(
        harmony_apply::apply_rows(X, R, n_clusters, W_all + n_pcs, s_rhs, n_pcs,
                                  n_pcs, n_clusters, seg_start, seg_group,
                                  n_seg, normalize, Z, stream),
        "correction apply");
}

// ---- nanobind registration ----

template <typename T, typename Device>
static void register_correction_batched(nb::module_& m) {
    m.def(
        "correction_batched",
        [](gpu_array_c<const T, Device> X, gpu_array_c<const T, Device> R,
           gpu_array_c<const T, Device> O,
           gpu_array_c<const T, Device> lambda_kb,
           gpu_array_c<const int, Device> seg_start,
           gpu_array_c<const int, Device> seg_group, std::vector<double> bounds,
           // workspace
           gpu_array_c<T, Device> Z, gpu_array_c<T, Device> inv_mats,
           gpu_array_c<T, Device> Phi_t_diag_R_X_all,
           gpu_array_c<T, Device> W_all, gpu_array_c<T, Device> g_factor,
           gpu_array_c<T, Device> g_P_row0, gpu_array_c<double, Device> scales,
           gpu_array_c<long long, Device> rhs_acc, bool normalize,
           std::optional<gpu_array_c<uint16_t, Device>> R_bf16,
           std::uintptr_t stream, std::uintptr_t handle) {
            int n_pcs = (int)X.shape(1), n_clusters = (int)R.shape(1);
            size_t limbs = sizeof(T) == 4 ? 1 : 2;
            if (scales.size() < bounds.size() ||
                rhs_acc.size() < bounds.size() * limbs * n_clusters * n_pcs ||
                seg_group.size() + 1 != seg_start.size())
                throw std::invalid_argument(
                    "correction_batched workspace is too small");
            auto run = [&](auto* r) {
                correction_batched_impl<T>(
                    X.data(), r, O.data(), lambda_kb.data(), n_pcs, n_clusters,
                    seg_start.data(), seg_group.data(), (int)seg_group.size(),
                    bounds, Z.data(), inv_mats.data(),
                    Phi_t_diag_R_X_all.data(), W_all.data(), g_factor.data(),
                    g_P_row0.data(), scales.data(), rhs_acc.data(), normalize,
                    (cudaStream_t)stream, (cublasHandle_t)handle);
            };
            if (!R_bf16) return run(R.data());
            if constexpr (std::is_same_v<T, float>)
                return run(
                    reinterpret_cast<const __nv_bfloat16*>(R_bf16->data()));
            throw std::invalid_argument("bfloat16 assignments require float32");
        },
        "X"_a, nb::kw_only(), "R"_a, "O"_a, "lambda_kb"_a, "seg_start"_a,
        "seg_group"_a, "bounds"_a, "Z"_a, "inv_mats"_a, "Phi_t_diag_R_X_all"_a,
        "W_all"_a, "g_factor"_a, "g_P_row0"_a, "scales"_a, "rhs_acc"_a,
        "normalize"_a = false, "R_bf16"_a = nb::none(), "stream"_a = 0,
        "handle"_a);

    m.def(
        "prepare_multi",
        [](gpu_array_c<const T, Device> X, gpu_array<const T, Device> R,
           gpu_array_c<const T, Device> O, gpu_array_c<const T, Device> joint_O,
           gpu_array_c<const int, Device> joint_cats,
           gpu_array_c<const int, Device> marginal_joint_offsets,
           gpu_array_c<const int, Device> marginal_joint_indices,
           gpu_array_c<const T, Device> lambda_kb,
           gpu_array_c<const uint8_t, Device> active, int n_batches,
           gpu_array_c<const int, Device> seg_start,
           gpu_array_c<const int, Device> seg_group, std::vector<double> bounds,
           gpu_array_c<T, Device> gram, gpu_array_c<T, Device> rhs,
           gpu_array_c<T, Device> joint_rhs, gpu_array_c<double, Device> scales,
           gpu_array_c<long long, Device> acc, std::uintptr_t stream) {
            if (R.ndim() != 2 || R.stride(1) != 1)
                throw std::invalid_argument("R must have contiguous rows");
            prepare_multi_impl<T>(
                X.data(), R.data(), (int)R.stride(0), O.data(), joint_O.data(),
                joint_cats.data(), marginal_joint_offsets.data(),
                marginal_joint_indices.data(), lambda_kb.data(), active.data(),
                (int)X.shape(1), (int)R.shape(1), n_batches,
                (int)joint_cats.shape(1), seg_start.data(), seg_group.data(),
                (int)seg_group.size(), bounds, gram.data(), rhs.data(),
                joint_rhs.data(), scales.data(), acc.data(),
                (cudaStream_t)stream);
        },
        "X"_a, nb::kw_only(), "R"_a, "O"_a, "joint_O"_a, "joint_cats"_a,
        "marginal_joint_offsets"_a, "marginal_joint_indices"_a, "lambda_kb"_a,
        "active_mask"_a, "n_batches"_a, "seg_start"_a, "seg_group"_a,
        "bounds"_a, "gram"_a, "rhs"_a, "joint_rhs"_a, "scales"_a, "acc"_a,
        "stream"_a = 0);

    m.def(
        "apply_multi",
        [](gpu_array_c<const T, Device> X, gpu_array<const T, Device> R,
           gpu_array_c<const T, Device> W_all,
           gpu_array_c<const int, Device> joint_cats, int n_batches,
           gpu_array_c<const int, Device> seg_start,
           gpu_array_c<const int, Device> seg_group, bool normalize,
           gpu_array_c<T, Device> W_joint, gpu_array_c<T, Device> Z,
           bool resume, bool finish, std::uintptr_t stream) {
            if (R.ndim() != 2 || R.stride(1) != 1)
                throw std::invalid_argument("R must have contiguous rows");
            apply_multi_impl<T>(
                X.data(), R.data(), (int)R.stride(0), W_all.data(),
                joint_cats.data(), (int)X.shape(1), (int)R.shape(1), n_batches,
                (int)joint_cats.shape(1), (int)joint_cats.shape(0),
                seg_start.data(), seg_group.data(), (int)seg_group.size(),
                normalize, resume, finish, W_joint.data(), Z.data(),
                (cudaStream_t)stream);
        },
        "X"_a, nb::kw_only(), "R"_a, "W_all"_a, "joint_cats"_a, "n_batches"_a,
        "seg_start"_a, "seg_group"_a, "normalize"_a, "W_joint"_a, "Z"_a,
        "resume"_a = false, "finish"_a = true, "stream"_a = 0);
}

template <typename Device>
void register_bindings(nb::module_& m) {
    register_correction_batched<float, Device>(m);
    register_correction_batched<double, Device>(m);
}

NB_MODULE(_harmony_correction_batched_cuda, m) {
    REGISTER_GPU_BINDINGS(register_bindings, m);
}
