#include <cuda_runtime.h>
#include <nanobind/stl/optional.h>
#include <algorithm>
#include <optional>
#include <stdexcept>
#include <vector>

#include "../../nb_types.h"

#include "../outer/kernels_outer.cuh"
#include "../scatter/kernels_scatter.cuh"
#include "../../cublas_helpers.cuh"
#include "../cutile/cutile_runtime.cuh"
#include "kernels_correction_fast.cuh"
#include "kernels_correction_multi.cuh"

using namespace nb::literals;

constexpr int WARP_SIZE = 32;
constexpr int MAX_BLOCK_DIM = 256;
constexpr int BLOCK_DIM_1D = 256;

static std::vector<int> host_offsets(const int* offsets, int n,
                                     cudaStream_t stream) {
    std::vector<int> h(n + 1);
    cudaMemcpyAsync(h.data(), offsets, (size_t)(n + 1) * sizeof(int),
                    cudaMemcpyDeviceToHost, stream);
    cudaStreamSynchronize(stream);
    return h;
}

// Multi-key regression systems for clusters [c0, c0 + n_clusters) of R (row
// stride ld_r). Cells are sorted by joint category, so R_j^T X_j is one GEMM
// per joint; the marginal and intercept rows are sums of those products.
template <typename T>
static void prepare_multi_impl(
    const T* X, const T* R, int ld_r, const T* O, const T* joint_O,
    const int* joint_cats, const int* joint_offsets,
    const int* marginal_joint_offsets, const int* marginal_joint_indices,
    const T* lambda_kb, const uint8_t* active, int n_pcs, int n_clusters,
    int n_batches, int n_covariates, int n_joint_categories, T* gram, T* rhs,
    T* joint_rhs, cudaStream_t stream, cublasHandle_t handle) {
    if (n_covariates < 2)
        throw std::invalid_argument(
            "prepare_multi requires at least two covariates");
    int nb1 = n_batches + 1;
    if (n_clusters == 0) return;
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

    cublas_check_status(cublasSetStream(handle, stream), "cublasSetStream");
    T one = T(1), zero = T(0);
    auto h_offsets = host_offsets(joint_offsets, n_joint_categories, stream);
    size_t joint_size = (size_t)n_clusters * n_pcs;
    for (int j = 0; j < n_joint_categories; j++) {
        int start = h_offsets[j], n_joint_cells = h_offsets[j + 1] - start;
        T* out = joint_rhs + j * joint_size;
        if (n_joint_cells == 0) {
            cudaMemsetAsync(out, 0, joint_size * sizeof(T), stream);
            continue;
        }
        cublas_check_status(
            cublas_gemm<T>(handle, CUBLAS_OP_N, CUBLAS_OP_T, n_pcs, n_clusters,
                           n_joint_cells, &one, X + (size_t)start * n_pcs,
                           n_pcs, R + (size_t)start * ld_r, ld_r, &zero, out,
                           n_pcs),
            "cublas_gemm(multi-key rhs)");
    }
    marginal_from_joint_rhs_kernel<T>
        <<<strided_grid((long long)n_clusters * nb1 * n_pcs, BLOCK_DIM_1D),
           BLOCK_DIM_1D, 0, stream>>>(
            joint_rhs, marginal_joint_offsets, marginal_joint_indices, active,
            rhs, n_pcs, n_clusters, n_batches, n_joint_categories);
    CUDA_CHECK_LAST_ERROR(marginal_from_joint_rhs_kernel);
}

// Z_j (-)= R_j W_joint[j] for every joint j over the cluster range; with
// `finish`, Z = X + Z (L2-normalized on request).
template <typename T>
static void apply_multi_impl(const T* X, const T* R, int ld_r, const T* W_all,
                             const int* joint_cats, const int* joint_offsets,
                             int n_cells, int n_pcs, int n_clusters,
                             int n_batches, int n_covariates,
                             int n_joint_categories, bool accumulate,
                             bool finish, bool normalize, T* W_joint, T* Z,
                             cudaStream_t stream, cublasHandle_t handle) {
    size_t joint_size = (size_t)n_clusters * n_pcs;
    if (n_clusters > 0) {
        joint_coefficients_kernel<T>
            <<<strided_grid((long long)(n_joint_categories * joint_size),
                            BLOCK_DIM_1D),
               BLOCK_DIM_1D, 0, stream>>>(W_all, joint_cats, W_joint, n_pcs,
                                          n_clusters, n_batches, n_covariates,
                                          n_joint_categories);
        CUDA_CHECK_LAST_ERROR(joint_coefficients_kernel);
        cublas_check_status(cublasSetStream(handle, stream), "cublasSetStream");
        T minus_one = T(-1), beta = accumulate ? T(1) : T(0);
        auto h_offsets =
            host_offsets(joint_offsets, n_joint_categories, stream);
        for (int j = 0; j < n_joint_categories; j++) {
            int start = h_offsets[j], n_joint_cells = h_offsets[j + 1] - start;
            if (n_joint_cells == 0) continue;
            cublas_check_status(
                cublas_gemm<T>(handle, CUBLAS_OP_N, CUBLAS_OP_N, n_pcs,
                               n_joint_cells, n_clusters, &minus_one,
                               W_joint + j * joint_size, n_pcs,
                               R + (size_t)start * ld_r, ld_r, &beta,
                               Z + (size_t)start * n_pcs, n_pcs),
                "cublas_gemm(multi-key apply)");
        }
    } else if (!accumulate) {
        cudaMemsetAsync(Z, 0, (size_t)n_cells * n_pcs * sizeof(T), stream);
    }
    if (!finish) return;
    add_rows_normalize_kernel<T>
        <<<strided_grid((long long)n_cells * 32, BLOCK_DIM_1D), BLOCK_DIM_1D, 0,
           stream>>>(X, Z, n_cells, n_pcs, normalize);
    CUDA_CHECK_LAST_ERROR(add_rows_normalize_kernel);
}

template <typename T>
static void correction_batched_impl(
    const T* X, const T* R, const T* O, const int* cat_offsets,
    const T* lambda_kb, int n_cells, int n_pcs, int n_clusters, int n_batches,
    // workspace
    T* Z, T* inv_mats, T* Phi_t_diag_R_X_all, T* W_all, T* g_factor,
    T* g_P_row0, bool normalize, const __nv_bfloat16* R_bf16,
    float* tile_partials, cudaStream_t stream, cublasHandle_t handle) {
    // Cells are sorted by batch, so per-batch products read X and R in place;
    // row 0 of each right-hand side is the sum of the batch rows. With
    // R_bf16 (float32 only) both per-batch products run as cuTile kernels.
    if (R_bf16 && (!tile_partials || !std::is_same_v<T, float>))
        throw std::invalid_argument(
            "bfloat16 correction needs float32 input and tile_partials");
    int nb1 = n_batches + 1;

    // inv_mats for all clusters at once (cluster_k = -1)
    int bdim = std::min(MAX_BLOCK_DIM,
                        std::max(WARP_SIZE, (n_batches + WARP_SIZE - 1) /
                                                WARP_SIZE * WARP_SIZE));
    compute_inv_mats_kernel<T><<<n_clusters, bdim, 0, stream>>>(
        O, lambda_kb, inv_mats, g_factor, g_P_row0, n_batches, n_clusters,
        /*cluster_k=*/-1);
    CUDA_CHECK_LAST_ERROR(compute_inv_mats_kernel);
    cublas_check_status(cublasSetStream(handle, stream), "cublasSetStream");
    T one = T(1), zero = T(0), minus_one = T(-1);

    // Phi_t_diag_R_X_all (n_clusters, nb1, n_pcs); empty batches stay zero.
    cudaMemsetAsync(Phi_t_diag_R_X_all, 0,
                    (size_t)n_clusters * nb1 * n_pcs * sizeof(T), stream);
    auto h_offsets = host_offsets(cat_offsets, n_batches, stream);
    for (int b = 0; b < n_batches; b++) {
        int start = h_offsets[b], n_batch_cells = h_offsets[b + 1] - start;
        if (n_batch_cells == 0) continue;
        T* rhs = Phi_t_diag_R_X_all + (b + 1) * n_pcs;  // R_b^T X_b
        if (R_bf16) {
            if constexpr (std::is_same_v<T, float>)
                cuda_check(harmony_cutile::rtz(
                               R_bf16 + (size_t)start * n_clusters,
                               X + (size_t)start * n_pcs, n_batch_cells,
                               n_clusters, n_pcs, tile_partials, rhs,
                               (int64_t)nb1 * n_pcs, stream),
                           "cuTile correction rhs");
        } else {
            cublas_check_status(
                cublas_gemm<T>(handle, CUBLAS_OP_N, CUBLAS_OP_T, n_pcs,
                               n_clusters, n_batch_cells, &one,
                               X + (size_t)start * n_pcs, n_pcs,
                               R + (size_t)start * n_clusters, n_clusters,
                               &zero, rhs, nb1 * n_pcs),
                "cublas_gemm(correction rhs)");
        }
    }
    sum_batch_rows_kernel<T>
        <<<strided_grid((long long)n_clusters * n_pcs, BLOCK_DIM_1D),
           BLOCK_DIM_1D, 0, stream>>>(Phi_t_diag_R_X_all, n_batches, n_pcs,
                                      n_clusters);
    CUDA_CHECK_LAST_ERROR(sum_batch_rows_kernel);

    // W_all = inv_mats @ Phi_t_diag_R_X_all per cluster (column-major view:
    // C(n_pcs, nb1) = B(n_pcs, nb1) @ A(nb1, nb1)), then W_all[:, 0, :] = 0.
    long long s_rhs = (long long)nb1 * n_pcs, s_inv = (long long)nb1 * nb1;
    cublas_check_status(cublas_gemm_strided_batched<T>(
                            handle, CUBLAS_OP_N, CUBLAS_OP_N, n_pcs, nb1, nb1,
                            &one, Phi_t_diag_R_X_all, n_pcs, s_rhs, inv_mats,
                            nb1, s_inv, &zero, W_all, n_pcs, s_rhs, n_clusters),
                        "cublas_gemm_strided_batched(W_all)");
    cudaMemset2DAsync(W_all, (size_t)nb1 * n_pcs * sizeof(T), 0,
                      n_pcs * sizeof(T), n_clusters, stream);

    // Z_b = -R_b W[:, b + 1, :] per batch, then Z = X + Z (normalized on
    // request).
    for (int b = 0; b < n_batches; b++) {
        int start = h_offsets[b], n_batch_cells = h_offsets[b + 1] - start;
        if (n_batch_cells == 0) continue;
        const T* w = W_all + (size_t)(b + 1) * n_pcs;
        T* z = Z + (size_t)start * n_pcs;
        if (R_bf16) {
            if constexpr (std::is_same_v<T, float>)
                cuda_check(harmony_cutile::neg_rw(
                               R_bf16 + (size_t)start * n_clusters, w,
                               (int64_t)nb1 * n_pcs, z, n_batch_cells,
                               n_clusters, n_pcs, stream),
                           "cuTile correction apply");
        } else {
            cublas_check_status(
                cublas_gemm<T>(handle, CUBLAS_OP_N, CUBLAS_OP_N, n_pcs,
                               n_batch_cells, n_clusters, &minus_one, w,
                               nb1 * n_pcs, R + (size_t)start * n_clusters,
                               n_clusters, &zero, z, n_pcs),
                "cublas_gemm(correction apply)");
        }
    }
    add_rows_normalize_kernel<T>
        <<<strided_grid((long long)n_cells * 32, BLOCK_DIM_1D), BLOCK_DIM_1D, 0,
           stream>>>(X, Z, n_cells, n_pcs, normalize);
    CUDA_CHECK_LAST_ERROR(add_rows_normalize_kernel);
}

// ---- nanobind registration ----

template <typename T, typename Device>
static void register_correction_batched(nb::module_& m) {
    m.def(
        "correction_batched",
        [](gpu_array_c<const T, Device> X, gpu_array_c<const T, Device> R,
           gpu_array_c<const T, Device> O,
           gpu_array_c<const int, Device> cat_offsets,
           gpu_array_c<const T, Device> lambda_kb, int n_cells, int n_pcs,
           int n_clusters, int n_batches,
           // workspace
           gpu_array_c<T, Device> Z, gpu_array_c<T, Device> inv_mats,
           gpu_array_c<T, Device> Phi_t_diag_R_X_all,
           gpu_array_c<T, Device> W_all, gpu_array_c<T, Device> g_factor,
           gpu_array_c<T, Device> g_P_row0, bool normalize,
           std::optional<gpu_array_c<uint16_t, Device>> R_bf16,
           std::optional<gpu_array_c<T, Device>> tile_partials,
           std::uintptr_t stream, std::uintptr_t handle) {
            correction_batched_impl<T>(
                X.data(), R.data(), O.data(), cat_offsets.data(),
                lambda_kb.data(), n_cells, n_pcs, n_clusters, n_batches,
                Z.data(), inv_mats.data(), Phi_t_diag_R_X_all.data(),
                W_all.data(), g_factor.data(), g_P_row0.data(), normalize,
                R_bf16 ? reinterpret_cast<const __nv_bfloat16*>(R_bf16->data())
                       : nullptr,
                tile_partials ? reinterpret_cast<float*>(tile_partials->data())
                              : nullptr,
                (cudaStream_t)stream, (cublasHandle_t)handle);
        },
        "X"_a, nb::kw_only(), "R"_a, "O"_a, "cat_offsets"_a, "lambda_kb"_a,
        "n_cells"_a, "n_pcs"_a, "n_clusters"_a, "n_batches"_a, "Z"_a,
        "inv_mats"_a, "Phi_t_diag_R_X_all"_a, "W_all"_a, "g_factor"_a,
        "g_P_row0"_a, "normalize"_a = false, "R_bf16"_a = nb::none(),
        "tile_partials"_a = nb::none(), "stream"_a = 0, "handle"_a);

    m.def(
        "prepare_multi",
        [](gpu_array_c<const T, Device> X, gpu_array<const T, Device> R,
           gpu_array_c<const T, Device> O, gpu_array_c<const T, Device> joint_O,
           gpu_array_c<const int, Device> joint_cats,
           gpu_array_c<const int, Device> joint_offsets,
           gpu_array_c<const int, Device> marginal_joint_offsets,
           gpu_array_c<const int, Device> marginal_joint_indices,
           gpu_array_c<const T, Device> lambda_kb,
           gpu_array_c<const uint8_t, Device> active, int n_batches,
           int n_covariates, gpu_array_c<T, Device> gram,
           gpu_array_c<T, Device> rhs, gpu_array_c<T, Device> joint_rhs,
           std::uintptr_t stream, std::uintptr_t handle) {
            if (R.ndim() != 2 || R.stride(1) != 1)
                throw std::invalid_argument("R must have contiguous rows");
            prepare_multi_impl<T>(
                X.data(), R.data(), (int)R.stride(0), O.data(), joint_O.data(),
                joint_cats.data(), joint_offsets.data(),
                marginal_joint_offsets.data(), marginal_joint_indices.data(),
                lambda_kb.data(), active.data(), (int)X.shape(1),
                (int)R.shape(1), n_batches, n_covariates,
                (int)joint_cats.shape(0), gram.data(), rhs.data(),
                joint_rhs.data(), (cudaStream_t)stream, (cublasHandle_t)handle);
        },
        "X"_a, nb::kw_only(), "R"_a, "O"_a, "joint_O"_a, "joint_cats"_a,
        "joint_offsets"_a, "marginal_joint_offsets"_a,
        "marginal_joint_indices"_a, "lambda_kb"_a, "active_mask"_a,
        "n_batches"_a, "n_covariates"_a, "gram"_a, "rhs"_a, "joint_rhs"_a,
        "stream"_a = 0, "handle"_a);

    m.def(
        "apply_multi",
        [](gpu_array_c<const T, Device> X, gpu_array<const T, Device> R,
           gpu_array_c<const T, Device> W_all,
           gpu_array_c<const int, Device> joint_cats,
           gpu_array_c<const int, Device> joint_offsets, int n_batches,
           bool accumulate, bool finish, bool normalize,
           gpu_array_c<T, Device> W_joint, gpu_array_c<T, Device> Z,
           std::uintptr_t stream, std::uintptr_t handle) {
            if (R.ndim() != 2 || R.stride(1) != 1)
                throw std::invalid_argument("R must have contiguous rows");
            apply_multi_impl<T>(
                X.data(), R.data(), (int)R.stride(0), W_all.data(),
                joint_cats.data(), joint_offsets.data(), (int)X.shape(0),
                (int)X.shape(1), (int)R.shape(1), n_batches,
                (int)joint_cats.shape(1), (int)joint_cats.shape(0), accumulate,
                finish, normalize, W_joint.data(), Z.data(),
                (cudaStream_t)stream, (cublasHandle_t)handle);
        },
        "X"_a, nb::kw_only(), "R"_a, "W_all"_a, "joint_cats"_a,
        "joint_offsets"_a, "n_batches"_a, "accumulate"_a, "finish"_a,
        "normalize"_a, "W_joint"_a, "Z"_a, "stream"_a = 0, "handle"_a);
}

template <typename Device>
void register_bindings(nb::module_& m) {
    register_correction_batched<float, Device>(m);
    register_correction_batched<double, Device>(m);
}

NB_MODULE(_harmony_correction_batched_cuda, m) {
    m.def(
        "cutile_bf16_available",
        [](int n_pcs, int n_clusters) {
            return harmony_cutile::available(n_pcs, n_clusters);
        },
        "n_pcs"_a, "n_clusters"_a);
    REGISTER_GPU_BINDINGS(register_bindings, m);
}
