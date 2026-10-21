#include <cuda_runtime.h>
#include <nanobind/stl/optional.h>
#include <optional>
#include <stdexcept>

#include "../../nb_types.h"

#include "../segment_gemm.cuh"
#include "apply_rows.cuh"
#include "kernels_correction_fast.cuh"
#include "kernels_correction_multi.cuh"

using namespace nb::literals;

constexpr int MAX_BLOCK_DIM = 256;
constexpr int BLOCK_DIM_1D = 256;

// out[g][d] = max |X_id| over the cells i of group g in the segments, as
// bit patterns of |x| (exact and order-free; NaN above inf), so every split
// of the cells gives the same maxima.
template <typename T>
__global__ void group_absmax_kernel(const T* X, int D, const int* seg_start,
                                    const int* seg_group, T* out) {
    using U = std::conditional_t<sizeof(T) == 4, unsigned, unsigned long long>;
    U* o = reinterpret_cast<U*>(out) + (size_t)seg_group[blockIdx.x] * D;
    for (int d = threadIdx.x; d < D; d += blockDim.x) {
        U m = 0;
        for (int i = seg_start[blockIdx.x] + threadIdx.y;
             i < seg_start[blockIdx.x + 1]; i += blockDim.y) {
            U u;
            memcpy(&u, X + (size_t)i * D + d, sizeof(T));
            m = max(m, u & (~U(0) >> 1));
        }
        atomicMax(o + d, m);
    }
}

// out[g * group_stride + k * ld_out + d] = sum over the cells of group g of
// R_ik X_id (R row stride ldr), exact over the segments (segment_gemm.cuh) in
// 1 (float32) or 2 limbs of 30 bits; bounds (n_groups x n_pcs): exponents of
// n_g max|X_gd| (segment_gemm.cuh).
template <typename T, typename RT, typename OT>
static void segment_sums(const T* X, const RT* R, int ldr, int n_pcs,
                         int n_clusters, const int* seg_start,
                         const int* seg_group, int n_seg, int n_groups,
                         const short* bounds, long long* acc, OT* out,
                         long long group_stride, int ld_out,
                         cudaStream_t stream) {
    int limbs = sizeof(T) == 4 ? 1 : 2;
    cudaMemsetAsync(
        acc, 0,
        (size_t)n_groups * limbs * n_clusters * n_pcs * sizeof(long long),
        stream);
    cuda_check(harmony_segments::segment_rtz(X, R, n_pcs, n_clusters, ldr,
                                             seg_start, seg_group, n_seg,
                                             bounds, acc, limbs, stream),
               "segment sums");
    cuda_check(harmony_segments::finalize_rtz(acc, n_groups, n_pcs, n_clusters,
                                              limbs, bounds, out, group_stride,
                                              ld_out, stream),
               "segment sums");
}

// Multi-key regression systems (double) for the clusters of R (a column slice
// with row stride ldr). Cells are sorted by joint category: the joint
// right-hand sides R_j^T X_j are exact segment sums, the marginal and
// intercept rows sums of those, and the Gram matrices sums of the joint counts.
template <typename T>
static void prepare_multi_impl(
    const T* X, const T* R, int ldr, const T* joint_O, const int* joint_cats,
    const int* marginal_joint_offsets, const int* marginal_joint_indices,
    const T* lambda_kb, const uint8_t* active, int n_pcs, int n_clusters,
    int n_batches, int n_first, int n_covariates, const int* seg_start,
    const int* seg_group, int n_seg, int n_joint_categories,
    const short* bounds, double* gram, double* rhs, double* joint_rhs,
    long long* acc, cudaStream_t stream) {
    int nb1 = n_batches + 1;
    cudaMemsetAsync(gram, 0, (size_t)n_clusters * nb1 * nb1 * sizeof(double),
                    stream);
    initialize_multi_gram_kernel<T><<<n_clusters, MAX_BLOCK_DIM, 0, stream>>>(
        lambda_kb, active, joint_O, marginal_joint_offsets,
        marginal_joint_indices, gram, n_batches, n_clusters,
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
                 n_joint_categories, bounds, acc, joint_rhs,
                 (long long)n_clusters * n_pcs, n_pcs, stream);
    marginal_from_joint_rhs_kernel<<<
        strided_grid((long long)n_clusters * n_batches * n_pcs, BLOCK_DIM_1D),
        BLOCK_DIM_1D, 0, stream>>>(joint_rhs, marginal_joint_offsets,
                                   marginal_joint_indices, rhs, n_pcs,
                                   n_clusters, n_batches);
    CUDA_CHECK_LAST_ERROR(marginal_from_joint_rhs_kernel);
    intercept_and_mask_kernel<<<strided_grid((long long)n_clusters * n_pcs,
                                             BLOCK_DIM_1D),
                                BLOCK_DIM_1D, 0, stream>>>(
        active, rhs, n_pcs, n_clusters, n_batches, n_first);
    CUDA_CHECK_LAST_ERROR(intercept_and_mask_kernel);
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

// Batched correction of cells sorted by batch, clusters k0 .. k0 + n_k - 1
// (all at once, or in chunks: resume / finish as in apply_rows). The
// right-hand sides S_b = R_b^T X_b are exact sums over segments
// (segment_gemm.cuh; bounds[b][d]: exponent of n_b max|X_bd|), the
// coefficients the closed form of ridge_coefficients; Z = X - R_i W[batch] is
// applied row by row and normalized on request (apply_rows.cuh), so results do
// not depend on how the cells are split. S and W are (n_k, n_batches, n_pcs).
template <typename T, typename RT>
static void correction_batched_impl(
    const T* X, const RT* R, const T* O, const T* lambda_kb, int n_pcs,
    int n_clusters, int k0, int n_k, const int* seg_start, const int* seg_group,
    int n_seg, int n_batches, const short* bounds, T* Z, T* S, T* W,
    long long* rhs_acc, bool normalize, bool resume, bool finish,
    cudaStream_t stream) {
    long long s_cluster = (long long)n_batches * n_pcs;
    segment_sums(X, R + k0, n_clusters, n_pcs, n_k, seg_start, seg_group, n_seg,
                 n_batches, bounds, rhs_acc, S, n_pcs, (int)s_cluster, stream);
    // rhs_acc (summed into S) holds the intercepts
    ridge_coefficients(O, lambda_kb, S, W, reinterpret_cast<double*>(rhs_acc),
                       s_cluster, n_batches, n_clusters, n_pcs, k0, n_k,
                       stream);
    cuda_check(
        harmony_apply::apply_rows(X, R + k0, n_clusters, W, s_cluster, n_pcs,
                                  n_pcs, n_k, seg_start, seg_group, n_seg,
                                  normalize, Z, stream, resume, finish),
        "correction apply");
}

// ---- nanobind registration ----

// The right-hand side bounds must be n_groups x n_pcs.
template <typename Bounds>
static int bound_groups(const Bounds& bounds, size_t n_pcs) {
    if (bounds.ndim() != 2 || bounds.shape(1) != n_pcs)
        throw std::invalid_argument("bounds must be groups x PCs");
    return (int)bounds.shape(0);
}

template <typename T, typename Device>
static void register_correction_batched(nb::module_& m) {
    m.def(
        "group_absmax",
        [](gpu_array_c<const T, Device> X,
           gpu_array_c<const int, Device> seg_start,
           gpu_array_c<const int, Device> seg_group, gpu_array_c<T, Device> out,
           std::uintptr_t stream) {
            if (out.ndim() != 2 || out.shape(1) != X.shape(1) ||
                seg_group.size() + 1 != seg_start.size())
                throw std::invalid_argument("group_absmax: bad shapes");
            auto s = (cudaStream_t)stream;
            cudaMemsetAsync(out.data(), 0, out.size() * sizeof(T), s);
            if (seg_group.size() == 0) return;
            group_absmax_kernel<T>
                <<<(unsigned)seg_group.size(), dim3(32, 8), 0, s>>>(
                    X.data(), (int)X.shape(1), seg_start.data(),
                    seg_group.data(), out.data());
            CUDA_CHECK_LAST_ERROR(group_absmax_kernel);
        },
        "X"_a.noconvert(), nb::kw_only(), "seg_start"_a.noconvert(),
        "seg_group"_a.noconvert(), "out"_a.noconvert(), "stream"_a = 0);

    m.def(
        "correction_batched",
        [](gpu_array_c<const T, Device> X, gpu_array_c<const T, Device> R,
           gpu_array_c<const T, Device> O,
           gpu_array_c<const T, Device> lambda_kb,
           gpu_array_c<const int, Device> seg_start,
           gpu_array_c<const int, Device> seg_group,
           gpu_array_c<const short, Device> bounds,
           // workspace
           gpu_array_c<T, Device> Z, gpu_array_c<T, Device> S,
           gpu_array_c<T, Device> W, gpu_array_c<long long, Device> rhs_acc,
           bool normalize, std::optional<gpu_array_c<uint16_t, Device>> R_bf16,
           int cluster_start, bool resume, bool finish, std::uintptr_t stream) {
            // S: the clusters of this call, from cluster_start
            int n_pcs = (int)X.shape(1), n_clusters = (int)R.shape(1),
                n_k = (int)S.shape(0), n_batches = bound_groups(bounds, n_pcs);
            size_t limbs = sizeof(T) == 4 ? 1 : 2,
                   n_coef = (size_t)n_batches * n_k * n_pcs;
            if (S.ndim() != 3 || cluster_start < 0 ||
                cluster_start + n_k > n_clusters)
                throw std::invalid_argument("correction_batched: bad clusters");
            if (rhs_acc.size() < limbs * n_coef || S.size() < n_coef ||
                W.size() < n_coef || seg_group.size() + 1 != seg_start.size())
                throw std::invalid_argument(
                    "correction_batched workspace is too small");
            auto run = [&](auto* r) {
                correction_batched_impl<T>(
                    X.data(), r, O.data(), lambda_kb.data(), n_pcs, n_clusters,
                    cluster_start, n_k, seg_start.data(), seg_group.data(),
                    (int)seg_group.size(), n_batches, bounds.data(), Z.data(),
                    S.data(), W.data(), rhs_acc.data(), normalize, resume,
                    finish, (cudaStream_t)stream);
            };
            if (!R_bf16) return run(R.data());
            if constexpr (std::is_same_v<T, float>)
                return run(
                    reinterpret_cast<const __nv_bfloat16*>(R_bf16->data()));
            throw std::invalid_argument("bfloat16 assignments require float32");
        },
        "X"_a.noconvert(), nb::kw_only(), "R"_a.noconvert(), "O"_a.noconvert(),
        "lambda_kb"_a.noconvert(), "seg_start"_a.noconvert(),
        "seg_group"_a.noconvert(), "bounds"_a.noconvert(), "Z"_a.noconvert(),
        "S"_a.noconvert(), "W"_a.noconvert(), "rhs_acc"_a.noconvert(),
        "normalize"_a = false, "R_bf16"_a.noconvert() = nb::none(),
        "cluster_start"_a = 0, "resume"_a = false, "finish"_a = true,
        "stream"_a = 0);

    m.def(
        "prepare_multi",
        [](gpu_array_c<const T, Device> X, gpu_array<const T, Device> R,
           gpu_array_c<const T, Device> joint_O,
           gpu_array_c<const int, Device> joint_cats,
           gpu_array_c<const int, Device> marginal_joint_offsets,
           gpu_array_c<const int, Device> marginal_joint_indices,
           gpu_array_c<const T, Device> lambda_kb,
           gpu_array_c<const uint8_t, Device> active, int n_batches,
           int n_first, gpu_array_c<const int, Device> seg_start,
           gpu_array_c<const int, Device> seg_group,
           gpu_array_c<const short, Device> bounds,
           gpu_array_c<double, Device> gram, gpu_array_c<double, Device> rhs,
           gpu_array_c<double, Device> joint_rhs,
           gpu_array_c<long long, Device> acc, std::uintptr_t stream) {
            if (R.ndim() != 2 || R.stride(1) != 1)
                throw std::invalid_argument("R must have contiguous rows");
            if (n_first < 1 || n_first > n_batches)
                throw std::invalid_argument(
                    "n_first must be in [1, n_batches]");
            int n_joint_categories = bound_groups(bounds, X.shape(1));
            prepare_multi_impl<T>(
                X.data(), R.data(), (int)R.stride(0), joint_O.data(),
                joint_cats.data(), marginal_joint_offsets.data(),
                marginal_joint_indices.data(), lambda_kb.data(), active.data(),
                (int)X.shape(1), (int)R.shape(1), n_batches, n_first,
                (int)joint_cats.shape(1), seg_start.data(), seg_group.data(),
                (int)seg_group.size(), n_joint_categories, bounds.data(),
                gram.data(), rhs.data(), joint_rhs.data(), acc.data(),
                (cudaStream_t)stream);
        },
        "X"_a.noconvert(), nb::kw_only(), "R"_a.noconvert(),
        "joint_O"_a.noconvert(), "joint_cats"_a.noconvert(),
        "marginal_joint_offsets"_a.noconvert(),
        "marginal_joint_indices"_a.noconvert(), "lambda_kb"_a.noconvert(),
        "active_mask"_a.noconvert(), "n_batches"_a, "n_first"_a,
        "seg_start"_a.noconvert(), "seg_group"_a.noconvert(),
        "bounds"_a.noconvert(), "gram"_a.noconvert(), "rhs"_a.noconvert(),
        "joint_rhs"_a.noconvert(), "acc"_a.noconvert(), "stream"_a = 0);

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
        "X"_a.noconvert(), nb::kw_only(), "R"_a.noconvert(),
        "W_all"_a.noconvert(), "joint_cats"_a.noconvert(), "n_batches"_a,
        "seg_start"_a.noconvert(), "seg_group"_a.noconvert(), "normalize"_a,
        "W_joint"_a.noconvert(), "Z"_a.noconvert(), "resume"_a = false,
        "finish"_a = true, "stream"_a = 0);
}

template <typename Device>
void register_bindings(nb::module_& m) {
    register_correction_batched<float, Device>(m);
    register_correction_batched<double, Device>(m);
}

NB_MODULE(_harmony_correction_batched_cuda, m) {
    REGISTER_GPU_BINDINGS(register_bindings, m);
}
