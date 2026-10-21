#include <algorithm>
#include <cuda_runtime.h>
#include <cusolverDn.h>
#include <nanobind/stl/optional.h>
#include <nanobind/stl/pair.h>
#include <optional>
#include <stdexcept>
#include <utility>
#include <vector>

#include "../../cublas_helpers.cuh"
#include "../../nb_types.h"
#include "../../rmm_scratch.h"

#include "../comm.cuh"
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

// out[g * group_stride + k * ld_out + d] = sum over the cells of group g (on
// all GPUs of comm) of R_ik X_id (R row stride ldr), exact over the segments
// (segment_gemm.cuh) in 1 (float32) or 2 limbs of 30 bits; bounds (n_groups x
// n_pcs): exponents of n_g max|X_gd| (segment_gemm.cuh).
template <typename T, typename RT, typename OT>
static void segment_sums(const T* X, const RT* R, int ldr, int n_pcs,
                         int n_clusters, const int* seg_start,
                         const int* seg_group, int n_seg, int n_groups,
                         const short* bounds, long long* acc, OT* out,
                         long long group_stride, int ld_out,
                         harmony_comm::Comm* comm, int rank,
                         cudaStream_t stream) {
    const int limbs = RHS_LIMBS<T>;
    cuda_check(cudaMemsetAsync(acc, 0,
                               (size_t)n_groups * limbs * n_clusters * n_pcs *
                                   sizeof(long long),
                               stream),
               "segment sums memset");
    cuda_check(harmony_segments::segment_rtz(X, R, n_pcs, n_clusters, ldr,
                                             seg_start, seg_group, n_seg,
                                             bounds, acc, limbs, stream),
               "segment sums");
    harmony_comm::allgather_groups(comm, rank, acc,
                                   (size_t)limbs * n_clusters * n_pcs, stream);
    cuda_check(harmony_segments::finalize_rtz(acc, n_groups, n_pcs, n_clusters,
                                              limbs, bounds, out, group_stride,
                                              ld_out, stream),
               "segment sums");
}

// The regression systems (double) of a chunk's n_clusters clusters from their
// joint counts joint_O (J x n_clusters) and joint right-hand sides R_j^T X_j
// (J x n_clusters x n_pcs; cells sorted by joint category): the marginal and
// intercept rows are sums of those, the Gram matrices sums of the counts.
template <typename T>
static void prepare_multi_impl(
    const T* X, const T* R, int ldr, const T* joint_O, const int* joint_cats,
    const int* marginal_joint_offsets, const int* marginal_joint_indices,
    const T* lambda_kb, const uint8_t* active, int n_pcs, int n_clusters,
    int n_batches, int n_first, int n_covariates, const int* seg_start,
    const int* seg_group, int n_seg, int n_joint_categories,
    const short* bounds, double* gram, double* rhs, double* joint_rhs,
    long long* acc, harmony_comm::Comm* comm, int rank, cudaStream_t stream) {
    int nb1 = n_batches + 1;
    cuda_check(
        cudaMemsetAsync(
            gram, 0, (size_t)n_clusters * nb1 * nb1 * sizeof(double), stream),
        "gram memset");
    initialize_multi_gram_kernel<T><<<n_clusters, MAX_BLOCK_DIM, 0, stream>>>(
        lambda, joint_O, marginal_joint_offsets, marginal_joint_indices, gram,
        n_batches, n_clusters, n_joint_categories);
    CUDA_CHECK_LAST_ERROR(initialize_multi_gram_kernel);
    dim3 grid(strided_grid((long long)n_batches * n_clusters, BLOCK_DIM_1D));
    if (n_covariates == 2)
        add_joint_cross_kernel<T, 2><<<grid, BLOCK_DIM_1D, 0, stream>>>(
            joint_O, joint_cats, marginal_joint_offsets, marginal_joint_indices,
            lambda, gram, n_covariates, n_batches, n_clusters);
    else if (n_covariates == 3)
        add_joint_cross_kernel<T, 3><<<grid, BLOCK_DIM_1D, 0, stream>>>(
            joint_O, joint_cats, marginal_joint_offsets, marginal_joint_indices,
            lambda, gram, n_covariates, n_batches, n_clusters);
    else
        add_joint_cross_kernel<T, 0><<<grid, BLOCK_DIM_1D, 0, stream>>>(
            joint_O, joint_cats, marginal_joint_offsets, marginal_joint_indices,
            lambda, gram, n_covariates, n_batches, n_clusters);
    CUDA_CHECK_LAST_ERROR(add_joint_cross_kernel);

    segment_sums(X, R, ldr, n_pcs, n_clusters, seg_start, seg_group, n_seg,
                 n_joint_categories, bounds, acc, joint_rhs,
                 (long long)n_clusters * n_pcs, n_pcs, comm, rank, stream);
    marginal_from_joint_rhs_kernel<<<
        strided_grid((long long)n_clusters * n_batches * n_pcs, BLOCK_DIM_1D),
        BLOCK_DIM_1D, 0, stream>>>(joint_rhs, marginal_joint_offsets,
                                   marginal_joint_indices, rhs, n_pcs,
                                   n_clusters, n_batches);
    CUDA_CHECK_LAST_ERROR(marginal_from_joint_rhs_kernel);
    intercept_and_mask_kernel<T>
        <<<strided_grid((long long)n_clusters * n_pcs, BLOCK_DIM_1D),
           BLOCK_DIM_1D, 0, stream>>>(lambda, rhs, n_pcs, n_clusters, n_batches,
                                      n_first);
    CUDA_CHECK_LAST_ERROR(intercept_and_mask_kernel);
}

// A chunk's k systems of n unknowns solve faster batched than one at a time
// (blocked factorization) unless they are few and large, n > 1000 k: measured
// on an RTX PRO 6000 Blackwell for n = 64 .. 6000, k = 1 .. 100, 20 .. 100 PCs.
// The two give different bits, and batched ones also depend on k (cuBLAS
// picks its batched trsm by the batch size), so this rule and the chunk
// budget (_multi_correction_cluster_chunk_size) set the results: changing
// either changes them for the designs it moves.
constexpr int ONE_AT_A_TIME_UNKNOWNS = 1000;  // per system of a chunk
static bool solve_one_at_a_time(int k, int n) {
    return n > ONE_AT_A_TIME_UNKNOWNS * k;
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
    harmony_comm::Comm* comm, int rank, cudaStream_t stream) {
    long long s_cluster = (long long)n_batches * n_pcs;
    segment_sums(X, R + k0, n_clusters, n_pcs, n_k, seg_start, seg_group, n_seg,
                 n_batches, bounds, rhs_acc, S, n_pcs, (int)s_cluster, comm,
                 rank, stream);
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
            cuda_check(
                cudaMemsetAsync(out.data(), 0, out.size() * sizeof(T), s),
                "group_absmax memset");
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
           int cluster_start, bool resume, bool finish, std::uintptr_t stream,
           std::uintptr_t comm, int rank) {
            // S: the clusters of this call, from cluster_start
            int n_pcs = (int)X.shape(1), n_clusters = (int)R.shape(1),
                n_batches = bound_groups(bounds, n_pcs);
            if (chunk < 1 || seg_group.size() + 1 != seg_start.size())
                throw std::invalid_argument("correction_batched: bad shapes");
            // S and W of one pass: (chunk, n_batches, n_pcs).
            long long s_cluster = (long long)n_batches * n_pcs;
            size_t n_coef = (size_t)s_cluster * chunk;
            RmmScratchPool pool;
            T *S = pool.alloc<T>(n_coef), *W = pool.alloc<T>(n_coef);
            long long* rhs_acc = pool.alloc<long long>(RHS_LIMBS<T> * n_coef);
            int n_seg = (int)seg_group.size();
            auto s = (cudaStream_t)stream;
            auto run = [&](auto* r) {
                correction_batched_impl<T>(
                    X.data(), r, O.data(), lambda_kb.data(), n_pcs, n_clusters,
                    cluster_start, n_k, seg_start.data(), seg_group.data(),
                    (int)seg_group.size(), n_batches, bounds.data(), Z.data(),
                    S.data(), W.data(), rhs_acc.data(), normalize, resume,
                    finish, (harmony_comm::Comm*)comm, rank,
                    (cudaStream_t)stream);
            };
            if (!R_bf16) return run(R.data());
            if constexpr (std::is_same_v<T, float>)
                return run(
                    reinterpret_cast<const __nv_bfloat16*>(R_bf16->data()));
            throw std::invalid_argument("bfloat16 assignments require float32");
        },
        nb::call_guard<nb::gil_scoped_release>(), "X"_a.noconvert(),
        nb::kw_only(), "R"_a.noconvert(), "O"_a.noconvert(),
        "lambda_kb"_a.noconvert(), "seg_start"_a.noconvert(),
        "seg_group"_a.noconvert(), "bounds"_a.noconvert(), "Z"_a.noconvert(),
        "S"_a.noconvert(), "W"_a.noconvert(), "rhs_acc"_a.noconvert(),
        "normalize"_a = false, "R_bf16"_a.noconvert() = nb::none(),
        "cluster_start"_a = 0, "resume"_a = false, "finish"_a = true,
        "stream"_a = 0, "comm"_a = 0, "rank"_a = 0);

    m.def(
        "correction_multi",
        [](gpu_array_c<const T, Device> X, gpu_array_c<const T, Device> R,
           gpu_array_c<const T, Device> lambda_kb,
           gpu_array_c<const int, Device> joint_cats,
           gpu_array_c<const int, Device> marginal_joint_offsets,
           gpu_array_c<const int, Device> marginal_joint_indices, int n_first,
           gpu_array_c<const int, Device> seg_start,
           gpu_array_c<const int, Device> seg_group,
           gpu_array_c<const short, Device> bounds,
           gpu_array_c<double, Device> gram, gpu_array_c<double, Device> rhs,
           gpu_array_c<double, Device> joint_rhs,
           gpu_array_c<long long, Device> acc, std::uintptr_t stream,
           std::uintptr_t comm, int rank) {
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
                (harmony_comm::Comm*)comm, rank, (cudaStream_t)stream);
        },
        nb::call_guard<nb::gil_scoped_release>(), "X"_a.noconvert(),
        nb::kw_only(), "R"_a.noconvert(), "joint_O"_a.noconvert(),
        "joint_cats"_a.noconvert(), "marginal_joint_offsets"_a.noconvert(),
        "marginal_joint_indices"_a.noconvert(), "lambda_kb"_a.noconvert(),
        "active_mask"_a.noconvert(), "n_batches"_a, "n_first"_a,
        "seg_start"_a.noconvert(), "seg_group"_a.noconvert(),
        "bounds"_a.noconvert(), "gram"_a.noconvert(), "rhs"_a.noconvert(),
        "joint_rhs"_a.noconvert(), "acc"_a.noconvert(), "stream"_a = 0,
        "comm"_a = 0, "rank"_a = 0);

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

    m.def(
        "solve_spd",
        [](gpu_array_c<double, Device> gram,  // batched: overwritten
           gpu_array_c<const double, Device> rhs, gpu_array_c<double, Device> x,
           bool one_at_a_time, std::uintptr_t stream, std::uintptr_t blas,
           std::uintptr_t solver) {
            if (gram.ndim() != 3 || gram.shape(1) != gram.shape(2) ||
                rhs.ndim() != 3 || rhs.shape(0) != gram.shape(0) ||
                rhs.shape(1) != gram.shape(1) || x.size() != rhs.size())
                throw std::invalid_argument("solve_spd: bad shapes");
            return solve_systems(gram.data(), rhs.data(), x.data(),
                                 (int)gram.shape(0), (int)gram.shape(1),
                                 (int)rhs.shape(2), one_at_a_time,
                                 (cudaStream_t)stream, (cublasHandle_t)blas,
                                 (cusolverDnHandle_t)solver);
        },
        nb::call_guard<nb::gil_scoped_release>(), "gram"_a.noconvert(),
        "rhs"_a.noconvert(), "x"_a.noconvert(), nb::kw_only(),
        "one_at_a_time"_a = false, "stream"_a = 0, "blas"_a, "solver"_a);
}

NB_MODULE(_harmony_correction_batched_cuda, m) {
    REGISTER_GPU_BINDINGS(register_bindings, m);
}
