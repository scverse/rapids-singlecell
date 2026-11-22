#include <algorithm>
#include <functional>
#include <cuda_runtime.h>
#include <cusolverDn.h>
#include <nanobind/stl/optional.h>
#include <nanobind/stl/pair.h>
#include <nanobind/stl/vector.h>
#include <optional>
#include <stdexcept>
#include <utility>
#include <vector>

#include "../../cublas_helpers.cuh"
#include "../../nb_types.h"
#include "../../rmm_scratch.h"

#include "../comm.cuh"
#include "../fixed_point.cuh"
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

// int64 limbs of 30 bits per exact right-hand side sum: 1 (float32) or 2.
template <typename T>
constexpr int RHS_LIMBS = sizeof(T) == 4 ? 1 : 2;

// out[g * group_stride + k * ld_out + d] = sum over the cells of group g (on
// all GPUs of comm) of R_ik X_id (R row stride ldr), exact over the segments
// (segment_gemm.cuh) in acc (n_groups x RHS_LIMBS x n_clusters x n_pcs);
// bounds (n_groups x n_pcs): exponents of n_g max|X_gd| (segment_gemm.cuh).
// Every GPU of comm receives the sums.
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

// A chunk's k systems of n unknowns solve faster batched than one at a time
// (blocked factorization) unless they are few and large, n > 1000 k: measured
// on an RTX PRO 6000 Blackwell for n = 64 .. 6000, k = 1 .. 100, 20 .. 100 PCs
// (float64; the float32 factorization below keeps the rule).
// The two give different bits, and batched ones also depend on k (cuBLAS
// picks its batched trsm by the batch size), so this rule and the chunk
// budget (_multi_correction_cluster_chunk_size) set the results: changing
// either changes them for the designs it moves.
constexpr int ONE_AT_A_TIME_UNKNOWNS = 1000;  // per system of a chunk
static bool solve_one_at_a_time(int k, int n) {
    return n > ONE_AT_A_TIME_UNKNOWNS * k;
}

// The pointers of the batched calls (systems, then right-hand sides), and in
// diag (k x n) the diagonals of the k systems (n x n), which they overwrite.
__global__ void system_pointers_kernel(double* gram, double* rhs, int k, int n,
                                       size_t rhs_stride, double** ptrs,
                                       double* diag) {
    long long e = blockIdx.x * (long long)blockDim.x + threadIdx.x;
    if (e >= (long long)k * n) return;
    long long i = e / n, j = e % n;
    diag[e] = gram[(i * n + j) * n + j];
    if (j == 0) {
        ptrs[i] = gram + i * n * n;
        ptrs[k + i] = rhs + i * rhs_stride;
    }
}

// Puts back the systems whose batched factorization failed (info) from their
// diagonals and C strict lower triangles, which it leaves alone (column-major
// upper): the systems are symmetric.
__global__ void restore_failed_kernel(double* gram, const double* diag,
                                      const int* info, int k, int n) {
    for (int i = blockIdx.y; i < k; i += gridDim.y) {
        if (!info[i]) continue;
        double* g = gram + (size_t)i * n * n;
        for (int r = blockIdx.x; r < n; r += gridDim.x)
            for (int c = r + threadIdx.x; c < n; c += blockDim.x)
                g[(size_t)r * n + c] =
                    c == r ? diag[(size_t)i * n + r] : g[(size_t)c * n + r];
    }
}

// The systems equilibrated for float32: D_i = 1/sqrt(diag gram_i) (into D)
// and As_i = D_i gram_i D_i.
__global__ void equilibrate_kernel(const double* gram, int k, int n, double* D,
                                   float* As) {
    long long total = (long long)k * n * n;
    for (long long e = blockIdx.x * (long long)blockDim.x + threadIdx.x;
         e < total; e += (long long)gridDim.x * blockDim.x) {
        long long i = e / ((long long)n * n), rem = e % ((long long)n * n);
        int r = (int)(rem / n), c = (int)(rem % n);
        const double* g = gram + i * (size_t)n * n;
        double dr = rsqrt(g[(size_t)r * n + r]);
        if (c == r) D[i * n + r] = dr;
        As[e] =
            (float)(g[(size_t)r * n + c] * dr * rsqrt(g[(size_t)c * n + c]));
    }
}

// out (+)= D in, row by row (k x n x p): right-hand sides and residuals to the
// equilibrated systems' scale (float32; a value beyond float32 sets *bad),
// solutions and corrections back to the systems' scale (float64).
template <typename In, typename Out>
__global__ void scale_rows_kernel(const In* in, const double* D, int k, int n,
                                  int p, bool add, Out* out, int* bad) {
    long long total = (long long)k * n * p;
    for (long long e = blockIdx.x * (long long)blockDim.x + threadIdx.x;
         e < total; e += (long long)gridDim.x * blockDim.x) {
        double v = D[e / p] * (double)in[e];
        if (add) v += out[e];
        out[e] = (Out)v;
        if (sizeof(Out) == 4 && !isfinite((float)v)) *bad = 1;
    }
}

__global__ void float_pointers_kernel(float* As, float* ys, int k,
                                      size_t gram_stride, size_t rhs_stride,
                                      float** ptrs) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= k) return;
    ptrs[i] = As + i * gram_stride;
    ptrs[k + i] = ys + i * rhs_stride;
}

// x = cupy.linalg.lstsq(G, b)[0], the minimum-norm solution by SVD, for a
// G (n x n) whose Cholesky factorization failed; b, x: n x p, C-ordered, on
// the current device and CuPy's current stream (the caller's).
static void least_squares(const double* G, const double* b, double* x, int n,
                          int p) {
    nb::gil_scoped_acquire gil;
    int device;
    cuda_check(cudaGetDevice(&device), "least squares");
    auto cupy = [&](const double* data, int cols) {
        return nb::cast(
            nb::ndarray<nb::cupy, double>(const_cast<double*>(data),
                                          {(size_t)n, (size_t)cols},
                                          nb::handle(), {}, nb::dtype<double>(),
                                          nb::device::cuda::value, device),
            nb::rv_policy::reference);
    };
    cupy(x, p)[nb::ellipsis()] =
        nb::module_::import_("cupy.linalg")
            .attr("lstsq")(cupy(G, n), cupy(b, p), "rcond"_a = nb::none())[0];
}

// Solves the k systems gram_i x_i = rhs_i (n x n, n x p; C-ordered) into x by
// float64 Cholesky factors of gram_i (a symmetric C-ordered matrix is its own
// column-major form), by least squares where gram_i is singular. Batched,
// gram_i is factored in place (gram is overwritten) and a C-ordered x_i is
// the column-major x_i^T: two right-side triangular solves give
// rhs_i^T inv(L_i L_i^T) = x_i^T in place. One at a time, a copy of gram_i is
// factored and a column-major copy of rhs_i solved (faster than right-side
// solves at that size). Returns how many needed least squares.
static int solve_systems_double(double* gram, const double* rhs, double* x,
                                int k, int n, int p, bool one_at_a_time,
                                cudaStream_t stream, cublasHandle_t blas,
                                cusolverDnHandle_t solver,
                                harmony_comm::Comm* comm) {
    if (k == 0) return 0;
    const double one = 1.0, zero = 0.0;
    const auto lower = CUBLAS_FILL_MODE_LOWER;
    const size_t gram_stride = (size_t)n * n, rhs_stride = (size_t)n * p;
    cublas_check_status(cublasSetStream(blas, stream), "cublasSetStream");
    cusolver_check(cusolverDnSetStream(solver, stream), "cusolverDnSetStream");
    RmmScratchPool pool;
    int* info;
    if (one_at_a_time) {
        int lwork = 0;
        cusolver_check(
            cusolverDnDpotrf_bufferSize(solver, lower, n, nullptr, n, &lwork),
            "cusolverDnDpotrf_bufferSize");
        double *L, *work, *Bt;  // Bt: rhs_i column-major
        int* solve_info;
        pool.alloc_all(info, k, L, gram_stride, work, lwork, Bt, rhs_stride,
                       solve_info, 1);
        for (int i = 0; i < k; ++i) {
            cuda_check(cudaMemcpyAsync(L, gram + i * gram_stride,
                                       gram_stride * sizeof(double),
                                       cudaMemcpyDeviceToDevice, stream),
                       "solve copy");
            cusolver_check(
                cusolverDnDpotrf(solver, lower, n, L, n, work, lwork, info + i),
                "cusolverDnDpotrf");
            double* x_i = x + i * rhs_stride;
            cublas_check_status(
                cublasDgeam(blas, CUBLAS_OP_T, CUBLAS_OP_N, n, p, &one,
                            rhs + i * rhs_stride, p, &zero, Bt, n, Bt, n),
                "cublasDgeam");
            cusolver_check(
                cusolverDnDpotrs(solver, lower, n, p, L, n, Bt, n, solve_info),
                "cusolverDnDpotrs");
            cublas_check_status(
                cublasDgeam(blas, CUBLAS_OP_T, CUBLAS_OP_N, p, n, &one, Bt, n,
                            &zero, x_i, p, x_i, p),
                "cublasDgeam");
        }
    } else {
        double* diag;
        double** ptrs;
        pool.alloc_all(info, k, diag, (size_t)k * n, ptrs, 2 * (size_t)k);
        cuda_check(cudaMemcpyAsync(x, rhs, k * rhs_stride * sizeof(double),
                                   cudaMemcpyDeviceToDevice, stream),
                   "solve copy");
        system_pointers_kernel<<<
            (unsigned)(((long long)k * n + BLOCK_DIM_1D - 1) / BLOCK_DIM_1D),
            BLOCK_DIM_1D, 0, stream>>>(gram, x, k, n, rhs_stride, ptrs, diag);
        CUDA_CHECK_LAST_ERROR(system_pointers_kernel);
        cusolver_check(
            cusolverDnDpotrfBatched(solver, lower, n, ptrs, n, info, k),
            "cusolverDnDpotrfBatched");
        for (auto op : {CUBLAS_OP_T, CUBLAS_OP_N})
            cublas_check_status(
                cublasDtrsmBatched(blas, CUBLAS_SIDE_RIGHT, lower, op,
                                   CUBLAS_DIAG_NON_UNIT, p, n, &one, ptrs, n,
                                   ptrs + k, p, k),
                "cublasDtrsmBatched");
        restore_failed_kernel<<<dim3(n, std::min(k, 65535)), BLOCK_DIM_1D, 0,
                                stream>>>(gram, diag, info, k, n);
        CUDA_CHECK_LAST_ERROR(restore_failed_kernel);
    }
    std::vector<int> failed(k);
    harmony_comm::synchronize(comm, stream, "solve");
    cuda_check(cudaMemcpyAsync(failed.data(), info, k * sizeof(int),
                               cudaMemcpyDeviceToHost, stream),
               "solve info");
    cuda_check(cudaStreamSynchronize(stream), "solve");
    int n_least_squares = 0;
    for (int i = 0; i < k; ++i)
        if (failed[i]) {
            least_squares(gram + i * gram_stride, rhs + i * rhs_stride,
                          x + i * rhs_stride, n, p);
            ++n_least_squares;
        }
    return n_least_squares;
}

// Solves the k systems in float32 after equilibration (D_i gram_i D_i has a
// unit diagonal and a condition number bounded by the ridge's share of it
// over the design's blocks, see _refinement_steps) and refines the solution
// `refine` times with float64 residuals: each step gains a factor of that
// condition number times 2^-24, so the caller picks the count from its bound,
// at a fraction of a float64 factorization's cost on GPUs with slow float64.
// refine < 0, a system whose float32 factorization fails (near-singular) or
// a value beyond float32 take the float64 path for the chunk
// (solve_systems_double; least squares for singular systems). Returns how
// many systems needed least squares; *fell_back: that the chunk took the
// float64 path.
static int solve_systems(double* gram, const double* rhs, double* x, int k,
                         int n, int p, bool one_at_a_time, int refine,
                         cudaStream_t stream, cublasHandle_t blas,
                         cusolverDnHandle_t solver,
                         harmony_comm::Comm* comm = nullptr,
                         bool* fell_back = nullptr) {
    if (fell_back) *fell_back = refine < 0;
    if (k == 0) return 0;
    if (refine < 0)
        return solve_systems_double(gram, rhs, x, k, n, p, one_at_a_time,
                                    stream, blas, solver, comm);
    const float onef = 1.0f, zerof = 0.0f;
    const double one = 1.0, minus_one = -1.0;
    const auto lower = CUBLAS_FILL_MODE_LOWER;
    const size_t gram_stride = (size_t)n * n, rhs_stride = (size_t)n * p;
    cublas_check_status(cublasSetStream(blas, stream), "cublasSetStream");
    cusolver_check(cusolverDnSetStream(solver, stream), "cusolverDnSetStream");
    // Plain float32 for the float32 solves (CuPy may have set TF32 on the
    // handle), and the residual GEMM from an aligned copy of x, so one GPU and
    // several (x at a rank's offset) get the same cuBLAS kernels.
    struct MathMode {
        cublasHandle_t h;
        cublasMath_t m;
        ~MathMode() {
            cublasSetMathMode(h, m);
        }
    } saved{blas, CUBLAS_DEFAULT_MATH};
    cublas_check_status(cublasGetMathMode(blas, &saved.m), "cublasGetMathMode");
    cublas_check_status(cublasSetMathMode(blas, CUBLAS_DEFAULT_MATH),
                        "cublasSetMathMode");
    std::vector<int> failed(k + 1);  // info of the k systems, then `bad`
    {
        RmmScratchPool pool;
        float *As, *ys, *work = nullptr, *Bt = nullptr;
        float** ptrs = nullptr;
        double *D, *res, *xs;
        int *info, *solve_info = nullptr, lwork = 0;
        if (one_at_a_time) {
            cusolver_check(cusolverDnSpotrf_bufferSize(solver, lower, n,
                                                       nullptr, n, &lwork),
                           "cusolverDnSpotrf_bufferSize");
            pool.alloc_all(As, k * gram_stride, ys, k * rhs_stride, D,
                           (size_t)k * n, res, k * rhs_stride, xs,
                           k * rhs_stride, info, (size_t)k + 1, work, lwork, Bt,
                           rhs_stride, solve_info, 1);
        } else {
            pool.alloc_all(As, k * gram_stride, ys, k * rhs_stride, D,
                           (size_t)k * n, res, k * rhs_stride, xs,
                           k * rhs_stride, info, (size_t)k + 1, ptrs,
                           2 * (size_t)k);
            float_pointers_kernel<<<(k + BLOCK_DIM_1D - 1) / BLOCK_DIM_1D,
                                    BLOCK_DIM_1D, 0, stream>>>(
                As, ys, k, gram_stride, rhs_stride, ptrs);
            CUDA_CHECK_LAST_ERROR(float_pointers_kernel);
        }
        int* bad = info + k;
        cuda_check(cudaMemsetAsync(info, 0, (k + 1) * sizeof(int), stream),
                   "solve info memset");
        equilibrate_kernel<<<strided_grid((long long)k * gram_stride,
                                          BLOCK_DIM_1D),
                             BLOCK_DIM_1D, 0, stream>>>(gram, k, n, D, As);
        CUDA_CHECK_LAST_ERROR(equilibrate_kernel);
        auto scale = [&](auto* in, bool add, auto* out) {
            scale_rows_kernel<<<strided_grid((long long)k * rhs_stride,
                                             BLOCK_DIM_1D),
                                BLOCK_DIM_1D, 0, stream>>>(in, D, k, n, p, add,
                                                           out, bad);
            CUDA_CHECK_LAST_ERROR(scale_rows_kernel);
        };
        // ys = As^-1 ys in place, by the factors of As (factored on the first
        // call).
        auto solve = [&](bool factor) {
            if (one_at_a_time) {
                for (int i = 0; i < k; ++i) {
                    float *A_i = As + i * gram_stride,
                          *y_i = ys + i * rhs_stride;
                    if (factor)
                        cusolver_check(
                            cusolverDnSpotrf(solver, lower, n, A_i, n, work,
                                             lwork, info + i),
                            "cusolverDnSpotrf");
                    cublas_check_status(
                        cublasSgeam(blas, CUBLAS_OP_T, CUBLAS_OP_N, n, p, &onef,
                                    y_i, p, &zerof, Bt, n, Bt, n),
                        "cublasSgeam");
                    cusolver_check(cusolverDnSpotrs(solver, lower, n, p, A_i, n,
                                                    Bt, n, solve_info),
                                   "cusolverDnSpotrs");
                    cublas_check_status(
                        cublasSgeam(blas, CUBLAS_OP_T, CUBLAS_OP_N, p, n, &onef,
                                    Bt, n, &zerof, y_i, p, y_i, p),
                        "cublasSgeam");
                }
                return;
            }
            if (factor)
                cusolver_check(
                    cusolverDnSpotrfBatched(solver, lower, n, ptrs, n, info, k),
                    "cusolverDnSpotrfBatched");
            for (auto op : {CUBLAS_OP_T, CUBLAS_OP_N})
                cublas_check_status(
                    cublasStrsmBatched(blas, CUBLAS_SIDE_RIGHT, lower, op,
                                       CUBLAS_DIAG_NON_UNIT, p, n, &onef, ptrs,
                                       n, ptrs + k, p, k),
                    "cublasStrsmBatched");
        };
        scale(rhs, false, ys);
        solve(true);
        scale(ys, false, xs);
        for (int step = 0; step < refine; ++step) {
            // res_i^T = rhs_i^T - xs_i^T gram_i (p x n column-major; gram_i
            // is symmetric)
            cuda_check(
                cudaMemcpyAsync(res, rhs, k * rhs_stride * sizeof(double),
                                cudaMemcpyDeviceToDevice, stream),
                "refine copy");
            cublas_check_status(
                cublas_gemm_strided_batched<double>(
                    blas, CUBLAS_OP_N, CUBLAS_OP_N, p, n, n, &minus_one, xs, p,
                    (long long)rhs_stride, gram, n, (long long)gram_stride,
                    &one, res, p, (long long)rhs_stride, k),
                "cublasDgemmStridedBatched");
            scale(res, false, ys);
            solve(false);
            scale(ys, true, xs);
        }
        cuda_check(cudaMemcpyAsync(x, xs, k * rhs_stride * sizeof(double),
                                   cudaMemcpyDeviceToDevice, stream),
                   "solve copy");
        harmony_comm::synchronize(comm, stream, "solve");
        cuda_check(cudaMemcpyAsync(failed.data(), info, (k + 1) * sizeof(int),
                                   cudaMemcpyDeviceToHost, stream),
                   "solve info");
        cuda_check(cudaStreamSynchronize(stream), "solve");
    }
    if (std::all_of(failed.begin(), failed.end(), [](int f) { return !f; }))
        return 0;
    if (fell_back) *fell_back = true;
    return solve_systems_double(gram, rhs, x, k, n, p, one_at_a_time, stream,
                                blas, solver, comm);
}

// acc (J x k int64) = the clustering's counts of R's columns (row stride ldr)
// per joint category, exact over the n_total cells on all GPUs of comm
// (segment_counts_kernel), on GPU `to` (-1: every one); the others take part
// in the exchange.
template <typename T>
static void joint_counts(const T* R, int ldr, int k, const int* seg_start,
                         const int* seg_group, int n_seg, int J,
                         long long n_total, long long* acc,
                         harmony_comm::Comm* comm, int rank, int to,
                         cudaStream_t stream) {
    FixedScales<T> scales(n_total, 0, k);
    size_t n = (size_t)J * k;
    cuda_check(cudaMemsetAsync(acc, 0, n * sizeof(long long), stream),
               "joint counts memset");
    segment_counts_kernel<T>
        <<<strided_grid((long long)n_seg * k, BLOCK_DIM_1D), BLOCK_DIM_1D, 0,
           stream>>>(R, ldr, k, seg_start, seg_group, n_seg, scales.count,
                     reinterpret_cast<unsigned long long*>(acc));
    CUDA_CHECK_LAST_ERROR(segment_counts_kernel);
    harmony_comm::allgather_groups(comm, rank, acc, k, stream, to);
}

// The multi-key correction of cells sorted by joint category (X, R: n x K),
// in chunks of `chunk` clusters: per chunk the exact joint counts (int64) and
// the exact per-level sums of R^T X (one pass per 8 keys, every level with
// its own bound), the systems (kernels_correction_multi.cuh) and their
// solutions, then the corrections into Z (one pass per 8 keys, each segment's
// coefficients summed over the keys). Several GPUs (comm) take rounds of one
// chunk each: all take part in the sums of every chunk, the counts gathered
// to its GPU and the level sums (levels span the GPUs) summed on every one;
// its GPU assembles and solves the chunk as one GPU would, then they exchange
// the solutions. seg_levels (n_covariates x n_seg): each segment's level
// within its key; level_offsets (n_covariates + 1): the first level of each
// key. Returns the systems solved here and how many of them by least squares.
template <typename T>
static std::pair<int, int> correction_multi_impl(
    const T* X, const T* R, const T* lambda_kb, const int* joint_cats,
    const int* seg_levels, const std::vector<int>& level_offsets, int n_pcs,
    int n_clusters, int J, const int* seg_start, const int* seg_group,
    int n_seg, const short* bounds, T* Z, bool normalize, int chunk, T suppress,
    long long n_total, int refine, cudaStream_t stream, cublasHandle_t blas,
    cusolverDnHandle_t solver, harmony_comm::Comm* comm, int rank) {
    const int C = (int)level_offsets.size() - 1, n_batches = level_offsets[C];
    const int nb1 = n_batches + 1, n_first = level_offsets[1];
    const int P = comm ? (int)comm->ranks.size() : 1;
    const size_t system = (size_t)nb1 * nb1, coefs = (size_t)nb1 * n_pcs;
    const size_t sums =
        (size_t)n_batches * RHS_LIMBS<T> * n_pcs;  // per cluster
    std::pair<int, int> solved{0, 0};
    for (int first = 0; first < n_clusters; first += P * chunk) {
        // GPU q's chunk of the round: clusters first + cut(q) .. cut(q + 1)
        int m = std::min(n_clusters - first, P * chunk);
        auto cut = [&](int q) { return std::min(m, q * chunk); };
        int mine = cut(rank + 1) - cut(rank);
        RmmScratchPool round;  // the round's solutions, GPU q's at cut(q)
        double* W = round.alloc<double>(m * coefs);
        {
            RmmScratchPool systems;  // this GPU's
            double *gram = nullptr, *rhs = nullptr;
            for (int q = 0; q < P && cut(q) < m; ++q) {
                int c0 = first + cut(q), k = cut(q + 1) - cut(q);
                bool own = q == rank;  // others' chunks: only their sums
                RmmScratchPool pool;
                long long *counts, *acc;
                pool.alloc_all(counts, (size_t)J * k, acc, sums * k);
                if (own) {
                    systems.alloc_all(gram, k * system, rhs, k * coefs);
                    cuda_check(
                        cudaMemsetAsync(gram, 0, k * system * sizeof(double),
                                        stream),
                        "gram memset");
                }
                joint_counts<T>(R + c0, n_clusters, k, seg_start, seg_group,
                                n_seg, J, n_total, counts, comm, rank, q,
                                stream);
                // the exact sums of every key's levels in one pass, summed
                // over the GPUs (levels span them): rows 1 .. of the
                // right-hand sides
                cuda_check(cudaMemsetAsync(acc, 0, sums * k * sizeof(long long),
                                           stream),
                           "level sums memset");
                for (int c = 0; c < C; c += harmony_segments::kMaxGroupings) {
                    harmony_segments::Groupings keys{};
                    keys.n = std::min(C - c, harmony_segments::kMaxGroupings);
                    for (int j = 0; j < keys.n; ++j) {
                        keys.seg_group[j] =
                            seg_levels + (size_t)(c + j) * n_seg;
                        keys.offset[j] = level_offsets[c + j];
                    }
                    cuda_check(
                        harmony_segments::segment_rtz(
                            X, R + c0, n_pcs, k, n_clusters, seg_start, keys,
                            n_seg, bounds, acc, RHS_LIMBS<T>, stream),
                        "level sums");
                }
                harmony_comm::allreduce(comm, rank, acc, sums * k, stream);
                if (!own) continue;
                cuda_check(harmony_segments::finalize_rtz(
                               acc, n_batches, n_pcs, k, RHS_LIMBS<T>, bounds,
                               rhs + n_pcs, n_pcs, nb1 * n_pcs, stream),
                           "level sums");
                count_blocks_kernel<<<strided_grid((long long)J * k,
                                                   BLOCK_DIM_1D),
                                      BLOCK_DIM_1D, 0, stream>>>(
                    reinterpret_cast<const unsigned long long*>(counts),
                    joint_cats, C, J, k, nb1,
                    reinterpret_cast<unsigned long long*>(gram));
                CUDA_CHECK_LAST_ERROR(count_blocks_kernel);
                FixedScales<T> scales(n_total, 0, k);  // as joint_counts
                Penalties<T> lambda{lambda_kb + c0, n_clusters, suppress};
                finish_gram_kernel<T><<<k, MAX_BLOCK_DIM, 0, stream>>>(
                    lambda, scales.count_inv, n_first, n_batches, k, gram);
                CUDA_CHECK_LAST_ERROR(finish_gram_kernel);
                finish_rhs_kernel<T>
                    <<<strided_grid((long long)k * n_pcs, BLOCK_DIM_1D),
                       BLOCK_DIM_1D, 0, stream>>>(lambda, n_first, n_batches,
                                                  n_pcs, k, rhs);
                CUDA_CHECK_LAST_ERROR(finish_rhs_kernel);
            }
            solved.second +=
                solve_systems(gram, rhs, W + cut(rank) * coefs, mine, nb1,
                              n_pcs, solve_one_at_a_time(mine, nb1), refine,
                              stream, blas, solver, comm);
            solved.first += mine;
        }
        if (P > 1)
            harmony_comm::allgather(
                comm, rank, reinterpret_cast<long long*>(W),
                [&](int q) {
                    return std::pair{cut(q) * coefs, cut(q + 1) * coefs};
                },
                stream);
        // Z = X - sum_k R_ik sum_c W_k[level_c(i)] (normalized on request):
        // one pass, each segment's coefficients summed over the keys. Each
        // chunk continues the sums of the earlier ones in Z and the last one
        // subtracts them from X, so no second buffer is needed.
        for (int c0 = first; c0 < first + m; c0 += chunk) {
            int k = std::min(chunk, n_clusters - c0);
            bool last = c0 + k == n_clusters;
            RmmScratchPool pool;
            T* table = pool.alloc<T>((size_t)n_batches * k * n_pcs);
            level_tables_kernel<T>
                <<<strided_grid((long long)n_batches * k * n_pcs, BLOCK_DIM_1D),
                   BLOCK_DIM_1D, 0, stream>>>(W + (c0 - first) * coefs, k, nb1,
                                              n_pcs, table);
            CUDA_CHECK_LAST_ERROR(level_tables_kernel);
            for (int c = 0; c < C; c += harmony_apply::kMaxTables) {
                harmony_apply::Tables<T> keys{};
                keys.n = std::min(C - c, harmony_apply::kMaxTables);
                for (int j = 0; j < keys.n; ++j) {
                    keys.w[j] =
                        table + (size_t)level_offsets[c + j] * k * n_pcs;
                    keys.seg_group[j] = seg_levels + (size_t)(c + j) * n_seg;
                }
                cuda_check(harmony_apply::apply_rows(
                               X, R + c0, n_clusters, keys, n_pcs,
                               (long long)k * n_pcs, n_pcs, k, seg_start, n_seg,
                               normalize && last, Z, stream, c0 > 0 || c > 0,
                               last && c + keys.n == C),
                           "multi-key apply");
            }
        }
    }
    return solved;
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
           gpu_array_c<const short, Device> bounds, gpu_array_c<T, Device> Z,
           int chunk, bool normalize,
           std::optional<gpu_array_c<uint16_t, Device>> R_bf16,
           std::uintptr_t stream, std::uintptr_t comm, int rank) {
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
            auto c = (harmony_comm::Comm*)comm;
            auto run = [&](auto* r) {
                // Cells sorted by batch: the right-hand sides S_b = R_b^T X_b
                // are exact sums over segments (bounds[b][d]: exponent of n_b
                // max|X_bd|), the coefficients the closed form of
                // ridge_coefficients, and Z = X - R_i W[batch] is applied row
                // by row and normalized on request (apply_rows.cuh), so results
                // do not depend on how the cells are split. Each pass
                // continues the sums of the earlier clusters in Z.
                for (int k0 = 0; k0 < n_clusters; k0 += chunk) {
                    int n_k = std::min(chunk, n_clusters - k0);
                    segment_sums(X.data(), r + k0, n_clusters, n_pcs, n_k,
                                 seg_start.data(), seg_group.data(), n_seg,
                                 n_batches, bounds.data(), rhs_acc, S, n_pcs,
                                 (int)s_cluster, c, rank, s);
                    // rhs_acc (summed into S) holds the intercepts
                    ridge_coefficients(O.data(), lambda_kb.data(), S, W,
                                       reinterpret_cast<double*>(rhs_acc),
                                       s_cluster, n_batches, n_clusters, n_pcs,
                                       k0, n_k, s);
                    cuda_check(harmony_apply::apply_rows(
                                   X.data(), r + k0, n_clusters, W, s_cluster,
                                   n_pcs, n_pcs, n_k, seg_start.data(),
                                   seg_group.data(), n_seg, normalize, Z.data(),
                                   s, k0 > 0, k0 + n_k == n_clusters),
                               "correction apply");
                }
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
        "chunk"_a, "normalize"_a = false, "R_bf16"_a.noconvert() = nb::none(),
        "stream"_a = 0, "comm"_a = 0, "rank"_a = 0);

    m.def(
        "correction_multi",
        [](gpu_array_c<const T, Device> X, gpu_array_c<const T, Device> R,
           gpu_array_c<const T, Device> lambda_kb,
           gpu_array_c<const int, Device> joint_cats,
           gpu_array_c<const int, Device> seg_levels,
           std::vector<int> level_offsets,
           gpu_array_c<const int, Device> seg_start,
           gpu_array_c<const int, Device> seg_group,
           gpu_array_c<const short, Device> bounds, gpu_array_c<T, Device> Z,
           bool normalize, int chunk, double suppress, long long n_total,
           int refine, std::uintptr_t stream, std::uintptr_t blas,
           std::uintptr_t solver, std::uintptr_t comm, int rank) {
            int C = (int)level_offsets.size() - 1;
            if (R.ndim() != 2 || R.shape(0) != X.shape(0) ||
                Z.size() != X.size() || lambda_kb.ndim() != 2 ||
                lambda_kb.shape(1) != R.shape(1) || C < 2 ||
                level_offsets[0] != 0 ||
                std::adjacent_find(level_offsets.begin(), level_offsets.end(),
                                   std::greater_equal<int>{}) !=
                    level_offsets.end() ||
                joint_cats.ndim() != 2 || joint_cats.shape(1) != (size_t)C ||
                seg_levels.ndim() != 2 || seg_levels.shape(0) != (size_t)C ||
                seg_group.size() + 1 != seg_start.size() ||
                seg_levels.shape(1) != seg_group.size() ||
                level_offsets[C] != (int)lambda_kb.shape(0) ||
                bound_groups(bounds, X.shape(1)) != level_offsets[C] ||
                chunk < 1)
                throw std::invalid_argument("correction_multi: bad shapes");
            return correction_multi_impl<T>(
                X.data(), R.data(), lambda_kb.data(), joint_cats.data(),
                seg_levels.data(), level_offsets, (int)X.shape(1),
                (int)R.shape(1), (int)joint_cats.shape(0), seg_start.data(),
                seg_group.data(), (int)seg_group.size(), bounds.data(),
                Z.data(), normalize, chunk, (T)suppress,
                n_total > 0 ? n_total : (long long)X.shape(0), refine,
                (cudaStream_t)stream, (cublasHandle_t)blas,
                (cusolverDnHandle_t)solver, (harmony_comm::Comm*)comm, rank);
        },
        nb::call_guard<nb::gil_scoped_release>(), "X"_a.noconvert(),
        nb::kw_only(), "R"_a.noconvert(), "lambda_kb"_a.noconvert(),
        "joint_cats"_a.noconvert(), "seg_levels"_a.noconvert(),
        "level_offsets"_a, "seg_start"_a.noconvert(), "seg_group"_a.noconvert(),
        "bounds"_a.noconvert(), "Z"_a.noconvert(), "normalize"_a = false,
        "chunk"_a, "suppress"_a, "n_total"_a = 0, "refine"_a, "stream"_a = 0,
        "blas"_a, "solver"_a, "comm"_a = 0, "rank"_a = 0);
}

template <typename Device>
void register_bindings(nb::module_& m) {
    register_correction_batched<float, Device>(m);
    register_correction_batched<double, Device>(m);

    m.def(
        "solve_spd",
        [](gpu_array_c<double, Device> gram,  // float64 path: overwritten
           gpu_array_c<const double, Device> rhs, gpu_array_c<double, Device> x,
           bool one_at_a_time, int refine, std::uintptr_t stream,
           std::uintptr_t blas, std::uintptr_t solver) {
            if (gram.ndim() != 3 || gram.shape(1) != gram.shape(2) ||
                rhs.ndim() != 3 || rhs.shape(0) != gram.shape(0) ||
                rhs.shape(1) != gram.shape(1) || x.size() != rhs.size())
                throw std::invalid_argument("solve_spd: bad shapes");
            bool fell_back;
            int n_least_squares = solve_systems(
                gram.data(), rhs.data(), x.data(), (int)gram.shape(0),
                (int)gram.shape(1), (int)rhs.shape(2), one_at_a_time, refine,
                (cudaStream_t)stream, (cublasHandle_t)blas,
                (cusolverDnHandle_t)solver, nullptr, &fell_back);
            return std::pair{n_least_squares, fell_back};
        },
        nb::call_guard<nb::gil_scoped_release>(), "gram"_a.noconvert(),
        "rhs"_a.noconvert(), "x"_a.noconvert(), nb::kw_only(),
        "one_at_a_time"_a = false, "refine"_a = 1, "stream"_a = 0, "blas"_a,
        "solver"_a);
}

NB_MODULE(_harmony_correction_batched_cuda, m) {
    REGISTER_GPU_BINDINGS(register_bindings, m);
}
