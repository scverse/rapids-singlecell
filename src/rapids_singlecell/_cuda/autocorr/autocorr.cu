#include <cuda_runtime.h>
#include "../nb_types.h"

#include <nanobind/stl/optional.h>
#include <optional>

#include "kernels_autocorr.cuh"

using namespace nb::literals;

constexpr int AUTOCORR_BLOCK = 256;

template <typename T, typename P>
static void moments_launch(const P* indptr, const int* indices, const T* data,
                           int n, double* out, cudaStream_t stream) {
    int device = 0, n_sm = 0;
    cuda_check(cudaGetDevice(&device), "cudaGetDevice");
    cuda_check(
        cudaDeviceGetAttribute(&n_sm, cudaDevAttrMultiProcessorCount, device),
        "cudaDeviceGetAttribute(MultiProcessorCount)");
    const int grid =
        std::min<long long>(strided_grid(n, AUTOCORR_BLOCK), 8LL * n_sm);
    autocorr_moments_kernel<<<grid, AUTOCORR_BLOCK, 0, stream>>>(
        indptr, indices, data, n, out);
    CUDA_CHECK_LAST_ERROR(autocorr_moments_kernel);
}

template <typename T, typename Device>
void def_sums(nb::module_& m) {
    m.def(
        "sums",
        [](gpu_array_c<const T, Device> x, gpu_array_c<const T, Device> z,
           gpu_array_c<const T, Device> r,
           std::optional<gpu_array_c<const unsigned int, Device>> keys,
           gpu_array_c<double, Device> out, bool geary, bool uniform,
           std::uintptr_t stream) {
            const int n = x.shape(0), n_genes = x.shape(1);
            const int n_perms = keys ? (int)keys->shape(0) : 0;
            nb_require(n_genes <= 32 && z.shape(0) == (size_t)n &&
                           z.shape(1) == (size_t)n_genes &&
                           r.shape(0) == (size_t)n,
                       "sums: x, z (n, <= 32 genes) and r (n) must match");
            nb_require(n_perms <= AUTOCORR_BATCH &&
                           (!keys || keys->shape(1) == AUTOCORR_ROUNDS),
                       "sums: keys must be (<= 16, 6)");
            nb_require(out.shape(0) == (size_t)(n_perms ? n_perms : 1) &&
                           out.shape(1) == 2 && out.shape(2) == (size_t)n_genes,
                       "sums: out must be (n_perms or 1, 2, n_genes)");
            int device = 0, n_sm = 0;
            cuda_check(cudaGetDevice(&device), "cudaGetDevice");
            cuda_check(cudaDeviceGetAttribute(
                           &n_sm, cudaDevAttrMultiProcessorCount, device),
                       "cudaDeviceGetAttribute(MultiProcessorCount)");
            const int grid = std::min<long long>(
                strided_grid((long long)n * 32, AUTOCORR_BLOCK), 8LL * n_sm);
            const unsigned int* k = keys ? keys->data() : nullptr;
            const auto kernel =
                geary ? (uniform ? autocorr_sums_kernel<T, true, true>
                                 : autocorr_sums_kernel<T, true, false>)
                      : (uniform ? autocorr_sums_kernel<T, false, true>
                                 : autocorr_sums_kernel<T, false, false>);
            kernel<<<grid, AUTOCORR_BLOCK, 0, (cudaStream_t)stream>>>(
                x.data(), z.data(), r.data(), n, n_genes, k, n_perms,
                out.data());
            CUDA_CHECK_LAST_ERROR(autocorr_sums_kernel);
        },
        "x"_a.noconvert(), "z"_a.noconvert(), "r"_a.noconvert(),
        "keys"_a.noconvert().none(), nb::kw_only(), "out"_a.noconvert(),
        "geary"_a, "uniform"_a = false, "stream"_a = 0);
}

template <typename T, typename P, typename Device>
void def_lag(nb::module_& m) {
    m.def(
        "lag",
        [](gpu_array_c<const P, Device> indptr,
           gpu_array_c<const int, Device> indices,
           gpu_array_c<const T, Device> weights, gpu_array_c<const T, Device> x,
           gpu_array_c<T, Device> z, std::uintptr_t stream) {
            const int n = x.shape(0), n_genes = x.shape(1);
            nb_require(n_genes <= 32 && indptr.shape(0) == (size_t)n + 1 &&
                           z.shape(0) == (size_t)n &&
                           z.shape(1) == (size_t)n_genes,
                       "lag: W (n, n) CSR, x and z (n, <= 32)");
            int device = 0, n_sm = 0;
            cuda_check(cudaGetDevice(&device), "cudaGetDevice");
            cuda_check(cudaDeviceGetAttribute(
                           &n_sm, cudaDevAttrMultiProcessorCount, device),
                       "cudaDeviceGetAttribute(MultiProcessorCount)");
            const int grid = std::min<long long>(
                strided_grid((long long)n * 32, AUTOCORR_BLOCK), 16LL * n_sm);
            autocorr_lag_kernel<<<grid, AUTOCORR_BLOCK, 0,
                                  (cudaStream_t)stream>>>(
                indptr.data(), indices.data(), weights.data(), x.data(), n,
                n_genes, z.data());
            CUDA_CHECK_LAST_ERROR(autocorr_lag_kernel);
        },
        "indptr"_a.noconvert(), "indices"_a.noconvert(),
        "weights"_a.noconvert(), "x"_a.noconvert(), nb::kw_only(),
        "z"_a.noconvert(), "stream"_a = 0);
}

template <typename T, typename Device>
void def_colstats(nb::module_& m) {
    m.def(
        "colstats",
        [](gpu_array_c<const T, Device> x, gpu_array_c<const double, Device> c,
           gpu_array_c<double, Device> out, std::uintptr_t stream) {
            const int n = x.shape(0), n_genes = x.shape(1);
            nb_require(n_genes <= 32 && c.shape(0) == (size_t)n &&
                           out.shape(0) == 4 && out.shape(1) == (size_t)n_genes,
                       "colstats: x (n, <= 32), c (n), out (4, n_genes)");
            int device = 0, n_sm = 0;
            cuda_check(cudaGetDevice(&device), "cudaGetDevice");
            cuda_check(cudaDeviceGetAttribute(
                           &n_sm, cudaDevAttrMultiProcessorCount, device),
                       "cudaDeviceGetAttribute(MultiProcessorCount)");
            const int grid = std::min<long long>(
                strided_grid((long long)n * 32, AUTOCORR_BLOCK), 8LL * n_sm);
            autocorr_colstats_kernel<<<grid, AUTOCORR_BLOCK, 0,
                                       (cudaStream_t)stream>>>(
                x.data(), c.data(), n, n_genes, out.data());
            CUDA_CHECK_LAST_ERROR(autocorr_colstats_kernel);
        },
        "x"_a.noconvert(), "c"_a.noconvert(), nb::kw_only(),
        "out"_a.noconvert(), "stream"_a = 0);
}

template <typename T, typename P, typename Device>
void def_moments(nb::module_& m) {
    m.def(
        "moments",
        [](gpu_array_c<const P, Device> indptr,
           gpu_array_c<const int, Device> indices,
           gpu_array_c<const T, Device> data, gpu_array_c<double, Device> out,
           std::uintptr_t stream) {
            const int n = indptr.shape(0) - 1;
            nb_require(out.shape(0) == 3, "moments: out must have 3 entries");
            moments_launch(indptr.data(), indices.data(), data.data(), n,
                           out.data(), (cudaStream_t)stream);
        },
        "indptr"_a.noconvert(), "indices"_a.noconvert(), "data"_a.noconvert(),
        nb::kw_only(), "out"_a.noconvert(), "stream"_a = 0);
}

template <typename Device>
void register_bindings(nb::module_& m) {
    def_moments<float, int, Device>(m);
    def_moments<double, int, Device>(m);
    def_moments<float, long long, Device>(m);
    def_moments<double, long long, Device>(m);
    def_colstats<float, Device>(m);
    def_colstats<double, Device>(m);
    def_sums<float, Device>(m);
    def_sums<double, Device>(m);
    def_lag<float, int, Device>(m);
    def_lag<double, int, Device>(m);
    def_lag<float, long long, Device>(m);
    def_lag<double, long long, Device>(m);
    m.def(
        "permutation",
        [](gpu_array_c<const unsigned int, Device> keys,
           gpu_array_c<int, Device> out, std::uintptr_t stream) {
            const int n = out.shape(0);
            nb_require(n >= 2 && keys.shape(0) == AUTOCORR_ROUNDS,
                       "permutation: n >= 2 and 6 keys");
            autocorr_permutation_kernel<<<(n + 255) / 256, 256, 0,
                                          (cudaStream_t)stream>>>(
                keys.data(), n, out.data());
            CUDA_CHECK_LAST_ERROR(autocorr_permutation_kernel);
        },
        "keys"_a, nb::kw_only(), "out"_a, "stream"_a = 0);
}

NB_MODULE(_autocorr_cuda, m) {
    m.attr("BATCH") = AUTOCORR_BATCH;
    m.attr("ROUNDS") = AUTOCORR_ROUNDS;
    REGISTER_GPU_BINDINGS(register_bindings, m);
}
