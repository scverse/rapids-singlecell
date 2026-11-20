#include <cuda_runtime.h>
#include "../nb_types.h"

#include <nanobind/stl/optional.h>
#include <optional>

#include "kernels_autocorr.cuh"

using namespace nb::literals;

constexpr int AUTOCORR_BLOCK = 256;

template <typename T, typename P>
static void moments_launch(const P* indptr, const int* indices, const T* data,
                           int n, double* row, double* col, double* out,
                           cudaStream_t stream) {
    const int n_sm = device_attribute(cudaDevAttrMultiProcessorCount);
    const int grid =
        std::min<long long>(strided_grid(n, AUTOCORR_BLOCK), 8LL * n_sm);
    autocorr_moments_kernel<<<grid, AUTOCORR_BLOCK, 0, stream>>>(
        indptr, indices, data, n, row, col, out);
    CUDA_CHECK_LAST_ERROR(autocorr_moments_kernel);
}

template <typename T, typename Device>
void def_sums(nb::module_& m) {
    m.def(
        "sums",
        [](gpu_array_c<const T, Device> x, gpu_array_c<const T, Device> z,
           gpu_array_c<const double, Device> r,
           gpu_array_c<const unsigned int, Device> keys,
           gpu_array_c<double, Device> out, bool geary, bool uniform,
           std::uintptr_t stream) {
            const int n = x.shape(0), n_genes = x.shape(1);
            const int n_perms = keys.shape(0);
            nb_require(n_genes <= 32 && z.shape(0) == (size_t)n &&
                           z.shape(1) == (size_t)n_genes &&
                           r.shape(0) == (size_t)n,
                       "sums: x, z (n, <= 32 genes) and r (n) must match");
            nb_require(keys.shape(1) == AUTOCORR_ROUNDS,
                       "sums: each permutation needs six round keys");
            nb_require(out.shape(0) == (size_t)n_perms &&
                           out.shape(1) == (size_t)(uniform ? 1 : 2) &&
                           out.shape(2) == (size_t)n_genes,
                       "sums: expected one output per score, term and gene");
            const int n_sm = device_attribute(cudaDevAttrMultiProcessorCount);
            const int grid = std::min<long long>(
                strided_grid((long long)n * 32, AUTOCORR_BLOCK), 8LL * n_sm);
            const auto kernel =
                geary ? (uniform ? autocorr_sums_kernel<T, true, true>
                                 : autocorr_sums_kernel<T, true, false>)
                      : (uniform ? autocorr_sums_kernel<T, false, true>
                                 : autocorr_sums_kernel<T, false, false>);
            const long long step = (long long)max_grid_dim_y() * AUTOCORR_BATCH;
            for (long long first = 0; first < n_perms; first += step) {
                dim3 blocks(grid,
                            strided_grid_y(n_perms - first, AUTOCORR_BATCH));
                kernel<<<blocks, AUTOCORR_BLOCK, 0, (cudaStream_t)stream>>>(
                    x.data(), z.data(), r.data(), n, n_genes, keys.data(),
                    n_perms, first, out.data());
                CUDA_CHECK_LAST_ERROR(autocorr_sums_kernel);
            }
        },
        "x"_a.noconvert(), "z"_a.noconvert(), "r"_a.noconvert(),
        "keys"_a.noconvert(), nb::kw_only(), "out"_a.noconvert(), "geary"_a,
        "uniform"_a = false, "stream"_a = 0);
}

template <typename T, typename P, typename Device>
void def_graph_ops(nb::module_& m) {
    m.def(
        "normalize",
        [](gpu_array_c<const P, Device> indptr,
           gpu_array_c<const T, Device> weights, gpu_array_c<T, Device> out,
           std::uintptr_t stream) {
            const int n = indptr.shape(0) - 1;
            nb_require(out.size() == weights.size(),
                       "normalize: expected one output per weight");
            autocorr_normalize_kernel<<<strided_grid(n, AUTOCORR_BLOCK),
                                        AUTOCORR_BLOCK, 0,
                                        (cudaStream_t)stream>>>(
                indptr.data(), weights.data(), n, out.data());
            CUDA_CHECK_LAST_ERROR(autocorr_normalize_kernel);
        },
        "indptr"_a.noconvert(), "weights"_a.noconvert(), nb::kw_only(),
        "out"_a.noconvert(), "stream"_a = 0);
    m.def(
        "observed",
        [](gpu_array_c<const P, Device> indptr,
           gpu_array_c<const int, Device> indices,
           gpu_array_c<const T, Device> weights, gpu_array_c<const T, Device> x,
           gpu_array_c<const double, Device> row,
           gpu_array_c<const double, Device> col,
           std::optional<gpu_array_c<T, Device>> lag,
           gpu_array_c<double, Device> stats, bool geary,
           std::uintptr_t stream) {
            const int n = x.shape(0), n_genes = x.shape(1);
            nb_require(n_genes <= 32 && indptr.size() == (size_t)n + 1 &&
                           indices.size() == weights.size() &&
                           row.size() == (size_t)n && col.size() == (size_t)n &&
                           (!lag || lag->size() == x.size()) &&
                           stats.shape(0) == 4 &&
                           stats.shape(1) == (size_t)n_genes,
                       "observed: expected CSR graph, row and column sums, and "
                       "up to 32 genes");
            const int n_sm = device_attribute(cudaDevAttrMultiProcessorCount);
            const int grid = std::min<long long>(
                strided_grid((long long)n * 32, AUTOCORR_BLOCK), 16LL * n_sm);
            const auto kernel =
                geary ? (lag ? autocorr_observed_kernel<T, P, true, true>
                             : autocorr_observed_kernel<T, P, true, false>)
                      : (lag ? autocorr_observed_kernel<T, P, false, true>
                             : autocorr_observed_kernel<T, P, false, false>);
            kernel<<<grid, AUTOCORR_BLOCK, 0, (cudaStream_t)stream>>>(
                indptr.data(), indices.data(), weights.data(), x.data(),
                row.data(), col.data(), n, n_genes, lag ? lag->data() : nullptr,
                stats.data());
            CUDA_CHECK_LAST_ERROR(autocorr_observed_kernel);
        },
        "indptr"_a.noconvert(), "indices"_a.noconvert(),
        "weights"_a.noconvert(), "x"_a.noconvert(), "row"_a.noconvert(),
        "col"_a.noconvert(), nb::kw_only(), "lag"_a.noconvert().none(),
        "stats"_a.noconvert(), "geary"_a, "stream"_a = 0);
}

template <typename T, typename P, typename Device>
void def_moments(nb::module_& m) {
    m.def(
        "moments",
        [](gpu_array_c<const P, Device> indptr,
           gpu_array_c<const int, Device> indices,
           gpu_array_c<const T, Device> data, gpu_array_c<double, Device> row,
           gpu_array_c<double, Device> col, gpu_array_c<double, Device> out,
           std::uintptr_t stream) {
            const int n = indptr.shape(0) - 1;
            nb_require(row.size() == (size_t)n && col.size() == (size_t)n &&
                           out.size() == 3 && data.size() == indices.size(),
                       "moments: expected CSR weights, n row and column sums, "
                       "and three moments");
            moments_launch(indptr.data(), indices.data(), data.data(), n,
                           row.data(), col.data(), out.data(),
                           (cudaStream_t)stream);
        },
        "indptr"_a.noconvert(), "indices"_a.noconvert(), "data"_a.noconvert(),
        nb::kw_only(), "row"_a.noconvert(), "col"_a.noconvert(),
        "out"_a.noconvert(), "stream"_a = 0);
}

template <typename Device>
void register_bindings(nb::module_& m) {
    def_moments<float, int, Device>(m);
    def_moments<double, int, Device>(m);
    def_moments<float, long long, Device>(m);
    def_moments<double, long long, Device>(m);
    def_sums<float, Device>(m);
    def_sums<double, Device>(m);
    def_graph_ops<float, int, Device>(m);
    def_graph_ops<double, int, Device>(m);
    def_graph_ops<float, long long, Device>(m);
    def_graph_ops<double, long long, Device>(m);
    m.def(
        "permutation",
        [](gpu_array_c<const unsigned int, Device> keys,
           gpu_array_c<int, Device> out, std::uintptr_t stream) {
            const int n = out.shape(0);
            nb_require(n >= 2 && keys.shape(0) == AUTOCORR_ROUNDS,
                       "permutation: n >= 2 and 6 keys");
            autocorr_permutation_kernel<<<strided_grid(n, 256), 256, 0,
                                          (cudaStream_t)stream>>>(
                keys.data(), n, out.data());
            CUDA_CHECK_LAST_ERROR(autocorr_permutation_kernel);
        },
        "keys"_a, nb::kw_only(), "out"_a, "stream"_a = 0);
}

NB_MODULE(_autocorr_cuda, m) {
    m.attr("ROUNDS") = AUTOCORR_ROUNDS;
    REGISTER_GPU_BINDINGS(register_bindings, m);
}
