#include <cuda_runtime.h>
#include <nanobind/stl/optional.h>
#include <algorithm>
#include <cstdint>
#include <optional>
#include "../nb_types.h"

#include "kernels_gsea.cuh"
#include "kernels_gsea_rank.cuh"

using namespace nb::literals;

namespace {
constexpr int GSEA_BLOCK_SIZE = 128;

static inline void launch_gsea_rank(const float* values, size_t n_rows,
                                    size_t n_genes, int* order, int* ranks,
                                    cudaStream_t stream) {
    if (!n_rows || !n_genes) {
        return;
    }
    int device, shared_limit;
    cuda_check(cudaGetDevice(&device), "GSEA rank device");
    cuda_check(
        cudaDeviceGetAttribute(&shared_limit,
                               cudaDevAttrMaxSharedMemoryPerBlockOptin, device),
        "GSEA rank shared memory");
    const auto grid =
        static_cast<unsigned>(std::min(n_rows, size_t(max_grid_dim_x())));
    const size_t shared_bytes = sizeof(std::uint16_t) * n_genes;
    if (n_genes <= 65536 &&
        shared_bytes + sizeof(gsea_rank::Work) <= size_t(shared_limit)) {
        cuda_check(
            cudaFuncSetAttribute(gsea_rank::rank_kernel<std::uint16_t>,
                                 cudaFuncAttributeMaxDynamicSharedMemorySize,
                                 shared_limit - sizeof(gsea_rank::Work)),
            "GSEA rank shared limit");
        gsea_rank::rank_kernel<std::uint16_t>
            <<<grid, gsea_rank::THREADS, shared_bytes, stream>>>(
                values, n_rows, n_genes, order, ranks);
    } else {
        gsea_rank::rank_kernel<int><<<grid, gsea_rank::THREADS, 0, stream>>>(
            values, n_rows, n_genes, order, ranks);
    }
    CUDA_CHECK_LAST_ERROR(rank_kernel);
}

template <typename Position, int Capacity = 8>
static inline void launch_gsea_sparse_typed(gsea::GseaInput data,
                                            size_t capacity, unsigned grid,
                                            double* scores, double* null_sum,
                                            long long* same_count,
                                            long long* extreme_count,
                                            cudaStream_t stream) {
    // Select the smallest hit buffer that holds every set in this group.
    if (capacity > Capacity) {
        if constexpr (Capacity < 2048) {
            launch_gsea_sparse_typed<Position, Capacity * 2>(
                data, capacity, grid, scores, null_sum, same_count,
                extreme_count, stream);
        }
        return;
    }
    gsea::sparse_kernel<Capacity, Position>
        <<<grid, GSEA_BLOCK_SIZE, 0, stream>>>(data, scores, null_sum,
                                               same_count, extreme_count);
    CUDA_CHECK_LAST_ERROR(sparse_kernel);
}

static inline void launch_gsea_sparse(gsea::GseaInput data, size_t capacity,
                                      double* scores, double* null_sum,
                                      long long* same_count,
                                      long long* extreme_count,
                                      cudaStream_t stream) {
    if (!data.n_rows || !data.n_sources || !data.n_permutations) {
        return;
    }
    // Observed scores use one thread per set; null scores use whole warps.
    const size_t width = null_sum ? ((data.n_permutations + 31) / 32) * 32 : 1;
    nb_require(data.n_sources <= INT64_MAX / width &&
                   data.n_rows <= INT64_MAX / (data.n_sources * width),
               "GSEA task dimensions exceed addressable memory");
    const auto grid =
        strided_grid(data.n_rows * data.n_sources * width, GSEA_BLOCK_SIZE);
    if (data.n_genes <= 65536) {
        launch_gsea_sparse_typed<std::uint16_t>(data, capacity, grid, scores,
                                                null_sum, same_count,
                                                extreme_count, stream);
    } else {
        launch_gsea_sparse_typed<int>(data, capacity, grid, scores, null_sum,
                                      same_count, extreme_count, stream);
    }
}

static inline void launch_gsea_dense(const float* values, const int* order,
                                     const bool* membership,
                                     const int* permutations, size_t n_genes,
                                     size_t n_sets, size_t n_permutations,
                                     size_t n_tasks, double* scores,
                                     cudaStream_t stream) {
    if (!n_tasks) {
        return;
    }
    const auto grid = strided_grid(n_tasks, GSEA_BLOCK_SIZE);
    gsea::dense_kernel<<<grid, GSEA_BLOCK_SIZE, 0, stream>>>(
        values, order, membership, permutations, n_genes, n_sets,
        n_permutations, n_tasks, scores);
    CUDA_CHECK_LAST_ERROR(dense_kernel);
}

template <typename T, typename Device, int Dimensions>
using Array = nb::ndarray<T, Device, nb::ndim<Dimensions>, nb::c_contig>;

template <typename Device>
void register_bindings(nb::module_& m) {
    using Values = Array<const float, Device, 2>;
    using Indices = Array<const int, Device, 1>;
    using IndexMatrix = Array<const int, Device, 2>;
    using Scores = Array<double, Device, 2>;
    using Counts = Array<long long, Device, 2>;
    using Permutations = nb::ndarray<const int, Device, nb::ndim<2>>;
    m.def(
        "rank",
        [](Values values, Array<int, Device, 2> order,
           Array<int, Device, 2> ranks, std::uintptr_t stream) {
            const size_t n_rows = values.shape(0), n_genes = values.shape(1);
            nb_require(n_genes <= INT32_MAX, "too many features");
            nb_require(order.shape(0) == n_rows && order.shape(1) == n_genes &&
                           ranks.shape(0) == n_rows &&
                           ranks.shape(1) == n_genes,
                       "rank output shape mismatch");
            launch_gsea_rank(values.data(), n_rows, n_genes, order.data(),
                             ranks.data(),
                             reinterpret_cast<cudaStream_t>(stream));
        },
        "values"_a, "order"_a, "ranks"_a, "stream"_a = 0);
    m.def(
        "sparse",
        [](Values values, IndexMatrix ranks, Indices targets, Indices starts,
           Indices sizes, Indices sources, Indices positive_counts,
           Permutations permutations, Scores scores, size_t capacity,
           std::optional<Scores> null_sum, std::optional<Counts> same_count,
           std::optional<Counts> extreme_count, std::uintptr_t stream) {
            const gsea::GseaInput data{
                values.data(),
                ranks.data(),
                targets.data(),
                starts.data(),
                sizes.data(),
                sources.data(),
                positive_counts.data(),
                permutations.data(),
                values.shape(0),
                values.shape(1),
                starts.size(),
                permutations.shape(1),
                sources.size(),
                static_cast<size_t>(permutations.stride(0))};
            nb_require(
                data.n_genes > 0 && data.n_genes <= INT32_MAX && capacity > 0 &&
                    capacity <= 2048,
                "GSEA dimensions and capacity must fit the scoring kernel");
            nb_require(
                ranks.shape(0) == data.n_rows &&
                    ranks.shape(1) == data.n_genes &&
                    sizes.size() == data.n_sets &&
                    positive_counts.size() == data.n_rows &&
                    permutations.shape(0) == data.n_genes &&
                    data.permutation_stride >= data.n_permutations &&
                    (data.n_permutations <= 1 || permutations.stride(1) == 1) &&
                    scores.shape(0) == data.n_rows &&
                    scores.shape(1) == data.n_sets,
                "Incompatible GSEA input shapes or permutation strides");
            nb_require(
                null_sum.has_value() == same_count.has_value() &&
                    null_sum.has_value() == extreme_count.has_value(),
                "GSEA requires all three null-statistic outputs together");
            if (null_sum) {
                nb_require(null_sum->shape(0) == data.n_rows &&
                               null_sum->shape(1) == data.n_sets &&
                               same_count->shape(0) == data.n_rows &&
                               same_count->shape(1) == data.n_sets &&
                               extreme_count->shape(0) == data.n_rows &&
                               extreme_count->shape(1) == data.n_sets,
                           "Incompatible GSEA statistic output shapes");
            } else {
                nb_require(
                    data.n_permutations == 1,
                    "Observed GSEA scoring requires one identity permutation");
            }
            launch_gsea_sparse(data, capacity, scores.data(),
                               null_sum ? null_sum->data() : nullptr,
                               same_count ? same_count->data() : nullptr,
                               extreme_count ? extreme_count->data() : nullptr,
                               reinterpret_cast<cudaStream_t>(stream));
        },
        "values"_a, "ranks"_a, "cnct"_a, "starts"_a, "lens"_a, "source_ids"_a,
        "positive_counts"_a, "inverseT"_a, "es"_a, nb::kw_only(), "capacity"_a,
        "null_sum"_a = nb::none(), "same_count"_a = nb::none(),
        "extreme_count"_a = nb::none(), "stream"_a = 0);
    m.def(
        "dense",
        [](Values values, IndexMatrix order,
           Array<const bool, Device, 2> membership, IndexMatrix permutations,
           Array<double, Device, 3> scores, std::uintptr_t stream) {
            const size_t n_rows = values.shape(0), n_genes = values.shape(1);
            const size_t n_sets = membership.shape(0);
            const size_t n_permutations = permutations.shape(0);
            nb_require(n_genes > 0 && n_genes <= INT32_MAX &&
                           order.shape(0) == n_rows &&
                           order.shape(1) == n_genes &&
                           membership.shape(1) == n_genes &&
                           permutations.shape(1) == n_genes &&
                           scores.shape(0) == n_rows &&
                           scores.shape(1) == n_permutations &&
                           scores.shape(2) == n_sets,
                       "Incompatible GSEA dense score shapes");
            launch_gsea_dense(values.data(), order.data(), membership.data(),
                              permutations.data(), n_genes, n_sets,
                              n_permutations, scores.size(), scores.data(),
                              reinterpret_cast<cudaStream_t>(stream));
        },
        "values"_a, "order"_a, "membership"_a, "forward_permutations"_a, "es"_a,
        nb::kw_only(), "stream"_a = 0);
}
}  // namespace

NB_MODULE(_gsea_cuda, m) {
    REGISTER_GPU_BINDINGS(register_bindings, m);
}
