#include <cuda_runtime.h>
#include <cstdint>
#include "../nb_types.h"
#include "kernels_viper.cuh"

using namespace nb::literals;

namespace {
constexpr int BLOCK_SIZE = 128;

template <typename T, typename Device, int Dimensions>
using Array = nb::ndarray<T, Device, nb::ndim<Dimensions>, nb::c_contig>;

template <typename A, typename B>
bool same_shape(const A& a, const B& b) {
    return a.shape(0) == b.shape(0) && a.shape(1) == b.shape(1);
}

template <typename T, typename Device>
void register_typed_bindings(nb::module_& m) {
    using Values = Array<const T, Device, 2>;
    using Indices = Array<const long long, Device, 1>;
    using Doubles = Array<const double, Device, 2>;
    using Output = Array<double, Device, 2>;
    m.def(
        "average_ranks",
        [](Values mat, Values sorted_values, Output ranks,
           std::uintptr_t stream) {
            nb_require(same_shape(mat, sorted_values) && same_shape(mat, ranks),
                       "VIPER rank shape mismatch");
            if (!mat.size()) return;
            viper::average_ranks<<<strided_grid(mat.size(), BLOCK_SIZE),
                                   BLOCK_SIZE, 0,
                                   reinterpret_cast<cudaStream_t>(stream)>>>(
                mat.data(), sorted_values.data(), ranks.data(), mat.shape(1),
                mat.size());
            CUDA_CHECK_LAST_ERROR(average_ranks);
        },
        "mat"_a.noconvert(), "sorted_values"_a.noconvert(),
        "ranks"_a.noconvert(), "stream"_a = 0);
    m.def(
        "gather_overlap",
        [](Values mat, Indices targets, Indices starts, Indices sizes,
           Indices rows, Indices sources, Array<T, Device, 2> values,
           std::uintptr_t stream) {
            nb_require(starts.size() == sizes.size() + 1 &&
                           rows.size() == sources.size() &&
                           values.shape(0) == rows.size(),
                       "VIPER gather shape mismatch");
            if (!values.size()) return;
            viper::gather_overlap<<<strided_grid(values.size(), BLOCK_SIZE),
                                    BLOCK_SIZE, 0,
                                    reinterpret_cast<cudaStream_t>(stream)>>>(
                mat.data(), targets.data(), starts.data(), sizes.data(),
                rows.data(), sources.data(), values.data(), mat.shape(1),
                values.shape(1), values.size());
            CUDA_CHECK_LAST_ERROR(gather_overlap);
        },
        "mat"_a.noconvert(), "targets"_a.noconvert(), "starts"_a.noconvert(),
        "sizes"_a.noconvert(), "rows"_a.noconvert(), "sources"_a.noconvert(),
        "values"_a.noconvert(), "stream"_a = 0);
    m.def(
        "overlap_score",
        [](Values net, Indices targets, Indices starts, Indices sizes,
           Indices rows, Indices sources, Doubles zsigned, Doubles magnitude,
           Array<const bool, Device, 2> significant, Doubles counts,
           Output score, long long n_targets, std::uintptr_t stream) {
            const size_t nsrc = net.shape(1), ntask = rows.size();
            nb_require(
                starts.size() == nsrc + 1 && sizes.size() == nsrc &&
                    sources.size() == ntask && zsigned.shape(0) == ntask &&
                    same_shape(zsigned, magnitude) &&
                    significant.shape(1) == nsrc && counts.shape(0) == nsrc &&
                    counts.shape(1) == nsrc && score.shape(0) == ntask &&
                    score.shape(1) == nsrc && n_targets >= 0,
                "VIPER overlap shape mismatch");
            if (!score.size()) return;
            viper::overlap_score<<<strided_grid(score.size(), BLOCK_SIZE),
                                   BLOCK_SIZE, 0,
                                   reinterpret_cast<cudaStream_t>(stream)>>>(
                net.data(), targets.data(), starts.data(), sizes.data(),
                rows.data(), sources.data(), zsigned.data(), magnitude.data(),
                significant.data(), counts.data(), score.data(), nsrc,
                zsigned.shape(1), n_targets, score.size());
            CUDA_CHECK_LAST_ERROR(overlap_score);
        },
        "net"_a.noconvert(), "targets"_a.noconvert(), "starts"_a.noconvert(),
        "sizes"_a.noconvert(), "rows"_a.noconvert(), "sources"_a.noconvert(),
        "zsigned"_a.noconvert(), "magnitude"_a.noconvert(),
        "significant"_a.noconvert(), "counts"_a.noconvert(),
        "score"_a.noconvert(), "n_targets"_a, "stream"_a = 0);
}

template <typename Device>
void register_bindings(nb::module_& m) {
    using Indices = Array<const long long, Device, 1>;
    using Vector = Array<const double, Device, 1>;
    using Doubles = Array<const double, Device, 2>;
    using Output = Array<double, Device, 2>;
    register_typed_bindings<float, Device>(m);
    register_typed_bindings<double, Device>(m);
    m.def(
        "unsigned_quantiles",
        [](Doubles quantiles, Output magnitude, std::uintptr_t stream) {
            nb_require(same_shape(quantiles, magnitude),
                       "VIPER unsigned quantile shape mismatch");
            if (!quantiles.size()) return;
            viper::unsigned_quantiles<<<
                strided_grid(quantiles.size(), BLOCK_SIZE), BLOCK_SIZE, 0,
                reinterpret_cast<cudaStream_t>(stream)>>>(
                quantiles.data(), magnitude.data(), quantiles.size());
            CUDA_CHECK_LAST_ERROR(unsigned_quantiles);
        },
        "quantiles"_a.noconvert(), "magnitude"_a.noconvert(), "stream"_a = 0);
    m.def(
        "initial_score",
        [](Doubles zsigned, Doubles magnitude, Vector weights, Indices targets,
           Indices starts, Output sum1, Output sum2, std::uintptr_t stream) {
            nb_require(
                starts.size() > 0 && starts.size() == sum1.shape(1) + 1 &&
                    sum1.shape(0) == zsigned.shape(0) &&
                    same_shape(zsigned, magnitude) && same_shape(sum1, sum2) &&
                    targets.size() == weights.size(),
                "VIPER initial score shape mismatch");
            if (!sum1.size()) return;
            viper::initial_score<<<strided_grid(sum1.size(), BLOCK_SIZE),
                                   BLOCK_SIZE, 0,
                                   reinterpret_cast<cudaStream_t>(stream)>>>(
                zsigned.data(), magnitude.data(), weights.data(),
                targets.data(), starts.data(), sum1.data(), sum2.data(),
                zsigned.shape(1), sum1.shape(1), sum1.size());
            CUDA_CHECK_LAST_ERROR(initial_score);
        },
        "zsigned"_a.noconvert(), "magnitude"_a.noconvert(),
        "weights"_a.noconvert(), "targets"_a.noconvert(),
        "starts"_a.noconvert(), "sum1"_a.noconvert(), "sum2"_a.noconvert(),
        "stream"_a = 0);
    m.def(
        "transform_overlap",
        [](Doubles rank, Indices sizes, Indices rows, Indices sources,
           Doubles nes, Output tail, Output signed_value,
           std::uintptr_t stream) {
            nb_require(
                rows.size() == rank.shape(0) && sources.size() == rows.size() &&
                    sizes.size() == nes.shape(1) && same_shape(rank, tail) &&
                    same_shape(rank, signed_value),
                "VIPER overlap transform shape mismatch");
            if (!rank.size()) return;
            viper::transform_overlap<<<
                strided_grid(rank.size(), BLOCK_SIZE), BLOCK_SIZE, 0,
                reinterpret_cast<cudaStream_t>(stream)>>>(
                rank.data(), sizes.data(), rows.data(), sources.data(),
                nes.data(), tail.data(), signed_value.data(), nes.shape(1),
                rank.shape(1), rank.size());
            CUDA_CHECK_LAST_ERROR(transform_overlap);
        },
        "rank"_a.noconvert(), "sizes"_a.noconvert(), "rows"_a.noconvert(),
        "sources"_a.noconvert(), "nes"_a.noconvert(), "tail"_a.noconvert(),
        "signed_value"_a.noconvert(), "stream"_a = 0);
    m.def(
        "magnitude_overlap",
        [](Doubles tail, Vector maximum, Output magnitude,
           std::uintptr_t stream) {
            nb_require(
                same_shape(tail, magnitude) && maximum.size() == tail.shape(0),
                "VIPER overlap magnitude shape mismatch");
            if (!tail.size()) return;
            viper::magnitude_overlap<<<
                strided_grid(tail.size(), BLOCK_SIZE), BLOCK_SIZE, 0,
                reinterpret_cast<cudaStream_t>(stream)>>>(
                tail.data(), maximum.data(), magnitude.data(), tail.shape(1),
                tail.size());
            CUDA_CHECK_LAST_ERROR(magnitude_overlap);
        },
        "tail"_a.noconvert(), "maximum"_a.noconvert(),
        "magnitude"_a.noconvert(), "stream"_a = 0);
    m.def(
        "penalize",
        [](Array<const int, Device, 2> index, Indices offsets,
           Array<const int, Device, 1> losers,
           Array<const int, Device, 1> winners, Vector factors,
           Indices features, Output likelihood, std::uintptr_t stream) {
            nb_require(offsets.size() == likelihood.shape(0) + 1 &&
                           losers.size() == winners.size() &&
                           losers.size() == factors.size(),
                       "VIPER penalty shape mismatch");
            if (!likelihood.shape(0) || !features.size()) return;
            nb_require(features.size() <= INT64_MAX / likelihood.shape(0),
                       "VIPER penalty dimensions exceed addressable memory");
            const long long nwork = likelihood.shape(0) * features.size();
            viper::penalize<<<strided_grid(nwork, BLOCK_SIZE), BLOCK_SIZE, 0,
                              reinterpret_cast<cudaStream_t>(stream)>>>(
                index.data(), offsets.data(), losers.data(), winners.data(),
                factors.data(), features.data(), likelihood.data(),
                features.size(), index.shape(1), likelihood.shape(1), nwork);
            CUDA_CHECK_LAST_ERROR(penalize);
        },
        "index"_a.noconvert(), "offsets"_a.noconvert(), "losers"_a.noconvert(),
        "winners"_a.noconvert(), "factors"_a.noconvert(),
        "features"_a.noconvert(), "likelihood"_a.noconvert(), "stream"_a = 0);
    m.def(
        "rescore",
        [](Doubles zsigned, Doubles magnitude, Vector weights, Indices targets,
           Indices starts, Doubles likelihood,
           Array<const bool, Device, 2> affected, Doubles initial,
           Output result, std::uintptr_t stream) {
            nb_require(same_shape(zsigned, magnitude) &&
                           zsigned.shape(0) == initial.shape(0) &&
                           targets.size() == weights.size() &&
                           starts.size() == initial.shape(1) + 1 &&
                           likelihood.shape(0) == initial.shape(0) &&
                           likelihood.shape(1) == weights.size() &&
                           same_shape(affected, initial) &&
                           same_shape(initial, result),
                       "VIPER rescore shape mismatch");
            if (!result.size()) return;
            viper::
                rescore<<<strided_grid(result.size(), BLOCK_SIZE), BLOCK_SIZE,
                          0, reinterpret_cast<cudaStream_t>(stream)>>>(
                    zsigned.data(), magnitude.data(), weights.data(),
                    targets.data(), starts.data(), likelihood.data(),
                    affected.data(), initial.data(), result.data(),
                    zsigned.shape(1), initial.shape(1), likelihood.shape(1),
                    result.size());
            CUDA_CHECK_LAST_ERROR(rescore);
        },
        "zsigned"_a.noconvert(), "magnitude"_a.noconvert(),
        "weights"_a.noconvert(), "targets"_a.noconvert(),
        "starts"_a.noconvert(), "likelihood"_a.noconvert(),
        "affected"_a.noconvert(), "initial"_a.noconvert(),
        "result"_a.noconvert(), "stream"_a = 0);
}
}  // namespace

NB_MODULE(_viper_cuda, m) {
    REGISTER_GPU_BINDINGS(register_bindings, m);
}
