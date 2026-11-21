#include <cuda_runtime.h>
#include <algorithm>
#include <initializer_list>
#include "../nb_types.h"

#include "kernels_cooc.cuh"

template <typename Array>
static bool has_shape(const Array& array, std::initializer_list<size_t> shape) {
    return array.ndim() == shape.size() &&
           std::equal(shape.begin(), shape.end(), array.shape_ptr());
}

template <typename Device>
void register_bindings(nb::module_& m) {
    m.def("spatial_keys", [](gpu_array_c<const float, Device> spatial,
                             gpu_array_c<const int, Device> labels,
                             gpu_array_c<const float, Device> lower,
                             gpu_array_c<const float, Device> upper,
                             gpu_array_c<unsigned long long, Device> keys,
                             std::uintptr_t stream) {
        nb_require(spatial.ndim() == 2 && spatial.size() > 0 &&
                       has_shape(labels, {spatial.shape(0)}) &&
                       has_shape(keys, {spatial.shape(0)}) &&
                       has_shape(lower, {spatial.shape(1)}) &&
                       has_shape(upper, {spatial.shape(1)}),
                   "spatial_keys: expected nonempty (cells, dims) coordinates, "
                   "one label and key per cell, and bounds per dimension");
        int n = keys.size();
        cooc_spatial_keys<<<(keys.size() + 255) / 256, 256, 0,
                            reinterpret_cast<cudaStream_t>(stream)>>>(
            spatial.data(), labels.data(), lower.data(), upper.data(),
            keys.data(), n, spatial.shape(1));
        CUDA_CHECK_LAST_ERROR(cooc_spatial_keys);
    });
    m.def("build_tiles", [](gpu_array_c<const float, Device> spatial,
                            gpu_array_c<const int, Device> offsets,
                            gpu_array_c<const int, Device> tile_offsets,
                            gpu_array_c<unsigned char, Device> tiles,
                            gpu_array_c<float, Device> bounds,
                            std::uintptr_t stream) {
        nb_require(spatial.ndim() == 2 && tiles.ndim() == 2 &&
                       tiles.shape(0) > 0 &&
                       tiles.shape(1) == sizeof(CoocSpatialTile) &&
                       has_shape(bounds, {tiles.shape(0), spatial.shape(1), 2}),
                   "build_tiles: expected tiles (n_tiles, TILE_BYTES) "
                   "and bounds (n_tiles, dims, 2)");
        cooc_build_tiles<<<tiles.shape(0), 32, 0,
                           reinterpret_cast<cudaStream_t>(stream)>>>(
            spatial.data(), spatial.shape(1), offsets.data(),
            tile_offsets.data(), offsets.size() - 1,
            reinterpret_cast<CoocSpatialTile*>(tiles.data()),
            reinterpret_cast<float2*>(bounds.data()));
        CUDA_CHECK_LAST_ERROR(cooc_build_tiles);
    });
    m.def("count_tiles", [](gpu_array_c<const float, Device> spatial,
                            gpu_array_c<const unsigned char, Device> tiles,
                            gpu_array_c<const float, Device> bounds,
                            gpu_array_c<const float, Device> thresholds,
                            gpu_array_c<cooc_count_t, Device> counts, int rank,
                            int n_devices, std::uintptr_t stream) {
        const bool within = counts.ndim() == 2;
        nb_require(thresholds.ndim() == 1 && thresholds.size() > 0 &&
                       (within || counts.ndim() == 3) && counts.shape(0) > 0 &&
                       counts.shape(counts.ndim() - 1) == thresholds.size() &&
                       (within || counts.shape(0) == counts.shape(1)),
                   "count_tiles: expected nonempty thresholds and counts "
                   "shaped (k, bins) or (k, k, bins)");
        int n_tiles = tiles.shape(0), bins = thresholds.size();
        dim3 grid((n_tiles + n_devices - 1) / n_devices,
                  std::min(128, (n_tiles + 3) / 4));
        auto kernel =
            spatial.shape(1) == 2 ? cooc_count_tiles<2> : cooc_count_tiles<0>;
        kernel<<<grid, 128, 0, reinterpret_cast<cudaStream_t>(stream)>>>(
            spatial.data(), spatial.shape(1),
            reinterpret_cast<const CoocSpatialTile*>(tiles.data()),
            reinterpret_cast<const float2*>(bounds.data()), n_tiles,
            thresholds.data(), bins, counts.shape(0), within, counts.data(),
            rank, n_devices);
        CUDA_CHECK_LAST_ERROR(cooc_count_tiles);
    });
    m.def("normalize", [](gpu_array_c<const cooc_count_t, Device> counts,
                          gpu_array_c<const int64_t, Device> order,
                          gpu_array_c<float, Device> marginals,
                          gpu_array_c<float, Device> out,
                          std::uintptr_t stream) {
        nb_require(
            order.ndim() == 1 && order.size() > 0 && counts.ndim() == 3,
            "normalize: expected a nonempty order vector and 3-D counts");
        const size_t k = counts.shape(0), bins = order.size();
        nb_require(has_shape(counts, {k, k, bins}) &&
                       has_shape(out, {k, k, bins}) &&
                       has_shape(marginals, {bins, k}),
                   "normalize: expected counts and output (k, k, bins) "
                   "and marginals (bins, k)");
        cooc_normalize<<<bins, 32, 0, reinterpret_cast<cudaStream_t>(stream)>>>(
            counts.data(), order.data(), marginals.data(), out.data(), k, bins);
        CUDA_CHECK_LAST_ERROR(cooc_normalize);
    });
}

NB_MODULE(_cooc_cuda, m) {
    m.attr("TILE_BYTES") = sizeof(CoocSpatialTile);
    REGISTER_GPU_BINDINGS(register_bindings, m);
}
