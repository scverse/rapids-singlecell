#include <cuda_runtime.h>
#include "../nb_types.h"

#include "kernels_cooc.cuh"

template <typename Device>
void register_bindings(nb::module_& m) {
    m.def("spatial_keys", [](gpu_array_c<const float, Device> spatial,
                             gpu_array_c<const int, Device> labels,
                             gpu_array_c<const float, Device> lower,
                             gpu_array_c<const float, Device> upper,
                             gpu_array_c<unsigned long long, Device> keys,
                             std::uintptr_t stream) {
        int n = keys.size();
        cooc_spatial_keys<<<(keys.size() + 255) / 256, 256, 0,
                            reinterpret_cast<cudaStream_t>(stream)>>>(
            spatial.data(), labels.data(), lower.data(), upper.data(),
            keys.data(), n, spatial.shape(1));
        CUDA_CHECK_LAST_ERROR(cooc_spatial_keys);
    });
    m.def("build_tiles",
          [](gpu_array_c<const float, Device> spatial,
             gpu_array_c<const int, Device> offsets,
             gpu_array_c<const int, Device> tile_offsets,
             gpu_array_c<unsigned char, Device> tiles,
             gpu_array_c<float, Device> bounds, std::uintptr_t stream) {
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
        int n_tiles = tiles.shape(0), bins = thresholds.size();
        dim3 grid((n_tiles + n_devices - 1) / n_devices,
                  std::min(128, (n_tiles + 3) / 4));
        auto kernel =
            spatial.shape(1) == 2 ? cooc_count_tiles<2> : cooc_count_tiles<0>;
        kernel<<<grid, 128, 0, reinterpret_cast<cudaStream_t>(stream)>>>(
            spatial.data(), spatial.shape(1),
            reinterpret_cast<const CoocSpatialTile*>(tiles.data()),
            reinterpret_cast<const float2*>(bounds.data()), n_tiles,
            thresholds.data(), bins, counts.shape(0), counts.ndim() == 2,
            counts.data(), rank, n_devices);
        CUDA_CHECK_LAST_ERROR(cooc_count_tiles);
    });
    m.def("normalize",
          [](gpu_array_c<const cooc_count_t, Device> counts,
             gpu_array_c<const int64_t, Device> order,
             gpu_array_c<float, Device> marginals,
             gpu_array_c<float, Device> out, std::uintptr_t stream) {
              cooc_normalize<<<order.size(), 32, 0,
                               reinterpret_cast<cudaStream_t>(stream)>>>(
                  counts.data(), order.data(), marginals.data(), out.data(),
                  counts.shape(0), order.size());
              CUDA_CHECK_LAST_ERROR(cooc_normalize);
          });
}

NB_MODULE(_cooc_cuda, m) {
    m.attr("TILE_BYTES") = sizeof(CoocSpatialTile);
    REGISTER_GPU_BINDINGS(register_bindings, m);
}
