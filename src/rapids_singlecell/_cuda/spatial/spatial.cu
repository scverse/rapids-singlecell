#include <cub/device/device_radix_sort.cuh>

#include "../nb_types.h"
#include "../rmm_scratch.h"
#include "kernels_delaunay.cuh"
#include "kernels_graph.cuh"
#include "kernels_spatial.cuh"

#include <cstdint>

using namespace nb::literals;

template <typename T, typename R, typename C, typename Device>
void register_distances(nb::module_& m) {
    m.def(
        "edge_distances",
        [](gpu_array_c<const T, Device> points,
           gpu_array_c<const R, Device> rows,
           gpu_array_c<const C, Device> columns,
           gpu_array_c<T, Device> distances, std::uintptr_t stream) {
            const long long n = rows.size();
            kd_edge_distances<<<(n + 256) / 256, 256, 0,
                                (cudaStream_t)stream>>>(
                points.data(), rows.data(), columns.data(), n, points.shape(1),
                distances.data());
            CUDA_CHECK_LAST_ERROR(kd_edge_distances);
        },
        "points"_a.noconvert(), "rows"_a.noconvert(), "columns"_a.noconvert(),
        "distances"_a.noconvert(), "stream"_a);
}

template <typename T, typename Device>
void register_tree(nb::module_& m) {
    using Ints = gpu_array_c<const int, Device>;
    using Offsets = gpu_array_c<const long long, Device>;
    m.def(
        "build_tree",
        [](gpu_array_c<const T, Device> points, Ints ranks, Ints codes,
           Offsets segments, int max_length, gpu_array_c<int, Device> index,
           gpu_array_c<T, Device> boxes, std::uintptr_t stream) {
            const int n = points.shape(0), dims = points.shape(1);
            const int bits = 64 - __builtin_clzll(n), blocks = (n + 255) / 256;
            const auto s = (cudaStream_t)stream;
            RmmScratchPool pool;
            cub::DoubleBuffer<unsigned long long> keys(
                pool.alloc<unsigned long long>(n),
                pool.alloc<unsigned long long>(n));
            cub::DoubleBuffer<int> ids(index.data(), pool.alloc<int>(n));
            size_t bytes = 0;
            cub::DeviceRadixSort::SortPairs(nullptr, bytes, keys, ids, n, 0,
                                            2 * bits);
            void* temp = pool.alloc<char>(bytes);
            const int levels = 31 - __builtin_clz(max_length);
            for (int level = -1; level < levels; ++level) {
                kd_keys<<<blocks, 256, 0, s>>>(
                    ranks.data(), codes.data(), segments.data(), n, dims, bits,
                    level, keys.Current(), ids.Current());
                cub::DeviceRadixSort::SortPairs(temp, bytes, keys, ids, n, 0,
                                                2 * bits, s);
            }
            if (ids.Current() != index.data())
                cudaMemcpyAsync(index.data(), ids.Current(), n * sizeof(int),
                                cudaMemcpyDeviceToDevice, s);
            for (int level = levels; level >= 0; --level)
                kd_boxes<<<blocks, 256, 0, s>>>(points.data(), index.data(),
                                                codes.data(), segments.data(),
                                                n, dims, level, boxes.data());
            CUDA_CHECK_LAST_ERROR(kd_boxes);
        },
        "points"_a.noconvert(), "ranks"_a.noconvert(),
        "codes"_a.noconvert().none(), "segments"_a.noconvert(), "max_length"_a,
        "index"_a.noconvert(), "boxes"_a.noconvert(), "stream"_a);
    // k > 0 finds k nearest neighbors; k == 0 counts radius neighbors into
    // offsets, or fills them if rows are given.
    m.def(
        "tree_search",
        [](gpu_array_c<const T, Device> points,
           gpu_array_c<const T, Device> tree, Ints index,
           gpu_array_c<const T, Device> boxes, Offsets segments, Ints codes,
           int k, double radius, gpu_array_c<long long, Device> offsets,
           gpu_array_c<long long, Device> rows,
           gpu_array_c<long long, Device> columns,
           gpu_array_c<T, Device> distances, std::uintptr_t stream) {
            const int n = points.shape(0);
            const auto kernel = k > 32        ? kd_search<T, 0>
                                : k > 16      ? kd_search<T, 32>
                                : k > 8       ? kd_search<T, 16>
                                : k           ? kd_search<T, 8>
                                : rows.data() ? kd_search<T, -1, true>
                                              : kd_search<T, -1>;
            kernel<<<(n + 127) / 128, 128, 0, (cudaStream_t)stream>>>(
                points.data(), tree.data(), index.data(), boxes.data(),
                segments.data(), codes.data(), n, points.shape(1), k, radius,
                offsets.data(), rows.data(), columns.data(), distances.data());
            CUDA_CHECK_LAST_ERROR(kd_search);
        },
        "points"_a.noconvert(), "tree"_a.noconvert(), "index"_a.noconvert(),
        "boxes"_a.noconvert(), "segments"_a.noconvert(),
        "codes"_a.noconvert().none(), "k"_a, "radius"_a,
        "offsets"_a.noconvert().none(),
        "rows"_a.noconvert().none() = nb::none(),
        "columns"_a.noconvert().none() = nb::none(),
        "distances"_a.noconvert().none() = nb::none(), "stream"_a);
    m.def(
        "library_percentile",
        [](gpu_array_c<const double, Device> index,
           gpu_array_c<const T, Device> values,
           gpu_array_c<const long long, Device> starts,
           gpu_array_c<const long long, Device> sizes,
           gpu_array_c<T, Device> out, std::uintptr_t stream) {
            const long long n = index.size();
            library_percentile<<<(n + 255) / 256, 256, 0,
                                 (cudaStream_t)stream>>>(
                index.data(), values.data(), starts.data(), sizes.data(), n,
                out.data());
            CUDA_CHECK_LAST_ERROR(library_percentile);
        },
        "index"_a.noconvert(), "values"_a.noconvert(), "starts"_a.noconvert(),
        "sizes"_a.noconvert(), "out"_a.noconvert(), "stream"_a);
    register_distances<T, int, int, Device>(m);
    register_distances<T, int, long long, Device>(m);
    register_distances<T, long long, int, Device>(m);
    register_distances<T, long long, long long, Device>(m);
}

template <typename R, typename C, typename I, typename T, typename Device>
void register_graph(nb::module_& m) {
    m.def(
        "assemble_graphs",
        [](gpu_array_c<const R, Device> rows, gpu_array_c<const C, Device> cols,
           gpu_array_c<const T, Device> values, float diag,
           gpu_array_c<I, Device> indptr, gpu_array_c<I, Device> indices,
           gpu_array_c<float, Device> adj, gpu_array_c<I, Device> dst_indices,
           gpu_array_c<T, Device> dst, std::uintptr_t stream) {
            const long long n_edges = rows.size(), n_obs = indptr.size() - 1;
            assemble_graphs<<<(n_edges + n_obs + 256) / 256, 256, 0,
                              (cudaStream_t)stream>>>(
                rows.data(), cols.data(), values.data(), n_edges, n_obs, diag,
                indptr.data(), indices.data(), adj.data(), dst_indices.data(),
                dst.size() ? dst.data() : nullptr);
            CUDA_CHECK_LAST_ERROR(assemble_graphs);
        },
        "rows"_a.noconvert(), "cols"_a.noconvert(), "values"_a.noconvert(),
        "diag"_a, "indptr"_a.noconvert(), "indices"_a.noconvert(),
        "adj"_a.noconvert(), "dst_indices"_a.noconvert(), "dst"_a.noconvert(),
        "stream"_a);
}

template <typename R, typename C, typename Device>
void register_graphs(nb::module_& m) {
    register_graph<R, C, int, float, Device>(m);
    register_graph<R, C, int, double, Device>(m);
    register_graph<R, C, long long, float, Device>(m);
    register_graph<R, C, long long, double, Device>(m);
}

template <typename Device>
void register_delaunay(nb::module_& m) {
    using Points = gpu_array_c<const double, Device>;
    using Ints = gpu_array_c<const int, Device>;
    using Out = gpu_array_c<int, Device>;
    const auto grid = [](long long n) { return (unsigned)((n + 127) / 128); };
    m.def(
        "delaunay_widen",
        [=](gpu_array_c<const float, Device> coords, int shift,
            gpu_array_c<double, Device> out, std::uintptr_t stream) {
            delaunay_widen<<<grid(out.size()), 128, 0, (cudaStream_t)stream>>>(
                coords.data(), out.size(), shift, out.data());
            CUDA_CHECK_LAST_ERROR(delaunay_widen);
        },
        "coords"_a.noconvert(), "shift"_a, "out"_a.noconvert(), "stream"_a);
    m.def(
        "delaunay_orientations",
        [=](Points points, Ints triangles, gpu_array_c<double, Device> out,
            std::uintptr_t stream) {
            const long long nt = out.size();
            delaunay_orientations<<<grid(nt), 128, 0, (cudaStream_t)stream>>>(
                points.data(), triangles.data(), nt, points.shape(0),
                out.data());
            CUDA_CHECK_LAST_ERROR(delaunay_orientations);
        },
        "points"_a.noconvert(), "triangles"_a.noconvert(), "out"_a.noconvert(),
        "stream"_a);
    m.def(
        "delaunay_validate",
        [=](Points points, Ints triangles, Ints opposite, Out state, Out seen,
            Out after, Out before, Out votes, std::uintptr_t stream) {
            const long long n = points.shape(0), nt = triangles.shape(0);
            const auto s = (cudaStream_t)stream;
            delaunay_validate_triangles<<<grid(nt), 128, 0, s>>>(
                points.data(), triangles.data(), opposite.data(), n, nt,
                state.data(), seen.data(), after.data(), before.data(),
                votes.data());
            delaunay_validate_boundary<<<grid(n), 128, 0, s>>>(
                points.data(), n, seen.data(), after.data(), before.data(),
                state.data());
            CUDA_CHECK_LAST_ERROR(delaunay_validate);
        },
        "points"_a.noconvert(), "triangles"_a.noconvert(),
        "opposite"_a.noconvert(), "state"_a.noconvert(), "seen"_a.noconvert(),
        "after"_a.noconvert(), "before"_a.noconvert(), "votes"_a.noconvert(),
        "stream"_a);
    m.def(
        "delaunay_fill_edges",
        [=](Ints triangles, Ints opposite, Out cursor, Out rows, Out cols,
            std::uintptr_t stream) {
            const long long nt = triangles.shape(0);
            delaunay_fill_edges<<<grid(nt), 128, 0, (cudaStream_t)stream>>>(
                triangles.data(), opposite.data(), nt, cursor.data(),
                rows.data(), cols.data());
            CUDA_CHECK_LAST_ERROR(delaunay_fill_edges);
        },
        "triangles"_a.noconvert(), "opposite"_a.noconvert(),
        "cursor"_a.noconvert(), "rows"_a.noconvert(), "cols"_a.noconvert(),
        "stream"_a);
}

template <typename Device>
void register_spatial(nb::module_& m) {
    register_tree<float, Device>(m);
    register_tree<double, Device>(m);
    register_graphs<int, int, Device>(m);
    register_graphs<int, long long, Device>(m);
    register_graphs<long long, int, Device>(m);
    register_graphs<long long, long long, Device>(m);
    register_delaunay<Device>(m);
}

NB_MODULE(_spatial_cuda, m) {
    REGISTER_GPU_BINDINGS(register_spatial, m);
}
