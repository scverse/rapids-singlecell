#include "../nb_types.h"
#include "kernels_spatial.cuh"
#include "kernels_graph.cuh"
#include "kernels_delaunay.cuh"

#include <cmath>
#include <cstdint>
#include <limits>
#include <type_traits>

using namespace nb::literals;

template <typename T, typename Device>
void check_tree(gpu_array_c<const T, Device> points,
                gpu_array_c<const T, Device> tree,
                gpu_array_c<const long long, Device> index, double radius = 0) {
    if (points.ndim() != 2 || !points.shape(0) || !points.shape(1) ||
        points.shape(1) > std::numeric_limits<int>::max() || tree.ndim() != 2 ||
        tree.shape(0) != points.shape(0) || tree.shape(1) != points.shape(1) ||
        index.ndim() != 1 || index.size() != points.shape(0) ||
        !std::isfinite(radius) || radius < 0)
        throw std::invalid_argument("Invalid spatial tree arrays or radius.");
}

template <typename T, typename Device>
void register_tree(nb::module_& m) {
    m.def(
        "knn",
        [](gpu_array_c<const T, Device> points,
           gpu_array_c<const T, Device> tree,
           gpu_array_c<const long long, Device> index, int k,
           gpu_array_c<long long, Device> columns,
           gpu_array_c<T, Device> distances, std::uintptr_t stream) {
            check_tree(points, tree, index);
            const long long n = points.shape(0);
            if (k < 1 || k >= n || columns.ndim() != 1 ||
                distances.ndim() != 1 || columns.size() != n * k ||
                distances.size() != columns.size())
                throw std::invalid_argument(
                    "Invalid spatial kNN output buffers or k.");
            spatial_tree_search<T, false, false>
                <<<(n + 127) / 128, 128, 0, (cudaStream_t)stream>>>(
                    points.data(), tree.data(), index.data(), n,
                    points.shape(1), k, 0, nullptr, nullptr, nullptr,
                    columns.data(), distances.data());
            CUDA_CHECK_LAST_ERROR(spatial_tree_search);
        },
        "points"_a.noconvert(), "tree"_a.noconvert(), "index"_a.noconvert(),
        nb::kw_only(), "k"_a, "columns"_a.noconvert(),
        "distances"_a.noconvert(), "stream"_a = 0);

    m.def(
        "radius_count",
        [](gpu_array_c<const T, Device> points,
           gpu_array_c<const T, Device> tree,
           gpu_array_c<const long long, Device> index, double radius,
           gpu_array_c<long long, Device> counts, std::uintptr_t stream) {
            check_tree(points, tree, index, radius);
            const long long n = points.shape(0);
            if (counts.ndim() != 1 || counts.size() != n)
                throw std::invalid_argument(
                    "Invalid spatial radius count buffer.");
            spatial_tree_search<T, true, false>
                <<<(n + 127) / 128, 128, 0, (cudaStream_t)stream>>>(
                    points.data(), tree.data(), index.data(), n,
                    points.shape(1), 0, radius, nullptr, counts.data(), nullptr,
                    nullptr, nullptr);
            CUDA_CHECK_LAST_ERROR(spatial_tree_search);
        },
        "points"_a.noconvert(), "tree"_a.noconvert(), "index"_a.noconvert(),
        nb::kw_only(), "radius"_a, "counts"_a.noconvert(), "stream"_a = 0);

    m.def(
        "radius_fill",
        [](gpu_array_c<const T, Device> points,
           gpu_array_c<const T, Device> tree,
           gpu_array_c<const long long, Device> index, double radius,
           gpu_array_c<const long long, Device> offsets,
           gpu_array_c<long long, Device> rows,
           gpu_array_c<long long, Device> columns,
           gpu_array_c<T, Device> distances, std::uintptr_t stream) {
            check_tree(points, tree, index, radius);
            const long long n = points.shape(0);
            if (offsets.ndim() != 1 || offsets.size() != n + 1 ||
                rows.ndim() != 1 || columns.ndim() != 1 ||
                distances.ndim() != 1 || rows.size() != columns.size() ||
                rows.size() != distances.size())
                throw std::invalid_argument(
                    "Invalid spatial radius output buffers.");
            spatial_tree_search<T, true, true>
                <<<(n + 127) / 128, 128, 0, (cudaStream_t)stream>>>(
                    points.data(), tree.data(), index.data(), n,
                    points.shape(1), 0, radius, offsets.data(), nullptr,
                    rows.data(), columns.data(), distances.data());
            CUDA_CHECK_LAST_ERROR(spatial_tree_search);
        },
        "points"_a.noconvert(), "tree"_a.noconvert(), "index"_a.noconvert(),
        nb::kw_only(), "radius"_a, "offsets"_a.noconvert(),
        "rows"_a.noconvert(), "columns"_a.noconvert(),
        "distances"_a.noconvert(), "stream"_a = 0);
}

template <typename T, typename R, typename C, typename Device>
void register_distances(nb::module_& m) {
    m.def(
        "edge_distances",
        [](gpu_array_c<const T, Device> points,
           gpu_array_c<const R, Device> rows,
           gpu_array_c<const C, Device> columns,
           gpu_array_c<T, Device> distances, std::uintptr_t stream) {
            if (points.ndim() != 2 || !points.shape(1) ||
                points.shape(1) > std::numeric_limits<int>::max() ||
                rows.ndim() != 1 || columns.ndim() != 1 ||
                distances.ndim() != 1 || (!points.shape(0) && rows.size()) ||
                rows.size() != columns.size() ||
                rows.size() != distances.size())
                throw std::invalid_argument(
                    "Invalid spatial edge-distance buffers.");
            const long long n_edges = rows.size();
            if (!n_edges) return;
            spatial_edge_distances<<<(n_edges + 255) / 256, 256, 0,
                                     (cudaStream_t)stream>>>(
                points.data(), rows.data(), columns.data(), distances.data(),
                n_edges, points.shape(1));
            CUDA_CHECK_LAST_ERROR(spatial_edge_distances);
        },
        "points"_a.noconvert(), "rows"_a.noconvert(), "columns"_a.noconvert(),
        nb::kw_only(), "distances"_a.noconvert(), "stream"_a = 0);
}

template <typename T, typename Device>
void register_spatial_type(nb::module_& m) {
    register_tree<T, Device>(m);
    register_distances<T, int, int, Device>(m);
    register_distances<T, int, long long, Device>(m);
    register_distances<T, long long, int, Device>(m);
    register_distances<T, long long, long long, Device>(m);
}

template <typename Device>
void register_spatial(nb::module_& m) {
    register_spatial_type<float, Device>(m);
    register_spatial_type<double, Device>(m);
}

template <typename R, typename C, typename I, typename T, typename Device>
void bind_graph(nb::module_& m) {
    m.def(
        "assemble_graphs",
        [](gpu_array_c<const R, Device> rows,
           gpu_array_c<const C, Device> columns,
           gpu_array_c<const T, Device> distances, bool set_diag,
           bool with_distances, gpu_array_c<I, Device> adj_indptr,
           gpu_array_c<I, Device> dst_indptr,
           gpu_array_c<I, Device> adj_columns,
           gpu_array_c<I, Device> dst_columns,
           gpu_array_c<float, Device> adj_data, gpu_array_c<T, Device> dst_data,
           std::uintptr_t stream) {
            if (rows.ndim() != 1 || columns.ndim() != 1 ||
                distances.ndim() != 1 || adj_indptr.ndim() != 1 ||
                dst_indptr.ndim() != 1 || adj_columns.ndim() != 1 ||
                dst_columns.ndim() != 1 || adj_data.ndim() != 1 ||
                dst_data.ndim() != 1 || adj_indptr.shape(0) == 0) {
                throw std::invalid_argument(
                    "assemble_graphs: expected 1D arrays and a nonempty "
                    "indptr");
            }
            const long long n_edges = rows.shape(0);
            const long long n_obs = adj_indptr.shape(0) - 1;
            const long long nnz = n_edges + n_obs;
            if (columns.shape(0) != n_edges || adj_columns.shape(0) != nnz ||
                adj_data.shape(0) != nnz ||
                (with_distances &&
                 (distances.shape(0) != n_edges ||
                  dst_indptr.shape(0) != n_obs + 1 ||
                  dst_columns.shape(0) != nnz || dst_data.shape(0) != nnz)) ||
                nnz > std::numeric_limits<I>::max()) {
                throw std::invalid_argument(
                    "assemble_graphs: incompatible array lengths or index "
                    "dtype");
            }
            const long long blocks =
                (n_edges + (n_edges < n_obs ? n_obs : 0) + 256) / 256;
            if (blocks > max_grid_dim_x()) {
                throw std::invalid_argument(
                    "assemble_graphs: too many edges for one launch");
            }
            const auto launch = [&](auto with_distances_tag) {
                constexpr bool WithDistances =
                    decltype(with_distances_tag)::value;
                assemble_graphs_kernel<R, C, I, T, WithDistances>
                    <<<blocks, 256, 0, (cudaStream_t)stream>>>(
                        rows.data(), columns.data(), distances.data(), n_edges,
                        n_obs, set_diag, adj_indptr.data(), dst_indptr.data(),
                        adj_columns.data(), dst_columns.data(), adj_data.data(),
                        dst_data.data());
            };
            if (with_distances)
                launch(std::true_type{});
            else
                launch(std::false_type{});
            CUDA_CHECK_LAST_ERROR(assemble_graphs_kernel);
        },
        "rows"_a.noconvert(), "columns"_a.noconvert(),
        "distances"_a.noconvert(), nb::kw_only(), "set_diag"_a,
        "with_distances"_a, "adj_indptr"_a.noconvert(),
        "dst_indptr"_a.noconvert(), "adj_columns"_a.noconvert(),
        "dst_columns"_a.noconvert(), "adj_data"_a.noconvert(),
        "dst_data"_a.noconvert(), "stream"_a = 0);
}

template <typename R, typename C, typename Device>
void bind_graph_outputs(nb::module_& m) {
    bind_graph<R, C, int, float, Device>(m);
    bind_graph<R, C, int, double, Device>(m);
    bind_graph<R, C, long long, float, Device>(m);
    bind_graph<R, C, long long, double, Device>(m);
}

template <typename Device>
void register_graph_bindings(nb::module_& m) {
    bind_graph_outputs<int, int, Device>(m);
    bind_graph_outputs<int, long long, Device>(m);
    bind_graph_outputs<long long, int, Device>(m);
    bind_graph_outputs<long long, long long, Device>(m);
}

template <typename Device>
static void register_delaunay_bindings(nb::module_& m) {
    using Points =
        nb::ndarray<const double, Device, nb::shape<-1, 2>, nb::c_contig>;
    using Input =
        nb::ndarray<const int, Device, nb::shape<-1, 3>, nb::c_contig>;
    using Output = nb::ndarray<int, Device, nb::ndim<1>, nb::c_contig>;
    m.def(
        "delaunay_normalize",
        [](gpu_array_c<const float, Device> points, int exponent,
           gpu_array_c<double, Device> output, std::uintptr_t stream) {
            if (points.ndim() != 2 || points.shape(1) != 2 ||
                output.ndim() != 2 || output.shape(0) != points.shape(0) ||
                output.shape(1) != 2 || exponent < -149 || exponent > 128)
                throw std::invalid_argument(
                    "Invalid Delaunay normalization arrays");
            const long long size = points.size();
            if (!size) return;
            delaunay_normalize<<<(size + 255) / 256, 256, 0,
                                 reinterpret_cast<cudaStream_t>(stream)>>>(
                points.data(), size, exponent, output.data());
            CUDA_CHECK_LAST_ERROR(delaunay_normalize);
        },
        "points"_a.noconvert(), nb::kw_only(), "exponent"_a,
        "output"_a.noconvert(), "stream"_a);
    m.def(
        "delaunay_orientations",
        [](Points points, int a, int b, gpu_array_c<double, Device> output,
           std::uintptr_t stream) {
            const auto n = points.shape(0);
            if (n == 0 || n >= (1u << 26) || a < 0 || b < 0 || a >= n ||
                b >= n || output.ndim() != 1 || output.size() != n)
                throw std::invalid_argument(
                    "Invalid Delaunay orientation arrays");
            delaunay_orientations<<<(n + 127) / 128, 128, 0,
                                    reinterpret_cast<cudaStream_t>(stream)>>>(
                points.data(), n, a, b, output.data());
            CUDA_CHECK_LAST_ERROR(delaunay_orientations);
        },
        "points"_a.noconvert(), "a"_a, "b"_a, nb::kw_only(),
        "output"_a.noconvert(), "stream"_a);
    m.def(
        "delaunay_triangle_orientations",
        [](Points points, Input triangles, gpu_array_c<double, Device> output,
           std::uintptr_t stream) {
            const auto n = points.shape(0), nt = triangles.shape(0);
            if (n >= (1u << 26) || nt > 2 * n || output.ndim() != 1 ||
                output.size() != nt)
                throw std::invalid_argument("Invalid Delaunay triangle arrays");
            if (!nt) return;
            delaunay_triangle_orientations<<<(nt + 127) / 128, 128, 0,
                                             reinterpret_cast<cudaStream_t>(
                                                 stream)>>>(
                points.data(), n, triangles.data(), nt, output.data());
            CUDA_CHECK_LAST_ERROR(delaunay_triangle_orientations);
        },
        "points"_a.noconvert(), "triangles"_a.noconvert(), nb::kw_only(),
        "output"_a.noconvert(), "stream"_a);
    m.def(
        "validate_delaunay",
        [](Points points, Input triangles, Input opposite, Output invalid,
           Output seen, Output next_boundary, Output prev_boundary,
           Output boundary_count, std::uintptr_t stream) {
            const auto n = points.shape(0), nt = triangles.shape(0);
            if (n < 3 || n >= (1u << 26) || nt < n - 2 || nt > 2 * n - 5 ||
                opposite.shape(0) != nt || invalid.size() != 1 ||
                boundary_count.size() != 2 || seen.size() != n ||
                next_boundary.size() != n || prev_boundary.size() != n)
                throw std::invalid_argument(
                    "Invalid Delaunay validation shapes");
            auto s = reinterpret_cast<cudaStream_t>(stream);
            validate_triangles<<<(nt + 127) / 128, 128, 0, s>>>(
                points.data(), triangles.data(), opposite.data(), n, nt,
                invalid.data(), seen.data(), next_boundary.data(),
                prev_boundary.data(), boundary_count.data());
            CUDA_CHECK_LAST_ERROR(validate_triangles);
            validate_boundary<<<(n + 127) / 128, 128, 0, s>>>(
                points.data(), n, nt, seen.data(), next_boundary.data(),
                prev_boundary.data(), boundary_count.data(), invalid.data());
            CUDA_CHECK_LAST_ERROR(validate_boundary);
            validate_boundary_winding<<<1, 1, 0, s>>>(boundary_count.data(),
                                                      invalid.data());
            CUDA_CHECK_LAST_ERROR(validate_boundary_winding);
        },
        "points"_a.noconvert(), "triangles"_a.noconvert(),
        "opposite"_a.noconvert(), "invalid"_a.noconvert(), "seen"_a.noconvert(),
        "next_boundary"_a.noconvert(), "prev_boundary"_a.noconvert(),
        "boundary_count"_a.noconvert(), "stream"_a);
    m.def(
        "exact_flip_votes",
        [](Points points, Input triangles, Input opposite, Output votes,
           std::uintptr_t stream) {
            const auto n = points.shape(0), nt = triangles.shape(0);
            if (n < 3 || n >= (1u << 26) || nt < n - 2 || nt > 2 * n - 5 ||
                opposite.shape(0) != nt || votes.size() != nt)
                throw std::invalid_argument(
                    "Invalid Delaunay flip-vote arrays");
            exact_flip_votes<<<(nt + 127) / 128, 128, 0,
                               reinterpret_cast<cudaStream_t>(stream)>>>(
                points.data(), triangles.data(), opposite.data(), nt,
                votes.data());
            CUDA_CHECK_LAST_ERROR(exact_flip_votes);
        },
        "points"_a.noconvert(), "triangles"_a.noconvert(),
        "opposite"_a.noconvert(), nb::kw_only(), "votes"_a.noconvert(),
        "stream"_a);
    m.def(
        "fill_delaunay_edges",
        [](Input triangles, Input opposite, Output cursor, Output rows,
           Output cols, std::uintptr_t stream) {
            const auto nt = triangles.shape(0), n = cursor.size();
            if (n < 3 || n >= (1u << 26) || nt < n - 2 || nt > 2 * n - 5 ||
                opposite.shape(0) != nt || rows.size() != 2 * (n + nt - 1) ||
                cols.size() != rows.size())
                throw std::invalid_argument("Invalid Delaunay edge shapes");
            fill_edges<<<(nt + 127) / 128, 128, 0,
                         reinterpret_cast<cudaStream_t>(stream)>>>(
                triangles.data(), opposite.data(), nt, cursor.data(),
                rows.data(), cols.data());
            CUDA_CHECK_LAST_ERROR(fill_edges);
        },
        "triangles"_a.noconvert(), "opposite"_a.noconvert(),
        "cursor"_a.noconvert(), "rows"_a.noconvert(), "cols"_a.noconvert(),
        "stream"_a);
}

NB_MODULE(_spatial_cuda, m) {
    REGISTER_GPU_BINDINGS(register_spatial, m);
    REGISTER_GPU_BINDINGS(register_graph_bindings, m);
    REGISTER_GPU_BINDINGS(register_delaunay_bindings, m);
}
