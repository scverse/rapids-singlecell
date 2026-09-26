#include <cuda_runtime.h>
#include "../nb_types.h"

#include "kernels_mv.cuh"

using namespace nb::literals;

template <typename T, typename IdxT>
static inline void launch_mean_var_major(const IdxT* indptr,
                                         const IdxT* indices, const T* data,
                                         double* means, double* vars, int major,
                                         int minor, cudaStream_t stream) {
    dim3 block(BLOCK_SIZE_MAJOR);
    dim3 grid(major);
    mean_var_major_kernel<T, IdxT><<<grid, block, 0, stream>>>(
        indptr, indices, data, means, vars, major, minor);
    CUDA_CHECK_LAST_ERROR(mean_var_major_kernel);
}

template <typename T, typename IdxT, typename Device>
void def_mean_var_major(nb::module_& m) {
    m.def(
        "mean_var_major",
        [](gpu_array_c<const IdxT, Device> indptr,
           gpu_array_c<const IdxT, Device> indices,
           gpu_array_c<const T, Device> data, gpu_array_c<double, Device> means,
           gpu_array_c<double, Device> vars, int major, int minor,
           std::uintptr_t stream) {
            launch_mean_var_major<T, IdxT>(
                indptr.data(), indices.data(), data.data(), means.data(),
                vars.data(), major, minor, (cudaStream_t)stream);
        },
        "indptr"_a, "indices"_a, "data"_a, "means"_a, "vars"_a, nb::kw_only(),
        "major"_a, "minor"_a, "stream"_a = 0);
}

// Order-agnostic minor-axis sums (one atomic per nonzero); no indptr needed.
template <typename T, typename IdxT, typename Device>
void def_mean_var_minor(nb::module_& m) {
    m.def(
        "mean_var_minor",
        [](gpu_array_c<const IdxT, Device> indices,
           gpu_array_c<const T, Device> data, gpu_array_c<double, Device> means,
           gpu_array_c<double, Device> vars, long long nnz,
           std::uintptr_t stream) {
            MeanVarOp<T> op{data.data(), means.data(), vars.data(), 0};
            minor_reduce_flat<IdxT>(indices.data(), op, nnz,
                                    (cudaStream_t)stream);
        },
        "indices"_a, "data"_a, "means"_a, "vars"_a, nb::kw_only(), "nnz"_a,
        "stream"_a = 0);
}

// Shared-memory tile sweep with atomic fallback. Returns whether unsorted rows
// were detected, so the caller can skip the attempt next time.
template <typename T, typename IdxT, typename Device>
void def_mean_var_minor_tiled(nb::module_& m) {
    m.def(
        "mean_var_minor_tiled",
        [](gpu_array_c<const IdxT, Device> indptr,
           gpu_array_c<const IdxT, Device> indices,
           gpu_array_c<const T, Device> data, gpu_array_c<double, Device> means,
           gpu_array_c<double, Device> vars, int major, int minor,
           long long nnz, bool assume_unsorted, std::uintptr_t stream) {
            MeanVarOp<T> op{data.data(), means.data(), vars.data(), 0};
            return minor_reduce<IdxT>(indptr.data(), indices.data(), op, major,
                                      minor, nnz, assume_unsorted,
                                      (cudaStream_t)stream);
        },
        "indptr"_a, "indices"_a, "data"_a, "means"_a, "vars"_a, nb::kw_only(),
        "major"_a, "minor"_a, "nnz"_a, "assume_unsorted"_a = false,
        "stream"_a = 0);
}

// Expose the planner so callers can ask which path a matrix would take:
// tile width, tile count, and whether unsorted rows would rescan or go atomic.
void def_tile_plan(nb::module_& m) {
    m.def(
        "tile_plan",
        [](long long nnz, int n_rows, int n_cols, size_t bytes_per_col) {
            require_arg(nnz >= 0 && n_rows > 0 && n_cols > 0 &&
                            bytes_per_col > 0 && bytes_per_col <= 4096,
                        "tile_plan: nnz must be >= 0, n_rows and n_cols "
                        "positive, and bytes_per_col in [1, 4096]");
            const TilePlan p = plan_tiles(nnz, n_rows, n_cols, bytes_per_col);
            nb::dict d;
            d["use_tiled"] = p.use_tiled;
            d["tile_size"] = p.tile_size;
            d["n_tiles"] = p.n_tiles;
            d["rows_per_block"] = p.rows_per_block;
            d["smem_bytes"] = p.smem_bytes;
            d["rescan_if_unsorted"] =
                p.use_tiled && p.n_tiles <= SWEEP_MAX_RESCAN_TILES;
            return d;
        },
        "nnz"_a, "n_rows"_a, "n_cols"_a,
        "bytes_per_col"_a = 2 * sizeof(double));
}

template <typename Device>
void register_bindings(nb::module_& m) {
    def_mean_var_major<float, int, Device>(m);
    def_mean_var_major<float, long long, Device>(m);
    def_mean_var_major<double, int, Device>(m);
    def_mean_var_major<double, long long, Device>(m);

    def_mean_var_minor<float, int, Device>(m);
    def_mean_var_minor<float, long long, Device>(m);
    def_mean_var_minor<double, int, Device>(m);
    def_mean_var_minor<double, long long, Device>(m);

    def_mean_var_minor_tiled<float, int, Device>(m);
    def_mean_var_minor_tiled<float, long long, Device>(m);
    def_mean_var_minor_tiled<double, int, Device>(m);
    def_mean_var_minor_tiled<double, long long, Device>(m);
}

NB_MODULE(_mean_var_cuda, m) {
    REGISTER_GPU_BINDINGS(register_bindings, m);
    def_tile_plan(m);
    register_scratch_allocator(m);
}
