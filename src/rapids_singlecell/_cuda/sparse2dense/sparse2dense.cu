#include <cuda_runtime.h>
#include "../nb_types.h"

#include "kernels_s2d.cuh"

using namespace nb::literals;

// Fully templated kernel launch - no runtime branches
template <typename T, typename IdxT, bool C_ORDER>
static inline void launch_sparse2dense(const IdxT* indptr, const IdxT* index,
                                       const T* data, T* out, long long major,
                                       long long minor, long long max_nnz,
                                       cudaStream_t stream) {
    if (max_nnz == 0 || major == 0) return;  // nothing to scatter

    constexpr int BLOCK_DIM = 32;
    dim3 block(BLOCK_DIM, BLOCK_DIM);
    unsigned grid_x = (unsigned)((major + BLOCK_DIM - 1) / BLOCK_DIM);
    // Cap grid Y via shared helper (y-axis has lower limit than x);
    // the kernel grid-strides internally to cover any remaining work.
    unsigned grid_y = strided_grid_y(max_nnz, BLOCK_DIM);
    dim3 grid(grid_x, grid_y);
    sparse2dense_kernel<T, IdxT, C_ORDER>
        <<<grid, block, 0, stream>>>(indptr, index, data, out, major, minor);
    CUDA_CHECK_LAST_ERROR(sparse2dense_kernel);
}

// Runtime dispatch wrapper - c_switch depends on both sparse format and output
// order
template <typename T, typename IdxT>
static inline void dispatch_sparse2dense(const IdxT* indptr, const IdxT* index,
                                         const T* data, T* out, long long major,
                                         long long minor, bool c_switch,
                                         long long max_nnz,
                                         cudaStream_t stream) {
    if (c_switch) {
        launch_sparse2dense<T, IdxT, true>(indptr, index, data, out, major,
                                           minor, max_nnz, stream);
    } else {
        launch_sparse2dense<T, IdxT, false>(indptr, index, data, out, major,
                                            minor, max_nnz, stream);
    }
}

// Helper to define sparse2dense for a given dtype, index type and output
// contiguity
template <typename T, typename IdxT, typename OutContig, typename Device>
void def_sparse2dense(nb::module_& m) {
    m.def(
        "sparse2dense",
        [](gpu_array_contig<const IdxT, Device, nb::c_contig> indptr,
           gpu_array_contig<const IdxT, Device, nb::c_contig> index,
           gpu_array_contig<const T, Device, nb::c_contig> data,
           gpu_array_contig<T, Device, OutContig> out, long long major,
           long long minor, bool c_switch, long long max_nnz,
           std::uintptr_t stream) {
            dispatch_sparse2dense<T, IdxT>(
                indptr.data(), index.data(), data.data(), out.data(), major,
                minor, c_switch, max_nnz, (cudaStream_t)stream);
        },
        "indptr"_a, "index"_a, "data"_a, nb::kw_only(), "out"_a, "major"_a,
        "minor"_a, "c_switch"_a, "max_nnz"_a, "stream"_a = 0);
}

template <typename Device>
void register_bindings(nb::module_& m) {
    // F-order output must come before C-order for proper dispatch
    def_sparse2dense<float, int, nb::f_contig, Device>(m);
    def_sparse2dense<float, int, nb::c_contig, Device>(m);
    def_sparse2dense<double, int, nb::f_contig, Device>(m);
    def_sparse2dense<double, int, nb::c_contig, Device>(m);
    def_sparse2dense<float, long long, nb::f_contig, Device>(m);
    def_sparse2dense<float, long long, nb::c_contig, Device>(m);
    def_sparse2dense<double, long long, nb::f_contig, Device>(m);
    def_sparse2dense<double, long long, nb::c_contig, Device>(m);
}

NB_MODULE(_sparse2dense_cuda, m) {
    REGISTER_GPU_BINDINGS(register_bindings, m);
}
