#include <cuda_runtime.h>
#include <nanobind/stl/optional.h>

#include <optional>

#include "../nb_types.h"

#include "kernels_aggr.cuh"

using namespace nb::literals;

constexpr int BLOCK_SIZE_SPARSE = 64;
constexpr int BLOCK_SIZE_DENSE = 256;

// Expand a runtime accumulator mask (1..7) into a compile-time MASK template
// argument. Each kernel instantiation emits only the requested atomicAdds, so
// inactive (null) output buffers are never written. Mask 0 means no output
// buffer was provided, which is always a caller bug.
#define AGGR_DISPATCH_MASK(active, LAUNCH)                                   \
    switch (active) {                                                        \
        case 1:                                                              \
            LAUNCH(1);                                                       \
            break;                                                           \
        case 2:                                                              \
            LAUNCH(2);                                                       \
            break;                                                           \
        case 3:                                                              \
            LAUNCH(3);                                                       \
            break;                                                           \
        case 4:                                                              \
            LAUNCH(4);                                                       \
            break;                                                           \
        case 5:                                                              \
            LAUNCH(5);                                                       \
            break;                                                           \
        case 6:                                                              \
            LAUNCH(6);                                                       \
            break;                                                           \
        case 7:                                                              \
            LAUNCH(7);                                                       \
            break;                                                           \
        default:                                                             \
            throw std::runtime_error(                                        \
                "aggr: at least one of out_sum/out_count/out_sqsum must be " \
                "provided");                                                 \
    }

template <typename T, typename IdxT, int MASK>
static inline void launch_sparse_aggr(bool is_csc, const IdxT* indptr,
                                      const IdxT* index, const T* data,
                                      double* out_sum, double* out_count,
                                      double* out_sqsum, const int* cats,
                                      const bool* mask, size_t n_cells,
                                      size_t n_genes, cudaStream_t stream) {
    dim3 block(BLOCK_SIZE_SPARSE);
    if (is_csc) {
        dim3 grid((unsigned)n_genes);
        csc_aggr_kernel<T, IdxT, MASK><<<grid, block, 0, stream>>>(
            indptr, index, data, out_sum, out_count, out_sqsum, cats, mask,
            n_cells, n_genes);
        CUDA_CHECK_LAST_ERROR(csc_aggr_kernel);
    } else {
        dim3 grid((unsigned)n_cells);
        csr_aggr_kernel<T, IdxT, MASK><<<grid, block, 0, stream>>>(
            indptr, index, data, out_sum, out_count, out_sqsum, cats, mask,
            n_cells, n_genes);
        CUDA_CHECK_LAST_ERROR(csr_aggr_kernel);
    }
}

template <typename T, int MASK>
static inline void launch_dense_aggr(bool is_fortran, const T* data,
                                     double* out_sum, double* out_count,
                                     double* out_sqsum, const int* cats,
                                     const bool* mask, size_t n_cells,
                                     size_t n_genes, cudaStream_t stream) {
    dim3 block(BLOCK_SIZE_DENSE);
    dim3 grid(strided_grid((long long)n_cells * n_genes, BLOCK_SIZE_DENSE));
    if (is_fortran) {
        dense_aggr_kernel_F<T, MASK><<<grid, block, 0, stream>>>(
            data, out_sum, out_count, out_sqsum, cats, mask, n_cells, n_genes);
        CUDA_CHECK_LAST_ERROR(dense_aggr_kernel_F);
    } else {
        dense_aggr_kernel_C<T, MASK><<<grid, block, 0, stream>>>(
            data, out_sum, out_count, out_sqsum, cats, mask, n_cells, n_genes);
        CUDA_CHECK_LAST_ERROR(dense_aggr_kernel_C);
    }
}

template <typename T, typename IdxT, typename OutIdxT>
static inline void launch_csr_to_coo(const IdxT* indptr, const IdxT* index,
                                     const T* data, OutIdxT* row, OutIdxT* col,
                                     double* ndata, const int* cats,
                                     const bool* mask, int n_cells,
                                     cudaStream_t stream) {
    dim3 grid((unsigned)n_cells);
    dim3 block(BLOCK_SIZE_SPARSE);
    csr_to_coo_kernel<T, IdxT, OutIdxT><<<grid, block, 0, stream>>>(
        indptr, index, data, row, col, ndata, cats, mask, n_cells);
    CUDA_CHECK_LAST_ERROR(csr_to_coo_kernel);
}

template <typename IdxT>
static inline void launch_sparse_var(const IdxT* indptr, const IdxT* index,
                                     double* data, const double* mean_data,
                                     double* n_cells, int dof, int n_groups,
                                     cudaStream_t stream) {
    dim3 grid((unsigned)n_groups);
    dim3 block(BLOCK_SIZE_SPARSE);
    sparse_var_kernel<IdxT><<<grid, block, 0, stream>>>(
        indptr, index, data, mean_data, n_cells, dof, n_groups);
    CUDA_CHECK_LAST_ERROR(sparse_var_kernel);
}

template <typename T, typename IdxT, typename Device>
void def_sparse_aggr(nb::module_& m) {
    m.def(
        "sparse_aggr",
        [](gpu_array_c<const IdxT, Device> indptr,
           gpu_array_c<const IdxT, Device> index,
           gpu_array_c<const T, Device> data,
           std::optional<gpu_array_c<double, Device>> out_sum,
           std::optional<gpu_array_c<double, Device>> out_count,
           std::optional<gpu_array_c<double, Device>> out_sqsum,
           gpu_array_c<const int, Device> cats,
           gpu_array_c<const bool, Device> mask, size_t n_cells, size_t n_genes,
           bool is_csc, std::uintptr_t stream) {
            double* ps = out_sum ? out_sum->data() : nullptr;
            double* pc = out_count ? out_count->data() : nullptr;
            double* pq = out_sqsum ? out_sqsum->data() : nullptr;
            int active = (ps ? AGGR_SUM : 0) | (pc ? AGGR_COUNT : 0) |
                         (pq ? AGGR_SQSUM : 0);
#define LAUNCH(M)                                                     \
    launch_sparse_aggr<T, IdxT, M>(                                   \
        is_csc, indptr.data(), index.data(), data.data(), ps, pc, pq, \
        cats.data(), mask.data(), n_cells, n_genes, (cudaStream_t)stream)
            AGGR_DISPATCH_MASK(active, LAUNCH);
#undef LAUNCH
        },
        "indptr"_a, "index"_a, "data"_a, nb::kw_only(),
        "out_sum"_a = nb::none(), "out_count"_a = nb::none(),
        "out_sqsum"_a = nb::none(), "cats"_a, "mask"_a, "n_cells"_a,
        "n_genes"_a, "is_csc"_a, "stream"_a = 0);
}

template <typename T, typename DataContig, typename Device>
void def_dense_aggr(nb::module_& m) {
    m.def(
        "dense_aggr",
        [](gpu_array_contig<const T, Device, DataContig> data,
           std::optional<gpu_array_c<double, Device>> out_sum,
           std::optional<gpu_array_c<double, Device>> out_count,
           std::optional<gpu_array_c<double, Device>> out_sqsum,
           gpu_array_c<const int, Device> cats,
           gpu_array_c<const bool, Device> mask, size_t n_cells, size_t n_genes,
           bool is_fortran, std::uintptr_t stream) {
            double* ps = out_sum ? out_sum->data() : nullptr;
            double* pc = out_count ? out_count->data() : nullptr;
            double* pq = out_sqsum ? out_sqsum->data() : nullptr;
            int active = (ps ? AGGR_SUM : 0) | (pc ? AGGR_COUNT : 0) |
                         (pq ? AGGR_SQSUM : 0);
            constexpr bool is_f = std::is_same_v<DataContig, nb::f_contig>;
#define LAUNCH(M)                                                       \
    launch_dense_aggr<T, M>(is_f, data.data(), ps, pc, pq, cats.data(), \
                            mask.data(), n_cells, n_genes,              \
                            (cudaStream_t)stream)
            AGGR_DISPATCH_MASK(active, LAUNCH);
#undef LAUNCH
        },
        "data"_a, nb::kw_only(), "out_sum"_a = nb::none(),
        "out_count"_a = nb::none(), "out_sqsum"_a = nb::none(), "cats"_a,
        "mask"_a, "n_cells"_a, "n_genes"_a, "is_fortran"_a, "stream"_a = 0);
}

template <typename T, typename IdxT, typename OutIdxT, typename Device>
void def_csr_to_coo(nb::module_& m) {
    m.def(
        "csr_to_coo",
        [](gpu_array_c<const IdxT, Device> indptr,
           gpu_array_c<const IdxT, Device> index,
           gpu_array_c<const T, Device> data,
           gpu_array_c<OutIdxT, Device> out_row,
           gpu_array_c<OutIdxT, Device> out_col,
           gpu_array_c<double, Device> out_data,
           gpu_array_c<const int, Device> cats,
           gpu_array_c<const bool, Device> mask, int n_cells,
           std::uintptr_t stream) {
            launch_csr_to_coo<T, IdxT, OutIdxT>(
                indptr.data(), index.data(), data.data(), out_row.data(),
                out_col.data(), out_data.data(), cats.data(), mask.data(),
                n_cells, (cudaStream_t)stream);
        },
        "indptr"_a, "index"_a, "data"_a, nb::kw_only(), "out_row"_a,
        "out_col"_a, "out_data"_a, "cats"_a, "mask"_a, "n_cells"_a,
        "stream"_a = 0);
}

template <typename IdxT, typename Device>
void def_sparse_var(nb::module_& m) {
    m.def(
        "sparse_var",
        [](gpu_array_c<const IdxT, Device> indptr,
           gpu_array_c<const IdxT, Device> index,
           gpu_array_c<double, Device> data,
           gpu_array_c<const double, Device> means,
           gpu_array_c<double, Device> n_cells, int dof, int n_groups,
           std::uintptr_t stream) {
            launch_sparse_var<IdxT>(indptr.data(), index.data(), data.data(),
                                    means.data(), n_cells.data(), dof, n_groups,
                                    (cudaStream_t)stream);
        },
        "indptr"_a, "index"_a, "data"_a, nb::kw_only(), "means"_a, "n_cells"_a,
        "dof"_a, "n_groups"_a, "stream"_a = 0);
}

template <typename Device>
void register_bindings(nb::module_& m) {
    def_sparse_aggr<float, int, Device>(m);
    def_sparse_aggr<float, long long, Device>(m);
    def_sparse_aggr<double, int, Device>(m);
    def_sparse_aggr<double, long long, Device>(m);

    // F-order must come before C-order for proper dispatch
    def_dense_aggr<float, nb::f_contig, Device>(m);
    def_dense_aggr<float, nb::c_contig, Device>(m);
    def_dense_aggr<double, nb::f_contig, Device>(m);
    def_dense_aggr<double, nb::c_contig, Device>(m);

    def_csr_to_coo<float, int, int, Device>(m);
    def_csr_to_coo<float, long long, long long, Device>(m);
    def_csr_to_coo<double, int, int, Device>(m);
    def_csr_to_coo<double, long long, long long, Device>(m);

    def_sparse_var<int, Device>(m);
    def_sparse_var<long long, Device>(m);
}

NB_MODULE(_aggr_cuda, m) {
    REGISTER_GPU_BINDINGS(register_bindings, m);
}
