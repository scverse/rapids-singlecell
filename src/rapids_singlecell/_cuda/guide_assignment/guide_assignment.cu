#include <cuda_runtime.h>

#include "../nb_types.h"

#include "kernels_guide_assignment.cuh"
#include "kernels_sparse.cuh"

using namespace nb::literals;
namespace ga = guide_assignment;

constexpr int SPARSE_BLOCK = 256;

template <typename T, typename IdxT, typename Device>
ga::CscColumns<T, IdxT> csc_columns(gpu_array_c<const IdxT, Device>& indptr,
                                    gpu_array_c<const IdxT, Device>& indices,
                                    gpu_array_c<const T, Device>& data) {
    return {indptr.data(), indices.data(), data.data()};
}

// Blocks for one warp per row.
inline unsigned int warp_blocks(long long n_rows) {
    return static_cast<unsigned int>((n_rows * 32 + SPARSE_BLOCK - 1) /
                                     SPARSE_BLOCK);
}

template <typename Columns, typename Device>
void histogram_columns(Columns columns, long long n_guides, long long chunk,
                       long long n_chunks,
                       gpu_array_c<unsigned int, Device>& hist,
                       gpu_array_c<unsigned long long, Device>& n_tail,
                       gpu_array_c<unsigned long long, Device>& tail,
                       std::uintptr_t stream) {
    if (n_guides == 0) return;
    const int cap = static_cast<int>(hist.shape(1));
    ga::value_histogram_columns_kernel<<<dim3(n_guides, n_chunks), SPARSE_BLOCK,
                                         cap * sizeof(unsigned int),
                                         (cudaStream_t)stream>>>(
        columns, chunk, cap, hist.data(), n_tail.data(), tail.data(),
        tail.shape(0));
    CUDA_CHECK_LAST_ERROR(value_histogram_columns_kernel);
}

// Fits every (guide, start) by MAP-EM, then keeps the best admissible start of
// every guide and its threshold in out.
template <typename Device>
void def_fit(nb::module_& m) {
    m.def(
        "fit_poisson_gauss",
        [](gpu_array_c<const float, Device> values,
           gpu_array_c<const float, Device> weights,
           gpu_array_c<const int, Device> n_values,
           gpu_array_c<const int, Device> max_counts,
           gpu_array_c<const float, Device> draws,
           gpu_array_c<float, Device> params, gpu_array_c<float, Device> loss,
           gpu_array_c<float, Device> out, int n_iter, float tol,
           std::uintptr_t stream) {
            const int n_guides = static_cast<int>(values.shape(0));
            const int n_starts = static_cast<int>(params.shape(0));
            if (n_guides == 0) return;
            ga::fit_poisson_gauss_kernel<<<dim3(n_guides, n_starts),
                                           ga::BLOCK_SIZE, 0,
                                           (cudaStream_t)stream>>>(
                values.data(), weights.data(), n_values.data(), draws.data(),
                params.data(), loss.data(), n_guides,
                static_cast<int>(values.shape(1)),
                static_cast<int>(draws.shape(1)), n_iter, tol);
            CUDA_CHECK_LAST_ERROR(fit_poisson_gauss_kernel);
            ga::select_threshold_kernel<<<n_guides, ga::BLOCK_SIZE, 0,
                                          (cudaStream_t)stream>>>(
                params.data(), loss.data(), max_counts.data(), out.data(),
                n_guides, n_starts);
            CUDA_CHECK_LAST_ERROR(select_threshold_kernel);
        },
        "values"_a.noconvert(), "weights"_a.noconvert(),
        "n_values"_a.noconvert(), "max_counts"_a.noconvert(),
        "draws"_a.noconvert(), "params"_a.noconvert(), "loss"_a.noconvert(),
        "out"_a.noconvert(), nb::kw_only(), "n_iter"_a, "tol"_a,
        "stream"_a = 0);
}

template <typename T, typename IdxT, typename Device>
void def_sparse(nb::module_& m) {
    m.def(
        "value_histogram_csr",
        [](gpu_array_c<const IdxT, Device> indices,
           gpu_array_c<const T, Device> data,
           gpu_array_c<unsigned int, Device> hist,
           gpu_array_c<unsigned long long, Device> n_tail,
           gpu_array_c<unsigned long long, Device> tail,
           std::uintptr_t stream) {
            const long long nnz = static_cast<long long>(data.shape(0));
            if (nnz == 0) return;
            ga::value_histogram_csr_kernel<<<strided_grid(nnz, SPARSE_BLOCK),
                                             SPARSE_BLOCK, 0,
                                             (cudaStream_t)stream>>>(
                indices.data(), data.data(), nnz,
                static_cast<int>(hist.shape(1)), hist.data(), n_tail.data(),
                tail.data(), tail.shape(0));
            CUDA_CHECK_LAST_ERROR(value_histogram_csr_kernel);
        },
        "indices"_a.noconvert(), "data"_a.noconvert(), "hist"_a.noconvert(),
        "n_tail"_a.noconvert(), "tail"_a.noconvert(), nb::kw_only(),
        "stream"_a = 0);
    m.def(
        "value_histogram_csc",
        [](gpu_array_c<const IdxT, Device> indptr,
           gpu_array_c<const IdxT, Device> indices,
           gpu_array_c<const T, Device> data,
           gpu_array_c<unsigned int, Device> hist,
           gpu_array_c<unsigned long long, Device> n_tail,
           gpu_array_c<unsigned long long, Device> tail, long long chunk,
           long long n_chunks, std::uintptr_t stream) {
            histogram_columns(
                csc_columns<T, IdxT, Device>(indptr, indices, data),
                static_cast<long long>(indptr.shape(0)) - 1, chunk, n_chunks,
                hist, n_tail, tail, stream);
        },
        "indptr"_a.noconvert(), "indices"_a.noconvert(), "data"_a.noconvert(),
        "hist"_a.noconvert(), "n_tail"_a.noconvert(), "tail"_a.noconvert(),
        nb::kw_only(), "chunk"_a, "n_chunks"_a, "stream"_a = 0);
    m.def(
        "assigned_csr",
        [](gpu_array_c<const IdxT, Device> indptr,
           gpu_array_c<const IdxT, Device> indices,
           gpu_array_c<const T, Device> data,
           gpu_array_c<const double, Device> thresholds,
           gpu_array_c<IdxT, Device> offsets,
           gpu_array_c<IdxT, Device> out_indices, bool fill,
           std::uintptr_t stream) {
            const long long n_rows =
                static_cast<long long>(indptr.shape(0)) - 1;
            if (n_rows <= 0) return;
            auto kernel = fill ? ga::assigned_csr_kernel<T, IdxT, true>
                               : ga::assigned_csr_kernel<T, IdxT, false>;
            kernel<<<warp_blocks(n_rows), SPARSE_BLOCK, 0,
                     (cudaStream_t)stream>>>(
                indptr.data(), indices.data(), data.data(), thresholds.data(),
                n_rows, offsets.data(), out_indices.data());
            CUDA_CHECK_LAST_ERROR(assigned_csr_kernel);
        },
        "indptr"_a.noconvert(), "indices"_a.noconvert(), "data"_a.noconvert(),
        "thresholds"_a.noconvert(), "offsets"_a.noconvert(),
        "out_indices"_a.noconvert(), nb::kw_only(), "fill"_a, "stream"_a = 0);
    m.def(
        "assigned_csc",
        [](gpu_array_c<const IdxT, Device> indptr,
           gpu_array_c<const IdxT, Device> indices,
           gpu_array_c<const T, Device> data,
           gpu_array_c<const double, Device> thresholds,
           gpu_array_c<IdxT, Device> offsets,
           gpu_array_c<IdxT, Device> out_indices, long long chunk,
           long long n_chunks, bool fill, std::uintptr_t stream) {
            const long long n_guides =
                static_cast<long long>(indptr.shape(0)) - 1;
            if (n_guides <= 0) return;
            auto kernel = fill ? ga::assigned_csc_kernel<T, IdxT, true>
                               : ga::assigned_csc_kernel<T, IdxT, false>;
            kernel<<<dim3(n_guides, n_chunks), SPARSE_BLOCK, 0,
                     (cudaStream_t)stream>>>(
                csc_columns<T, IdxT, Device>(indptr, indices, data),
                thresholds.data(), chunk, offsets.data(), out_indices.data());
            CUDA_CHECK_LAST_ERROR(assigned_csc_kernel);
        },
        "indptr"_a.noconvert(), "indices"_a.noconvert(), "data"_a.noconvert(),
        "thresholds"_a.noconvert(), "offsets"_a.noconvert(),
        "out_indices"_a.noconvert(), nb::kw_only(), "chunk"_a, "n_chunks"_a,
        "fill"_a, "stream"_a = 0);
    m.def(
        "max_guide_csr",
        [](gpu_array_c<const IdxT, Device> indptr,
           gpu_array_c<const IdxT, Device> indices,
           gpu_array_c<const T, Device> data,
           gpu_array_c<double, Device> values, gpu_array_c<int, Device> guides,
           std::uintptr_t stream) {
            const long long n_rows =
                static_cast<long long>(indptr.shape(0)) - 1;
            if (n_rows <= 0) return;
            ga::max_guide_csr_kernel<T, IdxT>
                <<<warp_blocks(n_rows), SPARSE_BLOCK, 0,
                   (cudaStream_t)stream>>>(indptr.data(), indices.data(),
                                           data.data(), n_rows, values.data(),
                                           guides.data());
            CUDA_CHECK_LAST_ERROR(max_guide_csr_kernel);
        },
        "indptr"_a.noconvert(), "indices"_a.noconvert(), "data"_a.noconvert(),
        "values"_a.noconvert(), "guides"_a.noconvert(), nb::kw_only(),
        "stream"_a = 0);
    m.def(
        "max_guide_csc",
        [](gpu_array_c<const IdxT, Device> indptr,
           gpu_array_c<const IdxT, Device> indices,
           gpu_array_c<const T, Device> data,
           gpu_array_c<unsigned long long, Device> keys,
           gpu_array_c<double, Device> values, gpu_array_c<int, Device> guides,
           long long chunk, long long n_chunks, std::uintptr_t stream) {
            const long long n_guides =
                static_cast<long long>(indptr.shape(0)) - 1;
            const long long n_rows = static_cast<long long>(keys.shape(0));
            if (n_rows == 0) return;
            const auto columns =
                csc_columns<T, IdxT, Device>(indptr, indices, data);
            const dim3 grid(n_guides, n_chunks);
            if (n_guides > 0) {
                ga::max_key_csc_kernel<<<grid, SPARSE_BLOCK, 0,
                                         (cudaStream_t)stream>>>(columns, chunk,
                                                                 keys.data());
                CUDA_CHECK_LAST_ERROR(max_key_csc_kernel);
                ga::max_guide_csc_kernel<<<grid, SPARSE_BLOCK, 0,
                                           (cudaStream_t)stream>>>(
                    columns, chunk, keys.data(), guides.data());
                CUDA_CHECK_LAST_ERROR(max_guide_csc_kernel);
            }
            ga::max_guide_finish_kernel<<<strided_grid(n_rows, SPARSE_BLOCK),
                                          SPARSE_BLOCK, 0,
                                          (cudaStream_t)stream>>>(
                keys.data(), n_rows, values.data(), guides.data());
            CUDA_CHECK_LAST_ERROR(max_guide_finish_kernel);
        },
        "indptr"_a.noconvert(), "indices"_a.noconvert(), "data"_a.noconvert(),
        "keys"_a.noconvert(), "values"_a.noconvert(), "guides"_a.noconvert(),
        nb::kw_only(), "chunk"_a, "n_chunks"_a, "stream"_a = 0);
}

template <typename T, typename Device>
void def_dense(nb::module_& m) {
    m.def(
        "value_histogram_dense",
        [](gpu_array<const T, Device> x, gpu_array_c<unsigned int, Device> hist,
           gpu_array_c<unsigned long long, Device> n_tail,
           gpu_array_c<unsigned long long, Device> tail, long long chunk,
           long long n_chunks, std::uintptr_t stream) {
            histogram_columns(
                ga::DenseColumns<T>{x.data(),
                                    static_cast<long long>(x.shape(0)),
                                    static_cast<long long>(x.stride(0)),
                                    static_cast<long long>(x.stride(1))},
                static_cast<long long>(x.shape(1)), chunk, n_chunks, hist,
                n_tail, tail, stream);
        },
        "x"_a.noconvert(), "hist"_a.noconvert(), "n_tail"_a.noconvert(),
        "tail"_a.noconvert(), nb::kw_only(), "chunk"_a, "n_chunks"_a,
        "stream"_a = 0);
}

template <typename Device>
void register_bindings(nb::module_& m) {
    def_fit<Device>(m);
    def_sparse<float, int, Device>(m);
    def_sparse<float, long long, Device>(m);
    def_sparse<double, int, Device>(m);
    def_sparse<double, long long, Device>(m);
    def_dense<float, Device>(m);
    def_dense<double, Device>(m);
}

NB_MODULE(_guide_assignment_cuda, m) {
    REGISTER_GPU_BINDINGS(register_bindings, m);
}
