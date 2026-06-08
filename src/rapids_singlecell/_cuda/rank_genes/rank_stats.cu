#include <cuda_runtime.h>

#include "../nb_types.h"
#include "csr_tile_to_dense.cuh"

using namespace nb::literals;

namespace {

constexpr int GROUP_STATS_BLOCK = 256;

// Benjamini-Hochberg step-up tail: in-place reverse cumulative minimum along
// each row (group) of an already BH-scaled, p-value-sorted matrix. NaNs are
// treated as 1.0. One block per row, single thread per row (serial scan).
__global__ void fdr_bh_reverse_cummin_kernel(double* values, const int n_cols) {
    const int row = blockIdx.x;
    double running = 1.0;
    double* row_values = values + static_cast<size_t>(row) * n_cols;
    for (int col = n_cols - 1; col >= 0; --col) {
        double value = row_values[col];
        if (!(value == value)) {  // NaN -> 1.0
            value = 1.0;
        }
        if (value < running) {
            running = value;
        }
        row_values[col] = running;
    }
}

// Per-group sum / sum-of-squares / nnz over a dense F-order (column-major)
// block of shape (n_rows x n_cols). group_codes maps each row to a group; rows
// with an out-of-range code are skipped. Outputs are (n_groups x n_cols),
// C-order, accumulated with atomics. Grid-strided so a chunk larger than the
// gridDim.x cap is still fully covered.
__global__ void group_chunk_stats_kernel(
    const double* block, const int* group_codes, double* group_sums,
    double* group_sum_sq, double* group_nnz, const int n_rows, const int n_cols,
    const int n_groups, const bool compute_nnz) {
    const long long total = static_cast<long long>(n_rows) * n_cols;
    const long long stride = static_cast<long long>(blockDim.x) * gridDim.x;
    for (long long idx =
             static_cast<long long>(blockIdx.x) * blockDim.x + threadIdx.x;
         idx < total; idx += stride) {
        const int row = idx % n_rows;
        const int col = idx / n_rows;
        const int group = group_codes[row];
        if (group < 0 || group >= n_groups) {
            continue;
        }
        const double value = block[idx];
        const long long out = static_cast<long long>(group) * n_cols + col;
        atomicAdd(group_sums + out, value);
        atomicAdd(group_sum_sq + out, value * value);
        if (compute_nnz && value != 0.0) {
            atomicAdd(group_nnz + out, 1.0);
        }
    }
}

}  // namespace

// CSR -> dense F-order (double) window densify, in a single fused pass.
template <typename TData, typename IndptrT, typename IndexT, typename Device>
static void def_csr_tile_to_dense(nb::module_& m) {
    m.def(
        "csr_tile_to_dense",
        [](gpu_array_c<const IndptrT, Device> indptr,
           gpu_array_c<const IndexT, Device> indices,
           gpu_array_c<const TData, Device> data,
           gpu_array_f<double, Device> out, int col_lb, int col_ub,
           std::uintptr_t stream) {
            const int n_cells = static_cast<int>(indptr.shape(0)) - 1;
            if (n_cells <= 0 || col_ub <= col_lb) {
                return;
            }
            if (col_lb < 0) {
                throw std::invalid_argument(
                    "csr_tile_to_dense: col_lb must be non-negative");
            }
            if (indices.shape(0) != data.shape(0)) {
                throw std::invalid_argument(
                    "csr_tile_to_dense: indices and data must have equal "
                    "length");
            }
            if (out.ndim() != 2 || static_cast<int>(out.shape(0)) != n_cells ||
                static_cast<long long>(out.shape(1)) <
                    static_cast<long long>(col_ub) - col_lb) {
                throw std::invalid_argument(
                    "csr_tile_to_dense: out must be a (n_cells, >= col_ub - "
                    "col_lb) array");
            }
            constexpr int CSR_TILE_BLOCK = 128;
            const unsigned int grid =
                (static_cast<unsigned int>(n_cells) + CSR_TILE_BLOCK - 1) /
                CSR_TILE_BLOCK;
            csr_tile_to_dense_kernel<TData, IndptrT, IndexT>
                <<<grid, CSR_TILE_BLOCK, 0, (cudaStream_t)stream>>>(
                    indptr.data(), indices.data(), data.data(), out.data(),
                    col_lb, col_ub, n_cells);
            CUDA_CHECK_LAST_ERROR(csr_tile_to_dense_kernel);
        },
        "indptr"_a, "indices"_a, "data"_a, "out"_a, nb::kw_only(), "col_lb"_a,
        "col_ub"_a, "stream"_a = 0);
}

template <typename Device>
void register_bindings(nb::module_& m) {
    def_csr_tile_to_dense<float, int, int, Device>(m);
    def_csr_tile_to_dense<float, long long, int, Device>(m);
    def_csr_tile_to_dense<float, int, long long, Device>(m);
    def_csr_tile_to_dense<float, long long, long long, Device>(m);
    def_csr_tile_to_dense<double, int, int, Device>(m);
    def_csr_tile_to_dense<double, long long, int, Device>(m);
    def_csr_tile_to_dense<double, int, long long, Device>(m);
    def_csr_tile_to_dense<double, long long, long long, Device>(m);

    m.def(
        "fdr_bh_reverse_cummin",
        [](gpu_array_c<double, Device> values, std::uintptr_t stream) {
            const int n_rows = static_cast<int>(values.shape(0));
            const int n_cols = static_cast<int>(values.shape(1));
            if (n_rows <= 0 || n_cols <= 0) {
                return;
            }
            fdr_bh_reverse_cummin_kernel<<<dim3(n_rows), dim3(1), 0,
                                           (cudaStream_t)stream>>>(
                values.data(), n_cols);
            CUDA_CHECK_LAST_ERROR(fdr_bh_reverse_cummin_kernel);
        },
        "values"_a, nb::kw_only(), "stream"_a = 0);

    m.def(
        "group_chunk_stats",
        [](gpu_array_f<const double, Device> block,
           gpu_array_c<const int, Device> group_codes,
           gpu_array_c<double, Device> group_sums,
           gpu_array_c<double, Device> group_sum_sq,
           gpu_array_c<double, Device> group_nnz, bool compute_nnz,
           std::uintptr_t stream) {
            if (block.ndim() != 2 || group_sums.ndim() != 2 ||
                group_sum_sq.ndim() != 2) {
                throw std::invalid_argument(
                    "group_chunk_stats: block, group_sums and group_sum_sq "
                    "must be 2-D");
            }
            const int n_rows = static_cast<int>(block.shape(0));
            const int n_cols = static_cast<int>(block.shape(1));
            const int n_groups = static_cast<int>(group_sums.shape(0));
            const long long total = static_cast<long long>(n_rows) * n_cols;
            if (total <= 0) {
                return;
            }
            if (static_cast<int>(group_codes.shape(0)) != n_rows) {
                throw std::invalid_argument(
                    "group_chunk_stats: group_codes length must equal block "
                    "rows");
            }
            if (static_cast<int>(group_sum_sq.shape(0)) != n_groups ||
                static_cast<int>(group_sums.shape(1)) != n_cols ||
                static_cast<int>(group_sum_sq.shape(1)) != n_cols) {
                throw std::invalid_argument(
                    "group_chunk_stats: group_sums and group_sum_sq must be "
                    "(n_groups, n_cols)");
            }
            if (compute_nnz &&
                (group_nnz.ndim() != 2 ||
                 static_cast<int>(group_nnz.shape(0)) != n_groups ||
                 static_cast<int>(group_nnz.shape(1)) != n_cols)) {
                throw std::invalid_argument(
                    "group_chunk_stats: group_nnz must be (n_groups, n_cols) "
                    "when compute_nnz is set");
            }
            const unsigned int grid = strided_grid(total, GROUP_STATS_BLOCK);
            group_chunk_stats_kernel<<<grid, GROUP_STATS_BLOCK, 0,
                                       (cudaStream_t)stream>>>(
                block.data(), group_codes.data(), group_sums.data(),
                group_sum_sq.data(), group_nnz.data(), n_rows, n_cols, n_groups,
                compute_nnz);
            CUDA_CHECK_LAST_ERROR(group_chunk_stats_kernel);
        },
        "block"_a, "group_codes"_a, "group_sums"_a, "group_sum_sq"_a,
        "group_nnz"_a, nb::kw_only(), "compute_nnz"_a, "stream"_a = 0);
}

NB_MODULE(_rank_stats_cuda, m) {
    REGISTER_GPU_BINDINGS(register_bindings, m);
}
