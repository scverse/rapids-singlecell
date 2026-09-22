#include <cuda_runtime.h>
#include "../nb_types.h"

#include "kernels_ora.cuh"

using namespace nb::literals;

template <typename Device>
void register_bindings(nb::module_& m) {
    m.def(
        "select",
        [](gpu_array_c<const float, Device> mat, int n_up, int n_bm,
           gpu_array_c<unsigned char, Device> selected, std::uintptr_t stream) {
            if (mat.shape(0) == 0) return;
            select_kernel<<<strided_grid(mat.shape(0), 1), 256, 0,
                            (cudaStream_t)stream>>>(mat.data(), mat.shape(0),
                                                    (int)mat.shape(1), n_up,
                                                    n_bm, selected.data());
            CUDA_CHECK_LAST_ERROR(select_kernel);
        },
        "mat"_a, "n_up"_a, "n_bm"_a, "selected"_a, "stream"_a = 0);
    m.def(
        "overlap",
        [](gpu_array_c<const unsigned char, Device> selected,
           gpu_array_c<const int, Device> cnct,
           gpu_array_c<const int, Device> starts,
           gpu_array_c<const int, Device> offsets,
           gpu_array_c<int, Device> overlaps, std::uintptr_t stream) {
            if (selected.ndim() != 2 || overlaps.ndim() != 2 ||
                overlaps.shape(0) != starts.size() ||
                offsets.size() != starts.size() ||
                overlaps.shape(1) != selected.shape(1)) {
                throw std::invalid_argument(
                    "overlap: incompatible array shapes");
            }
            if (overlaps.size() == 0) return;
            const int rows_per_block =
                selected.shape(1) < 32 ? WARPS_PER_BLOCK : BLOCK_SIZE;
            const dim3 grid(strided_grid(selected.shape(1), rows_per_block),
                            strided_grid_y(starts.size(), 1));
            overlap_kernel<<<grid, BLOCK_SIZE, 0, (cudaStream_t)stream>>>(
                selected.data(), cnct.data(), starts.data(), offsets.data(),
                selected.shape(1), (int)starts.size(), overlaps.data());
            CUDA_CHECK_LAST_ERROR(overlap_kernel);
        },
        "selected"_a, nb::kw_only(), "cnct"_a, "starts"_a, "offsets"_a,
        "overlaps"_a, "stream"_a = 0);
    m.def(
        "ora_lookup",
        [](gpu_array_c<const unsigned char, Device> selected,
           gpu_array_c<const int, Device> cnct,
           gpu_array_c<const int, Device> starts,
           gpu_array_c<const int, Device> offsets,
           gpu_array_c<const int, Device> groups,
           gpu_array_c<const double, Device> score_table,
           gpu_array_c<const double, Device> p_table, int count,
           double background, gpu_array_c<double, Device> es,
           gpu_array_c<double, Device> pv, gpu_array_c<int, Device> invalid,
           std::uintptr_t stream) {
            if (selected.ndim() != 2 || es.ndim() != 2 || pv.ndim() != 2 ||
                es.shape(0) != starts.size() ||
                es.shape(1) != selected.shape(1) ||
                pv.shape(0) != es.shape(0) || pv.shape(1) != es.shape(1) ||
                offsets.size() != starts.size() ||
                groups.size() != starts.size() || score_table.ndim() != 2 ||
                p_table.ndim() != 2 ||
                p_table.shape(0) != score_table.shape(0) ||
                p_table.shape(1) != score_table.shape(1) ||
                invalid.size() != 1) {
                throw std::invalid_argument(
                    "ora_lookup: incompatible array shapes");
            }
            if (es.size() == 0) return;
            const int rows_per_block =
                selected.shape(1) < 32 ? WARPS_PER_BLOCK : BLOCK_SIZE;
            const dim3 grid(strided_grid(selected.shape(1), rows_per_block),
                            strided_grid_y(starts.size(), 1));
            overlap_kernel<true><<<grid, BLOCK_SIZE, 0, (cudaStream_t)stream>>>(
                selected.data(), cnct.data(), starts.data(), offsets.data(),
                selected.shape(1), (int)starts.size(), nullptr,
                {groups.data(), score_table.data(), p_table.data(),
                 (int)score_table.shape(1), count, background, es.data(),
                 pv.data(), invalid.data()});
            CUDA_CHECK_LAST_ERROR(overlap_kernel);
        },
        "selected"_a, nb::kw_only(), "cnct"_a, "starts"_a, "offsets"_a,
        "groups"_a, "score_table"_a, "p_table"_a, "count"_a, "background"_a,
        "es"_a, "pv"_a, "invalid"_a, "stream"_a = 0);
    m.def(
        "fisher",
        [](gpu_array_c<const int, Device> a,
           gpu_array_c<const int, Device> counts,
           gpu_array_c<const int, Device> sizes, double background,
           gpu_array_c<double, Device> pv, int alternative,
           gpu_array_c<const double, Device> gamma, long long tail_offset,
           std::uintptr_t stream) {
            const size_t ncols = a.ndim() == 2 ? a.shape(1) : 1;
            if ((a.ndim() != 1 && a.ndim() != 2) || pv.size() != a.size() ||
                (counts.size() != 1 && counts.size() != ncols) ||
                (sizes.size() != 1 && sizes.size() != a.shape(0))) {
                throw std::invalid_argument(
                    "fisher: incompatible array shapes");
            }
            if (a.size() == 0) return;
            fisher_kernel<<<strided_grid(a.size(), WARPS_PER_BLOCK), BLOCK_SIZE,
                            0, (cudaStream_t)stream>>>(
                a.data(), counts.data(), sizes.data(), a.size(), ncols,
                counts.size() == 1, sizes.size() == 1, background, alternative,
                gamma.size() ? gamma.data() : nullptr, tail_offset, pv.data());
            CUDA_CHECK_LAST_ERROR(fisher_kernel);
        },
        "a"_a, nb::kw_only(), "counts"_a, "sizes"_a, "background"_a, "pv"_a,
        "alternative"_a, "gamma"_a, "tail_offset"_a, "stream"_a = 0);
}

NB_MODULE(_ora_cuda, m) {
    REGISTER_GPU_BINDINGS(register_bindings, m);
}
