#include <cuda_runtime.h>
#include <cstdint>
#include "../nb_types.h"

#include "kernels_wnn.cuh"

using namespace nb::literals;

constexpr int WNN_BLOCK_SIZE = 256;

// Launches `kernel` with `smem` bytes of dynamic shared memory, opting in
// above the default limit.
template <typename Kernel, typename... Args>
static void launch(Kernel kernel, unsigned grid, int block, size_t smem,
                   std::uintptr_t stream, const char* name, Args... args) {
    cuda_check(
        cudaFuncSetAttribute(
            kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, (int)smem),
        name);
    kernel<<<grid, block, smem, (cudaStream_t)stream>>>(args...);
    cuda_check_last_error(name);
}

// Checks shared by the SNN launches; returns whether there are rows.
template <typename Knn, typename Rows, typename Seg>
static bool snn_rows(const Knn& knn, int s, const Rows& rows, const Seg& seg,
                     bool presorted) {
    nb_require(s >= 1 && s <= (int)knn.shape(1),
               "s must be in [1, knn.shape[1]]");
    nb_require(!presorted || seg.size() == rows.size() + 1,
               "seg must have rows + 1 entries");
    nb_require(rows.size() <= (size_t)max_grid_dim_x(), "too many rows");
    return rows.size() > 0;
}

template <typename Device>
void register_bindings(nb::module_& m) {
    m.def(
        "impute_dist",
        [](gpu_array_c<const float, Device> emb,
           gpu_array_c<const int, Device> knn, int k,
           gpu_array_c<const float, Device> nearest,
           gpu_array_c<float, Device> out, std::uintptr_t stream) {
            const int n_obs = (int)emb.shape(0);
            const int k_max = (int)knn.shape(1);
            nb_require(knn.shape(0) == (size_t)n_obs, "knn rows != n_obs");
            nb_require(k >= 2 && k <= k_max, "k must be in [2, knn.shape[1]]");
            if (n_obs == 0) return;
            unsigned grid = strided_grid((long long)n_obs * 32, WNN_BLOCK_SIZE);
            launch(wnn_impute_dist_kernel, grid, WNN_BLOCK_SIZE, 0, stream,
                   "wnn_impute_dist_kernel", emb.data(), (int)emb.shape(1),
                   knn.data(), k_max, k, nearest.data(), out.data(), n_obs);
        },
        "emb"_a, "knn"_a, nb::kw_only(), "k"_a, "nearest"_a, "out"_a,
        "stream"_a = 0);

    m.def(
        "snn_postings",
        [](gpu_array_c<const int, Device> knn, int s,
           gpu_array_c<long long, Device> cursor,
           gpu_array_c<int, Device> postings, std::uintptr_t stream) {
            const long long n_obs = (long long)knn.shape(0);
            nb_require(s >= 1 && s <= (int)knn.shape(1),
                       "s must be in [1, knn.shape[1]]");
            nb_require(cursor.shape(0) == (size_t)n_obs,
                       "cursor size != n_obs");
            nb_require(
                !postings.size() || postings.size() == (size_t)(n_obs * s),
                "postings must have n_obs * s entries");
            if (n_obs == 0) return;
            launch(wnn_postings_kernel, strided_grid(n_obs * s, WNN_BLOCK_SIZE),
                   WNN_BLOCK_SIZE, 0, stream, "wnn_postings_kernel", knn.data(),
                   (int)knn.shape(1), s, n_obs,
                   reinterpret_cast<unsigned long long*>(cursor.data()),
                   postings.size() ? postings.data() : nullptr);
        },
        "knn"_a, nb::kw_only(), "s"_a, "cursor"_a, "postings"_a.none(),
        "stream"_a = 0);

    m.def(
        "snn_emit",
        [](gpu_array_c<const int, Device> knn, int s,
           gpu_array_c<const int, Device> rows,
           gpu_array_c<const long long, Device> post_off,
           gpu_array_c<const int, Device> postings,
           gpu_array_c<const long long, Device> seg,
           gpu_array_c<unsigned long long, Device> keys,
           std::uintptr_t stream) {
            if (!snn_rows(knn, s, rows, seg, true)) return;
            launch(wnn_snn_emit_kernel, (unsigned)rows.size(), WNN_SNN_THREADS,
                   0, stream, "wnn_snn_emit_kernel", rows.data(), knn.data(),
                   (int)knn.shape(1), s, post_off.data(), postings.data(),
                   seg.data(), keys.data());
        },
        "knn"_a, nb::kw_only(), "s"_a, "rows"_a, "post_off"_a, "postings"_a,
        "seg"_a, "keys"_a, "stream"_a = 0);

    m.def(
        "snn_bandwidth",
        [](gpu_array_c<const int, Device> knn, int s,
           gpu_array_c<const int, Device> rows,
           gpu_array_c<const long long, Device> post_off,
           gpu_array_c<const int, Device> postings, int P,
           gpu_array_c<unsigned int, Device> sorted,
           gpu_array_c<const long long, Device> seg,
           gpu_array_c<float, Device> dist,
           gpu_array_c<const float, Device> emb,
           gpu_array_c<const float, Device> nearest,
           gpu_array_c<float, Device> sigma, std::uintptr_t stream) {
            nb_require(emb.shape(0) == knn.shape(0), "emb rows != knn rows");
            nb_require(
                nearest.size() == knn.shape(0) && sigma.size() == knn.shape(0),
                "nearest and sigma need n_obs entries");
            nb_require(dist.size() >= sorted.size(),
                       "dist shorter than sorted");
            const bool presorted = seg.size() > 0;
            if (!snn_rows(knn, s, rows, seg, presorted)) return;
            launch(presorted ? wnn_snn_bandwidth_kernel<true>
                             : wnn_snn_bandwidth_kernel<false>,
                   (unsigned)rows.size(), WNN_SNN_THREADS,
                   sizeof(int) * (s + 1 + (presorted ? 0 : 2 * (size_t)P)),
                   stream, "wnn_snn_bandwidth_kernel", rows.data(), knn.data(),
                   (int)knn.shape(1), s, post_off.data(), postings.data(), P,
                   sorted.data(), seg.data(), dist.data(), emb.data(),
                   (int)emb.shape(1), nearest.data(), sigma.data());
        },
        "knn"_a, nb::kw_only(), "s"_a, "rows"_a, "post_off"_a, "postings"_a,
        "P"_a, "sorted"_a.none(), "seg"_a.none(), "dist"_a.none(), "emb"_a,
        "nearest"_a, "sigma"_a, "stream"_a = 0);

    m.def(
        "snn_graph",
        [](gpu_array_c<const int, Device> knn, int s,
           gpu_array_c<const int, Device> rows,
           gpu_array_c<const long long, Device> post_off,
           gpu_array_c<const int, Device> postings, int P,
           gpu_array_c<unsigned int, Device> sorted,
           gpu_array_c<const long long, Device> seg, float prune,
           gpu_array_c<long long, Device> indptr, gpu_array_c<int, Device> cols,
           gpu_array_c<float, Device> vals, bool fill, std::uintptr_t stream) {
            nb_require(indptr.shape(0) == knn.shape(0) + 1,
                       "indptr must have n_obs + 1 entries");
            nb_require(!fill || (cols.size() && cols.size() == vals.size()),
                       "fill needs cols and vals of the same size");
            const bool presorted = seg.size() > 0;
            if (!snn_rows(knn, s, rows, seg, presorted)) return;
            launch(presorted ? wnn_snn_graph_kernel<true>
                             : wnn_snn_graph_kernel<false>,
                   (unsigned)rows.size(), WNN_SNN_THREADS,
                   presorted ? 0 : sizeof(int) * (size_t)P, stream,
                   "wnn_snn_graph_kernel", rows.data(), knn.data(),
                   (int)knn.shape(1), s, post_off.data(), postings.data(),
                   sorted.data(), seg.data(), prune, fill, indptr.data(),
                   cols.data(), vals.data());
        },
        "knn"_a, nb::kw_only(), "s"_a, "rows"_a, "post_off"_a, "postings"_a,
        "P"_a, "sorted"_a.none(), "seg"_a.none(), "prune"_a, "indptr"_a,
        "cols"_a.none(), "vals"_a.none(), "fill"_a, "stream"_a = 0);

    m.def(
        "multimodal_knn",
        [](gpu_array_c<const float, Device> emb,
           gpu_array_c<const int, Device> dim_off,
           gpu_array_c<const int, Device> knn, int knn_range,
           gpu_array_c<const float, Device> weight,
           gpu_array_c<const float, Device> sigma,
           gpu_array_c<const float, Device> nearest, int k_out,
           gpu_array_c<int, Device> out_idx,
           gpu_array_c<float, Device> out_dissim, std::uintptr_t stream) {
            const int n_obs = (int)emb.shape(0);
            const int n_mod = (int)knn.shape(0);
            const int k_max = (int)knn.shape(2);
            nb_require(knn.shape(1) == (size_t)n_obs, "knn rows != n_obs");
            nb_require(knn_range >= 2 && knn_range <= k_max,
                       "knn_range must be in [2, knn.shape[2]]");
            nb_require(dim_off.shape(0) == (size_t)n_mod + 1,
                       "dim_off must have n_mod + 1 entries");
            nb_require(k_out <= knn_range - 1, "k_out must be < knn_range");
            if (n_obs == 0) return;
            nb_require(n_obs <= max_grid_dim_x(), "too many cells per launch");
            int P = 1;
            while (P < n_mod * (knn_range - 1)) P <<= 1;
            launch(wnn_multimodal_knn_kernel, (unsigned)n_obs, WNN_BLOCK_SIZE,
                   (size_t)P * (2 * sizeof(int) + sizeof(float)), stream,
                   "wnn_multimodal_knn_kernel", emb.data(), (int)emb.shape(1),
                   dim_off.data(), knn.data(), k_max, knn_range, n_mod,
                   weight.data(), sigma.data(), nearest.data(), n_obs, P, k_out,
                   out_idx.data(), out_dissim.data());
        },
        "emb"_a, nb::kw_only(), "dim_off"_a, "knn"_a, "knn_range"_a, "weight"_a,
        "sigma"_a, "nearest"_a, "k_out"_a, "out_idx"_a, "out_dissim"_a,
        "stream"_a = 0);
}

NB_MODULE(_wnn_cuda, m) {
    REGISTER_GPU_BINDINGS(register_bindings, m);
}
