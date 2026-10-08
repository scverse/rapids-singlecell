#include <cuda_runtime.h>

#include <algorithm>
#include <cstdint>
#include <stdexcept>

#include "../nb_types.h"
#include "kernels_dsb.cuh"

using namespace nb::literals;

constexpr int WARPS_PER_BLOCK = 8;
constexpr int COOP_THREADS = 256;
constexpr long long COOP_MIN_CHUNK = 2048;

// Cooperative work split: chunks of >= COOP_MIN_CHUNK values, at most one
// chunk per co-resident block, no more blocks than (pair, chunk) items.
struct CoopLayout {
    long long grid, chunk, n_chunks;
};

template <typename T>
static CoopLayout coop_layout(long long n_fits, int n) {
    int device = 0, n_sm = 0, per_sm = 0;
    cuda_check(cudaGetDevice(&device), "cudaGetDevice");
    cuda_check(
        cudaDeviceGetAttribute(&n_sm, cudaDevAttrMultiProcessorCount, device),
        "cudaDeviceGetAttribute");
    cuda_check(cudaOccupancyMaxActiveBlocksPerMultiprocessor(
                   &per_sm, dsb::mix2_coop_kernel<T>, COOP_THREADS, 0),
               "cudaOccupancyMaxActiveBlocksPerMultiprocessor");
    const long long max_grid = (long long)n_sm * (per_sm > 0 ? per_sm : 1);
    const long long chunk =
        std::max(COOP_MIN_CHUNK, ((long long)n + max_grid - 1) / max_grid);
    const long long n_chunks = ((long long)n + chunk - 1) / chunk;
    return {std::min(max_grid, 2 * n_fits * n_chunks), chunk, n_chunks};
}

template <typename T>
static size_t coop_workspace_bytes(long long n_fits, int n) {
    return 2 * n_fits *
           (sizeof(dsb::FitState<T>) +
            coop_layout<T>(n_fits, n).n_chunks * 7 * sizeof(double));
}

template <typename T, typename Device>
static void def_mix2(nb::module_& m) {
    // init: empty -> fit every row; (n_fits, 2, 6) -> only write start values
    m.def(
        "mix2_warp",
        [](gpu_array_c<const T, Device> X, gpu_array_c<T, Device> params,
           gpu_array_c<int, Device> status, gpu_array_c<T, Device> q,
           gpu_array_c<T, Device> init, int max_iter, std::uintptr_t stream) {
            const long long n_fits = X.shape(0);
            if (n_fits == 0) return;
            const unsigned int grid = strided_grid(n_fits, WARPS_PER_BLOCK);
            dsb::mix2_warp_kernel<T><<<grid, WARPS_PER_BLOCK * dsb::kWarp, 0,
                                       (cudaStream_t)stream>>>(
                X.data(), n_fits, (int)X.shape(1), max_iter, params.data(),
                status.data(), q.data(), init.size() ? init.data() : nullptr);
            CUDA_CHECK_LAST_ERROR(mix2_warp_kernel);
        },
        "X"_a, "params"_a, "status"_a, "q"_a, "init"_a, nb::kw_only(),
        "max_iter"_a, "stream"_a = 0);

    m.def(
        "mix2_coop",
        [](gpu_array_c<const T, Device> X, gpu_array_c<const T, Device> init,
           gpu_array_c<T, Device> params, gpu_array_c<int, Device> status,
           gpu_array_c<std::uint8_t, Device> work, int max_iter,
           std::uintptr_t stream) {
            long long n_fits = X.shape(0);
            int n = (int)X.shape(1);
            if (n_fits == 0) return;
            nb_require(work.size() >= coop_workspace_bytes<T>(n_fits, n),
                       "dsb workspace too small");
            CoopLayout l = coop_layout<T>(n_fits, n);
            const T* x = X.data();
            const T* ip = init.data();
            auto* states = reinterpret_cast<dsb::FitState<T>*>(work.data());
            double* partials = reinterpret_cast<double*>(states + 2 * n_fits);
            T* pp = params.data();
            int* sp = status.data();
            void* args[] = {&x,        &n_fits,     &n,  &max_iter,
                            &l.chunk,  &l.n_chunks, &ip, &states,
                            &partials, &pp,         &sp};
            cuda_check(cudaLaunchCooperativeKernel(
                           (void*)dsb::mix2_coop_kernel<T>, dim3(l.grid),
                           dim3(COOP_THREADS), args, 0, (cudaStream_t)stream),
                       "dsb: cooperative launch for vectors longer than 2000 "
                       "values (unsupported on some MPS / virtualized GPUs)");
        },
        "X"_a, "init"_a, "params"_a, "status"_a, "work"_a, nb::kw_only(),
        "max_iter"_a, "stream"_a = 0);
}

template <typename Device>
void register_bindings(nb::module_& m) {
    def_mix2<float, Device>(m);
    def_mix2<double, Device>(m);
}

NB_MODULE(_dsb_cuda, m) {
    m.doc() = "Batched two-component univariate Gaussian mixtures for dsb.";
    m.def(
        "mix2_coop_workspace",
        [](long long n_fits, int n, bool double_precision) {
            return double_precision ? coop_workspace_bytes<double>(n_fits, n)
                                    : coop_workspace_bytes<float>(n_fits, n);
        },
        "n_fits"_a, "n"_a, "double_precision"_a);
    REGISTER_GPU_BINDINGS(register_bindings, m);
}
