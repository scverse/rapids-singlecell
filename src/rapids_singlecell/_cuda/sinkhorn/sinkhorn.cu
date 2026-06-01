#include <cuda_runtime.h>

#include <cstdint>

#include "../nb_types.h"
#include "kernels_sinkhorn.cuh"

using namespace nb::literals;

namespace {
constexpr int SINKHORN_BLOCK = 256;
}  // namespace

// Each binding launches one kernel on the caller-supplied stream, so the Python
// driver can dispatch work to several devices' streams (async multi-stream work
// queue). B, N, M are taken from the cost tensor shape (B, N, M).

template <typename T, typename Device>
static void def_kernels(nb::module_& m) {
    constexpr int BLOCK = SINKHORN_BLOCK;

    m.def(
        "auto_eps",
        [](gpu_array_c<const T, Device> cost,
           gpu_array_c<const bool, Device> mask_a,
           gpu_array_c<const bool, Device> mask_b,
           gpu_array_c<const T, Device> total, double scale, double floor,
           gpu_array_c<T, Device> eps, std::uintptr_t stream_ptr) {
            const int B = (int)cost.shape(0);
            const int N = (int)cost.shape(1);
            const int M = (int)cost.shape(2);
            cudaStream_t stream = reinterpret_cast<cudaStream_t>(stream_ptr);
            sinkhorn::auto_eps_kernel<T, BLOCK><<<B, BLOCK, 0, stream>>>(
                cost.data(), mask_a.data(), mask_b.data(), total.data(),
                static_cast<T>(scale), static_cast<T>(floor), eps.data(), N, M);
            CUDA_CHECK_LAST_ERROR(auto_eps_kernel);
        },
        "cost"_a, "mask_a"_a, "mask_b"_a, "total"_a, "scale"_a, "floor"_a,
        "eps"_a, "stream"_a = 0);

    m.def(
        "update_g",
        [](gpu_array_c<const T, Device> cost,
           gpu_array_c<const bool, Device> mask_a,
           gpu_array_c<const bool, Device> mask_b,
           gpu_array_c<const T, Device> f, gpu_array_c<const T, Device> eps,
           gpu_array_c<const T, Device> log_b,
           gpu_array_c<const int, Device> conv, gpu_array_c<T, Device> g,
           double omega, std::uintptr_t stream_ptr) {
            const int B = (int)cost.shape(0);
            const int N = (int)cost.shape(1);
            const int M = (int)cost.shape(2);
            cudaStream_t stream = reinterpret_cast<cudaStream_t>(stream_ptr);
            dim3 grid(B, (M + BLOCK - 1) / BLOCK);
            sinkhorn::update_g_kernel<T, BLOCK><<<grid, BLOCK, 0, stream>>>(
                cost.data(), mask_a.data(), mask_b.data(), f.data(), eps.data(),
                log_b.data(), conv.data(), g.data(), N, M,
                static_cast<T>(omega));
            CUDA_CHECK_LAST_ERROR(update_g_kernel);
        },
        "cost"_a, "mask_a"_a, "mask_b"_a, "f"_a, "eps"_a, "log_b"_a, "conv"_a,
        "g"_a, "omega"_a = 1.0, "stream"_a = 0);

    m.def(
        "update_f",
        [](gpu_array_c<const T, Device> cost,
           gpu_array_c<const bool, Device> mask_a,
           gpu_array_c<const bool, Device> mask_b,
           gpu_array_c<const T, Device> g, gpu_array_c<const T, Device> eps,
           gpu_array_c<const T, Device> log_a,
           gpu_array_c<const int, Device> conv, gpu_array_c<T, Device> f,
           double omega, std::uintptr_t stream_ptr) {
            const int B = (int)cost.shape(0);
            const int N = (int)cost.shape(1);
            const int M = (int)cost.shape(2);
            cudaStream_t stream = reinterpret_cast<cudaStream_t>(stream_ptr);
            sinkhorn::update_f_kernel<T, BLOCK><<<B * N, BLOCK, 0, stream>>>(
                cost.data(), mask_a.data(), mask_b.data(), g.data(), eps.data(),
                log_a.data(), conv.data(), f.data(), N, M,
                static_cast<T>(omega));
            CUDA_CHECK_LAST_ERROR(update_f_kernel);
        },
        "cost"_a, "mask_a"_a, "mask_b"_a, "g"_a, "eps"_a, "log_a"_a, "conv"_a,
        "f"_a, "omega"_a = 1.0, "stream"_a = 0);

    // Fused gather + pairwise distance cost build (replaces the cuBLAS
    // gather+GEMM). cost[b, i, j] = dist(emb[cidx_l[b, i]], emb[cidx_r[b, j]])
    // for the chosen ``metric`` (see distances::Metric). The 2-Wasserstein cost
    // uses SQEUCLIDEAN; the other policies are wired so callers can plug in
    // more metrics without a new kernel.
    m.def(
        "build_cost",
        [](gpu_array_c<const T, Device> emb,
           gpu_array_c<const int, Device> cidx_l,
           gpu_array_c<const int, Device> cidx_r, gpu_array_c<T, Device> cost,
           int metric, std::uintptr_t stream_ptr) {
            const int B = (int)cidx_l.shape(0);
            const int N = (int)cidx_l.shape(1);
            const int M = (int)cidx_r.shape(1);
            const int D = (int)emb.shape(1);
            cudaStream_t stream = reinterpret_cast<cudaStream_t>(stream_ptr);
            constexpr int TILE = 16;
            // Features are streamed in FEAT_TILE-wide chunks so shared memory
            // is 2 * TILE * FEAT_TILE * sizeof(T) (16 KB f32 / 32 KB f64),
            // independent of D. D <= FEAT_TILE (typical PCA latents) is a
            // single chunk; larger D (e.g. raw-feature layers) just adds
            // chunks.
            constexpr int FEAT_TILE = 128;
            dim3 grid(B, (N + TILE - 1) / TILE, (M + TILE - 1) / TILE);
            dim3 block(TILE, TILE);
            // +1 padded row stride (matches the kernel) to avoid bank
            // conflicts.
            size_t shmem = (size_t)2 * TILE * (FEAT_TILE + 1) * sizeof(T);
            switch (metric) {
                case distances::EUCLIDEAN:
                    sinkhorn::pairwise_cost_kernel<T, TILE, FEAT_TILE,
                                                   distances::Euclidean<T>>
                        <<<grid, block, shmem, stream>>>(
                            emb.data(), cidx_l.data(), cidx_r.data(),
                            cost.data(), N, M, D);
                    break;
                case distances::MANHATTAN:
                    sinkhorn::pairwise_cost_kernel<T, TILE, FEAT_TILE,
                                                   distances::Manhattan<T>>
                        <<<grid, block, shmem, stream>>>(
                            emb.data(), cidx_l.data(), cidx_r.data(),
                            cost.data(), N, M, D);
                    break;
                default:
                    sinkhorn::pairwise_cost_kernel<T, TILE, FEAT_TILE,
                                                   distances::SqEuclidean<T>>
                        <<<grid, block, shmem, stream>>>(
                            emb.data(), cidx_l.data(), cidx_r.data(),
                            cost.data(), N, M, D);
                    break;
            }
            CUDA_CHECK_LAST_ERROR(pairwise_cost_kernel);
        },
        "emb"_a, "cidx_l"_a, "cidx_r"_a, "cost"_a, "metric"_a, "stream"_a = 0);
}

template <typename Device>
void register_bindings(nb::module_& m) {
    def_kernels<double, Device>(m);
    def_kernels<float, Device>(m);
}

NB_MODULE(_sinkhorn_cuda, m) {
    REGISTER_GPU_BINDINGS(register_bindings, m);
}
