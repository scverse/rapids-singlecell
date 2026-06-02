#include <cuda_runtime.h>

#include <cstdint>

#include "../nb_types.h"
#include "kernels_sinkhorn.cuh"

using namespace nb::literals;

namespace {
constexpr int SINKHORN_BLOCK = 256;
}  // namespace

// RAGGED (jagged) batched Sinkhorn bindings -- no padding, no masks. Pairs are
// stored flat with per-pair int64 offsets (cost_off / f_off / g_off) and int32
// sizes (n / m); row2pair / col2pair map flat rows / columns back to their
// pair. Each binding launches one kernel on the caller-supplied stream so the
// Python driver can dispatch work to several devices' streams.

template <typename T, typename Device>
static void def_kernels(nb::module_& m) {
    constexpr int BLOCK = SINKHORN_BLOCK;

    // eps[b] = scale * std(cost block b), floored.
    m.def(
        "auto_eps",
        [](gpu_array_c<const T, Device> cost,
           gpu_array_c<const int64_t, Device> cost_off,
           gpu_array_c<const int, Device> n, gpu_array_c<const int, Device> m,
           double scale, double floor, gpu_array_c<T, Device> eps,
           std::uintptr_t stream_ptr) {
            const int B = (int)n.shape(0);
            cudaStream_t stream = reinterpret_cast<cudaStream_t>(stream_ptr);
            sinkhorn::auto_eps_kernel<T, BLOCK><<<B, BLOCK, 0, stream>>>(
                cost.data(), cost_off.data(), n.data(), m.data(),
                static_cast<T>(scale), static_cast<T>(floor), eps.data());
            CUDA_CHECK_LAST_ERROR(auto_eps_kernel);
        },
        "cost"_a, "cost_off"_a, "n"_a, "m"_a, "scale"_a, "floor"_a, "eps"_a,
        "stream"_a = 0);

    // g update: one thread per flat column.
    m.def(
        "update_g",
        [](gpu_array_c<const T, Device> cost,
           gpu_array_c<const int64_t, Device> cost_off,
           gpu_array_c<const int, Device> n, gpu_array_c<const int, Device> m,
           gpu_array_c<const T, Device> f,
           gpu_array_c<const int64_t, Device> f_off, gpu_array_c<T, Device> g,
           gpu_array_c<const int64_t, Device> g_off,
           gpu_array_c<const int, Device> col2pair,
           gpu_array_c<const T, Device> eps, gpu_array_c<const T, Device> log_b,
           gpu_array_c<const int, Device> conv, double omega,
           std::uintptr_t stream_ptr) {
            const int64_t total_cols = (int64_t)col2pair.shape(0);
            cudaStream_t stream = reinterpret_cast<cudaStream_t>(stream_ptr);
            const unsigned int grid =
                (unsigned int)((total_cols + BLOCK - 1) / BLOCK);
            sinkhorn::update_g_kernel<T><<<grid, BLOCK, 0, stream>>>(
                cost.data(), cost_off.data(), n.data(), m.data(), f.data(),
                f_off.data(), g.data(), g_off.data(), col2pair.data(),
                eps.data(), log_b.data(), conv.data(), static_cast<T>(omega),
                total_cols);
            CUDA_CHECK_LAST_ERROR(update_g_kernel);
        },
        "cost"_a, "cost_off"_a, "n"_a, "m"_a, "f"_a, "f_off"_a, "g"_a,
        "g_off"_a, "col2pair"_a, "eps"_a, "log_b"_a, "conv"_a, "omega"_a = 1.0,
        "stream"_a = 0);

    // f update: one block per flat row.
    m.def(
        "update_f",
        [](gpu_array_c<const T, Device> cost,
           gpu_array_c<const int64_t, Device> cost_off,
           gpu_array_c<const int, Device> m, gpu_array_c<const T, Device> g,
           gpu_array_c<const int64_t, Device> g_off, gpu_array_c<T, Device> f,
           gpu_array_c<const int64_t, Device> f_off,
           gpu_array_c<const int, Device> row2pair,
           gpu_array_c<const T, Device> eps, gpu_array_c<const T, Device> log_a,
           gpu_array_c<const int, Device> conv, double omega,
           std::uintptr_t stream_ptr) {
            const int64_t total_rows = (int64_t)row2pair.shape(0);
            cudaStream_t stream = reinterpret_cast<cudaStream_t>(stream_ptr);
            sinkhorn::update_f_kernel<T, BLOCK>
                <<<(unsigned int)total_rows, BLOCK, 0, stream>>>(
                    cost.data(), cost_off.data(), m.data(), g.data(),
                    g_off.data(), f.data(), f_off.data(), row2pair.data(),
                    eps.data(), log_a.data(), conv.data(),
                    static_cast<T>(omega), total_rows);
            CUDA_CHECK_LAST_ERROR(update_f_kernel);
        },
        "cost"_a, "cost_off"_a, "m"_a, "g"_a, "g_off"_a, "f"_a, "f_off"_a,
        "row2pair"_a, "eps"_a, "log_a"_a, "conv"_a, "omega"_a = 1.0,
        "stream"_a = 0);

    // Per-pair convergence: one block per pair (no atomics), sets conv[b].
    m.def(
        "check_convergence",
        [](gpu_array_c<const T, Device> f, gpu_array_c<const T, Device> f_prev,
           gpu_array_c<const int64_t, Device> f_off,
           gpu_array_c<const int, Device> n, gpu_array_c<const T, Device> g,
           gpu_array_c<const T, Device> g_prev,
           gpu_array_c<const int64_t, Device> g_off,
           gpu_array_c<const int, Device> m_sizes, double tol,
           gpu_array_c<int, Device> conv, std::uintptr_t stream_ptr) {
            const int B = (int)n.shape(0);
            cudaStream_t stream = reinterpret_cast<cudaStream_t>(stream_ptr);
            sinkhorn::converged_kernel<T, BLOCK><<<B, BLOCK, 0, stream>>>(
                f.data(), f_prev.data(), f_off.data(), n.data(), g.data(),
                g_prev.data(), g_off.data(), m_sizes.data(),
                static_cast<T>(tol), conv.data());
            CUDA_CHECK_LAST_ERROR(converged_kernel);
        },
        "f"_a, "f_prev"_a, "f_off"_a, "n"_a, "g"_a, "g_prev"_a, "g_off"_a,
        "m"_a, "tol"_a, "conv"_a, "stream"_a = 0);

    // Fused gather + squared-Euclidean cost build over the flat tile schedule.
    m.def(
        "build_cost",
        [](gpu_array_c<const T, Device> emb,
           gpu_array_c<const int, Device> cidx_l,
           gpu_array_c<const int64_t, Device> f_off,
           gpu_array_c<const int, Device> cidx_r,
           gpu_array_c<const int64_t, Device> g_off,
           gpu_array_c<const int, Device> n, gpu_array_c<const int, Device> m,
           gpu_array_c<const int64_t, Device> cost_off,
           gpu_array_c<const int, Device> tile_pair,
           gpu_array_c<const int, Device> tile_i0,
           gpu_array_c<const int, Device> tile_j0, gpu_array_c<T, Device> cost,
           std::uintptr_t stream_ptr) {
            const int num_tiles = (int)tile_pair.shape(0);
            const int D = (int)emb.shape(1);
            cudaStream_t stream = reinterpret_cast<cudaStream_t>(stream_ptr);
            // shared = 2*TILE*(FEAT_TILE+1)*sizeof(T), independent of D; the +1
            // stride avoids bank conflicts (see kernel).
            constexpr int TILE = 16;
            constexpr int FEAT_TILE = 128;
            dim3 block(TILE, TILE);
            size_t shmem = (size_t)2 * TILE * (FEAT_TILE + 1) * sizeof(T);
            sinkhorn::pairwise_cost_kernel<T, TILE, FEAT_TILE>
                <<<(unsigned int)num_tiles, block, shmem, stream>>>(
                    emb.data(), cidx_l.data(), f_off.data(), cidx_r.data(),
                    g_off.data(), n.data(), m.data(), cost_off.data(),
                    tile_pair.data(), tile_i0.data(), tile_j0.data(),
                    cost.data(), D);
            CUDA_CHECK_LAST_ERROR(pairwise_cost_kernel);
        },
        "emb"_a, "cidx_l"_a, "f_off"_a, "cidx_r"_a, "g_off"_a, "n"_a, "m"_a,
        "cost_off"_a, "tile_pair"_a, "tile_i0"_a, "tile_j0"_a, "cost"_a,
        "stream"_a = 0);
}

template <typename Device>
void register_bindings(nb::module_& m) {
    def_kernels<double, Device>(m);
    def_kernels<float, Device>(m);
}

NB_MODULE(_sinkhorn_cuda, m) {
    REGISTER_GPU_BINDINGS(register_bindings, m);
}
