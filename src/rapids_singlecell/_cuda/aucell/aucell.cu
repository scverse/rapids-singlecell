#include <cuda_runtime.h>
#include "../nb_types.h"

using namespace nb::literals;

__global__ void auc_kernel(const int* __restrict__ ranks, size_t R, size_t C,
                           const int* __restrict__ cnct,
                           const int* __restrict__ starts,
                           const int* __restrict__ lens, size_t n_sets,
                           int n_up, const float* __restrict__ max_aucs,
                           float* __restrict__ es) {
    for (size_t set = blockIdx.x; set < n_sets; set += gridDim.x) {
        const int start = starts[set];
        const int end = start + lens[set];
        const size_t row_stride = static_cast<size_t>(blockDim.x) * gridDim.y;

        for (size_t row =
                 static_cast<size_t>(blockIdx.y) * blockDim.x + threadIdx.x;
             row < R; row += row_stride) {
            int r = 0;
            int s = 0;

            for (int i = start; i < end; ++i) {
                const int g = cnct[i];
                const int rk = ranks[row * C + g];
                if (rk <= n_up) {
                    r += 1;
                    s += rk;
                }
            }
            const float val =
                (float)((static_cast<long long>(r) * n_up) - s) / max_aucs[set];
            es[row * n_sets + set] = val;
        }
    }
}

static inline void launch_auc(const int* ranks, size_t R, size_t C,
                              const int* cnct, const int* starts,
                              const int* lens, size_t n_sets, int n_up,
                              const float* max_aucs, float* es,
                              cudaStream_t stream) {
    constexpr int BLOCK_SIZE = 32;
    dim3 block(BLOCK_SIZE);
    dim3 grid(strided_grid(static_cast<long long>(n_sets), 1),
              strided_grid_y(static_cast<long long>(R), BLOCK_SIZE));
    auc_kernel<<<grid, block, 0, stream>>>(ranks, R, C, cnct, starts, lens,
                                           n_sets, n_up, max_aucs, es);
    CUDA_CHECK_LAST_ERROR(auc_kernel);
}

template <typename Device>
void register_bindings(nb::module_& m) {
    m.def(
        "auc",
        [](gpu_array_c<const int, Device> ranks, size_t R, size_t C,
           gpu_array_c<const int, Device> cnct,
           gpu_array_c<const int, Device> starts,
           gpu_array_c<const int, Device> lens, size_t n_sets, int n_up,
           gpu_array_c<const float, Device> max_aucs,
           gpu_array_c<float, Device> es, std::uintptr_t stream) {
            launch_auc(ranks.data(), R, C, cnct.data(), starts.data(),
                       lens.data(), n_sets, n_up, max_aucs.data(), es.data(),
                       (cudaStream_t)stream);
        },
        "ranks"_a, nb::kw_only(), "R"_a, "C"_a, "cnct"_a, "starts"_a, "lens"_a,
        "n_sets"_a, "n_up"_a, "max_aucs"_a, "es"_a, "stream"_a = 0);
}

NB_MODULE(_aucell_cuda, m) {
    REGISTER_GPU_BINDINGS(register_bindings, m);
}
