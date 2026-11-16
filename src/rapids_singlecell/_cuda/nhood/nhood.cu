#include <cub/device/device_radix_sort.cuh>
#include <cuda_runtime.h>
#include "../nb_types.h"
#include "../rmm_scratch.h"

#include <nanobind/stl/optional.h>

#include <algorithm>
#include <optional>

#include "kernels_nhood.cuh"

using namespace nb::literals;

static constexpr int NHOOD_BLOCK = 256;

// Count edges for a batch of label vectors: out[b] += pair counts of buf[b].
static void launch_count(const int* rows, const int* cols, long long nnz,
                         const int* buf, long long n, int k, int n_batch,
                         unsigned long long* out, int n_sm, int max_optin,
                         cudaStream_t s) {
    // Fill the device across the whole batch.
    unsigned int nbx = strided_grid(nnz, NHOOD_BLOCK);
    unsigned int fill = (unsigned int)((8LL * n_sm + n_batch - 1) / n_batch);
    dim3 grid(nbx < fill ? nbx : fill, n_batch);
    // uint32 shared bins cannot overflow while nnz < 2^32.
    const size_t shmem = (size_t)k * k * sizeof(unsigned int);
    if (shmem <= (size_t)max_optin && nnz < (1LL << 32)) {
        cuda_check(cudaFuncSetAttribute(
                       nhood_count_kernel<true>,
                       cudaFuncAttributeMaxDynamicSharedMemorySize, (int)shmem),
                   "cudaFuncSetAttribute(nhood_count_kernel)");
        nhood_count_kernel<true>
            <<<grid, NHOOD_BLOCK, shmem, s>>>(rows, cols, nnz, buf, n, k, out);
    } else {
        nhood_count_kernel<false>
            <<<grid, NHOOD_BLOCK, 0, s>>>(rows, cols, nnz, buf, n, k, out);
    }
    CUDA_CHECK_LAST_ERROR(nhood_count_kernel);
}

template <typename Device>
void register_bindings(nb::module_& m) {
    m.def(
        "permuted_counts",
        [](gpu_array_c<const int, Device> rows,
           gpu_array_c<const int, Device> cols,
           gpu_array_c<const int, Device> labels,
           gpu_array_c<const int, Device> pos,
           gpu_array_c<const int, Device> group_off,
           gpu_array_c<const unsigned long long, Device> seeds,
           gpu_array_c<unsigned long long, Device> out, int k, int batch,
           std::optional<gpu_array_c<int, Device>> labels_out,
           std::uintptr_t stream) {
            const long long n = labels.shape(0);
            const long long nnz = rows.shape(0);
            const long long n_perms = seeds.shape(0);
            const int n_groups = (int)group_off.shape(0) - 1;
            nb_require(cols.shape(0) == (size_t)nnz &&
                           pos.shape(0) == (size_t)n && n > 0 &&
                           n < (1LL << 31) && n_groups > 0,
                       "permuted_counts: inconsistent edge or cell arrays");
            nb_require(batch > 0 && batch <= max_grid_dim_y() &&
                           (long long)batch * n_groups < (1LL << 32),
                       "permuted_counts: batch out of range");
            nb_require(out.ndim() == 3 && out.shape(0) == (size_t)n_perms &&
                           out.shape(1) == (size_t)k &&
                           out.shape(2) == (size_t)k,
                       "permuted_counts: out must be (n_perms, k, k)");
            nb_require(
                !labels_out || (labels_out->shape(0) == (size_t)n_perms &&
                                labels_out->shape(1) == (size_t)n),
                "permuted_counts: labels_out must be (n_perms, n_cells)");
            cudaStream_t s = (cudaStream_t)stream;
            int device = 0, n_sm = 0, max_optin = 0;
            cuda_check(cudaGetDevice(&device), "cudaGetDevice");
            cuda_check(cudaDeviceGetAttribute(
                           &n_sm, cudaDevAttrMultiProcessorCount, device),
                       "cudaDeviceGetAttribute(MultiProcessorCount)");
            cuda_check(cudaDeviceGetAttribute(
                           &max_optin, cudaDevAttrMaxSharedMemoryPerBlockOptin,
                           device),
                       "cudaDeviceGetAttribute(MaxSharedMemoryPerBlockOptin)");

            // Scratch for one batch, reused by every batch.
            const long long cap = (long long)batch * n;
            const unsigned long long n_segs =
                (unsigned long long)batch * n_groups;
            const int seg_bits =
                n_segs > 1 ? 64 - __builtin_clzll(n_segs - 1) : 0;
            RmmScratchPool pool;
            int* buf = pool.alloc<int>(cap);
            cub::DoubleBuffer<unsigned long long> sort_keys(
                pool.alloc<unsigned long long>(cap),
                pool.alloc<unsigned long long>(cap));
            cub::DoubleBuffer<int> order(pool.alloc<int>(cap),
                                         pool.alloc<int>(cap));
            size_t bytes = 0;
            cuda_check(
                cub::DeviceRadixSort::SortPairs(
                    nullptr, bytes, sort_keys, order, cap, 0, 32 + seg_bits, s),
                "cub::DeviceRadixSort::SortPairs");
            void* temp = pool.alloc<char>(bytes);

            for (long long start = 0; start < n_perms; start += batch) {
                const int b = (int)std::min<long long>(batch, n_perms - start);
                const long long total = (long long)b * n;
                // Shuffle each group by sorting (permutation, group, random)
                // keys.
                dim3 kgrid(strided_grid(n, NHOOD_BLOCK), b);
                shuffle_keys_kernel<<<kgrid, NHOOD_BLOCK, 0, s>>>(
                    group_off.data(), n_groups, seeds.data() + start, (int)n,
                    sort_keys.Current(), order.Current());
                CUDA_CHECK_LAST_ERROR(shuffle_keys_kernel);
                cuda_check(cub::DeviceRadixSort::SortPairs(
                               temp, bytes, sort_keys, order, total, 0,
                               32 + seg_bits, s),
                           "cub::DeviceRadixSort::SortPairs");
                scatter_labels_kernel<<<strided_grid(total, NHOOD_BLOCK),
                                        NHOOD_BLOCK, 0, s>>>(
                    labels.data(), pos.data(), order.Current(), total, (int)n,
                    buf);
                CUDA_CHECK_LAST_ERROR(scatter_labels_kernel);
                if (labels_out) {
                    cuda_check(cudaMemcpyAsync(labels_out->data() + start * n,
                                               buf, total * sizeof(int),
                                               cudaMemcpyDeviceToDevice, s),
                               "cudaMemcpyAsync(labels_out)");
                }
                if (nnz > 0) {
                    launch_count(rows.data(), cols.data(), nnz, buf, n, k, b,
                                 out.data() + start * k * k, n_sm, max_optin,
                                 s);
                }
            }
        },
        "rows"_a, "cols"_a, "labels"_a, "pos"_a, "group_off"_a, "seeds"_a,
        nb::kw_only(), "out"_a, "k"_a, "batch"_a, "labels_out"_a = nb::none(),
        "stream"_a = 0);
}

NB_MODULE(_nhood_cuda, m) {
    register_scratch_allocator(m);
    REGISTER_GPU_BINDINGS(register_bindings, m);
}
