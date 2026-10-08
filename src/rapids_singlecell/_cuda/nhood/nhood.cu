#include <cuda_runtime.h>
#include "../nb_types.h"

#include "kernels_nhood.cuh"

using namespace nb::literals;

static constexpr int NHOOD_BLOCK = 256;

template <typename Device>
void register_bindings(nb::module_& m) {
    m.def(
        "permuted_counts",
        [](gpu_array_c<const int, Device> rows,
           gpu_array_c<const int, Device> cols,
           gpu_array_c<const int, Device> labels,
           gpu_array_c<const int, Device> pos,
           gpu_array_c<const int, Device> group_off,
           gpu_array_c<const unsigned int, Device> keys,
           gpu_array_c<int, Device> buf,
           gpu_array_c<unsigned long long, Device> out, int k,
           std::uintptr_t stream) {
            const long long n = labels.shape(0);
            const long long nnz = rows.shape(0);
            const int n_batch = (int)keys.shape(0);
            const int n_groups = (int)group_off.shape(0) - 1;
            nb_require(cols.shape(0) == (size_t)nnz &&
                           pos.shape(0) == (size_t)n && n < (1LL << 31) &&
                           n_groups > 0,
                       "permuted_counts: inconsistent edge or cell arrays");
            nb_require(keys.ndim() == 2 && keys.shape(1) == FEISTEL_ROUNDS &&
                           n_batch > 0 && n_batch <= max_grid_dim_y(),
                       "permuted_counts: keys must be (n_batch, 24)");
            nb_require(
                buf.shape(0) >= (size_t)n_batch && buf.shape(1) == (size_t)n &&
                    out.shape(0) == (size_t)n_batch &&
                    out.shape(1) == (size_t)k && out.shape(2) == (size_t)k,
                "permuted_counts: buf/out shape mismatch");
            cudaStream_t s = (cudaStream_t)stream;

            dim3 pgrid(strided_grid(n, NHOOD_BLOCK), n_batch);
            permute_labels_kernel<<<pgrid, NHOOD_BLOCK, 0, s>>>(
                labels.data(), pos.data(), group_off.data(), n_groups,
                keys.data(), (int)n, buf.data());
            CUDA_CHECK_LAST_ERROR(permute_labels_kernel);
            if (nnz == 0) {
                return;
            }

            int device = 0, n_sm = 0, max_optin = 0;
            cuda_check(cudaGetDevice(&device), "cudaGetDevice");
            cuda_check(cudaDeviceGetAttribute(
                           &n_sm, cudaDevAttrMultiProcessorCount, device),
                       "cudaDeviceGetAttribute(MultiProcessorCount)");
            cuda_check(cudaDeviceGetAttribute(
                           &max_optin, cudaDevAttrMaxSharedMemoryPerBlockOptin,
                           device),
                       "cudaDeviceGetAttribute(MaxSharedMemoryPerBlockOptin)");
            // Fill the device across the whole batch.
            unsigned int nbx = strided_grid(nnz, NHOOD_BLOCK);
            unsigned int fill =
                (unsigned int)((8LL * n_sm + n_batch - 1) / n_batch);
            dim3 cgrid(nbx < fill ? nbx : fill, n_batch);
            // uint32 shared bins cannot overflow while nnz < 2^32.
            const size_t shmem = (size_t)k * k * sizeof(unsigned int);
            if (shmem <= (size_t)max_optin && nnz < (1LL << 32)) {
                cuda_check(cudaFuncSetAttribute(
                               nhood_count_kernel<true>,
                               cudaFuncAttributeMaxDynamicSharedMemorySize,
                               (int)shmem),
                           "cudaFuncSetAttribute(nhood_count_kernel)");
                nhood_count_kernel<true><<<cgrid, NHOOD_BLOCK, shmem, s>>>(
                    rows.data(), cols.data(), nnz, buf.data(), n, k,
                    out.data());
            } else {
                nhood_count_kernel<false><<<cgrid, NHOOD_BLOCK, 0, s>>>(
                    rows.data(), cols.data(), nnz, buf.data(), n, k,
                    out.data());
            }
            CUDA_CHECK_LAST_ERROR(nhood_count_kernel);
        },
        "rows"_a, "cols"_a, "labels"_a, "pos"_a, "group_off"_a, "keys"_a,
        nb::kw_only(), "buf"_a, "out"_a, "k"_a, "stream"_a = 0);
}

NB_MODULE(_nhood_cuda, m) {
    REGISTER_GPU_BINDINGS(register_bindings, m);
}
