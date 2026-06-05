#include <cuda_runtime.h>

#include <stdexcept>
#include <string>

#include "../nb_types.h"

#include "kernels_mixscale.cuh"

using namespace nb::literals;

constexpr size_t DEFAULT_DYNAMIC_SMEM_LIMIT = 48 * 1024;

static inline void cuda_check_runtime(cudaError_t err, const char* what) {
    if (err != cudaSuccess) {
        throw std::runtime_error(std::string(what) +
                                 " failed: " + cudaGetErrorString(err));
    }
}

template <typename T>
static inline void launch_mixscale_project_score(
    const T* X, long long n_vars, const int* row_ids, const int* col_ids,
    const int* n_per_gene, const int* k_per_gene, const int* cell_offsets,
    const int* feat_offsets, const bool* is_guide, const bool* nt_in_all,
    int n_genes, int max_k, bool do_scale, T* pvec_scratch, T* scores_out,
    cudaStream_t stream) {
    if (n_genes == 0 || max_k == 0) return;
    dim3 block(MIXSCALE_THREADS);
    dim3 grid(n_genes);
    // Dynamic shared holds vec, col_mean, col_std (3 x k each).
    size_t dyn_shmem = (size_t)3 * max_k * sizeof(T);
    // Opt in to >48KB shared only when the dynamic buffers plus the static
    // reduction buffer exceed the default.
    cudaFuncAttributes attr{};
    cuda_check_runtime(
        cudaFuncGetAttributes(&attr, mixscale_project_score_kernel<T>),
        "cudaFuncGetAttributes(mixscale_project_score_kernel)");
    size_t total_shmem = attr.sharedSizeBytes + dyn_shmem;
    if (total_shmem > DEFAULT_DYNAMIC_SMEM_LIMIT) {
        int device = 0;
        cuda_check_runtime(cudaGetDevice(&device), "cudaGetDevice");
        int max_optin = 0;
        cuda_check_runtime(
            cudaDeviceGetAttribute(
                &max_optin, cudaDevAttrMaxSharedMemoryPerBlockOptin, device),
            "cudaDeviceGetAttribute(MaxSharedMemoryPerBlockOptin)");
        if (total_shmem > (size_t)max_optin) {
            throw std::runtime_error(
                "mixscale_project_score requires " +
                std::to_string(total_shmem) + " B of shared memory (max " +
                std::to_string(max_optin) + " B); too many DE genes per gene.");
        }
        cuda_check_runtime(
            cudaFuncSetAttribute(mixscale_project_score_kernel<T>,
                                 cudaFuncAttributeMaxDynamicSharedMemorySize,
                                 (int)dyn_shmem),
            "cudaFuncSetAttribute(mixscale_project_score_kernel)");
    }
    mixscale_project_score_kernel<T><<<grid, block, dyn_shmem, stream>>>(
        X, n_vars, row_ids, col_ids, n_per_gene, k_per_gene, cell_offsets,
        feat_offsets, is_guide, nt_in_all, n_genes, do_scale, pvec_scratch,
        scores_out);
    CUDA_CHECK_LAST_ERROR(mixscale_project_score_kernel);
}

template <typename T, typename Device>
void def_project_score(nb::module_& m) {
    m.def(
        "project_score",
        [](gpu_array_c<const T, Device> X, long long n_vars,
           gpu_array_c<const int, Device> row_ids,
           gpu_array_c<const int, Device> col_ids,
           gpu_array_c<const int, Device> n_per_gene,
           gpu_array_c<const int, Device> k_per_gene,
           gpu_array_c<const int, Device> cell_offsets,
           gpu_array_c<const int, Device> feat_offsets,
           gpu_array_c<const bool, Device> is_guide,
           gpu_array_c<const bool, Device> nt_in_all,
           gpu_array_c<T, Device> pvec_scratch,
           gpu_array_c<T, Device> scores_out, int n_genes, int max_k,
           bool do_scale, std::uintptr_t stream) {
            launch_mixscale_project_score<T>(
                X.data(), n_vars, row_ids.data(), col_ids.data(),
                n_per_gene.data(), k_per_gene.data(), cell_offsets.data(),
                feat_offsets.data(), is_guide.data(), nt_in_all.data(), n_genes,
                max_k, do_scale, pvec_scratch.data(), scores_out.data(),
                (cudaStream_t)stream);
        },
        "X"_a, "n_vars"_a, "row_ids"_a, "col_ids"_a, "n_per_gene"_a,
        "k_per_gene"_a, "cell_offsets"_a, "feat_offsets"_a, "is_guide"_a,
        "nt_in_all"_a, "pvec_scratch"_a, "scores_out"_a, nb::kw_only(),
        "n_genes"_a, "max_k"_a, "do_scale"_a, "stream"_a = 0);
}

template <typename Device>
void register_bindings(nb::module_& m) {
    def_project_score<float, Device>(m);
    def_project_score<double, Device>(m);
}

NB_MODULE(_mixscale_cuda, m) {
    m.doc() = "Mixscale continuous perturbation score (projection + z-score).";
    REGISTER_GPU_BINDINGS(register_bindings, m);
}
