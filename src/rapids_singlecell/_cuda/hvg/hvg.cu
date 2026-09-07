#include <cuda_runtime.h>
#include "../minor_tiles.cuh"
#include "../nb_types.h"

using namespace nb::literals;

template <typename T>
__global__ void expected_zeros_kernel(const T* __restrict__ scaled_means,
                                      const T* __restrict__ total_counts,
                                      T* __restrict__ expected, int n_genes,
                                      int n_cells) {
    int gene = blockDim.x * blockIdx.x + threadIdx.x;
    if (gene >= n_genes) return;

    T sm = scaled_means[gene];
    T sum = T(0.0);

    for (int c = 0; c < n_cells; c++) {
        sum += exp(-sm * total_counts[c]);
    }

    expected[gene] = sum / T(n_cells);
}

template <typename T>
static void launch_expected_zeros(const T* scaled_means, const T* total_counts,
                                  T* expected, int n_genes, int n_cells,
                                  cudaStream_t stream) {
    constexpr int BLOCK_SIZE = 256;
    int grid_size = (n_genes + BLOCK_SIZE - 1) / BLOCK_SIZE;
    expected_zeros_kernel<T><<<grid_size, BLOCK_SIZE, 0, stream>>>(
        scaled_means, total_counts, expected, n_genes, n_cells);
    CUDA_CHECK_LAST_ERROR(expected_zeros_kernel);
}

/// min(value, clip) that propagates NaN like cupy.minimum on the dense path
/// (fmin would return the finite operand).
__device__ inline double clip_min(double value, double clip) {
    return (isnan(value) || isnan(clip)) ? NAN : fmin(value, clip);
}

/// Per-column sum and sum-of-squares of min(value, clip[col]) for seurat_v3
/// (see minor_tiles.cuh). Layout: double sq-sums, double sums, double clips.
template <typename T>
struct ClipSumOp {
    const T* data;
    const double* clip;
    double* sq_sum;
    double* sum;
    int tile_size;
    static constexpr size_t bytes_per_col = 3 * sizeof(double);
    static constexpr bool needs_rows = false;
    __device__ bool row_active(int) const {
        return true;
    }
    __device__ void zero_col(char* acc, int g, int col) const {
        double* s = reinterpret_cast<double*>(acc);
        s[g] = 0.0;
        s[tile_size + g] = 0.0;
        s[2 * tile_size + g] = clip[col];
    }
    __device__ void add(char* acc, long long q, int g) const {
        double* s = reinterpret_cast<double*>(acc);
        const double e =
            clip_min(static_cast<double>(data[q]), s[2 * tile_size + g]);
        atomicAdd(&s[g], e * e);
        atomicAdd(&s[tile_size + g], e);
    }
    __device__ void flush_col(const char* acc, int, int col, int g) const {
        const double* s = reinterpret_cast<const double*>(acc);
        if (s[g] != 0.0 || s[tile_size + g] != 0.0) {
            atomicAdd(&sq_sum[col], s[g]);
            atomicAdd(&sum[col], s[tile_size + g]);
        }
    }
    __device__ void add_global(long long q, int col, int) const {
        const double e = clip_min(static_cast<double>(data[q]), clip[col]);
        atomicAdd(&sq_sum[col], e * e);
        atomicAdd(&sum[col], e);
    }
    void zero_outputs(int minor, int, cudaStream_t stream) const {
        cuda_check(
            cudaMemsetAsync(sq_sum, 0, (size_t)minor * sizeof(double), stream),
            "cudaMemsetAsync(ClipSumOp outputs)");
        cuda_check(
            cudaMemsetAsync(sum, 0, (size_t)minor * sizeof(double), stream),
            "cudaMemsetAsync(ClipSumOp outputs)");
    }
};

// Returns whether unsorted rows were detected.
template <typename T, typename IdxT, typename Device>
void def_clip_square_sum(nb::module_& m) {
    m.def(
        "clip_square_sum",
        [](gpu_array_c<const IdxT, Device> indptr,
           gpu_array_c<const IdxT, Device> indices,
           gpu_array_c<const T, Device> data,
           gpu_array_c<const double, Device> clip_val,
           gpu_array_c<double, Device> sq_sum, gpu_array_c<double, Device> sum,
           bool assume_unsorted, std::uintptr_t stream) {
            require_csr_arrays("clip_square_sum", indptr, indices, data);
            require_arg(clip_val.shape(0) == sum.shape(0) &&
                            sq_sum.shape(0) == sum.shape(0),
                        "clip_square_sum: clip_val, sq_sum and sum must have "
                        "one entry per column");
            ClipSumOp<T> op{data.data(), clip_val.data(), sq_sum.data(),
                            sum.data(), 0};
            return minor_reduce<IdxT>(
                indptr.data(), indices.data(), op, (int)indptr.shape(0) - 1,
                (int)sum.shape(0), (long long)data.shape(0), assume_unsorted,
                (cudaStream_t)stream);
        },
        "indptr"_a, "indices"_a, "data"_a, nb::kw_only(), "clip_val"_a,
        "sq_sum"_a, "sum"_a, "assume_unsorted"_a = false, "stream"_a = 0);
}

template <typename T, typename Device>
void def_expected_zeros(nb::module_& m) {
    m.def(
        "expected_zeros",
        [](gpu_array_c<const T, Device> scaled_means,
           gpu_array_c<const T, Device> total_counts,
           gpu_array_c<T, Device> expected, int n_genes, int n_cells,
           std::uintptr_t stream) {
            launch_expected_zeros<T>(scaled_means.data(), total_counts.data(),
                                     expected.data(), n_genes, n_cells,
                                     reinterpret_cast<cudaStream_t>(stream));
        },
        "scaled_means"_a, "total_counts"_a, "expected"_a, "n_genes"_a,
        "n_cells"_a, "stream"_a = 0);
}

template <typename Device>
void register_bindings(nb::module_& m) {
    def_expected_zeros<float, Device>(m);
    def_expected_zeros<double, Device>(m);

    def_clip_square_sum<float, int, Device>(m);
    def_clip_square_sum<float, long long, Device>(m);
    def_clip_square_sum<double, int, Device>(m);
    def_clip_square_sum<double, long long, Device>(m);
}

NB_MODULE(_hvg_cuda, m) {
    REGISTER_GPU_BINDINGS(register_bindings, m);
    register_scratch_allocator(m);
}
