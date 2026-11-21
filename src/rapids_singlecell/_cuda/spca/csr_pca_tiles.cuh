#pragma once

// Each warp sums disjoint sorted CSR rows. Its private histogram is updated in
// row order, then the warp histograms are combined in a fixed order.
constexpr int PCA_COL_BLOCK = 512;

template <typename T, typename I>
__global__ void pca_column_sums(const I* indptr, const I* columns,
                                const T* data, double* partial, int genes,
                                size_t nrows) {
    constexpr int warps = 4;
    __shared__ double sums[warps][PCA_COL_BLOCK];
    int lane = threadIdx.x % 32, warp = threadIdx.x / 32;
    const int col_base = blockIdx.y * PCA_COL_BLOCK;
    for (int i = lane; i < PCA_COL_BLOCK; i += 32) sums[warp][i] = 0;
    __syncwarp();
    const size_t rows_per_block = (nrows + gridDim.x - 1) / gridDim.x;
    const size_t start_row = static_cast<size_t>(blockIdx.x) * rows_per_block;
    for (size_t r = start_row + warp;
         r < nrows && r < start_row + rows_per_block; r += warps) {
        I first = indptr[r], last = indptr[r + 1];
        if (lane == 0 && col_base) {
            I hi = last;
            while (first < hi) {
                I mid = first + (hi - first) / 2;
                if (columns[mid] < col_base)
                    first = mid + 1;
                else
                    hi = mid;
            }
        }
        first = __shfl_sync(0xffffffff, first, 0);
        for (size_t i = first; i < static_cast<size_t>(last); i += 32) {
            I col = (i + lane < static_cast<size_t>(last))
                        ? columns[i + lane] - col_base
                        : PCA_COL_BLOCK;
            bool in_block = col >= 0 && col < PCA_COL_BLOCK;
            if (!__any_sync(0xffffffff, in_block)) break;
            if (in_block)
                sums[warp][col] += static_cast<double>(data[i + lane]);
        }
        __syncwarp();
    }
    __syncthreads();
    for (int i = threadIdx.x; i < PCA_COL_BLOCK && col_base + i < genes;
         i += blockDim.x) {
        double sum = 0;
#pragma unroll
        for (int w = 0; w < warps; ++w) sum += sums[w][i];
        partial[static_cast<size_t>(blockIdx.x) * genes + col_base + i] = sum;
    }
}

template <typename T, typename I>
__global__ void pca_center_csr(const I* indptr, const I* columns, const T* data,
                               const double* mean, T* out, int genes) {
    const int row = blockIdx.x;
    T* dst = out + static_cast<size_t>(row) * genes;
    for (int j = threadIdx.x; j < genes; j += blockDim.x)
        dst[j] = static_cast<T>(-mean[j]);
    __syncthreads();
    for (size_t i = static_cast<size_t>(indptr[row]) + threadIdx.x;
         i < static_cast<size_t>(indptr[row + 1]); i += blockDim.x) {
        auto j = columns[i];
        dst[j] = static_cast<T>(static_cast<double>(data[i]) - mean[j]);
    }
}

template <typename T, typename I, typename Device>
void register_pca_tiles(nb::module_& m) {
    m.def("pca_column_sums",
          [](gpu_array_c<const I, Device> indptr,
             gpu_array_c<const I, Device> columns,
             gpu_array_c<const T, Device> data,
             gpu_array_c<double, Device> partial, std::uintptr_t stream) {
              size_t n = indptr.size() - 1;
              int genes = partial.shape(1);
              dim3 grid(partial.shape(0),
                        (genes + PCA_COL_BLOCK - 1) / PCA_COL_BLOCK);
              pca_column_sums<<<grid, 128, 0,
                                reinterpret_cast<cudaStream_t>(stream)>>>(
                  indptr.data(), columns.data(), data.data(), partial.data(),
                  genes, n);
              CUDA_CHECK_LAST_ERROR(pca_column_sums);
          });
    m.def("pca_center_csr",
          [](gpu_array_c<const I, Device> indptr,
             gpu_array_c<const I, Device> columns,
             gpu_array_c<const T, Device> data,
             gpu_array_c<const double, Device> mean, gpu_array_c<T, Device> out,
             std::uintptr_t stream) {
              pca_center_csr<<<out.shape(0), 128, 0,
                               reinterpret_cast<cudaStream_t>(stream)>>>(
                  indptr.data(), columns.data(), data.data(), mean.data(),
                  out.data(), out.shape(1));
              CUDA_CHECK_LAST_ERROR(pca_center_csr);
          });
}
