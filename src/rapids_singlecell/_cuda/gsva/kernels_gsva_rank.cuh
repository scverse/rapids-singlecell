#pragma once

#include <algorithm>
#include <cstdint>

#include "../nb_types.h"
#include "../gsea/kernels_gsea_rank.cuh"

namespace gsva_rank {

using Key = unsigned long long;

// Descending values, then ascending feature index. Reuse GSEA's cooperative
// sort while keeping only the inverse ranks in global memory.
__global__ void rank_kernel(const float* values, size_t n_rows, int n_columns,
                            int* ranks) {
    extern __shared__ std::uint16_t row_order[];
    __shared__ gsea_rank::Work work;
    for (size_t row = blockIdx.x; row < n_rows; row += gridDim.x) {
        const float* row_values = values + row * n_columns;
        for (int gene = threadIdx.x; gene < n_columns;
             gene += gsea_rank::THREADS) {
            row_order[gene] = std::uint16_t(gene);
        }
        __syncthreads();
        gsea_rank::sort_order<true>(row_values, row_order, n_columns, work);
        for (int position = threadIdx.x; position < n_columns;
             position += gsea_rank::THREADS) {
            ranks[row * n_columns + row_order[position]] = position + 1;
        }
        __syncthreads();
    }
}

__global__ void sparse_ecdf_keys_kernel(const float* values, const int* columns,
                                        long long nnz, Key* keys) {
    const long long stride = static_cast<long long>(blockDim.x) * gridDim.x;
    for (long long i =
             static_cast<long long>(blockIdx.x) * blockDim.x + threadIdx.x;
         i < nnz; i += stride) {
        // IEEE signed zeros represent the same ECDF value.
        const unsigned bits =
            values[i] == 0.0f ? 0u : __float_as_uint(values[i]);
        const unsigned ordered =
            bits ^ ((bits >> 31) ? 0xffffffffu : 0x80000000u);
        keys[i] = (static_cast<Key>(columns[i]) << 32) | ordered;
    }
}

__global__ void sparse_ecdf_ranks_kernel(const Key* __restrict__ keys,
                                         const long long* __restrict__ order,
                                         const long long* __restrict__ starts,
                                         const int* __restrict__ counts,
                                         long long nnz, long long nobs,
                                         int* ranks) {
    const long long stride = static_cast<long long>(blockDim.x) * gridDim.x;
    for (long long idx =
             static_cast<long long>(blockIdx.x) * blockDim.x + threadIdx.x;
         idx < nnz; idx += stride) {
        const Key key = keys[idx];
        const int col = int(key >> 32);
        long long right = idx + 1;
        if (right < nnz && keys[right] == key) {
            long long low = right;
            long long high = starts[col] + counts[col];
            while (low < high) {
                const long long middle = low + (high - low) / 2;
                if (keys[middle] <= key) {
                    low = middle + 1;
                } else {
                    high = middle;
                }
            }
            right = low;
        }
        long long rank = right - starts[col];
        if (unsigned(key) >= 0x80000000u) {
            rank += nobs - counts[col];
        }
        ranks[order[idx]] = int(rank);
    }
}

__global__ void sparse_base_keys_kernel(const int* zero_ranks, int nvar,
                                        long long nobs, Key* keys) {
    for (int gene = blockIdx.x * blockDim.x + threadIdx.x; gene < nvar;
         gene += blockDim.x * gridDim.x) {
        keys[gene] = (static_cast<Key>(nobs - zero_ranks[gene]) << 16) |
                     static_cast<Key>(gene);
    }
}

__device__ __forceinline__ int lower_bound_key(const Key* values, int size,
                                               Key key) {
    int low = 0, high = size;
    while (low < high) {
        const int middle = low + (high - low) / 2;
        if (values[middle] < key) {
            low = middle + 1;
        } else {
            high = middle;
        }
    }
    return low;
}

__device__ __forceinline__ int row_value(const int* indices, const int* values,
                                         long long start, long long end,
                                         int gene, const int* zero_ranks) {
    long long low = start, high = end;
    while (low < high) {
        const long long middle = low + (high - low) / 2;
        if (indices[middle] < gene) {
            low = middle + 1;
        } else {
            high = middle;
        }
    }
    return low < end && indices[low] == gene ? values[low] : zero_ranks[gene];
}

__device__ __forceinline__ void sort_keys(Key* keys, int capacity) {
    for (int width = 2; width <= capacity; width <<= 1) {
        for (int distance = width >> 1; distance; distance >>= 1) {
            for (int item = threadIdx.x; item < capacity; item += blockDim.x) {
                const int other = item ^ distance;
                if (other > item) {
                    const bool ascending = (item & width) == 0;
                    const auto left = keys[item];
                    const auto right = keys[other];
                    if ((left > right) == ascending) {
                        keys[item] = right;
                        keys[other] = left;
                    }
                }
            }
            __syncthreads();
        }
    }
}

__global__ void sparse_rank_kernel(
    const int* __restrict__ indptr, const int* __restrict__ indices,
    const int* __restrict__ values, const int* __restrict__ zero_ranks,
    const Key* __restrict__ base_keys, const int* __restrict__ target_genes,
    const int* __restrict__ row_ids, int nrows_group, int capacity, int nvar,
    int ntarget, long long nobs, int* dos, int* srs) {
    extern __shared__ Key keys[];
    for (int group = blockIdx.x; group < nrows_group; group += gridDim.x) {
        const int row = row_ids[group];
        const long long start = indptr[row];
        const int count = int(indptr[row + 1] - start);
        for (int item = threadIdx.x; item < capacity; item += blockDim.x) {
            if (item < count) {
                const int gene = indices[start + item];
                keys[item] =
                    (static_cast<Key>(nobs - values[start + item]) << 16) |
                    static_cast<Key>(gene);
            } else {
                keys[item] = 0xffffffffffffffffULL;
            }
        }
        __syncthreads();
        sort_keys(keys, capacity);
        for (int target = threadIdx.x; target < ntarget; target += blockDim.x) {
            const int gene = target_genes[target];
            const int value = row_value(indices, values, start, start + count,
                                        gene, zero_ranks);
            const Key key =
                (static_cast<Key>(nobs - value) << 16) | static_cast<Key>(gene);
            dos[static_cast<long long>(row) * ntarget + target] =
                lower_bound_key(keys, count, key);
        }
        __syncthreads();
        for (int item = threadIdx.x; item < capacity; item += blockDim.x) {
            if (item < count) {
                const int gene = indices[start + item];
                keys[item] = (static_cast<Key>(nobs - zero_ranks[gene]) << 16) |
                             static_cast<Key>(gene);
            } else {
                keys[item] = 0xffffffffffffffffULL;
            }
        }
        __syncthreads();
        sort_keys(keys, capacity);
        for (int target = threadIdx.x; target < ntarget; target += blockDim.x) {
            const int gene = target_genes[target];
            const int value = row_value(indices, values, start, start + count,
                                        gene, zero_ranks);
            const Key key =
                (static_cast<Key>(nobs - value) << 16) | static_cast<Key>(gene);
            const int rank =
                1 + lower_bound_key(base_keys, nvar, key) -
                lower_bound_key(keys, count, key) +
                dos[static_cast<long long>(row) * ntarget + target];
            dos[static_cast<long long>(row) * ntarget + target] = rank;
            srs[static_cast<long long>(row) * ntarget + target] =
                abs(2 * rank - nvar - 2) / 2;
        }
        __syncthreads();
    }
}

inline int shared_limit() {
    int device, limit;
    cuda_check(cudaGetDevice(&device), "GSVA rank device");
    cuda_check(cudaDeviceGetAttribute(
                   &limit, cudaDevAttrMaxSharedMemoryPerBlockOptin, device),
               "GSVA rank shared memory");
    return limit;
}

inline unsigned sparse_grid(size_t nwork, unsigned threads) {
    return static_cast<unsigned>(
        std::min((nwork + threads - 1) / threads, size_t(65535)));
}

template <typename T, typename Device, int Dimensions>
using Array = nb::ndarray<T, Device, nb::ndim<Dimensions>, nb::c_contig>;

template <typename Device>
void register_bindings(nb::module_& m) {
    using namespace nb::literals;
    using Indices = Array<const int, Device, 1>;
    using Keys = Array<const Key, Device, 1>;
    using LongIndices = Array<const long long, Device, 1>;
    using Output = Array<int, Device, 1>;
    using Matrix = Array<int, Device, 2>;
    using KeyOutput = Array<Key, Device, 1>;
    m.def(
        "rank",
        [](Array<const float, Device, 2> values, Matrix ranks,
           std::uintptr_t stream) {
            const size_t nobs = values.shape(0), nvar = values.shape(1);
            nb_require(nobs <= INT32_MAX && nvar <= 65536,
                       "GSVA rank dimensions exceed supported limits");
            nb_require(ranks.shape(0) == nobs && ranks.shape(1) == nvar,
                       "GSVA rank output shape mismatch");
            if (!nobs || !nvar) {
                return;
            }
            const size_t shared_bytes = sizeof(std::uint16_t) * nvar;
            cudaFuncAttributes attributes;
            cuda_check(cudaFuncGetAttributes(&attributes, rank_kernel),
                       "GSVA rank kernel attributes");
            nb_require(shared_bytes + attributes.sharedSizeBytes <=
                           size_t(shared_limit()),
                       "GSVA rank exceeds device shared memory");
            cuda_check(
                cudaFuncSetAttribute(
                    rank_kernel, cudaFuncAttributeMaxDynamicSharedMemorySize,
                    static_cast<int>(shared_bytes)),
                "GSVA rank shared limit");
            const auto grid =
                static_cast<unsigned>(std::min(nobs, size_t(max_grid_dim_x())));
            rank_kernel<<<grid, gsea_rank::THREADS, shared_bytes,
                          reinterpret_cast<cudaStream_t>(stream)>>>(
                values.data(), nobs, static_cast<int>(nvar), ranks.data());
            CUDA_CHECK_LAST_ERROR(rank_kernel);
        },
        "values"_a, "ranks"_a, "stream"_a = 0);
    m.def(
        "sparse_ecdf_keys",
        [](Array<const float, Device, 1> values, Indices columns,
           KeyOutput keys, std::uintptr_t stream) {
            nb_require(values.size() <= INT32_MAX,
                       "GSVA sparse values exceed int32 index range");
            nb_require(
                columns.size() == values.size() && keys.size() == values.size(),
                "GSVA sparse ECDF key shape mismatch");
            if (!values.size()) {
                return;
            }
            sparse_ecdf_keys_kernel<<<sparse_grid(values.size(), 128), 128, 0,
                                      reinterpret_cast<cudaStream_t>(stream)>>>(
                values.data(), columns.data(), values.size(), keys.data());
            CUDA_CHECK_LAST_ERROR(sparse_ecdf_keys_kernel);
        },
        "values"_a, "columns"_a, "keys"_a, "stream"_a = 0);
    m.def(
        "sparse_ecdf_ranks",
        [](Keys keys, LongIndices order, LongIndices starts, Indices counts,
           long long nobs, Output ranks, std::uintptr_t stream) {
            nb_require(nobs >= 0 && nobs <= INT32_MAX &&
                           keys.size() <= INT32_MAX && counts.size() <= 65536,
                       "GSVA sparse ECDF dimensions exceed supported limits");
            nb_require(order.size() == keys.size() &&
                           ranks.size() == keys.size() &&
                           starts.size() == counts.size() + 1,
                       "GSVA sparse ECDF rank shape mismatch");
            if (!keys.size()) {
                return;
            }
            nb_require(nobs > 0 && counts.size() > 0,
                       "GSVA nonempty ECDF values require rows and features");
            sparse_ecdf_ranks_kernel<<<sparse_grid(keys.size(), 256), 256, 0,
                                       reinterpret_cast<cudaStream_t>(
                                           stream)>>>(
                keys.data(), order.data(), starts.data(), counts.data(),
                keys.size(), nobs, ranks.data());
            CUDA_CHECK_LAST_ERROR(sparse_ecdf_ranks_kernel);
        },
        "keys"_a, "order"_a, "starts"_a, "counts"_a, "nobs"_a, "ranks"_a,
        "stream"_a = 0);
    m.def(
        "sparse_base_keys",
        [](Indices zero_ranks, long long nobs, KeyOutput keys,
           std::uintptr_t stream) {
            nb_require(
                nobs >= 0 && nobs <= INT32_MAX && zero_ranks.size() <= 65536,
                "GSVA sparse base-key dimensions exceed supported limits");
            nb_require(keys.size() == zero_ranks.size(),
                       "GSVA sparse base-key shape mismatch");
            if (!keys.size()) {
                return;
            }
            sparse_base_keys_kernel<<<sparse_grid(keys.size(), 128), 128, 0,
                                      reinterpret_cast<cudaStream_t>(stream)>>>(
                zero_ranks.data(), static_cast<int>(zero_ranks.size()), nobs,
                keys.data());
            CUDA_CHECK_LAST_ERROR(sparse_base_keys_kernel);
        },
        "zero_ranks"_a, "nobs"_a, "keys"_a, "stream"_a = 0);
    m.def(
        "sparse_rank",
        [](Indices indptr, Indices indices, Indices values, Indices zero_ranks,
           Keys base_keys, Indices target_genes, Indices row_ids, int capacity,
           Matrix dos, Matrix srs, std::uintptr_t stream) {
            const size_t nobs = dos.shape(0), ntarget = dos.shape(1);
            const size_t nvar = zero_ranks.size();
            nb_require(nobs <= INT32_MAX && nvar <= 65536 && ntarget <= nvar &&
                           indices.size() <= INT32_MAX &&
                           row_ids.size() <= nobs,
                       "GSVA sparse rank dimensions exceed supported limits");
            nb_require(
                capacity > 0 && capacity <= 8192 &&
                    (capacity & (capacity - 1)) == 0,
                "GSVA sparse rank capacity must be a power of two <= 8192");
            nb_require(indptr.size() == nobs + 1 &&
                           values.size() == indices.size() &&
                           base_keys.size() == nvar &&
                           target_genes.size() == ntarget &&
                           srs.shape(0) == nobs && srs.shape(1) == ntarget,
                       "GSVA sparse rank shape mismatch");
            if (!row_ids.size() || !ntarget) {
                return;
            }
            const size_t shared_bytes = sizeof(Key) * capacity;
            nb_require(shared_bytes <= size_t(shared_limit()),
                       "GSVA sparse rank exceeds device shared memory");
            cuda_check(cudaFuncSetAttribute(
                           sparse_rank_kernel,
                           cudaFuncAttributeMaxDynamicSharedMemorySize,
                           static_cast<int>(shared_bytes)),
                       "GSVA sparse rank shared limit");
            sparse_rank_kernel<<<sparse_grid(row_ids.size(), 1), 256,
                                 shared_bytes,
                                 reinterpret_cast<cudaStream_t>(stream)>>>(
                indptr.data(), indices.data(), values.data(), zero_ranks.data(),
                base_keys.data(), target_genes.data(), row_ids.data(),
                static_cast<int>(row_ids.size()), capacity,
                static_cast<int>(nvar), static_cast<int>(ntarget), nobs,
                dos.data(), srs.data());
            CUDA_CHECK_LAST_ERROR(sparse_rank_kernel);
        },
        "indptr"_a, "indices"_a, "values"_a, "zero_ranks"_a, "base_keys"_a,
        "target_genes"_a, "row_ids"_a, "capacity"_a, "dos"_a, "srs"_a,
        "stream"_a = 0);
}

}  // namespace gsva_rank
