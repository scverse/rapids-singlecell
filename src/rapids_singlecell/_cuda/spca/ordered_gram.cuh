#pragma once

#include <cmath>
#include <type_traits>

namespace ordered_gram {
constexpr int warps = 2;
constexpr int max_columns = 2048;

template <typename I>
__device__ I lower_bound(const I* col, I left, I right, int k) {
    while (left < right) {
        I mid = left + (right - left) / 2;
        if (col[mid] < k)
            left = mid + 1;
        else
            right = mid;
    }
    return left;
}

template <typename I>
__global__ void mark_rows(const I* ptr, const I* cols, size_t n,
                          unsigned* masks, int stride) {
    using U = std::make_unsigned_t<I>;
    for (size_t row = size_t(blockIdx.x) * blockDim.y + threadIdx.y; row < n;
         row += size_t(gridDim.x) * blockDim.y)
        for (U k = U(ptr[row]) + threadIdx.x; k < U(ptr[row + 1]); k += 32)
            atomicOr(masks + size_t(cols[k]) * stride + row / 32,
                     1u << (row % 32));
}

template <typename T, typename I>
__global__ void upper(const I* row_ptr, const I* cols, const T* values,
                      const unsigned* masks, int stride, int words, T* out,
                      int p, int width, bool direct, bool accumulate) {
    extern __shared__ double workspace[];
    auto* sums = reinterpret_cast<T*>(workspace);
    int warp = threadIdx.x / 32, lane = threadIdx.x % 32;
    int anchor = direct ? blockIdx.x * (blockDim.x / 32) + warp : blockIdx.x;
    int part = blockIdx.y, panel = blockIdx.z * width;
    int low = max(anchor, panel), high = min(p, panel + width);
    if (low >= high) return;
    T* own;
    if (direct) {
        // Each output row in this partition has a single writer warp.
        own = out + (size_t(part) * p + anchor) * p;
    } else {
        own = sums + warp * width;
        for (int j = low + lane; j < high; j += 32) own[j - panel] = 0;
        __syncwarp();
    }
    int segments = direct ? gridDim.y : gridDim.y * warps;
    int segment = direct ? part : part * warps + warp;
    int first = (size_t(words) * segment) / segments;
    int last = (size_t(words) * (segment + 1)) / segments;
    using U = std::make_unsigned_t<I>;
    // Each lane consumes one bitmap word in ascending bit order. The warp
    // accumulates those rows in lane order on every iteration.
    masks += size_t(anchor) * stride;
    for (int page = first; page < last; page += 32) {
        int word_index = page + lane;
        unsigned word = word_index < last ? masks[word_index] : 0u;
        unsigned hits;
        while ((hits = __ballot_sync(0xffffffff, word != 0))) {
            I left = 0, right = 0;
            T value = 0;
            if (word) {
                size_t row = size_t(word_index) * 32 + (__ffs(word) - 1);
                word &= word - 1;
                right = row_ptr[row + 1];
                left = lower_bound(cols, row_ptr[row], right, anchor);
                value = values[left];
                if (width < p) {
                    left = lower_bound(cols, left, right, low);
                    right = lower_bound(cols, left, right, high);
                }
            }
            while (hits) {
                int source = __ffs(hits) - 1;
                hits &= hits - 1;
                I a = __shfl_sync(0xffffffff, left, source);
                I b = __shfl_sync(0xffffffff, right, source);
                T factor = __shfl_sync(0xffffffff, value, source);
                for (U k = U(a) + lane; k < U(b); k += 32) {
                    int j = cols[k];
                    own[j - panel] = fma(factor, values[k], own[j - panel]);
                }
                // Canonical CSR gives one writer per feature within a row.
                __syncwarp();
            }
        }
    }
    if (!direct) {
        __syncthreads();
        for (int j = low + threadIdx.x; j < high; j += blockDim.x) {
            size_t pos = (size_t(part) * p + anchor) * p + j;
            T sum = sums[j - panel] + sums[width + j - panel];
            out[pos] = accumulate ? out[pos] + sum : sum;
        }
    }
}

// The maximum exponent and inverted minimum exponent share a zero initializer.
template <typename T>
__global__ void bounds(const T* data, size_t n, unsigned* out) {
    constexpr int mask = sizeof(T) == 4 ? 255 : 2047;
    unsigned hi = 0, inverse = 0;
    for (size_t k = size_t(blockIdx.x) * blockDim.x + threadIdx.x; k < n;
         k += size_t(gridDim.x) * blockDim.x) {
        T v = data[k];
        if (!v) continue;
        unsigned e;
        if constexpr (sizeof(T) == 4)
            e = (__float_as_uint(v) >> 23) & mask;
        else
            e = (__double_as_longlong(v) >> 52) & mask;
        hi = max(hi, e);
        inverse = max(inverse, mask - e);
    }
    for (int s = 16; s > 0; s /= 2) {
        hi = max(hi, __shfl_down_sync(0xffffffff, hi, s));
        inverse = max(inverse, __shfl_down_sync(0xffffffff, inverse, s));
    }
    if (!(threadIdx.x % 32)) {
        atomicMax(out, hi);
        atomicMax(out + 1, inverse);
    }
}

template <typename T>
__device__ long long nearest(T x) {
    if constexpr (sizeof(T) == 4)
        return __float2ll_rn(x);
    else
        return __double2ll_rn(x);
}

__device__ size_t upper_offset(size_t row, size_t p) {
    return row * p - row * (row + 1) / 2;
}

template <typename T, typename I, int mode>
__global__ void accumulate(const I* ptr, const I* col, const T* data, size_t n,
                           int p, unsigned long long* bins, T scale, T radix) {
    size_t triangle = size_t(p) * (p + 1) / 2;
    for (size_t row = size_t(blockIdx.x) * blockDim.y + threadIdx.y; row < n;
         row += size_t(gridDim.x) * blockDim.y) {
        I end = ptr[row + 1];
        for (I a = ptr[row]; a < end; ++a) {
            size_t base = upper_offset(col[a], p);
            T x = data[a];
            using U = std::make_unsigned_t<I>;
            for (U b = U(a) + threadIdx.x; b < U(end); b += blockDim.x) {
                T product = x * data[b];
                T scaled = product * scale;
                size_t j = base + col[b];
                if constexpr (mode == 2) {
                    // The entire magnitude fits in an unsigned word. Under the
                    // row-count bound the net carry/borrow fits in a signed
                    // word.
                    auto magnitude = __double2ull_rn(fabs(double(scaled)));
                    if (!magnitude) continue;
                    bool negative = scaled < 0;
                    auto old =
                        atomicAdd(bins + j, negative ? -magnitude : magnitude);
                    if (negative && old < magnitude)
                        atomicAdd(bins + triangle + j, ~0ULL);
                    if (!negative && old > ~magnitude)
                        atomicAdd(bins + triangle + j, 1ULL);
                } else {
                    long long high = nearest(scaled);
                    if (high)
                        atomicAdd(bins + j,
                                  static_cast<unsigned long long>(high));
                    if constexpr (mode == 1) {
                        long long low = nearest((scaled - T(high)) * radix);
                        if (low)
                            atomicAdd(bins + triangle + j,
                                      static_cast<unsigned long long>(low));
                    }
                }
            }
        }
    }
}

template <typename T>
__global__ void unpack(const long long* bins, T* gram, int p, T unit,
                       int bits) {
    size_t triangle = size_t(p) * (p + 1) / 2;
    size_t col = size_t(blockIdx.x) * blockDim.x + threadIdx.x;
    if (col >= p) return;
    for (size_t row = size_t(blockIdx.y) * blockDim.y + threadIdx.y; row < p;
         row += size_t(gridDim.y) * blockDim.y) {
        T result = 0;
        if (row <= col) {
            size_t j = upper_offset(row, p) + col;
            // Recombine before rounding so cancellation across the two signed
            // bins and ties at the destination precision are handled exactly.
            T integer;
            if (bits == 64)
                integer = T(__int128(bins[triangle + j]) * (__int128(1) << 64) +
                            static_cast<unsigned long long>(bins[j]));
            else if (bits)
                integer =
                    T(__int128(bins[j]) * (1LL << bits) + bins[triangle + j]);
            else
                integer = T(bins[j]);
            result = integer * unit;
        }
        gram[row * p + col] = result;
    }
}

template <typename T, typename I, typename Device>
void bind(nb::module_& m) {
    if constexpr (sizeof(I) == 4) {
        m.def("gram_bounds",
              [](gpu_array_c<const T, Device> data,
                 gpu_array_c<unsigned, Device> limits, std::uintptr_t stream) {
                  nb_require(limits.ndim() == 1 && limits.size() == 2,
                             "gram_bounds: expected two exponent limits");
                  // The kernel updates the caller's zero-initialized maxima.
                  if (!data.size()) return;
                  int grid = min((data.size() + 255) / 256, size_t(2048));
                  bounds<<<grid, 256, 0, (cudaStream_t)stream>>>(
                      data.data(), data.size(), limits.data());
                  CUDA_CHECK_LAST_ERROR(bounds);
              });
    }
    m.def(
        "integer_gram",
        [](gpu_array_c<const I, Device> ptr, gpu_array_c<const I, Device> col,
           gpu_array_c<const T, Device> data,
           gpu_array_c<long long, Device> bins, gpu_array_c<T, Device> gram,
           int unit, int bits, std::uintptr_t stream) {
            nb_require(ptr.ndim() == 1 && ptr.size() > 0 && col.ndim() == 1 &&
                           data.ndim() == 1 && col.size() == data.size() &&
                           gram.ndim() == 2 && gram.shape(0) == gram.shape(1),
                       "integer_gram: invalid CSR arrays or Gram matrix");
            size_t n = ptr.size() - 1;
            int p = gram.shape(0);
            nb_require(bins.ndim() == 2 && bins.shape(0) == 1 + (bits != 0) &&
                           bins.shape(1) == size_t(p) * (p + 1) / 2,
                       "integer_gram: incompatible packed triangle bins");
            auto s = (cudaStream_t)stream;
            if (!p) return;
            if (!n) {
                cuda_check(cudaMemsetAsync(gram.data(), 0, gram.nbytes(), s),
                           "zero empty-row Gram matrix");
                return;
            }
            dim3 block = data.size() / n < 64 ? dim3(32, 4) : dim3(128);
            int grid = min((n + block.y - 1) / block.y, size_t(65535));
            int low_bits = bits == 64 ? 0 : bits;
            T scale = std::ldexp(1.0, -unit - low_bits),
              radix = std::ldexp(1.0, low_bits);
            auto counters = reinterpret_cast<unsigned long long*>(bins.data());
            auto launch = [&](auto mode) {
                accumulate<T, I, decltype(mode)::value>
                    <<<grid, block, 0, s>>>(ptr.data(), col.data(), data.data(),
                                            n, p, counters, scale, radix);
            };
            if (bits == 64)
                launch(std::integral_constant<int, 2>{});
            else if (bits)
                launch(std::integral_constant<int, 1>{});
            else
                launch(std::integral_constant<int, 0>{});
            CUDA_CHECK_LAST_ERROR(accumulate);
            dim3 shape((p + 31) / 32, min((p + 7) / 8, 65535));
            unpack<<<shape, dim3(32, 8), 0, s>>>(
                bins.data(), gram.data(), p, T(std::ldexp(1.0, unit)), bits);
            CUDA_CHECK_LAST_ERROR(unpack);
        });
    m.def("ordered_gram", [](gpu_array_c<const I, Device> row_ptr,
                             gpu_array_c<const I, Device> cols,
                             gpu_array_c<const T, Device> values,
                             gpu_array_c<unsigned, Device> masks,
                             gpu_array_c<T, Device> partial,
                             std::uintptr_t stream) {
        nb_require(row_ptr.ndim() == 1 && row_ptr.size() && cols.ndim() == 1 &&
                       values.ndim() == 1 && cols.size() == values.size(),
                   "ordered_gram: incompatible CSR arrays");
        nb_require(
            partial.ndim() == 3 && partial.shape(0) &&
                partial.shape(1) == partial.shape(2),
            "ordered_gram: expected a nonempty stack of square Gram matrices");
        size_t n = row_ptr.size() - 1;
        int p = partial.shape(1);
        if (!p) return;
        nb_require(!n || (masks.ndim() == 2 && masks.shape(0) == size_t(p) &&
                          masks.shape(1) > 0 &&
                          masks.shape(1) <= size_t(INT_MAX - 31)),
                   "ordered_gram: incompatible row-presence masks");
        auto s = (cudaStream_t)stream;
        cuda_check(cudaMemsetAsync(partial.data(), 0, partial.nbytes(), s),
                   "initialize sparse Gram matrix");
        if (!n) return;
        int stride = masks.shape(1);
        // One warp can update a wide output row directly when few of its
        // entries are touched by each CSR row.
        bool direct = p > max_columns && values.size() / n < 32;
        int width = direct ? p : min(p, max_columns);
        int shared_bytes = direct ? 0 : width * warps * sizeof(T);
        nb_require(shared_bytes <=
                       device_attribute(cudaDevAttrMaxSharedMemoryPerBlock),
                   "ordered_gram: shared memory exceeds the device limit");
        int anchors_per_block = direct ? 4 : 1;
        int threads = direct ? 32 * anchors_per_block : 32 * warps;
        dim3 grid((p + anchors_per_block - 1) / anchors_per_block,
                  partial.shape(0), (p + width - 1) / width);
        for (size_t start = 0; start < n; start += size_t(stride) * 32) {
            size_t count = min(n - start, size_t(stride) * 32);
            int words = (count + 31) / 32;
            cuda_check(cudaMemsetAsync(masks.data(), 0, masks.nbytes(), s),
                       "clear row-presence masks");
            mark_rows<<<strided_grid(count, 8), dim3(32, 8), 0, s>>>(
                row_ptr.data() + start, cols.data(), count, masks.data(),
                stride);
            CUDA_CHECK_LAST_ERROR(mark_rows);
            upper<<<grid, threads, shared_bytes, s>>>(
                row_ptr.data() + start, cols.data(), values.data(),
                masks.data(), stride, words, partial.data(), p, width, direct,
                start != 0);
            CUDA_CHECK_LAST_ERROR(upper);
        }
    });
}
}  // namespace ordered_gram
