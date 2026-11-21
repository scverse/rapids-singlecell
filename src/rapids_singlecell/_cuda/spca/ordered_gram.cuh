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

template <typename T, typename I>
__global__ void upper(const I* row_ptr, const I* cols, const T* values,
                      const I* col_ptr, const I* rows, const T* transposed,
                      T* out, int p, int width) {
    extern __shared__ double workspace[];
    auto* sums = reinterpret_cast<T*>(workspace);
    int anchor = blockIdx.x, part = blockIdx.y, panel = blockIdx.z * width;
    int low = max(anchor, panel), high = min(p, panel + width);
    if (low >= high) return;
    int warp = threadIdx.x / 32, lane = threadIdx.x % 32;
    T* own = sums + warp * width;
    for (int j = low + lane; j < high; j += 32) own[j - panel] = 0;
    __syncwarp();

    I begin = col_ptr[anchor], count = col_ptr[anchor + 1] - begin;
    int segments = gridDim.y * warps, segment = part * warps + warp;
    I length = count / segments, extra = count % segments;
    I first = begin + segment * length + min(I(segment), extra);
    I last = first + length + (segment < extra);
    for (I page = first; page < last; page += 32) {
        I left = 0, right = 0;
        T value = 0;
        if (page + lane < last) {
            I r = rows[page + lane];
            left = row_ptr[r];
            right = row_ptr[r + 1];
            value = transposed[page + lane];
            if (width < p) {
                left = lower_bound(cols, left, right, low);
                right = lower_bound(cols, left, right, high);
            }
        }
        for (int q = 0; q < min(I(32), last - page); ++q) {
            I a = __shfl_sync(0xffffffff, left, q);
            I b = __shfl_sync(0xffffffff, right, q);
            T factor = __shfl_sync(0xffffffff, value, q);
            for (I k = a + lane; k < b; k += 32) {
                int j = cols[k];
                if (j >= low)
                    own[j - panel] = fma(factor, values[k], own[j - panel]);
            }
            // A canonical CSR row has one writer per feature. Finish its
            // updates before the next row, which may assign that feature to
            // another lane.
            __syncwarp();
        }
    }
    __syncthreads();
    for (int j = low + threadIdx.x; j < high; j += blockDim.x) {
        out[(size_t(part) * p + anchor) * p + j] =
            sums[j - panel] + sums[width + j - panel];
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
            for (I b = a + threadIdx.x; b < end; b += blockDim.x) {
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
            size_t n = ptr.size() - 1;
            int p = gram.shape(0);
            dim3 block = data.size() / n < 64 ? dim3(32, 4) : dim3(128);
            int grid = min((n + block.y - 1) / block.y, size_t(65535));
            int low_bits = bits == 64 ? 0 : bits;
            T scale = std::ldexp(1.0, -unit - low_bits),
              radix = std::ldexp(1.0, low_bits);
            auto counters = reinterpret_cast<unsigned long long*>(bins.data());
            auto s = (cudaStream_t)stream;
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
    m.def("ordered_gram",
          [](gpu_array_c<const I, Device> row_ptr,
             gpu_array_c<const I, Device> cols,
             gpu_array_c<const T, Device> values,
             gpu_array_c<const I, Device> col_ptr,
             gpu_array_c<const I, Device> rows,
             gpu_array_c<const T, Device> transposed,
             gpu_array_c<T, Device> partial, std::uintptr_t stream) {
              int p = col_ptr.size() - 1;
              if (!p) return;
              int width = min(p, max_columns);
              dim3 grid(p, partial.shape(0), (p + width - 1) / width);
              upper<<<grid, warps * 32, width * warps * sizeof(T),
                      (cudaStream_t)stream>>>(
                  row_ptr.data(), cols.data(), values.data(), col_ptr.data(),
                  rows.data(), transposed.data(), partial.data(), p, width);
              CUDA_CHECK_LAST_ERROR(upper);
          });
}
}  // namespace ordered_gram
