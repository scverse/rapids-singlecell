#pragma once

#include <cmath>
#include <type_traits>

namespace ordered_gram {
using Word = unsigned long long;
constexpr int width = 256;
constexpr int warps = 8;

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

// Canonical CSR stores each gene at most once per row. Pack the offset of
// the first entry in each group of 32 genes alongside its presence bits.
template <typename I>
__global__ void group_index(const I* ptr, const I* cols, size_t rows,
                            int groups, Word* bounds) {
    size_t row = size_t(blockIdx.x) * blockDim.y + threadIdx.y;
    if (row >= rows) return;
    I start = ptr[row], end = ptr[row + 1];
    for (int g = threadIdx.x; g < groups; g += 32) {
        I left = start, right = end;
        while (left < right) {
            I mid = left + (right - left) / 2;
            if (cols[mid] / 32 < g)
                left = mid + 1;
            else
                right = mid;
        }
        I first = left;
        unsigned mask = 0;
        for (; left < end && cols[left] / 32 == g; ++left)
            mask |= 1u << (cols[left] % 32);
        bounds[size_t(g) * rows + row] = (Word(first - start) << 32) | mask;
    }
}

// A 64-bit product splits into 48 low bits and at most 16 high bits.
// Each warp sees at most 4,096 rows per batch, so both local sums fit.
enum Mode { floating, one_word, split48, split };

__device__ size_t upper_offset(size_t row, size_t p) {
    return row * p - row * (row + 1) / 2;
}

__device__ Word shift_mantissa(Word m, int shift) {
    if (shift >= 64 || shift <= -64) return 0;
    return shift < 0 ? m >> -shift : m << shift;
}

template <typename T>
__device__ Word mantissa(T v, int unit, int& shift, bool& negative) {
    if constexpr (sizeof(T) == 4) {
        unsigned raw = __float_as_uint(v);
        negative = raw >> 31;
        unsigned exponent = (raw >> 23) & 255;
        shift = int(exponent) - (127 + 23) - unit;
        return (raw << 1) == 0 ? 0 : (raw & 0x7fffff) | (1u << 23);
    } else {
        Word raw = __double_as_longlong(v);
        negative = raw >> 63;
        unsigned exponent = (raw >> 52) & 2047;
        shift = int(exponent) - (1023 + 52) - unit;
        return (raw << 1) == 0
                   ? 0
                   : (raw & ((Word(1) << 52) - 1)) | (Word(1) << 52);
    }
}

template <Mode mode, typename T, typename I>
__global__ void product(const I* ptr, const I* col, const T* data,
                        const Word* index, int rows, unsigned char* storage,
                        size_t triangle, int p, int unit, int bits,
                        bool accumulate) {
    using Low = std::conditional_t<mode == floating, T, Word>;
    using High = std::conditional_t<mode == split48, int, long long>;
    extern __shared__ double scratch[];
    int anchor = blockIdx.x, panel = blockIdx.z * width;
    int low = max(anchor, panel), high = min(p, panel + width);
    if (low >= high) return;
    int lane = threadIdx.x % 32, warp = threadIdx.x / 32;
    auto* low_words = reinterpret_cast<Low*>(scratch);
    auto* high_words = reinterpret_cast<High*>(low_words + warps * width);
    Low* sums = low_words + warp * width;
    High* high_sums = nullptr;
    if constexpr (mode >= split48) high_sums = high_words + warp * width;
    for (int j = low + lane; j < high; j += 32) {
        sums[j - panel] = 0;
        if constexpr (mode >= split48) high_sums[j - panel] = 0;
    }
    __syncwarp();

    unsigned bit = 1u << (anchor % 32);
    int part = blockIdx.y, serial = part * warps + warp;
    int segments = gridDim.y * warps;
    int begin = size_t(rows) * serial / segments;
    int end = size_t(rows) * (serial + 1) / segments;
    int radix = mode == split48 ? 48 : bits;
    // Visit matching rows in lane order so each warp owns its partial sums.
    for (int packet = begin; packet < end; packet += 32) {
        int row = packet + lane;
        Word info = row < end ? index[size_t(anchor / 32) * rows + row] : 0;
        unsigned hits = __ballot_sync(0xffffffff, (unsigned(info) & bit) != 0);
        I left = 0, right = 0;
        T value = 0;
        if (unsigned(info) & bit) {
            I base = ptr[row];
            I a = base + I(info >> 32) + __popc(unsigned(info) & (bit - 1));
            value = data[a];
            left = anchor >= panel
                       ? a
                       : base + I(index[size_t(panel / 32) * rows + row] >> 32);
            right = high == p
                        ? ptr[row + 1]
                        : base + I(index[size_t(high / 32) * rows + row] >> 32);
        }
        while (hits) {
            int source = __ffs(hits) - 1;
            hits &= hits - 1;
            I a = __shfl_sync(0xffffffff, left, source),
              b = __shfl_sync(0xffffffff, right, source);
            T factor = __shfl_sync(0xffffffff, value, source);
            using U = std::make_unsigned_t<I>;
            for (U k = U(a) + lane; k < U(b); k += 32) {
                int j = col[k] - panel;
                if constexpr (mode == floating) {
                    sums[j] = fma(factor, data[k], sums[j]);
                } else {
                    // Match the correctly rounded product in the input dtype.
                    T val = factor * data[k];
                    int shift;
                    bool negative;
                    Word m = mantissa(val, unit, shift, negative);
                    Word magnitude = shift_mantissa(m, shift);
                    if constexpr (mode >= split48) {
                        Word top;
                        if constexpr (mode == split48)
                            top = magnitude >> 48;
                        else
                            top = shift_mantissa(m, shift - radix);
                        high_sums[j] += negative ? -High(top) : High(top);
                        magnitude &= (Word(1) << radix) - 1;
                    }
                    sums[j] += negative ? -magnitude : magnitude;
                }
            }
            __syncwarp();
        }
    }
    __syncthreads();
    auto* partial = reinterpret_cast<Low*>(storage);
    size_t base =
        mode == floating ? size_t(anchor) * p : upper_offset(anchor, p);
    size_t entries = mode == floating ? size_t(p) * p : triangle;
    for (int j = low + threadIdx.x; j < high; j += blockDim.x) {
        size_t at = part * entries + base + j;
        if constexpr (mode < split48) {
            Low sum = 0;
#pragma unroll
            for (int w = 0; w < warps; ++w)
                sum += low_words[w * width + j - panel];
            partial[at] = accumulate ? partial[at] + sum : sum;
        } else {
            auto* upper = partial + gridDim.y * triangle;
            __int128 sum = 0;
            if (accumulate)
                sum = __int128((long long)upper[at]) * (__int128(1) << 64) +
                      partial[at];
#pragma unroll
            for (int w = 0; w < warps; ++w) {
                int i = w * width + j - panel;
                sum += __int128(high_words[i]) * (__int128(1) << radix) +
                       (long long)low_words[i];
            }
            partial[at] = Word(sum);
            upper[at] = Word(sum >> 64);
        }
    }
}

template <Mode mode, typename T>
__global__ void finish(const unsigned char* storage, T* out, size_t triangle,
                       int p, int parts, int unit) {
    size_t row = size_t(blockIdx.y) * blockDim.y + threadIdx.y;
    size_t col = size_t(blockIdx.x) * blockDim.x + threadIdx.x;
    if (col >= size_t(p)) return;
    for (; row < size_t(p); row += size_t(gridDim.y) * blockDim.y) {
        T result = 0;
        if (row <= col) {
            size_t j = upper_offset(row, p) + col;
            if constexpr (mode == floating) {
                auto* v = reinterpret_cast<const T*>(storage);
                for (int part = 0; part < parts; part++)
                    result += v[(size_t(part) * p + row) * p + col];
            } else {
                auto* v = reinterpret_cast<const Word*>(storage);
                __int128 total = 0;
                for (int part = 0; part < parts; part++) {
                    size_t i = part * triangle + j;
                    if constexpr (mode >= split48)
                        total += __int128((long long)v[i + parts * triangle]) *
                                     (__int128(1) << 64) +
                                 v[i];
                    else
                        total += (long long)v[i];
                }
                result = ldexp(T(total), unit);
            }
        }
        out[row * p + col] = result;
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
    m.def("ordered_gram", [](gpu_array_c<const I, Device> ptr,
                             gpu_array_c<const I, Device> col,
                             gpu_array_c<const T, Device> data,
                             gpu_array_c<Word, Device> index,
                             gpu_array_c<unsigned char, Device> partial,
                             gpu_array_c<T, Device> gram, int unit, int bits,
                             std::uintptr_t stream) {
        nb_require(ptr.ndim() == 1 && ptr.size() && col.ndim() == 1 &&
                       data.ndim() == 1 && col.size() == data.size() &&
                       gram.ndim() == 2 && gram.shape(0) == gram.shape(1) &&
                       gram.shape(0) <= size_t(INT_MAX),
                   "ordered_gram: invalid CSR arrays or Gram matrix");
        int p = gram.shape(0);
        size_t n = ptr.size() - 1;
        if (!p) return;
        auto s = (cudaStream_t)stream;
        if (!n || !data.size()) {
            cuda_check(cudaMemsetAsync(gram.data(), 0, gram.nbytes(), s),
                       "zero empty Gram");
            return;
        }
        int groups = (p - 1) / 32 + 1;
        size_t triangle = size_t(p) * (p + size_t(1)) / 2;
        size_t entries = bits < 0 ? size_t(p) * p : triangle;
        size_t item = bits < 0 ? sizeof(T) : (bits ? 16 : 8);
        size_t alignment = bits < 0 ? alignof(T) : alignof(Word);
        nb_require(
            ((bits >= -1 && bits <= 52) || bits == 64) && unit >= -4096 &&
                unit <= 4096 && index.ndim() == 2 &&
                index.shape(0) == size_t(groups) && index.shape(1) > 0 &&
                index.shape(1) <= 32768 && partial.ndim() == 2 &&
                partial.shape(0) > 0 && partial.shape(0) <= 8 &&
                reinterpret_cast<std::uintptr_t>(partial.data()) % alignment ==
                    0 &&
                partial.shape(1) / item == entries &&
                partial.shape(1) % item == 0,
            "ordered_gram: invalid mode or workspace");
        int chunk = index.shape(1), parts = partial.shape(0);
        auto launch = [&](auto kind) {
            constexpr Mode mode = decltype(kind)::value;
            int shared = warps * width *
                         (mode == floating   ? sizeof(T)
                          : mode == one_word ? 8
                          : mode == split48  ? 12
                                             : 16);
            for (size_t start = 0; start < n; start += chunk) {
                int rows = min(size_t(chunk), n - start);
                group_index<<<(rows + 7) / 8, dim3(32, 8), 0, s>>>(
                    ptr.data() + start, col.data(), size_t(rows), groups,
                    index.data());
                CUDA_CHECK_LAST_ERROR(group_index);
                product<mode><<<dim3(p, parts, (p - 1) / width + 1), 32 * warps,
                                shared, s>>>(
                    ptr.data() + start, col.data(), data.data(), index.data(),
                    rows, partial.data(), triangle, p, unit, bits, start != 0);
                CUDA_CHECK_LAST_ERROR(product);
            }
            dim3 grid((p - 1) / 32 + 1, min((p - 1) / 8 + 1, 65535));
            finish<mode><<<grid, dim3(32, 8), 0, s>>>(
                partial.data(), gram.data(), triangle, p, parts, unit);
            CUDA_CHECK_LAST_ERROR(finish);
        };
        if (bits < 0)
            launch(std::integral_constant<Mode, floating>{});
        else if (!bits)
            launch(std::integral_constant<Mode, one_word>{});
        else if (bits == 64)
            launch(std::integral_constant<Mode, split48>{});
        else
            launch(std::integral_constant<Mode, split>{});
    });
}
}  // namespace ordered_gram
