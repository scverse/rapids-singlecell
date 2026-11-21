#pragma once

#include <cuda_fp16.h>
#include <mma.h>

// Each 32-row slab of two 64-feature tiles is reduced independently. The scale
// and FP16 residual preserve small differences between nearby FP32 values.
namespace tensor_gram {
constexpr int features = 64;
constexpr int slab = 32;
constexpr int threads = 128;

__global__ void feature_range(const float* x, unsigned* ranges, int n, int p) {
    int lane = threadIdx.x % 32, warp = threadIdx.x / 32;
    int col = blockIdx.x * 32 + lane;
    unsigned hi = 0, neg_lo = 0;
    for (int row = blockIdx.y * 64 + warp;
         col < p && row < min(n, int((blockIdx.y + 1) * 64)); row += 4) {
        unsigned bits = __float_as_uint(fabsf(x[size_t(row) * p + col]));
        hi = max(hi, bits);
        if (bits != 0) neg_lo = max(neg_lo, 0x7fffffffU - bits);
    }
    __shared__ unsigned upper[4][32], lower[4][32];
    upper[warp][lane] = hi;
    lower[warp][lane] = neg_lo;
    __syncthreads();
    if (warp == 0 && col < p) {
        for (int w = 1; w < 4; ++w) {
            hi = max(hi, upper[w][lane]);
            neg_lo = max(neg_lo, lower[w][lane]);
        }
        atomicMax(ranges + col, hi);
        atomicMax(ranges + p + col, neg_lo);
    }
}

__device__ inline int scale_exponent(unsigned bits) {
    // Map the column maximum to [2^14, 2^15). The rounded high part
    // then has absolute error <= 8, so its residual scaled by 2^12 fits FP16.
    return bits && bits < 0x7f800000U ? ilogbf(__uint_as_float(bits)) - 14 : 0;
}

__global__ void split(const float* x, const unsigned* ranges, __half* hi,
                      __half* lo, int n, int p, int ld) {
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    if (col >= ld) return;
    float v = 0;
    if (row < n && col < p)
        v = scalbnf(x[size_t(row) * p + col], -scale_exponent(ranges[col]));
    __half h = __float2half_rn(v);
    size_t i = size_t(row) * ld + col;
    hi[i] = h;
    lo[i] = __float2half_rn((v - __half2float(h)) * 0x1p12f);
}

__device__ inline bool wide_range(const unsigned* ranges, int col, int p) {
    if (col >= p || ranges[col] == 0) return false;
    unsigned hi = ranges[col], lo = 0x7fffffffU - ranges[p + col];
    // Every nonzero high part is normal within a 28-exponent span. Thus
    // each omitted low-low product is at most 2^-22 of the original product.
    return hi >= 0x7f800000U || scale_exponent(hi) - scale_exponent(lo) > 28;
}

struct Panels {
    // Skew each row by eight halves to reduce shared-memory bank conflicts.
    __half a[slab][features + 8], b[slab][features + 8];
    __half da[slab][features + 8], db[slab][features + 8];
};

__global__ void upper(const float* x, const __half* hi, const __half* lo,
                      const unsigned* ranges, double* gram, int n, int p,
                      int ld) {
    using namespace nvcuda::wmma;
    int y = blockIdx.y * features, z = blockIdx.x * features;
    if (y > z) return;
    const int t = threadIdx.x;
    // Mark features needing the FP64 reduction below.
    __shared__ unsigned masks[4];
#if __CUDA_ARCH__ >= 700
    bool wide = wide_range(ranges, (t < features ? y : z) + t % features, p);
#else
    bool wide = true;
#endif
    unsigned mask = __ballot_sync(0xffffffffU, wide);
    if (t % 32 == 0) masks[t / 32] = mask;
    __syncthreads();

    __shared__ __align__(32) union {
        Panels panel;
        float out[features][features];
        double partial[threads];
    } shared;
#if __CUDA_ARCH__ >= 700
    int wy = (t / 64) * 32, wz = ((t / 32) % 2) * 32;
    using A = fragment<matrix_a, 16, 16, 16, __half, col_major>;
    using B = fragment<matrix_b, 16, 16, 16, __half, row_major>;
    using C = fragment<accumulator, 16, 16, 16, float>;
    C sum[2][2];
#pragma unroll
    for (int i = 0; i < 2; ++i)
        for (int j = 0; j < 2; ++j) fill_fragment(sum[i][j], 0.f);

    for (int k = 0; k < n; k += slab) {
#pragma unroll
        for (int u = t; u < slab * features / 8; u += threads) {
            int r = u / (features / 8), c = (u % (features / 8)) * 8;
            size_t off = size_t(k + r) * ld + c;
            *reinterpret_cast<uint4*>(&shared.panel.a[r][c]) =
                *reinterpret_cast<const uint4*>(hi + off + y);
            *reinterpret_cast<uint4*>(&shared.panel.da[r][c]) =
                *reinterpret_cast<const uint4*>(lo + off + y);
            *reinterpret_cast<uint4*>(&shared.panel.b[r][c]) =
                *reinterpret_cast<const uint4*>(hi + off + z);
            *reinterpret_cast<uint4*>(&shared.panel.db[r][c]) =
                *reinterpret_cast<const uint4*>(lo + off + z);
        }
        __syncthreads();
        C main[2][2], residual[2][2];
#pragma unroll
        for (int i = 0; i < 2; ++i)
            for (int j = 0; j < 2; ++j) {
                fill_fragment(main[i][j], 0.f);
                fill_fragment(residual[i][j], 0.f);
            }
#pragma unroll
        for (int q = 0; q < slab; q += 16) {
            A a[2], da[2];
            B b[2], db[2];
#pragma unroll
            for (int i = 0; i < 2; ++i) {
                int col = wy + i * 16;
                load_matrix_sync(a[i], &shared.panel.a[q][col], features + 8);
                load_matrix_sync(da[i], &shared.panel.da[q][col], features + 8);
                col = wz + i * 16;
                load_matrix_sync(b[i], &shared.panel.b[q][col], features + 8);
                load_matrix_sync(db[i], &shared.panel.db[q][col], features + 8);
            }
#pragma unroll
            for (int i = 0; i < 2; ++i)
                for (int j = 0; j < 2; ++j) {
                    mma_sync(main[i][j], a[i], b[j], main[i][j]);
                    mma_sync(residual[i][j], da[i], b[j], residual[i][j]);
                    mma_sync(residual[i][j], a[i], db[j], residual[i][j]);
                }
        }
#pragma unroll
        for (int i = 0; i < 2; ++i)
            for (int j = 0; j < 2; ++j)
                for (int f = 0; f < C::num_elements; ++f)
                    sum[i][j].x[f] +=
                        fmaf(residual[i][j].x[f], 0x1p-12f, main[i][j].x[f]);
        __syncthreads();
    }
#pragma unroll
    for (int i = 0; i < 2; ++i)
        for (int j = 0; j < 2; ++j)
            store_matrix_sync(&shared.out[wy + i * 16][wz + j * 16], sum[i][j],
                              features, mem_row_major);
    __syncthreads();
    for (int i = t; i < features * features; i += threads) {
        int a = y + i / features, b = z + i % features;
        bool exact =
            (masks[(i / features) / 32] & (1u << ((i / features) % 32))) ||
            (masks[2 + (i % features) / 32] & (1u << (i % 32)));
        if (a <= b && b < p && !exact) {
            int scale = scale_exponent(ranges[a]) + scale_exponent(ranges[b]);
            gram[size_t(a) * p + b] +=
                scalbn(double(shared.out[i / features][i % features]), scale);
        }
    }
#endif
    if (!(masks[0] | masks[1] | masks[2] | masks[3])) return;
    __syncthreads();
    for (int group = 0; group < 4; ++group) {
        for (unsigned bits = masks[group]; bits; bits &= bits - 1) {
            int fixed =
                (group < 2 ? y : z) + (group % 2) * 32 + __ffs(bits) - 1;
            int other = (group < 2 ? z : y) + t % features;
            int a = group < 2 ? fixed : other, b = group < 2 ? other : fixed;
            // A pair with two exceptional features was already visited by its
            // row.
            bool needed =
                a <= b && b < p &&
                (group < 2 || !(masks[(t % features) / 32] & (1u << (t % 32))));
            double partial = 0;
            if (needed) {
                for (int k = t / features; k < n; k += threads / features)
                    partial += double(x[size_t(k) * p + fixed]) *
                               x[size_t(k) * p + other];
            }
            shared.partial[t] = partial;
            __syncthreads();
            if (t < features && needed)
                gram[size_t(a) * p + b] +=
                    shared.partial[t] + shared.partial[t + features];
            __syncthreads();
        }
    }
}

template <typename Device>
void bind(nb::module_& m) {
    m.def("gram_tensorcore", [](gpu_array_c<const float, Device> x,
                                gpu_array_c<unsigned short, Device> planes,
                                gpu_array_c<unsigned, Device> ranges,
                                gpu_array_c<double, Device> gram,
                                std::uintptr_t stream) {
        if (x.ndim() != 2)
            throw std::invalid_argument("expected a 2D PCA tile");
        int n = x.shape(0), p = x.shape(1);
        int nr = (n + slab - 1) / slab * slab,
            ld = (p + features - 1) / features * features;
        if (n <= 0 || p <= 0 || planes.ndim() != 3 || planes.shape(0) != 2 ||
            planes.shape(1) < size_t(nr) || planes.shape(2) != size_t(ld) ||
            ranges.size() < 2 * size_t(p) || gram.ndim() != 2 ||
            gram.shape(0) != size_t(p) || gram.shape(1) != size_t(p))
            throw std::invalid_argument(
                "incompatible Tensor Core Gram workspace");
        cudaStream_t s = reinterpret_cast<cudaStream_t>(stream);
        cuda_check(cudaMemsetAsync(ranges.data(), 0,
                                   2 * size_t(p) * sizeof(unsigned), s),
                   "reset PCA feature ranges");
        feature_range<<<dim3((p + 31) / 32, (n + 63) / 64), threads, 0, s>>>(
            x.data(), ranges.data(), n, p);
        CUDA_CHECK_LAST_ERROR(feature_range);
        __half* hi = reinterpret_cast<__half*>(planes.data());
        __half* lo = hi + planes.shape(1) * ld;
        split<<<dim3((ld + 31) / 32, nr / 8), dim3(32, 8), 0, s>>>(
            x.data(), ranges.data(), hi, lo, n, p, ld);
        CUDA_CHECK_LAST_ERROR(split);
        upper<<<dim3(ld / features, ld / features), threads, 0, s>>>(
            x.data(), hi, lo, ranges.data(), gram.data(), n, p, ld);
        CUDA_CHECK_LAST_ERROR(upper);
    });
}
}  // namespace tensor_gram
