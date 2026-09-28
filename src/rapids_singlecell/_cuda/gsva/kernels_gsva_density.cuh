#pragma once

#include <algorithm>
#include <climits>
#include <cstdint>

#include "../nb_types.h"

namespace gsva_density {

constexpr int THREADS = 128;

__global__ void bandwidth_kernel(const float* mat, int nobs, int nvar,
                                 double* bandwidth) {
    const long long stride = (long long)blockDim.x * gridDim.x;
    for (long long i = (long long)blockIdx.x * blockDim.x + threadIdx.x;
         i < nvar; i += stride) {
        double mean = 0.0;
        for (long long row = 0; row < nobs; ++row) {
            mean += mat[row * nvar + i];
        }
        mean /= double(nobs);
        double variance = 0.0;
        for (long long row = 0; row < nobs; ++row) {
            const double delta = double(mat[row * nvar + i]) - mean;
            variance += delta * delta;
        }
        bandwidth[i] = sqrt(variance / double(nobs - 1)) / 4.0;
    }
}

__global__ void gaussian_density_32_kernel(const double* __restrict__ scaled,
                                           const double* __restrict__ cdf,
                                           int nobs, int nvar, int size,
                                           double* density) {
    const int stride = blockDim.x * gridDim.x;
    for (int idx = blockIdx.x * blockDim.x + threadIdx.x; idx < size;
         idx += stride) {
        const int col = idx % nvar;
        const double value = scaled[idx];
        double left = 0.0;
        for (int other = 0; other < nobs; ++other) {
            const double diff = value - scaled[other * nvar + col];
            if (diff > 10.0) {
                left += 1.0;
            } else if (diff >= -10.0) {
                const double item = cdf[(int)(fabs(diff) * 1000.0)];
                left += diff < 0.0 ? 1.0 - item : item;
            }
        }
        left /= double(nobs);
        density[idx] = -log((1.0 - left) / left);
    }
}

__global__ void gaussian_density_64_kernel(const double* scaled,
                                           const double* cdf, int nobs,
                                           int nvar, long long size,
                                           double* density) {
    const long long stride = (long long)blockDim.x * gridDim.x;
    for (long long i = (long long)blockIdx.x * blockDim.x + threadIdx.x;
         i < size; i += stride) {
        const long long row = i / nvar;
        const long long col = i - row * nvar;
        double left = 0.0;
        for (long long other = 0; other < nobs; ++other) {
            const double diff = scaled[i] - scaled[other * nvar + col];
            if (diff > 10.0) {
                left += 1.0;
            } else if (diff >= -10.0) {
                const double value =
                    cdf[(long long)(fabs(diff) / 10.0 * 10000.0)];
                left += diff < 0.0 ? 1.0 - value : value;
            }
        }
        left /= double(nobs);
        density[i] = -log((1.0 - left) / left);
    }
}

template <typename T>
__global__ void poisson_lookup_kernel(const T* mat, const double* cdfs,
                                      long long ncounts, long long nobs,
                                      long long nvar, long long size,
                                      double* density) {
    const long long stride = (long long)blockDim.x * gridDim.x;
    for (long long i = (long long)blockIdx.x * blockDim.x + threadIdx.x;
         i < size; i += stride) {
        const long long k = i / nvar;
        const long long col = i - k * nvar;
        double left = 0.0;
        for (long long other = 0; other < nobs; ++other) {
            const long long count = (long long)mat[other * nvar + col];
            left += cdfs[k * ncounts + count];
        }
        left /= double(nobs);
        density[i] = -log((1.0 - left) / left);
    }
}

template <typename T>
__global__ void poisson_gather_kernel(const T* mat, const double* lookup,
                                      long long nvar, long long size,
                                      double* density) {
    const long long stride = (long long)blockDim.x * gridDim.x;
    for (long long i = (long long)blockIdx.x * blockDim.x + threadIdx.x;
         i < size; i += stride) {
        const long long col = i % nvar;
        const long long k = (long long)mat[i];
        density[i] = lookup[k * nvar + col];
    }
}

__global__ void poisson_gather_tile_kernel(
    const long long* counts, const double* lookup, long long start,
    long long end, long long size, long long output_stride, double* density) {
    const long long stride = (long long)blockDim.x * gridDim.x;
    for (long long i = (long long)blockIdx.x * blockDim.x + threadIdx.x;
         i < size; i += stride) {
        const long long count = counts[i];
        if (count >= start && count < end) {
            density[i * output_stride] = lookup[count - start];
        }
    }
}

__global__ void ecdf_kernel(const float* mat, const float* sorted,
                            long long nobs, long long nvar, long long size,
                            float* density) {
    const long long stride = (long long)blockDim.x * gridDim.x;
    for (long long i = (long long)blockIdx.x * blockDim.x + threadIdx.x;
         i < size; i += stride) {
        const long long row = i / nvar;
        const long long col = i - row * nvar;
        const float value = mat[i];
        long long low = 0, high = nobs;
        while (low < high) {
            const long long middle = low + (high - low) / 2;
            if (value < sorted[middle * nvar + col]) {
                high = middle;
            } else {
                low = middle + 1;
            }
        }
        density[i] = float(low) / float(nobs);
    }
}

template <typename T, typename Device, int Dimensions>
using Array = nb::ndarray<T, Device, nb::ndim<Dimensions>, nb::c_contig>;

template <typename T, typename Device>
void register_poisson_bindings(nb::module_& m) {
    using namespace nb::literals;
    using Counts = gpu_array_c<const T, Device>;
    using Input = gpu_array_c<const double, Device>;
    using Output = gpu_array_c<double, Device>;
    m.def(
        "poisson_lookup",
        [](Counts counts, Array<const double, Device, 2> cdfs, Output out,
           long long nobs, long long nvar, std::uintptr_t stream) {
            nb_require(nobs > 0 && nvar > 0 && nobs <= LLONG_MAX / nvar &&
                           counts.size() == static_cast<size_t>(nobs * nvar),
                       "GSVA Poisson count shape mismatch");
            nb_require(cdfs.shape(1) > 0 && out.size() <= LLONG_MAX &&
                           out.size() % nvar == 0 &&
                           out.size() / nvar <= cdfs.shape(0),
                       "GSVA Poisson lookup shape mismatch");
            if (!out.size()) return;
            poisson_lookup_kernel<T>
                <<<strided_grid(out.size(), THREADS), THREADS, 0,
                   reinterpret_cast<cudaStream_t>(stream)>>>(
                    counts.data(), cdfs.data(), cdfs.shape(1), nobs, nvar,
                    out.size(), out.data());
            CUDA_CHECK_LAST_ERROR(poisson_lookup_kernel);
        },
        "counts"_a, "cdfs"_a, "out"_a, "nobs"_a, "nvar"_a, "stream"_a = 0);
    m.def(
        "poisson_gather",
        [](Counts counts, Input lookup, Output out, long long nvar,
           std::uintptr_t stream) {
            nb_require(nvar > 0 && counts.size() <= LLONG_MAX &&
                           counts.size() % nvar == 0 &&
                           out.size() == counts.size() &&
                           lookup.size() % nvar == 0 &&
                           (!counts.size() || lookup.size() > 0),
                       "GSVA Poisson gather shape mismatch");
            if (!out.size()) return;
            poisson_gather_kernel<T>
                <<<strided_grid(out.size(), THREADS), THREADS, 0,
                   reinterpret_cast<cudaStream_t>(stream)>>>(
                    counts.data(), lookup.data(), nvar, out.size(), out.data());
            CUDA_CHECK_LAST_ERROR(poisson_gather_kernel);
        },
        "counts"_a, "lookup"_a, "out"_a, "nvar"_a, "stream"_a = 0);
}

template <typename Device>
void register_bindings(nb::module_& m) {
    using namespace nb::literals;
    using Values = Array<const float, Device, 2>;
    using Density = Array<double, Device, 2>;
    m.def(
        "bandwidth",
        [](Values mat, Array<double, Device, 1> out, std::uintptr_t stream) {
            nb_require(mat.shape(0) <= INT_MAX && mat.shape(1) <= INT_MAX &&
                           out.size() == mat.shape(1),
                       "GSVA bandwidth shape mismatch");
            if (!out.size()) return;
            bandwidth_kernel<<<strided_grid(out.size(), THREADS), THREADS, 0,
                               reinterpret_cast<cudaStream_t>(stream)>>>(
                mat.data(), static_cast<int>(mat.shape(0)),
                static_cast<int>(mat.shape(1)), out.data());
            CUDA_CHECK_LAST_ERROR(bandwidth_kernel);
        },
        "mat"_a, "out"_a, "stream"_a = 0);
    m.def(
        "gaussian_density",
        [](Array<const double, Device, 2> scaled,
           Array<const double, Device, 1> cdf, Density out,
           std::uintptr_t stream) {
            nb_require(scaled.shape(0) <= INT_MAX &&
                           scaled.shape(1) <= INT_MAX &&
                           scaled.size() <= LLONG_MAX && cdf.size() >= 10001 &&
                           out.shape(0) == scaled.shape(0) &&
                           out.shape(1) == scaled.shape(1),
                       "GSVA Gaussian density shape mismatch");
            if (!out.size()) return;
            const int nobs = static_cast<int>(scaled.shape(0));
            const int nvar = static_cast<int>(scaled.shape(1));
            const auto cuda_stream = reinterpret_cast<cudaStream_t>(stream);
            if (scaled.size() <= INT_MAX) {
                const auto blocks = static_cast<unsigned>(std::min(
                    (scaled.size() + THREADS - 1) / THREADS, size_t(65535)));
                gaussian_density_32_kernel<<<blocks, THREADS, 0, cuda_stream>>>(
                    scaled.data(), cdf.data(), nobs, nvar,
                    static_cast<int>(scaled.size()), out.data());
                CUDA_CHECK_LAST_ERROR(gaussian_density_32_kernel);
            } else {
                gaussian_density_64_kernel<<<strided_grid(scaled.size(),
                                                          THREADS),
                                             THREADS, 0, cuda_stream>>>(
                    scaled.data(), cdf.data(), nobs, nvar, scaled.size(),
                    out.data());
                CUDA_CHECK_LAST_ERROR(gaussian_density_64_kernel);
            }
        },
        "scaled"_a, "cdf"_a, "out"_a, "stream"_a = 0);
    register_poisson_bindings<float, Device>(m);
    register_poisson_bindings<int, Device>(m);
    register_poisson_bindings<long long, Device>(m);
    m.def(
        "poisson_gather_tile",
        [](Array<const long long, Device, 1> counts,
           gpu_array_c<const double, Device> lookup, long long start,
           long long end, nb::ndarray<double, Device, nb::ndim<1>> out,
           std::uintptr_t stream) {
            nb_require(start >= 0 && end >= start &&
                           lookup.size() >= static_cast<size_t>(end - start) &&
                           out.size() == counts.size() &&
                           out.size() <= LLONG_MAX &&
                           (out.size() <= 1 || out.stride(0) != 0),
                       "GSVA Poisson tile shape mismatch");
            if (!out.size()) return;
            poisson_gather_tile_kernel<<<
                strided_grid(out.size(), THREADS), THREADS, 0,
                reinterpret_cast<cudaStream_t>(stream)>>>(
                counts.data(), lookup.data(), start, end, out.size(),
                out.stride(0), out.data());
            CUDA_CHECK_LAST_ERROR(poisson_gather_tile_kernel);
        },
        "counts"_a, "lookup"_a, "start"_a, "end"_a, "out"_a, "stream"_a = 0);
    m.def(
        "ecdf",
        [](Values mat, Values sorted, Array<float, Device, 2> out,
           std::uintptr_t stream) {
            nb_require(mat.size() <= LLONG_MAX &&
                           sorted.shape(0) == mat.shape(0) &&
                           sorted.shape(1) == mat.shape(1) &&
                           out.shape(0) == mat.shape(0) &&
                           out.shape(1) == mat.shape(1),
                       "GSVA ECDF shape mismatch");
            if (!out.size()) return;
            ecdf_kernel<<<strided_grid(out.size(), THREADS), THREADS, 0,
                          reinterpret_cast<cudaStream_t>(stream)>>>(
                mat.data(), sorted.data(), mat.shape(0), mat.shape(1),
                mat.size(), out.data());
            CUDA_CHECK_LAST_ERROR(ecdf_kernel);
        },
        "mat"_a, "sorted"_a, "out"_a, "stream"_a = 0);
}

}  // namespace gsva_density
