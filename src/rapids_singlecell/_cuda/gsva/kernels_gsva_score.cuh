#pragma once

#include "../nb_types.h"

#include <algorithm>
#include <climits>
#include <cmath>
#include <cstdint>

namespace gsva_score {

__global__ void score_kernel(const int* dos, const int* srs, const int* cnct,
                             const int* starts, const int* sizes,
                             const int* source_ids, long long nobs,
                             long long nfeat, long long nvar, long long nsrc,
                             long long ngroups, int maxdiff, int absrnk,
                             double tau, double* scores) {
    __shared__ double reduction[128];
    __shared__ double negative_reduction[128];
    for (long long task = blockIdx.x; task < nobs * ngroups;
         task += gridDim.x) {
        const long long row = task / ngroups;
        const int source = source_ids[task - row * ngroups];
        const int size = sizes[source];
        const int start = starts[source];
        double total = 0.0;
        for (int hit = threadIdx.x; hit < size; hit += blockDim.x) {
            const int gene = cnct[start + hit];
            const double rank = double(srs[row * nfeat + gene]);
            total += tau == 1.0 ? rank : pow(rank, tau);
        }
        reduction[threadIdx.x] = total;
        __syncthreads();
        for (int stride = blockDim.x / 2; stride; stride /= 2) {
            if (threadIdx.x < stride) {
                reduction[threadIdx.x] += reduction[threadIdx.x + stride];
            }
            __syncthreads();
        }
        total = reduction[0];
        if (total <= 0.0 || size >= nvar) {
            if (threadIdx.x == 0) {
                const double infinity =
                    __longlong_as_double(0x7ff0000000000000ULL);
                const double not_a_number =
                    __longlong_as_double(0x7ff8000000000000ULL);
                scores[row * nsrc + source] =
                    maxdiff ? (absrnk ? -infinity : not_a_number) : infinity;
            }
            __syncthreads();
            continue;
        }
        const double decrement = 1.0 / double(nvar - size);
        double positive = 0.0, negative = 0.0;
        for (int hit = threadIdx.x; hit < size; hit += blockDim.x) {
            const int gene = cnct[start + hit];
            const int position = dos[row * nfeat + gene] - 1;
            const double rank = double(srs[row * nfeat + gene]);
            const double weight = tau == 1.0 ? rank : pow(rank, tau);
            double cumulative = 0.0;
            int hit_count = 0;
            for (int other = 0; other < size; ++other) {
                const int other_gene = cnct[start + other];
                if (dos[row * nfeat + other_gene] - 1 <= position) {
                    const double other_rank =
                        double(srs[row * nfeat + other_gene]);
                    cumulative +=
                        tau == 1.0 ? other_rank : pow(other_rank, tau);
                    ++hit_count;
                }
            }
            const double missed =
                maxdiff
                    ? double(position - hit_count + 1) * decrement
                    : double(position - hit_count + 1) / double(nvar - size);
            positive = fmax(positive, cumulative / total - missed);
            negative = fmin(negative, (cumulative - weight) / total - missed);
        }
        reduction[threadIdx.x] = positive;
        negative_reduction[threadIdx.x] = negative;
        __syncthreads();
        for (int stride = blockDim.x / 2; stride; stride /= 2) {
            if (threadIdx.x < stride) {
                reduction[threadIdx.x] = fmax(reduction[threadIdx.x],
                                              reduction[threadIdx.x + stride]);
                negative_reduction[threadIdx.x] =
                    fmin(negative_reduction[threadIdx.x],
                         negative_reduction[threadIdx.x + stride]);
            }
            __syncthreads();
        }
        if (threadIdx.x == 0) {
            positive = reduction[0];
            negative = negative_reduction[0];
            scores[row * nsrc + source] =
                maxdiff
                    ? (absrnk ? positive - negative : positive + negative)
                    : (fabs(positive) > fabs(negative) ? positive : negative);
        }
        __syncthreads();
    }
}

__global__ void sorted_kernel(const int* dos, const int* srs, const int* cnct,
                              const int* starts, const int* sizes,
                              const int* source_ids, long long nobs,
                              long long nfeat, long long nvar, long long nsrc,
                              long long ngroups, int capacity, int maxdiff,
                              int absrnk, double tau, double* scores) {
    extern __shared__ unsigned char scratch[];
    __shared__ double positive_reduction[128];
    __shared__ double negative_reduction[128];
    unsigned long long* keys = reinterpret_cast<unsigned long long*>(scratch);
    double* buffer = reinterpret_cast<double*>(keys + capacity);
    int* positions = reinterpret_cast<int*>(buffer + capacity);
    for (long long task = blockIdx.x; task < nobs * ngroups;
         task += gridDim.x) {
        const long long row = task / ngroups;
        const int source = source_ids[task - row * ngroups];
        const int size = sizes[source];
        const int start = starts[source];
        for (int hit = threadIdx.x; hit < capacity; hit += blockDim.x) {
            if (hit < size) {
                const int gene = cnct[start + hit];
                const unsigned position = dos[row * nfeat + gene] - 1;
                keys[hit] = (static_cast<unsigned long long>(position) << 32) |
                            static_cast<unsigned>(gene);
            } else {
                keys[hit] = 0xffffffffffffffffULL;
            }
        }
        __syncthreads();
        for (int width = 2; width <= capacity; width <<= 1) {
            for (int distance = width >> 1; distance; distance >>= 1) {
                for (int hit = threadIdx.x; hit < capacity; hit += blockDim.x) {
                    const int other = hit ^ distance;
                    if (other > hit) {
                        const bool ascending = (hit & width) == 0;
                        const auto left = keys[hit];
                        const auto right = keys[other];
                        if ((left > right) == ascending) {
                            keys[hit] = right;
                            keys[other] = left;
                        }
                    }
                }
                __syncthreads();
            }
        }
        for (int hit = threadIdx.x; hit < capacity; hit += blockDim.x) {
            if (hit < size) {
                const auto key = keys[hit];
                positions[hit] = int(key >> 32);
                const int gene = int(unsigned(key));
                const double rank = double(srs[row * nfeat + gene]);
                buffer[hit] = tau == 1.0 ? rank : pow(rank, tau);
            } else {
                buffer[hit] = 0.0;
            }
        }
        __syncthreads();
        double* input = buffer;
        double* output = reinterpret_cast<double*>(keys);
        for (int offset = 1; offset < capacity; offset <<= 1) {
            for (int hit = threadIdx.x; hit < capacity; hit += blockDim.x) {
                output[hit] =
                    input[hit] + (hit >= offset ? input[hit - offset] : 0.0);
            }
            __syncthreads();
            double* temporary = input;
            input = output;
            output = temporary;
        }
        const double total = input[size - 1];
        if (total <= 0.0 || size >= nvar) {
            if (threadIdx.x == 0) {
                const double infinity =
                    __longlong_as_double(0x7ff0000000000000ULL);
                const double not_a_number =
                    __longlong_as_double(0x7ff8000000000000ULL);
                scores[row * nsrc + source] =
                    maxdiff ? (absrnk ? -infinity : not_a_number) : infinity;
            }
            __syncthreads();
            continue;
        }
        const double decrement = 1.0 / double(nvar - size);
        double positive = 0.0, negative = 0.0;
        for (int hit = threadIdx.x; hit < size; hit += blockDim.x) {
            const double missed =
                maxdiff ? double(positions[hit] - hit) * decrement
                        : double(positions[hit] - hit) / double(nvar - size);
            positive = fmax(positive, input[hit] / total - missed);
            const double previous = hit ? input[hit - 1] : 0.0;
            negative = fmin(negative, previous / total - missed);
        }
        positive_reduction[threadIdx.x] = positive;
        negative_reduction[threadIdx.x] = negative;
        __syncthreads();
        for (int stride = blockDim.x / 2; stride; stride >>= 1) {
            if (threadIdx.x < stride) {
                positive_reduction[threadIdx.x] =
                    fmax(positive_reduction[threadIdx.x],
                         positive_reduction[threadIdx.x + stride]);
                negative_reduction[threadIdx.x] =
                    fmin(negative_reduction[threadIdx.x],
                         negative_reduction[threadIdx.x + stride]);
            }
            __syncthreads();
        }
        if (threadIdx.x == 0) {
            positive = positive_reduction[0];
            negative = negative_reduction[0];
            scores[row * nsrc + source] =
                maxdiff
                    ? (absrnk ? positive - negative : positive + negative)
                    : (fabs(positive) > fabs(negative) ? positive : negative);
        }
        __syncthreads();
    }
}

__global__ void small_kernel(const int* dos, const int* srs, const int* cnct,
                             const int* starts, const int* sizes,
                             const int* source_ids, long long nobs,
                             long long nfeat, long long nvar, long long nsrc,
                             long long ngroups, int maxdiff, int absrnk,
                             double tau, double* scores) {
    const int lane = threadIdx.x & 31;
    const int warp = threadIdx.x >> 5;
    const long long warps_per_block = blockDim.x >> 5;
    const long long stride = (long long)gridDim.x * warps_per_block;
    for (long long task = (long long)blockIdx.x * warps_per_block + warp;
         task < nobs * ngroups; task += stride) {
        const long long row = task / ngroups;
        const int source = source_ids[task - row * ngroups];
        const int size = sizes[source];
        const int start = starts[source];
        int position = 0x7fffffff;
        double weight = 0.0;
        if (lane < size) {
            const int gene = cnct[start + lane];
            position = dos[row * nfeat + gene] - 1;
            const double rank = double(srs[row * nfeat + gene]);
            weight = tau == 1.0 ? rank : pow(rank, tau);
        }
        for (int width = 2; width <= 32; width <<= 1) {
            for (int distance = width >> 1; distance; distance >>= 1) {
                const int other_position =
                    __shfl_xor_sync(0xffffffffu, position, distance);
                const double other_weight =
                    __shfl_xor_sync(0xffffffffu, weight, distance);
                const bool ascending = (lane & width) == 0;
                const bool keep_minimum = ((lane & distance) == 0) == ascending;
                const bool take_other = keep_minimum
                                            ? other_position < position
                                            : other_position > position;
                if (take_other) {
                    position = other_position;
                    weight = other_weight;
                }
            }
        }
        double cumulative = weight;
        for (int offset = 1; offset < 32; offset <<= 1) {
            const double prefix =
                __shfl_up_sync(0xffffffffu, cumulative, offset);
            if (lane >= offset) cumulative += prefix;
        }
        const double total = __shfl_sync(0xffffffffu, cumulative, size - 1);
        if (total <= 0.0 || size >= nvar) {
            if (lane == 0) {
                const double infinity =
                    __longlong_as_double(0x7ff0000000000000ULL);
                const double not_a_number =
                    __longlong_as_double(0x7ff8000000000000ULL);
                scores[row * nsrc + source] =
                    maxdiff ? (absrnk ? -infinity : not_a_number) : infinity;
            }
            continue;
        }
        const double decrement = 1.0 / double(nvar - size);
        double positive = 0.0, negative = 0.0;
        if (lane < size) {
            // Signed extrema can change sign on a last-bit difference.
            // Preserve the reference division when comparing magnitudes.
            const double missed =
                maxdiff ? double(position - lane) * decrement
                        : double(position - lane) / double(nvar - size);
            positive = fmax(positive, cumulative / total - missed);
            negative = fmin(negative, (cumulative - weight) / total - missed);
        }
        for (int offset = 16; offset; offset >>= 1) {
            positive =
                fmax(positive, __shfl_down_sync(0xffffffffu, positive, offset));
            negative =
                fmin(negative, __shfl_down_sync(0xffffffffu, negative, offset));
        }
        if (lane == 0) {
            scores[row * nsrc + source] =
                maxdiff
                    ? (absrnk ? positive - negative : positive + negative)
                    : (fabs(positive) > fabs(negative) ? positive : negative);
        }
    }
}

template <typename T, typename Device, int Dimensions>
using Array = nb::ndarray<T, Device, nb::ndim<Dimensions>, nb::c_contig>;

template <typename Device>
void register_bindings(nb::module_& m) {
    using namespace nb::literals;
    using RankMatrix = Array<const int, Device, 2>;
    using Indices = Array<const int, Device, 1>;
    using Scores = Array<double, Device, 2>;
    m.def(
        "score",
        [](RankMatrix dos, RankMatrix srs, Indices cnct, Indices starts,
           Indices sizes, Indices source_ids, Scores scores, long long nvar,
           int capacity, bool maxdiff, bool absrnk, double tau,
           std::uintptr_t stream) {
            const size_t nobs = dos.shape(0), nfeat = dos.shape(1);
            const size_t nsrc = starts.size(), ngroups = source_ids.size();
            nb_require(srs.shape(0) == nobs && srs.shape(1) == nfeat &&
                           sizes.size() == nsrc && scores.shape(0) == nobs &&
                           scores.shape(1) == nsrc,
                       "GSVA score input/output shapes do not match");
            nb_require(nvar > 0 && nvar <= INT32_MAX &&
                           nfeat <= static_cast<size_t>(nvar) &&
                           nsrc <= INT32_MAX && cnct.size() <= INT32_MAX &&
                           ngroups <= nsrc,
                       "Invalid GSVA score feature or network dimensions");
            nb_require(capacity == 0 || (capacity >= 32 && capacity <= 2048 &&
                                         (capacity & (capacity - 1)) == 0),
                       "GSVA score capacity must be zero or a power of two "
                       "between 32 and 2048");
            nb_require(
                nobs <= INT64_MAX && (!ngroups || nobs <= INT64_MAX / ngroups),
                "GSVA score task dimensions exceed addressable memory");
            if (!nobs || !ngroups) {
                return;
            }
            const auto cuda_stream = reinterpret_cast<cudaStream_t>(stream);
            const long long ntasks = static_cast<long long>(nobs * ngroups);
            const long long nrows = static_cast<long long>(nobs);
            const long long nfeatures = static_cast<long long>(nfeat);
            const long long nsources = static_cast<long long>(nsrc);
            const long long ng = static_cast<long long>(ngroups);
            const long long nblocks =
                capacity == 32 ? 1 + (ntasks - 1) / 4 : ntasks;
            const auto grid = static_cast<unsigned>(std::min(nblocks, 65535LL));
            if (capacity == 32) {
                small_kernel<<<grid, 128, 0, cuda_stream>>>(
                    dos.data(), srs.data(), cnct.data(), starts.data(),
                    sizes.data(), source_ids.data(), nrows, nfeatures, nvar,
                    nsources, ng, maxdiff, absrnk, tau, scores.data());
                CUDA_CHECK_LAST_ERROR(gsva_score_small);
            } else if (capacity) {
                const int threads =
                    capacity <= 128 ? 32 : (capacity == 256 ? 64 : 128);
                const size_t shared_bytes =
                    static_cast<size_t>(capacity) *
                    (sizeof(unsigned long long) + sizeof(double) + sizeof(int));
                sorted_kernel<<<grid, threads, shared_bytes, cuda_stream>>>(
                    dos.data(), srs.data(), cnct.data(), starts.data(),
                    sizes.data(), source_ids.data(), nrows, nfeatures, nvar,
                    nsources, ng, capacity, maxdiff, absrnk, tau,
                    scores.data());
                CUDA_CHECK_LAST_ERROR(gsva_score_sorted);
            } else {
                score_kernel<<<grid, 128, 0, cuda_stream>>>(
                    dos.data(), srs.data(), cnct.data(), starts.data(),
                    sizes.data(), source_ids.data(), nrows, nfeatures, nvar,
                    nsources, ng, maxdiff, absrnk, tau, scores.data());
                CUDA_CHECK_LAST_ERROR(gsva_score);
            }
        },
        "dos"_a, "srs"_a, "cnct"_a, "starts"_a, "sizes"_a, "source_ids"_a,
        "scores"_a, "nvar"_a, "capacity"_a, "maxdiff"_a, "absrnk"_a, "tau"_a,
        "stream"_a = 0);
}

}  // namespace gsva_score
