// VIPER calculations adapted from decoupler 2.2.0; see LICENSE-decoupler.
#pragma once

#include <cuda_runtime.h>
#include <cfloat>

namespace viper {

__device__ double zero_cancellation(double sum, double absolute,
                                    long long nvar) {
    // Include rank-quantile rounding near the inverse-normal tails as well as
    // summation error. Do not amplify an unresolved sign with unsigned signal.
    return isfinite(sum) && isfinite(absolute) &&
                   fabs(sum) <= 8 * DBL_EPSILON * nvar * absolute
               ? 0.0
               : sum;
}

__global__ void unsigned_quantiles(const double* quantiles, double* magnitude,
                                   long long nwork) {
    for (long long i =
             static_cast<long long>(blockIdx.x) * blockDim.x + threadIdx.x;
         i < nwork; i += static_cast<long long>(blockDim.x) * gridDim.x) {
        magnitude[i] = fabs(quantiles[i] - 0.5) * 2;
    }
}

template <typename T>
__global__ void average_ranks(const T* mat, const T* sorted_values,
                              double* ranks, long long nvar, long long nwork) {
    for (long long i =
             static_cast<long long>(blockIdx.x) * blockDim.x + threadIdx.x;
         i < nwork; i += static_cast<long long>(blockDim.x) * gridDim.x) {
        const T value = mat[i];
        const long long offset = (i / nvar) * nvar;
        long long lo = 0, hi = nvar;
        while (lo < hi) {
            const long long mid = (lo + hi) / 2;
            if (sorted_values[offset + mid] < value)
                lo = mid + 1;
            else
                hi = mid;
        }
        const long long first = lo;
        hi = nvar;
        while (lo < hi) {
            const long long mid = (lo + hi) / 2;
            if (sorted_values[offset + mid] <= value)
                lo = mid + 1;
            else
                hi = mid;
        }
        ranks[i] = (first + lo + 1) * 0.5;
    }
}

__global__ void initial_score(const double* zsigned, const double* magnitude,
                              const double* weights, const long long* targets,
                              const long long* starts, double* sum1_out,
                              double* sum2_out, long long nvar, long long nsrc,
                              long long nwork) {
    for (long long i =
             static_cast<long long>(blockIdx.x) * blockDim.x + threadIdx.x;
         i < nwork; i += static_cast<long long>(blockDim.x) * gridDim.x) {
        const long long obs = i / nsrc, source = i % nsrc;
        const double w = 1.0 / (starts[source + 1] - starts[source]);
        double sum1 = 0, sum2 = 0, absolute = 0;
        for (long long e = starts[source]; e < starts[source + 1]; ++e) {
            const long long feature = obs * nvar + targets[e];
            const double term = zsigned[feature] * (w * weights[e]);
            sum1 += term;
            sum2 += magnitude[feature] * ((1 - fabs(weights[e])) * w);
            absolute += fabs(term);
        }
        sum1_out[i] = sum2 > 0 ? zero_cancellation(sum1, absolute, nvar) : sum1;
        sum2_out[i] = sum2;
    }
}

template <typename T>
__global__ void gather_overlap(const T* mat, const long long* targets,
                               const long long* starts, const long long* sizes,
                               const long long* rows, const long long* sources,
                               T* values, long long nvar, long long width,
                               long long nwork) {
    for (long long i =
             static_cast<long long>(blockIdx.x) * blockDim.x + threadIdx.x;
         i < nwork; i += static_cast<long long>(blockDim.x) * gridDim.x) {
        const long long task = i / width, k = i % width;
        const long long source = sources[task];
        const bool present = k < sizes[source];
        const long long target = present ? targets[starts[source] + k] : 0;
        values[i] = present ? mat[rows[task] * nvar + target] : (1.0 / 0.0);
    }
}

__global__ void transform_overlap(const double* rank, const long long* sizes,
                                  const long long* rows,
                                  const long long* sources, const double* nes,
                                  double* tail, double* signed_value,
                                  long long nsrc, long long width,
                                  long long nwork) {
    for (long long i =
             static_cast<long long>(blockIdx.x) * blockDim.x + threadIdx.x;
         i < nwork; i += static_cast<long long>(blockDim.x) * gridDim.x) {
        const long long task = i / width, source = sources[task];
        const double q = rank[i] / (sizes[source] + 1);
        tail[i] =
            i % width < sizes[source] ? fabs(q * 2 - 1) * 2 - 1 : -(1.0 / 0.0);
        const double score = nes[rows[task] * nsrc + source];
        const double sign = score == 0 ? 0 : copysign(1.0, score);
        signed_value[i] = i % width < sizes[source]
                              ? normcdfinv((q * 2 - 1) / 2 + 0.5) * sign
                              : 0;
    }
}

__global__ void magnitude_overlap(const double* tail, const double* maximum,
                                  double* magnitude, long long width,
                                  long long nwork) {
    for (long long i =
             static_cast<long long>(blockIdx.x) * blockDim.x + threadIdx.x;
         i < nwork; i += static_cast<long long>(blockDim.x) * gridDim.x) {
        magnitude[i] =
            normcdfinv((tail[i] + (1 - maximum[i / width]) / 2) / 2 + 0.5);
    }
}

template <typename T>
__global__ void overlap_score(const T* net, const long long* targets,
                              const long long* starts, const long long* sizes,
                              const long long* rows, const long long* sources,
                              const double* zsigned, const double* magnitude,
                              const bool* significant, const double* counts,
                              double* scores, long long nsrc, long long width,
                              long long n_targets, long long nwork) {
    for (long long i =
             static_cast<long long>(blockIdx.x) * blockDim.x + threadIdx.x;
         i < nwork; i += static_cast<long long>(blockDim.x) * gridDim.x) {
        const long long task = i / nsrc, other = i % nsrc;
        const long long row = rows[task], source = sources[task];
        const double count = counts[source * nsrc + other];
        double score = nan("");
        if (other != source && significant[row * nsrc + other] &&
            count > n_targets) {
            double sum1 = 0, sum2 = 0, absolute = 0;
            // Both paths visit shared features in ascending order.
            const bool smaller = sizes[other] * 4 < sizes[source];
            const long long size = smaller ? sizes[other] : sizes[source];
            for (long long k = 0; k < size; ++k) {
                const long long feature =
                    targets[starts[smaller ? other : source] + k];
                if (net[feature * nsrc + (smaller ? source : other)] != 0) {
                    long long focal_k = k;
                    if (smaller) {
                        long long lo = 0, hi = sizes[source];
                        while (lo < hi) {
                            const long long mid = (lo + hi) / 2;
                            if (targets[starts[source] + mid] < feature)
                                lo = mid + 1;
                            else
                                hi = mid;
                        }
                        focal_k = lo;
                    }
                    const double weight = net[feature * nsrc + source];
                    const double term =
                        weight * zsigned[task * width + focal_k];
                    sum1 += term;
                    absolute += fabs(term);
                    sum2 +=
                        (1 - fabs(weight)) * magnitude[task * width + focal_k];
                }
            }
            if (sum2 > 0)
                sum1 = zero_cancellation(sum1, absolute, sizes[source]);
            const double sign = sum1 == 0 ? 1 : copysign(1.0, sum1);
            score =
                (fabs(sum1) + sum2 * (sum2 > 0)) / count * sign * sqrt(count);
            score = 1 - normcdf(score);
        }
        scores[i] = score;
    }
}

__global__ void penalize(const int* index, const long long* offsets,
                         const int* losers, const int* winners,
                         const double* factors, const long long* features,
                         double* likelihood, long long nfeature, long long nvar,
                         long long nedge, long long nwork) {
    for (long long i =
             static_cast<long long>(blockIdx.x) * blockDim.x + threadIdx.x;
         i < nwork; i += static_cast<long long>(blockDim.x) * gridDim.x) {
        const long long obs = i / nfeature, feature = features[i % nfeature];
        for (long long p = offsets[obs]; p < offsets[obs + 1]; ++p) {
            const int x =
                index[static_cast<long long>(losers[p]) * nvar + feature];
            const int y =
                index[static_cast<long long>(winners[p]) * nvar + feature];
            if (x >= 0 && y >= 0) {
                double& value = likelihood[obs * nedge + x];
                if (value != 0 && likelihood[obs * nedge + y] != 0)
                    value /= factors[p];
            }
        }
    }
}

__global__ void rescore(const double* zsigned, const double* magnitude,
                        const double* weights, const long long* targets,
                        const long long* starts, const double* likelihood,
                        const bool* affected, const double* initial,
                        double* result, long long nvar, long long nsrc,
                        long long nedge, long long nwork) {
    for (long long i =
             static_cast<long long>(blockIdx.x) * blockDim.x + threadIdx.x;
         i < nwork; i += static_cast<long long>(blockDim.x) * gridDim.x) {
        double score = initial[i];
        if (affected[i]) {
            const long long obs = i / nsrc, source = i % nsrc;
            double maximum = 0, total = 0, scale = 0, sum1 = 0, sum2 = 0,
                   absolute = 0;
            for (long long e = starts[source]; e < starts[source + 1]; ++e) {
                const double w = likelihood[obs * nedge + e];
                maximum = isnan(w) ? w : (w > maximum ? w : maximum);
            }
            for (long long e = starts[source]; e < starts[source + 1]; ++e) {
                const double w = likelihood[obs * nedge + e] / maximum;
                total += w;
                scale += w * w;
            }
            for (long long e = starts[source]; e < starts[source + 1]; ++e) {
                const double w =
                    (likelihood[obs * nedge + e] / maximum) / total;
                const long long feature = obs * nvar + targets[e];
                const double term = zsigned[feature] * (w * weights[e]);
                sum1 += term;
                absolute += fabs(term);
                sum2 += magnitude[feature] * ((1 - fabs(weights[e])) * w);
            }
            if (sum2 > 0) sum1 = zero_cancellation(sum1, absolute, nvar);
            const double positive = sum2 < 0 ? 0 : sum2;
            const double sign = sum1 == 0 ? 0 : copysign(1.0, sum1);
            score = (fabs(sum1) + positive) * sign * sqrt(scale);
        }
        result[i] = score;
    }
}

}  // namespace viper
