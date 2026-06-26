#pragma once

#include <cuda_runtime.h>
#include <math_constants.h>

template <typename T>
__device__ __forceinline__ T neg_infinity();

template <>
__device__ __forceinline__ float neg_infinity<float>() {
    return -CUDART_INF_F;
}

template <>
__device__ __forceinline__ double neg_infinity<double>() {
    return -CUDART_INF;
}

template <typename T>
__global__ void gaussian_kde_2d_kernel(const T* __restrict__ xy,
                                       T* __restrict__ out, const int n,
                                       const T a, const T b, const T c) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;

    const T xi = xy[2 * i];
    const T yi = xy[2 * i + 1];

    T running_max = neg_infinity<T>();
    T running_sum = T(0);

    for (int j = 0; j < n; j++) {
        const T dx = xi - xy[2 * j];
        const T dy = yi - xy[2 * j + 1];
        const T log_k = a * dx * dx + b * dx * dy + c * dy * dy;

        if (log_k > running_max) {
            running_sum = running_sum * exp(running_max - log_k) + T(1);
            running_max = log_k;
        } else {
            running_sum += exp(log_k - running_max);
        }
    }

    out[i] = log(running_sum) + running_max;
}
