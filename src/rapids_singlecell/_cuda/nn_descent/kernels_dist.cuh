#pragma once

#include <cuda_runtime.h>

__global__ void compute_distances_sqeuclidean_kernel(
    const float* __restrict__ data, float* __restrict__ out,
    const unsigned int* __restrict__ pairs, long long n_samples,
    long long n_features, long long n_neighbors) {
    long long i1 = blockDim.x * blockIdx.x + threadIdx.x;
    if (i1 >= n_samples) {
        return;
    }
    for (long long j = 0; j < n_neighbors; ++j) {
        long long i2 = static_cast<long long>(pairs[i1 * n_neighbors + j]);
        float dist = 0.0f;
        long long base1 = i1 * n_features;
        long long base2 = i2 * n_features;
        for (long long d = 0; d < n_features; ++d) {
            float diff = data[base1 + d] - data[base2 + d];
            dist += diff * diff;  // powf(diff, 2)
        }
        out[i1 * n_neighbors + j] = dist;
    }
}

__global__ void compute_distances_cosine_kernel(
    const float* __restrict__ data, float* __restrict__ out,
    const unsigned int* __restrict__ pairs, long long n_samples,
    long long n_features, long long n_neighbors) {
    long long i1 = blockDim.x * blockIdx.x + threadIdx.x;
    if (i1 >= n_samples) {
        return;
    }
    float sum_i1 = 0.0f;
    long long base1 = i1 * n_features;
    for (long long d = 0; d < n_features; ++d) {
        float v = data[base1 + d];
        sum_i1 += v * v;  // powf(v, 2)
    }
    float inv_norm_i1 = (sum_i1 > 0.0f) ? rsqrtf(sum_i1) : 0.0f;
    for (long long j = 0; j < n_neighbors; ++j) {
        long long i2 = static_cast<long long>(pairs[i1 * n_neighbors + j]);
        float dot = 0.0f;
        float sum_i2 = 0.0f;
        long long base2 = i2 * n_features;
        for (long long d = 0; d < n_features; ++d) {
            float v1 = data[base1 + d];
            float v2 = data[base2 + d];
            dot += v1 * v2;
            sum_i2 += v2 * v2;
        }
        float inv_norm_i2 = (sum_i2 > 0.0f) ? rsqrtf(sum_i2) : 0.0f;
        out[i1 * n_neighbors + j] = 1.0f - dot * inv_norm_i1 * inv_norm_i2;
    }
}

__global__ void compute_distances_inner_kernel(
    const float* __restrict__ data, float* __restrict__ out,
    const unsigned int* __restrict__ pairs, long long n_samples,
    long long n_features, long long n_neighbors) {
    long long i1 = blockDim.x * blockIdx.x + threadIdx.x;
    if (i1 >= n_samples) {
        return;
    }
    for (long long j = 0; j < n_neighbors; ++j) {
        long long i2 = static_cast<long long>(pairs[i1 * n_neighbors + j]);
        float val = 0.0f;
        long long base1 = i1 * n_features;
        long long base2 = i2 * n_features;
        for (long long d = 0; d < n_features; ++d) {
            val += data[base1 + d] * data[base2 + d];
        }
        out[i1 * n_neighbors + j] = val;
    }
}
