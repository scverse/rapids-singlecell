#pragma once

#include <cuda_runtime.h>
#include <stdint.h>

template <typename T>
struct kmeans_err_vec_traits;

template <>
struct kmeans_err_vec_traits<float> {
    using Vec = float4;
    static constexpr size_t width = 4;
    static __device__ __forceinline__ float accumulate(const Vec& r,
                                                       const Vec& dot) {
        return r.x * 2.f * (1.f - dot.x) + r.y * 2.f * (1.f - dot.y) +
               r.z * 2.f * (1.f - dot.z) + r.w * 2.f * (1.f - dot.w);
    }
};

template <>
struct kmeans_err_vec_traits<double> {
    using Vec = double2;
    static constexpr size_t width = 2;
    static __device__ __forceinline__ double accumulate(const Vec& r,
                                                        const Vec& dot) {
        return r.x * 2.0 * (1.0 - dot.x) + r.y * 2.0 * (1.0 - dot.y);
    }
};

template <typename T>
__global__ void kmeans_err_kernel(const T* __restrict__ r,
                                  const T* __restrict__ dot, size_t n,
                                  T* __restrict__ out) {
    T acc = (T)0;
    using VecTraits = kmeans_err_vec_traits<T>;
    using Vec = typename VecTraits::Vec;

    size_t tid = blockIdx.x * blockDim.x + threadIdx.x;
    const size_t scalar_stride = gridDim.x * blockDim.x;
    const size_t vec_width = VecTraits::width;
    const size_t vec_align = sizeof(Vec);
    const uintptr_t r_addr = (uintptr_t)r;
    const uintptr_t dot_addr = (uintptr_t)dot;
    const bool has_common_vector_alignment =
        ((r_addr ^ dot_addr) & (vec_align - 1)) == 0;

    if (!has_common_vector_alignment) {
        for (size_t i = tid; i < n; i += scalar_stride) {
            T rv = r[i];
            T dotv = dot[i];
            acc += rv * (T)2 * ((T)1 - dotv);
        }
    } else {
        size_t prefix = 0;
        size_t misalignment = r_addr & (vec_align - 1);
        if (misalignment != 0) prefix = (vec_align - misalignment) / sizeof(T);
        if (prefix > n) prefix = n;

        for (size_t i = tid; i < prefix; i += scalar_stride) {
            T rv = r[i];
            T dotv = dot[i];
            acc += rv * (T)2 * ((T)1 - dotv);
        }

        size_t vector_elements = ((n - prefix) / vec_width) * vec_width;
        size_t vector_chunks = vector_elements / vec_width;
        for (size_t chunk = tid; chunk < vector_chunks;
             chunk += scalar_stride) {
            size_t i = prefix + chunk * vec_width;
            Vec rv = *(const Vec*)(r + i);
            Vec dotv = *(const Vec*)(dot + i);
            acc += VecTraits::accumulate(rv, dotv);
        }

        size_t tail = prefix + vector_elements;
        for (size_t i = tail + tid; i < n; i += scalar_stride) {
            T rv = r[i];
            T dotv = dot[i];
            acc += rv * (T)2 * ((T)1 - dotv);
        }
    }

#pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1)
        acc += __shfl_down_sync(0xffffffff, acc, offset);
    __shared__ T s[32];
    if ((threadIdx.x & 31) == 0) s[threadIdx.x >> 5] = acc;
    __syncthreads();
    if (threadIdx.x < 32) {
        T val = (threadIdx.x < (blockDim.x >> 5)) ? s[threadIdx.x] : (T)0;
#pragma unroll
        for (int offset = 16; offset > 0; offset >>= 1)
            val += __shfl_down_sync(0xffffffff, val, offset);
        if (threadIdx.x == 0) atomicAdd(out, val);
    }
}
