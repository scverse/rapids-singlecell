#pragma once

// Storage type of Harmony's assignments R: the compute type T, or bfloat16
// (dtype="bfloat16"), which is converted to and from T on every access.
// Also the fixed-order warp and block reductions shared by the kernels.

#include <cuda_bf16.h>

#include <type_traits>

template <typename S, typename T>
__device__ __forceinline__ S to_storage(T value) {
    if constexpr (std::is_same_v<S, __nv_bfloat16>)
        return __float2bfloat16_rn(static_cast<float>(value));
    else
        return static_cast<S>(value);
}

template <typename T, typename S>
__device__ __forceinline__ T from_storage(S value) {
    if constexpr (std::is_same_v<S, __nv_bfloat16>)
        return static_cast<T>(__bfloat162float(value));
    else
        return static_cast<T>(value);
}

// Butterfly reductions: every lane gets the result (lane 0 the same sum as a
// shuffle-down tree).
template <typename T>
__device__ __forceinline__ T warp_sum(T value) {
#pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1)
        value += __shfl_xor_sync(0xffffffff, value, offset);
    return value;
}

template <typename T>
__device__ __forceinline__ T warp_max(T value) {
#pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1)
        value = max(value, __shfl_xor_sync(0xffffffff, value, offset));
    return value;
}

// Sum over a block of whole warps, returned to every thread.
template <typename T>
__device__ T block_sum(T value) {
    __shared__ T warp_sums[32];
    int lane = threadIdx.x & 31, warp = threadIdx.x >> 5;
    value = warp_sum(value);
    if (lane == 0) warp_sums[warp] = value;
    __syncthreads();
    if (warp == 0) {
        value = warp_sum(lane < (blockDim.x >> 5) ? warp_sums[lane] : T(0));
        if (lane == 0) warp_sums[0] = value;
    }
    __syncthreads();
    value = warp_sums[0];
    __syncthreads();
    return value;
}
