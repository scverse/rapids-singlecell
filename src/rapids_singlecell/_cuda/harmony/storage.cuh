#pragma once

// Storage type of Harmony's assignments R: the compute type T, or bfloat16
// (dtype="bfloat16"), which is converted to and from T on every access.

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
