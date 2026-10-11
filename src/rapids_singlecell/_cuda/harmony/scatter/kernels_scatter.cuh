#pragma once

#include <cuda_runtime.h>
#include <stdint.h>
#include <type_traits>

#include "../storage.cuh"

// a[cat + 1, pc0 .. pc0 + 1] = sum over the category's cells of v * bias; one
// block per (category, pair of PCs).
template <typename T>
__global__ void scatter_add_kernel_with_bias_block(
    const T* __restrict__ v, const int* __restrict__ cat_offsets,
    const int* __restrict__ cell_indices, int n_cells, int n_pcs, int n_batches,
    T* __restrict__ a, const T* __restrict__ bias) {
    using VecPC = std::conditional_t<std::is_same_v<T, float>, float2, double2>;
    int pairs = (n_pcs + 1) / 2;
    if ((int)blockIdx.x >= n_batches * pairs) return;
    int cat = blockIdx.x / pairs + 1, pc0 = blockIdx.x % pairs * 2;
    bool has_pc1 = pc0 + 1 < n_pcs;
    T acc0 = T(0), acc1 = T(0);
    for (int i = cat_offsets[cat - 1] + threadIdx.x; i < cat_offsets[cat];
         i += blockDim.x) {
        int cell = cell_indices[i];
        const T* ptr = v + (size_t)cell * n_pcs + pc0;
        T bb = __ldg(bias + cell);
        if (has_pc1 && ((uintptr_t)ptr & (sizeof(VecPC) - 1)) == 0) {
            VecPC vv = *(const VecPC*)ptr;
            acc0 += vv.x * bb;
            acc1 += vv.y * bb;
        } else {
            acc0 += ptr[0] * bb;
            if (has_pc1) acc1 += ptr[1] * bb;
        }
    }
    acc0 = block_sum(acc0);
    acc1 = block_sum(acc1);
    if (threadIdx.x == 0) {
        a[cat * n_pcs + pc0] = acc0;
        if (has_pc1) a[cat * n_pcs + pc0 + 1] = acc1;
    }
}
