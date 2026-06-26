#pragma once

#include <cuda_runtime.h>

__global__ void rev_cummin64_kernel(const double* __restrict__ x,
                                    double* __restrict__ y, int n_rows, int m) {
    int r = blockDim.x * blockIdx.x + threadIdx.x;
    if (r >= n_rows) return;

    const double* xr = x + (size_t)r * m;
    double* yr = y + (size_t)r * m;

    double cur = xr[m - 1];
    yr[m - 1] = cur;

    for (int j = m - 2; j >= 0; --j) {
        double v = xr[j];
        cur = (v < cur) ? v : cur;
        yr[j] = cur;
    }
}
