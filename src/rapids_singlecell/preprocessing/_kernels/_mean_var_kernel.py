from __future__ import annotations

import cupy as cp

sq_sum = cp.ReductionKernel(
    "T x",
    "float64 y",
    "x * x",
    "a + b",
    "y = a",
    "0",
    "sqsum64",
)

mean_sum = cp.ReductionKernel(
    "T x",
    "float64 y",
    "x",
    "a + b",
    "y = a",
    "0",
    "sum64",
)
