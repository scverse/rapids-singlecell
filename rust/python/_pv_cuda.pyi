"""Rust/cuda-oxide GPU entry points; streams and allocations are borrowed."""

from cupy import ndarray

def rev_cummin64(
    x: ndarray, *, out: ndarray, n_rows: int, m: int, stream: int = 0
) -> None: ...
