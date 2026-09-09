"""Rust/cuda-oxide GPU entry points; streams and allocations are borrowed."""

from cupy import ndarray

def gaussian_kde_2d(
    xy: ndarray, *, out: ndarray, n: int, a: float, b: float, c: float, stream: int = 0
) -> None: ...
