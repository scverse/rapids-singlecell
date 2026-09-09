"""Rust/cuda-oxide GPU entry points; streams and allocations are borrowed."""

from cupy import ndarray

def paired_squared(
    X: ndarray,
    Y: ndarray,
    *,
    out: ndarray,
    n_pairs: int,
    n_features: int,
    stream: int = 0,
) -> None: ...
def paired_abs_mean(
    X: ndarray,
    Y: ndarray,
    *,
    out: ndarray,
    n_pairs: int,
    n_features: int,
    stream: int = 0,
) -> None: ...
def pairwise_squared(
    X: ndarray,
    Y: ndarray,
    *,
    out: ndarray,
    n_x: int,
    n_y: int,
    n_features: int,
    stream: int = 0,
) -> None: ...
def pairwise_abs_mean(
    X: ndarray,
    Y: ndarray,
    *,
    out: ndarray,
    n_x: int,
    n_y: int,
    n_features: int,
    stream: int = 0,
) -> None: ...
