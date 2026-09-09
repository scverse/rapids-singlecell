"""Rust/cuda-oxide GPU entry points; streams and allocations are borrowed."""

from cupy import ndarray

def sqeuclidean(
    data: ndarray,
    *,
    out: ndarray,
    pairs: ndarray,
    n_samples: int,
    n_features: int,
    n_neighbors: int,
    stream: int = 0,
) -> None: ...
def cosine(
    data: ndarray,
    *,
    out: ndarray,
    pairs: ndarray,
    n_samples: int,
    n_features: int,
    n_neighbors: int,
    stream: int = 0,
) -> None: ...
def inner(
    data: ndarray,
    *,
    out: ndarray,
    pairs: ndarray,
    n_samples: int,
    n_features: int,
    n_neighbors: int,
    stream: int = 0,
) -> None: ...
