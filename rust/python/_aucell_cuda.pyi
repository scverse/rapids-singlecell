"""Rust/cuda-oxide GPU entry points; streams and allocations are borrowed."""

from cupy import ndarray

def auc(
    ranks: ndarray,
    *,
    R: int,
    C: int,
    cnct: ndarray,
    starts: ndarray,
    lens: ndarray,
    n_sets: int,
    n_up: int,
    max_aucs: ndarray,
    es: ndarray,
    stream: int = 0,
) -> None: ...
