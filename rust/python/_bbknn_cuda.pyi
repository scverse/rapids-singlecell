"""Rust/cuda-oxide GPU entry points; streams and allocations are borrowed."""

from cupy import ndarray

def find_top_k_per_row(
    data: ndarray,
    indptr: ndarray,
    *,
    n_rows: int,
    trim: int,
    vals: ndarray,
    stream: int = 0,
) -> None: ...
def find_top_k_per_row_sorted(
    data: ndarray,
    indptr: ndarray,
    *,
    n_rows: int,
    trim: int,
    vals: ndarray,
    stream: int = 0,
) -> None: ...
def sort_tile_size() -> int: ...
def cut_smaller(
    indptr: ndarray,
    index: ndarray,
    data: ndarray,
    *,
    vals: ndarray,
    n_rows: int,
    stream: int = 0,
) -> None: ...
