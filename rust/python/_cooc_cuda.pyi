"""Rust/cuda-oxide GPU entry points; streams and allocations are borrowed."""

from cupy import ndarray

def count_csr_catpairs(
    spatial: ndarray,
    *,
    thresholds: ndarray,
    cat_offsets: ndarray,
    cell_indices: ndarray,
    pair_left: ndarray,
    pair_right: ndarray,
    counts: ndarray,
    num_pairs: int,
    k: int,
    l_val: int,
    blocks_per_pair: int,
    cell_tile: int,
    block_size: int,
    shared_mem: int,
    stream: int = 0,
) -> None: ...
def count_pairwise(
    spatial: ndarray,
    *,
    thresholds: ndarray,
    labels: ndarray,
    result: ndarray,
    n: int,
    k: int,
    l_val: int,
    stream: int = 0,
) -> None: ...
def reduce_shared(
    result: ndarray, *, out: ndarray, k: int, l_val: int, format: int, stream: int = 0
) -> bool: ...
def reduce_global(
    result: ndarray,
    *,
    inter_out: ndarray,
    out: ndarray,
    k: int,
    l_val: int,
    format: int,
    stream: int = 0,
) -> None: ...
def get_kernel_config(
    l_val: int, n_cells: int, k: int
) -> tuple[int, int, int, int] | None: ...
