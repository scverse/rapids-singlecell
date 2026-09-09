"""Rust/cuda-oxide GPU entry points; streams and allocations are borrowed."""

from cupy import ndarray

def compute_distances_sparse(
    indptr: ndarray,
    indices: ndarray,
    data: ndarray,
    cat_offsets: ndarray,
    cell_indices: ndarray,
    pair_left: ndarray,
    pair_right: ndarray,
    pairwise_sums: ndarray,
    num_pairs: int,
    n_features: int,
    blocks_per_pair: int,
    cell_tile: int,
    feat_tile: int,
    block_size: int,
    shared_mem: int,
    stream: int = 0,
) -> None: ...
def compute_distances(
    embedding: ndarray,
    cat_offsets: ndarray,
    cell_indices: ndarray,
    pair_left: ndarray,
    pair_right: ndarray,
    pairwise_sums: ndarray,
    num_pairs: int,
    n_features: int,
    blocks_per_pair: int,
    cell_tile: int,
    feat_tile: int,
    block_size: int,
    shared_mem: int,
    stream: int = 0,
) -> None: ...
def get_kernel_config(
    n_features: int, is_double: bool
) -> tuple[int, int, int, int] | None: ...
