"""Rust/cuda-oxide GPU entry points; streams and allocations are borrowed."""

from cupy import ndarray

def gram_csr_upper(
    indptr: ndarray,
    index: ndarray,
    data: ndarray,
    *,
    nrows: int,
    ncols: int,
    out: ndarray,
    stream: int = 0,
) -> None: ...
def copy_upper_to_lower(*, out: ndarray, ncols: int, stream: int = 0) -> None: ...
def cov_from_gram(
    gram: ndarray,
    meanx: ndarray,
    meany: ndarray,
    *,
    cov: ndarray,
    ncols: int,
    stream: int = 0,
) -> None: ...
def check_zero_genes(
    indices: ndarray, *, out: ndarray, nnz: int, num_genes: int, stream: int = 0
) -> None: ...
