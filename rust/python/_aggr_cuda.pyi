"""Rust/cuda-oxide GPU entry points; streams and allocations are borrowed."""

from cupy import ndarray

def sparse_aggr(
    indptr: ndarray,
    index: ndarray,
    data: ndarray,
    *,
    out_sum: ndarray | None = None,
    out_count: ndarray | None = None,
    out_sqsum: ndarray | None = None,
    cats: ndarray,
    mask: ndarray,
    n_cells: int,
    n_genes: int,
    is_csc: bool,
    stream: int = 0,
) -> None: ...
def dense_aggr(
    data: ndarray,
    *,
    out_sum: ndarray | None = None,
    out_count: ndarray | None = None,
    out_sqsum: ndarray | None = None,
    cats: ndarray,
    mask: ndarray,
    n_cells: int,
    n_genes: int,
    is_fortran: bool,
    stream: int = 0,
) -> None: ...
def csr_to_coo(
    indptr: ndarray,
    index: ndarray,
    data: ndarray,
    *,
    out_row: ndarray,
    out_col: ndarray,
    out_data: ndarray,
    cats: ndarray,
    mask: ndarray,
    n_cells: int,
    stream: int = 0,
) -> None: ...
def sparse_var(
    indptr: ndarray,
    index: ndarray,
    data: ndarray,
    *,
    means: ndarray,
    n_cells: ndarray,
    dof: int,
    n_groups: int,
    stream: int = 0,
) -> None: ...
