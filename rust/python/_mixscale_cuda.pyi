"""Rust/cuda-oxide GPU entry points; streams and allocations are borrowed."""

from cupy import ndarray

def project_score(
    X: ndarray,
    n_vars: int,
    row_ids: ndarray,
    col_ids: ndarray,
    n_per_gene: ndarray,
    k_per_gene: ndarray,
    cell_offsets: ndarray,
    feat_offsets: ndarray,
    is_guide: ndarray,
    nt_in_all: ndarray,
    pvec_scratch: ndarray,
    scores_out: ndarray,
    *,
    n_genes: int,
    max_k: int,
    do_scale: bool,
    stream: int = 0,
) -> None: ...
