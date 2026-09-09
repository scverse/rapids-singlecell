"""Rust/cuda-oxide GPU entry points; streams and allocations are borrowed."""

from cupy import ndarray

def assign_threshold_dense(
    X: ndarray,
    valid_guides: ndarray,
    lam: ndarray,
    mu: ndarray,
    sigma: ndarray,
    pi0: ndarray,
    assignments: ndarray,
    thresholds: ndarray,
    *,
    n_cells: int,
    n_guides: int,
    n_valid_guides: int,
    posterior_threshold: float,
    stream: int = 0,
) -> None: ...
def fit_assign_dense(
    X: ndarray,
    assignments: ndarray,
    thresholds: ndarray,
    lam: ndarray,
    mu: ndarray,
    sigma: ndarray,
    pi0: ndarray,
    valid_mask: ndarray,
    nonzero_counts: ndarray,
    max_counts: ndarray,
    *,
    n_cells: int,
    n_guides: int,
    max_iter: int,
    tol: float,
    posterior_threshold: float,
    stream: int = 0,
) -> None: ...
