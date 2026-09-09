"""Rust/cuda-oxide GPU entry points; streams and allocations are borrowed."""

from cupy import ndarray

def sum_count_dense(
    data: ndarray,
    *,
    clusters: ndarray,
    sum: ndarray,
    count: ndarray,
    rows: int,
    cols: int,
    ncls: int,
    stream: int = 0,
) -> None: ...
def sum_count_sparse(
    indptr: ndarray,
    index: ndarray,
    data: ndarray,
    *,
    clusters: ndarray,
    sum: ndarray,
    count: ndarray,
    rows: int,
    ncls: int,
    stream: int = 0,
) -> None: ...
def mean_dense(
    data: ndarray,
    *,
    clusters: ndarray,
    g: ndarray,
    rows: int,
    cols: int,
    ncls: int,
    stream: int = 0,
) -> None: ...
def mean_sparse(
    indptr: ndarray,
    index: ndarray,
    data: ndarray,
    *,
    clusters: ndarray,
    g: ndarray,
    rows: int,
    ncls: int,
    stream: int = 0,
) -> None: ...
def elementwise_diff(
    g: ndarray, *, total_counts: ndarray, n_genes: int, n_clusters: int, stream: int = 0
) -> None: ...
def interaction(
    interactions: ndarray,
    *,
    interaction_clusters: ndarray,
    mean: ndarray,
    res: ndarray,
    mask: ndarray,
    g: ndarray,
    n_iter: int,
    n_inter_clust: int,
    ncls: int,
    stream: int = 0,
) -> None: ...
def res_mean(
    interactions: ndarray,
    *,
    interaction_clusters: ndarray,
    mean: ndarray,
    res_mean: ndarray,
    n_inter: int,
    n_inter_clust: int,
    ncls: int,
    stream: int = 0,
) -> None: ...
