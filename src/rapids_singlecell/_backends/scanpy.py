from __future__ import annotations

from typing import TYPE_CHECKING

from rapids_singlecell.get import aggregate
from rapids_singlecell.preprocessing import (
    bbknn,
    calculate_qc_metrics,
    filter_cells,
    filter_genes,
    filter_highly_variable,
    flag_gene_family,
    harmony_integrate,
    highly_variable_genes,
    neighbors,
    normalize_pearson_residuals,
    normalize_total,
    regress_out,
    scrublet,
    scrublet_simulate_doublets,
)
from rapids_singlecell.preprocessing import log1p as _log1p
from rapids_singlecell.preprocessing import pca as _pca
from rapids_singlecell.preprocessing import scale as _scale
from rapids_singlecell.preprocessing._pca import _empty
from rapids_singlecell.tools import (
    diffmap,
    draw_graph,
    embedding_density,
    kmeans,
    leiden,
    louvain,
    rank_genes_groups,
    rank_genes_groups_logreg,
    score_genes,
    score_genes_cell_cycle,
    tsne,
    umap,
)

if TYPE_CHECKING:
    from anndata import AnnData
    from numpy.typing import DTypeLike, NDArray

name = "rapids-singlecell"
aliases = ["cuda", "rapids", "rapids_singlecell"]


def log1p(
    data: AnnData,
    *,
    base: float | None = None,
    layer: str | None = None,
    obsm: str | None = None,
    inplace: bool = True,
    copy: bool = False,
):
    return _log1p(
        data,
        base=base,
        layer=layer,
        obsm=obsm,
        inplace=inplace,
        copy=copy,
    )


def pca(
    data: AnnData,
    n_comps: int | None = None,
    *,
    layer: str | None = None,
    zero_center: bool = True,
    svd_solver: str | None = None,
    chunked: bool = False,
    chunk_size: int | None = None,
    rng=None,
    mask_var: NDArray | str | None = _empty,
    dtype: DTypeLike = "float32",
    key_added: str | None = None,
    copy: bool = False,
    random_state: int | None = 0,
    use_highly_variable: bool | None = None,
    **kwargs,
) -> None | AnnData:
    if rng is not None:
        random_state = rng
    return _pca(
        data,
        n_comps=n_comps,
        layer=layer,
        zero_center=zero_center,
        svd_solver=svd_solver,
        random_state=random_state,
        mask_var=mask_var,
        use_highly_variable=use_highly_variable,
        dtype=dtype,
        chunked=chunked,
        chunk_size=chunk_size,
        key_added=key_added,
        copy=copy,
        **kwargs,
    )


def scale(
    data: AnnData,
    *,
    zero_center: bool = True,
    max_value: float | None = None,
    copy: bool = False,
    layer: str | None = None,
    obsm: str | None = None,
    mask_obs: NDArray | str | None = None,
    inplace: bool = True,
):
    return _scale(
        data,
        zero_center=zero_center,
        max_value=max_value,
        copy=copy,
        layer=layer,
        obsm=obsm,
        mask_obs=mask_obs,
        inplace=inplace,
    )


__all__ = [
    "aggregate",
    "bbknn",
    "calculate_qc_metrics",
    "diffmap",
    "draw_graph",
    "embedding_density",
    "filter_cells",
    "filter_genes",
    "filter_highly_variable",
    "flag_gene_family",
    "harmony_integrate",
    "highly_variable_genes",
    "kmeans",
    "leiden",
    "log1p",
    "louvain",
    "neighbors",
    "normalize_pearson_residuals",
    "normalize_total",
    "pca",
    "rank_genes_groups",
    "rank_genes_groups_logreg",
    "regress_out",
    "scale",
    "score_genes",
    "score_genes_cell_cycle",
    "scrublet",
    "scrublet_simulate_doublets",
    "tsne",
    "umap",
]
