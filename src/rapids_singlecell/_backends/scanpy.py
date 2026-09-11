"""Scanpy backend exports with RAPIDS-singlecell's native semantics."""

from __future__ import annotations

from functools import partial

from rapids_singlecell.get import aggregate
from rapids_singlecell.preprocessing import (
    bbknn,
    filter_cells,
    filter_genes,
    filter_highly_variable,
    flag_gene_family,
    harmony_integrate,
    highly_variable_genes,
    log1p,
    neighbors,
    normalize_pearson_residuals,
    normalize_total,
    pca,
    regress_out,
    scale,
    scrublet,
    scrublet_simulate_doublets,
    sqrt,
)
from rapids_singlecell.preprocessing import (
    calculate_qc_metrics as _calculate_qc_metrics,
)
from rapids_singlecell.tools import (
    diffmap,
    draw_graph,
    embedding_density,
    ingest,
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

name = "rapids-singlecell"
aliases = ["cuda", "rapids", "rapids_singlecell"]

calculate_qc_metrics = partial(_calculate_qc_metrics, inplace=False)


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
    "ingest",
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
    "sqrt",
    "tsne",
    "umap",
]
