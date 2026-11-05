from __future__ import annotations

from .tools import (
    diffmap,
    draw_graph,
    embedding_density,
    ingest,
    leiden,
    pca,
    rank_genes_groups,
    score_genes,
    score_genes_cell_cycle,
    tsne,
    umap,
)
from .tools import (
    kmeans as kmeans,
)
from .tools import (
    louvain as louvain,
)
from .tools import (
    rank_genes_groups_logreg as rank_genes_groups_logreg,
)

__all__ = [
    "diffmap",
    "draw_graph",
    "embedding_density",
    "ingest",
    "leiden",
    "pca",
    "rank_genes_groups",
    "score_genes",
    "score_genes_cell_cycle",
    "tsne",
    "umap",
]

__deprecated_exports__ = {
    "kmeans": "Deprecated; do not use in new analyses.",
    "louvain": louvain.__deprecated__,
    "rank_genes_groups_logreg": rank_genes_groups_logreg.__deprecated__,
}
