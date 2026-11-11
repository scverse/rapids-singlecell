from __future__ import annotations

from .tools import (
    CellCycleOut,
    CellCycleResult,
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
    "CellCycleOut",
    "CellCycleResult",
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

# `out=True` results and `out` mappings have no public members beyond their fields.
__api_members__: dict[str, list[str]] = {n: [] for n in __all__ if n[0].isupper()}

__deprecated_exports__ = {
    "kmeans": "Deprecated; do not use in new analyses.",
    "louvain": louvain.__deprecated__,
    "rank_genes_groups_logreg": rank_genes_groups_logreg.__deprecated__,
}
