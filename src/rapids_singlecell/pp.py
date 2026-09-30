from __future__ import annotations

from .preprocessing import (
    bbknn,
    calculate_qc_metrics,
    filter_cells,
    filter_genes,
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
from .preprocessing import (
    filter_highly_variable as filter_highly_variable,
)
from .preprocessing import (
    flag_gene_family as flag_gene_family,
)

__all__ = [
    "bbknn",
    "calculate_qc_metrics",
    "filter_cells",
    "filter_genes",
    "harmony_integrate",
    "highly_variable_genes",
    "log1p",
    "neighbors",
    "normalize_pearson_residuals",
    "normalize_total",
    "pca",
    "regress_out",
    "scale",
    "scrublet",
    "scrublet_simulate_doublets",
    "sqrt",
]

__deprecated_exports__ = {
    "filter_highly_variable": "Deprecated; do not use in new analyses.",
    "flag_gene_family": "Deprecated; do not use in new analyses.",
}
