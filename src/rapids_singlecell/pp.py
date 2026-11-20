from __future__ import annotations

from .preprocessing import (
    NeighborsOut,
    NeighborsResult,
    NormalizeTotalOut,
    NormalizeTotalResult,
    PcaOut,
    PcaResult,
    QcMetricsOut,
    QcMetricsResult,
    ScaleOut,
    ScaleResult,
    ScrubletOut,
    ScrubletResult,
    WnnResult,
    bbknn,
    calculate_qc_metrics,
    dsb,
    filter_cells,
    filter_genes,
    harmony_integrate,
    highly_variable_genes,
    log1p,
    neighbors,
    normalize_pearson_residuals,
    normalize_shifted_clr,
    normalize_total,
    pca,
    regress_out,
    scale,
    scrublet,
    scrublet_simulate_doublets,
    sqrt,
    wnn,
)
from .preprocessing import (
    filter_highly_variable as filter_highly_variable,
)
from .preprocessing import (
    flag_gene_family as flag_gene_family,
)

__all__ = [
    "NeighborsOut",
    "NeighborsResult",
    "NormalizeTotalOut",
    "NormalizeTotalResult",
    "PcaOut",
    "PcaResult",
    "QcMetricsOut",
    "QcMetricsResult",
    "ScaleOut",
    "ScaleResult",
    "ScrubletOut",
    "ScrubletResult",
    "WnnResult",
    "bbknn",
    "calculate_qc_metrics",
    "dsb",
    "filter_cells",
    "filter_genes",
    "harmony_integrate",
    "highly_variable_genes",
    "log1p",
    "neighbors",
    "normalize_pearson_residuals",
    "normalize_shifted_clr",
    "normalize_total",
    "pca",
    "regress_out",
    "scale",
    "scrublet",
    "scrublet_simulate_doublets",
    "sqrt",
    "wnn",
]

# `out=True` results and `out` mappings have no public members beyond their fields.
__api_members__: dict[str, list[str]] = {n: [] for n in __all__ if n[0].isupper()}

__deprecated_exports__ = {
    "filter_highly_variable": "Deprecated; do not use in new analyses.",
    "flag_gene_family": "Deprecated; do not use in new analyses.",
}
