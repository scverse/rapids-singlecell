from __future__ import annotations

from typing import Any, Literal

from anndata import AnnData

from rapids_singlecell.squidpy_gpu import calculate_niche as _calculate_niche
from rapids_singlecell.squidpy_gpu import co_occurrence, ligrec, spatial_autocorr

name = "rapids-singlecell"
aliases = ["cuda", "rapids", "rapids_singlecell"]


def calculate_niche(  # noqa: PLR0917
    data: AnnData,
    flavor: Literal["neighborhood", "utag", "cellcharter", "spatialleiden"],
    library_key: str | None = None,
    mask: Any | None = None,
    groups: str | None = None,
    n_neighbors: int | None = None,
    resolutions: float
    | tuple[float, float]
    | list[float | tuple[float, float]]
    | None = None,
    min_niche_size: int | None = None,
    scale: bool = True,  # noqa: FBT001, FBT002
    abs_nhood: bool = False,  # noqa: FBT001, FBT002
    distance: int | None = None,
    n_hop_weights: list[float] | None = None,
    aggregation: str | None = None,
    n_components: int | None = None,
    random_state: int = 42,
    spatial_connectivities_key: str = "spatial_connectivities",
    latent_connectivities_key: str = "connectivities",
    layer_ratio: float = 1.0,
    n_iterations: int = -1,
    use_weights: bool | tuple[bool, bool] = True,  # noqa: FBT001, FBT002
    use_rep: str | None = None,
    inplace: bool = True,  # noqa: FBT001, FBT002
    *,
    table_key: str | None = None,
) -> AnnData | None:
    """Adapt Squidpy's niche API to RAPIDS SingleCell's GPU implementation."""
    if not isinstance(data, AnnData):
        raise TypeError(
            "The RAPIDS SingleCell backend currently supports AnnData inputs only."
        )
    if flavor == "spatialleiden":
        raise NotImplementedError(
            "The RAPIDS SingleCell backend does not support flavor='spatialleiden'."
        )
    if library_key is not None or mask is not None or table_key is not None:
        raise NotImplementedError(
            "library_key, mask, and table_key are not supported by the RAPIDS backend."
        )
    if (
        latent_connectivities_key != "connectivities"
        or layer_ratio != 1.0
        or n_iterations != -1
        or use_weights is not True
    ):
        raise NotImplementedError(
            "The selected niche parameters are not supported by the RAPIDS backend."
        )

    return _calculate_niche(
        data,
        flavor=flavor,
        groups=groups,
        n_neighbors=15 if n_neighbors is None else n_neighbors,
        resolutions=(0.5,) if resolutions is None else resolutions,
        min_niche_size=min_niche_size,
        scale=scale,
        abs_nhood=abs_nhood,
        distance=distance,
        n_hop_weights=n_hop_weights,
        aggregation="mean" if aggregation is None else aggregation,
        n_components=10 if n_components is None else n_components,
        random_state=random_state,
        spatial_connectivities_key=spatial_connectivities_key,
        use_rep=use_rep,
        copy=not inplace,
    )


__all__ = ["calculate_niche", "co_occurrence", "ligrec", "spatial_autocorr"]
