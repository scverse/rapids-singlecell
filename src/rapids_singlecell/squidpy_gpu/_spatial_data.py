from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
from anndata import AnnData

if TYPE_CHECKING:
    from spatialdata import SpatialData


def _extract_adata(
    adata: AnnData | SpatialData, *, table_key: str | None = None
) -> AnnData:
    """Resolve an optional SpatialData input without importing it for AnnData."""
    if isinstance(adata, AnnData):
        return adata
    message = (
        f"Expected `adata` to be of type `anndata.AnnData` or "
        f"`spatialdata.SpatialData`, found `{type(adata).__name__}`."
    )
    try:
        from spatialdata import SpatialData
    except ImportError:
        # Without spatialdata installed, the input cannot be a SpatialData object.
        raise TypeError(message) from None

    if not isinstance(adata, SpatialData):
        raise TypeError(message)
    if table_key is None:
        raise TypeError("SpatialData input requires `table_key`.")
    if table_key not in adata.tables:
        raise ValueError(
            f"Table {table_key!r} not found in SpatialData. "
            f"Available tables: {list(adata.tables)}"
        )
    return adata.tables[table_key]


def _resolve_spatial_data(
    adata: AnnData | SpatialData,
    *,
    table_key: str | None,
    elements_to_coordinate_systems: dict[str, str] | None,
    spatial_key: str,
    library_key: str | None,
) -> tuple[AnnData, str | None]:
    """Match transformed element centroids to table rows by region and instance."""
    table = _extract_adata(adata, table_key=table_key)
    if isinstance(adata, AnnData):
        return table, library_key

    from spatialdata import get_centroids
    from spatialdata.models import get_table_keys

    if elements_to_coordinate_systems is None:
        raise ValueError("SpatialData input requires `elements_to_coordinate_systems`.")
    _, region_key, instance_key = get_table_keys(table)
    if region_key is None or instance_key is None:
        raise ValueError("The SpatialData table must annotate spatial elements.")
    regions = table.obs[region_key]
    if regions.isna().any():
        raise ValueError(f"`adata.obs[{region_key!r}]` has missing labels.")
    missing = set(regions) - elements_to_coordinate_systems.keys()
    if missing:
        raise ValueError(
            f"Missing coordinate systems for table regions: {sorted(missing)}."
        )
    coords = np.empty((table.n_obs, 2), dtype=np.float64)
    for region in regions.unique():
        centroids = get_centroids(
            adata[region], coordinate_system=elements_to_coordinate_systems[region]
        )[["x", "y"]].compute()
        rows = np.flatnonzero(regions == region)
        instances = table.obs[instance_key].iloc[rows]
        if not instances.isin(centroids.index).all():
            raise ValueError(
                f"Table instances are missing from spatial element {region!r}."
            )
        coords[rows] = centroids.loc[instances].to_numpy()
    table.obsm[spatial_key] = coords
    return table, region_key
