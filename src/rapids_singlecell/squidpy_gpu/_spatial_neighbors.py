from __future__ import annotations

from typing import TYPE_CHECKING, Literal, NamedTuple

import cupy as cp
import numpy as np
import scipy.sparse

from . import neighbors as nb
from ._spatial_data import _resolve_spatial_data
from ._utils import _assert_categorical_obs, _assert_spatial_basis

if TYPE_CHECKING:
    from anndata import AnnData
    from spatialdata import SpatialData

_SHARED = (
    "spatial_key",
    "library_key",
    "key_added",
    "copy",
    "table_key",
    "elements_to_coordinate_systems",
)
# Exact built-in types build all libraries in one pass; subclasses may override.
_BUILDERS = (nb.KNNBuilder, nb.RadiusBuilder, nb.DelaunayBuilder, nb.GridBuilder)
_STEPS = {
    nb.DistanceIntervalPostprocessor,
    nb.PercentilePostprocessor,
    nb.TransformPostprocessor,
}


class SpatialNeighborsResult(NamedTuple):
    """Spatial connectivity and distance matrices returned with ``copy=True``."""

    #: SciPy CSR matrix of edge weights, with shape ``(n_obs, n_obs)``.
    connectivities: scipy.sparse.csr_matrix
    #: SciPy CSR matrix of Euclidean distances, or shortest directed hop
    #: counts for grid graphs, with shape ``(n_obs, n_obs)``.
    distances: scipy.sparse.csr_matrix


def spatial_neighbors_knn(
    adata: AnnData | SpatialData,
    *,
    spatial_key: str = "spatial",
    table_key: str | None = None,
    elements_to_coordinate_systems: dict[str, str] | None = None,
    library_key: str | None = None,
    n_neighs: int = 6,
    percentile: float | None = None,
    transform: Literal["spectral", "cosine"] | None = None,
    set_diag: bool = False,
    key_added: str = "spatial",
    copy: bool = False,
) -> SpatialNeighborsResult | None:
    """Build a directed kNN graph from ``adata.obsm[spatial_key]``.

    ``n_neighs`` counts other observations, including coincident ones, and must
    be smaller than every library of the categorical ``library_key``; libraries
    get independent graphs. ``percentile`` prunes distances above that
    percentile of each library, counting stored zeros such as the diagonal.
    Coincident neighbors stay connected but have no stored distance.
    ``transform`` rescales connectivities by column degrees (``'spectral'``) or
    row similarity (``'cosine'``) after ``set_diag`` adds self-loops. Graphs use
    float32. With ``copy=True``, return SciPy CSR matrices; otherwise store
    ``'{key_added}_connectivities'`` and ``'{key_added}_distances'`` in
    ``adata.obsp`` and the parameters in ``adata.uns['{key_added}_neighbors']``.

    Parameters
    ----------
    adata
        Annotated data matrix, or a SpatialData object containing the selected table.
        SpatialData is optional and imported only for SpatialData input.
    spatial_key
        Key in ``adata.obsm`` containing coordinates. For SpatialData, transformed
        element centroids (x, y) are written here in table observation order.
    table_key
        Key in ``SpatialData.tables``. Required for SpatialData input; ignored
        for AnnData input.
    elements_to_coordinate_systems
        Mapping from spatial element names to coordinate system names. Required
        for SpatialData input and must cover all regions annotated by the table.
        Graphs are built independently per region; ignored for AnnData input.
    library_key
        Categorical column in ``adata.obs`` defining independent libraries.
        For SpatialData, the table region key is used instead.
    n_neighs
        Number of nearest other observations. Must be smaller than the number
        of observations in each library when using a kNN base graph.
    percentile
        Prune edges above this distance percentile (0 to 100) independently
        per library, after radius pruning. Stored zeros, including the diagonal,
        contribute to the percentile. ``None`` disables percentile pruning.
    transform
        Connectivity transform: ``None``, ``"spectral"`` (normalization by column
        degrees), or ``"cosine"`` (similarity between connectivity rows). Applied
        after pruning and adding the requested diagonal; distances are unchanged.
    set_diag
        Whether to set connectivity self-loops before transforming the graph.
        Distance diagonals are zero; cosine can create connectivity self-loops
        even when this is ``False``.
    key_added
        Prefix for graph keys in ``adata.obsp`` and neighbor metadata in
        ``adata.uns``. For SpatialData, results are stored in the selected table.
    copy
        Return the graph matrices instead of storing graphs and metadata.
        For SpatialData, centroid coordinates are still written to the table.

    Returns
    -------
    SpatialNeighborsResult | None
        With ``copy=True``, return connectivity and distance matrices as SciPy CSR
        matrices. Built-in builders store float32 connectivities and distances;
        Squidpy stores Euclidean distances as float64. Otherwise, return ``None``
        and store ``'{key_added}_connectivities'`` and ``'{key_added}_distances'``
        in ``adata.obsp``, and graph parameters in
        ``adata.uns['{key_added}_neighbors']``.
        For SpatialData, ``adata`` here denotes the selected table.
    """
    return _run(nb.KNNBuilder, **locals())


def spatial_neighbors_radius(
    adata: AnnData | SpatialData,
    *,
    radius: float | tuple[float, float],
    spatial_key: str = "spatial",
    table_key: str | None = None,
    elements_to_coordinate_systems: dict[str, str] | None = None,
    library_key: str | None = None,
    percentile: float | None = None,
    transform: Literal["spectral", "cosine"] | None = None,
    set_diag: bool = False,
    key_added: str = "spatial",
    copy: bool = False,
) -> SpatialNeighborsResult | None:
    """Connect observations within an inclusive ``radius``.

    A pair ``radius`` is an interval: neighbors up to its maximum are found,
    then shorter edges are pruned before ``percentile``. Zero connects
    coincident points.

    Parameters
    ----------
    adata
        Annotated data matrix, or a SpatialData object containing the selected table.
        SpatialData is optional and imported only for SpatialData input.
    radius
        Maximum inclusive Euclidean distance, or an inclusive ``(min, max)``
        interval. Zero connects coincident observations.
    spatial_key
        Key in ``adata.obsm`` containing coordinates. For SpatialData, transformed
        element centroids (x, y) are written here in table observation order.
    table_key
        Key in ``SpatialData.tables``. Required for SpatialData input; ignored
        for AnnData input.
    elements_to_coordinate_systems
        Mapping from spatial element names to coordinate system names. Required
        for SpatialData input and must cover all regions annotated by the table.
        Graphs are built independently per region; ignored for AnnData input.
    library_key
        Categorical column in ``adata.obs`` defining independent libraries.
        For SpatialData, the table region key is used instead.
    percentile
        Prune edges above this distance percentile (0 to 100) independently
        per library, after radius pruning. Stored zeros, including the diagonal,
        contribute to the percentile. ``None`` disables percentile pruning.
    transform
        Connectivity transform: ``None``, ``"spectral"`` (normalization by column
        degrees), or ``"cosine"`` (similarity between connectivity rows). Applied
        after pruning and adding the requested diagonal; distances are unchanged.
    set_diag
        Whether to set connectivity self-loops before transforming the graph.
        Distance diagonals are zero; cosine can create connectivity self-loops
        even when this is ``False``.
    key_added
        Prefix for graph keys in ``adata.obsp`` and neighbor metadata in
        ``adata.uns``. For SpatialData, results are stored in the selected table.
    copy
        Return the graph matrices instead of storing graphs and metadata.
        For SpatialData, centroid coordinates are still written to the table.

    Returns
    -------
    SpatialNeighborsResult | None
        With ``copy=True``, return connectivity and distance matrices as SciPy CSR
        matrices. Built-in builders store float32 connectivities and distances;
        Squidpy stores Euclidean distances as float64. Otherwise, return ``None``
        and store ``'{key_added}_connectivities'`` and ``'{key_added}_distances'``
        in ``adata.obsp``, and graph parameters in
        ``adata.uns['{key_added}_neighbors']``.
        For SpatialData, ``adata`` here denotes the selected table.
    """
    return _run(nb.RadiusBuilder, **locals())


def spatial_neighbors_delaunay(
    adata: AnnData | SpatialData,
    *,
    spatial_key: str = "spatial",
    table_key: str | None = None,
    elements_to_coordinate_systems: dict[str, str] | None = None,
    library_key: str | None = None,
    radius: float | tuple[float, float] | None = None,
    percentile: float | None = None,
    transform: Literal["spectral", "cosine"] | None = None,
    set_diag: bool = False,
    key_added: str = "spatial",
    copy: bool = False,
) -> SpatialNeighborsResult | None:
    """Build a Delaunay graph with Euclidean edge lengths.

    ``radius`` optionally prunes edges to an inclusive interval before
    ``percentile``; a scalar means ``(0, radius)``. Larger 2D inputs use
    validated GPU triangulation; small, 3D, and failed inputs use SciPy/Qhull.
    Of duplicate coordinates, one observation gets the edges.

    Parameters
    ----------
    adata
        Annotated data matrix, or a SpatialData object containing the selected table.
        SpatialData is optional and imported only for SpatialData input.
    spatial_key
        Key in ``adata.obsm`` containing coordinates. For SpatialData, transformed
        element centroids (x, y) are written here in table observation order.
    table_key
        Key in ``SpatialData.tables``. Required for SpatialData input; ignored
        for AnnData input.
    elements_to_coordinate_systems
        Mapping from spatial element names to coordinate system names. Required
        for SpatialData input and must cover all regions annotated by the table.
        Graphs are built independently per region; ignored for AnnData input.
    library_key
        Categorical column in ``adata.obs`` defining independent libraries.
        For SpatialData, the table region key is used instead.
    radius
        Optional inclusive Euclidean distance cutoff. A scalar means
        ``(0, radius)``; a pair specifies the interval. ``None`` keeps all edges.
    percentile
        Prune edges above this distance percentile (0 to 100) independently
        per library, after radius pruning. Stored zeros, including the diagonal,
        contribute to the percentile. ``None`` disables percentile pruning.
    transform
        Connectivity transform: ``None``, ``"spectral"`` (normalization by column
        degrees), or ``"cosine"`` (similarity between connectivity rows). Applied
        after pruning and adding the requested diagonal; distances are unchanged.
    set_diag
        Whether to set connectivity self-loops before transforming the graph.
        Distance diagonals are zero; cosine can create connectivity self-loops
        even when this is ``False``.
    key_added
        Prefix for graph keys in ``adata.obsp`` and neighbor metadata in
        ``adata.uns``. For SpatialData, results are stored in the selected table.
    copy
        Return the graph matrices instead of storing graphs and metadata.
        For SpatialData, centroid coordinates are still written to the table.

    Returns
    -------
    SpatialNeighborsResult | None
        With ``copy=True``, return connectivity and distance matrices as SciPy CSR
        matrices. Built-in builders store float32 connectivities and distances;
        Squidpy stores Euclidean distances as float64. Otherwise, return ``None``
        and store ``'{key_added}_connectivities'`` and ``'{key_added}_distances'``
        in ``adata.obsp``, and graph parameters in
        ``adata.uns['{key_added}_neighbors']``.
        For SpatialData, ``adata`` here denotes the selected table.
    """
    return _run(nb.DelaunayBuilder, **locals())


def spatial_neighbors_grid(
    adata: AnnData | SpatialData,
    *,
    spatial_key: str = "spatial",
    table_key: str | None = None,
    elements_to_coordinate_systems: dict[str, str] | None = None,
    library_key: str | None = None,
    n_neighs: int = 6,
    n_rings: int = 1,
    delaunay: bool = False,
    transform: Literal["spectral", "cosine"] | None = None,
    set_diag: bool = False,
    key_added: str = "spatial",
    copy: bool = False,
) -> SpatialNeighborsResult | None:
    """Build a lattice graph whose distances count shortest directed hops.

    The kNN base keeps edges shorter than 1.3 times the median distance; use
    ``n_neighs=6`` for hexagonal and ``4`` for square grids. ``n_rings`` adds
    hops; ``delaunay=True`` uses Delaunay edges as the base instead.

    Parameters
    ----------
    adata
        Annotated data matrix, or a SpatialData object containing the selected table.
        SpatialData is optional and imported only for SpatialData input.
    spatial_key
        Key in ``adata.obsm`` containing coordinates. For SpatialData, transformed
        element centroids (x, y) are written here in table observation order.
    table_key
        Key in ``SpatialData.tables``. Required for SpatialData input; ignored
        for AnnData input.
    elements_to_coordinate_systems
        Mapping from spatial element names to coordinate system names. Required
        for SpatialData input and must cover all regions annotated by the table.
        Graphs are built independently per region; ignored for AnnData input.
    library_key
        Categorical column in ``adata.obs`` defining independent libraries.
        For SpatialData, the table region key is used instead.
    n_neighs
        Number of nearest other observations. Must be smaller than the number
        of observations in each library when using a kNN base graph.
    n_rings
        Number of grid hops to include. Distances count shortest directed hops.
    delaunay
        Use Delaunay edges as the grid base instead of kNN edges filtered to
        distances below 1.3 times the per-library median.
    transform
        Connectivity transform: ``None``, ``"spectral"`` (normalization by column
        degrees), or ``"cosine"`` (similarity between connectivity rows). Applied
        after pruning and adding the requested diagonal; distances are unchanged.
    set_diag
        Whether to set connectivity self-loops before transforming the graph.
        Distance diagonals are zero; cosine can create connectivity self-loops
        even when this is ``False``.
    key_added
        Prefix for graph keys in ``adata.obsp`` and neighbor metadata in
        ``adata.uns``. For SpatialData, results are stored in the selected table.
    copy
        Return the graph matrices instead of storing graphs and metadata.
        For SpatialData, centroid coordinates are still written to the table.

    Returns
    -------
    SpatialNeighborsResult | None
        With ``copy=True``, return connectivity and distance matrices as SciPy CSR
        matrices. Built-in builders use float32. Otherwise, return ``None`` and
        store ``'{key_added}_connectivities'`` and ``'{key_added}_distances'`` in
        ``adata.obsp``, and graph parameters in ``adata.uns['{key_added}_neighbors']``.
        For SpatialData, ``adata`` here denotes the selected table.
    """
    return _run(nb.GridBuilder, **locals())


def _run(builder, adata, **kwargs):
    """Call :func:`spatial_neighbors_from_builder` with the mode's builder."""
    shared = {key: kwargs.pop(key) for key in _SHARED}
    return spatial_neighbors_from_builder(adata, builder(**kwargs), **shared)


def spatial_neighbors_from_builder(
    adata: AnnData | SpatialData,
    builder: nb.GraphBuilder,
    *,
    spatial_key: str = "spatial",
    table_key: str | None = None,
    elements_to_coordinate_systems: dict[str, str] | None = None,
    library_key: str | None = None,
    key_added: str = "spatial",
    copy: bool = False,
) -> SpatialNeighborsResult | None:
    """Build spatial graphs with a built-in or custom builder.

    ``builder.build(coords)`` receives finite float32 CuPy coordinates and
    returns connectivity and distance matrices; ``uns_params()`` supplies the
    metadata. With ``library_key``, ``combine(graphs, indices)`` merges the
    per-library graphs in observation order. Builders from
    :mod:`rapids_singlecell.gr.neighbors` with built-in postprocessors build all
    libraries in one pass.

    Parameters
    ----------
    adata
        Annotated data matrix, or a SpatialData object containing the selected table.
        SpatialData is optional and imported only for SpatialData input.
    builder
        Graph construction strategy. ``build(coords)`` receives finite float32
        CuPy coordinates and returns connectivity and distance matrices.
        ``uns_params()`` supplies metadata; ``combine(graphs, indices)`` must
        support independent libraries when ``library_key`` is used.
    spatial_key
        Key in ``adata.obsm`` containing coordinates. For SpatialData, transformed
        element centroids (x, y) are written here in table observation order.
    table_key
        Key in ``SpatialData.tables``. Required for SpatialData input; ignored
        for AnnData input.
    elements_to_coordinate_systems
        Mapping from spatial element names to coordinate system names. Required
        for SpatialData input and must cover all regions annotated by the table.
        Graphs are built independently per region; ignored for AnnData input.
    library_key
        Categorical column in ``adata.obs`` defining independent libraries.
        For SpatialData, the table region key is used instead.
    key_added
        Prefix for graph keys in ``adata.obsp`` and neighbor metadata in
        ``adata.uns``. For SpatialData, results are stored in the selected table.
    copy
        Return the graph matrices instead of storing graphs and metadata.
        For SpatialData, centroid coordinates are still written to the table.

    Returns
    -------
    SpatialNeighborsResult | None
        With ``copy=True``, return connectivity and distance matrices as SciPy CSR
        matrices. Built-in builders store float32 connectivities and distances;
        Squidpy stores Euclidean distances as float64. Otherwise, return ``None``
        and store ``'{key_added}_connectivities'`` and ``'{key_added}_distances'``
        in ``adata.obsp``, and graph parameters in
        ``adata.uns['{key_added}_neighbors']``.
        For SpatialData, ``adata`` here denotes the selected table.
    """
    adata, library_key = _resolve_spatial_data(
        adata,
        table_key=table_key,
        elements_to_coordinate_systems=elements_to_coordinate_systems,
        spatial_key=spatial_key,
        library_key=library_key,
    )
    _assert_spatial_basis(adata, key=spatial_key)
    coords = nb._validate_coordinates(adata.obsm[spatial_key])
    codes = None
    if library_key is not None:
        _assert_categorical_obs(adata, key=library_key)
        codes = adata.obs[library_key].cat.codes.to_numpy()
        if np.any(codes < 0):
            raise ValueError(f"`adata.obs[{library_key!r}]` has missing labels.")
    single = codes is None or codes.min() == codes.max()
    if type(builder) in _BUILDERS and {*map(type, builder.postprocessors())} <= _STEPS:
        result = builder._build(coords, None if single else cp.asarray(codes, "int32"))
    elif codes is None:
        result = builder.build(coords)
    else:
        order = np.argsort(codes, kind="stable")
        groups = np.split(order, np.flatnonzero(np.diff(codes[order])) + 1)
        graphs = [builder.build(coords[cp.asarray(group)]) for group in groups]
        result = builder.combine(graphs, order)
    result = SpatialNeighborsResult(
        *(scipy.sparse.csr_matrix(g.get() if hasattr(g, "get") else g) for g in result)
    )
    if copy:
        return result
    if adata.is_view:
        adata._init_as_actual(adata.copy())
    conns_key, dists_key = f"{key_added}_connectivities", f"{key_added}_distances"
    adata.obsp[conns_key], adata.obsp[dists_key] = result
    keys = {"connectivities_key": conns_key, "distances_key": dists_key}
    adata.uns[f"{key_added}_neighbors"] = {**keys, "params": builder.uns_params()}
