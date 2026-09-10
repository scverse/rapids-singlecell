from __future__ import annotations

from typing import TYPE_CHECKING, Literal, NamedTuple

import cupy as cp
import numpy as np
from scipy import sparse as sc_sparse

from ._utils import _assert_categorical_obs, _assert_spatial_basis
from .neighbors import (
    DelaunayBuilder,
    GridBuilder,
    KNNBuilder,
    RadiusBuilder,
    _validate_coordinates,
)

if TYPE_CHECKING:
    from anndata import AnnData

    from .neighbors import GraphBuilder

_Transform = Literal["spectral", "cosine"] | None


class SpatialNeighborsResult(NamedTuple):
    """Spatial connectivity and distance matrices returned with ``copy=True``."""

    connectivities: sc_sparse.csr_matrix
    distances: sc_sparse.csr_matrix


def spatial_neighbors_knn(
    adata: AnnData,
    *,
    spatial_key: str = "spatial",
    library_key: str | None = None,
    n_neighs: int = 6,
    percentile: float | None = None,
    transform: _Transform = None,
    set_diag: bool = False,
    key_added: str = "spatial",
    copy: bool = False,
) -> SpatialNeighborsResult | None:
    """Build a directed kNN graph from ``adata.obsm[spatial_key]``.

    ``n_neighs`` counts other observations, including coincident points, and
    must be smaller than every library. Coordinates must be dense and real.
    A categorical ``library_key`` builds independent graphs in observation
    order; missing labels are rejected and unused categories are ignored.

    ``transform=None`` gives binary connectivity. ``'spectral'`` normalizes
    by column degrees; ``'cosine'`` compares adjacency rows. ``set_diag``
    adds self-loops before transformation; cosine can introduce a diagonal
    even when it is false. Distances are not transformed.

    ``percentile`` prunes long edges independently per library. Stored
    zero distances, including the diagonal, contribute to the percentile.
    Zero-distance neighbors remain connected even though sparse distance
    zeros are removed.

    With ``copy=True``, return :class:`SpatialNeighborsResult` of SciPy CSR
    matrices without mutation. Otherwise store ``'{key_added}_connectivities'``
    and ``'{key_added}_distances'`` in ``adata.obsp``, parameters in
    ``adata.uns['{key_added}_neighbors']``, and return ``None``.

    Search and distances use float32, without changing the stored coordinates.
    Search uses CuPy's GPU KDTree.
    Coordinates outside float32's finite range must be rescaled.
    """
    builder = KNNBuilder(
        n_neighs=n_neighs, percentile=percentile, transform=transform, set_diag=set_diag
    )
    return spatial_neighbors_from_builder(
        adata,
        builder,
        spatial_key=spatial_key,
        library_key=library_key,
        key_added=key_added,
        copy=copy,
    )


def spatial_neighbors_radius(
    adata: AnnData,
    *,
    radius: float | tuple[float, float],
    spatial_key: str = "spatial",
    library_key: str | None = None,
    percentile: float | None = None,
    transform: _Transform = None,
    set_diag: bool = False,
    key_added: str = "spatial",
    copy: bool = False,
) -> SpatialNeighborsResult | None:
    """Build a graph of observations within an inclusive radius.

    ``radius`` is a nonnegative finite maximum distance or a pair defining
    an interval (sorted before use). Zero connects distinct coincident points.
    Percentile pruning follows interval pruning and includes its stored zeros.

    Shared parameters and outputs follow :func:`spatial_neighbors_knn`.
    Coordinates and graph data use float32, as in the other spatial modes.
    The GPU index counts then fills edges without a dense pairwise matrix.
    """
    builder = RadiusBuilder(
        radius=radius, percentile=percentile, transform=transform, set_diag=set_diag
    )
    return spatial_neighbors_from_builder(
        adata,
        builder,
        spatial_key=spatial_key,
        library_key=library_key,
        key_added=key_added,
        copy=copy,
    )


def spatial_neighbors_delaunay(
    adata: AnnData,
    *,
    spatial_key: str = "spatial",
    library_key: str | None = None,
    radius: float | tuple[float, float] | None = None,
    percentile: float | None = None,
    transform: _Transform = None,
    set_diag: bool = False,
    key_added: str = "spatial",
    copy: bool = False,
) -> SpatialNeighborsResult | None:
    """Build a Delaunay graph with Euclidean edge lengths.

    ``radius`` optionally prunes to an inclusive interval; a scalar means
    ``(0, radius)``. Percentile pruning follows radius pruning.
    Shared parameters and outputs follow :func:`spatial_neighbors_knn`.

    Suitable larger 2D inputs use validated GPU triangulation. Small inputs,
    3D, duplicates, and ambiguous geometry fall back to SciPy/Qhull. Distances,
    pruning, and connectivity transforms run on the GPU in either case.
    Inputs and graph data use float32; CuPy/Qhull triangulation requires double
    internally, including the GPU geometry validation predicates.
    """
    builder = DelaunayBuilder(
        radius=radius, percentile=percentile, transform=transform, set_diag=set_diag
    )
    return spatial_neighbors_from_builder(
        adata,
        builder,
        spatial_key=spatial_key,
        library_key=library_key,
        key_added=key_added,
        copy=copy,
    )


def spatial_neighbors_grid(
    adata: AnnData,
    *,
    spatial_key: str = "spatial",
    library_key: str | None = None,
    n_neighs: int = 6,
    n_rings: int = 1,
    delaunay: bool = False,
    transform: _Transform = None,
    set_diag: bool = False,
    key_added: str = "spatial",
    copy: bool = False,
) -> SpatialNeighborsResult | None:
    """Build a lattice graph with shortest directed hop distances.

    The float32 kNN base keeps candidates closer than 1.3 times the median distance.
    Use ``n_neighs=6`` for hexagonal or ``4`` for square grids, and
    ``n_rings`` for additional hops. With ``delaunay=True``, the automatic
    Delaunay backend supplies the base graph and ``n_neighs`` is ignored.
    Candidates tied at the cutoff can differ from Squidpy's selection.

    Shared parameters and outputs follow :func:`spatial_neighbors_knn`;
    distances are ring numbers, not physical lengths.
    """
    builder = GridBuilder(
        n_neighs=n_neighs,
        n_rings=n_rings,
        delaunay=delaunay,
        transform=transform,
        set_diag=set_diag,
    )
    return spatial_neighbors_from_builder(
        adata,
        builder,
        spatial_key=spatial_key,
        library_key=library_key,
        key_added=key_added,
        copy=copy,
    )


def spatial_neighbors_from_builder(
    adata: AnnData,
    builder: GraphBuilder,
    *,
    spatial_key: str = "spatial",
    library_key: str | None = None,
    key_added: str = "spatial",
    copy: bool = False,
) -> SpatialNeighborsResult | None:
    """Build spatial graphs with a built-in or custom builder.

    ``builder.build(coords)`` receives contiguous float32 CuPy coordinates and returns
    connectivity and distance matrices; ``uns_params()`` supplies metadata.
    With ``library_key``, ``combine(graphs, indices)`` must combine the library
    graphs and restore observation order. Built-in builders are available in
    :mod:`rapids_singlecell.gr.neighbors`.

    Coordinate, library, storage, and ``copy`` behavior follow
    :func:`spatial_neighbors_knn`.
    """
    _assert_spatial_basis(adata, key=spatial_key)
    coords = _validate_coordinates(adata.obsm[spatial_key])

    if library_key is None:
        groups = None
    else:
        _assert_categorical_obs(adata, key=library_key)
        codes = adata.obs[library_key].cat.codes.to_numpy()
        if np.any(codes < 0):
            raise ValueError(
                f"Library labels in `adata.obs[{library_key!r}]` must not be missing."
            )
        groups = [np.flatnonzero(codes == code) for code in np.unique(codes)]
    # Coordinates were checked once above, including all libraries. Only bypass
    # repeated validation for the exact built-in types: custom subclasses may
    # override build() and must still receive their normal public entry point.
    build = (
        builder._build_validated
        if type(builder) in (KNNBuilder, RadiusBuilder, DelaunayBuilder, GridBuilder)
        else builder.build
    )
    if groups is None:
        result = build(coords)
    else:
        graphs = [build(coords[cp.asarray(group)]) for group in groups]
        result = builder.combine(graphs, np.concatenate(groups))
    result = SpatialNeighborsResult(
        *(
            sc_sparse.csr_matrix(graph.get() if hasattr(graph, "get") else graph)
            for graph in result
        )
    )
    if copy:
        return result
    if adata.is_view:
        adata._init_as_actual(adata.copy())
    conns_key, dists_key = f"{key_added}_connectivities", f"{key_added}_distances"
    adata.obsp[conns_key] = result.connectivities
    adata.obsp[dists_key] = result.distances
    adata.uns[f"{key_added}_neighbors"] = {
        "connectivities_key": conns_key,
        "distances_key": dists_key,
        "params": builder.uns_params(),
    }
    return None
