from __future__ import annotations

from typing import TYPE_CHECKING, Literal, NamedTuple

import cupy as cp
import numpy as np
import scipy.sparse

from . import neighbors as nb
from ._utils import _assert_categorical_obs, _assert_spatial_basis

if TYPE_CHECKING:
    from anndata import AnnData

_SHARED = ("spatial_key", "library_key", "key_added", "copy")
# Exact built-in types build all libraries in one pass; subclasses may override.
_BUILDERS = (nb.KNNBuilder, nb.RadiusBuilder, nb.DelaunayBuilder, nb.GridBuilder)
_STEPS = {
    nb.DistanceIntervalPostprocessor,
    nb.PercentilePostprocessor,
    nb.TransformPostprocessor,
}


class SpatialNeighborsResult(NamedTuple):
    """Spatial connectivity and distance matrices returned with ``copy=True``."""

    connectivities: scipy.sparse.csr_matrix
    distances: scipy.sparse.csr_matrix


def spatial_neighbors_knn(
    adata: AnnData,
    *,
    spatial_key: str = "spatial",
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
    """
    return _run(nb.KNNBuilder, **locals())


def spatial_neighbors_radius(
    adata: AnnData,
    *,
    radius: float | tuple[float, float],
    spatial_key: str = "spatial",
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
    coincident points. Other parameters follow :func:`spatial_neighbors_knn`.
    """
    return _run(nb.RadiusBuilder, **locals())


def spatial_neighbors_delaunay(
    adata: AnnData,
    *,
    spatial_key: str = "spatial",
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
    Of duplicate coordinates, one observation gets the edges. Other parameters
    follow :func:`spatial_neighbors_knn`.
    """
    return _run(nb.DelaunayBuilder, **locals())


def spatial_neighbors_grid(
    adata: AnnData,
    *,
    spatial_key: str = "spatial",
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
    hops; ``delaunay=True`` uses Delaunay edges as the base instead. Other
    parameters follow :func:`spatial_neighbors_knn`.
    """
    return _run(nb.GridBuilder, **locals())


def _run(builder, adata, **kwargs):
    """Call :func:`spatial_neighbors_from_builder` with the mode's builder."""
    shared = {key: kwargs.pop(key) for key in _SHARED}
    return spatial_neighbors_from_builder(adata, builder(**kwargs), **shared)


def spatial_neighbors_from_builder(
    adata: AnnData,
    builder: nb.GraphBuilder,
    *,
    spatial_key: str = "spatial",
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
    libraries in one pass. Other parameters follow :func:`spatial_neighbors_knn`.
    """
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
    elif single:
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
