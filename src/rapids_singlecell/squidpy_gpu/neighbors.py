"""GPU graph builders and reusable spatial graph postprocessors."""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from numbers import Integral, Real
from typing import Any, TypeVar

import cupy as cp
import numpy as np
from cupyx.scipy import sparse as cp_sparse

from ._spatial_delaunay import _delaunay_edges
from ._spatial_graph import build_adjacency, build_graphs
from ._spatial_neighbors_backend import _edge_distances, _knn_edges, _radius_edges

__all__ = [
    "GraphMatrixT",
    "GraphBuilder",
    "GraphBuilderCSR",
    "GraphPostprocessor",
    "DistanceIntervalPostprocessor",
    "PercentilePostprocessor",
    "TransformPostprocessor",
    "KNNBuilder",
    "RadiusBuilder",
    "DelaunayBuilder",
    "GridBuilder",
]

GraphMatrixT = TypeVar("GraphMatrixT")
GraphPostprocessor = Callable[
    [GraphMatrixT, GraphMatrixT], tuple[GraphMatrixT, GraphMatrixT]
]
type CSR = cp_sparse.csr_matrix
type CSRPair = tuple[CSR, CSR]


class GraphBuilder[CoordT, GraphMatrixT](ABC):
    """Base interface for spatial graph construction strategies.

    Implement :meth:`build_graph` and :meth:`uns_params`; postprocessors run in
    order. Override :meth:`combine` for custom matrices with independent libraries.
    """

    def __init__(
        self,
        transform: str | None = None,
        *,
        set_diag: bool = False,
        percentile: float | None = None,
        postprocessors: Sequence[GraphPostprocessor[GraphMatrixT]] = (),
    ) -> None:
        self.transform = _validate_transform(transform)
        self.set_diag = set_diag
        self.percentile = _validate_percentile(percentile)
        self._postprocessors = list(postprocessors)

    def build(self, coords: CoordT) -> tuple[GraphMatrixT, GraphMatrixT]:
        """Construct a graph and apply its postprocessors in order."""
        adj, dst = self.build_graph(coords)
        for postprocessor in self.postprocessors():
            adj, dst = postprocessor(adj, dst)
        return adj, dst

    @abstractmethod
    def build_graph(self, coords: CoordT) -> tuple[GraphMatrixT, GraphMatrixT]:
        """Construct raw adjacency and distance matrices."""

    def postprocessors(self) -> Sequence[GraphPostprocessor[GraphMatrixT]]:
        """Return processing steps for the raw adjacency and distances."""
        return self._postprocessors

    @abstractmethod
    def uns_params(self) -> dict[str, Any]:
        """Return graph parameters to store in ``adata.uns``."""

    def combine(
        self,
        mats: Sequence[tuple[GraphMatrixT, GraphMatrixT]],
        ixs: Sequence[int],
    ) -> tuple[GraphMatrixT, GraphMatrixT]:
        """Combine per-library graphs in original observation order."""
        raise NotImplementedError(
            "Using `library_key` with this graph builder is not implemented yet."
        )


class GraphBuilderCSR(GraphBuilder[cp.ndarray, cp_sparse.csr_matrix], ABC):
    """CuPy CSR strategy with float32 coordinate validation and library combination."""

    def build(self, coords: cp.ndarray) -> CSRPair:
        return self._build_validated(_validate_coordinates(coords))

    def _build_validated(self, coords: cp.ndarray) -> CSRPair:
        """Build from coordinates already validated by the AnnData runner."""
        return super().build(coords)

    def combine(self, mats: Sequence[CSRPair], ixs: Sequence[int]) -> CSRPair:
        if (
            len(mats) == 1
            and isinstance(ixs, np.ndarray)
            and not np.any(np.diff(ixs) < 0)
        ):
            return mats[0]
        adj = cp_sparse.block_diag([mat[0] for mat in mats], format="csr")
        dst = cp_sparse.block_diag([mat[1] for mat in mats], format="csr")
        indices = cp.asarray(ixs)
        if indices.size and bool(cp.any(cp.diff(indices) < 0)):
            order = cp.argsort(indices)
            adj = adj[order, :][:, order]
            dst = dst[order, :][:, order]
        return adj, dst


class _SpatialBuilder(GraphBuilderCSR):
    """Shared setup and metadata for the built-in graph strategies."""

    _coord_type = "generic"
    _parameters: tuple[str, ...] = ()

    def __init__(
        self,
        transform: str | None,
        *,
        set_diag: bool,
        percentile: float | None = None,
    ) -> None:
        steps = []
        interval = getattr(self, "radius", None)
        if isinstance(interval, tuple):
            steps.append(DistanceIntervalPostprocessor(interval))
        if percentile is not None:
            steps.append(PercentilePostprocessor(percentile))
        steps.append(TransformPostprocessor(transform))
        super().__init__(
            transform, set_diag=set_diag, percentile=percentile, postprocessors=steps
        )

    def uns_params(self) -> dict[str, Any]:
        params = {name: getattr(self, name) for name in self._parameters}
        if "n_neighs" in params:
            params["n_neighbors"] = params.pop("n_neighs")
        return {"coord_type": self._coord_type, **params, "transform": self.transform}

    def _from_edges(
        self, coords: cp.ndarray, edges: tuple[cp.ndarray, cp.ndarray, cp.ndarray]
    ) -> CSRPair:
        rows, cols, values = edges
        _validate_distances(values)
        return build_graphs(rows, cols, values, len(coords), set_diag=self.set_diag)


class KNNBuilder(_SpatialBuilder):
    """Find ``n_neighs`` other observations using float32 coordinates and distances."""

    _parameters = ("n_neighs",)

    def __init__(
        self,
        n_neighs: int = 6,
        *,
        transform: str | None = None,
        set_diag: bool = False,
        percentile: float | None = None,
    ) -> None:
        self.n_neighs = _validate_n_neighs(n_neighs)
        super().__init__(transform, set_diag=set_diag, percentile=percentile)

    def build_graph(self, coords: cp.ndarray) -> CSRPair:
        _validate_neighbor_count(coords, self.n_neighs)
        return self._from_edges(coords, _knn_edges(coords, self.n_neighs))


class RadiusBuilder(_SpatialBuilder):
    """Connect observations within a scalar radius or inclusive radius interval."""

    _parameters = ("radius",)

    def __init__(
        self,
        radius: float | tuple[float, float],
        *,
        transform: str | None = None,
        set_diag: bool = False,
        percentile: float | None = None,
    ) -> None:
        self.radius = _validate_radius(radius)
        super().__init__(transform, set_diag=set_diag, percentile=percentile)

    def build_graph(self, coords: cp.ndarray) -> CSRPair:
        upper = max(self.radius) if isinstance(self.radius, tuple) else self.radius
        return self._from_edges(coords, _radius_edges(coords, upper))


class DelaunayBuilder(_SpatialBuilder):
    """Connect Delaunay neighbors and store their Euclidean distances.

    Larger, nondegenerate 2D inputs use validated GPU triangulation; 3D,
    duplicates, and ambiguous/unsupported geometry use SciPy/Qhull. Distances,
    pruning, and transforms run on GPU. Scalar ``radius`` prunes to ``(0, radius)``.
    Coordinates and graph data use float32; triangulation requires double internally.
    """

    _parameters = ("radius",)

    def __init__(
        self,
        radius: float | tuple[float, float] | None = None,
        *,
        transform: str | None = None,
        set_diag: bool = False,
        percentile: float | None = None,
    ) -> None:
        if radius is not None:
            radius = _validate_radius(radius)
            if not isinstance(radius, tuple):
                radius = (0.0, radius)
        self.radius = radius
        super().__init__(transform, set_diag=set_diag, percentile=percentile)

    def build_graph(self, coords: cp.ndarray) -> CSRPair:
        rows, cols = _delaunay_edges(coords)
        values = _edge_distances(coords, rows, cols)
        return self._from_edges(coords, (rows, cols, values))


class GridBuilder(_SpatialBuilder):
    """Build a lattice graph whose distances count shortest directed grid hops.

    The base uses neighbors closer than 1.3 times the median distance, or
    Delaunay edges with ``delaunay=True``. Additional rings expand on the GPU.
    """

    _coord_type = "grid"
    _parameters = ("n_neighs", "n_rings", "delaunay")

    def __init__(
        self,
        n_neighs: int = 6,
        *,
        n_rings: int = 1,
        delaunay: bool = False,
        transform: str | None = None,
        set_diag: bool = False,
    ) -> None:
        self.n_neighs = _validate_n_neighs(n_neighs)
        self.n_rings = _validate_n_neighs(n_rings, name="n_rings")
        self.delaunay = delaunay
        super().__init__(transform, set_diag=set_diag)

    def build_graph(self, coords: cp.ndarray) -> CSRPair:
        if self.n_rings == 1:
            adj = self._base_adjacency(coords, set_diag=self.set_diag)
            dst = adj.copy()
        else:
            base = self._base_adjacency(coords, set_diag=True)
            visited = frontier = dst = base
            for ring in range(2, self.n_rings + 1):
                frontier = (frontier @ base).tocsr()
                frontier.data[:] = 1.0
                frontier = frontier - frontier.multiply(visited)
                frontier.eliminate_zeros()
                if not frontier.nnz:
                    break
                if ring < self.n_rings:
                    visited = visited + frontier
                dst = dst + frontier * float(ring)
            dst.setdiag(float(self.set_diag))
            dst.eliminate_zeros()
            adj = dst.copy()
            adj.data[:] = 1.0
        dst.setdiag(0.0)
        return adj, dst

    def _base_adjacency(self, coords: cp.ndarray, *, set_diag: bool) -> CSR:
        if self.delaunay:
            rows, cols = _delaunay_edges(coords)
        else:
            _validate_neighbor_count(coords, self.n_neighs)
            rows, cols, values = _knn_edges(coords, self.n_neighs)
            _validate_distances(values)
            mask = values < cp.median(values) * 1.3
            rows, cols = rows[mask], cols[mask]
        adj = build_adjacency(rows, cols, len(coords), set_diag=set_diag)
        # Searches emit unique neighbors and the assembly kernel adds one
        # diagonal per row. Sorting therefore also establishes canonical CSR,
        # avoiding a COO duplicate-reduction round trip before ring expansion.
        adj.sort_indices()
        adj.has_canonical_format = True
        return adj


@dataclass(frozen=True)
class DistanceIntervalPostprocessor:
    """Prune distances outside an inclusive interval, preserving adjacency diagonal."""

    interval: tuple[float, float]

    def __post_init__(self) -> None:
        interval = _validate_radius(self.interval)
        if not isinstance(interval, tuple):
            raise ValueError("`interval` must contain two nonnegative finite numbers.")
        object.__setattr__(self, "interval", tuple(sorted(interval)))

    def __call__(self, adj: CSR, dst: CSR) -> CSRPair:
        lower, upper = self.interval
        # Keep Python/float64 interval bounds precise for float32 distances.
        # Casting a bound such as 1 + 1e-8 to float32 would retain length 1.
        values = dst.data.astype(cp.float64, copy=False)
        outside = (values < lower) | (values > upper)
        diagonal = adj.diagonal()
        adj = _remove_distance_edges(adj, dst, outside)
        adj.setdiag(diagonal)
        return adj, dst


@dataclass(frozen=True)
class PercentilePostprocessor:
    """Prune distances above a percentile, including all stored zeros in its input."""

    percentile: float

    def __post_init__(self) -> None:
        value = _validate_percentile(self.percentile)
        if value is None:
            raise ValueError("`percentile` must be a finite number between 0 and 100.")
        object.__setattr__(self, "percentile", value)

    def __call__(self, adj: CSR, dst: CSR) -> CSRPair:
        if dst.nnz:
            threshold = cp.percentile(dst.data, self.percentile)
            adj = _remove_distance_edges(adj, dst, dst.data > threshold)
        return adj, dst


@dataclass(frozen=True)
class TransformPostprocessor:
    """Remove stored zeros and apply spectral or cosine connectivity transformation."""

    transform: str | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "transform", _validate_transform(self.transform))

    def __call__(self, adj: CSR, dst: CSR) -> CSRPair:
        adj.eliminate_zeros()
        dst.eliminate_zeros()
        if self.transform == "spectral":
            degrees = cp.asarray(adj.sum(axis=0)).ravel()
            scale = cp.zeros_like(degrees)
            positive = degrees > 0
            scale[positive] = 1 / cp.sqrt(degrees[positive])
            adj = adj.multiply(scale[:, None]).multiply(scale[None, :]).tocsr()
        elif self.transform == "cosine":
            norm = cp.sqrt(cp.asarray(adj.multiply(adj).sum(axis=1)).ravel())
            scale = cp.zeros_like(norm)
            positive = norm > 0
            scale[positive] = 1 / norm[positive]
            normalized = adj.multiply(scale[:, None]).tocsr()
            adj = (normalized @ normalized.T).tocsr()
        if self.transform is not None:
            adj.eliminate_zeros()
        adj.sort_indices()
        dst.sort_indices()
        return adj, dst


def _remove_distance_edges(adj: CSR, dst: CSR, outside: cp.ndarray) -> CSR:
    # Adjacency and distances need not share CSR storage, especially after a
    # previous postprocessor removed zero-valued adjacency edges. Locate the
    # edges through a sparse mask rather than indexing adjacency.data directly.
    removal = dst.copy()
    removal.data = outside.astype(adj.dtype)
    removal.eliminate_zeros()
    dst.data[outside] = 0.0
    return (adj - adj.multiply(removal)).tocsr()


def _validate_coordinates(coords: cp.ndarray) -> cp.ndarray:
    if hasattr(coords, "to_numpy"):
        coords = coords.to_numpy()
    if (
        not isinstance(coords, (np.ndarray, cp.ndarray))
        or coords.dtype.kind not in "iuf"
    ):
        raise TypeError("Spatial coordinates must be a dense array of real numbers.")
    if coords.ndim != 2 or not coords.shape[0] or not coords.shape[1]:
        raise ValueError(
            "Spatial coordinates must be a nonempty two-dimensional array."
        )
    coords = cp.asarray(coords, dtype=cp.float32, order="C")
    if not bool(cp.isfinite(coords).all()):
        raise ValueError("Spatial coordinates must be finite in float32. Rescale them.")
    return coords


def _validate_distances(values: cp.ndarray) -> None:
    if not bool(cp.isfinite(values).all()):
        raise ValueError(
            "Spatial distances exceed the coordinate dtype's finite range. "
            "Rescale the coordinates."
        )


def _validate_neighbor_count(coords: cp.ndarray, n_neighs: int) -> None:
    if len(coords) <= n_neighs:
        raise ValueError(
            "`n_neighs` must be smaller than the number of observations in every library."
        )


def _validate_n_neighs(value: int, *, name: str = "n_neighs") -> int:
    if (
        isinstance(value, (bool, np.bool_))
        or not isinstance(value, Integral)
        or value < 1
    ):
        raise ValueError(f"`{name}` must be a positive integer.")
    return int(value)


def _validate_radius(
    radius: float | tuple[float, float],
) -> float | tuple[float, float]:
    values = radius if isinstance(radius, tuple) else (radius,)
    if (isinstance(radius, tuple) and len(radius) != 2) or any(
        isinstance(value, (bool, np.bool_))
        or not isinstance(value, Real)
        or not np.isfinite(value)
        or value < 0
        for value in values
    ):
        raise ValueError(
            "`radius` must be a nonnegative finite number or a pair of them."
        )
    return (
        tuple(float(value) for value in radius)
        if isinstance(radius, tuple)
        else float(radius)
    )


def _validate_transform(transform: str | None) -> str | None:
    transform = getattr(transform, "value", transform)
    if transform not in (None, "spectral", "cosine"):
        raise ValueError("`transform` must be None, 'spectral', or 'cosine'.")
    return transform


def _validate_percentile(percentile: float | None) -> float | None:
    if percentile is not None and (
        isinstance(percentile, (bool, np.bool_))
        or not isinstance(percentile, Real)
        or not np.isfinite(percentile)
        or not 0 <= percentile <= 100
    ):
        raise ValueError("`percentile` must be a finite number between 0 and 100.")
    return None if percentile is None else float(percentile)
