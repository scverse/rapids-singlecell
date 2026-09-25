from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from numbers import Integral, Real
from typing import Any

import cupy as cp
import numpy as np
from cupyx.scipy import sparse as cp_sparse

from rapids_singlecell._cuda import _spatial_cuda

from ._spatial_delaunay import _delaunay_edges
from ._spatial_graph import build_adjacency, build_graphs
from ._spatial_neighbors_backend import _edge_distances, _knn_edges, _radius_edges

type GraphPostprocessor[M] = Callable[[M, M], tuple[M, M]]
type CSR = cp_sparse.csr_matrix
type CSRPair = tuple[CSR, CSR]


class GraphBuilder[CoordT, GraphMatrixT](ABC):
    """Base class for spatial graph construction strategies.

    Implement :meth:`build_graph` and :meth:`uns_params`; :meth:`build` applies
    the postprocessors in order. Override :meth:`combine` to support libraries.
    """

    def __init__(
        self,
        transform: str | None = None,
        *,
        set_diag: bool = False,
        percentile: float | None = None,
        postprocessors: Sequence[GraphPostprocessor[GraphMatrixT]] = (),
    ) -> None:
        self.transform = _transform(transform)
        self.set_diag = set_diag
        self.percentile = percentile
        self._postprocessors = list(postprocessors)

    def build(self, coords: CoordT) -> tuple[GraphMatrixT, GraphMatrixT]:
        """Construct a graph and apply its postprocessors."""
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
        self, mats: Sequence[tuple[GraphMatrixT, GraphMatrixT]], ixs: Sequence[int]
    ) -> tuple[GraphMatrixT, GraphMatrixT]:
        """Combine per-library graphs in original observation order."""
        raise NotImplementedError("This graph builder does not support `library_key`.")


class GraphBuilderCSR(GraphBuilder[cp.ndarray, cp_sparse.csr_matrix], ABC):
    """CuPy CSR strategy that builds from finite float32 coordinates."""

    def build(self, coords: cp.ndarray) -> CSRPair:
        return super().build(_validate_coordinates(coords))

    def combine(self, mats: Sequence[CSRPair], ixs: Sequence[int]) -> CSRPair:
        adj, dst = (cp_sparse.block_diag(m, format="csr") for m in zip(*mats))
        ixs = cp.asarray(ixs)
        if bool((cp.diff(ixs) < 0).any()):
            order = cp.argsort(ixs)
            adj, dst = adj[order][:, order], dst[order][:, order]
        return adj, dst


class _SpatialBuilder(GraphBuilderCSR):
    """Postprocessors, graph assembly, and metadata of the built-in builders."""

    _coord_type = "generic"
    _parameters: tuple[str, ...] = ()

    def __init__(self, transform, percentile=None, interval=None, *, set_diag) -> None:
        super().__init__(transform, set_diag=set_diag, percentile=percentile)
        prune = [] if interval is None else [DistanceIntervalPostprocessor(interval)]
        prune += [] if percentile is None else [PercentilePostprocessor(percentile)]
        self._postprocessors = [*prune, TransformPostprocessor(transform)]

    def build_graph(self, coords: cp.ndarray, codes: cp.ndarray | None = None):
        """Build raw graphs, independently within each library of ``codes``."""
        rows, cols, values = self._edges(coords, codes)
        adj, dst = build_graphs(rows, cols, values, len(coords), set_diag=self.set_diag)
        del rows, cols, values  # Sort once, after freeing the edge lists.
        adj.sort_indices()
        dst.sort_indices()
        # Sharing the identical patterns lets postprocessors prune in place.
        dst.indices, dst.indptr = adj.indices, adj.indptr
        return adj, dst

    def _build(self, coords: cp.ndarray, codes: cp.ndarray | None) -> CSRPair:
        """Build validated coordinates of all libraries in one pass."""
        adj, dst = self.build_graph(coords, codes)
        for step in self.postprocessors():
            by_library = type(step) is PercentilePostprocessor
            adj, dst = step(adj, dst, codes) if by_library else step(adj, dst)
        return adj, dst

    def uns_params(self) -> dict[str, Any]:
        params = {name: getattr(self, name) for name in self._parameters}
        if isinstance(params.get("radius"), tuple):
            params["radius"] = list(params["radius"])
        if "n_neighs" in params:
            params["n_neighbors"] = params.pop("n_neighs")
        return {"coord_type": self._coord_type, **params, "transform": self.transform}


class KNNBuilder(_SpatialBuilder):
    """Connect each observation to its ``n_neighs`` nearest other observations."""

    _parameters = ("n_neighs",)

    def __init__(
        self,
        n_neighs: int = 6,
        *,
        transform: str | None = None,
        set_diag: bool = False,
        percentile: float | None = None,
    ) -> None:
        self.n_neighs = _count(n_neighs, "n_neighs")
        super().__init__(transform, percentile, set_diag=set_diag)

    def _edges(self, coords, codes):
        return _knn(coords, self.n_neighs, codes)


class RadiusBuilder(_SpatialBuilder):
    """Connect observations within a radius or an inclusive ``(min, max)`` interval."""

    _parameters = ("radius",)

    def __init__(
        self,
        radius: float | tuple[float, float],
        *,
        transform: str | None = None,
        set_diag: bool = False,
        percentile: float | None = None,
    ) -> None:
        self.radius = _radius(radius)
        interval = self.radius if isinstance(self.radius, tuple) else None
        super().__init__(transform, percentile, interval, set_diag=set_diag)

    def _edges(self, coords, codes):
        rows, cols, values = _radius_edges(coords, np.max(self.radius), codes)
        return rows, cols, _finite(values)


class DelaunayBuilder(_SpatialBuilder):
    """Connect Delaunay neighbors; a scalar ``radius`` prunes to ``(0, radius)``."""

    _parameters = ("radius",)

    def __init__(
        self,
        radius: float | tuple[float, float] | None = None,
        *,
        transform: str | None = None,
        set_diag: bool = False,
        percentile: float | None = None,
    ) -> None:
        self.radius = None if radius is None else _radius(radius)
        interval = (0.0, self.radius) if isinstance(self.radius, float) else self.radius
        super().__init__(transform, percentile, interval, set_diag=set_diag)

    def _edges(self, coords, codes):
        rows, cols = _delaunay_edges(coords, codes)
        return rows, cols, _finite(_edge_distances(coords, rows, cols))


class GridBuilder(_SpatialBuilder):
    """Build a lattice graph whose distances count shortest directed grid hops."""

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
        self.n_neighs = _count(n_neighs, "n_neighs")
        self.n_rings = _count(n_rings, "n_rings")
        self.delaunay = delaunay
        super().__init__(transform, set_diag=set_diag)

    def build_graph(self, coords: cp.ndarray, codes: cp.ndarray | None = None):
        if self.delaunay:
            rows, cols = _delaunay_edges(coords, codes)
        else:
            rows, cols, dist = _knn(coords, self.n_neighs, codes)
            median = cp.median(dist) if codes is None else _median(dist, codes[rows])
            keep = dist < median * 1.3
            rows, cols = rows[keep], cols[keep]
        rings = self.n_rings > 1
        base = build_adjacency(rows, cols, len(coords), set_diag=self.set_diag or rings)
        base.sort_indices()
        adj, dst, visited, frontier = base, base.copy(), base, base
        for ring in range(2, self.n_rings + 1):
            # The base diagonal keeps every frontier entry in the product.
            frontier = _sparse_product(frontier, base, min_nnz=frontier.nnz)
            frontier.data[:] = 1.0
            frontier = frontier - frontier.multiply(visited)
            frontier.eliminate_zeros()
            if not frontier.nnz:
                break
            if ring < self.n_rings:
                visited = visited + frontier
            dst = dst + frontier * float(ring)
        if rings:
            dst.setdiag(float(self.set_diag))
            dst.eliminate_zeros()
            adj = dst.copy()
            adj.data[:] = 1.0
        dst.setdiag(0.0)
        return adj, dst


@dataclass(frozen=True)
class DistanceIntervalPostprocessor:
    """Prune distances outside an inclusive interval, keeping the adjacency diagonal."""

    interval: tuple[float, float]

    def __post_init__(self) -> None:
        lower, upper = sorted(_number(value, "interval") for value in self.interval)
        object.__setattr__(self, "interval", (lower, upper))

    def __call__(self, adj: CSR, dst: CSR) -> CSRPair:
        lower, upper = self.interval
        # Compare in float64: a float32 bound 1 + 1e-8 would keep length 1.
        values = dst.data.astype(cp.float64, copy=False)
        return _prune(adj, dst, (values < lower) | (values > upper), keep_diagonal=True)


@dataclass(frozen=True)
class PercentilePostprocessor:
    """Prune distances above a percentile of all stored distances, including zeros.

    Given per-observation library ``codes``, each library uses its own percentile.
    """

    percentile: float

    def __post_init__(self) -> None:
        percentile = _number(self.percentile, "percentile", 100)
        object.__setattr__(self, "percentile", percentile)

    def __call__(self, adj: CSR, dst: CSR, codes: cp.ndarray | None = None) -> CSRPair:
        if not dst.nnz:
            return adj, dst
        if codes is None:
            threshold = cp.percentile(dst.data, self.percentile)
        else:
            library = codes[_entry_rows(dst)]
            values, starts, sizes = _library_sort(dst.data, library)
            index = self.percentile / 100 * (sizes - 1.0)
            threshold = cp.empty(len(sizes), dtype=values.dtype)
            stream = cp.cuda.get_current_stream().ptr
            _spatial_cuda.library_percentile(
                index, values, starts, sizes, threshold, stream
            )
            threshold = threshold[library]
        return _prune(adj, dst, dst.data > threshold)


@dataclass(frozen=True)
class TransformPostprocessor:
    """Remove stored zeros, then apply a spectral or cosine connectivity transform."""

    transform: str | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "transform", _transform(self.transform))

    def __call__(self, adj: CSR, dst: CSR) -> CSRPair:
        adj.eliminate_zeros()
        dst.eliminate_zeros()
        if self.transform == "spectral":
            degrees = cp.asarray(adj.sum(axis=0)).ravel()
            scale = cp.where(degrees > 0, 1 / cp.sqrt(degrees), 0)
            adj = adj.multiply(scale[:, None]).multiply(scale[None, :]).tocsr()
        elif self.transform == "cosine":
            norms = cp.asarray(adj.multiply(adj).sum(axis=1)).ravel()
            scale = cp.where(norms > 0, 1 / cp.sqrt(norms), 0)
            adj = adj.multiply(scale[:, None]).tocsr()
            # Each nonempty row keeps its unit self-similarity.
            adj = _sparse_product(adj, adj.T, min_nnz=int(cp.count_nonzero(scale)))
        if self.transform is not None:
            adj.eliminate_zeros()
        adj.sort_indices()
        dst.sort_indices()
        return adj, dst


def _prune(adj: CSR, dst: CSR, outside: cp.ndarray, *, keep_diagonal=False) -> CSRPair:
    # Zero pruned distances and adjacency edges. Built-in graphs share their
    # pattern arrays and keep stored zeros for TransformPostprocessor to remove.
    edges = outside & (dst.indices != _entry_rows(dst)) if keep_diagonal else outside
    shared = adj.indptr is dst.indptr and adj.indices is dst.indices
    if shared and not cp.may_share_memory(adj.data, dst.data):
        adj.data[edges] = 0
    else:
        mask = (edges.astype(adj.dtype), dst.indices, dst.indptr)
        adj = (adj - adj.multiply(cp_sparse.csr_matrix(mask, shape=dst.shape))).tocsr()
    dst.data[outside] = 0
    return adj, dst


def _entry_rows(matrix: CSR) -> cp.ndarray:
    return cp.searchsorted(matrix.indptr, cp.arange(matrix.nnz), side="right") - 1


def _sparse_product(left: CSR, right: Any, *, min_nnz: int) -> CSR:
    """Multiply sparse matrices around cuSPARSE SpGEMM limitations.

    Near 2**31 intermediate products, SpGEMM fails or CuPy silently returns an
    empty product, so row blocks stay within 2**28 products.
    """
    right = right.tocsr()
    # Intermediate products before each row of ``left``.
    work = cp.cumsum(cp.diff(right.indptr)[left.indices], dtype=cp.int64)
    work = cp.concatenate((cp.zeros(1, cp.int64), work))[left.indptr]
    splits = cp.searchsorted(work, cp.arange(2**28, int(work[-1]), 2**28)).get()
    bounds = np.unique(np.r_[0, splits, left.shape[0]])
    if len(bounds) == 2:
        product = _spgemm(left, right)
    else:
        blocks = [_spgemm(left[a:b], right) for a, b in zip(bounds, bounds[1:])]
        product = cp_sparse.vstack(blocks, "csr")
    if product.nnz < min_nnz:
        raise RuntimeError("cuSPARSE returned an incomplete sparse matrix product.")
    return product


def _spgemm(left: CSR, right: CSR) -> CSR:
    try:
        return (left @ right).tocsr()
    except cp.cuda.cusparse.CuSparseError:
        # int32 SpGEMM fails once nonempty rows span 2**30 cells.
        left, right = (cp_sparse.csr_matrix(m) for m in (left, right))
        for m in (left, right):
            m.indices, m.indptr = m.indices.astype(cp.int64), m.indptr.astype(cp.int64)
        product = left @ right
        # The constructor restores int32 indices when they fit.
        return cp_sparse.csr_matrix(
            (product.data, product.indices, product.indptr), shape=product.shape
        )


def _library_sort(values: cp.ndarray, library: cp.ndarray) -> tuple[cp.ndarray, ...]:
    """Sort nonnegative float32 values (bits sort alike) within libraries."""
    keys = cp.sort((library.astype(cp.uint64) << 32) | values.view(cp.uint32))
    sizes = cp.bincount(library)
    values = (keys & 0xFFFFFFFF).astype(cp.uint32).view(cp.float32)
    return values, cp.cumsum(sizes) - sizes, sizes


def _median(values: cp.ndarray, library: cp.ndarray) -> cp.ndarray:
    """``cp.median`` of each value's library."""
    values, starts, sizes = _library_sort(values, library)
    low = values[cp.clip(starts + (sizes - 1) // 2, 0, len(values) - 1)]
    high = values[cp.clip(starts + sizes // 2, 0, len(values) - 1)]
    return cp.where(sizes % 2 == 1, low, (low + high) / 2)[library]


def _knn(coords: cp.ndarray, n_neighs: int, codes: cp.ndarray | None):
    smallest = len(coords) if codes is None else int(cp.bincount(codes)[codes].min())
    if smallest <= n_neighs:
        raise ValueError("Every library needs more than `n_neighs` observations.")
    rows, cols, values = _knn_edges(coords, n_neighs, codes)
    return rows, cols, _finite(values)


def _validate_coordinates(coords: Any) -> cp.ndarray:
    coords = coords.to_numpy() if hasattr(coords, "to_numpy") else coords
    coords = cp.asarray(coords, dtype=cp.float32, order="C")
    if coords.ndim != 2 or 0 in coords.shape:
        raise ValueError("Spatial coordinates must be a nonempty 2D array.")
    if not bool(cp.isfinite(coords).all()):
        raise ValueError("Spatial coordinates must be finite in float32. Rescale them.")
    return coords


def _finite(distances: cp.ndarray) -> cp.ndarray:
    if not bool(cp.isfinite(distances).all()):
        raise ValueError("Spatial distances overflow float32. Rescale the coordinates.")
    return distances


def _count(value: Any, name: str) -> int:
    if not isinstance(value, Integral) or value < 1:
        raise ValueError(f"`{name}` must be a positive integer.")
    return int(value)


def _number(value: Any, name: str, upper: float = np.inf) -> float:
    if not isinstance(value, Real) or not 0 <= value < np.inf or value > upper:
        raise ValueError(f"`{name}` must be a nonnegative finite number <= {upper}.")
    return float(value)


def _radius(radius: Any) -> float | tuple[float, float]:
    if not isinstance(radius, tuple):
        return _number(radius, "radius")
    lower, upper = radius
    return _number(lower, "radius"), _number(upper, "radius")


def _transform(transform: Any) -> str | None:
    transform = getattr(transform, "value", transform)  # squidpy's Transform enum
    if transform not in (None, "spectral", "cosine"):
        raise ValueError("`transform` must be None, 'spectral', or 'cosine'.")
    return transform
