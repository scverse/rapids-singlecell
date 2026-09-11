# CuPy Delaunay bug fixes; adapted portions use the MIT license (see LICENSE).
# Copyright (c) 2015 Preferred Infrastructure, Inc.
# Copyright (c) 2015 Preferred Networks, Inc.

from __future__ import annotations

import math
import warnings

import cupy as cp
import numpy as np
from cupyx.scipy.spatial.delaunay_2d._kernels import (
    flip,
    init_point_location_exact,
    init_point_location_fast,
    init_predicate,
    make_first_tri,
    mark_rejected_flips,
    update_opp,
)
from cupyx.scipy.spatial.delaunay_2d._tri import GDel2D, get_morton_number
from scipy.spatial import Delaunay

from rapids_singlecell._cuda import _spatial_cuda

_GPU_MIN_POINTS = 32_768
_INCIRCLE_FAILURE = 2


class _UnsafeTriangulation(ValueError):
    """GPU triangulation did not complete."""


class _FullDelaunay(GDel2D):
    """Fix CuPy's point-loss bug for distinct coordinates sharing a Morton key."""

    def _init_for_flip(self):
        self.min_val = self.points.min()
        self.max_val = self.points.max()
        self.range_val = self.max_val - self.min_val
        get_morton_number(
            self.points,
            self.n_points - 1,
            self.min_val,
            self.range_val,
            self.values,
        )
        self.values[-1] = 2**31 - 1
        self.points_idx = cp.argsort(self.values).astype(cp.int32)
        self.point_vec = self.point_vec[self.points_idx]
        self._construct_initial_triangles()

    def _construct_initial_triangles(self):
        v0 = int(cp.argmin(self.point_vec[:-1, 0]))
        v1 = int(cp.argmax(self.point_vec[:-1, 0]))
        orientations = cp.empty(self.n_points - 1, dtype=cp.float64)
        _spatial_cuda.delaunay_orientations(
            self.point_vec[:-1],
            v0,
            v1,
            output=orientations,
            stream=cp.cuda.get_current_stream().ptr,
        )
        v2 = int(cp.argmax(cp.abs(orientations)))
        orientation = float(orientations[v2])
        if orientation == 0:
            raise ValueError("Delaunay triangulation requires non-collinear points.")
        tri = cp.asarray(
            [v0, v1, v2] if orientation > 0 else [v1, v0, v2], dtype=cp.int32
        )
        self.point_vec[-1] = self.point_vec[tri].mean(axis=0)
        self.pred_consts = cp.empty(18, dtype=cp.float64)
        init_predicate(self.pred_consts)
        make_first_tri(
            self.triangles,
            self.triangle_opp,
            self.triangle_info,
            tri,
            self.n_points - 1,
        )
        self._renew_counters()
        exact_check = cp.empty(self.n_points, dtype=cp.int32)
        for locate in (init_point_location_fast, init_point_location_exact):
            locate(
                self.vert_tri,
                self.n_points,
                exact_check,
                self.counters,
                tri,
                self.n_points - 1,
                self.point_vec,
                self.points_idx,
                self.pred_consts,
            )
        self.available_points = self.n_points - 4
        self.tri_num = 4

    def _split_tri(self):
        remaining = self.available_points
        super()._split_tri()
        if self.available_points >= remaining:
            raise _UnsafeTriangulation("GPU point insertion made no progress.")

    def _renew_counters(self):
        self.counters_offset += self.counters_size
        if self.counters_offset + self.counters_size > len(self._counters):
            self.counters_offset = 0
            self._counters[:] = 0


def _validated_edges(
    points: cp.ndarray,
    triangles: cp.ndarray,
    opposite: cp.ndarray,
) -> tuple[cp.ndarray, cp.ndarray] | None:
    """Accept only a complete Delaunay planar triangulation."""
    n, nt = len(points), len(triangles)
    if not n - 2 <= nt <= 2 * n - 5:
        return None
    orientations = cp.empty(nt, dtype=cp.float64)
    _spatial_cuda.delaunay_triangle_orientations(
        points,
        triangles,
        output=orientations,
        stream=cp.cuda.get_current_stream().ptr,
    )
    if bool(cp.any(orientations == 0)):
        # Symbolic perturbation can add zero-area fans along straight hull edges.
        keep = orientations != 0
        mapping = cp.full(nt, -1, dtype=cp.int32)
        mapping[keep] = cp.arange(int(cp.count_nonzero(keep)), dtype=cp.int32)
        triangles, opposite = triangles[keep], opposite[keep]
        neighbors = opposite >> 4
        interior = opposite >= 0
        if bool(cp.any(neighbors[interior] >= nt)):
            return None
        neighbors[interior] = mapping[neighbors[interior]]
        opposite = cp.where(neighbors >= 0, (neighbors << 4) | (opposite & 15), -1)
        nt = len(triangles)
        if not n - 2 <= nt <= 2 * n - 5:
            return None
    del orientations
    invalid = cp.zeros(1, dtype=cp.int32)
    seen = cp.zeros(n, dtype=cp.int32)
    next_boundary = cp.full(n, -1, dtype=cp.int32)
    prev_boundary = cp.full(n, -1, dtype=cp.int32)
    boundary_count = cp.zeros(2, dtype=cp.int32)
    while True:
        _spatial_cuda.validate_delaunay(
            points,
            triangles,
            opposite,
            invalid,
            seen,
            next_boundary,
            prev_boundary,
            boundary_count,
            cp.cuda.get_current_stream().ptr,
        )
        failure = int(invalid[0])
        if not failure:
            break
        if failure != _INCIRCLE_FAILURE or not _flip_delaunay_edges(
            points, triangles, opposite
        ):
            return None
        invalid.fill(0)
        seen.fill(0)
        next_boundary.fill(-1)
        prev_boundary.fill(-1)
        boundary_count.fill(0)
    # Euler's formula gives the exact directed edge count once validation has
    # established a triangulated disk. The prefix sum groups output by row.
    rows = cp.empty(2 * (n + nt - 1), dtype=cp.int32)
    cols = cp.empty_like(rows)
    cursor = cp.cumsum(seen, dtype=cp.int32) - seen
    _spatial_cuda.fill_delaunay_edges(
        triangles, opposite, cursor, rows, cols, cp.cuda.get_current_stream().ptr
    )
    return rows, cols


def _flip_delaunay_edges(
    points: cp.ndarray, triangles: cp.ndarray, opposite: cp.ndarray
) -> bool:
    """Flip a nonconflicting batch of strictly non-Delaunay interior edges."""
    nt = len(triangles)
    votes = cp.full(nt, np.iinfo(np.int32).max, dtype=cp.int32)
    _spatial_cuda.exact_flip_votes(
        points,
        triangles,
        opposite,
        votes=votes,
        stream=cp.cuda.get_current_stream().ptr,
    )
    active = cp.arange(nt, dtype=cp.int32)
    triangle_info = cp.zeros(nt, dtype=cp.int8)
    flip_to_tri = cp.empty(nt, dtype=cp.int32)
    mark_rejected_flips(active, opposite, votes, triangle_info, flip_to_tri, nt)
    flip_to_tri = flip_to_tri[flip_to_tri >= 0]
    if not flip_to_tri.size:
        return False
    # Each batch starts a fresh flip history; unflipped triangles have no message.
    messages = cp.full((nt, 2), -1, dtype=cp.int32)
    history = cp.empty((len(flip_to_tri), 4), dtype=cp.int32)
    flip(
        flip_to_tri,
        triangles,
        opposite,
        triangle_info,
        messages,
        active,
        history,
        0,
        nt,
        0,
    )
    update_opp(history, opposite, messages, flip_to_tri, 0, len(flip_to_tri))
    return True


def _gpu_delaunay_edges(coords: cp.ndarray) -> tuple[cp.ndarray, cp.ndarray] | None:
    if coords.shape[1] != 2 or len(coords) < _GPU_MIN_POINTS:
        return None
    bounds = cp.asnumpy(cp.stack((coords.min(axis=0), coords.max(axis=0)))).astype(
        np.float64, copy=False
    )
    magnitude = float(np.max(np.abs(bounds)))
    if not math.isfinite(magnitude):
        raise ValueError("Delaunay coordinates must be finite.")
    exponent = math.frexp(magnitude)[1]
    # Power-of-two scaling preserves float32 coordinates exactly in double.
    # Avoid centering, which can merge points with very different magnitudes.
    if coords.dtype == cp.float32:
        points = cp.empty(coords.shape, dtype=cp.float64)
        _spatial_cuda.delaunay_normalize(
            cp.ascontiguousarray(coords),
            exponent=exponent,
            output=points,
            stream=cp.cuda.get_current_stream().ptr,
        )
    else:
        points = cp.ldexp(coords.astype(cp.float64, copy=False), -exponent)
    points = cp.ascontiguousarray(points)
    order = cp.lexsort(cp.stack((points[:, 1], points[:, 0])))
    ordered = points[order]
    distinct = cp.empty(len(points), dtype=cp.bool_)
    distinct[0] = True
    distinct[1:] = cp.any(ordered[1:] != ordered[:-1], axis=1)
    representatives = None
    if not bool(cp.all(distinct)):
        representatives = cp.sort(order[distinct])
        points = points[representatives]
    if len(points) < 3:
        raise ValueError(
            "Delaunay triangulation requires at least three distinct points."
        )
    # Internal triangle references reserve four low bits for edge metadata.
    if len(points) >= 2**26:
        return None
    try:
        triangles, opposite = _FullDelaunay(points).compute()
    except ValueError:  # Degenerate geometry or failed convergence: defer to Qhull.
        return None
    edges = _validated_edges(points, triangles, opposite)
    if edges is not None and representatives is not None:
        rows, cols = edges
        return representatives[rows], representatives[cols]
    return edges


def _delaunay_edges(coords: cp.ndarray) -> tuple[cp.ndarray, cp.ndarray]:
    """GPU Delaunay for suitable 2D inputs; Qhull for all other cases."""
    if coords.shape[1] != 2:
        warnings.warn(
            f"GPU Delaunay triangulation is not yet implemented for {coords.shape[1]}D "
            "coordinates; using CPU triangulation. To request GPU support, open a "
            "feature request at https://github.com/scverse/rapids_singlecell/issues.",
            UserWarning,
            stacklevel=2,
        )
    edges = _gpu_delaunay_edges(coords)
    if edges is not None:
        return edges
    indptr, indices = Delaunay(cp.asnumpy(coords)).vertex_neighbor_vertices
    rows = cp.repeat(
        cp.arange(len(coords), dtype=cp.int64), cp.asarray(np.diff(indptr))
    )
    return rows, cp.asarray(indices)
