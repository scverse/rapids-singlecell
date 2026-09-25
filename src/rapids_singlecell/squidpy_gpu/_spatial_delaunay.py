# CuPy Delaunay bug fixes; adapted portions use the MIT license (see LICENSE).
# Copyright (c) 2015 Preferred Infrastructure, Inc.
# Copyright (c) 2015 Preferred Networks, Inc.

from __future__ import annotations

import math
import os
import warnings
from concurrent.futures import ThreadPoolExecutor

import cupy as cp
import numpy as np
from scipy.spatial import Delaunay

try:  # Private CuPy internals: if they move, every input uses Qhull.
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
except ImportError:
    GDel2D = object

from rapids_singlecell._cuda import _spatial_cuda

_GPU_MIN_POINTS = 32_768


def _orientations(points: cp.ndarray, triangles: cp.ndarray) -> cp.ndarray:
    out = cp.empty(len(triangles), dtype=cp.float64)
    _spatial_cuda.delaunay_orientations(points, triangles, out, _stream())
    return out


def _stream() -> int:
    return cp.cuda.get_current_stream().ptr


class _FullDelaunay(GDel2D):
    """CuPy's GDel2D without point loss, stalls, or counter overruns."""

    def _init_for_flip(self):
        # CuPy drops distinct points that share a Morton key; keep them all.
        low = self.points.min()
        get_morton_number(
            self.points, self.n_points - 1, low, self.points.max() - low, self.values
        )
        self.values[-1] = 2**31 - 1
        self.points_idx = cp.argsort(self.values).astype(cp.int32)
        self.point_vec = self.point_vec[self.points_idx]
        self._construct_initial_triangles()

    def _construct_initial_triangles(self):
        # The widest triangle on the extreme-x points, by exact orientation.
        n = self.n_points - 1
        x = cp.ascontiguousarray(self.point_vec[:-1, 0])
        v0, v1 = int(x.argmin()), int(x.argmax())
        tri = cp.empty((n, 3), dtype=cp.int32)
        tri[:, 0], tri[:, 1], tri[:, 2] = v0, v1, cp.arange(n)
        area = _orientations(self.point_vec[:-1], tri)
        v2 = int(cp.abs(area).argmax())
        sign = float(area[v2])
        if sign == 0:
            raise ValueError("Delaunay triangulation requires non-collinear points.")
        tri = cp.asarray([v0, v1, v2] if sign > 0 else [v1, v0, v2], cp.int32)
        self.point_vec[-1] = self.point_vec[tri].mean(axis=0)
        self.pred_consts = cp.empty(18, dtype=cp.float64)
        init_predicate(self.pred_consts)
        make_first_tri(self.triangles, self.triangle_opp, self.triangle_info, tri, n)
        self._renew_counters()
        exact_check = cp.empty(self.n_points, dtype=cp.int32)
        for locate in (init_point_location_fast, init_point_location_exact):
            locate(
                self.vert_tri,
                self.n_points,
                exact_check,
                self.counters,
                tri,
                n,
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
            raise ValueError("GPU point insertion made no progress.")

    def _renew_counters(self):
        self.counters_offset += self.counters_size
        if self.counters_offset + self.counters_size > len(self._counters):
            self.counters_offset = 0
            self._counters[:] = 0


def _validated_edges(
    points: cp.ndarray, triangles: cp.ndarray, opposite: cp.ndarray
) -> tuple[cp.ndarray, cp.ndarray] | None:
    """Row-grouped edges if the mesh is a Delaunay triangulation, else ``None``.

    Exact checks certify positively oriented triangles, consistent adjacency and
    a convex hull boundary traversed once; illegal edges are flipped.
    """
    n = len(points)
    area = _orientations(points, triangles)
    if not bool(area.all()):
        # Symbolic perturbation can add zero-area fans along straight hull edges.
        keep = area != 0
        index = cp.where(keep, cp.cumsum(keep, dtype=cp.int32) - 1, -1)
        triangles, opposite = triangles[keep], opposite[keep]
        other = cp.where(opposite >= 0, index[opposite >> 4], -1)
        opposite = cp.where(other >= 0, (other << 4) | (opposite & 15), -1)
    nt = len(triangles)
    while True:
        state = cp.zeros(3, dtype=cp.int32)
        seen = cp.zeros(n, dtype=cp.int32)
        after, before = cp.full((2, n), -1, dtype=cp.int32)
        votes = cp.full(nt, 2**31 - 1, dtype=cp.int32)
        args = state, seen, after, before, votes, _stream()
        _spatial_cuda.delaunay_validate(points, triangles, opposite, *args)
        failure, boundary, lowest = state.tolist()
        if boundary < 3 or nt != 2 * n - boundary - 2 or lowest != 1:
            failure |= 1
        if not failure:
            break
        if failure != 2 or not _flip_delaunay_edges(triangles, opposite, votes):
            return None
    # Euler's formula gives the edge count; the degree prefix sum groups rows.
    rows = cp.empty(2 * (n + nt - 1), dtype=cp.int32)
    cols = cp.empty_like(rows)
    cursor = cp.cumsum(seen, dtype=cp.int32) - seen
    _spatial_cuda.delaunay_fill_edges(
        triangles, opposite, cursor, rows, cols, _stream()
    )
    return rows, cols


def _flip_delaunay_edges(
    triangles: cp.ndarray, opposite: cp.ndarray, votes: cp.ndarray
) -> bool:
    """Flip a nonconflicting batch of illegal edges with CuPy's kernels."""
    nt = len(triangles)
    active = cp.arange(nt, dtype=cp.int32)
    info = cp.zeros(nt, dtype=cp.int8)
    flips = cp.empty(nt, dtype=cp.int32)
    mark_rejected_flips(active, opposite, votes, info, flips, nt)
    flips = flips[flips >= 0]
    if not flips.size:
        return False
    # Each batch starts a fresh flip history; unflipped triangles have no message.
    messages = cp.full((nt, 2), -1, dtype=cp.int32)
    history = cp.empty((len(flips), 4), dtype=cp.int32)
    flip(flips, triangles, opposite, info, messages, active, history, 0, nt, 0)
    update_opp(history, opposite, messages, flips, 0, len(flips))
    return True


def _gpu_delaunay_edges(coords: cp.ndarray) -> tuple[cp.ndarray, cp.ndarray] | None:
    if GDel2D is object or coords.shape[1] != 2 or len(coords) < _GPU_MIN_POINTS:
        return None
    magnitude = float(cp.abs(coords).max())
    if not math.isfinite(magnitude):
        raise ValueError("Delaunay coordinates must be finite.")
    # Power-of-two scaling is exact; centering could merge distant points.
    shift = -math.frexp(magnitude)[1]
    if coords.dtype == cp.float32:
        points = cp.empty(coords.shape, dtype=cp.float64)
        _spatial_cuda.delaunay_widen(
            cp.ascontiguousarray(coords), shift, points, _stream()
        )
    else:
        points = cp.ldexp(coords.astype(cp.float64), shift)
    # Duplicates are represented by their first observation.
    order = cp.lexsort(points.T)
    ordered = points[order]
    distinct = cp.ones(len(points), dtype=cp.bool_)
    distinct[1:] = (ordered[1:] != ordered[:-1]).any(axis=1)
    representatives = None if bool(distinct.all()) else cp.sort(order[distinct])
    if representatives is not None:
        points = points[representatives]
    if len(points) < 3:
        raise ValueError(
            "Delaunay triangulation requires at least three distinct points."
        )
    # Triangle references reserve four low bits for edge metadata.
    if len(points) >= 2**26:
        return None
    try:
        edges = _validated_edges(points, *_FullDelaunay(points).compute())
    except (ValueError, AttributeError, TypeError):
        # Degenerate geometry, stalled insertion or changed CuPy internals.
        return None
    if edges is None or representatives is None:
        return edges
    return representatives[edges[0]], representatives[edges[1]]


def _qhull_edges(points: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    indptr, indices = Delaunay(points).vertex_neighbor_vertices
    return np.repeat(np.arange(len(points)), np.diff(indptr)), indices


def _delaunay_edges(
    coords: cp.ndarray, codes: cp.ndarray | None = None
) -> tuple[cp.ndarray, cp.ndarray]:
    """GPU Delaunay for suitable 2D inputs; Qhull for all other cases.

    With ``codes``, each library is triangulated independently, and edges use
    observation indices grouped by row.
    """
    if coords.shape[1] != 2:
        warnings.warn(
            f"GPU Delaunay triangulation is not yet implemented for {coords.shape[1]}D "
            "coordinates; using CPU triangulation. To request GPU support, open a "
            "feature request at https://github.com/scverse/rapids_singlecell/issues.",
            UserWarning,
            stacklevel=2,
        )
    if codes is None:
        edges = _gpu_delaunay_edges(coords) or _qhull_edges(cp.asnumpy(coords))
        return tuple(cp.asarray(edge) for edge in edges)
    host, codes = cp.asnumpy(coords), cp.asnumpy(codes)
    order = np.argsort(codes, kind="stable")
    groups = np.split(order, np.flatnonzero(np.diff(codes[order])) + 1)
    gpu = coords.shape[1] == 2 and GDel2D is not object
    small = [not gpu or len(group) < _GPU_MIN_POINTS for group in groups]
    threaded = sum(small) > 1  # A lone Qhull run is faster on this thread.
    rows, cols = [], []
    with ThreadPoolExecutor(min(8, len(os.sched_getaffinity(0)))) as pool:
        # Qhull releases the GIL, so small libraries run in threads while large
        # ones use the GPU. Results in library order keep the loop's errors.
        jobs = [
            pool.submit(_qhull_edges, host[group]) if threaded and is_small else None
            for group, is_small in zip(groups, small, strict=True)
        ]
        for group, job in zip(groups, jobs, strict=True):
            edges = None if job else _gpu_delaunay_edges(coords[cp.asarray(group)])
            if edges is None:
                edges = job.result() if job else _qhull_edges(host[group])
            else:
                group = cp.asarray(group)
            rows.append(cp.asarray(group[edges[0]]))
            cols.append(cp.asarray(group[edges[1]]))
    rows, cols = cp.concatenate(rows), cp.concatenate(cols)
    order = cp.argsort(rows)
    return rows[order], cols[order]
