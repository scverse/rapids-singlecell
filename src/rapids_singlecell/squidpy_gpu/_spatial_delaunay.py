# The GDel2D initialization and counter adapter below is adapted from CuPy,
# distributed under the following MIT license:
#
# Copyright (c) 2015 Preferred Infrastructure, Inc.
# Copyright (c) 2015 Preferred Networks, Inc.
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
# THE SOFTWARE.

from __future__ import annotations

import math

import cupy as cp
import numpy as np
from cupyx.scipy.spatial.delaunay_2d._tri import GDel2D, get_morton_number
from scipy.spatial import Delaunay

from rapids_singlecell._cuda import _spatial_cuda

_GPU_MIN_POINTS = 32_768


class _UnsafeTriangulation(ValueError):
    """The GPU candidate cannot safely reproduce the Qhull graph."""


class _FullDelaunay(GDel2D):
    """Preserve distinct coordinates sharing CuPy's Morton ordering key."""

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
        self._flip_steps = 0
        self._construct_initial_triangles()

    def _split_tri(self):
        remaining = self.available_points
        super()._split_tri()
        if self.available_points >= remaining:
            raise _UnsafeTriangulation("GPU point insertion made no progress.")

    def _do_flipping(self, check_mode):
        self._flip_steps += 1
        if self._flip_steps > 4096:
            raise _UnsafeTriangulation("GPU edge flipping did not converge.")
        return super()._do_flipping(check_mode)

    def _renew_counters(self):
        self.counters_offset += self.counters_size
        if self.counters_offset + self.counters_size > len(self._counters):
            self.counters_offset = 0
            self._counters[:] = 0


def _validated_edges(
    points: cp.ndarray,
    triangles: cp.ndarray,
    opposite: cp.ndarray,
    *,
    tolerance: float,
) -> tuple[cp.ndarray, cp.ndarray] | None:
    """Accept only a complete, strictly Delaunay planar triangulation."""
    n, nt = len(points), len(triangles)
    if not n - 2 <= nt <= 2 * n - 5:
        return None
    invalid = cp.zeros(1, dtype=cp.int32)
    seen = cp.zeros(n, dtype=cp.int32)
    next_boundary = cp.full(n, -1, dtype=cp.int32)
    prev_boundary = cp.full(n, -1, dtype=cp.int32)
    boundary_count = cp.zeros(1, dtype=cp.int32)
    _spatial_cuda.validate_delaunay(
        points,
        triangles,
        opposite,
        tolerance,
        invalid,
        seen,
        next_boundary,
        prev_boundary,
        boundary_count,
        cp.cuda.get_current_stream().ptr,
    )
    if int(invalid[0]):
        return None
    # Euler's formula gives the exact directed edge count once validation has
    # established a triangulated disk. The prefix sum groups output by row.
    rows = cp.empty(2 * (n + nt - 1), dtype=cp.int32)
    cols = cp.empty_like(rows)
    cursor = cp.cumsum(seen, dtype=cp.int32) - seen
    _spatial_cuda.fill_delaunay_edges(
        triangles, opposite, cursor, rows, cols, cp.cuda.get_current_stream().ptr
    )
    return rows, cols


def _gpu_delaunay_edges(coords: cp.ndarray) -> tuple[cp.ndarray, cp.ndarray] | None:
    if coords.shape[1] != 2 or len(coords) < _GPU_MIN_POINTS:
        return None
    # Internal triangle references reserve four low bits for edge metadata.
    if len(coords) >= 2**26:
        return None
    bounds = cp.asnumpy(cp.stack((coords.min(axis=0), coords.max(axis=0)))).astype(
        np.float64, copy=False
    )
    span = float(np.max(bounds[1] - bounds[0]))
    magnitude = float(np.max(np.abs(bounds)))
    if not math.isfinite(span) or span <= 0 or magnitude / span > 1e6:
        return None
    exponent = math.frexp(span)[1]
    # Normalization must not turn a numerically extreme input that Qhull may
    # reject into an apparently ordinary GPU graph. Retain the CPU semantics
    # outside this conservative range of original coordinate scales.
    if not -128 <= exponent <= 128:
        return None
    # CuPy's triangulation ABI and our conservative predicates require double,
    # even when the public input geometry and output distances are float32.
    points = cp.ldexp(coords.astype(cp.float64, copy=False), -exponent)
    center = np.ldexp(bounds[0] / 2 + bounds[1] / 2, -exponent)
    points -= cp.asarray(center)
    points = cp.ascontiguousarray(points)
    order = cp.lexsort(cp.stack((points[:, 1], points[:, 0])))
    ordered = points[order]
    if bool(cp.any(cp.all(ordered[1:] == ordered[:-1], axis=1))):
        return None
    # The tolerance also accounts for Qhull's sensitivity to translation.
    tolerance = 4 * np.finfo(np.float64).eps * max(1, magnitude / span)
    try:
        triangles, opposite = _FullDelaunay(points).compute()
    except ValueError:  # Degenerate geometry or failed convergence: defer to Qhull.
        return None
    return _validated_edges(points, triangles, opposite, tolerance=tolerance)


def _delaunay_edges(coords: cp.ndarray) -> tuple[cp.ndarray, cp.ndarray]:
    """GPU Delaunay for suitable 2D inputs; Qhull for all other cases."""
    edges = _gpu_delaunay_edges(coords)
    if edges is not None:
        return edges
    indptr, indices = Delaunay(cp.asnumpy(coords)).vertex_neighbor_vertices
    rows = cp.repeat(
        cp.arange(len(coords), dtype=cp.int64), cp.asarray(np.diff(indptr))
    )
    return rows, cp.asarray(indices)
