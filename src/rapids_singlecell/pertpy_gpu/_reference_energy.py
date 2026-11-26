from __future__ import annotations

from math import isqrt

import cupy as cp
import numpy as np
from cuml.metrics import pairwise_distances

_PAIRWISE_SCRATCH = 128 << 20

# A warp sums one selected row up to its diagonal. The three blocks represent
# the symmetric pooled distance matrix without copying the reference square.
_selected_rows = cp.RawKernel(  # type: ignore[attr-defined]
    r"""
    extern "C" __global__ void selected_rows(
        const int *picked, int k, int m, int t, const double *rr,
        const double *tr, const double *tt, double *out) {
        int lane = threadIdx.x % 32;
        int i = blockIdx.x * (blockDim.x / 32) + threadIdx.x / 32;
        if (i >= k) return;
        long long base = (long long)blockIdx.y * k;
        int x = picked[base + i];
        double sum = 0;
        for (int j = lane; j < i; j += 32) {
            int y = picked[base + j];
            if (x < m && y < m) sum += rr[(long long)x * m + y];
            else if (x >= m && y >= m) sum += tt[(long long)(x - m) * t + y - m];
            else sum += tr[(long long)(max(x, y) - m) * m + min(x, y)];
        }
        for (int d = 16; d; d /= 2) sum += __shfl_down_sync(0xffffffffu, sum, d);
        if (lane == 0) out[base + i] = 2 * sum;
    }
    """,
    "selected_rows",
)


# Oversize comparisons retain only marginals. A warp evaluates distances from
# one selected cell to its preceding selected cells directly from the features.
_selected_feature_rows = cp.RawKernel(  # type: ignore[attr-defined]
    r"""
    extern "C" __global__ void selected_feature_rows(
        const int *picked, int k, int m, int d, const double *ref,
        const double *target, const double *offset, double *out) {
        int lane = threadIdx.x % 32;
        int i = blockIdx.x * (blockDim.x / 32) + threadIdx.x / 32;
        if (i >= k) return;
        long long base = (long long)blockIdx.y * k;
        int x = picked[base + i];
        const double *a = x < m ? ref + (long long)x*d : target + (long long)(x-m)*d;
        double sum = 0;
        for (int j = 0; j < i; ++j) {
            int y = picked[base + j];
            const double *b = y < m ? ref + (long long)y*d : target + (long long)(y-m)*d;
            double squared = 0;
            for (int f = lane; f < d; f += 32) {
                double delta = a[f] - b[f];
                squared += delta * delta;
            }
            for (int step = 16; step; step /= 2)
                squared += __shfl_down_sync(0xffffffffu, squared, step);
            if (lane == 0) sum += sqrt(squared) - *offset;
        }
        if (lane == 0) out[base + i] = 2 * sum;
    }
    """,
    "selected_feature_rows",
)


def _distances(x, y):
    # Limit additional cuML output storage even for a reference of tens of thousands.
    rows = max(1, _PAIRWISE_SCRATCH // (8 * len(y)))
    if rows >= len(x):
        return cp.asarray(pairwise_distances(x, y, metric="euclidean"), order="C")
    out = cp.empty((len(x), len(y)), dtype=cp.float64)
    for start in range(0, len(x), rows):
        out[start : start + rows] = pairwise_distances(x[start : start + rows], y, metric="euclidean")
    return out


def offdiagonal_energy(cross, within, other, small, large, *, offset):
    """Combine centered pair sums, omitting self-pairs from both within-group means."""
    a, b = max(small - 1, 1), max(large - 1, 1)
    numerator = 2 * a * b * cross - large * b * within - small * a * other
    return numerator / (float(small) * large * a * b) + offset * ((small == 1) + (large == 1))


def _tile_marginals(x, y, offset, budget):
    """Sum a rectangular distance matrix by row and column, visiting self-pairs once."""
    symmetric = x is y
    rows = cp.zeros(len(x), dtype=cp.float64)
    cols = rows if symmetric else cp.zeros(len(y), dtype=cp.float64)
    side = min(4096, max(1, isqrt(budget // 8)))
    for i in range(0, len(x), side):
        a = slice(i, i + side)
        for j in range(0, i + 1 if symmetric else len(y), side):
            b = slice(j, j + side)
            tile = _distances(x[a], y[b])
            tile -= offset
            if symmetric and i == j:
                cp.fill_diagonal(tile, 0)
            rows[a] += tile.sum(axis=1)
            if not symmetric or i != j:
                cols[b] += tile.sum(axis=0)
            del tile
    return rows, cols


class ReferenceEnergy:
    """Cache reference distances or their marginals for off-diagonal Euclidean E-tests."""

    def __init__(self, reference, *, max_distance_memory=None):
        self.reference = cp.asarray(reference, dtype=cp.float64, order="C")
        self.max_distance_memory = max_distance_memory
        m = len(self.reference)
        self.rr = None
        rr_bytes = 8 * m**2
        if max_distance_memory is None or rr_bytes + min(rr_bytes, _PAIRWISE_SCRATCH) <= max_distance_memory:
            try:
                self.rr = _distances(self.reference, self.reference)
            except MemoryError:
                # The reference can outgrow available VRAM after choosing the budget.
                self.max_distance_memory = max(8, min(_PAIRWISE_SCRATCH, rr_bytes // 4))
        if self.rr is not None:
            self.offset = self.rr[0, 1].copy() if m > 1 else cp.asarray(0.0)
            self.rr -= self.offset
            cp.fill_diagonal(self.rr, 0)
            self.marginal = self.rr.sum(axis=1)
        else:
            self.offset = _distances(self.reference[:1], self.reference[1:2])[0, 0] if m > 1 else cp.asarray(0.0)
            self.marginal, _ = _tile_marginals(self.reference, self.reference, self.offset, self.max_distance_memory)
        self.total = self.marginal.sum()

    def scores(self, target, membership, batches, n_perms, *, statistic="contrast"):
        m, t = len(self.reference), len(target)
        k, large = sorted((m, t))
        target = cp.asarray(target, dtype=cp.float64, order="C")
        budget = self.max_distance_memory
        tr_bytes, tt_bytes = 8 * m * t, 8 * t**2
        scratch = min(max(tr_bytes, tt_bytes), _PAIRWISE_SCRATCH)
        tr = tt = None
        if self.rr is not None and (budget is None or self.rr.nbytes + tr_bytes + tt_bytes + scratch <= budget):
            try:
                tr = _distances(target, self.reference)
                tt = _distances(target, target)
            except MemoryError:
                # The retained marginals suffice for another target or a streamed retry.
                self.rr = tr = None
                self.max_distance_memory = budget = max(8, min(_PAIRWISE_SCRATCH, max(tr_bytes, tt_bytes) // 4))
        if tr is not None and tt is not None:
            tr -= self.offset
            tt -= self.offset
            cp.fill_diagonal(tt, 0)
            target_cross, ref_cross, within_target = tr.sum(axis=1), tr.sum(axis=0), tt.sum(axis=1)
            kernel = _selected_rows
            inputs = (np.int32(t), self.rr, tr, tt)
        else:
            available = budget if self.rr is None else budget - self.rr.nbytes
            target_cross, ref_cross = _tile_marginals(target, self.reference, self.offset, available)
            within_target, _ = _tile_marginals(target, target, self.offset, available)
            kernel = _selected_feature_rows
            inputs = (np.int32(target.shape[1]), self.reference, target, self.offset)
        marginal = cp.concatenate((self.marginal + ref_cross, target_cross + within_target))
        total = self.total + 2 * ref_cross.sum() + within_target.sum()

        # Shuffles use AnnData observation order. Translate them to reference-first indices.
        order = np.empty(m + t, dtype=np.int32)
        order[~membership] = np.arange(m)
        order[membership] = np.arange(m, m + t)
        order = cp.asarray(order)
        scores = cp.empty(n_perms + 1, dtype=cp.float64)
        for start, indices in batches:
            selected = order[indices]
            parts = cp.empty(selected.shape, dtype=cp.float64)
            kernel(
                ((k + 7) // 8, len(selected)),
                (256,),
                (selected, np.int32(k), np.int32(m), *inputs, parts),
            )
            within = parts.sum(axis=1)
            rows = marginal[selected].sum(axis=1)
            cross = rows - within
            if statistic == "mean_pairwise":
                batch = cross / (k * large) + self.offset
            else:
                other = total - 2 * rows + within
                if statistic == "edistance":
                    # The V statistic includes zero diagonals in the within-group averages.
                    batch = 2 * cross / (k * large) - within / k**2 - other / large**2
                    batch += self.offset * (1 / k + 1 / large)
                else:
                    batch = offdiagonal_energy(cross, within, other, k, large, offset=self.offset)
            scores[start : start + len(selected)] = batch
        return scores.get()
