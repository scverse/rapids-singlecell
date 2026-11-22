from __future__ import annotations

import cupy as cp
import numpy as np


def _group_sum(keys, values, *, sorted_keys=False):
    """Sum values for each integer key, in a stable order on the GPU."""
    if not keys.size:
        return keys, values
    if not sorted_keys:
        order = cp.argsort(keys, kind="stable")
        keys, values = keys[order], values[order]
    starts = cp.concatenate((cp.zeros(1, dtype=cp.int64), cp.flatnonzero(keys[1:] != keys[:-1]) + 1))
    return keys[starts], cp.add.reduceat(values, starts)


def _totals(keys, values, n):
    keys, sums = _group_sum(keys, values)
    out = cp.zeros(n, dtype=values.dtype)
    out[keys] = sums
    return out


def move(layers, labels, weights, resolutions, *, parent=None, rng):
    """Move vertices on the shared partition, or refine within a fixed parent partition."""
    n = labels.size
    labels = labels.astype(cp.int32, copy=True)
    if n < 2:
        return labels
    keys, values, null = [], [], []
    for layer, weight, gamma in zip(layers, weights, resolutions, strict=True):
        if not weight or not layer.nnz:
            continue
        degree = cp.asarray(layer.sum(axis=1)).ravel()
        total = float(degree.sum())
        if not total:
            continue
        if gamma:
            null.append((degree, weight * gamma * (degree / total)))
        coo = layer.tocoo()
        keep = coo.row != coo.col
        if parent is not None:
            keep &= parent[coo.row] == parent[coo.col]
        keys.append(coo.row[keep].astype(cp.int64) * n + coo.col[keep])
        values.append(weight * coo.data[keep])
    if not keys:
        return labels
    edges, value = _group_sum(cp.concatenate(keys), cp.concatenate(values))
    row, col = (edges // n).astype(cp.int32), (edges % n).astype(cp.int32)
    priority = cp.asarray(rng.permutation(n), dtype=cp.int32)
    vertices = cp.arange(n, dtype=cp.int32)
    eps = 16 * np.finfo(np.float64).eps
    if parent is not None:
        parent_volume = [_totals(parent, degree, n) for degree, _ in null]

    while True:
        pair, observed = _group_sum(row.astype(cp.int64) * n + labels[col], value)
        v, target = (pair // n).astype(cp.int32), (pair % n).astype(cp.int32)
        own = target == labels[v]
        stay_edges = cp.zeros(n, dtype=cp.float64)
        stay_edges[v[own]] = observed[own]
        stay_null = cp.zeros_like(stay_edges)
        target_null = cp.zeros_like(observed)
        counts = cp.bincount(labels, minlength=n)
        if parent is not None:
            cut = _totals(labels[v[~own]], observed[~own], n)
            bound = cp.zeros_like(cut)
            representative = cp.zeros(n, dtype=cp.int32)
            cp.maximum.at(representative, labels, vertices)
        for i, (degree, penalty) in enumerate(null):
            volume = _totals(labels, degree, n)
            stay_null += penalty * (volume[labels] - degree)
            target_null += penalty[v] * volume[target]
            if parent is not None:
                # A refined group can merge while its cut to the rest of its parent
                # meets the weighted configuration-model expectation.
                mass = _totals(labels, penalty, n)
                bound += mass * (parent_volume[i][parent[representative]] - volume)

        stay = stay_edges - stay_null
        gain = (observed - target_null) - stay[v]
        improving = ~own & (gain > eps * (observed + target_null + stay_edges[v] + stay_null[v]))
        if parent is not None:
            well = cut + eps * (cut + bound) >= bound
            improving &= (counts[labels[v]] == 1) & well[labels[v]] & well[target]
            charge = cp.maximum(0, observed - 2 * target_null + bound[labels[v]])[improving]
        v, target, gain = v[improving], target[improving], gain[improving]
        if parent is None:
            # Leaving for a new singleton has score zero (self-loops cancel).
            free = vertices[(counts[labels] > 1) & (-stay > eps * (stay_edges + stay_null))]
            v = cp.concatenate((v, free))
            target = cp.concatenate((target, cp.full(free.size, n, dtype=cp.int32)))
            gain = cp.concatenate((gain, -stay[free]))
        if not v.size:
            return labels

        # Choose each vertex's best destination and seed disjoint source/target
        # pairs. Each source contributes at most one vertex to a batch.
        order = cp.lexsort(cp.stack((target, -gain, v)))
        v, target, gain = v[order], target[order], gain[order]
        first = cp.concatenate((cp.ones(1, dtype=cp.bool_), v[1:] != v[:-1]))
        v, target, gain = v[first], target[first], gain[first]
        source, rank = labels[v], priority[v]
        owner = cp.full(n + 1, n, dtype=cp.int32)
        cp.minimum.at(owner, source, rank)
        leads = owner[source] == rank
        cp.minimum.at(owner, target, rank)
        seed = (owner[source] == rank) & ((target == n) | (owner[target] == rank))
        receives = cp.zeros(n + 1, dtype=cp.bool_)
        receives[target[seed]] = True
        batch = leads & receives[target] & ~receives[source]
        v, target, gain, seed = v[batch], target[batch], gain[batch], seed[batch]
        # More sources can join each seeded target if the move stays profitable
        # even after every other entrant's strength has been added to its volume.
        extra = cp.zeros_like(gain)
        for degree, penalty in null:
            incoming = _totals(target, degree[v], n + 1)
            extra += penalty[v] * (incoming[target] - degree[v])
        commit = seed | (target == n) | (gain > (1 + eps) * extra)
        if parent is not None:
            # Joining a singleton reduces the target's surplus over its parent
            # cut bound by at most charge, including all edges to other entrants.
            # With j prior entrants, a surplus >= j * max_charge certifies the cut.
            charge = charge[order][first][batch]
            by_target = cp.argsort(target.astype(cp.int64) * n + priority[v], kind="stable")
            t = target[by_target]
            starts = cp.concatenate((cp.zeros(1, dtype=cp.int64), cp.flatnonzero(t[1:] != t[:-1]) + 1))
            lengths = cp.diff(cp.append(starts, t.size))
            position = cp.arange(t.size) - cp.repeat(starts, lengths)
            max_charge = cp.repeat(cp.maximum.reduceat(charge[by_target], starts), lengths)
            commit[by_target] &= cut[t] + eps * (cut[t] + bound[t]) >= bound[t] + position * max_charge
        v, target = v[commit], target[commit]
        singleton = cp.flatnonzero(target == n)
        if singleton.size:
            target[singleton] = cp.flatnonzero(counts == 0)[: singleton.size]
        labels[v] = target
