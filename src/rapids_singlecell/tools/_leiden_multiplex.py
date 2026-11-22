from __future__ import annotations

import math
from copy import copy
from types import SimpleNamespace

import cupy as cp
import numpy as np
from cupyx.scipy import sparse

from ._leiden import (
    _N_ITER_WARN,
    _as_csr,
    _compact_initial_membership,
    _cuda_memory_errors,
    _drop_ignored_entries,
    _ingest,
    _input_device,
    _workspace_cache_applies,
)
from ._leiden_multiplex_moves import _group_sum, move

# An explicit non-flushing conversion keeps float32 subnormal edges in both weight modes.
_to_float64 = cp.ElementwiseKernel(  # type: ignore[attr-defined]  # missing from cupy-stubs
    "float32 x", "float64 y", 'asm("cvt.f64.f32 %0, %1;" : "=d"(y) : "f"(x));', "multiplex_float64"
)


def _layer(adjacency, use_weights):
    """Apply native Leiden's CSR and weight semantics before moving or contracting vertices."""
    device = cp.cuda.Device().id
    source = _input_device(adjacency)
    with cp.cuda.Device(device if source is None else source):
        csr = _as_csr(adjacency)
    if source not in (None, device):
        from ._rank_genes_groups._wilcoxon_host import _copy_device_sparse_to_device

        csr = _copy_device_sparse_to_device(csr, device)
    if csr.nnz:
        if csr.dtype == np.float32:
            csr = copy(csr)
            csr.data = _to_float64(csr.data) if isinstance(csr.data, cp.ndarray) else csr.data.astype(np.float64)
        graph, info, weighted = _ingest(csr, use_weights=use_weights, host_input=False)
        if info["nnz_counted"]:
            if info["nnz_counted"] < graph.indices.size:
                graph = _drop_ignored_entries(graph)
            data = (
                graph.data.astype(cp.float64, copy=False) if weighted else cp.ones(graph.indices.size, dtype=cp.float64)
            )
            return sparse.csr_matrix((data, graph.indices, graph.indptr), shape=csr.shape)
    return sparse.csr_matrix(csr.shape, dtype=cp.float64)


def _contract(layer, groups, count):
    # Each layer keeps its own coarse graph. Its diagonal carries internal edges to the next level.
    edges = layer.tocoo(copy=False)
    keys = groups[edges.row].astype(cp.int64) * count + groups[edges.col]
    keys, data = _group_sum(keys, edges.data)
    rows = (keys // count).astype(cp.int32)
    offsets = cp.searchsorted(rows, cp.arange(count + 1, dtype=cp.int64))
    return sparse.csr_matrix((data, (keys % count).astype(cp.int32), offsets), shape=(count, count))


def _connected(layers, labels):
    """Split disconnected communities on the union of the positive-weight layers."""
    sources, targets = [], []
    for layer in layers:
        edges = layer.tocoo(copy=False)
        keep = (edges.row < edges.col) & (labels[edges.row] == labels[edges.col])
        sources.append(edges.row[keep])
        targets.append(edges.col[keep])
    u, v = cp.concatenate(sources), cp.concatenate(targets)
    roots = cp.arange(labels.size, dtype=cp.int32)
    while u.size:
        cp.minimum.at(roots, cp.maximum(u, v), cp.minimum(u, v))
        while not cp.array_equal(roots, shortcut := roots[roots]):
            roots = shortcut
        u, v = roots[u], roots[v]
        pending = u != v
        u, v = u[pending], v[pending]
    return cp.unique(roots, return_inverse=True)[1].astype(cp.int32)


def _iteration(layers, labels, weights, resolutions, rng):
    level = layers
    projection = cp.arange(labels.size, dtype=cp.int32)
    while True:
        labels = move(level, labels, weights, resolutions, rng=rng)
        singletons = cp.arange(labels.size, dtype=cp.int32)
        refined = move(level, singletons, weights, resolutions, parent=labels, rng=rng)
        _, representatives, groups = cp.unique(refined, return_index=True, return_inverse=True)
        count = representatives.size
        if count == labels.size:
            return _connected(layers, labels[projection])
        groups = groups.astype(cp.int32)
        projection = groups[projection]
        level = [_contract(layer, groups, count) for layer in level]
        labels = cp.unique(labels[representatives], return_inverse=True)[1].astype(cp.int32)


def _modularity(layer, labels, resolution):
    """Normalized RB quality on a unit-volume layer."""
    if not layer.nnz:
        return 0.0
    edges = layer.tocoo(copy=False)
    communities = labels[edges.row]
    internal = edges.data[communities == labels[edges.col]].sum()
    _, strengths = _group_sum(labels, cp.asarray(layer.sum(axis=1)).ravel())
    return float(internal - resolution * (strengths @ strengths))


@_cuda_memory_errors()
def leiden_multiplex(graphs, resolutions, weights, *, use_weights, n_iterations, rng, initial_membership=None):
    """Optimize one partition for the layer-weighted sum of RBConfiguration qualities."""
    layers = [_layer(graph, use_weights) for graph in graphs]
    n = layers[0].shape[0]
    if any(layer.shape != (n, n) for layer in layers):
        raise ValueError("All multiplex graphs must have the same shape.")
    labels = _compact_initial_membership(initial_membership, n)
    if labels is None:
        labels = cp.arange(n, dtype=cp.int32)

    # Normalize each layer for the gain calculations. Its original volume restores the
    # unnormalized RB objective, so scaling one input graph still changes its influence.
    volumes = [float(layer.data.sum()) for layer in layers]
    masses = np.asarray(volumes, dtype=np.longdouble) * weights
    if not np.isfinite(volumes).all() or not np.isfinite(masses).all():
        raise ValueError("Multiplex layer volumes must be finite. Rescale the input weights.")
    for layer, volume in zip(layers, volumes, strict=True):
        if volume:
            layer.data = layer.data / volume
    modularity = [0.0] * len(layers)
    active = np.flatnonzero(masses)
    if active.size:
        level = [layers[i] for i in active]
        gamma = [resolutions[i] for i in active]
        factors = np.asarray(masses[active] / max(masses), dtype=np.float64)
        if any(
            float(layer.data.min()) * w < np.finfo(np.float64).tiny for layer, w in zip(level, factors, strict=True)
        ):
            raise ValueError("Multiplex edge weights span too wide a range for float64. Rescale the layers or weights.")
        previous = -math.inf
        # Reuse the short-lived move and refinement buffers within this call when RMM
        # would otherwise synchronize on each free. The caller supplies the memory.
        allocator = cp.cuda.get_allocator()
        if _workspace_cache_applies():
            allocator = cp.cuda.MemoryPool(allocator).malloc
        with cp.cuda.using_allocator(allocator):
            for iteration in range(_N_ITER_WARN if n_iterations == -1 else n_iterations):
                candidate = _iteration(level, labels, factors, gamma, rng)
                if iteration and cp.array_equal(candidate, labels):
                    break
                scores = [_modularity(layer, candidate, res) for layer, res in zip(level, gamma, strict=True)]
                score = math.fsum(factors * scores)
                if score <= previous:
                    break
                labels, previous = candidate, score
                for i, value in zip(active, scores, strict=True):
                    modularity[i] = value
    else:
        labels = cp.arange(n, dtype=cp.int32)

    for i in np.flatnonzero(np.asarray(weights) == 0):
        modularity[i] = _modularity(layers[i], labels, resolutions[i])
    quality = float(masses @ np.asarray(modularity, dtype=np.longdouble))
    if not math.isfinite(quality):
        raise ValueError("Multiplex quality exceeds float64 range. Rescale the layer weights.")
    # Match native Leiden's label order: largest first, then smallest original vertex.
    _, inverse, counts = np.unique(labels.get(), return_inverse=True, return_counts=True)
    order = np.argsort(-counts, kind="stable")
    membership = np.argsort(order)[inverse].astype(np.int32)
    return SimpleNamespace(membership=membership, modularity=modularity, quality=quality, n_clusters=len(counts))
