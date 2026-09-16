from __future__ import annotations

from copy import deepcopy

import cupy as cp
import numpy as np

from rapids_singlecell._cuda import _harmony_clustering_cuda as _clustering_cuda

_ASSIGNMENT_BATCH_ROWS = 65_536
_DISTANCE_ERROR_FACTOR = 4
_WEIGHT_TILE_ROWS = 1024
_HOST_COUNTS_MAX_CLUSTERS = 4096

_refine_squared_distances = cp.ElementwiseKernel(
    "T distance, raw T X, raw T centers, raw T x_norm, raw T center_norm, "
    "int32 n_features, int32 n_clusters, T error_factor",
    "T out",
    """
    int row = i / n_clusters;
    int cluster = i % n_clusters;
    out = distance + x_norm[row] + center_norm[cluster];
    if (out <= error_factor * (x_norm[row] + center_norm[cluster])) {
        T sum = 0;
        for (int feature = 0; feature < n_features; ++feature) {
            T delta = X[(size_t)row * n_features + feature]
                    - centers[(size_t)cluster * n_features + feature];
            sum += delta * delta;
        }
        out = sum;
    }
    """,
    "harmony_refine_squared_distances",
)


def _squared_distances(X, centers, *, squared_norms):
    center_norms = cp.sum(centers * centers, axis=1)
    distances = -2 * (X @ centers.T)
    # Recompute near-zero pairs where subtracting the dot product loses precision.
    error_factor = _DISTANCE_ERROR_FACTOR * np.finfo(X.dtype).eps * (X.shape[1] + 2)
    _refine_squared_distances(
        distances,
        X,
        centers,
        squared_norms,
        center_norms,
        X.shape[1],
        centers.shape[0],
        X.dtype.type(error_factor),
        distances,
    )
    return distances


def _assign_labels(X, centers, *, squared_norms, labels, minimum):
    for begin in range(0, X.shape[0], _ASSIGNMENT_BATCH_ROWS):
        end = min(begin + _ASSIGNMENT_BATCH_ROWS, X.shape[0])
        distances = _squared_distances(
            X[begin:end], centers, squared_norms=squared_norms[begin:end]
        )
        batch_labels = cp.argmin(distances, axis=1).astype(cp.int32)
        labels[begin:end] = batch_labels
        minimum[begin:end] = distances[cp.arange(end - begin), batch_labels]


def _kmeans(X, n_clusters, *, max_iter, rng):
    """Initialize Harmony with k-means++ and fixed-order Lloyd reductions."""
    n_rows, n_cols = X.shape
    if not 1 <= n_clusters <= n_rows:
        raise ValueError("n_clusters must be between 1 and the number of sampled cells")
    rng = np.random.default_rng(rng)
    squared_norms = cp.sum(X * X, axis=1)
    centers = cp.empty((n_clusters, n_cols), dtype=X.dtype)
    closest = cp.full(n_rows, cp.inf, dtype=X.dtype)
    index = int(rng.integers(n_rows))
    centers[0] = X[index]
    uniforms = cp.asarray(deepcopy(rng).random(n_clusters - 1))
    n_draws = cp.zeros(1, dtype=cp.int32)
    totals = cp.empty(
        (n_rows + _WEIGHT_TILE_ROWS - 1) // _WEIGHT_TILE_ROWS, dtype=cp.float64
    )
    for cluster in range(n_clusters):
        candidate = _squared_distances(
            X, centers[cluster : cluster + 1], squared_norms=squared_norms
        ).ravel()
        cp.minimum(closest, candidate, out=closest)
        if cluster + 1 < n_clusters:
            _clustering_cuda.select_kmeans_center(
                X,
                weights=closest,
                uniforms=uniforms,
                centers=centers,
                totals=totals,
                n_draws=n_draws,
                cluster=cluster + 1,
                stream=cp.cuda.get_current_stream().ptr,
            )
    # Zero-weight fallback choices consume no random draw.
    rng.random(int(n_draws[0]))

    labels = cp.empty(n_rows, dtype=cp.int32)
    previous = cp.full(n_rows, -1, dtype=cp.int32)
    minimum = cp.empty(n_rows, dtype=X.dtype)
    sums = cp.empty_like(centers)
    workspace = cp.empty(
        _clustering_cuda.get_scatter_temp_bytes(
            n_rows=n_rows,
            n_cols=n_cols,
            n_categories=n_clusters,
            itemsize=X.itemsize,
        ),
        dtype=cp.uint8,
    )
    has_empty = True
    for _ in range(max_iter):
        _assign_labels(
            X, centers, squared_norms=squared_norms, labels=labels, minimum=minimum
        )
        if not has_empty and bool(cp.array_equal(labels, previous)):
            break
        previous[:] = labels
        counts = cp.bincount(labels, minlength=n_clusters)
        sums.fill(0)
        _clustering_cuda.scatter_add(
            X,
            categories=labels,
            out=sums,
            workspace=workspace,
            n_rows=n_rows,
            n_cols=n_cols,
            n_categories=n_clusters,
            switcher=1,
            stream=cp.cuda.get_current_stream().ptr,
        )
        cp.divide(sums, counts[:, None], out=centers)
        empty = (
            np.flatnonzero(counts.get() == 0)
            if n_clusters <= _HOST_COUNTS_MAX_CLUSTERS
            else cp.flatnonzero(counts == 0).get()
        )
        has_empty = bool(empty.size)
        if has_empty:
            host_distance = minimum.get()
            for cluster in empty:
                index = int(np.argmax(host_distance))
                centers[cluster] = X[index]
                host_distance[index] = -np.inf
    return centers
