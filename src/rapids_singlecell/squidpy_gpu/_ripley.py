from __future__ import annotations

from typing import TYPE_CHECKING, Literal

import cupy as cp
import numpy as np
import pandas as pd
from scipy.spatial import ConvexHull

from rapids_singlecell._cuda import _spatial_cuda
from rapids_singlecell._utils import _create_category_index_mapping

from ._co_oc import _co_occurrence_gpu
from ._spatial_data import _extract_adata
from ._spatial_neighbors_backend import _build_kdtree
from ._utils import _assert_categorical_obs, _assert_spatial_basis

if TYPE_CHECKING:
    from anndata import AnnData
    from spatialdata import SpatialData

# Libraries per pair-count launch; its counts buffer grows with the square.
_PAIR_CHUNK = 128


def ripley(
    adata: AnnData | SpatialData,
    cluster_key: str,
    *,
    mode: Literal["F", "G", "L"] = "F",
    spatial_key: str = "spatial",
    metric: str = "euclidean",
    n_neigh: int = 2,
    n_simulations: int = 100,
    n_observations: int = 1000,
    max_dist: float | None = None,
    n_steps: int = 50,
    rng: int | np.random.Generator | None = None,
    copy: bool = False,
    table_key: str | None = None,
) -> dict[str, pd.DataFrame | np.ndarray] | None:
    r"""
    Calculate Ripley's statistics for point processes.

    GPU port of :func:`squidpy.gr.ripley`. Depending on ``mode``, it computes
    Ripley's ``'F'``, ``'G'`` or ``'L'`` statistic per cluster and compares it
    to simulated Spatial Poisson Point Processes on the convex hull of all
    observations.

    ``'F'`` and ``'G'`` are defined as

    .. math::

        F(t),G(t)=P( d_{i,j} \le t )

    where :math:`d_{i,j}` are the distances from random points of a Spatial
    Poisson Point Process (``'F'``) or from points of other clusters (``'G'``)
    to their ``n_neigh`` nearest points of the cluster. ``'L'`` stabilizes the
    variance of

    .. math::

        K(t) = \frac{1}{\lambda} \sum_{i \ne j} \frac{I(d_{i,j} \le t)}{n}

    as :math:`L(t) = (K(t) / \pi)^{1/2}`.

    Parameters
    ----------
    adata
        Annotated data matrix or a SpatialData object containing the selected table.
    cluster_key
        Key in :attr:`anndata.AnnData.obs` with the cluster labels.
    mode
        Which Ripley's statistic to compute.
    spatial_key
        Key in :attr:`anndata.AnnData.obsm` with the 2D spatial coordinates.
    metric
        Distance metric. Only ``'euclidean'`` is supported.
    n_neigh
        Number of nearest neighbors whose distances enter ``'F'`` and ``'G'``.
    n_simulations
        How many simulations to run for computing p-values.
    n_observations
        How many observations to generate for each Spatial Poisson Point Process.
    max_dist
        Maximum distance of the support. If `None`, ``max_dist`` is
        :math:`\sqrt{area / 2}` of the convex hull.
    n_steps
        Number of steps of the support.
    rng
        Seed or :class:`numpy.random.Generator` that seeds the GPU random
        number generator for the simulated point processes.
    copy
        If ``True``, return the result, otherwise save it to ``adata``.
    table_key
        Key in ``SpatialData.tables``; required for SpatialData input.
        All reads and writes use this table; ignored for AnnData input.

    Returns
    -------
    If ``copy = True``, returns a :class:`dict` with the following keys:

        - ``'{mode}_stat'`` - :class:`pandas.DataFrame` with the statistic per cluster and bin.
        - ``'sims_stat'`` - :class:`pandas.DataFrame` with the statistic per simulation and bin.
        - ``'bins'`` - the support of the statistic.
        - ``'pvalues'`` - p-values of shape ``(n_clusters, n_steps)``.

    Otherwise, modifies the ``adata`` with the following key:

        - :attr:`anndata.AnnData.uns` ``['{cluster_key}_ripley_{mode}']`` - the above mentioned :class:`dict`.

    Notes
    -----
    Distances for ``'F'`` and ``'G'`` are exact in double precision. ``'L'``
    counts pairs in single precision around the centroid of the coordinates,
    so pairs within rounding error of a support value may land in the
    neighboring bin. Random points are drawn on the GPU, so simulations do not
    reproduce :func:`squidpy.gr.ripley` for the same ``rng``.
    """
    adata = _extract_adata(adata, table_key=table_key)
    _assert_categorical_obs(adata, key=cluster_key)
    _assert_spatial_basis(adata, key=spatial_key)
    if mode not in ("F", "G", "L"):
        raise ValueError(
            f"Expected `mode` to be one of 'F', 'G' or 'L', found {mode!r}."
        )
    if metric != "euclidean":
        raise ValueError(
            f"Unsupported metric {metric!r}. Ripley's statistics on the GPU "
            "support only 'euclidean'."
        )

    spatial = adata.obsm[spatial_key]
    coordinates = np.ascontiguousarray(
        spatial.get() if isinstance(spatial, cp.ndarray) else spatial,
        dtype=np.float64,
    )
    if coordinates.ndim != 2 or coordinates.shape[1] != 2:
        raise ValueError(
            f"Expected 2D spatial coordinates, found shape {coordinates.shape}."
        )
    clusters = adata.obs[cluster_key]
    if clusters.isna().any():
        raise ValueError(
            f"`adata.obs[{cluster_key!r}]` has `{clusters.isna().sum()}` missing "
            "labels; drop or label those observations first."
        )
    classes, cluster_idx = np.unique(np.asarray(clusters), return_inverse=True)
    n_clusters = len(classes)
    if mode != "L":
        sizes = np.bincount(cluster_idx, minlength=n_clusters)
        if sizes.min() < n_neigh:
            small = classes[sizes.argmin()]
            raise ValueError(
                f"Cluster {small!r} has `{sizes.min()}` observations, fewer than "
                f"`n_neigh={n_neigh}`."
            )

    N = coordinates.shape[0]
    hull = ConvexHull(coordinates)
    area = hull.volume
    if max_dist is None:
        max_dist = (area / 2) ** 0.5
    support = np.linspace(0, max_dist, n_steps)

    # Every random draw, F's per-cluster points first, comes from one stream.
    n_patterns = n_simulations + (n_clusters if mode == "F" else 0)
    generator = cp.random.default_rng(
        np.random.default_rng(rng).integers(np.iinfo(np.int64).max)
    )
    random = _ppp(hull, n_patterns * n_observations, generator)
    random = random.reshape(n_patterns, n_observations, 2)
    sims_random = random[n_patterns - n_simulations :]
    sim_codes = cp.repeat(cp.arange(n_simulations, dtype=cp.int32), n_observations)
    coords = cp.asarray(coordinates)
    codes = cp.asarray(cluster_idx, dtype=cp.int32)

    if mode == "L":
        counts = (
            _pair_counts(coords, codes, n_clusters, support),
            _pair_counts(sims_random.reshape(-1, 2), sim_codes, n_simulations, support),
        )
        # Unordered pairs count twice as ordered pairs.
        obs_arr, sims = (np.sqrt(2 * c / N / (N / area) / np.pi) for c in counts)
    else:
        support_gpu = cp.asarray(support)
        tree = _build_kdtree(coords, codes)
        sim_tree = _build_kdtree(sims_random.reshape(-1, 2), sim_codes)
        obs_hist = cp.empty((n_clusters, n_steps - 1), dtype=cp.int64)
        sim_hist = cp.empty((n_simulations, n_steps - 1), dtype=cp.int64)
        for i in range(n_clusters):
            queries = random[i] if mode == "F" else coords[codes != i]
            distances = _query(queries, tree, i, n_neigh)
            obs_hist[i] = cp.histogram(distances, support_gpu)[0]
        # Simulations query with F's points of the last cluster, like squidpy.
        queries = random[n_clusters - 1] if mode == "F" else coords
        for i in range(n_simulations):
            sim_hist[i] = cp.histogram(_query(queries, sim_tree, i, 1), support_gpu)[0]
        obs_arr, sims = (_f_g_function(h.get()) for h in (obs_hist, sim_hist))

    pvalues = 1 + (sims[None, :, :] >= obs_arr[:, None, :]).sum(axis=1)
    pvalues = pvalues / (n_simulations + 1)
    pvalues = np.minimum(pvalues, 1 - pvalues)

    obs_df = _reshape_res(
        obs_arr.T, columns=classes, index=support, var_name=cluster_key
    )
    sims_df = _reshape_res(
        sims.T,
        columns=np.arange(n_simulations),
        index=support,
        var_name="simulations",
    )
    res = {
        f"{mode}_stat": obs_df,
        "sims_stat": sims_df,
        "bins": support,
        "pvalues": pvalues,
    }
    if copy:
        return res
    adata.uns[f"{cluster_key}_ripley_{mode}"] = res


def _ppp(hull: ConvexHull, n: int, generator: cp.random.Generator) -> cp.ndarray:
    """Draw ``n`` uniform points in a convex hull by rejection from its bounding box."""
    lower, upper = hull.min_bound, hull.max_bound
    acceptance = hull.volume / np.prod(upper - lower)
    # Inside points satisfy normal @ point + offset <= 0 for every facet.
    equations = cp.asarray(hull.equations)
    max_batch = max((1 << 26) // len(equations), 1024)
    lower, extent = cp.asarray(lower), cp.asarray(upper - lower)
    parts, found = [], 0
    while found < n:
        batch = min(int((n - found) / acceptance * 1.1) + 64, max_batch)
        candidates = lower + generator.random((batch, 2)) * extent
        inside = (candidates @ equations[:, :2].T + equations[:, 2] <= 0).all(axis=1)
        parts.append(candidates[inside])
        found += len(parts[-1])
    return cp.ascontiguousarray(cp.concatenate(parts)[:n])


def _query(
    queries: cp.ndarray,
    tree: tuple[cp.ndarray, cp.ndarray, cp.ndarray, cp.ndarray],
    library: int,
    k: int,
) -> cp.ndarray:
    """Distances from ``queries`` to their ``k`` nearest points of one library."""
    points, index, boxes, segments = tree
    columns = cp.empty((len(queries), k), dtype=cp.int64)
    distances = cp.empty((len(queries), k), dtype=queries.dtype)
    _spatial_cuda.tree_search(
        cp.ascontiguousarray(queries),
        points,
        index,
        boxes,
        segments[library : library + 2],
        None,
        k,
        0.0,
        None,
        columns=columns,
        distances=distances,
        stream=cp.cuda.get_current_stream().ptr,
        external=True,
    )
    return distances


def _f_g_function(hist: np.ndarray) -> np.ndarray:
    """Cumulative fractions of each histogram, after a leading zero."""
    with np.errstate(invalid="ignore", divide="ignore"):
        fracs = np.cumsum(hist, axis=1) / hist.sum(axis=1, keepdims=True)
    return np.pad(fracs, ((0, 0), (1, 0)))


def _pair_counts(
    points: cp.ndarray, codes: cp.ndarray, n_libraries: int, support: np.ndarray
) -> np.ndarray:
    """Unordered pairs of each library within each support distance."""
    # Centering keeps more float32 bits; distances do not change.
    points = (points - points.mean(axis=0)).astype(cp.float32)
    thresholds = cp.asarray(support**2, dtype=cp.float32)
    device = cp.cuda.Device().id
    counts = np.empty((n_libraries, len(support)), dtype=np.int64)
    for start in range(0, n_libraries, _PAIR_CHUNK):
        stop = min(start + _PAIR_CHUNK, n_libraries)
        in_chunk = (codes >= start) & (codes < stop)
        chunk_points = cp.ascontiguousarray(points[in_chunk])
        cat_offsets, cell_indices = _create_category_index_mapping(
            codes[in_chunk] - start, stop - start
        )
        diagonal = cp.arange(stop - start, dtype=cp.int32)
        chunk, ok = _co_occurrence_gpu(
            spatial=chunk_points,
            thresholds=thresholds,
            cat_offsets=cat_offsets,
            cell_indices=cell_indices,
            pair_left=diagonal,
            pair_right=diagonal,
            n_cells=len(chunk_points),
            k=stop - start,
            l_val=len(support),
            device_ids=[device],
        )
        if not ok:
            raise ValueError(
                f"`n_steps={len(support)}` needs more shared memory than the "
                "GPU provides for Ripley's L."
            )
        counts[start:stop] = chunk[diagonal, diagonal].get()
    return counts


def _reshape_res(
    results: np.ndarray, columns: np.ndarray, index: np.ndarray, var_name: str
) -> pd.DataFrame:
    df = pd.DataFrame(results, columns=columns, index=index)
    df.index.set_names(["bins"], inplace=True)
    df = df.melt(var_name=var_name, value_name="stats", ignore_index=False)
    df[var_name] = df[var_name].astype("category", copy=True)
    df.reset_index(inplace=True)
    return df
