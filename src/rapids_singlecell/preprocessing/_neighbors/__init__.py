from __future__ import annotations

from types import MappingProxyType
from typing import TYPE_CHECKING, Any, Literal, NamedTuple, TypedDict, get_args
from warnings import warn

import cupy as cp
import numpy as np
from scverse_misc import Deprecation, deprecated_arg

from rapids_singlecell._utils._random import (
    RNGLike,
    SeedLike,
    _accepts_legacy_random_state,
    _LegacyRng,
)
from rapids_singlecell.get._out import (
    DEPR_KEY_ADDED,
    _key,
    _out_dests,
    _resolve,
    _resolve_obs,
    _use_to_uns,
    _write_dests,
    deprecated_copy,
)
from rapids_singlecell.get._utils import _get_vec, _to_numpy_1d
from rapids_singlecell.preprocessing._neighbors._helper import (
    _check_knn_complete,
    _check_metrics,
    _check_neighbors_X,
    _fix_self_distances,
    _trimming,
)
from rapids_singlecell.preprocessing._neighbors._neighbors import (
    KNN_ALGORITHMS,
    _Algorithms,
    _build_sparse_distances,
    _calc_connectivities,
    _connectivities_to_host_csr,
    _Metrics,
)
from rapids_singlecell.preprocessing._neighbors._wnn import WnnResult, wnn
from rapids_singlecell.tools._utils import _choose_representation

if TYPE_CHECKING:
    from collections.abc import Mapping

    from anndata import AnnData

    from rapids_singlecell.get._utils import AdRef, GraphAcc, LayerAcc, MultiAcc

_Algorithms_bbknn = Literal["brute", "cagra", "ivfflat", "ivfpq", "mg_ivfflat", "mg_ivfpq"]

_DEFAULT_SEED = 0


class NeighborsResult(NamedTuple):
    """Outputs of :func:`~rapids_singlecell.pp.neighbors` and :func:`~rapids_singlecell.pp.bbknn`."""

    connectivities: Any
    """Weighted adjacency matrix of the neighborhood graph."""
    distances: Any
    """Distances to each observation's nearest neighbors."""


class NeighborsOut(TypedDict, total=False):
    """Where a mapping `out` of :func:`~rapids_singlecell.pp.neighbors` writes each output."""

    connectivities: GraphAcc | str
    distances: GraphAcc | str


def _neighbors_dests(key_added: str | None) -> NeighborsResult:
    prefix = "" if key_added is None else f"{key_added}_"
    return NeighborsResult({"obsp": f"{prefix}connectivities"}, {"obsp": f"{prefix}distances"})


def _neighbors_out(out, key_added):
    return _out_dests(out, key_added, dests=_neighbors_dests, kinds=NeighborsResult({"obsp"}, {"obsp"}), uns=True)


def _store_neighbors(adata: AnnData, dests, vals: tuple, params: dict) -> tuple | None:
    """Write the graphs to `dests` (from `_out_dests`), and the `.uns` entry naming them."""
    if dests is True:
        return vals
    key, dests = dests
    entry = {f"{name}_key": _key(d) for name, d in zip(vals._fields, dests, strict=True) if d is not None}
    adata.uns[key or "neighbors"] = entry | {"params": params}
    _write_dests(adata, vals, dests)
    return None


@_accepts_legacy_random_state(_DEFAULT_SEED)
@deprecated_arg("use_rep", Deprecation("0.19.0", "Use `use` instead."))
@deprecated_arg("key_added", DEPR_KEY_ADDED)
@deprecated_copy
def neighbors(
    adata: AnnData,
    n_neighbors: int = 15,
    n_pcs: int | None = None,
    *,
    use: LayerAcc | MultiAcc | str | None = None,
    rng: SeedLike | RNGLike | None = None,
    algorithm: _Algorithms = "brute",
    metric: _Metrics = "euclidean",
    metric_kwds: Mapping[str, Any] = MappingProxyType({}),
    algorithm_kwds: Mapping[str, Any] = MappingProxyType({}),
    method: Literal["umap", "gauss", "jaccard"] = "umap",
    out: GraphAcc | str | NeighborsOut | bool = False,
    # deprecated
    use_rep: str | None = None,
    key_added: str | None = None,
    copy: bool = False,
) -> AnnData | NeighborsResult | None:
    """Compute a neighborhood graph of observations :cite:p:`Ootomo2024`.

    The neighbor search efficiency of this heavily relies on cuVS,
    which also provides a method for estimating connectivities of data points -
    the connectivity of the manifold.

    Parameters
    ----------
    adata
        Annotated data matrix.
    n_neighbors
        The size of local neighborhood (in terms of number of neighboring data
        points) used for manifold approximation. Larger values result in more
        global views of the manifold, while smaller values result in more local
        data being preserved. In general values should be in the range 2 to 100.
    n_pcs
        Use this many PCs. If `n_pcs==0` use `.X` if `use is None`.
    use
        The representation to use, e.g. `A.obsm['pca']` or `A.X`.
        Strings are resolved by :meth:`anndata.acc.AdAcc.resolve`, e.g. `'obsm.pca'`.
        If `None`, `.X` is used for `.n_vars < 50`, otherwise the PCA embedding
        (under the key the active ``rapids_singlecell.settings.preset`` uses), computed if missing.
    rng
        Random seed or :class:`~numpy.random.Generator` for reproducibility.
        The superseded `random_state` argument is still accepted.
    algorithm
        The query algorithm to use. Valid options are:
            `'brute'`
                Brute-force search that computes distances to all data points, guaranteeing exact results.

            `'ivfflat'`
                Uses inverted file indexing to partition the dataset into coarse quantizer cells and performs the search within the relevant cells.

            `'ivfpq'`
                Combines inverted file indexing with product quantization to encode sub-vectors of the dataset, facilitating faster distance computation.

            `'cagra'`
                Employs the Compressed, Accurate Graph-based search to quickly find nearest neighbors by traversing a graph structure.

            `'nn_descent'`
                Uses the NN-descent algorithm to approximate the k-nearest neighbors.
                Note: Performance may be degraded when Dask is active.

            `'all_neighbors'`
                Uses the all-neighbors algorithm to approximate the k-nearest neighbors.
                Note: Performance may be degraded when Dask is active and algorithm is `nn_descent`.

            `'mg_ivfflat'`
                Uses the Multi-GPU inverted file indexing to partition the dataset into coarse quantizer cells and performs the search within the relevant cells.

            `'mg_ivfpq'`
                Combines Multi-GPU inverted file indexing with product quantization to encode sub-vectors of the dataset, facilitating faster distance computation.

        Please ensure that the chosen algorithm is compatible with your dataset and the specific requirements of your search problem.
    metric
        A known metric's name or a callable that returns a distance.
        For ``inner_product``, ``distances`` stores raw similarities; UMAP and
        Gaussian weighting use positive score gaps, with self at distance zero.
    metric_kwds
        Options for the metric.
    method
        Method for computing connectivities.

        ``'umap'``
            UMAP fuzzy simplicial set (default).
        ``'gauss'``
            Adaptive Gaussian kernel.
        ``'jaccard'``
            PhenoGraph Jaccard index.
    algorithm_kwds
        Options for the algorithm.
        For `ivfflat` and `ivfpq` algorithms, the following parameters can be specified:

        * 'n_lists': Number of inverted lists for IVF indexing. Default is 2 * next_power_of_2(sqrt(n_samples)).

        * 'n_probes': Number of lists to probe during search. Default is 20. Higher values
          increase accuracy but reduce speed.

        For `nn_descent` algorithm, the following parameters can be specified:

        * 'intermediate_graph_degree': The degree of the intermediate graph. Default is cuVS's default (96),
          raised to `1.5 * n_neighbors` for larger `n_neighbors`. A smaller user-supplied value is raised
          to `n_neighbors`.

        For `cagra` algorithm, the following parameters can be specified:

        * 'itopk_size': Number of candidates kept during the search. Default is cuVS's default (64), raised
          to `n_neighbors` if smaller. Higher values increase accuracy but reduce speed.

        For `all_neighbors` algorithm, the following parameters can be specified:

        * 'algo': The algorithm to use. Valid options are: 'ivf_pq' and 'nn_descent'. Default is 'nn_descent'. `ivf_pq` is restricted to the `euclidean` and `sqeuclidean` metrics; use `nn_descent` for `cosine` and `inner_product`.

        * 'n_clusters': Number of clusters/batches to partition the dataset into (> overlap_factor). Default is 1 if the data fits on a single GPU, otherwise sized to the data and free memory (e.g. 24 for 100M cells on 8 GPUs).

        * 'overlap_factor': Number of clusters each point is assigned to. Must be < n_clusters when the build is batched (`n_clusters > 1`). Default is 1 unbatched, 2 with automatic `n_clusters` or clusters of at least 20,000 cells, and `min(max(2, ceil(log2(n_clusters))), n_clusters - 1)` for smaller ones. Lower values are faster but lose neighbors at cluster boundaries.

        * 'n_lists': Number of inverted lists for IVF indexing. Default is 2 * next_power_of_2(sqrt(n_samples)). Only available for `ivf_pq` algorithm.

        * 'graph_degree': The degree of the graph nn-descent builds before selecting the final `n_neighbors`. Default is 64 unbatched and `n_neighbors` batched, raised to `n_neighbors` if smaller. Only available for `nn_descent` algorithm.

        * 'intermediate_graph_degree': The degree of the intermediate graph. Default is `max(128, int(1.5 * graph_degree))`, following the recommended `>= 1.5 * graph_degree`. A smaller user-supplied value is raised to `graph_degree`. Only available for `nn_descent` algorithm.

        For `mg_ivfflat` and `mg_ivfpq` algorithms, the following parameters can be specified:

        * 'distribution_mode': The distribution mode to use. Valid options are: 'replicated' and 'shared'. Default is 'replicated'.

        * 'n_lists': Number of inverted lists for IVF indexing. Default is 2 * next_power_of_2(sqrt(n_samples)).

        * 'n_probes': Number of lists to probe during search. Default is 20. Higher values
          increase accuracy but reduce speed.

    out
        Where to write the graphs. A single accessor `A.obsp['k']` writes the connectivities to `.obsp['k']`,
        the distances to `.obsp['k_distances']` and the metadata to `.uns['k']`.
        A mapping (:class:`~rapids_singlecell.pp.NeighborsOut`) writes only the matrices it names,
        e.g. `{'connectivities': A.obsp['conns']}`.
        If `True`, they are returned as a :class:`~rapids_singlecell.pp.NeighborsResult`.
        If `False`, they are written to `.obsp['connectivities']`, `.obsp['distances']` and `.uns['neighbors']`.

    Returns
    -------
    A :class:`~rapids_singlecell.pp.NeighborsResult` if `out=True`, else `None`.
    See `out` for where these are written:

        **connectivities** : sparse matrix of dtype `float32`.
            Weighted adjacency matrix of the neighborhood graph of data
            points. Weights should be interpreted as connectivities.
        **distances** : sparse matrix of dtype `float32`.
            Instead of decaying weights, this stores distances for each pair of
            neighbors.

    """
    rng = np.random.default_rng(rng)
    meta_random_state = {"random_state": rng.arg} if isinstance(rng, _LegacyRng) else {}
    if method != "umap" and meta_random_state.get("random_state") != _DEFAULT_SEED:
        # `random_state` different from default (or any `rng`) was passed, but
        # only the UMAP connectivities are randomized.
        warn(
            f"Parameter `rng`/`random_state` ignored if `method={method!r}`.",
            UserWarning,
            stacklevel=2,
        )
        meta_random_state.pop("random_state", None)

    use = _resolve_obs(use)
    dests = _neighbors_out(out, key_added)
    if adata.is_view:
        adata._init_as_actual(adata.copy())

    if algorithm not in get_args(_Algorithms):
        raise ValueError(f"Invalid algorithm '{algorithm}' for KNN. Valid options are: {get_args(_Algorithms)}.")

    X = _choose_representation(adata, use_rep=use_rep, n_pcs=n_pcs, use=use)
    X_contiguous = _check_neighbors_X(X, algorithm)
    _check_metrics(algorithm, metric)

    knn_indices, knn_dist = KNN_ALGORITHMS[algorithm](
        X_contiguous,
        X_contiguous,
        k=n_neighbors,
        metric=metric,
        metric_kwds=metric_kwds,
        algorithm_kwds=algorithm_kwds,
    )
    _check_knn_complete(knn_indices, knn_dist, n_index=X_contiguous.shape[0], algorithm=algorithm)
    knn_dist = _fix_self_distances(knn_dist, metric)
    # free the GPU copy of the representation: the connectivities need the memory for large data
    del X, X_contiguous

    params = dict(
        n_neighbors=n_neighbors,
        method="rapids",
        **meta_random_state,
        metric=metric,
        **({"metric_kwds": metric_kwds} if metric_kwds else {}),
        **({"algorithm_kwds": algorithm_kwds} if algorithm_kwds else {}),
        **({"use_rep": use_rep} if use_rep is not None else _use_to_uns(use, rep=True)),
        **({"n_pcs": n_pcs} if n_pcs is not None else {}),
    )

    n_obs = adata.shape[0]
    distances = _build_sparse_distances(knn_indices, knn_dist, n_obs=n_obs)
    connectivities = _calc_connectivities(
        knn_indices,
        knn_dist,
        n_obs=n_obs,
        n_neighbors=n_neighbors,
        rng=rng,
        method=method,
        metric=metric,
    )
    connectivities = _connectivities_to_host_csr(connectivities)
    return _store_neighbors(adata, dests, NeighborsResult(connectivities, distances), params)


@_accepts_legacy_random_state(_DEFAULT_SEED)
@deprecated_arg("batch_key", Deprecation("0.19.0", "Use `batches` instead, e.g. `batches=A.obs['b']`."))
@deprecated_arg("use_rep", Deprecation("0.19.0", "Use `use` instead."))
@deprecated_arg("key_added", DEPR_KEY_ADDED)
@deprecated_copy
def bbknn(
    adata: AnnData,
    neighbors_within_batch: int = 3,
    n_pcs: int | None = None,
    *,
    batches: AdRef | str | None = None,
    use: LayerAcc | MultiAcc | str | None = None,
    rng: SeedLike | RNGLike | None = None,
    algorithm: _Algorithms_bbknn = "brute",
    metric: _Metrics = "euclidean",
    metric_kwds: Mapping[str, Any] = MappingProxyType({}),
    algorithm_kwds: Mapping[str, Any] = MappingProxyType({}),
    trim: int | None = None,
    out: GraphAcc | str | NeighborsOut | bool = False,
    # deprecated
    batch_key: str | None = None,
    use_rep: str | None = None,
    key_added: str | None = None,
    copy: bool = False,
) -> AnnData | NeighborsResult | None:
    """Batch balanced KNN :cite:p:`Polanski2019`.

    Batch balanced KNN alters the KNN procedure to identify each cell's top neighbours in
    each batch separately instead of the entire cell pool with no accounting for batch.

    The nearest neighbours for each batch are then merged to create a final list of
    neighbours for the cell.

    Parameters
    ----------
    adata
        Annotated data matrix.
    neighbors_within_batch
        How many top neighbours to report for each batch; total number of neighbours
        in the initial k-nearest-neighbours computation will be this number times
        the number of batches. This then serves as the basis for the construction
        of a symmetrical matrix of connectivities.
    n_pcs
        Use this many PCs. If `n_pcs==0` and `use is None`, use `.X`.
    batches
        The batch labels to balance, e.g. `A.obs['batch']` (or `'obs.batch'`). Required.
    use
        The representation to use, see :func:`~rapids_singlecell.pp.neighbors`.
    rng
        Random seed or :class:`~numpy.random.Generator` for reproducibility.
        The superseded `random_state` argument is still accepted.
    algorithm
        The query algorithm to use. Valid options are:

        `'brute'`
            Brute-force search that computes distances to all data points, guaranteeing exact results.
        `'ivfflat'`
            Uses inverted file indexing to partition the dataset into coarse quantizer cells and performs the search within the relevant cells.
        `'ivfpq'`
            Combines inverted file indexing with product quantization to encode sub-vectors of the dataset, facilitating faster distance computation.
        `'cagra'`
            Employs the Compressed, Accurate Graph-based search to quickly find nearest neighbors by traversing a graph structure.

        `'mg_ivfflat'`
            Uses the Multi-GPU inverted file indexing to partition the dataset into coarse quantizer cells and performs the search within the relevant cells.

        `'mg_ivfpq'`
            Combines Multi-GPU inverted file indexing with product quantization to encode sub-vectors of the dataset, facilitating faster distance computation.

        Please ensure that the chosen algorithm is compatible with your dataset and the specific requirements of your search problem.
    metric
        A known metric's name or a callable that returns a distance.
        For ``inner_product``, ``distances`` stores raw similarities; graph
        weighting uses positive score gaps, with self at distance zero.
    metric_kwds
        Options for the metric.
    algorithm_kwds
        Options for the algorithm. For `ivfflat` and `ivfpq` algorithms, the following
        parameters can be specified:

        * 'n_lists': Number of inverted lists for IVF indexing. Default is 2 * next_power_of_2(sqrt(n_samples)).
        * 'n_probes': Number of lists to probe during search. Default is 20. Higher values
          increase accuracy but reduce speed.

        For `cagra` algorithm, the following parameters can be specified:

        * 'itopk_size': Number of candidates kept during the search. Default is cuVS's default (64), raised
          to `neighbors_within_batch` if smaller. Higher values increase accuracy but reduce speed.

        For `mg_ivfflat` and `mg_ivfpq` algorithms, the following parameters can be specified:

        * 'distribution_mode': The distribution mode to use. Valid options are: 'replicated' and 'shared'. Default is 'replicated'.

        * 'n_lists': Number of inverted lists for IVF indexing. Default is 2 * next_power_of_2(sqrt(n_samples)).

        * 'n_probes': Number of lists to probe during search. Default is 20. Higher values
          increase accuracy but reduce speed.

    trim
        Trim the neighbours of each cell to these many top connectivities.
        May help with population independence and improve the tidiness of clustering.
        The lower the value the more independent the individual populations,
        at the cost of more conserved batch effect.
        If `None`, sets the parameter value automatically to
        10 times `neighbors_within_batch` times the number of batches.
        Set to 0 to skip.
    out
        Where to write the graphs, see :func:`~rapids_singlecell.pp.neighbors`.

    Returns
    -------
    A :class:`~rapids_singlecell.pp.NeighborsResult` if `out=True`, else `None`.
    See `out` for where these are written:

    connectivities : sparse matrix of dtype `float32`.
        Weighted adjacency matrix of the neighborhood graph of data
        points. Weights should be interpreted as connectivities.
    distances : sparse matrix of dtype `float32`.
        Instead of decaying weights, this stores distances for each pair of
        neighbors.
    """
    rng = np.random.default_rng(rng)
    meta_random_state = {"random_state": rng.arg} if isinstance(rng, _LegacyRng) else {}

    if batch_key is not None:
        if batches is not None:
            raise TypeError("Pass either `batches` or `batch_key`, not both.")
        if batch_key not in adata.obs:
            raise ValueError(f"Batch key '{batch_key}' not present in `adata.obs`.")
        batch_array = adata.obs[batch_key].to_numpy()
    elif batches is None:
        raise ValueError("Please provide `batches` to perform batch-balanced KNN.")
    else:
        batch_array = np.asarray(_to_numpy_1d(_get_vec(adata, _resolve(batches, vec=True), dim="obs")))
    use = _resolve_obs(use)
    dests = _neighbors_out(out, key_added)

    if algorithm not in get_args(_Algorithms_bbknn):
        raise ValueError(
            f"Invalid algorithm '{algorithm}' for batch-balanced KNN. Valid options are: {get_args(_Algorithms_bbknn)}."
        )
    if adata.is_view:
        adata._init_as_actual(adata.copy())

    X = _choose_representation(adata, use_rep=use_rep, n_pcs=n_pcs, use=use)
    X_contiguous = _check_neighbors_X(X, algorithm)
    _check_metrics(algorithm, metric)

    n_obs = adata.shape[0]
    unique_batches = np.unique(batch_array)
    total_neighbors = neighbors_within_batch * len(unique_batches)
    knn_dist = cp.zeros((X.shape[0], total_neighbors), dtype=np.float32)
    knn_indices = cp.zeros_like(knn_dist).astype(int)

    index_array = cp.arange(n_obs)

    for idx, batch in enumerate(unique_batches):
        mask_to = batch_array == batch
        X_to = X_contiguous[mask_to]
        ind_to = index_array[mask_to]

        sub_ind, sub_dist = KNN_ALGORITHMS[algorithm](
            X_to,
            X_contiguous,
            k=neighbors_within_batch,
            metric=metric,
            metric_kwds=metric_kwds,
            algorithm_kwds=algorithm_kwds,
        )
        _check_knn_complete(sub_ind, sub_dist, n_index=X_to.shape[0], algorithm=algorithm)

        col_range = cp.arange(idx * neighbors_within_batch, (idx + 1) * neighbors_within_batch)
        knn_indices[:, col_range] = ind_to[sub_ind]
        knn_dist[:, col_range] = sub_dist

    # Sort each row so neighbors are ordered closest-first across all batches.
    # fuzzy_simplicial_set uses the first non-zero distance per row as the
    # local-connectivity rho; unsorted input collapses sigma and weights.
    # ``inner_product`` stores similarities, so larger is closer.
    order = cp.argsort(-knn_dist if metric == "inner_product" else knn_dist, axis=1)
    row_idx = cp.arange(n_obs)[:, None]
    knn_dist = knn_dist[row_idx, order]
    knn_indices = knn_indices[row_idx, order]
    knn_dist = _fix_self_distances(knn_dist, metric)

    if trim is None:
        trim = 10 * total_neighbors

    params = dict(
        n_neighbors=total_neighbors,
        method="rapids",
        **meta_random_state,
        metric=metric,
        trim=trim,
        **({"metric_kwds": metric_kwds} if metric_kwds else {}),
        **({"algorithm_kwds": algorithm_kwds} if algorithm_kwds else {}),
        **({"use_rep": use_rep} if use_rep is not None else _use_to_uns(use, rep=True)),
        **({"n_pcs": n_pcs} if n_pcs is not None else {}),
    )

    distances = _build_sparse_distances(knn_indices, knn_dist, n_obs=n_obs)
    connectivities = _calc_connectivities(
        knn_indices,
        knn_dist,
        n_obs=n_obs,
        n_neighbors=total_neighbors,
        rng=rng,
        metric=metric,
        batch_codes=cp.asarray(np.searchsorted(unique_batches, batch_array)) if metric == "inner_product" else None,
    )
    if connectivities.nnz >= np.iinfo(np.int32).max:
        connectivities = connectivities.get().tocsr()
    else:
        connectivities = connectivities.tocsr()
        if trim > 0:
            connectivities = _trimming(connectivities, trim)
        connectivities = connectivities.get()
    return _store_neighbors(adata, dests, NeighborsResult(connectivities, distances), params)
