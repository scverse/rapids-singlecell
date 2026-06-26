from __future__ import annotations

from types import MappingProxyType
from typing import TYPE_CHECKING, Any, Literal, get_args

import cupy as cp
import numpy as np

from rapids_singlecell.preprocessing._neighbors._helper import (
    _check_metrics,
    _check_neighbors_X,
    _fix_self_distances,
    _trimming,
)
from rapids_singlecell.preprocessing._neighbors._neighbors import (
    KNN_ALGORITHMS,
    AnyRandom,
    _Algorithms,
    _build_sparse_distances,
    _calc_connectivities,
    _Metrics,
)
from rapids_singlecell.tools._utils import _choose_representation

if TYPE_CHECKING:
    from collections.abc import Mapping

    from anndata import AnnData

_Algorithms_bbknn = Literal[
    "brute", "cagra", "ivfflat", "ivfpq", "mg_ivfflat", "mg_ivfpq"
]


def neighbors(
    adata: AnnData,
    n_neighbors: int = 15,
    n_pcs: int | None = None,
    *,
    use_rep: str | None = None,
    random_state: AnyRandom = 0,
    algorithm: _Algorithms = "brute",
    metric: _Metrics = "euclidean",
    metric_kwds: Mapping[str, Any] = MappingProxyType({}),
    algorithm_kwds: Mapping[str, Any] = MappingProxyType({}),
    method: Literal["umap", "gauss", "jaccard"] = "umap",
    key_added: str | None = None,
    copy: bool = False,
) -> AnnData | None:
    """\
    Compute a neighborhood graph of observations :cite:p:`Ootomo2024`.

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
        Use this many PCs. If `n_pcs==0` use `.X` if `use_rep is None`.
    use_rep
        Use the indicated representation. `'X'` or any key for `.obsm` is valid.
        If None, the representation is chosen automatically: For .n_vars < 50, .X
        is used, otherwise `'X_pca'` is used. If `'X_pca'` is not present, it's
        computed with default parameters or `n_pcs` if present.
    random_state
        A numpy random seed.
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

        * 'intermediate_graph_degree': The degree of the intermediate graph. Default is None.
          It is recommended to set it to `>= 1.5 * n_neighbors`.

        For `all_neighbors` algorithm, the following parameters can be specified:

        * 'algo': The algorithm to use. Valid options are: 'ivf_pq' and 'nn_descent'. Default is 'nn_descent'.

        * 'n_clusters': Number of clusters/batches to partition the dataset into (> overlap_factor). Default is number of GPUs.

        * 'overlap_factor': Number of clusters each point is assigned to (must be < n_clusters). Default is 1.

        * 'n_lists': Number of inverted lists for IVF indexing. Default is 2 * next_power_of_2(sqrt(n_samples)). Only available for `ivf_pq` algorithm.

        * 'intermediate_graph_degree': The degree of the intermediate graph. Default is None. It is recommended to set it to `>= 1.5 * n_neighbors`. Only available for `nn_descent` algorithm.

        For `mg_ivfflat` and `mg_ivfpq` algorithms, the following parameters can be specified:

        * 'distribution_mode': The distribution mode to use. Valid options are: 'replicated' and 'shared'. Default is 'replicated'.

        * 'n_lists': Number of inverted lists for IVF indexing. Default is 2 * next_power_of_2(sqrt(n_samples)).

        * 'n_probes': Number of lists to probe during search. Default is 20. Higher values
          increase accuracy but reduce speed.

    key_added
        If not specified, the neighbors data is stored in .uns['neighbors'],
        distances and connectivities are stored in .obsp['distances'] and
        .obsp['connectivities'] respectively.
        If specified, the neighbors data is added to .uns[key_added],
        distances are stored in .obsp[key_added+'_distances'] and
        connectivities in .obsp[key_added+'_connectivities'].
    copy
        Return a copy instead of writing to adata.

    Returns
    -------
    Depending on `copy`, updates or returns `adata` with the following:

    See `key_added` parameter description for the storage path of
    connectivities and distances.

        **connectivities** : sparse matrix of dtype `float32`.
            Weighted adjacency matrix of the neighborhood graph of data
            points. Weights should be interpreted as connectivities.
        **distances** : sparse matrix of dtype `float32`.
            Instead of decaying weights, this stores distances for each pair of
            neighbors.

    """
    adata = adata.copy() if copy else adata

    if adata.is_view:
        adata._init_as_actual(adata.copy())

    if algorithm not in get_args(_Algorithms):
        raise ValueError(
            f"Invalid algorithm '{algorithm}' for KNN. "
            f"Valid options are: {get_args(_Algorithms)}."
        )

    X = _choose_representation(adata, use_rep=use_rep, n_pcs=n_pcs)
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
    knn_dist = _fix_self_distances(knn_dist, metric)

    params = dict(
        n_neighbors=n_neighbors,
        method="rapids",
        random_state=random_state,
        metric=metric,
        **({"metric_kwds": metric_kwds} if metric_kwds else {}),
        **({"algorithm_kwds": algorithm_kwds} if algorithm_kwds else {}),
        **({"use_rep": use_rep} if use_rep is not None else {}),
        **({"n_pcs": n_pcs} if n_pcs is not None else {}),
    )

    n_obs = adata.shape[0]
    distances = _build_sparse_distances(knn_indices, knn_dist, n_obs=n_obs)
    connectivities = _calc_connectivities(
        knn_indices,
        knn_dist,
        n_obs=n_obs,
        n_neighbors=n_neighbors,
        random_state=random_state,
        metric=metric,
        method=method,
    )
    if connectivities.nnz >= np.iinfo(np.int32).max:
        connectivities = connectivities.get().tocsr()
    else:
        connectivities = connectivities.tocsr().get()

    # Store results
    neighbors_key = key_added if key_added is not None else "neighbors"
    if neighbors_key == "neighbors":
        conns_key = "connectivities"
        dists_key = "distances"
    else:
        conns_key = neighbors_key + "_connectivities"
        dists_key = neighbors_key + "_distances"

    adata.uns[neighbors_key] = {
        "connectivities_key": conns_key,
        "distances_key": dists_key,
        "params": params,
    }
    adata.obsp[dists_key] = distances
    adata.obsp[conns_key] = connectivities

    return adata if copy else None


def bbknn(
    adata: AnnData,
    neighbors_within_batch: int = 3,
    n_pcs: int | None = None,
    *,
    batch_key: str | None = None,
    use_rep: str | None = None,
    random_state: AnyRandom = 0,
    algorithm: _Algorithms_bbknn = "brute",
    metric: _Metrics = "euclidean",
    metric_kwds: Mapping[str, Any] = MappingProxyType({}),
    algorithm_kwds: Mapping[str, Any] = MappingProxyType({}),
    trim: int | None = None,
    key_added: str | None = None,
    copy: bool = False,
) -> AnnData | None:
    """\
    Batch balanced KNN :cite:p:`Polanski2019`, altering the KNN procedure to identify each cell's top neighbours in
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
        Use this many PCs. If `n_pcs==0` and `use_rep is None`, use `.X`.
    use_rep
        Use the indicated representation. `'X'` or any key for `.obsm` is valid.
        If `None`, the representation is chosen automatically: For `.n_vars < 50`, `.X`
        is used, otherwise `'X_pca'` is used. If `'X_pca'` is not present, it's
        computed with default parameters or `n_pcs` if present.
    random_state
        A numpy random seed.
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
    metric_kwds
        Options for the metric.
    algorithm_kwds
        Options for the algorithm. For `ivfflat` and `ivfpq` algorithms, the following
        parameters can be specified:

        * 'n_lists': Number of inverted lists for IVF indexing. Default is 2 * next_power_of_2(sqrt(n_samples)).
        * 'nprobes': Number of lists to probe during search. Default is 1. Higher values
          increase accuracy but reduce speed.

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
    key_added
        If not specified, the neighbors data is stored in `.uns['neighbors']`,
        distances and connectivities are stored in `.obsp['distances']` and
        `.obsp['connectivities']` respectively.
        If specified, the neighbors data is added to `.uns[key_added]`,
        distances are stored in `.obsp[f'{key_added}_distances']` and
        connectivities in `.obsp[f'{key_added}_connectivities']`.
    copy
        Return a copy instead of writing to adata.

    Returns
    -------
    Depending on `copy`, updates or returns `adata` with the following:

    connectivities : sparse matrix of dtype `float32`.
        Weighted adjacency matrix of the neighborhood graph of data
        points. Weights should be interpreted as connectivities.
    distances : sparse matrix of dtype `float32`.
        Instead of decaying weights, this stores distances for each pair of
        neighbors.

    See `key_added` parameter description for the storage path of
    connectivities and distances.
    """

    if batch_key is None:
        raise ValueError("Please provide a batch key to perform batch-balanced KNN.")

    if batch_key not in adata.obs:
        raise ValueError(f"Batch key '{batch_key}' not present in `adata.obs`.")

    if algorithm not in get_args(_Algorithms_bbknn):
        raise ValueError(
            f"Invalid algorithm '{algorithm}' for batch-balanced KNN. "
            f"Valid options are: {get_args(_Algorithms_bbknn)}."
        )
    adata = adata.copy() if copy else adata
    if adata.is_view:
        adata._init_as_actual(adata.copy())

    X = _choose_representation(adata, use_rep=use_rep, n_pcs=n_pcs)
    X_contiguous = _check_neighbors_X(X, algorithm)
    _check_metrics(algorithm, metric)

    n_obs = adata.shape[0]
    batch_array = adata.obs[batch_key].values
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

        col_range = cp.arange(
            idx * neighbors_within_batch, (idx + 1) * neighbors_within_batch
        )
        knn_indices[:, col_range] = ind_to[sub_ind]
        knn_dist[:, col_range] = sub_dist

    # Sort each row so neighbors are ordered closest-first across all batches.
    # fuzzy_simplicial_set uses the first non-zero distance per row as the
    # local-connectivity rho; unsorted input collapses sigma and weights.
    order = cp.argsort(knn_dist, axis=1)
    row_idx = cp.arange(n_obs)[:, None]
    knn_dist = knn_dist[row_idx, order]
    knn_indices = knn_indices[row_idx, order]

    if trim is None:
        trim = 10 * total_neighbors

    params = dict(
        n_neighbors=total_neighbors,
        method="rapids",
        random_state=random_state,
        metric=metric,
        trim=trim,
        **({"metric_kwds": metric_kwds} if metric_kwds else {}),
        **({"algorithm_kwds": algorithm_kwds} if algorithm_kwds else {}),
        **({"use_rep": use_rep} if use_rep is not None else {}),
        **({"n_pcs": n_pcs} if n_pcs is not None else {}),
    )

    distances = _build_sparse_distances(knn_indices, knn_dist, n_obs=n_obs)
    connectivities = _calc_connectivities(
        knn_indices,
        knn_dist,
        n_obs=n_obs,
        n_neighbors=total_neighbors,
        random_state=random_state,
        metric=metric,
    )
    if connectivities.nnz >= np.iinfo(np.int32).max:
        connectivities = connectivities.get().tocsr()
    else:
        connectivities = connectivities.tocsr()
        if trim > 0:
            connectivities = _trimming(connectivities, trim)
        connectivities = connectivities.get()

    # Store results
    neighbors_key = key_added if key_added is not None else "neighbors"
    if neighbors_key == "neighbors":
        conns_key = "connectivities"
        dists_key = "distances"
    else:
        conns_key = neighbors_key + "_connectivities"
        dists_key = neighbors_key + "_distances"

    adata.uns[neighbors_key] = {
        "connectivities_key": conns_key,
        "distances_key": dists_key,
        "params": params,
    }
    adata.obsp[dists_key] = distances
    adata.obsp[conns_key] = connectivities

    return adata if copy else None
