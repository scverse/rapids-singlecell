from __future__ import annotations

import warnings
from typing import TYPE_CHECKING

import cudf
import cupy as cp
import numpy as np
import pandas as pd
from natsort import natsorted
from scanpy.tools._utils import _choose_graph
from scanpy.tools._utils_clustering import rename_groups, restrict_adjacency

from ._utils import _choose_representation

if TYPE_CHECKING:
    from collections.abc import Sequence

    from anndata import AnnData
    from scipy import sparse


def _check_dtype(dtype: str | np.dtype | cp.dtype) -> str | np.dtype | cp.dtype:
    if isinstance(dtype, str):
        if dtype not in ["float32", "float64"]:
            raise ValueError("dtype must be one of ['float32', 'float64']")
        else:
            return dtype
    elif dtype is np.float32 or dtype is np.float64:
        return dtype
    else:
        raise ValueError("dtype must be one of ['float32', 'float64']")


def _create_graph(adjacency, dtype=np.float64, *, use_weights=True):
    from cugraph import Graph

    sources, targets = adjacency.nonzero()
    weights = adjacency[sources, targets]
    vertices = list(range(adjacency.shape[0]))
    if isinstance(weights, np.matrix):
        weights = weights.A1
    df = cudf.DataFrame({"source": sources, "destination": targets, "weights": weights})
    df.weights = df.weights.astype(dtype)
    g = Graph()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        if use_weights:
            g.from_cudf_edgelist(
                df,
                source="source",
                destination="destination",
                weight="weights",
                vertices=vertices,
            )
        else:
            g.from_cudf_edgelist(
                df,
                source="source",
                destination="destination",
                vertices=vertices,
            )
    return g


def _create_graph_dask(adjacency, dtype=np.float64, *, use_weights=True):
    import cudf
    import dask.dataframe as dd
    from cugraph import Graph

    rows = np.repeat(np.arange(adjacency.shape[0]), np.diff(adjacency.indptr)).astype(
        np.int32
    )
    cols = adjacency.indices
    weights = adjacency.data

    n_devices = cp.cuda.runtime.getDeviceCount()
    chunksize = int((adjacency.nnz + n_devices - 1) / n_devices)

    boundaries = list(range(0, adjacency.nnz, chunksize))
    pairs = [(start, min(start + chunksize, adjacency.nnz)) for start in boundaries]

    def mapper(pair):
        start, end = pair
        return cudf.DataFrame(
            {
                "src": rows[start:end].astype(np.int64),
                "dst": cols[start:end].astype(np.int64),
                "weight": weights[start:end].astype(dtype),
            }
        )

    # meta must match the actual columns
    meta = {
        "src": np.int64,
        "dst": np.int64,
        "weight": dtype,
    }

    ddf = dd.from_map(mapper, pairs, meta=meta).to_backend("cudf").persist()
    import cugraph.dask.comms.comms as Comms

    Comms.initialize(p2p=True)
    g = Graph()
    if use_weights:
        g.from_dask_cudf_edgelist(
            ddf,
            source="src",
            destination="dst",
            weight="weight",
        )
    else:
        g.from_dask_cudf_edgelist(
            ddf,
            source="src",
            destination="dst",
        )
    return g


def leiden(
    adata: AnnData,
    resolution: float | list[float] = 1.0,
    *,
    random_state: int | None = 0,
    theta: float = 1.0,
    restrict_to: tuple[str, Sequence[str]] | None = None,
    key_added: str = "leiden",
    adjacency: sparse.spmatrix | None = None,
    n_iterations: int = 100,
    use_weights: bool = True,
    neighbors_key: str | None = None,
    obsp: str | None = None,
    dtype: str | np.dtype | cp.dtype = np.float32,
    use_dask: bool = False,
    copy: bool = False,
) -> AnnData | None:
    """
    Cluster cells into subgroups using the Leiden algorithm :cite:p:`Traag2019`.

    Performs Leiden clustering using cuGraph, an improved version of the
    Louvain algorithm :cite:p:`Blondel2008`.

    Parameters
    ----------
        adata :
            annData object

        resolution
            A parameter value or a list of parameter values controlling the coarseness of the clustering.
            (called gamma in the modularity formula). Higher values lead to
            more clusters. If a list of values is provided, the Leiden algorithm will be run for each value in the list.

        random_state
            Change the initialization of the optimization. Defaults to 0.

        theta
            Called theta in the Leiden algorithm, this is used to scale modularity
            gain in Leiden refinement phase, to compute the probability of joining
            a random leiden community.

        restrict_to
            Restrict the clustering to the categories within the key for
            sample annotation, tuple needs to contain
            `(obs_key, list_of_categories)`.

        key_added
            `adata.obs` key under which to add the cluster labels.

        adjacency
            Sparse adjacency matrix of the graph, defaults to neighbors
            connectivities.

        n_iterations
            This controls the maximum number of levels/iterations of the
            Leiden algorithm. When specified, the algorithm will terminate
            after no more than the specified number of iterations. No error
            occurs when the algorithm terminates early in this manner.

        use_weights
            If `True`, edge weights from the graph are used in the
            computation (placing more emphasis on stronger edges).

        neighbors_key
            If not specified, `leiden` looks at `.obsp['connectivities']`
            for neighbors connectivities. If specified, `leiden` looks at
            `.obsp[.uns[neighbors_key]['connectivities_key']]` for neighbors
            connectivities.

        obsp
            Use .obsp[obsp] as adjacency. You can't specify both
            `obsp` and `neighbors_key` at the same time.

        dtype
            Data type to use for the adjacency matrix.

        use_dask
            If `True`, use Dask to create the graph and cluster. This will use all GPUs available. This feature is experimental. For datasets with less than 10 Million cells, it is recommended to use `use_dask=False`.

        copy
            Whether to copy `adata` or modify it in place.
    """
    # Adjacency graph

    adata = adata.copy() if copy else adata

    dtype = _check_dtype(dtype)

    if adjacency is None:
        adjacency = _choose_graph(adata, obsp, neighbors_key)
    if restrict_to is not None:
        restrict_key, restrict_categories = restrict_to
        adjacency, restrict_indices = restrict_adjacency(
            adata=adata,
            restrict_key=restrict_key,
            restrict_categories=restrict_categories,
            adjacency=adjacency,
        )
    if use_dask:
        from cugraph.dask import leiden as culeiden

        g = _create_graph_dask(adjacency, dtype, use_weights=use_weights)
    else:
        from cugraph import leiden as culeiden

        g = _create_graph(adjacency, dtype, use_weights=use_weights)
    # Cluster
    if isinstance(resolution, float | int):
        resolutions = [resolution]
    else:
        resolutions = resolution
    modularities = []
    for resolution in resolutions:
        leiden_parts, modularity = culeiden(
            g,
            resolution=resolution,
            random_state=random_state,
            theta=theta,
            max_iter=n_iterations,
        )
        if use_dask:
            leiden_parts = leiden_parts.to_backend("pandas").compute()
        else:
            leiden_parts = leiden_parts.to_pandas()
        modularities.append(modularity)

        # Format output
        groups = leiden_parts.sort_values("vertex")[["partition"]].to_numpy().ravel()
        key_added_to_use = key_added
        if restrict_to is not None:
            if key_added == "leiden":
                key_added_to_use += "_R"
            groups = rename_groups(
                adata,
                key_added=key_added_to_use,
                restrict_key=restrict_key,
                restrict_categories=restrict_categories,
                restrict_indices=restrict_indices,
                groups=groups,
            )
        if len(resolutions) > 1:
            key_added_to_use += f"_{resolution}"

        adata.obs[key_added_to_use] = pd.Categorical(
            values=groups.astype("U"),
            categories=natsorted(map(str, np.unique(groups))),
        )
    if use_dask:
        import cugraph.dask.comms.comms as Comms

        Comms.destroy()
    # store information on the clustering parameters
    adata.uns[key_added] = {}
    adata.uns[key_added]["params"] = {
        "resolution": resolutions if len(resolutions) > 1 else resolutions[0],
        "random_state": random_state,
        "n_iterations": n_iterations,
    }
    adata.uns[key_added]["modularity"] = (
        modularities if len(modularities) > 1 else modularities[0]
    )
    return adata if copy else None


def louvain(
    adata: AnnData,
    resolution: float | list[float] = 1.0,
    *,
    restrict_to: tuple[str, Sequence[str]] | None = None,
    key_added: str = "louvain",
    adjacency: sparse.spmatrix | None = None,
    n_iterations: int = 100,
    threshold: float = 1e-7,
    use_weights: bool = True,
    neighbors_key: int | None = None,
    obsp: str | None = None,
    dtype: str | np.dtype | cp.dtype = np.float32,
    use_dask: bool = False,
    copy: bool = False,
) -> AnnData | None:
    """
    Cluster cells into subgroups using the Louvain algorithm :cite:p:`Blondel2008`.

    Parameters
    ----------
        adata :
            annData object

        resolution
            A parameter value or a list of parameter values controlling the coarseness of the clustering.
            (called gamma in the modularity formula). Higher values lead to
            more clusters. If a list of values is provided, the Leiden algorithm will be run for each value in the list.

        restrict_to
            Restrict the clustering to the categories within the key for
            sample annotation, tuple needs to contain
            `(obs_key, list_of_categories)`.

        key_added
            `adata.obs` key under which to add the cluster labels.

        adjacency
            Sparse adjacency matrix of the graph, defaults to neighbors
            connectivities.

        n_iterations
            This controls the maximum number of levels/iterations of the
            Louvain algorithm. When specified the algorithm will terminate
            after no more than the specified number of iterations. No error
            occurs when the algorithm terminates early in this manner.
            Capped at 500 to prevent excessive runtime.

        threshold
            Modularity gain threshold for each level/iteration. If the gain
            of modularity between two levels of the algorithm is less than
            the given threshold then the algorithm stops and returns the
            resulting communities. Defaults to 1e-7.

        use_weights
            If `True`, edge weights from the graph are used in the
            computation (placing more emphasis on stronger edges).

        neighbors_key
            If not specified, `louvain` looks at `.obsp['connectivities']`
            for neighbors connectivities. If specified, `louvain` looks at
            `.obsp[.uns[neighbors_key]['connectivities_key']]` for neighbors
            connectivities.

        obsp
            Use `.obsp[obsp]` as adjacency. You can't specify both `obsp`
            and `neighbors_key` at the same time.

        dtype
            Data type to use for the adjacency matrix.

        use_dask
            If `True`, use Dask to create the graph and cluster. This will use all GPUs available. This feature is experimental. For datasets with less than 10 Million cells, it is recommended to use `use_dask=False`.

        copy
            Whether to copy `adata` or modify it in place.

    """
    # Adjacency graph
    dtype = _check_dtype(dtype)

    adata = adata.copy() if copy else adata
    if adjacency is None:
        adjacency = _choose_graph(adata, obsp, neighbors_key)
    if restrict_to is not None:
        restrict_key, restrict_categories = restrict_to
        adjacency, restrict_indices = restrict_adjacency(
            adata,
            restrict_key,
            restrict_categories=restrict_categories,
            adjacency=adjacency,
        )
    # Cluster
    if use_dask:
        from cugraph.dask import louvain as culouvain

        g = _create_graph_dask(adjacency, dtype, use_weights=use_weights)
    else:
        from cugraph import louvain as culouvain

        g = _create_graph(adjacency, dtype, use_weights=use_weights)

    if isinstance(resolution, float | int):
        resolutions = [resolution]
    else:
        resolutions = resolution
    modularities = []
    for resolution in resolutions:
        louvain_parts, modularity = culouvain(
            g,
            resolution=resolution,
            max_level=n_iterations,
            threshold=threshold,
        )
        if use_dask:
            louvain_parts = louvain_parts.to_backend("pandas").compute()
        else:
            louvain_parts = louvain_parts.to_pandas()
        modularities.append(modularity)

        # Format output
        groups = louvain_parts.sort_values("vertex")[["partition"]].to_numpy().ravel()
        key_added_to_use = key_added
        if restrict_to is not None:
            if key_added == "louvain":
                key_added_to_use += "_R"
            groups = rename_groups(
                adata,
                key_added=key_added_to_use,
                restrict_key=restrict_key,
                restrict_categories=restrict_categories,
                restrict_indices=restrict_indices,
                groups=groups,
            )
        if len(resolutions) > 1:
            key_added_to_use += f"_{resolution}"

        adata.obs[key_added_to_use] = pd.Categorical(
            values=groups.astype("U"),
            categories=natsorted(map(str, np.unique(groups))),
        )
    if use_dask:
        import cugraph.dask.comms.comms as Comms

        Comms.destroy()
    adata.uns[key_added] = {}
    adata.uns[key_added]["params"] = {
        "resolution": resolutions if len(resolutions) > 1 else resolutions[0],
        "n_iterations": n_iterations,
        "threshold": threshold,
    }
    adata.uns[key_added]["modularity"] = (
        modularities if len(modularities) > 1 else modularities[0]
    )
    return adata if copy else None


def kmeans(
    adata: AnnData,
    n_clusters: int = 8,
    n_pcs: int = 50,
    *,
    use_rep: str = "X_pca",
    n_init: int = 1,
    random_state: float = 42,
    key_added: str = "kmeans",
    copy: bool = False,
    **kwargs,
) -> None:
    """
    KMeans is a basic but powerful clustering method which is optimized via Expectation Maximization. It randomly selects K data points in X, and computes which samples are close to these points. For every cluster of points, a mean is computed (hence the name), and this becomes the new centroid.

    Parameters
    ----------
        adata
            Annotated data matrix.
        n_clusters
            Number of clusters to compute
        n_pcs
            Use this many PCs. If `n_pcs==0` use `.X` if `use_rep is None`.
        use_rep
            Use the indicated representation. `'X'` or any key for `.obsm` is valid.
            If None, the representation is chosen automatically: For .n_vars < 50, .X
            is used, otherwise `'X_pca'` is used. If `'X_pca'` is not present, it's
            computed with default parameters or `n_pcs` if present.
        n_init
            Number of initializations to run the KMeans algorithm
        random_state
            if you want results to be the same when you restart Python, select a
            state. Default is 42.
        key_added
            `adata.obs` key under which to add the cluster labels.
        copy
            Whether to copy `adata` or modify it in place.
        **kwargs
            Additional keyword arguments for KMeans.

    """
    from cuml.cluster import KMeans

    adata = adata.copy() if copy else adata
    X = _choose_representation(adata, use_rep=use_rep, n_pcs=n_pcs)

    kmeans_out = KMeans(
        n_clusters=n_clusters, n_init=n_init, random_state=random_state, **kwargs
    ).fit(X)
    groups = kmeans_out.labels_

    adata.obs[key_added] = pd.Categorical(
        values=groups.astype("U"),
        categories=natsorted(map(str, np.unique(groups))),
    )

    return adata if copy else None
