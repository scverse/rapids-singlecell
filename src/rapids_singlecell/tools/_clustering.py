from __future__ import annotations

import contextlib
import warnings
from typing import TYPE_CHECKING

import cudf
import cupy as cp
import numpy as np
import pandas as pd
from natsort import natsorted
from scanpy.tools._utils import _choose_graph
from scanpy.tools._utils_clustering import rename_groups, restrict_adjacency

from rapids_singlecell._keys import _resolve_obsm_key
from rapids_singlecell._utils._random import (
    RNGLike,
    SeedLike,
    _accepts_legacy_random_state,
    _LegacyRng,
    _seed_from_rng,
)

from ._utils import _choose_representation, _is_canonical_csr, _is_symmetric

if TYPE_CHECKING:
    from collections.abc import Sequence

    from anndata import AnnData
    from scipy import sparse


def _check_dtype(dtype: str | np.dtype) -> str | np.dtype:
    if isinstance(dtype, str):
        if dtype not in ["float32", "float64"]:
            raise ValueError("dtype must be one of ['float32', 'float64']")
        else:
            return dtype
    elif dtype is np.float32 or dtype is np.float64:
        return dtype
    else:
        raise ValueError("dtype must be one of ['float32', 'float64']")


# cuDF columns hold at most this many rows
_CUDF_MAX_ROWS = np.iinfo(np.int32).max


def _upper_suffices(adjacency) -> bool:
    """Whether cuGraph needs only the upper triangle of `adjacency`.

    The graph is undirected, so cuGraph adds the reverse of every edge: for a symmetric
    adjacency, the upper triangle gives the same graph (and clusters) in half the memory and time.
    """
    return (
        not isinstance(adjacency.indptr, cp.ndarray)
        and _is_canonical_csr(adjacency)
        and _is_symmetric(adjacency)
    )


def _create_graph(adjacency, dtype=np.float64, *, use_weights=True):
    from cugraph import Graph

    adjacency = adjacency.tocsr()
    xp = cp if isinstance(adjacency.indptr, cp.ndarray) else np
    n = adjacency.shape[0]
    sources = xp.repeat(xp.arange(n, dtype=np.int64), xp.diff(adjacency.indptr))
    keep = adjacency.data != 0  # the stored non-zeros, like `adjacency.nonzero()`
    # cuGraph's edge offsets exceed int32 for this many edges; int64 vertex ids switch them over
    idx_dtype = np.int64 if int(keep.sum()) > _CUDF_MAX_ROWS else np.int32
    if _upper_suffices(adjacency):
        keep &= adjacency.indices >= sources
    df = cudf.DataFrame(
        {
            "source": sources[keep].astype(idx_dtype),
            "destination": adjacency.indices[keep].astype(idx_dtype),
            "weights": adjacency.data[keep],
        }
    )
    del sources, keep
    vertices = cudf.Series(cp.arange(n, dtype=idx_dtype))
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


def _edge_frame(arrays, dtype) -> cudf.DataFrame:
    src, dst, weight = arrays
    return cudf.DataFrame(
        {
            "src": src.astype(np.int64),
            "dst": dst.astype(np.int64),
            "weight": weight.astype(dtype),
        }
    )


def _create_graph_dask(adjacency, dtype=np.float64, *, use_weights=True):
    import dask.dataframe as dd
    from cugraph import Graph
    from distributed import default_client, wait

    client = default_client()
    workers = list(client.nthreads())
    adjacency = adjacency.tocsr()
    rows = np.repeat(
        np.arange(adjacency.shape[0], dtype=np.int32), np.diff(adjacency.indptr)
    )
    keep = adjacency.data != 0
    if _upper_suffices(adjacency):
        keep &= adjacency.indices >= rows
    rows, cols, weights = rows[keep], adjacency.indices[keep], adjacency.data[keep]
    del keep
    # one partition per worker, sent straight to it (not through the scheduler)
    bounds = np.linspace(0, len(rows), len(workers) + 1).astype(np.int64)
    parts = []
    for worker, start, stop in zip(workers, bounds[:-1], bounds[1:], strict=True):
        [arrays] = client.scatter(
            [(rows[start:stop], cols[start:stop], weights[start:stop])],
            workers=[worker],
            direct=True,
        )
        parts.append(
            client.submit(_edge_frame, arrays, dtype, workers=[worker], pure=False)
        )
    wait(parts)
    del rows, cols, weights
    meta = cudf.DataFrame(
        {
            "src": np.empty(0, np.int64),
            "dst": np.empty(0, np.int64),
            "weight": np.empty(0, dtype),
        }
    )
    ddf = dd.from_delayed(parts, meta=meta, verify_meta=False)
    import cugraph.dask.comms.comms as Comms

    try:
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
    except BaseException:
        # A worker dying mid-handshake leaves cugraph's global communicator
        # half-initialized; tear it down (best effort) so later calls don't
        # fail with "already initialized".
        with contextlib.suppress(Exception):
            Comms.destroy()
        raise
    return g


@_accepts_legacy_random_state(0)
def leiden(
    adata: AnnData,
    resolution: float | list[float] = 1.0,
    *,
    rng: SeedLike | RNGLike | None = None,
    theta: float = 1.0,
    restrict_to: tuple[str, Sequence[str]] | None = None,
    key_added: str = "leiden",
    adjacency: sparse.spmatrix | None = None,
    n_iterations: int = 100,
    use_weights: bool = True,
    neighbors_key: str | None = None,
    obsp: str | None = None,
    dtype: str | np.dtype = np.float32,
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

        rng
            Random seed or :class:`~numpy.random.Generator` changing the
            initialization of the optimization. Defaults to 0.
            The superseded `random_state` argument is still accepted.

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

    rng = np.random.default_rng(rng)
    meta_random_state = {"random_state": rng.arg} if isinstance(rng, _LegacyRng) else {}

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
    try:
        for resolution in resolutions:
            leiden_parts, modularity = culeiden(
                g,
                resolution=resolution,
                # cuGraph's leiden is seeded, so draw the seed right here
                random_state=_seed_from_rng(rng),
                theta=theta,
                max_iter=n_iterations,
            )
            if use_dask:
                leiden_parts = leiden_parts.to_backend("pandas").compute()
            else:
                leiden_parts = leiden_parts.to_pandas()
            modularities.append(modularity)

            # Format output
            groups = (
                leiden_parts.sort_values("vertex")[["partition"]].to_numpy().ravel()
            )
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
    finally:
        # Always tear down the raft/NCCL comms, even if clustering raised, so a
        # single failure can't leak the global cugraph communicator and make
        # every later dask clustering call fail with "already initialized".
        if use_dask:
            import cugraph.dask.comms.comms as Comms

            Comms.destroy()
    # store information on the clustering parameters
    adata.uns[key_added] = {}
    adata.uns[key_added]["params"] = {
        "resolution": resolutions if len(resolutions) > 1 else resolutions[0],
        **meta_random_state,
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
    dtype: str | np.dtype = np.float32,
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
    try:
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
            groups = (
                louvain_parts.sort_values("vertex")[["partition"]].to_numpy().ravel()
            )
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
    finally:
        # Always tear down the raft/NCCL comms, even if clustering raised, so a
        # single failure can't leak the global cugraph communicator and make
        # every later dask clustering call fail with "already initialized".
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


@_accepts_legacy_random_state(42)
def kmeans(
    adata: AnnData,
    n_clusters: int = 8,
    n_pcs: int = 50,
    *,
    use_rep: str = "X_pca",
    n_init: int = 1,
    rng: SeedLike | RNGLike | None = None,
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
            Either spelling of the PCA key resolves to whichever one the data
            actually uses, so the default works under any
            ``rapids_singlecell.settings.preset``.
        n_init
            Number of initializations to run the KMeans algorithm
        rng
            Random seed or :class:`~numpy.random.Generator`; fix it if you want
            results to be the same when you restart Python. Default is 42.
            The superseded `random_state` argument is still accepted.
        key_added
            `adata.obs` key under which to add the cluster labels.
        copy
            Whether to copy `adata` or modify it in place.
        **kwargs
            Additional keyword arguments for KMeans.

    """
    rng = np.random.default_rng(rng)

    from cuml.cluster import KMeans

    adata = adata.copy() if copy else adata
    use_rep = _resolve_obsm_key(adata, use_rep)
    X = _choose_representation(adata, use_rep=use_rep, n_pcs=n_pcs)

    kmeans_out = KMeans(
        # cuML's KMeans is seeded, so draw the seed right here
        n_clusters=n_clusters,
        n_init=n_init,
        random_state=_seed_from_rng(rng),
        **kwargs,
    ).fit(X)
    groups = kmeans_out.labels_

    adata.obs[key_added] = pd.Categorical(
        values=groups.astype("U"),
        categories=natsorted(map(str, np.unique(groups))),
    )

    return adata if copy else None
