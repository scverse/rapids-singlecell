from __future__ import annotations

import contextlib
import warnings
from typing import TYPE_CHECKING

import cudf
import cupy as cp
import numpy as np
import pandas as pd
from natsort import natsorted
from scanpy.tools._utils_clustering import restrict_adjacency
from scverse_misc import Deprecation, deprecated, deprecated_arg

from rapids_singlecell._keys import _resolve_obsm_key
from rapids_singlecell._utils._random import (
    RNGLike,
    SeedLike,
    _accepts_legacy_random_state,
    _LegacyRng,
    _seed_from_rng,
)
from rapids_singlecell.get._out import (
    DEPR_KEY_ADDED,
    DEPR_NEIGHBORS_KEY,
    DEPR_OBSP,
    _out_key,
    _resolve_obs,
    deprecated_copy,
)

from ._utils import _choose_representation, _use_graph

if TYPE_CHECKING:
    from collections.abc import Sequence

    from anndata import AnnData
    from scipy import sparse

    from rapids_singlecell.get._utils import AdRef, GraphAcc, LayerAcc, MultiAcc

_DEPR_ADJACENCY = Deprecation("0.19.0", "Store the graph in `.obsp` and pass `use=A.obsp[...]` instead.")


def _cluster_labels(key, resolutions, labels: dict, *, out, adata: AnnData, params: dict, modularities: list):
    """Return the labels for `out=True`, or write them and the `.uns` entry."""
    single = len(resolutions) == 1
    if out is True:
        return labels[resolutions[0]] if single else labels
    for resolution, cat in labels.items():
        adata.obs[key[0] if single else f"{key[0]}_{resolution}"] = cat
    adata.uns[key[1]] = {
        "params": {"resolution": resolutions[0] if single else resolutions, **params},
        "modularity": modularities[0] if single else modularities,
    }
    return None


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

    rows = np.repeat(np.arange(adjacency.shape[0]), np.diff(adjacency.indptr)).astype(np.int32)
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
@deprecated_arg("key_added", DEPR_KEY_ADDED)
@deprecated_arg("adjacency", _DEPR_ADJACENCY)
@deprecated_arg("neighbors_key", DEPR_NEIGHBORS_KEY)
@deprecated_arg("obsp", DEPR_OBSP)
@deprecated_copy
def leiden(
    adata: AnnData,
    resolution: float | list[float] = 1.0,
    *,
    rng: SeedLike | RNGLike | None = None,
    theta: float = 1.0,
    restrict_to: tuple[str, Sequence[str]] | None = None,
    n_iterations: int = 100,
    use_weights: bool = True,
    dtype: str | np.dtype = np.float32,
    use_dask: bool = False,
    use: GraphAcc | str | None = None,
    out: AdRef | str | bool = False,
    # deprecated
    key_added: str | None = None,
    adjacency: sparse.spmatrix | None = None,
    neighbors_key: str | None = None,
    obsp: str | None = None,
    copy: bool = False,
) -> AnnData | pd.Categorical | dict[float, pd.Categorical] | None:
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


        n_iterations
            This controls the maximum number of levels/iterations of the
            Leiden algorithm. When specified, the algorithm will terminate
            after no more than the specified number of iterations. No error
            occurs when the algorithm terminates early in this manner.

        use_weights
            If `True`, edge weights from the graph are used in the
            computation (placing more emphasis on stronger edges).

        dtype
            Data type to use for the adjacency matrix.

        use_dask
            If `True`, use Dask to create the graph and cluster. This will use all GPUs available. This feature is experimental. For datasets with less than 10 Million cells, it is recommended to use `use_dask=False`.

        use
            The neighborhood graph (connectivities) to cluster, e.g. `A.obsp['k']`.
            Strings are resolved by :meth:`anndata.acc.AdAcc.resolve`, e.g. `'obsp.k'`.
            If `None`, the one :func:`~rapids_singlecell.pp.neighbors` wrote by default.

        out
            Where to write the cluster labels, e.g. `A.obs['clusters']`; the parameters go to `.uns` under the same key.
            If `True`, the labels are returned instead (a dict by resolution for several resolutions).
            If `False`, they are written to `.obs['leiden']` and `.uns['leiden']` (`'leiden_R'` with `restrict_to`).
            Several resolutions are written to `.obs[f'{key}_{resolution}']`.
    """
    # Adjacency graph

    rng = np.random.default_rng(rng)
    meta_random_state = {"random_state": rng.arg} if isinstance(rng, _LegacyRng) else {}

    dtype = _check_dtype(dtype)
    key = _out_key(out, key_added, kind="obs", default="leiden")
    # like `key_added`, the default key gets a suffix, keys from `out` are literal
    col = f"{key}_R" if restrict_to is not None and key == "leiden" and out is False else key
    adjacency = _use_graph(adata, use, adjacency=adjacency, obsp=obsp, neighbors_key=neighbors_key)
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
    modularities, labels = [], {}
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
            groups = leiden_parts.sort_values("vertex")[["partition"]].to_numpy().ravel()
            if restrict_to is not None:
                groups = _rename_groups(
                    adata,
                    restrict_key=restrict_key,
                    restrict_categories=restrict_categories,
                    restrict_indices=restrict_indices,
                    groups=groups,
                )
            labels[resolution] = pd.Categorical(
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
    params = {**meta_random_state, "n_iterations": n_iterations}
    return _cluster_labels(
        (col, key), resolutions, labels, out=out, adata=adata, params=params, modularities=modularities
    )


@deprecated(Deprecation("0.19.0", "Use `rapids_singlecell.tl.leiden` instead."))
@deprecated_arg("key_added", DEPR_KEY_ADDED)
@deprecated_arg("adjacency", _DEPR_ADJACENCY)
@deprecated_arg("neighbors_key", DEPR_NEIGHBORS_KEY)
@deprecated_arg("obsp", DEPR_OBSP)
@deprecated_copy
def louvain(
    adata: AnnData,
    resolution: float | list[float] = 1.0,
    *,
    restrict_to: tuple[str, Sequence[str]] | None = None,
    n_iterations: int = 100,
    threshold: float = 1e-7,
    use_weights: bool = True,
    dtype: str | np.dtype = np.float32,
    use_dask: bool = False,
    use: GraphAcc | str | None = None,
    out: AdRef | str | bool = False,
    # deprecated
    key_added: str | None = None,
    adjacency: sparse.spmatrix | None = None,
    neighbors_key: str | None = None,
    obsp: str | None = None,
    copy: bool = False,
) -> AnnData | pd.Categorical | dict[float, pd.Categorical] | None:
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

        dtype
            Data type to use for the adjacency matrix.

        use_dask
            If `True`, use Dask to create the graph and cluster. This will use all GPUs available. This feature is experimental. For datasets with less than 10 Million cells, it is recommended to use `use_dask=False`.

        use
            The neighborhood graph (connectivities) to cluster, e.g. `A.obsp['k']`.
            Strings are resolved by :meth:`anndata.acc.AdAcc.resolve`, e.g. `'obsp.k'`.
            If `None`, the one :func:`~rapids_singlecell.pp.neighbors` wrote by default.

        out
            Where to write the cluster labels, e.g. `A.obs['clusters']`; the parameters go to `.uns` under the same key.
            If `True`, the labels are returned instead (a dict by resolution for several resolutions).
            If `False`, they are written to `.obs['louvain']` and `.uns['louvain']` (`'louvain_R'` with `restrict_to`).
            Several resolutions are written to `.obs[f'{key}_{resolution}']`.

    """
    # Adjacency graph
    dtype = _check_dtype(dtype)
    key = _out_key(out, key_added, kind="obs", default="louvain")
    col = f"{key}_R" if restrict_to is not None and key == "louvain" and out is False else key
    adjacency = _use_graph(adata, use, adjacency=adjacency, obsp=obsp, neighbors_key=neighbors_key)
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
    modularities, labels = [], {}
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
            groups = louvain_parts.sort_values("vertex")[["partition"]].to_numpy().ravel()
            if restrict_to is not None:
                groups = _rename_groups(
                    adata,
                    restrict_key=restrict_key,
                    restrict_categories=restrict_categories,
                    restrict_indices=restrict_indices,
                    groups=groups,
                )
            labels[resolution] = pd.Categorical(
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
    params = {"n_iterations": n_iterations, "threshold": threshold}
    return _cluster_labels(
        (col, key), resolutions, labels, out=out, adata=adata, params=params, modularities=modularities
    )


@_accepts_legacy_random_state(42)
@deprecated_arg("use_rep", Deprecation("0.19.0", "Use `use` instead."))
@deprecated_arg("key_added", DEPR_KEY_ADDED)
@deprecated_copy
def kmeans(
    adata: AnnData,
    n_clusters: int = 8,
    n_pcs: int = 50,
    *,
    n_init: int = 1,
    rng: SeedLike | RNGLike | None = None,
    use: LayerAcc | MultiAcc | str | None = None,
    out: AdRef | str | bool = False,
    # deprecated
    use_rep: str | None = None,
    key_added: str | None = None,
    copy: bool = False,
    **kwargs,
) -> AnnData | pd.Categorical | None:
    """
    KMeans is a basic but powerful clustering method which is optimized via Expectation Maximization. It randomly selects K data points in X, and computes which samples are close to these points. For every cluster of points, a mean is computed (hence the name), and this becomes the new centroid.

    Parameters
    ----------
        adata
            Annotated data matrix.
        n_clusters
            Number of clusters to compute
        n_pcs
            Use this many PCs. If `n_pcs==0` use `.X`.
        n_init
            Number of initializations to run the KMeans algorithm
        rng
            Random seed or :class:`~numpy.random.Generator`; fix it if you want
            results to be the same when you restart Python. Default is 42.
            The superseded `random_state` argument is still accepted.
        use
            The representation to cluster, e.g. `A.obsm['pca']` or `A.X`.
            Strings are resolved by :meth:`anndata.acc.AdAcc.resolve`, e.g. `'obsm.pca'`.
            If `None`, the PCA embedding, under whichever key the data uses.
        out
            Where to write the cluster labels, e.g. `A.obs['clusters']`.
            If `True`, the labels are returned instead. If `False`, they are written to `.obs['kmeans']`.
        **kwargs
            Additional keyword arguments for KMeans.

    """
    rng = np.random.default_rng(rng)

    from cuml.cluster import KMeans

    key = _out_key(out, key_added, kind="obs", default="kmeans")
    use = _resolve_obs(use)
    if use is None:
        use_rep = _resolve_obsm_key(adata, "X_pca" if use_rep is None else use_rep)
    X = _choose_representation(adata, use_rep=use_rep, n_pcs=n_pcs, use=use)

    kmeans_out = KMeans(
        # cuML's KMeans is seeded, so draw the seed right here
        n_clusters=n_clusters,
        n_init=n_init,
        random_state=_seed_from_rng(rng),
        **kwargs,
    ).fit(X)
    groups = kmeans_out.labels_

    labels = pd.Categorical(values=groups.astype("U"), categories=natsorted(map(str, np.unique(groups))))
    if key is True:
        return labels
    adata.obs[key] = labels
    return None


def _rename_groups(adata, *, restrict_key, restrict_categories, restrict_indices, groups):
    """Prefix the subclusters of `restrict_categories` with their parents, keeping the other labels."""
    all_groups = adata.obs[restrict_key].astype("U")
    all_groups.iloc[restrict_indices] = [f"{'-'.join(restrict_categories)},{g}" for g in groups.astype("U")]
    return all_groups
