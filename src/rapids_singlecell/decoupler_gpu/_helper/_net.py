from __future__ import annotations

import numpy as np
import pandas as pd

from rapids_singlecell.decoupler_gpu._helper._docs import docs
from rapids_singlecell.decoupler_gpu._helper._log import _log


@docs.dedent
def prune(
    features: np.ndarray | None,
    net: pd.DataFrame,
    tmin: int = 5,
    *,
    verbose: bool = False,
) -> pd.DataFrame:
    """
    Removes sources of a ``net`` with less than ``tmin`` targets shared with ``mat``.

    Parameters
    ----------
    %(features)s
    %(net)s
    %(tmin)s
    %(verbose)s

    Returns
    -------
    Filtered net in long format.
    """
    # Validate
    vnet = _validate_net(net, verbose=verbose)
    assert isinstance(tmin, int | float) and tmin >= 0, "tmin must be numeric and >= 0"
    # Find shared targets between mat and net
    if features is not None:
        msk = vnet["target"].isin(set(features))
        vnet = vnet.loc[msk]
    # Find unique sources with tmin
    sources = vnet["source"].value_counts()
    sources = set(sources[sources >= tmin].index)
    # Filter
    msk = vnet["source"].isin(sources)
    vnet = vnet[msk]
    assert not vnet.empty, (
        f"No sources with more than tmin={tmin} targets after\n \
    filtering by shared features in mat.\n \
    Make sure mat and net have shared target features or\n \
    reduce the number assigned to tmin"
    )
    return vnet


def _validate_net(net, *, verbose: bool = False) -> pd.DataFrame:
    assert isinstance(net, pd.DataFrame), "net must be a DataFrame"
    assert {"source", "target"}.issubset(net.columns), (
        "DataFrame must have 'source' and 'target' columns\n \
    If present but with a different names use:\n \
    net = net.rename(columns={'...' : 'source', '...': 'target'})"
    )
    assert not net.duplicated(subset=["source", "target"]).any(), (
        "net has duplicate rows, use:\n \
    net = net.drop_duplicates(subset=['source', 'target'])"
    )
    if "weight" not in net.columns:
        vnet = net[["source", "target"]].copy()
        vnet["weight"] = 1.0
        m = "weight not found in net.columns, adding it as:\nnet['weight'] = 1"
        _log(m, level="warn", verbose=verbose)
    else:
        vnet = net[["source", "target", "weight"]].copy()
    vnet["source"] = vnet["source"].astype("U")
    vnet["target"] = vnet["target"].astype("U")
    vnet["weight"] = vnet["weight"].astype(float)
    return vnet


def _adj(
    net: pd.DataFrame,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    # Pivot df to a wider format
    X = net.pivot(columns="source", index="target", values="weight").fillna(0)
    # Store node names and weights
    sources = X.columns.values.astype("U")
    targets = X.index.values.astype("U")
    X = X.values.astype(float)
    return sources, targets, X


def _order(
    features: np.ndarray,
    targets: np.ndarray,
    adjmat: np.ndarray,
) -> np.ndarray:
    # Init empty madjmat
    madjmat = np.zeros((len(features), adjmat.shape[1]))
    # Create an index array for rows of features corresponding to targets
    features_dict = {gene: i for i, gene in enumerate(features)}
    idxs = [features_dict[gene] for gene in targets if gene in features_dict]
    assert len(idxs) > 0, "No overlap found between features and targets"
    # Populate madjmat using advanced indexing
    madjmat[idxs, :] = adjmat[: len(idxs), :]
    return madjmat


@docs.dedent
def adjmat(
    features: np.ndarray,
    net: pd.DataFrame,
    *,
    verbose: bool = False,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Converts a network in long format into a regulatory adjacency matrix (targets x sources).

    Parameters
    ----------
    %(net)s

    Returns
    -------
    Returns the source names (columns), target names (rows), and the adjacency matrix of weights.
    """
    # Extract adj mat
    sources, targets, adjm = _adj(net=net)
    # Sort adjmat to match features
    adjm = _order(features, targets, adjm)
    m = f"Network adjacency matrix has {targets.size} unique features and {sources.size} unique sources"
    _log(m, level="info", verbose=verbose)
    return sources, targets, adjm


@docs.dedent
def idxmat(
    features: np.ndarray,
    net: pd.DataFrame,
    *,
    verbose: bool = False,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Indexes and returns feature sets as a decomposed sparse matrix.

    Parameters
    ----------
    %(features)s
    %(net)s

    Returns
    -------
    List of sources, concatenated indexes, starts and offsets.
    """
    # Transform targets to indxs
    table = {name: i for i, name in enumerate(features)}
    net["idx_target"] = [table[target] for target in net["target"]]
    # Find sets
    cnct = net.groupby("source", observed=True)["idx_target"].apply(
        lambda x: np.array(x, dtype=int)
    )
    net.drop(columns=["idx_target"], inplace=True)
    sources = cnct.index.values.astype("U")
    # Flatten net and get offsets
    offsets = cnct.apply(lambda x: len(x)).values
    cnct = np.concatenate(cnct.values)
    # Define starts to subset offsets
    starts = np.zeros(offsets.shape[0], dtype=int)
    starts[1:] = np.cumsum(offsets)[:-1]
    targets = np.unique(cnct)
    m = f"Network has {targets.size} unique features and {sources.size} unique sources"
    _log(m, level="info", verbose=verbose)
    return sources, cnct, starts, offsets
