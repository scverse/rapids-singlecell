from __future__ import annotations

import logging

import numpy as np
import pytest

import rapids_singlecell.decoupler_gpu as dc


@pytest.mark.parametrize("verbose", [True, False])
def test_validate_net(
    net,
    verbose,
    caplog,
):
    with caplog.at_level(logging.WARNING):
        vnet = dc._helper._net._validate_net(net, verbose=verbose)
    assert caplog.text == ""
    assert net.shape == vnet.shape
    assert "source" in vnet.columns
    assert "target" in vnet.columns
    assert "weight" in vnet.columns

    net.drop(columns=["weight"], inplace=True)
    assert "weight" not in net.columns
    with caplog.at_level(logging.WARNING):
        vnet = dc._helper._net._validate_net(net, verbose=verbose)
    if verbose:
        assert len(caplog.text) > 0
    assert "weight" in vnet.columns

    enet = net.rename(columns={"source": "tf"})
    with pytest.raises(AssertionError):
        dc._helper._net._validate_net(enet)


@pytest.mark.parametrize(
    "tmin,nrow_equal,raise_err",
    [
        [0, True, False],
        [3, True, False],
        [4, False, False],
        [5, False, True],
    ],
)
def test_prune(
    net,
    tmin,
    nrow_equal,
    raise_err,
):
    features = [f"G{i + 1:02d}" for i in range(20)]
    if raise_err:
        with pytest.raises(AssertionError):
            dc._helper._net.prune(features=features, net=net, tmin=tmin)
    else:
        pnet = dc._helper._net.prune(features=features, net=net, tmin=tmin)
        if nrow_equal:
            assert pnet.shape[0] == net.shape[0]
        else:
            assert pnet.shape[0] < net.shape[0]


def test_adj(
    net,
):
    sources, targets, adj = dc._helper._net._adj(net=net)
    sources, targets = list(sources), list(targets)
    net = net.set_index(["source", "target"])
    for s in sources:
        j = sources.index(s)
        snet = net.loc[s]
        for t in snet.index:
            i = targets.index(t)
            w_adj = adj[i, j]
            w_net = snet.loc[t]["weight"]
            print(s, t, i, j, w_net, w_adj)
            assert w_net == w_adj
    assert (len(targets), len(sources)) == adj.shape
    assert adj.dtype == float


def test_order(net):
    sources, targets, adjmat = dc._helper._net._adj(net=net)

    rtargets = targets[::-1]
    radjmat = adjmat[::-1]
    oadjmat = dc._helper._net._order(features=targets, targets=rtargets, adjmat=radjmat)
    assert (adjmat == oadjmat).all()

    mfeatures = list(targets) + ["x", "y", "z"]
    madjmat = dc._helper._net._order(features=mfeatures, targets=targets, adjmat=adjmat)
    assert np.all(madjmat == 0, axis=1).sum() == 3

    lfeatures = targets[:5]
    assert lfeatures.size < targets.size
    ladjmat = dc._helper._net._order(features=lfeatures, targets=targets, adjmat=adjmat)
    assert ladjmat.shape[0] < adjmat.shape[0]
    assert np.all(ladjmat == 0, axis=1).sum() == 0


@pytest.mark.parametrize("verbose", [True, False])
def test_adjmat(
    adata,
    net,
    unwnet,
    verbose,
    caplog,
):
    features = adata.var_names
    with caplog.at_level(logging.INFO):
        sources, targets, adjmat = dc._helper._net.adjmat(features=features, net=net, verbose=verbose)
    adjmat = adjmat.ravel()
    non_zero_adjmat = adjmat[adjmat != 0.0]
    assert not all(non_zero_adjmat == 1.0)
    if verbose:
        assert len(caplog.text) > 0
    else:
        assert caplog.text == ""
    unwnet = dc._helper._net._validate_net(unwnet)
    sources, targets, adjmat = dc._helper._net.adjmat(features=features, net=unwnet)
    adjmat = adjmat.ravel()
    non_zero_adjmat = adjmat[adjmat != 0.0]
    assert all(non_zero_adjmat == 1.0)


@pytest.mark.parametrize("verbose", [True, False])
def test_idxmat(
    adata,
    net,
    verbose,
    caplog,
):
    features = adata.var_names
    with caplog.at_level(logging.INFO):
        sources, cnct, starts, offsets = dc._helper._net.idxmat(features=features, net=net, verbose=verbose)
    if verbose:
        assert len(caplog.text) > 0
    else:
        assert caplog.text == ""
    assert sources.size == starts.size
    assert starts.size == offsets.size
    assert (net.groupby("source")["target"].size().loc[sources] == offsets).all()
    assert cnct.size == offsets.sum()
