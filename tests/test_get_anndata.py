from __future__ import annotations

import numpy as np
import pytest
from anndata import AnnData
from scipy.sparse import csc_array, csc_matrix, csr_array, csr_matrix

import rapids_singlecell as rsc
from rapids_singlecell.preprocessing._utils import _check_gpu_X


def _get_obs_rep_adata() -> AnnData:
    adata = AnnData(np.arange(6).reshape(3, 2))
    adata.layers["counts"] = np.full(adata.shape, 2)
    adata.raw = AnnData(np.full(adata.shape, 3))
    adata.obsm["embedding"] = np.full((adata.n_obs, 4), 4)
    adata.obsp["connectivities"] = np.full((adata.n_obs, adata.n_obs), 5)
    return adata


@pytest.mark.parametrize(
    ("kwargs", "expected"),
    [
        pytest.param({}, "X", id="X"),
        pytest.param({"layer": "counts"}, "layer", id="layer"),
        pytest.param({"use_raw": True}, "raw", id="raw"),
        pytest.param({"obsm": "embedding"}, "obsm", id="obsm"),
        pytest.param({"obsp": "connectivities"}, "obsp", id="obsp"),
    ],
)
def test_get_obs_rep(kwargs, expected):
    adata = _get_obs_rep_adata()
    expected_value = {
        "X": adata.X,
        "layer": adata.layers["counts"],
        "raw": adata.raw.X,
        "obsm": adata.obsm["embedding"],
        "obsp": adata.obsp["connectivities"],
    }[expected]

    assert rsc.get._get_obs_rep(adata, **kwargs) is expected_value


@pytest.mark.parametrize(
    ("kwargs", "shape"),
    [
        pytest.param({}, (3, 2), id="X"),
        pytest.param({"layer": "result"}, (3, 2), id="layer"),
        pytest.param({"obsm": "result"}, (3, 4), id="obsm"),
        pytest.param({"obsp": "result"}, (3, 3), id="obsp"),
    ],
)
def test_set_obs_rep(kwargs, shape):
    adata = _get_obs_rep_adata()
    value = np.full(shape, 6)

    rsc.get._set_obs_rep(adata, value, **kwargs)

    assert rsc.get._get_obs_rep(adata, **kwargs) is value


def test_set_obs_rep_raw_is_read_only():
    with pytest.raises(AttributeError):
        rsc.get._set_obs_rep(_get_obs_rep_adata(), np.full((3, 2), 6), use_raw=True)


def test_get_obs_rep_invalid_use_raw():
    with pytest.raises(TypeError, match="use_raw expected to be bool"):
        rsc.get._get_obs_rep(_get_obs_rep_adata(), use_raw=1)


def test_get_obs_rep_conflicting_choices():
    with pytest.raises(ValueError, match="Only one of `layer`, or `obsm`"):
        rsc.get._get_obs_rep(_get_obs_rep_adata(), layer="counts", obsm="embedding")


def test_set_obs_rep_conflicting_choices():
    with pytest.raises(AssertionError):
        rsc.get._set_obs_rep(
            _get_obs_rep_adata(),
            np.zeros((3, 2)),
            layer="counts",
            obsm="embedding",
        )


@pytest.mark.parametrize(
    "mtype", [csc_matrix, csr_matrix, csc_array, csr_array, "dense"]
)
def test_utils(mtype):
    X = np.arange(12, dtype=np.float32).reshape(3, 4)
    adata = AnnData(X if mtype == "dense" else mtype(X))
    # check X
    rsc.get.anndata_to_GPU(adata)
    rsc.preprocessing._utils._check_gpu_X(adata.X)
    rsc.get.anndata_to_CPU(adata)
    assert isinstance(adata.X, np.ndarray | csr_matrix | csc_matrix)
    # check layers
    adata.layers["test"] = adata.X.copy()
    rsc.get.anndata_to_GPU(adata, convert_all=True)
    _check_gpu_X(adata.layers["test"])
    rsc.get.anndata_to_CPU(adata, convert_all=True)
    assert isinstance(adata.layers["test"], np.ndarray | csr_matrix | csc_matrix)
