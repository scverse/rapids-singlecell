from __future__ import annotations

import inspect
import tomllib
from copy import copy
from functools import wraps
from pathlib import Path

import numpy as np
import pytest
from anndata import AnnData

import rapids_singlecell as rsc
from rapids_singlecell._backends import squidpy as squidpy_backend


def test_squidpy_backend_identity_and_exports():
    assert squidpy_backend.name == "rapids-singlecell"
    assert squidpy_backend.aliases == ["cuda", "rapids", "rapids_singlecell"]
    assert squidpy_backend.__all__ == [
        "calculate_niche",
        "calculate_niche_cellcharter",
        "calculate_niche_neighborhood",
        "calculate_niche_utag",
        "co_occurrence",
        "ligrec",
        "spatial_autocorr",
    ]


def test_squidpy_backend_exports_public_gr_api():
    public = {
        name
        for name, value in vars(rsc.gr).items()
        if not name.startswith("_") and inspect.isfunction(value)
    }

    assert public <= set(squidpy_backend.__all__)


def test_squidpy_backend_entrypoint_is_declared():
    pyproject = tomllib.loads(
        (Path(__file__).resolve().parents[1] / "pyproject.toml").read_text()
    )

    assert pyproject["project"]["entry-points"]["squidpy.backends"] == {
        "rapids-singlecell": "rapids_singlecell._backends.squidpy"
    }


def test_calculate_niche_routes_to_flavor_function(monkeypatch):
    adata = AnnData(np.ones((2, 2), dtype=np.float32))
    captured = {}

    @wraps(squidpy_backend._utag)
    def fake_utag(data, **kwargs):
        captured["data"] = data
        captured.update(kwargs)
        return None if kwargs["inplace"] else data

    monkeypatch.setattr(squidpy_backend, "_utag", fake_utag)

    with pytest.warns(UserWarning, match="groups are not used for flavor 'utag'"):
        result = squidpy_backend.calculate_niche(
            adata,
            flavor="utag",
            groups="cluster",
            n_neighbors=10,
            resolutions=0.5,
            layer_ratio=2.0,
            inplace=False,
        )

    assert result is adata
    assert captured.pop("data") is adata
    assert captured == {
        "spatial_connectivities_key": "spatial_connectivities",
        "inplace": False,
        "table_key": None,
        "n_neighbors": 10,
        "resolutions": 0.5,
    }


def test_calculate_niche_cellcharter_maps_squidpy_conventions(monkeypatch):
    adata = AnnData(np.ones((2, 2), dtype=np.float32))
    captured = {}

    @wraps(squidpy_backend._cellcharter)
    def fake_cellcharter(data, **kwargs):
        captured["data"] = data
        captured.update(kwargs)
        return data

    monkeypatch.setattr(squidpy_backend, "_cellcharter", fake_cellcharter)

    result = squidpy_backend.calculate_niche_cellcharter(
        adata, distance=2, rng=np.random.default_rng(0), copy=True
    )

    assert result is adata
    assert captured.pop("data") is adata
    assert isinstance(captured.pop("random_state"), int)
    assert captured == {
        "inplace": False,
        "distance": 2,
        "aggregation": "mean",
        "spatial_connectivities_key": "spatial_connectivities",
        "n_components": 10,
        "use_rep": None,
        "min_niche_size": None,
        "mask": None,
        "library_key": None,
        "table_key": None,
    }


def test_calculate_niche_rejects_spatialleiden():
    adata = AnnData(np.ones((2, 2), dtype=np.float32))

    with pytest.raises(NotImplementedError, match="spatialleiden"):
        squidpy_backend.calculate_niche(adata, flavor="spatialleiden")


def test_calculate_niche_forwards_spatialdata_table(monkeypatch):
    spatialdata = pytest.importorskip("spatialdata")
    adata = AnnData(np.ones((2, 2), dtype=np.float32))
    sdata = spatialdata.SpatialData(tables={"table": adata})
    captured = {}

    @wraps(squidpy_backend._cellcharter)
    def fake_cellcharter(data, **kwargs):
        captured["data"] = data
        captured.update(kwargs)

    monkeypatch.setattr(squidpy_backend, "_cellcharter", fake_cellcharter)

    squidpy_backend.calculate_niche(
        sdata, flavor="cellcharter", distance=2, aggregation="mean", table_key="table"
    )

    assert captured["data"] is sdata
    assert captured["table_key"] == "table"


def test_calculate_niche_requires_squidpy_arguments():
    adata = AnnData(np.ones((2, 2), dtype=np.float32))

    with pytest.raises(ValueError, match="'resolutions' is required for flavor 'utag'"):
        squidpy_backend.calculate_niche(adata, flavor="utag", n_neighbors=10)


def test_ligrec_drops_cpu_only_options(monkeypatch):
    adata = AnnData(np.ones((2, 2), dtype=np.float32))
    captured = {}

    def fake_ligrec(data, cluster_key, **kwargs):
        captured.update(kwargs)

    monkeypatch.setattr(squidpy_backend, "_ligrec", fake_ligrec)

    with pytest.warns(UserWarning, match="seed, n_jobs have no effect"):
        squidpy_backend.ligrec(adata, "cluster", seed=0, n_jobs=4, n_perms=5)

    assert captured == {"n_perms": 5}
    assert inspect.signature(squidpy_backend.ligrec) == inspect.signature(rsc.gr.ligrec)


def test_squidpy_backend_dispatch_smoke(monkeypatch):
    squidpy_backends = pytest.importorskip("squidpy._backends")
    sq = pytest.importorskip("squidpy")

    registry = squidpy_backends.dispatcher._registry
    dispatch_impl = squidpy_backends.dispatcher._dispatch_impl
    old_backend = squidpy_backends.settings.backend
    old_state = {
        "_backends": copy(registry._backends),
        "_alias_map": copy(registry._alias_map),
        "_load_errors": copy(registry._load_errors),
        "_registration_errors": copy(registry._registration_errors),
        "_warned_untrusted": copy(registry._warned_untrusted),
        "_discovered": registry._discovered,
        "_sig_cache": copy(dispatch_impl._sig_cache),
    }

    @wraps(squidpy_backend._utag)
    def fake_utag(data, **kwargs):
        data.uns["squidpy_backend_called"] = kwargs
        return None if kwargs["inplace"] else data

    monkeypatch.setattr(squidpy_backend, "_utag", fake_utag)

    try:
        squidpy_backends.settings._backend_var.set("cpu")
        registry._backends.clear()
        registry._alias_map.clear()
        registry._load_errors.clear()
        registry._registration_errors.clear()
        registry._warned_untrusted.clear()
        registry._discovered = True
        registry._register_backend(
            squidpy_backend,
            entrypoint_name="rapids-singlecell",
            distribution_name="rapids-singlecell",
            object_ref="rapids_singlecell._backends.squidpy",
        )
        dispatch_impl._sig_cache.clear()
        dispatch_impl._update_signatures()

        adata = AnnData(np.ones((2, 2), dtype=np.float32))
        result = sq.gr.calculate_niche(
            adata,
            flavor="utag",
            n_neighbors=10,
            resolutions=0.5,
            backend="cuda",
            inplace=False,
        )

        assert result is adata
        assert adata.uns["squidpy_backend_called"]["inplace"] is False
    finally:
        squidpy_backends.settings._backend_var.set(old_backend)
        registry._backends.clear()
        registry._backends.update(old_state["_backends"])
        registry._alias_map.clear()
        registry._alias_map.update(old_state["_alias_map"])
        registry._load_errors.clear()
        registry._load_errors.update(old_state["_load_errors"])
        registry._registration_errors.clear()
        registry._registration_errors.update(old_state["_registration_errors"])
        registry._warned_untrusted.clear()
        registry._warned_untrusted.update(old_state["_warned_untrusted"])
        registry._discovered = old_state["_discovered"]
        dispatch_impl._sig_cache.clear()
        dispatch_impl._sig_cache.update(old_state["_sig_cache"])
        dispatch_impl._update_signatures()
