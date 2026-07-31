from __future__ import annotations

import inspect
import tomllib
from copy import copy
from pathlib import Path

import numpy as np
import pytest
from anndata import AnnData

import rapids_singlecell as rsc
from rapids_singlecell._backends import scanpy as scanpy_backend


def _public_functions(module):
    return {
        name
        for name, value in vars(module).items()
        if not name.startswith("_") and inspect.isfunction(value)
    }


@pytest.mark.parametrize("module", [rsc.pp, rsc.tl])
def test_scanpy_backend_exports_public_scanpy_api(module):
    assert _public_functions(module) <= set(scanpy_backend.__all__)


def test_scanpy_backend_exports_aggregate():
    assert scanpy_backend.aggregate is rsc.get.aggregate
    assert "aggregate" in scanpy_backend.__all__


@pytest.mark.parametrize("name", ["log1p", "pca", "scale"])
def test_scanpy_backend_data_first_signature(name):
    first_param = next(
        iter(inspect.signature(getattr(scanpy_backend, name)).parameters)
    )

    assert first_param == "data"


@pytest.mark.parametrize("name", ["log1p", "pca", "scale"])
def test_public_pp_data_first_signature(name):
    params = inspect.signature(getattr(rsc.pp, name)).parameters
    first_param = next(iter(params))

    assert first_param == "data"
    assert "adata" not in params


def test_scanpy_backend_entrypoint_is_declared():
    pyproject = tomllib.loads(Path("pyproject.toml").read_text())

    assert pyproject["project"]["entry-points"]["scanpy.backends"] == {
        "rapids-singlecell": "rapids_singlecell._backends.scanpy"
    }


def test_scanpy_backend_dispatch_smoke(monkeypatch):
    scanpy_backends = pytest.importorskip("scanpy._backends")
    import scanpy as sc

    registry = scanpy_backends.dispatcher._registry
    dispatch_impl = scanpy_backends.dispatcher._dispatch_impl
    old_backend = scanpy_backends.settings.backend
    old_state = {
        "_backends": copy(registry._backends),
        "_alias_map": copy(registry._alias_map),
        "_load_errors": copy(registry._load_errors),
        "_registration_errors": copy(registry._registration_errors),
        "_warned_untrusted": copy(registry._warned_untrusted),
        "_discovered": registry._discovered,
        "_sig_cache": copy(dispatch_impl._sig_cache),
    }

    def fake_normalize_total(
        adata: AnnData, *, target_sum: float | None = None
    ) -> None:
        adata.X *= target_sum
        adata.uns["scanpy_backend_called"] = "normalize_total"

    def fake_scale(data: AnnData, **kwargs) -> None:
        data.X *= 2
        data.uns["scanpy_scale_backend_called"] = kwargs

    def fake_pca(data: AnnData, n_comps: int | None = None, **kwargs) -> None:
        data.uns["scanpy_pca_backend_called"] = {
            "n_comps": n_comps,
            **kwargs,
        }

    monkeypatch.setattr(scanpy_backend, "normalize_total", fake_normalize_total)
    monkeypatch.setattr(scanpy_backend, "_scale", fake_scale)
    monkeypatch.setattr(scanpy_backend, "_pca", fake_pca)

    try:
        scanpy_backends.settings._backend_var.set("cpu")
        registry._backends.clear()
        registry._alias_map.clear()
        registry._load_errors.clear()
        registry._registration_errors.clear()
        registry._warned_untrusted.clear()
        registry._discovered = True
        registry._register_backend(
            scanpy_backend,
            entrypoint_name="rapids-singlecell",
            distribution_name="rapids-singlecell",
            object_ref="rapids_singlecell._backends.scanpy",
        )
        dispatch_impl._sig_cache.clear()
        dispatch_impl._update_signatures()

        adata = AnnData(np.ones((2, 2), dtype=np.float32))
        sc.pp.normalize_total(adata, target_sum=3, backend="cuda")
        sc.pp.scale(adata, max_value=5, backend="cuda")
        sc.pp.pca(adata, n_comps=1, backend="cuda")

        np.testing.assert_allclose(adata.X, 6)
        assert adata.uns["scanpy_backend_called"] == "normalize_total"
        assert adata.uns["scanpy_scale_backend_called"]["max_value"] == 5
        assert adata.uns["scanpy_pca_backend_called"]["n_comps"] == 1
    finally:
        scanpy_backends.settings._backend_var.set(old_backend)
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
