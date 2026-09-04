from __future__ import annotations

import tomllib
from copy import copy
from pathlib import Path

import numpy as np
import pytest
from anndata import AnnData

from rapids_singlecell._backends import squidpy as squidpy_backend


def test_squidpy_backend_identity_and_exports():
    assert squidpy_backend.name == "rapids-singlecell"
    assert squidpy_backend.aliases == ["cuda", "rapids", "rapids_singlecell"]
    assert squidpy_backend.__all__ == [
        "calculate_niche",
        "co_occurrence",
        "ligrec",
        "spatial_autocorr",
    ]


def test_squidpy_backend_entrypoint_is_declared():
    pyproject = tomllib.loads(Path("pyproject.toml").read_text())

    assert pyproject["project"]["entry-points"]["squidpy.backends"] == {
        "rapids-singlecell": "rapids_singlecell._backends.squidpy"
    }


def test_calculate_niche_maps_inplace_to_copy(monkeypatch):
    adata = AnnData(np.ones((2, 2), dtype=np.float32))
    captured = {}

    def fake_calculate_niche(data, **kwargs):
        captured["data"] = data
        captured.update(kwargs)
        return data if kwargs["copy"] else None

    monkeypatch.setattr(squidpy_backend, "_calculate_niche", fake_calculate_niche)

    result = squidpy_backend.calculate_niche(adata, flavor="utag", inplace=False)

    assert result is adata
    assert captured["data"] is adata
    assert captured["copy"] is True


def test_squidpy_backend_dispatch_smoke(monkeypatch):
    squidpy_backends = pytest.importorskip("squidpy._backends")
    import squidpy as sq

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

    def fake_calculate_niche(data, **kwargs):
        data.uns["squidpy_backend_called"] = kwargs
        return data if kwargs["copy"] else None

    monkeypatch.setattr(squidpy_backend, "_calculate_niche", fake_calculate_niche)

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
            adata, flavor="utag", backend="cuda", inplace=False
        )

        assert result is adata
        assert adata.uns["squidpy_backend_called"]["copy"] is True
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
