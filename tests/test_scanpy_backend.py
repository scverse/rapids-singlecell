from __future__ import annotations

import inspect
import tomllib
from copy import copy
from functools import wraps
from pathlib import Path

import cupy as cp
import numpy as np
import pytest
import scanpy as sc
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
def test_public_pp_data_first_signature(name):
    params = inspect.signature(getattr(rsc.pp, name)).parameters
    first_param = next(iter(params))

    assert first_param == "data"
    assert "adata" not in params


@pytest.mark.parametrize(
    "name",
    [
        "filter_cells",
        "filter_genes",
        "regress_out",
        "diffmap",
        "draw_graph",
        "rank_genes_groups",
    ],
)
def test_scanpy_backend_adds_copy(name):
    params = inspect.signature(getattr(scanpy_backend, name)).parameters

    assert params["copy"].default is False
    assert set(
        inspect.signature(
            getattr(rsc.pp, name, None) or getattr(rsc.tl, name)
        ).parameters
    ) <= set(params)


def test_scanpy_backend_copy_returns_filtered_copy():
    adata = AnnData(cp.asarray(np.arange(1, 13, dtype=np.float32).reshape(4, 3)))

    # the dispatcher passes the data object by keyword
    result = scanpy_backend.filter_cells(data=adata, min_counts=10, copy=True)

    assert result is not adata
    assert (result.n_obs, adata.n_obs) == (3, 4)
    assert scanpy_backend.filter_cells(adata, min_counts=10) is None
    assert adata.n_obs == 3


def test_scanpy_backend_hvg_subset_and_inplace():
    rng = np.random.default_rng(0)
    adata = AnnData(cp.asarray(rng.poisson(1.0, (200, 50)).astype(np.float32)))

    metrics = scanpy_backend.highly_variable_genes(adata, n_top_genes=10, inplace=False)

    assert "highly_variable" not in adata.var
    assert metrics["highly_variable"].sum() == 10
    assert (
        len(
            scanpy_backend.highly_variable_genes(
                adata, n_top_genes=10, inplace=False, subset=True
            )
        )
        == 10
    )
    scanpy_backend.highly_variable_genes(adata, n_top_genes=10, subset=True)
    assert adata.n_vars == 10


def test_scanpy_backend_entrypoint_is_declared():
    pyproject = tomllib.loads(
        (Path(__file__).resolve().parents[1] / "pyproject.toml").read_text()
    )

    assert pyproject["project"]["entry-points"]["scanpy.backends"] == {
        "rapids-singlecell": "rapids_singlecell._backends.scanpy"
    }


def test_scanpy_backend_dispatch_smoke(monkeypatch):
    scanpy_backends = pytest.importorskip("scanpy._backends")

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

    @wraps(scanpy_backend.normalize_total)
    def fake_normalize_total(adata: AnnData, **kwargs) -> None:
        adata.X *= kwargs["target_sum"]
        adata.uns["scanpy_backend_called"] = "normalize_total"

    @wraps(scanpy_backend.scale)
    def fake_scale(data: AnnData, **kwargs) -> None:
        data.X *= 2
        data.uns["scanpy_scale_backend_called"] = kwargs

    @wraps(scanpy_backend.pca)
    def fake_pca(data: AnnData, n_comps: int | None = None, **kwargs) -> None:
        data.uns["scanpy_pca_backend_called"] = {
            "n_comps": n_comps,
            **kwargs,
        }

    monkeypatch.setattr(scanpy_backend, "normalize_total", fake_normalize_total)
    monkeypatch.setattr(scanpy_backend, "scale", fake_scale)
    monkeypatch.setattr(scanpy_backend, "pca", fake_pca)

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

        gpu = AnnData(cp.ones((2, 2), dtype=cp.float32))
        obs_metrics, var_metrics = sc.pp.calculate_qc_metrics(gpu, backend="cuda")
        assert "total_counts" in obs_metrics and "total_counts" in var_metrics
        assert "total_counts" not in gpu.obs
        sc.pp.calculate_qc_metrics(gpu, inplace=True, backend="cuda")
        assert "total_counts" in gpu.obs
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
