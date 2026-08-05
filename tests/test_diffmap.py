from __future__ import annotations

import numpy as np
import pytest
import scanpy as sc
from scanpy.datasets import pbmc68k_reduced

from rapids_singlecell.tools._diffmap import (
    _compute_eigen,
    _compute_transitions,
    _load_connectivities,
    diffmap,
)


@pytest.fixture
def adata():
    return pbmc68k_reduced()


class TestLoadConnectivities:
    def test_default_key(self, adata):
        conn = _load_connectivities(adata)
        assert conn.shape == (700, 700)

    def test_custom_key(self, adata):
        adata.uns["custom_neighbors"] = {
            "connectivities_key": "custom_conn",
            "distances_key": "custom_dist",
            "params": {"n_neighbors": 15},
        }
        adata.obsp["custom_conn"] = adata.obsp["connectivities"].copy()
        adata.obsp["custom_dist"] = adata.obsp["distances"].copy()

        conn = _load_connectivities(adata, neighbors_key="custom_neighbors")
        assert conn.shape == (700, 700)

    def test_missing_neighbors_raises(self, adata):
        with pytest.raises(ValueError, match="No neighbors found"):
            _load_connectivities(adata, neighbors_key="no_such_key")

    def test_missing_connectivities_raises(self, adata):
        adata.uns["bad"] = {
            "connectivities_key": "missing_conn",
            "distances_key": "missing_dist",
            "params": {"n_neighbors": 10},
        }
        with pytest.raises(ValueError, match="No connectivities found"):
            _load_connectivities(adata, neighbors_key="bad")


class TestComputeTransitions:
    def test_with_density_normalize(self, adata):
        conn = _load_connectivities(adata)
        tsym, _Z = _compute_transitions(conn, density_normalize=True)

        assert tsym.shape == (700, 700)
        # Symmetric: T_sym == T_sym^T
        diff = tsym - tsym.T
        assert abs(diff).max() < 1e-6

    def test_without_density_normalize(self, adata):
        conn = _load_connectivities(adata)
        tsym, _Z = _compute_transitions(conn, density_normalize=False)

        assert tsym.shape == (700, 700)


class TestComputeEigen:
    def test_shapes(self, adata):
        conn = _load_connectivities(adata)
        tsym, _Z = _compute_transitions(conn)
        evals, evecs = _compute_eigen(tsym, n_comps=15)

        assert evals.shape == (15,)
        assert evecs.shape == (700, 15)

    def test_decreasing_order(self, adata):
        conn = _load_connectivities(adata)
        tsym, _Z = _compute_transitions(conn)
        evals, _evecs = _compute_eigen(tsym, n_comps=10, sort="decrease")

        evals_np = evals.get()
        assert np.all(evals_np[:-1] >= evals_np[1:])

    def test_transitions_required(self, adata):
        """_compute_eigen needs a valid matrix, not None."""
        with pytest.raises((TypeError, ValueError, AttributeError)):
            _compute_eigen(None, n_comps=15)


class TestDiffmap:
    def test_output_keys(self, adata):
        diffmap(adata, n_comps=15)
        assert "X_diffmap" in adata.obsm
        assert "diffmap_evals" in adata.uns
        assert adata.obsm["X_diffmap"].shape == (700, 15)
        assert adata.uns["diffmap_evals"].shape == (15,)

    def test_matches_scanpy(self, adata):
        adata_sc = adata.copy()
        sc.tl.diffmap(adata_sc, n_comps=15)

        diffmap(adata, n_comps=15)

        # Eigenvalues should match closely
        np.testing.assert_allclose(
            adata.uns["diffmap_evals"],
            adata_sc.uns["diffmap_evals"],
            atol=1e-5,
        )
        # Eigenvectors are sign-ambiguous: compare absolute values
        np.testing.assert_allclose(
            np.abs(adata.obsm["X_diffmap"]),
            np.abs(adata_sc.obsm["X_diffmap"]),
            atol=1e-4,
        )

    def test_custom_neighbors_key(self, adata):
        adata.uns["custom_neighbors"] = {
            "connectivities_key": "custom_conn",
            "distances_key": "custom_dist",
            "params": {"n_neighbors": 10},
        }
        adata.obsp["custom_conn"] = adata.obsp["connectivities"].copy()
        adata.obsp["custom_dist"] = adata.obsp["distances"].copy()

        diffmap(adata, n_comps=10, neighbors_key="custom_neighbors")
        assert adata.obsm["X_diffmap"].shape == (700, 10)

    def test_output_on_cpu(self, adata):
        diffmap(adata, n_comps=10)
        assert isinstance(adata.obsm["X_diffmap"], np.ndarray)
        assert isinstance(adata.uns["diffmap_evals"], np.ndarray)
