from __future__ import annotations

from typing import TYPE_CHECKING, Any

import cupy as cp
import numpy as np
import pandas as pd
import pytest
import scipy.sparse as sparse
from anndata import AnnData, concat
from anndata.tests.helpers import assert_equal
from cupyx.scipy import sparse as cp_sparse
from cupyx.scipy.sparse import coo_matrix
from numpy.testing import assert_allclose, assert_array_equal

import rapids_singlecell as rsc
from rapids_singlecell.preprocessing._scrublet.sparse_utils import subsample_counts

if TYPE_CHECKING:
    from collections.abc import Callable


def pbmc200() -> AnnData:
    from scanpy.datasets import pbmc3k

    return pbmc3k()[200:400].copy()


def paul500() -> AnnData:
    from scanpy.datasets import paul15

    return paul15()[:500].copy()


@pytest.mark.parametrize(
    ("mk_data", "expected_idx", "expected_scores"),
    [
        pytest.param(pbmc200, [13, 138], [0.149254] * 2, id="sparse"),
        pytest.param(paul500, [180], [0.219178], id="dense"),
    ],
)
def test_scrublet(
    mk_data: Callable[[], AnnData],
    expected_idx: list[int],
    expected_scores: list[float],
):
    """Check that scrublet runs and detects some doublets."""
    adata = mk_data()
    rsc.get.anndata_to_GPU(adata)
    rsc.pp.scrublet(adata, use_approx_neighbors=False)

    doublet_idx = np.flatnonzero(adata.obs["predicted_doublet"]).tolist()
    assert doublet_idx == expected_idx
    assert_allclose(
        adata.obs["doublet_score"].iloc[doublet_idx],
        expected_scores,
        atol=1e-5,
        rtol=1e-5,
    )


def test_scrublet_batched():
    """Test that Scrublet run works with batched data."""
    adata = pbmc200()
    rsc.get.anndata_to_GPU(adata)
    adata.obs["batch"] = 100 * ["a"] + 100 * ["b"]
    split = [adata[adata.obs["batch"] == x].copy() for x in ("a", "b")]

    rsc.pp.scrublet(adata, use_approx_neighbors=False, batch_key="batch")

    doublet_idx = np.flatnonzero(adata.obs["predicted_doublet"]).tolist()
    # only one in the first batch (<100)
    assert doublet_idx == [0, 2, 8, 15, 43, 88, 108, 113, 115, 132, 135, 175]
    assert_allclose(
        adata.obs["doublet_score"].iloc[doublet_idx],
        np.array([0.109375, 0.164835])[([0] * 4 + [1] + [0] * 3 + [1] + [0] * 3)],
        atol=1e-5,
        rtol=1e-5,
    )
    assert adata.uns["scrublet"]["batches"].keys() == {"a", "b"}

    # Check that results are independent
    for s in split:
        rsc.pp.scrublet(s, use_approx_neighbors=False)
    merged = concat(split)

    pd.testing.assert_frame_equal(adata.obs[merged.obs.columns], merged.obs)


def _preprocess_for_scrublet(adata: AnnData) -> AnnData:
    adata_pp = adata.copy()
    rsc.pp.filter_genes(adata_pp, min_cells=3)
    rsc.pp.filter_cells(adata_pp, min_genes=3)
    adata_pp.layers["raw"] = adata_pp.X.copy()
    rsc.pp.normalize_total(adata_pp)
    logged = rsc.pp.log1p(adata_pp, copy=True)
    rsc.pp.highly_variable_genes(logged)
    return adata_pp[:, logged.var["highly_variable"]].copy()


def _create_sim_from_parents(adata: AnnData, parents: np.ndarray) -> AnnData:
    """Simulate doublets based on the randomly selected parents used previously."""
    n_sim = parents.shape[0]
    I = sparse.coo_matrix(
        (
            np.ones(2 * n_sim),
            (np.repeat(np.arange(n_sim), 2), parents.flat),
        ),
        (n_sim, adata.n_obs),
    )
    X = coo_matrix(I) @ adata.layers["raw"]
    return AnnData(
        X,
        var=pd.DataFrame(index=adata.var_names),
        obs={"total_counts": np.ravel(X.sum(axis=1).get())},
        obsm={"doublet_parents": parents.copy()},
    )


def test_scrublet_data():
    """
    Test that Scrublet processing is arranged correctly.

    Check that simulations run on raw data.
    """
    random_state = 1234
    adata = rsc.get.anndata_to_GPU(pbmc200(), copy=True)
    # Run Scrublet and let the main function run simulations
    adata_scrublet_auto_sim = rsc.pp.scrublet(
        adata,
        use_approx_neighbors=False,
        copy=True,
        random_state=random_state,
    )

    # Now make our own simulated data so we can check the result from function
    # is the same, and by inference that the processing steps have not been
    # broken

    # Replicate the preprocessing steps used by the main function
    adata_in = rsc.get.anndata_to_GPU(pbmc200(), copy=True)
    adata_obs = _preprocess_for_scrublet(adata_in)
    # Simulate doublets using the same parents
    adata_sim = _create_sim_from_parents(
        adata_obs, adata_scrublet_auto_sim.uns["scrublet"]["doublet_parents"]
    )

    # Apply the same post-normalisation the Scrublet function would
    rsc.pp.normalize_total(adata_obs, target_sum=1e6)
    rsc.pp.normalize_total(adata_sim, target_sum=1e6)

    adata_scrublet_manual_sim = rsc.pp.scrublet(
        adata_obs,
        adata_sim=adata_sim,
        use_approx_neighbors=False,
        copy=True,
        random_state=random_state,
    )

    # Require that the doublet scores are the same whether simulation is via
    # the main function or manually provided
    assert_allclose(
        adata_scrublet_manual_sim.obs["doublet_score"],
        adata_scrublet_auto_sim.obs["doublet_score"],
        atol=1e-15,
        rtol=1e-15,
    )


@pytest.fixture(scope="module")
def _scrub_small_sess() -> AnnData:
    # Reduce size of input for faster test
    adata = pbmc200()
    rsc.get.anndata_to_GPU(adata)
    rsc.pp.filter_genes(adata, min_cells=100)
    rsc.pp.scrublet(adata, use_approx_neighbors=False)
    return adata


@pytest.fixture()
def scrub_small(_scrub_small_sess: AnnData):
    return _scrub_small_sess.copy()


test_params = {
    "expected_doublet_rate": 0.1,
    "synthetic_doublet_umi_subsampling": 0.8,
    "knn_dist_metric": "manhattan",
    "normalize_variance": False,
    "log_transform": True,
    "mean_center": False,
    "n_prin_comps": 10,
    "n_neighbors": 2,
    "threshold": 0.1,
}


@pytest.mark.parametrize(("param", "value"), test_params.items())
def test_scrublet_params(scrub_small: AnnData, param: str, value: Any):
    """
    Test that Scrublet args are passed.

    Check that changes to parameters change scrublet results.
    """
    curr = rsc.pp.scrublet(
        adata=scrub_small, use_approx_neighbors=False, copy=True, **{param: value}
    )
    with pytest.raises(AssertionError):
        assert_equal(scrub_small, curr)


def test_scrublet_simulate_doublets():
    """Check that doublet simulation runs and simulates some doublets."""
    adata_obs = pbmc200()
    rsc.get.anndata_to_GPU(adata_obs)
    rsc.pp.filter_genes(adata_obs, min_cells=3)
    rsc.pp.filter_cells(adata_obs, min_genes=3)
    adata_obs.layers["raw"] = adata_obs.X
    rsc.pp.normalize_total(adata_obs)
    logged = rsc.pp.log1p(adata_obs, copy=True)

    _ = rsc.pp.highly_variable_genes(logged)
    adata_obs = adata_obs[:, logged.var["highly_variable"]]

    adata_sim = rsc.pp.scrublet_simulate_doublets(
        adata_obs, sim_doublet_ratio=0.02, layer="raw"
    )

    assert_array_equal(
        adata_sim.obsm["doublet_parents"],
        np.array([[13, 132], [106, 43], [152, 3], [160, 103]]),
    )


@pytest.mark.parametrize("dtype", [np.float32, np.float64])
def test_subsample_counts_uses_seeded_gpu_rng(dtype):
    rate = 0.6
    counts = cp.asarray([[4, 0, 2, 1], [0, 3, 0, 5], [1, 1, 6, 0]], dtype=dtype)
    matrix = cp_sparse.csr_matrix(counts)
    original_totals = matrix.sum(axis=1).ravel()

    expected_host_rng = np.random.default_rng(17)
    seed = int(expected_host_rng.integers(0, 2**32))
    expected_gpu_rng = cp.random.default_rng(seed)
    expected = matrix.copy()
    expected.data = expected_gpu_rng.binomial(
        cp.round(expected.data).astype(np.int64), rate
    ).astype(dtype, copy=False)
    expected_current_totals = expected.sum(axis=1).ravel()
    expected_unsampled_totals = expected_gpu_rng.binomial(
        cp.round(original_totals - expected_current_totals).astype(np.int64), rate
    ).astype(dtype, copy=False)

    rng = np.random.default_rng(17)
    actual, actual_totals = subsample_counts(
        matrix.copy(), rate=rate, original_totals=original_totals, rng=rng
    )

    cp.testing.assert_array_equal(actual.toarray(), expected.toarray())
    cp.testing.assert_array_equal(
        actual_totals, expected_current_totals + expected_unsampled_totals
    )
    assert actual.dtype == dtype
    assert actual_totals.dtype == dtype
    assert rng.integers(0, 2**32) == expected_host_rng.integers(0, 2**32)
