from __future__ import annotations

import itertools

import cuvs
import numpy as np
import pytest
from packaging.version import parse as parse_version
from scanpy.datasets import pbmc68k_reduced

import rapids_singlecell as rsc


def _calc_recall(distances, reference_distances, tolerance=0.9):
    hits = 0
    total = 0
    for (p_start, p_stop), (g_start, g_stop) in zip(
        itertools.pairwise(distances.indptr),
        itertools.pairwise(reference_distances.indptr),
    ):
        inter = np.intersect1d(
            distances.indices[p_start:p_stop],
            reference_distances.indices[g_start:g_stop],
        )
        hits += inter.size
        total += p_stop - p_start  # number of predicted neighbors

    recall = hits / total
    assert recall > tolerance


@pytest.mark.parametrize("algo", ["mg_ivfflat", "mg_ivfpq"])
def test_mg_neighbors(algo):
    if parse_version(cuvs.__version__) <= parse_version("25.08"):
        pytest.skip("Skipping Multi-GPU neighbors")

    if algo == "mg_ivfflat":
        other_algo = "ivfflat"
    else:
        other_algo = "ivfpq"

    k = 15
    adata = pbmc68k_reduced()
    n_rows = adata.shape[0]
    rsc.pp.neighbors(adata, n_pcs=50, n_neighbors=k, algorithm=algo)

    assert adata.obsp["distances"].shape == (n_rows, n_rows)
    assert adata.obsp["connectivities"].shape == (n_rows, n_rows)
    distances = adata.obsp["distances"].copy()
    rsc.pp.neighbors(adata, n_pcs=50, n_neighbors=k, algorithm=other_algo)
    np.testing.assert_array_equal(adata.obsp["distances"].indptr, distances.indptr)
    _calc_recall(distances, adata.obsp["distances"])


@pytest.mark.parametrize("algo", ["nn_descent", "ivfpq"])
def test_all_neighbors(algo):
    if parse_version(cuvs.__version__) <= parse_version("25.08"):
        pytest.skip("Skipping All-Neighbors")
    adata = pbmc68k_reduced()
    n_rows = adata.shape[0]
    if algo == "ivfpq":
        algorithm_kwds = {
            "algo": "ivf_pq",
            "n_lists": 15,
        }
        tolerance = 0.85
    else:
        algorithm_kwds = {
            "algo": "nn_descent",
        }
        tolerance = 0.95
    rsc.pp.neighbors(
        adata,
        n_pcs=50,
        n_neighbors=15,
        algorithm="all_neighbors",
        algorithm_kwds=algorithm_kwds,
    )
    assert adata.obsp["distances"].shape == (n_rows, n_rows)
    assert adata.obsp["connectivities"].shape == (n_rows, n_rows)
    distances = adata.obsp["distances"].copy()
    rsc.pp.neighbors(adata, n_pcs=50, n_neighbors=15, algorithm=algo)
    _calc_recall(distances, adata.obsp["distances"], tolerance=tolerance)


@pytest.mark.parametrize(
    ("n_devices", "expected"),
    [(1, (1, 1)), (2, (4, 2)), (3, (3, 2)), (4, (4, 2)), (8, (8, 3)), (16, (16, 4))],
)
def test_all_neighbors_batching_defaults(monkeypatch, n_devices, expected):
    import cupy as cp

    from rapids_singlecell.preprocessing._neighbors._algorithms._all_neighbors import (
        _all_neighbors_batching,
    )

    monkeypatch.setattr(cp.cuda.runtime, "getDeviceCount", lambda: n_devices)
    n_clusters, overlap_factor = _all_neighbors_batching({})
    assert (n_clusters, overlap_factor) == expected
    assert n_clusters == 1 or overlap_factor < n_clusters


def test_all_neighbors_batching_overrides():
    from rapids_singlecell.preprocessing._neighbors._algorithms._all_neighbors import (
        _all_neighbors_batching,
    )

    assert _all_neighbors_batching({"n_clusters": 16}) == (16, 4)
    assert _all_neighbors_batching({"n_clusters": 8, "overlap_factor": 2}) == (8, 2)
    # The default overlap is capped at n_clusters - 1, so small explicit cluster
    # counts stay usable instead of tripping the guard below.
    assert _all_neighbors_batching({"n_clusters": 2}) == (2, 1)
    assert _all_neighbors_batching({"n_clusters": 3}) == (3, 2)
    with pytest.raises(ValueError, match="must be greater than"):
        _all_neighbors_batching({"n_clusters": 3, "overlap_factor": 3})


@pytest.mark.parametrize("n_clusters", [4, 8])
def test_all_neighbors_batched(n_clusters):
    """These recall 0.91 and 0.47 with the previous ``overlap_factor=1``."""
    if parse_version(cuvs.__version__) <= parse_version("25.08"):
        pytest.skip("Skipping All-Neighbors")
    adata = pbmc68k_reduced()
    rsc.pp.neighbors(
        adata,
        n_pcs=50,
        n_neighbors=15,
        algorithm="all_neighbors",
        algorithm_kwds={"n_clusters": n_clusters},
    )
    distances = adata.obsp["distances"].copy()
    rsc.pp.neighbors(adata, n_pcs=50, n_neighbors=15, algorithm="brute")
    _calc_recall(distances, adata.obsp["distances"], tolerance=0.95)


@pytest.mark.parametrize("n_clusters", [1, 4])
@pytest.mark.parametrize("metric", ["cosine", "sqeuclidean"])
def test_all_neighbors_metrics(metric, n_clusters):
    if parse_version(cuvs.__version__) <= parse_version("25.08"):
        pytest.skip("Skipping All-Neighbors")
    adata = pbmc68k_reduced()
    rsc.pp.neighbors(
        adata,
        n_pcs=50,
        n_neighbors=15,
        algorithm="all_neighbors",
        metric=metric,
        algorithm_kwds={"n_clusters": n_clusters},
    )
    distances = adata.obsp["distances"].copy()
    rsc.pp.neighbors(adata, n_pcs=50, n_neighbors=15, algorithm="brute", metric=metric)
    _calc_recall(distances, adata.obsp["distances"], tolerance=0.95)


@pytest.mark.parametrize("n_clusters", [1, 4])
def test_all_neighbors_inner_product(n_clusters):
    """``inner_product`` used to fail in the connectivity step, not the build."""
    if parse_version(cuvs.__version__) <= parse_version("25.08"):
        pytest.skip("Skipping All-Neighbors")
    adata = pbmc68k_reduced()
    rsc.pp.neighbors(
        adata,
        n_pcs=50,
        n_neighbors=15,
        algorithm="all_neighbors",
        metric="inner_product",
        algorithm_kwds={"n_clusters": n_clusters},
    )
    distances = adata.obsp["distances"].copy()
    rsc.pp.neighbors(
        adata, n_pcs=50, n_neighbors=15, algorithm="brute", metric="inner_product"
    )
    # inner_product is a similarity, so the top-k are the *largest* dot products and
    # a point is not its own nearest neighbor. nn-descent refines a similarity graph
    # less well than a distance one: recall measures 0.933 here, against 0.995 for
    # the metrics in ``test_all_neighbors_metrics``.
    _calc_recall(distances, adata.obsp["distances"], tolerance=0.9)


@pytest.mark.parametrize("algo", ["mg_ivfflat", "mg_ivfpq"])
def test_mg_bbknn(algo):
    if parse_version(cuvs.__version__) <= parse_version("25.08"):
        pytest.skip("Skipping Multi-GPU neighbors")

    if algo == "mg_ivfflat":
        other_algo = "ivfflat"
    else:
        other_algo = "ivfpq"

    adata = pbmc68k_reduced()
    n_rows = adata.shape[0]
    rsc.pp.bbknn(adata, algorithm=algo, batch_key="phase")

    assert adata.obsp["distances"].shape == (n_rows, n_rows)
    assert adata.obsp["connectivities"].shape == (n_rows, n_rows)
    distances = adata.obsp["distances"].copy()
    rsc.pp.bbknn(adata, batch_key="phase", algorithm=other_algo)
    np.testing.assert_array_equal(adata.obsp["distances"].indptr, distances.indptr)
    _calc_recall(distances, adata.obsp["distances"])
