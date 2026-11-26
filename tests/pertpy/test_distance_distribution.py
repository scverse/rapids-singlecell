from __future__ import annotations

import cupy as cp
import cupyx.scipy.sparse as cpsp
import numpy as np
import pandas as pd
import pytest
from anndata import AnnData
from scipy import sparse, stats
from scipy.spatial.distance import cdist, mahalanobis
from sklearn.metrics.pairwise import cosine_distances, linear_kernel, polynomial_kernel, rbf_kernel
from sklearn.neighbors import KernelDensity

from rapids_singlecell.pertpy_gpu import Distance


@pytest.fixture
def samples():
    rng = np.random.default_rng(328)
    return rng.lognormal(size=(13, 4)), rng.lognormal(mean=0.8, size=(21, 4))


@pytest.mark.parametrize("metric", ["euclidean", "cityblock"])
def test_mean_pairwise(samples, metric):
    X, Y = samples
    result = Distance("mean_pairwise", cell_wise_metric=metric)(X, Y)
    np.testing.assert_allclose(result, cdist(X, Y, metric).mean())


def test_mean_pairwise_cosine_with_empty_cells(samples):
    X, Y = samples
    X[0] = Y[-1] = 0
    np.testing.assert_allclose(
        Distance("mean_pairwise", cell_wise_metric="cosine")(X, Y), cosine_distances(X, Y).mean()
    )


@pytest.mark.parametrize(
    ("options", "sklearn_kernel", "kernel_kwargs"),
    [
        ({}, linear_kernel, {}),
        ({"kernel": "rbf", "gamma": 0.3}, rbf_kernel, {"gamma": 0.3}),
        ({"kernel": "poly", "gamma": 0.7, "degree": 3}, polynomial_kernel, {"gamma": 0.7, "degree": 3, "coef0": 0}),
    ],
)
def test_mmd(samples, options, sklearn_kernel, kernel_kwargs):
    X, Y = samples
    weights = np.r_[np.full(len(X), 1 / len(X)), np.full(len(Y), -1 / len(Y))]
    kernel = sklearn_kernel(np.vstack((X, Y)), **kernel_kwargs)
    np.testing.assert_allclose(Distance("mmd", **options)(X, Y), weights @ kernel @ weights)


def test_symmetric_kl(samples):
    X, Y = samples
    X[:, 0] = 2
    epsilon = 0.05

    def directed(A, B):
        sd_a, sd_b = A.std(axis=0) + epsilon, B.std(axis=0) + epsilon
        delta = A.mean(axis=0) - B.mean(axis=0)
        return (np.log(sd_b / sd_a) + (sd_a**2 + delta**2) / (2 * sd_b**2) - 0.5).mean()

    np.testing.assert_allclose(Distance("sym_kldiv", epsilon=epsilon)(X, Y), directed(X, Y) + directed(Y, X))


def test_t_statistic(samples):
    X, Y = samples
    epsilon = 0.01
    t, _ = stats.ttest_ind_from_stats(
        X.mean(axis=0),
        np.sqrt(X.var(axis=0, ddof=1) + epsilon),
        len(X),
        Y.mean(axis=0),
        np.sqrt(Y.var(axis=0, ddof=1) + epsilon),
        len(Y),
        equal_var=False,
    )
    np.testing.assert_allclose(Distance("t_test", epsilon=epsilon)(X, Y), np.abs(t).mean())


def test_ks_includes_entire_tie_at_cdf_step(samples):
    X, Y = (np.round(data) for data in samples)
    X[:, 0] = Y[:, 0] = 1
    statistics = [stats.ks_2samp(x, y).statistic for x, y in zip(X.T, Y.T, strict=True)]
    np.testing.assert_allclose(Distance("ks_test")(X, Y), np.mean(statistics))


def test_mahalanobis_uses_first_group_covariance(samples):
    distance = Distance("mahalanobis")
    for X, Y in (samples, samples[::-1]):
        expected = mahalanobis(X.mean(axis=0), Y.mean(axis=0), np.linalg.inv(np.cov(X.T)))
        np.testing.assert_allclose(distance(X, Y), expected)


def test_mean_var_kde(samples):
    X, Y = samples
    X[:, 0] = 0
    points = []
    for data in (X, Y):
        moments = np.stack((data.mean(axis=0), data.var(axis=0)), axis=1)
        points.append(np.log(moments[moments[:, 0] > 0]))

    axes = []
    for coords in np.concatenate(points).T:
        step = np.ptp(coords) / 98
        axes.append(np.arange(coords.min() - step / 2, coords.max() + step, step))
    grid = np.array(np.meshgrid(*axes)).T.reshape(-1, 2)
    densities = [
        np.exp(KernelDensity(kernel="exponential", bandwidth="silverman").fit(p).score_samples(grid)) for p in points
    ]
    np.testing.assert_allclose(Distance("mean_var_distribution")(X, Y), np.mean(np.diff(densities, axis=0) ** 2))


@pytest.fixture(params=["host_csc", "device_csr"])
def sparse_groups(request):
    rng = np.random.default_rng(437)
    labels = np.tile(["a", "b", "c"], 20)
    obs = pd.DataFrame(
        {"group": pd.Categorical(labels, categories=["c", "b", "a", "unused"]), "batch": np.tile(["x", "y"], 30)}
    ).rename(str)
    X = rng.poisson(2, (len(labels), 3))
    X[labels == "a", 0] *= 3
    layer = sparse.csc_matrix(X) if request.param == "host_csc" else cpsp.csr_matrix(cp.asarray(X, dtype=cp.float32))
    return AnnData(X, obs=obs, layers={"counts": layer}), X


def test_grouped_directional_distance_sparse(sparse_groups):
    adata, X = sparse_groups
    distance = Distance("mahalanobis", layer_key="counts")
    refs = ["a", "c"]
    onesided = distance.onesided_distances(adata, "group", refs, groups=["b"])
    assert list(onesided.index) == ["c", "b", "a"]
    assert list(onesided.columns) == refs
    for target in onesided.index:
        for ref in refs:
            A, B = (X[adata.obs["group"] == g] for g in (target, ref))
            expected = mahalanobis(A.mean(axis=0), B.mean(axis=0), np.linalg.inv(np.cov(A.T)))
            np.testing.assert_allclose(onesided.loc[target, ref], expected)
    pairwise = distance.pairwise(adata, "group", groups=refs)
    assert list(pairwise.index) == ["c", "a"]
    np.testing.assert_allclose(pairwise, [[0, onesided.loc["c", "a"]], [onesided.loc["c", "a"], 0]])


@pytest.mark.parametrize("split", [None, "batch"])
def test_grouped_contrasts_sparse(sparse_groups, split):
    adata, X = sparse_groups
    distance = Distance("mmd", layer_key="counts")
    contrasts = Distance.create_contrasts(adata, "group", ["a", "b"], split_by=split).iloc[::-1]
    result = distance.contrast_distances(adata, contrasts)
    pd.testing.assert_frame_equal(result.drop(columns="mmd"), contrasts)
    for row in result.itertuples():
        keep = np.ones(len(X), dtype=bool) if split is None else adata.obs["batch"] == row.batch
        A, B = (X[keep & (adata.obs["group"] == g)] for g in (row.group, row.reference))
        np.testing.assert_allclose(row.mmd, np.sum((A.mean(axis=0) - B.mean(axis=0)) ** 2))


def test_grouped_bootstrap_sparse(sparse_groups):
    adata, _ = sparse_groups
    distance = Distance("mmd", layer_key="counts")
    groups = ["c", "a"]
    options = {"n_bootstrap": 9, "random_state": 123}
    X, Y = (adata.layers["counts"][np.flatnonzero(adata.obs["group"] == g)] for g in groups)
    mu, var = distance.bootstrap(X, Y, **options)
    assert var > 0
    pairwise = distance.pairwise(adata, "group", groups=groups, bootstrap=True, **options)
    onesided = distance.onesided_distances(adata, "group", "a", groups=groups, bootstrap=True, **options)
    for frame, series, expected in zip(pairwise, onesided, (mu, var), strict=True):
        np.testing.assert_allclose(frame.loc["c", "a"], expected)
        pd.testing.assert_series_equal(series, frame["a"])
