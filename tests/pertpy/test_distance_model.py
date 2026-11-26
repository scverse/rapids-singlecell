from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from anndata import AnnData
from scipy.optimize import minimize_scalar
from scipy.sparse import csr_matrix
from scipy.stats import nbinom
from sklearn.linear_model import LogisticRegression

from rapids_singlecell.pertpy_gpu import Distance


def test_negative_binomial_likelihood():
    rng = np.random.default_rng(100)
    X = rng.negative_binomial([2, 1], [0.3, 0.2], size=(90, 2))
    Y = rng.poisson(3, size=(70, 2))
    expected = []
    for x, y in zip(X.T, Y.T, strict=True):
        mu = x.mean()
        fit = minimize_scalar(
            lambda size, counts, mean: -nbinom.logpmf(counts, size, size / (size + mean)).mean(),
            args=(x, mu),
            bounds=(0.01, 100),
            method="bounded",
        )
        expected.append(-nbinom.logpmf(y, fit.x, fit.x / (fit.x + mu)).mean())
    # All-zero training genes with low target expression contribute zero.
    X, Y = np.column_stack((X, np.zeros(len(X)))), np.column_stack((Y, np.ones(len(Y))))
    assert Distance("nb_ll")(X, Y) == pytest.approx(sum(expected) / 3, rel=1e-6)


def test_negative_binomial_invalid_counts_and_failed_fits():
    distance = Distance("nb_ll")
    with pytest.raises(ValueError, match="raw counts"):
        distance(np.array([[0.5]]), np.array([[1]]))
    with pytest.raises(AttributeError, match="over half"):
        distance(np.ones((5, 2)), np.ones((4, 2)))


def test_classifier_proba_scores_held_out_second_group():
    rng = np.random.default_rng(1)
    X, Y = rng.normal(size=(24, 4)), rng.normal(1, size=(17, 4))
    order = np.random.default_rng(42).permutation(len(Y))
    test, train = np.split(order, [len(Y) // 5])
    model = LogisticRegression().fit(np.vstack((X, Y[train])), [0] * len(X) + [1] * len(train))
    expected = model.predict_proba(Y[test])[:, 1].mean()
    assert Distance("classifier_proba", random_state=42)(X, Y) == pytest.approx(expected, abs=5e-4)


def test_classifier_cp_projects_onto_all_other_groups():
    rng = np.random.default_rng(2)
    labels = np.repeat(["z", "ref", "a", "b"], 15)
    X = rng.normal(size=(len(labels), 3)) + np.repeat([0.5, 1, -1, 2], 15)[:, None]
    obs = pd.DataFrame({"group": pd.Categorical(labels)}, index=np.arange(len(labels)).astype(str))
    adata = AnnData(X=csr_matrix(X), obs=obs)
    adata.obsm["X_pca"] = np.zeros_like(X)
    training = labels != "ref"
    clf = LogisticRegression().fit(X[training], labels[training])
    expected = pd.Series(1 - clf.predict_proba(X[~training]).mean(axis=0), index=clf.classes_)
    expected["ref"] = 0.0

    distance = Distance("classifier_cp")
    groups = ["z", "ref", "a"]
    result = distance.onesided_distances(adata, "group", "ref", groups=groups)
    assert result.index.tolist() == groups
    np.testing.assert_allclose(result, expected.loc[groups], atol=5e-4)
    with pytest.raises(NotImplementedError, match="bootstrap"):
        distance.onesided_distances(adata, "group", "ref", bootstrap=True)
    with pytest.raises(NotImplementedError, match="onesided"):
        distance.pairwise(adata, "group")
