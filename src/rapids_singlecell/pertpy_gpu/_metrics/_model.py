from __future__ import annotations

import cupy as cp
import numpy as np
import pandas as pd
from cupyx.scipy.special import digamma, gammaln  # type: ignore[import-not-found]

from rapids_singlecell.get import X_to_GPU

from ._base_metric import BaseMetric
from ._group_metric import GroupMetric


class NegativeBinomialDistance(GroupMetric):
    """Negative log-likelihood of Y under per-gene negative binomials fitted to X."""

    def __init__(self, *, epsilon: float = 1e-8, **kwargs):
        super().__init__(**kwargs)
        self.epsilon = epsilon

    def _compute(self, X, Y):
        for counts in (X, Y):
            if not cp.all((counts >= 0) & (cp.abs(counts - cp.rint(counts)) < 1e-6)):
                raise ValueError("nb_ll requires nonnegative raw counts.")

        X = cp.rint(X).astype(cp.float64, copy=False)
        means = X.mean(axis=0)
        fit = X.var(axis=0) > means
        log_likelihood = cp.where((means == 0) & (Y.mean(axis=0) < 10), 0.0, cp.nan)
        if fit.any():
            x, y, mu = X[:, fit], Y[:, fit], means[fit]
            # Profile out the mean and bisect the likelihood score in p = size / (size + mu).
            lower, upper = cp.zeros_like(mu), cp.ones_like(mu)
            for _ in range(48):
                p = (lower + upper) / 2
                size = mu * p / (1 - p)
                score = (digamma(x + size) - digamma(size)).mean(axis=0) + cp.log(p)
                lower = cp.where(score > 0, p, lower)
                upper = cp.where(score > 0, upper, p)

            eps = self.epsilon
            log_norm = cp.log(size + mu + eps)
            log_likelihood[fit] = (
                -size * cp.log1p(mu / (size + eps))
                + y * (cp.log(mu + eps) - log_norm)
                + gammaln(y + size)
                - gammaln(size)
                - gammaln(y.astype(cp.float64, copy=False) + 1)
            ).mean(axis=0)

        skipped = int(cp.isnan(log_likelihood).sum())
        if skipped > len(means) / 2:
            raise AttributeError(f"{skipped} genes could not be fit, which is over half.")
        return -cp.nanmean(log_likelihood)


class ClassifierProbaDistance(GroupMetric):
    """Binary logistic-regression probability on held-out cells of the second group."""

    def __init__(self, *, random_state: int = 0, **kwargs):
        super().__init__(**kwargs)
        self.random_state = random_state

    def _compute(self, X, Y):
        from cuml.linear_model import LogisticRegression

        if len(Y) < 2:
            raise ValueError("classifier_proba needs at least two cells in the second group for its holdout.")
        order = cp.asarray(np.random.default_rng(self.random_state).permutation(len(Y)))
        n_test = max(1, len(Y) // 5)
        train = cp.concatenate((X, Y[order[n_test:]]))
        labels = cp.arange(len(train)) >= len(X)
        model = LogisticRegression(output_type="cupy").fit(train, labels.astype(cp.int32))
        return model.predict_proba(Y[order[:n_test]])[:, 1].mean()


class ClassifierClassProjection(BaseMetric):
    """One-sided projection of reference cells onto the other perturbation classes."""

    def __init__(self, *, metric_name, **kwargs):
        super().__init__(**kwargs)

    def pairwise(self, *args, **kwargs):
        raise NotImplementedError("classifier_cp only supports onesided_distances.")

    def onesided_distances(
        self,
        adata,
        groupby,
        selected_group,
        *,
        groups=None,
        bootstrap=False,
        n_bootstrap=100,
        random_state=0,
        multi_gpu=None,
    ):
        from cuml.linear_model import LogisticRegression

        if bootstrap:
            raise NotImplementedError("classifier_cp does not support bootstrap.")
        refs, single_ref, _ = self._resolve_onesided_inputs(adata, groupby, selected_group, groups)
        labels = adata.obs[groupby].cat.remove_unused_categories()
        targets = list(labels.cat.categories if groups is None else groups)
        missing = set(targets) - set(labels.cat.categories)
        if missing:
            raise ValueError(f"No cells found for groups: {sorted(missing)}")
        if len(labels.cat.categories) < 3:
            raise ValueError("classifier_cp needs at least two training groups in addition to the reference.")

        # Pertpy always trains on X and includes every observed non-reference class.
        X = X_to_GPU(adata.X.astype(np.result_type(adata.X.dtype, np.float32), copy=False))
        result = pd.DataFrame(
            index=pd.Index(targets, name=groupby), columns=pd.Index(refs, name="selected_group"), dtype=float
        )
        for ref in refs:
            testing = labels == ref
            training = labels.notna() & ~testing
            train_labels = labels[training].cat.remove_unused_categories()
            model = LogisticRegression(output_type="cupy").fit(
                X[cp.asarray(np.flatnonzero(training))], cp.asarray(train_labels.cat.codes.to_numpy(), dtype=cp.int32)
            )
            probabilities = model.predict_proba(X[cp.asarray(np.flatnonzero(testing))]).mean(axis=0)
            scores = pd.Series(1 - probabilities.get(), index=train_labels.cat.categories)
            scores.loc[ref] = 0.0
            result[ref] = scores.reindex(targets).to_numpy()
        if single_ref:
            return result[refs[0]].rename(f"classifier_cp to {refs[0]}")
        return result


MODEL_METRICS = {
    "nb_ll": NegativeBinomialDistance,
    "classifier_proba": ClassifierProbaDistance,
    "classifier_cp": ClassifierClassProjection,
}
