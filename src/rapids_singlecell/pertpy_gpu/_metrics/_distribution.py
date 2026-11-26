from __future__ import annotations

import cupy as cp
from cupyx import errstate  # type: ignore[attr-defined]
from cupyx.scipy.spatial.distance import cdist  # type: ignore[import-not-found]

from ._group_metric import GroupMetric


def _pairwise_mean(X, Y, pair_values):
    # Bound the temporary pair matrix even when both groups have many cells.
    block_size = 1024
    dtype = cp.result_type(X.dtype, Y.dtype, cp.float32)
    X = cp.ascontiguousarray(X, dtype=dtype)
    Y = cp.ascontiguousarray(Y, dtype=dtype)
    total = cp.zeros((), dtype=cp.float64)
    for i in range(0, len(X), block_size):
        for j in range(0, len(Y), block_size):
            values = pair_values(X[i : i + block_size], Y[j : j + block_size])
            total += values.sum(dtype=cp.float64)
    return total / (len(X) * len(Y))


class MeanPairwiseMetric(GroupMetric):
    """Average distance over all cross-group pairs of cells."""

    def __init__(self, *, cell_wise_metric="euclidean", **kwargs):
        super().__init__(**kwargs)
        aliases = {"manhattan": "cityblock", "l1": "cityblock", "l2": "euclidean"}
        self.cell_wise_metric = aliases.get(cell_wise_metric, cell_wise_metric)

    def _compute(self, X, Y):
        return _pairwise_mean(X, Y, self._distances)

    def _distances(self, X, Y):
        distances = cdist(X, Y, metric=self.cell_wise_metric)
        if self.cell_wise_metric == "cosine":
            # sklearn defines the cosine similarity of a zero cell as zero.
            nonzero = X.any(axis=1)[:, None] & Y.any(axis=1)[None, :]
            return cp.where(nonzero, cp.clip(distances, 0, 2), 1)
        return distances


class MMDMetric(GroupMetric):
    """Biased squared maximum mean discrepancy, including within-group self pairs."""

    def __init__(self, *, kernel="linear", gamma=1.0, degree=2, **kwargs):
        super().__init__(**kwargs)
        if kernel not in {"linear", "rbf", "poly"}:
            raise ValueError(f"Unknown MMD kernel: {kernel!r}")
        if kernel != "linear" and gamma is not None and gamma < 0:
            raise ValueError("MMD kernel gamma must be nonnegative")
        if kernel == "poly" and degree < 1:
            raise ValueError("MMD polynomial degree must be at least one")
        self.kernel = kernel
        self.gamma = gamma
        self.degree = degree

    def _kernel_values(self, X, Y):
        gamma = 1 / X.shape[1] if self.gamma is None else self.gamma
        if self.kernel == "rbf":
            return cp.exp(-gamma * cdist(X, Y, metric="sqeuclidean"))
        return (gamma * (X @ Y.T)) ** self.degree

    def _compute(self, X, Y):
        if self.kernel == "linear":
            return cp.sum((X.mean(axis=0, dtype=cp.float64) - Y.mean(axis=0, dtype=cp.float64)) ** 2)
        return (
            _pairwise_mean(X, X, self._kernel_values)
            + _pairwise_mean(Y, Y, self._kernel_values)
            - 2 * _pairwise_mean(X, Y, self._kernel_values)
        )


class SymmetricKLMetric(GroupMetric):
    """Mean sum of the two directed Gaussian KL divergences per feature."""

    def __init__(self, *, epsilon=1e-8, **kwargs):
        super().__init__(**kwargs)
        self.epsilon = epsilon

    def _compute(self, X, Y):
        delta = X.mean(axis=0, dtype=cp.float64) - Y.mean(axis=0, dtype=cp.float64)
        return self._from_moments(
            delta, X.var(axis=0, dtype=cp.float64), Y.var(axis=0, dtype=cp.float64), len(X), len(Y)
        )

    def _from_moments(self, delta, var_x, var_y, n_x, n_y):
        var_x = (cp.sqrt(var_x) + self.epsilon) ** 2
        var_y = (cp.sqrt(var_y) + self.epsilon) ** 2
        return 0.5 * (((var_x - var_y) ** 2 / (var_x * var_y)) + delta**2 * (1 / var_x + 1 / var_y)).mean(axis=-1)


class TTestMetric(GroupMetric):
    """Mean absolute Welch t statistic across features."""

    def __init__(self, *, epsilon=1e-8, **kwargs):
        super().__init__(**kwargs)
        self.epsilon = epsilon

    def _compute(self, X, Y):
        delta = X.mean(axis=0, dtype=cp.float64) - Y.mean(axis=0, dtype=cp.float64)
        return self._from_moments(
            delta, X.var(axis=0, dtype=cp.float64), Y.var(axis=0, dtype=cp.float64), len(X), len(Y)
        )

    def _from_moments(self, delta, var_x, var_y, n_x, n_y):
        # Convert population variances to unbiased estimates before adding the pseudocount.
        var_x = (var_x * n_x / (n_x - 1) + self.epsilon) / n_x
        var_y = (var_y * n_y / (n_y - 1) + self.epsilon) / n_y
        return (cp.abs(delta) / cp.sqrt(var_x + var_y)).mean(axis=-1)


class KSTestMetric(GroupMetric):
    """Mean two-sided Kolmogorov-Smirnov statistic across features."""

    def _compute(self, X, Y):
        pooled = cp.concatenate((X, Y))
        order = cp.argsort(pooled, axis=0)
        ordered = cp.take_along_axis(pooled, order, axis=0)
        rank = cp.arange(1, len(pooled) + 1)[:, None]
        count_x = cp.cumsum(order < len(X), axis=0)
        cdf_gap = cp.abs(count_x * len(pooled) - rank * len(X))
        # The empirical CDF steps once for each distinct value, including all ties.
        cdf_gap[:-1] *= ordered[:-1] != ordered[1:]
        return cp.where(cp.isnan(ordered[-1]), cp.nan, cdf_gap.max(axis=0)).mean() / (len(X) * len(Y))


class MahalanobisMetric(GroupMetric):
    """Mean-vector distance in the covariance geometry of the first group."""

    def _compute(self, X, Y):
        if len(X) <= X.shape[1]:
            raise cp.linalg.LinAlgError("Mahalanobis distance requires a nonsingular first-group covariance")
        delta = X.mean(axis=0, dtype=cp.float64) - Y.mean(axis=0, dtype=cp.float64)
        covariance = cp.atleast_2d(cp.cov(X, rowvar=False))
        with errstate(linalg="raise"):
            return cp.sqrt(delta @ cp.linalg.solve(covariance, delta))


class MeanVarDistributionMetric(GroupMetric):
    """Squared difference of exponential KDEs in log mean-variance space."""

    @staticmethod
    def _points(data):
        mean = data.mean(axis=0, dtype=cp.float64)
        var = data.var(axis=0, dtype=cp.float64)
        points = cp.column_stack((mean, var))[mean > 0]
        if not len(points) or cp.any(points[:, 1] <= 0):
            raise ValueError("Mean-variance KDE requires positive-mean features with nonzero variance in each group")
        return cp.log(points)

    def _compute(self, X, Y):
        from cuml.neighbors import KernelDensity

        x, y = self._points(X), self._points(Y)
        points = cp.concatenate((x, y))
        low, high = points.min(axis=0), points.max(axis=0)
        # Evaluate at the 100 bin centers spanning both groups, with one extra bin at each end.
        centers = low + (cp.arange(100)[:, None] - 0.5) * ((high - low) / 98)
        grid = cp.stack(cp.meshgrid(centers[:, 0], centers[:, 1]), axis=-1).reshape(-1, 2)
        kde = KernelDensity(kernel="exponential", bandwidth="silverman", output_type="cupy")
        density_x = cp.exp(kde.fit(x).score_samples(grid))
        density_y = cp.exp(kde.fit(y).score_samples(grid))
        return cp.square(density_x - density_y).mean()


DISTRIBUTION_METRICS = {
    "mean_pairwise": MeanPairwiseMetric,
    "mmd": MMDMetric,
    "sym_kldiv": SymmetricKLMetric,
    "t_test": TTestMetric,
    "ks_test": KSTestMetric,
    "mahalanobis": MahalanobisMetric,
    "mean_var_distribution": MeanVarDistributionMetric,
}
