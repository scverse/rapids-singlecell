from __future__ import annotations

from abc import abstractmethod
from itertools import combinations, product

import cupy as cp
import numpy as np
import pandas as pd

from rapids_singlecell.squidpy_gpu._utils import _assert_categorical_obs

from ._base_metric import BaseMetric, _as_gpu_data, _check_n_bootstrap, _is_sparse, _subset_data


def _as_dense(X) -> cp.ndarray:
    if isinstance(X, pd.DataFrame):
        X = X.to_numpy()
    if X.ndim != 2 or 0 in X.shape:
        raise ValueError("Expected a nonempty two-dimensional cell matrix.")
    if X.dtype not in (np.float32, np.float64):
        X = X.astype(np.float64)
    gpu = _as_gpu_data(X.tocsr() if _is_sparse(X) else X)
    return gpu if isinstance(gpu, cp.ndarray) else gpu.toarray()


class GroupMetric(BaseMetric):
    """Distances evaluated on two cell matrices on the current GPU."""

    def __init__(self, *, metric_name, layer_key=None, obsm_key="X_pca"):
        super().__init__(layer_key=layer_key, obsm_key=obsm_key)
        self.metric_name = metric_name

    @abstractmethod
    def _compute(self, X, Y): ...

    def compute_distance(self, X, Y):
        """Compute the distance for a pair of host or device matrices."""
        X, Y = _as_dense(X), _as_dense(Y)
        if X.shape[1] != Y.shape[1]:
            raise ValueError("X and Y must have the same number of features.")
        return float(self._compute(X, Y))

    def _group_arrays(self, adata, groupby, groups):
        _assert_categorical_obs(adata, key=groupby)
        indices = adata.obs.groupby(groupby, observed=True).indices
        wanted = set(indices) if groups is None else set(groups)
        if missing := wanted.difference(indices):
            raise ValueError(f"No cells found for groups: {sorted(missing)}")
        data = self._get_embedding(adata)
        return {g: _as_dense(_subset_data(data, rows)) for g, rows in indices.items() if g in wanted}

    def _evaluate(self, arrays, pairs, *, bootstrap, n_bootstrap, random_state):
        def scores(sample):
            values = [cp.asarray(self._compute(sample[a], sample[b])) for a, b in pairs]
            return cp.stack(values).astype(cp.float64, copy=False) if values else cp.empty(0)

        if not bootstrap:
            return (scores(arrays).get(),)
        _check_n_bootstrap(n_bootstrap)
        rng = cp.random.default_rng(random_state)
        mean = cp.zeros(len(pairs))
        m2 = cp.zeros_like(mean)
        for i in range(1, n_bootstrap + 1):
            sample = {g: x[rng.integers(0, len(x), size=len(x))] for g, x in arrays.items()}
            values = scores(sample)
            delta = values - mean
            mean += delta / i
            m2 += delta * (values - mean)
        return mean.get(), (m2 / n_bootstrap).get()

    def bootstrap_arrays(self, X, Y, *, n_bootstrap=100, random_state=0):
        """Independently resample the cells in each group with replacement."""
        X, Y = _as_dense(X), _as_dense(Y)
        if X.shape[1] != Y.shape[1]:
            raise ValueError("X and Y must have the same number of features.")
        mean, var = self._evaluate(
            {0: X, 1: Y}, [(0, 1)], bootstrap=True, n_bootstrap=n_bootstrap, random_state=random_state
        )
        return float(mean[0]), float(var[0])

    def pairwise(
        self, adata, groupby, *, groups=None, bootstrap=False, n_bootstrap=100, random_state=0, multi_gpu=None
    ):
        """Evaluate the upper triangle and mirror it, following pertpy's convention."""
        arrays = self._group_arrays(adata, groupby, groups)
        names = pd.Index(arrays, name=groupby)
        pairs = list(combinations(arrays, 2))
        results = []
        for flat in self._evaluate(
            arrays, pairs, bootstrap=bootstrap, n_bootstrap=n_bootstrap, random_state=random_state
        ):
            matrix = np.zeros((len(arrays), len(arrays)))
            upper = np.triu_indices(len(arrays), k=1)
            matrix[upper] = matrix[upper[::-1]] = flat
            result = pd.DataFrame(matrix, index=names, columns=names)
            result.name = f"pairwise {self.metric_name}" + (" variance" if results else "")
            results.append(result)
        return tuple(results) if bootstrap else results[0]

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
        """Evaluate each target against the selected references."""
        refs, single, needed = self._resolve_onesided_inputs(adata, groupby, selected_group, groups)
        arrays = self._group_arrays(adata, groupby, needed)
        pairs = [(a, b) for a, b in product(arrays, refs) if a != b]
        results = []
        for flat in self._evaluate(
            arrays, pairs, bootstrap=bootstrap, n_bootstrap=n_bootstrap, random_state=random_state
        ):
            result = pd.DataFrame(
                0.0, index=pd.Index(arrays, name=groupby), columns=pd.Index(refs, name="selected_group")
            )
            for (a, b), value in zip(pairs, flat, strict=True):
                result.loc[a, b] = value
            results.append(result.iloc[:, 0] if single else result)
        return tuple(results) if bootstrap else results[0]

    def contrast_distances(self, adata, contrasts, *, multi_gpu=None):
        """Compare target and reference cells within each requested stratum."""
        groupby, splits = self._parse_contrasts(adata, contrasts)
        indices = self._contrast_indices(adata.obs, contrasts, groupby, splits)
        data = self._get_embedding(adata)
        arrays = {}
        pairs = []
        for target, reference, *levels in contrasts[[groupby, "reference", *splits]].itertuples(index=False, name=None):
            left, right = ((target, *levels), (reference, *levels)) if splits else (target, reference)
            for key in (left, right):
                if key not in indices:
                    raise ValueError(f"No cells found for contrast condition {key}")
                if key not in arrays:
                    arrays[key] = _as_dense(_subset_data(data, indices[key]))
            pairs.append((left, right))
        result = contrasts.copy()
        result[self.metric_name] = [float(self._compute(arrays[a], arrays[b])) if a != b else 0.0 for a, b in pairs]
        return result
