from __future__ import annotations

from typing import TYPE_CHECKING

import cupy as cp
import numpy as np
import pandas as pd
from anndata import AnnData
from cupyx.scipy import sparse as cp_sparse

from rapids_singlecell._utils import _create_category_index_mapping
from rapids_singlecell.get import X_to_GPU, aggregate
from rapids_singlecell.preprocessing._utils import _get_mean_var
from rapids_singlecell.squidpy_gpu._utils import _assert_categorical_obs

from ._base_metric import BaseMetric
from ._utils._pseudobulk import (
    paired_abs_mean,
    paired_squared,
    pairwise_abs_mean,
    pairwise_squared,
)

if TYPE_CHECKING:
    from collections.abc import Sequence

    from scipy.sparse import csc_matrix as csc_matrix_cpu
    from scipy.sparse import csr_matrix as csr_matrix_cpu

    _MatrixLike = (
        np.ndarray
        | cp.ndarray
        | csr_matrix_cpu
        | csc_matrix_cpu
        | cp_sparse.csr_matrix
        | cp_sparse.csc_matrix
    )
    _GPUMatrixLike = cp.ndarray | cp_sparse.csr_matrix | cp_sparse.csc_matrix


def _as_gpu_data(data: pd.DataFrame | _MatrixLike) -> _GPUMatrixLike:
    """Unwrap pandas DataFrames before delegating to :func:`X_to_GPU`."""
    if isinstance(data, pd.DataFrame):
        data = data.to_numpy()
    return X_to_GPU(data)


def _subset_data(data, mask: np.ndarray):
    if isinstance(data, pd.DataFrame):
        return data.iloc[mask]
    if isinstance(data, cp.ndarray) or cp_sparse.issparse(data):
        return data[cp.asarray(mask)]
    return data[mask]


def _check_n_bootstrap(n_bootstrap: int) -> None:
    if n_bootstrap < 1:
        raise ValueError("n_bootstrap must be a positive integer.")


def _pairwise_squared_euclidean(
    X_mean: cp.ndarray,
    Y_mean: cp.ndarray,
) -> cp.ndarray:
    distances = cp.sum(X_mean * X_mean, axis=1)[:, None]
    distances = distances + cp.sum(Y_mean * Y_mean, axis=1)[None, :]
    distances = distances - 2 * (X_mean @ Y_mean.T)
    return cp.maximum(distances, 0)


class PseudobulkMetric(BaseMetric):
    """Base class for metrics computed between grouped mean vectors.

    Multi-GPU is intentionally not supported: cells are reduced to one mean
    vector per group up front, so the actual distance kernel runs on a small
    K×K matrix that is cheap on a single GPU.
    """

    supports_multi_gpu: bool = False

    def __init__(
        self,
        *,
        metric_name: str,
        layer_key: str | None = None,
        obsm_key: str | None = "X_pca",
    ):
        super().__init__(layer_key=layer_key, obsm_key=obsm_key)
        self.metric_name = metric_name

    def _get_embedding(self, adata: AnnData):
        """Return raw obsm/layer data without coercion.

        Overrides ``BaseMetric._get_embedding`` to preserve ``pd.DataFrame``
        and sparse inputs so they can be routed through
        :func:`rapids_singlecell.get.aggregate`, which handles those formats
        natively.
        """
        if self.layer_key is not None:
            return adata.layers[self.layer_key]
        return adata.obsm[self.obsm_key]

    def _aggregate_means(
        self,
        adata: AnnData,
        by: str | Sequence[str],
        *,
        mask: np.ndarray | None = None,
    ) -> tuple[cp.ndarray, pd.DataFrame]:
        obs = adata.obs
        data = self._get_embedding(adata)
        if mask is not None:
            obs = obs[mask]
            data = _subset_data(data, mask)

        gpu_data = _as_gpu_data(data)
        tmp = AnnData(X=gpu_data, obs=obs.copy())
        aggregated = aggregate(tmp, by=by, func="mean")
        means = aggregated.layers["mean"]
        if cp_sparse.issparse(means):
            means = means.toarray()
        return means, aggregated.obs.reset_index(drop=True)

    def _get_group_means(
        self,
        adata: AnnData,
        groupby: str,
        groups: Sequence[str] | None,
    ) -> tuple[cp.ndarray, list[str]]:
        _assert_categorical_obs(adata, key=groupby)

        mask = None
        if groups is not None:
            mask = adata.obs[groupby].isin(groups).to_numpy()

        means, obs = self._aggregate_means(adata, groupby, mask=mask)
        return means, list(obs[groupby])

    def _array_mean(self, X) -> cp.ndarray:
        X_gpu = _as_gpu_data(X)
        if X_gpu.ndim != 2:
            raise ValueError("Input arrays must be two-dimensional.")
        if X_gpu.shape[0] == 0:
            raise ValueError("Neither X nor Y can be empty.")
        mean, _ = _get_mean_var(X_gpu, axis=0)
        return mean.reshape(1, -1)

    def _distance_between_means(
        self,
        X_mean: cp.ndarray,
        Y_mean: cp.ndarray,
    ) -> cp.ndarray:
        raise NotImplementedError

    def _distance_between_pairs(
        self,
        X_mean: cp.ndarray,
        Y_mean: cp.ndarray,
    ) -> cp.ndarray:
        raise NotImplementedError

    def _pairwise_from_means(self, means: cp.ndarray) -> cp.ndarray:
        # Default: full K*K via `_distance_between_means` so subclasses can use
        # GEMM (Euclidean/MSE/Pearson/Cosine) or the pairwise kernel (MAE) as
        # the right primitive. R2 overrides to enforce pertpy's upper-triangle
        # convention because its `_distance_between_means` is asymmetric in X.
        distances = self._distance_between_means(means, means)
        cp.fill_diagonal(distances, 0)
        return distances

    def compute_distance(self, X, Y) -> float:
        X_mean = self._array_mean(X)
        Y_mean = self._array_mean(Y)
        return float(self._distance_between_pairs(X_mean, Y_mean)[0])

    def _subset_to_groups(
        self,
        adata: AnnData,
        groupby: str,
        needed_groups: Sequence[str] | None,
    ) -> tuple[_GPUMatrixLike, cp.ndarray, cp.ndarray, list[str]]:
        _assert_categorical_obs(adata, key=groupby)

        obs_col = adata.obs[groupby]
        data = self._get_embedding(adata)
        if needed_groups is not None:
            mask = obs_col.isin(needed_groups).to_numpy()
            obs_col = obs_col[mask].cat.remove_unused_categories()
            data = _subset_data(data, mask)

        groups_list = list(obs_col.cat.categories)
        group_labels = cp.asarray(obs_col.cat.codes.to_numpy(), dtype=cp.int32)
        cat_offsets, cell_indices = _create_category_index_mapping(
            group_labels, len(groups_list)
        )
        return _as_gpu_data(data), cat_offsets, cell_indices, groups_list

    def _bootstrap_sample_cells(
        self,
        *,
        cat_offsets: cp.ndarray,
        cell_indices: cp.ndarray,
        group_sizes_gpu: cp.ndarray,
        seed: int,
    ) -> cp.ndarray:
        """Generate a bootstrap sample using edistance-style index semantics."""
        rng = cp.random.default_rng(seed)
        total_cells = cell_indices.shape[0]

        if total_cells == 0:
            return cell_indices

        random_floats = rng.random(total_cells, dtype=cp.float32)
        group_sizes_list = group_sizes_gpu.get().tolist()
        cell_group_sizes = cp.repeat(group_sizes_gpu, group_sizes_list)
        bootstrap_local_idx = (random_floats * cell_group_sizes).astype(cp.int32)
        cell_group_offsets = cp.repeat(cat_offsets[:-1], group_sizes_list)
        bootstrap_global_idx = bootstrap_local_idx + cell_group_offsets
        return cell_indices[bootstrap_global_idx]

    def _pairwise_bootstrap(
        self,
        adata: AnnData,
        groupby: str,
        groups: Sequence[str] | None,
        *,
        n_bootstrap: int,
        random_state: int,
    ) -> tuple[pd.DataFrame, pd.DataFrame]:
        _check_n_bootstrap(n_bootstrap)
        embedding, cat_offsets, cell_indices, group_names = self._subset_to_groups(
            adata, groupby, groups
        )
        group_sizes = cp.diff(cat_offsets)
        group_sizes_cpu = group_sizes.get().astype(np.intp, copy=False)
        boot_obs = pd.DataFrame(
            {
                groupby: pd.Categorical(
                    np.repeat(group_names, group_sizes_cpu),
                    categories=group_names,
                )
            }
        )
        boot_obs.index = boot_obs.index.astype(str)

        mean = None
        m2 = None
        for i in range(n_bootstrap):
            boot_cell_indices = self._bootstrap_sample_cells(
                cat_offsets=cat_offsets,
                cell_indices=cell_indices,
                group_sizes_gpu=group_sizes,
                seed=random_state + i,
            )
            tmp = AnnData(X=embedding[boot_cell_indices], obs=boot_obs.copy())
            aggregated = aggregate(tmp, by=groupby, func="mean")
            means = aggregated.layers["mean"]
            if cp_sparse.issparse(means):
                means = means.toarray()
            distances = self._pairwise_from_means(means)
            distances = distances.astype(cp.float64, copy=False)
            if mean is None:
                mean = distances.copy()
                m2 = cp.zeros_like(distances)
            else:
                assert m2 is not None
                count = i + 1
                delta = distances - mean
                mean = mean + delta / count
                m2 = m2 + delta * (distances - mean)

        assert mean is not None and m2 is not None
        variance = m2 / n_bootstrap

        df = pd.DataFrame(mean.get(), index=group_names, columns=group_names)
        df.index.name = groupby
        df.columns.name = groupby
        df.name = f"pairwise {self.metric_name}"

        df_var = pd.DataFrame(variance.get(), index=group_names, columns=group_names)
        df_var.index.name = groupby
        df_var.columns.name = groupby
        df_var.name = f"pairwise {self.metric_name} variance"
        return df, df_var

    def _onesided_bootstrap(
        self,
        adata: AnnData,
        groupby: str,
        selected_groups: list[str],
        *,
        needed_groups: Sequence[str] | None,
        n_bootstrap: int,
        random_state: int,
    ) -> tuple[pd.DataFrame, pd.DataFrame]:
        _check_n_bootstrap(n_bootstrap)
        embedding, cat_offsets, cell_indices, group_names = self._subset_to_groups(
            adata, groupby, needed_groups
        )
        group_map = {name: idx for idx, name in enumerate(group_names)}
        selected_indices = [group_map[group] for group in selected_groups]
        selected = cp.asarray(selected_indices, dtype=cp.intp)
        group_sizes = cp.diff(cat_offsets)
        group_sizes_cpu = group_sizes.get().astype(np.intp, copy=False)
        boot_obs = pd.DataFrame(
            {
                groupby: pd.Categorical(
                    np.repeat(group_names, group_sizes_cpu),
                    categories=group_names,
                )
            }
        )
        boot_obs.index = boot_obs.index.astype(str)

        mean = None
        m2 = None
        for i in range(n_bootstrap):
            boot_cell_indices = self._bootstrap_sample_cells(
                cat_offsets=cat_offsets,
                cell_indices=cell_indices,
                group_sizes_gpu=group_sizes,
                seed=random_state + i,
            )
            tmp = AnnData(X=embedding[boot_cell_indices], obs=boot_obs.copy())
            aggregated = aggregate(tmp, by=groupby, func="mean")
            means = aggregated.layers["mean"]
            if cp_sparse.issparse(means):
                means = means.toarray()
            distances = self._distance_between_means(means, means[selected])
            for column, selected_idx in enumerate(selected_indices):
                distances[selected_idx, column] = 0
            distances = distances.astype(cp.float64, copy=False)
            if mean is None:
                mean = distances.copy()
                m2 = cp.zeros_like(distances)
            else:
                assert m2 is not None
                count = i + 1
                delta = distances - mean
                mean = mean + delta / count
                m2 = m2 + delta * (distances - mean)

        assert mean is not None and m2 is not None
        variance = m2 / n_bootstrap

        distances = pd.DataFrame(mean.get(), index=group_names, columns=selected_groups)
        distances.index.name = groupby
        distances.columns.name = "selected_group"

        variances = pd.DataFrame(
            variance.get(), index=group_names, columns=selected_groups
        )
        variances.index.name = groupby
        variances.columns.name = "selected_group"
        return distances, variances

    def bootstrap_arrays(
        self,
        X,
        Y,
        *,
        n_bootstrap: int = 100,
        random_state: int = 0,
    ) -> tuple[float, float]:
        """Bootstrap mean and variance of the distance between two arrays.

        Each iteration resamples ``X`` and ``Y`` independently with
        replacement (each to its own size) and recomputes the distance.
        """
        _check_n_bootstrap(n_bootstrap)
        X_gpu = _as_gpu_data(X)
        Y_gpu = _as_gpu_data(Y)
        if X_gpu.shape[0] == 0 or Y_gpu.shape[0] == 0:
            raise ValueError("Neither X nor Y can be empty.")

        rng = cp.random.default_rng(random_state)
        n_x = X_gpu.shape[0]
        n_y = Y_gpu.shape[0]
        distances = np.empty(n_bootstrap, dtype=np.float64)
        for i in range(n_bootstrap):
            x_idx = rng.integers(0, n_x, size=n_x)
            y_idx = rng.integers(0, n_y, size=n_y)
            distances[i] = self.compute_distance(X_gpu[x_idx], Y_gpu[y_idx])
        return float(np.mean(distances)), float(np.var(distances))

    def pairwise(
        self,
        adata: AnnData,
        groupby: str,
        *,
        groups: Sequence[str] | None = None,
        bootstrap: bool = False,
        n_bootstrap: int = 100,
        random_state: int = 0,
        multi_gpu: bool | list[int] | str | None = None,
    ) -> pd.DataFrame | tuple[pd.DataFrame, pd.DataFrame]:
        if bootstrap:
            return self._pairwise_bootstrap(
                adata,
                groupby,
                groups,
                n_bootstrap=n_bootstrap,
                random_state=random_state,
            )

        means, group_names = self._get_group_means(adata, groupby, groups)
        distances = self._pairwise_from_means(means)
        df = pd.DataFrame(distances.get(), index=group_names, columns=group_names)
        df.index.name = groupby
        df.columns.name = groupby
        df.name = f"pairwise {self.metric_name}"
        return df

    def onesided_distances(
        self,
        adata: AnnData,
        groupby: str,
        selected_group: str | Sequence[str],
        *,
        groups: Sequence[str] | None = None,
        bootstrap: bool = False,
        n_bootstrap: int = 100,
        random_state: int = 0,
        multi_gpu: bool | list[int] | str | None = None,
    ) -> (
        pd.Series
        | pd.DataFrame
        | tuple[pd.Series, pd.Series]
        | tuple[pd.DataFrame, pd.DataFrame]
    ):
        selected_groups, single_control, needed_groups = self._resolve_onesided_inputs(
            adata, groupby, selected_group, groups
        )

        if bootstrap:
            distances, variances = self._onesided_bootstrap(
                adata,
                groupby,
                selected_groups,
                needed_groups=needed_groups,
                n_bootstrap=n_bootstrap,
                random_state=random_state,
            )
            if single_control:
                key = selected_groups[0]
                return distances[key], variances[key]
            return distances, variances

        means, group_names = self._get_group_means(adata, groupby, needed_groups)
        group_map = {name: idx for idx, name in enumerate(group_names)}
        selected_indices = [group_map[group] for group in selected_groups]
        distances = self._distance_between_means(means, means[selected_indices])
        for column, selected_idx in enumerate(selected_indices):
            distances[selected_idx, column] = 0

        result = pd.DataFrame(
            distances.get(),
            index=group_names,
            columns=selected_groups,
        )
        result.index.name = groupby
        result.columns.name = "selected_group"
        if single_control:
            return result[selected_groups[0]]
        return result

    def contrast_distances(
        self,
        adata: AnnData,
        contrasts: pd.DataFrame,
        *,
        multi_gpu: bool | list[int] | str | None = None,
    ) -> pd.DataFrame:
        groupby, split_by = self._parse_contrasts(adata, contrasts)
        by = [groupby, *split_by]

        means, obs = self._aggregate_means(adata, by)

        obs_by_arrays = [obs[col].to_numpy() for col in by]
        condition_to_idx: dict[tuple, int] = {
            key: idx for idx, key in enumerate(zip(*obs_by_arrays))
        }

        target_vals = contrasts[groupby].to_numpy()
        ref_vals = contrasts["reference"].to_numpy()
        split_arrays = [contrasts[col].to_numpy() for col in split_by]

        left_idx: list[int] = []
        right_idx: list[int] = []
        for i in range(len(contrasts)):
            split_key = tuple(arr[i] for arr in split_arrays)
            target_key = (target_vals[i], *split_key)
            reference_key = (ref_vals[i], *split_key)
            try:
                left_idx.append(condition_to_idx[target_key])
                right_idx.append(condition_to_idx[reference_key])
            except KeyError as err:
                raise ValueError(
                    f"No cells found for contrast condition {err.args[0]}"
                ) from None

        left = cp.asarray(left_idx, dtype=cp.intp)
        right = cp.asarray(right_idx, dtype=cp.intp)
        values = self._distance_between_pairs(means[left], means[right])
        values = cp.where(left == right, 0, values)

        result = contrasts.copy()
        result[self.metric_name] = values.get()
        return result


class EuclideanDistance(PseudobulkMetric):
    """Euclidean distance between grouped mean vectors."""

    def _distance_between_means(
        self,
        X_mean: cp.ndarray,
        Y_mean: cp.ndarray,
    ) -> cp.ndarray:
        return cp.sqrt(_pairwise_squared_euclidean(X_mean, Y_mean))

    def _distance_between_pairs(
        self,
        X_mean: cp.ndarray,
        Y_mean: cp.ndarray,
    ) -> cp.ndarray:
        return cp.sqrt(paired_squared(X_mean, Y_mean))


class MeanSquaredDistance(PseudobulkMetric):
    """Mean squared distance between grouped mean vectors."""

    def _distance_between_means(
        self,
        X_mean: cp.ndarray,
        Y_mean: cp.ndarray,
    ) -> cp.ndarray:
        return _pairwise_squared_euclidean(X_mean, Y_mean) / X_mean.shape[1]

    def _distance_between_pairs(
        self,
        X_mean: cp.ndarray,
        Y_mean: cp.ndarray,
    ) -> cp.ndarray:
        return paired_squared(X_mean, Y_mean) / X_mean.shape[1]


class MeanAbsoluteDistance(PseudobulkMetric):
    """Mean absolute distance between grouped mean vectors."""

    def _distance_between_means(
        self,
        X_mean: cp.ndarray,
        Y_mean: cp.ndarray,
    ) -> cp.ndarray:
        return pairwise_abs_mean(X_mean, Y_mean)

    def _distance_between_pairs(
        self,
        X_mean: cp.ndarray,
        Y_mean: cp.ndarray,
    ) -> cp.ndarray:
        return paired_abs_mean(X_mean, Y_mean)


class PearsonDistance(PseudobulkMetric):
    """Pearson distance between grouped mean vectors.

    Matches pertpy's ``1 - scipy.stats.pearsonr`` exactly, which means a
    constant (zero-variance) mean vector yields NaN — same as scipy.
    """

    def _distance_between_means(
        self,
        X_mean: cp.ndarray,
        Y_mean: cp.ndarray,
    ) -> cp.ndarray:
        X_centered = X_mean - cp.mean(X_mean, axis=1, keepdims=True)
        Y_centered = Y_mean - cp.mean(Y_mean, axis=1, keepdims=True)
        numerator = X_centered @ Y_centered.T
        denominator = (
            cp.linalg.norm(X_centered, axis=1)[:, None]
            * cp.linalg.norm(Y_centered, axis=1)[None, :]
        )
        return 1 - numerator / denominator

    def _distance_between_pairs(
        self,
        X_mean: cp.ndarray,
        Y_mean: cp.ndarray,
    ) -> cp.ndarray:
        X_centered = X_mean - cp.mean(X_mean, axis=1, keepdims=True)
        Y_centered = Y_mean - cp.mean(Y_mean, axis=1, keepdims=True)
        numerator = cp.sum(X_centered * Y_centered, axis=1)
        denominator = cp.linalg.norm(X_centered, axis=1) * cp.linalg.norm(
            Y_centered, axis=1
        )
        return 1 - numerator / denominator


class CosineDistance(PseudobulkMetric):
    """Cosine distance between grouped mean vectors.

    Matches pertpy's ``scipy.spatial.distance.cosine`` exactly, which means
    a zero-norm mean vector yields NaN — same as scipy.
    """

    def _distance_between_means(
        self,
        X_mean: cp.ndarray,
        Y_mean: cp.ndarray,
    ) -> cp.ndarray:
        numerator = X_mean @ Y_mean.T
        denominator = (
            cp.linalg.norm(X_mean, axis=1)[:, None]
            * cp.linalg.norm(Y_mean, axis=1)[None, :]
        )
        return cp.clip(1 - numerator / denominator, 0, 2)

    def _distance_between_pairs(
        self,
        X_mean: cp.ndarray,
        Y_mean: cp.ndarray,
    ) -> cp.ndarray:
        numerator = cp.sum(X_mean * Y_mean, axis=1)
        denominator = cp.linalg.norm(X_mean, axis=1) * cp.linalg.norm(Y_mean, axis=1)
        return cp.clip(1 - numerator / denominator, 0, 2)


class R2ScoreDistance(PseudobulkMetric):
    """One minus coefficient of determination between grouped mean vectors."""

    def _distance_between_means(
        self,
        X_mean: cp.ndarray,
        Y_mean: cp.ndarray,
    ) -> cp.ndarray:
        ss_res = pairwise_squared(X_mean, Y_mean)
        centered = X_mean - cp.mean(X_mean, axis=1, keepdims=True)
        ss_tot = cp.sum(centered * centered, axis=1)[:, None]
        return cp.where(ss_tot != 0, ss_res / ss_tot, cp.where(ss_res == 0, 0, 1))

    def _distance_between_pairs(
        self,
        X_mean: cp.ndarray,
        Y_mean: cp.ndarray,
    ) -> cp.ndarray:
        ss_res = paired_squared(X_mean, Y_mean)
        centered = X_mean - cp.mean(X_mean, axis=1, keepdims=True)
        ss_tot = cp.sum(centered * centered, axis=1)
        return cp.where(ss_tot != 0, ss_res / ss_tot, cp.where(ss_res == 0, 0, 1))

    def _pairwise_from_means(self, means: cp.ndarray) -> cp.ndarray:
        # R2 is asymmetric in X (ss_tot is keyed off X). Pertpy's convention is
        # to take the upper-triangle entries as canonical and mirror them onto
        # the lower triangle. We operate on the K×K distance matrix (small),
        # not on means (K×d), so this stays cheap.
        distances = self._distance_between_means(means, means)
        k = len(means)
        if k <= 1:
            return cp.zeros((k, k), dtype=distances.dtype)
        row_idx, col_idx = cp.triu_indices(k, k=1)
        symmetric = cp.zeros_like(distances)
        upper = distances[row_idx, col_idx]
        symmetric[row_idx, col_idx] = upper
        symmetric[col_idx, row_idx] = upper
        return symmetric


PSEUDOBULK_METRICS = {
    "euclidean": EuclideanDistance,
    "root_mean_squared_error": EuclideanDistance,
    "mse": MeanSquaredDistance,
    "mean_absolute_error": MeanAbsoluteDistance,
    "pearson_distance": PearsonDistance,
    "cosine_distance": CosineDistance,
    "r2_distance": R2ScoreDistance,
}
