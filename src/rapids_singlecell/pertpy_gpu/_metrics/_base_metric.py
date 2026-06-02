from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING

import cupy as cp
import numpy as np

from rapids_singlecell._utils import _create_category_index_mapping, parse_device_ids
from rapids_singlecell.squidpy_gpu._utils import _assert_categorical_obs

if TYPE_CHECKING:
    from collections.abc import Sequence

    from anndata import AnnData

# Re-export for backwards compatibility
__all__ = ["BaseMetric", "parse_device_ids"]


class BaseMetric(ABC):
    """
    Abstract base class for distance metrics.

    All distance metric implementations should inherit from this class
    and implement the required methods.

    Parameters
    ----------
    layer_key
        Key in adata.layers for cell data. Mutually exclusive with obsm_key.
    obsm_key
        Key in adata.obsm for embeddings (default: 'X_pca')

    Attributes
    ----------
    supports_multi_gpu
        Whether this metric supports multi-GPU computation.
        Subclasses should override this to True if they implement multi-GPU.
    """

    supports_multi_gpu: bool = False

    def __init__(
        self,
        layer_key: str | None = None,
        obsm_key: str | None = "X_pca",
    ):
        """Initialize base metric."""
        if layer_key is not None and obsm_key is not None:
            raise ValueError(
                "Cannot use 'layer_key' and 'obsm_key' at the same time. "
                "Please provide only one of the two keys."
            )
        self.layer_key = layer_key
        self.obsm_key = obsm_key

    def _get_embedding(self, adata: AnnData) -> np.ndarray | cp.ndarray:
        """Get embedding from adata using layer_key or obsm_key.

        Returns the embedding in its original format (numpy or cupy).
        Preserves the input dtype (float32 or float64) for precision control.
        """
        if self.layer_key is not None:
            data = adata.layers[self.layer_key]
        else:
            data = adata.obsm[self.obsm_key]

        if isinstance(data, (cp.ndarray, np.ndarray)):
            return data
        return np.asarray(data)

    def _subset_to_groups(
        self,
        adata: AnnData,
        groupby: str,
        needed_groups: Sequence[str] | None,
    ) -> tuple[cp.ndarray, cp.ndarray, cp.ndarray, list[str]]:
        """Subset the embedding to ``groupby`` and build its category mapping.

        Unused (zero-cell) categories are always dropped so they never become
        empty groups. When ``needed_groups`` is given, only those cells are kept
        and a requested group with no cells raises ``ValueError``.

        Returns
        -------
        embedding
            Cell embeddings (CuPy) for the kept cells; input dtype preserved.
        cat_offsets, cell_indices
            Category offsets and the cell indices grouped by category.
        groups_list
            Ordered group names matching the category indices.
        """
        obs_col = adata.obs[groupby]
        embedding_raw = self._get_embedding(adata)

        if needed_groups is not None:
            mask = obs_col.isin(needed_groups).values
            obs_col = obs_col[mask].cat.remove_unused_categories()
            embedding = cp.asarray(embedding_raw[mask])
        else:
            obs_col = obs_col.cat.remove_unused_categories()
            embedding = cp.asarray(embedding_raw)

        groups_list = list(obs_col.cat.categories)
        if needed_groups is not None:
            missing = sorted(set(needed_groups) - set(groups_list))
            if missing:
                raise ValueError(f"No cells found for groups: {missing}")
        group_labels = cp.array(obs_col.cat.codes.values, dtype=cp.int32)
        cat_offsets, cell_indices = _create_category_index_mapping(
            group_labels, len(groups_list)
        )
        return embedding, cat_offsets, cell_indices, groups_list

    @staticmethod
    def _parse_contrasts(adata: AnnData, contrasts) -> tuple[str, list[str]]:
        """Validate a contrasts DataFrame and decompose its columns.

        Returns ``(groupby, split_by)`` per the layout enforced by
        :meth:`Distance.validate_contrasts` — first column is the groupby,
        ``"reference"`` is reserved, and every remaining column is a
        stratification filter.
        """
        from rapids_singlecell.pertpy_gpu._distance import Distance

        Distance.validate_contrasts(adata, contrasts)
        groupby = contrasts.columns[0]
        split_by = [c for c in contrasts.columns if c not in (groupby, "reference")]
        return groupby, split_by

    def _resolve_onesided_inputs(
        self,
        adata: AnnData,
        groupby: str,
        selected_group: str | Sequence[str],
        groups: Sequence[str] | None,
    ) -> tuple[list[str], bool, list[str] | None]:
        """Validate `selected_group`, normalize to a list, compute `needed` groups.

        Asserts that ``groupby`` is categorical and that every entry in
        ``selected_group`` is one of its categories with at least one cell.

        Returns
        -------
        selected_groups
            ``selected_group`` normalized to ``list[str]``.
        single_control
            ``True`` if ``selected_group`` was a single string (caller uses
            this to decide whether to return a Series or a DataFrame).
        needed
            Union of ``groups`` and ``selected_groups`` when ``groups`` was
            given; ``None`` otherwise (= use all categories).
        """
        _assert_categorical_obs(adata, key=groupby)

        single_control = isinstance(selected_group, str)
        selected_groups = [selected_group] if single_control else list(selected_group)

        missing = set(selected_groups) - set(adata.obs[groupby].cat.categories.values)
        if missing:
            raise ValueError(
                f"Selected groups {missing} not found in groupby '{groupby}'"
            )

        empty = set(selected_groups) - set(
            adata.obs[groupby].cat.remove_unused_categories().cat.categories.values
        )
        if empty:
            raise ValueError(f"No cells found for selected groups: {sorted(empty)}")

        needed = None
        if groups is not None:
            needed = list(set(groups) | set(selected_groups))

        return selected_groups, single_control, needed

    @abstractmethod
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
    ):
        """
        Compute pairwise distances between all cell groups.

        Parameters
        ----------
        adata
            Annotated data matrix
        groupby
            Key in adata.obs for grouping cells
        groups
            Specific groups to compute (if None, use all)
        bootstrap
            Whether to compute bootstrap variance estimates
        n_bootstrap
            Number of bootstrap iterations (if bootstrap=True)
        random_state
            Random seed for reproducibility
        multi_gpu
            GPU selection:
            - None: Use all GPUs if metric supports it, else GPU 0 (default)
            - True: Use all available GPUs
            - False: Use only GPU 0
            - list[int]: Use specific GPU IDs (e.g., [0, 2])
            - str: Comma-separated GPU IDs (e.g., "0,2")

        Returns
        -------
        result
            Result object containing distances and optional variance information.
        """
        ...

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
    ):
        """
        Compute distances from selected reference group(s) to all other groups.

        Parameters
        ----------
        adata
            Annotated data matrix
        groupby
            Key in adata.obs for grouping cells
        selected_group
            Reference group(s) to compute distances from. Can be a single
            group name or a sequence of group names.
        groups
            Specific groups to compute distances to (if None, use all)
        bootstrap
            Whether to compute bootstrap variance estimates
        n_bootstrap
            Number of bootstrap iterations (if bootstrap=True)
        random_state
            Random seed for reproducibility
        multi_gpu
            GPU selection:
            - None: Use all GPUs if metric supports it, else GPU 0 (default)
            - True: Use all available GPUs
            - False: Use only GPU 0
            - list[int]: Use specific GPU IDs (e.g., [0, 2])
            - str: Comma-separated GPU IDs (e.g., "0,2")

        Returns
        -------
        distances
            DataFrame with distances from selected_group(s) to other groups.
            If bootstrap=True, returns tuple of (distances, distances_var).
        """
        raise NotImplementedError(
            f"{self.__class__.__name__} does not implement onesided_distances"
        )

    def bootstrap(
        self,
        adata: AnnData,
        groupby: str,
        group_a: str,
        group_b: str,
        *,
        n_bootstrap: int = 100,
        random_state: int = 0,
        multi_gpu: bool | list[int] | str | None = None,
    ):
        """
        Compute bootstrap mean and variance for distance between two specific groups.

        Parameters
        ----------
        adata
            Annotated data matrix
        groupby
            Key in adata.obs for grouping cells
        group_a
            First group name
        group_b
            Second group name
        n_bootstrap
            Number of bootstrap iterations
        random_state
            Random seed for reproducibility
        multi_gpu
            GPU selection:
            - None: Use all GPUs if metric supports it, else GPU 0 (default)
            - True: Use all available GPUs
            - False: Use only GPU 0
            - list[int]: Use specific GPU IDs (e.g., [0, 2])
            - str: Comma-separated GPU IDs (e.g., "0,2")

        Returns
        -------
        mean
            Bootstrap mean distance
        variance
            Bootstrap variance
        """
        raise NotImplementedError(
            f"{self.__class__.__name__} does not implement bootstrap"
        )

    def contrast_distances(
        self,
        adata: AnnData,
        contrasts,
        *,
        multi_gpu: bool | list[int] | str | None = None,
    ):
        """
        Compute distances for contrasts.

        Parameters
        ----------
        adata
            Annotated data matrix
        contrasts
            DataFrame with a groupby column, a ``reference`` column,
            and optional split columns.
        multi_gpu
            GPU selection:
            - None: Use all GPUs if metric supports it, else GPU 0 (default)
            - True: Use all available GPUs
            - False: Use only GPU 0
            - list[int]: Use specific GPU IDs (e.g., [0, 2])
            - str: Comma-separated GPU IDs (e.g., "0,2")

        Returns
        -------
        pd.DataFrame
            Copy of the input DataFrame with an added distance column.
        """
        raise NotImplementedError(
            f"{self.__class__.__name__} does not implement contrast_distances"
        )
