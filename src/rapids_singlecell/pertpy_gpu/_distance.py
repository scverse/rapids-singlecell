from __future__ import annotations

import warnings
from typing import TYPE_CHECKING, Literal, NamedTuple

if TYPE_CHECKING:
    from collections.abc import Sequence

    import cupy as cp
    import numpy as np
    import pandas as pd
    from anndata import AnnData


class MeanVar(NamedTuple):
    """Result of bootstrap computation containing mean and variance."""

    mean: float
    variance: float


Metric = Literal[
    "edistance",
    "euclidean",
    "root_mean_squared_error",
    "mse",
    "mean_absolute_error",
    "pearson_distance",
    "cosine_distance",
    "r2_distance",
    "wasserstein",
]

SUPPORTED_METRICS = [
    "edistance",
    "euclidean",
    "root_mean_squared_error",
    "mse",
    "mean_absolute_error",
    "pearson_distance",
    "cosine_distance",
    "r2_distance",
    "wasserstein",
]


class Distance:
    """
    GPU-accelerated distance computation between groups of cells.

    API compatible with pertpy's Distance class.

    Currently supported metrics:

    - ``"edistance"``: Energy distance (default).
        Twice the mean pairwise distance between cells of two groups minus
        the mean pairwise distance between cells within each group. See
        `Peidli et al. (2023) <https://doi.org/10.1101/2022.08.20.504663>`__.
    - ``"euclidean"`` and ``"root_mean_squared_error"``: Euclidean distance
        between group mean vectors.
    - ``"mse"``: Mean squared distance between group mean vectors.
    - ``"mean_absolute_error"``: Mean absolute distance between group mean
        vectors.
    - ``"pearson_distance"``: Pearson distance between group mean vectors.
    - ``"cosine_distance"``: Cosine distance between group mean vectors.
    - ``"r2_distance"``: One minus the coefficient of determination between
        group mean vectors.
    - ``"wasserstein"``: Entropy-regularized 2-Wasserstein via Sinkhorn.
        Squared-Euclidean ground cost; per-pair auto-epsilon defaulting to
        ``0.05 * std(C)`` to match OTT-JAX. Returns OTT's
        ``reg_ot_cost`` value.

    Parameters
    ----------
    metric
        Distance metric to use.
    layer_key
        Key in adata.layers for cell data. Mutually exclusive with ``obsm_key``.
    obsm_key
        Key in adata.obsm for embeddings. Mutually exclusive with ``layer_key``.
        Defaults to ``"X_pca"`` if neither is specified.

    Notes
    -----
    The ``edistance`` bootstrap implementation differs from pertpy: rather
    than precomputing an n×n cell distance matrix and sampling from it, this
    implementation resamples cells and recomputes distances from scratch each
    iteration. This scales better for large datasets (O(n) vs O(n²) memory)
    and leverages multi-GPU parallelism for each bootstrap iteration.

    ``"edistance"`` and ``"wasserstein"`` use multi-GPU (pairs are split across
    devices). Pseudobulk metrics aggregate cells into K group-mean vectors
    before computing distances, and the resulting K×K kernel is cheap enough on
    a single GPU that distributing it is not worth the cost. Passing
    ``multi_gpu=True`` for those metrics falls back to a single device with a
    warning.

    Examples
    --------
    >>> import rapids_singlecell as rsc
    >>> distance = rsc.ptg.Distance(metric='edistance')
    >>> result = distance.pairwise(adata, groupby='perturbation')

    >>> # Direct computation on arrays
    >>> d = distance(X, Y)
    """

    def __init__(
        self,
        metric: Metric = "edistance",
        layer_key: str | None = None,
        obsm_key: str | None = None,
        **kwargs,
    ):
        """Initialize Distance calculator with specified metric.

        Extra keyword arguments are forwarded to the selected metric's
        constructor, exposing metric-specific options (e.g. ``relaxation`` for
        ``metric="wasserstein"``). Passing an option the chosen metric does not
        accept raises ``TypeError``.
        """
        if layer_key is not None and obsm_key is not None:
            raise ValueError(
                "Cannot use 'layer_key' and 'obsm_key' at the same time.\n"
                "Please provide only one of the two keys."
            )
        if layer_key is None and obsm_key is None:
            obsm_key = "X_pca"

        self.metric = metric
        self.layer_key = layer_key
        self.obsm_key = obsm_key
        self._metric_kwargs = kwargs
        self._metric_impl = None
        self._initialize_metric()

    def _initialize_metric(self):
        """Initialize the metric implementation based on the metric type."""
        if self.metric == "edistance":
            from rapids_singlecell.pertpy_gpu._metrics._edistance import (
                EDistanceMetric,
            )

            self._metric_impl = EDistanceMetric(
                layer_key=self.layer_key,
                obsm_key=self.obsm_key,
                **self._metric_kwargs,
            )
        elif self.metric == "wasserstein":
            from rapids_singlecell.pertpy_gpu._metrics._wasserstein import (
                WassersteinMetric,
            )

            self._metric_impl = WassersteinMetric(
                layer_key=self.layer_key,
                obsm_key=self.obsm_key,
                **self._metric_kwargs,
            )
        elif self.metric in SUPPORTED_METRICS:
            from rapids_singlecell.pertpy_gpu._metrics._pseudobulk import (
                PSEUDOBULK_METRICS,
            )

            self._metric_impl = PSEUDOBULK_METRICS[self.metric](
                metric_name=self.metric,
                layer_key=self.layer_key,
                obsm_key=self.obsm_key,
                **self._metric_kwargs,
            )
        else:
            raise ValueError(
                f"Unknown metric: {self.metric}. Supported metrics: {SUPPORTED_METRICS}"
            )

    def _check_multi_gpu_support(
        self, *, multi_gpu: bool | list[int] | str | None
    ) -> bool | list[int] | str:
        """Check if metric supports multi-GPU and resolve None default.

        Parameters
        ----------
        multi_gpu
            The multi_gpu parameter passed by the user. None means use default
            (True if supported, False otherwise).

        Returns
        -------
        multi_gpu
            Returns False if metric doesn't support multi-GPU, otherwise
            returns the resolved value (True if None was passed and supported).
        """
        # If None, default to True if supported, False otherwise
        if multi_gpu is None:
            return self._metric_impl.supports_multi_gpu

        if not self._metric_impl.supports_multi_gpu:
            # Check if user explicitly requested multi-GPU
            uses_multi_gpu = (
                multi_gpu is True
                or (isinstance(multi_gpu, list) and len(multi_gpu) > 1)
                or (isinstance(multi_gpu, str) and "," in multi_gpu)
            )
            if uses_multi_gpu:
                warnings.warn(
                    f"Metric '{self.metric}' does not support multi-GPU. "
                    "Falling back to single GPU (device 0).",
                    UserWarning,
                    stacklevel=3,
                )
            return False
        return multi_gpu

    def __call__(
        self,
        X: np.ndarray | cp.ndarray,
        Y: np.ndarray | cp.ndarray,
    ) -> float:
        """
        Compute distance between two cell groups directly from arrays.

        This provides pertpy-compatible API for direct distance computation.

        Parameters
        ----------
        X
            First array of shape (n_samples_x, n_features)
        Y
            Second array of shape (n_samples_y, n_features)

        Returns
        -------
        float
            Distance between X and Y

        Examples
        --------
        >>> distance = Distance(metric='edistance')
        >>> X = adata.obsm["X_pca"][adata.obs["group"] == "A"]
        >>> Y = adata.obsm["X_pca"][adata.obs["group"] == "B"]
        >>> d = distance(X, Y)
        """
        if not hasattr(self._metric_impl, "compute_distance"):
            raise NotImplementedError(
                f"Metric '{self.metric}' does not support direct distance computation"
            )
        return self._metric_impl.compute_distance(X, Y)

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
            DataFrame with pairwise distances. If bootstrap=True, returns
            tuple of (distances, distances_var) DataFrames.

        Examples
        --------
        >>> distance = Distance(metric='edistance')
        >>> result = distance.pairwise(adata, groupby='condition')
        """
        multi_gpu = self._check_multi_gpu_support(multi_gpu=multi_gpu)
        return self._metric_impl.pairwise(
            adata=adata,
            groupby=groupby,
            groups=groups,
            bootstrap=bootstrap,
            n_bootstrap=n_bootstrap,
            random_state=random_state,
            multi_gpu=multi_gpu,
        )

    def onesided_distances(
        self,
        adata: AnnData,
        groupby: str,
        selected_group: Sequence[str] | str,
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
        """
        Compute distances from one selected group to all other groups.

        Parameters
        ----------
        adata
            Annotated data matrix
        groupby
            Key in adata.obs for grouping cells
        selected_group
            Reference group to compute distances from
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
            Series containing distances from selected_group to all other groups.
            If bootstrap=True, returns tuple of (distances, distances_var).

        Examples
        --------
        >>> distance = Distance(metric='edistance')
        >>> distances = distance.onesided_distances(
        ...     adata, groupby='condition', selected_group='control'
        ... )
        """
        if not hasattr(self._metric_impl, "onesided_distances"):
            raise NotImplementedError(
                f"Metric '{self.metric}' does not support onesided_distances"
            )
        multi_gpu = self._check_multi_gpu_support(multi_gpu=multi_gpu)
        return self._metric_impl.onesided_distances(
            adata=adata,
            groupby=groupby,
            selected_group=selected_group,
            groups=groups,
            bootstrap=bootstrap,
            n_bootstrap=n_bootstrap,
            random_state=random_state,
            multi_gpu=multi_gpu,
        )

    def bootstrap(
        self,
        X: np.ndarray | cp.ndarray,
        Y: np.ndarray | cp.ndarray,
        *,
        n_bootstrap: int = 100,
        random_state: int = 0,
    ) -> MeanVar:
        """
        Compute bootstrap mean and variance for distance between two arrays.

        This provides pertpy-compatible API for bootstrap computation directly
        on arrays without requiring an AnnData object.

        Parameters
        ----------
        X
            First array of shape (n_samples_x, n_features)
        Y
            Second array of shape (n_samples_y, n_features)
        n_bootstrap
            Number of bootstrap iterations
        random_state
            Random seed for reproducibility

        Returns
        -------
        result
            Named tuple containing mean and variance of bootstrapped distances

        Examples
        --------
        >>> distance = Distance(metric='edistance')
        >>> X = adata.obsm["X_pca"][adata.obs["group"] == "A"]
        >>> Y = adata.obsm["X_pca"][adata.obs["group"] == "B"]
        >>> result = distance.bootstrap(X, Y, n_bootstrap=100)
        >>> print(f"Distance: {result.mean:.3f} ± {result.variance**0.5:.3f}")
        """
        if not hasattr(self._metric_impl, "bootstrap_arrays"):
            raise NotImplementedError(
                f"Metric '{self.metric}' does not support bootstrap"
            )
        mean, variance = self._metric_impl.bootstrap_arrays(
            X=X,
            Y=Y,
            n_bootstrap=n_bootstrap,
            random_state=random_state,
        )
        return MeanVar(mean=mean, variance=variance)

    @staticmethod
    def create_contrasts(
        adata: AnnData,
        groupby: str,
        selected_group: str | Sequence[str],
        *,
        groups: Sequence[str] | None = None,
        split_by: str | Sequence[str] | None = None,
    ) -> pd.DataFrame:
        """
        Build a contrasts DataFrame for use with :meth:`contrast_distances`.

        Each row represents one contrast: comparing a group against the
        reference, optionally within each level of ``split_by`` columns.
        The resulting DataFrame can be filtered or modified before passing
        to :meth:`contrast_distances`.

        The output layout is:

        - **First column** (``groupby``): the target values to compare
        - **``reference`` column**: the control value in the groupby column
        - **Remaining columns** (``split_by``): stratification filters

        Parameters
        ----------
        adata
            Annotated data matrix
        groupby
            Column in ``adata.obs`` whose levels are compared against
            ``selected_group``
        selected_group
            The reference (control) value(s) in the ``groupby`` column.
            When a sequence is passed, each target is compared against
            every reference, producing one row per (target, reference)
            combination.
        groups
            Specific groups to include. If None, all non-reference groups
            are included.
        split_by
            Column(s) in ``adata.obs`` to stratify by. If provided,
            contrasts are computed within each unique combination of
            these columns. Only combinations where the reference group
            exists are included.

        Returns
        -------
        pd.DataFrame
            One row per contrast. First column is ``groupby``, then
            ``reference``, then any ``split_by`` columns.

        Examples
        --------
        >>> # All targets vs control, ignoring celltype
        >>> contrasts = Distance.create_contrasts(
        ...     adata, groupby="target_gene", selected_group="Non_target"
        ... )

        >>> # Multiple references
        >>> contrasts = Distance.create_contrasts(
        ...     adata, groupby="target_gene",
        ...     selected_group=["Non_target", "Scramble"],
        ... )

        >>> # Stratified by celltype
        >>> contrasts = Distance.create_contrasts(
        ...     adata, groupby="target_gene", selected_group="Non_target",
        ...     split_by="group_name",
        ... )

        >>> # Filter before computing
        >>> contrasts = contrasts[contrasts["group_name"] != "rare_type"]
        >>> result = distance.contrast_distances(adata, contrasts=contrasts)

        >>> # Manual construction (no helper needed)
        >>> import pandas as pd
        >>> contrasts = pd.DataFrame({
        ...     "target_gene": ["Irf7", "Ski"],
        ...     "reference": ["Non_target", "Non_target"],
        ...     "group_name": ["CD4", "CD4"],
        ... })
        """
        import pandas as pd

        # Normalize to list
        if isinstance(selected_group, str):
            selected_groups = [selected_group]
        else:
            selected_groups = list(selected_group)

        obs_values = set(adata.obs[groupby].values)
        for sg in selected_groups:
            if sg not in obs_values:
                raise ValueError(f"Reference '{sg}' not found in column '{groupby}'")

        if split_by is None:
            split_cols: list[str] = []
        elif isinstance(split_by, str):
            split_cols = [split_by]
        else:
            split_cols = list(split_by)

        allowed_groups = set(groups) if groups is not None else None
        selected_set = set(selected_groups)
        all_cols = [groupby, *split_cols]

        parts: list[pd.DataFrame] = []
        for sg in selected_groups:
            if split_cols:
                existing = adata.obs[all_cols].drop_duplicates().reset_index(drop=True)

                ref_rows = existing[existing[groupby] == sg]
                if len(ref_rows) == 0:
                    continue
                ref_splits = ref_rows[split_cols]
                targets = existing[~existing[groupby].isin(selected_set)]
                if allowed_groups is not None:
                    targets = targets[targets[groupby].isin(allowed_groups)]
                matched = targets.merge(ref_splits, on=split_cols, how="inner")
                if len(matched) == 0:
                    continue
                matched = matched[all_cols].copy()
            else:
                target_vals = [
                    t
                    for t in adata.obs[groupby].unique()
                    if t not in selected_set
                    and (allowed_groups is None or t in allowed_groups)
                ]
                if not target_vals:
                    continue
                matched = pd.DataFrame({groupby: target_vals})

            matched.insert(1, "reference", sg)
            parts.append(matched)

        if not parts:
            cols = [groupby, "reference", *split_cols]
            return pd.DataFrame(columns=cols)

        df = pd.concat(parts, ignore_index=True)
        sort_cols = ["reference", *split_cols, groupby]
        df = df.sort_values(sort_cols).reset_index(drop=True)

        return df

    @staticmethod
    def validate_contrasts(
        adata: AnnData,
        contrasts: pd.DataFrame,
    ) -> None:
        """
        Validate a contrasts DataFrame against an AnnData object.

        Expects the DataFrame layout produced by :meth:`create_contrasts`:
        first column is the groupby column, ``reference`` column contains
        the control value, remaining columns are split-by filters.

        Parameters
        ----------
        adata
            Annotated data matrix
        contrasts
            DataFrame to validate

        Raises
        ------
        ValueError
            If validation fails.
        """
        if "reference" not in contrasts.columns:
            raise ValueError(
                "Contrasts DataFrame must have a 'reference' column. "
                "Use Distance.create_contrasts() or add it manually."
            )

        groupby = contrasts.columns[0]
        if groupby == "reference":
            raise ValueError(
                "First column cannot be 'reference'. The first column "
                "must be the groupby column."
            )

        split_by = [c for c in contrasts.columns if c not in (groupby, "reference")]

        # Check columns exist in adata.obs
        for col in [groupby, *split_by]:
            if col not in adata.obs.columns:
                raise ValueError(
                    f"Column '{col}' not found in adata.obs. "
                    f"Available columns: {list(adata.obs.columns)}"
                )

        # Check reference values exist in adata
        obs_groupby_values = set(adata.obs[groupby].unique())
        ref_values = set(contrasts["reference"].unique())
        missing_refs = ref_values - obs_groupby_values
        if missing_refs:
            raise ValueError(
                f"Reference values not found in adata.obs['{groupby}']: {missing_refs}"
            )

        # Check target values exist in adata
        target_values = set(contrasts[groupby].unique())
        missing_targets = target_values - obs_groupby_values
        if missing_targets:
            raise ValueError(
                f"Groups not found in adata.obs['{groupby}']: {missing_targets}"
            )

        # Check split_by values exist in adata
        for col in split_by:
            obs_vals = set(adata.obs[col].unique())
            contrast_vals = set(contrasts[col].unique())
            missing_split = contrast_vals - obs_vals
            if missing_split:
                raise ValueError(
                    f"Values not found in adata.obs['{col}']: {missing_split}"
                )

    def contrast_distances(
        self,
        adata: AnnData,
        contrasts: pd.DataFrame,
        *,
        multi_gpu: bool | list[int] | str | None = None,
    ) -> pd.DataFrame:
        """
        Compute distances for contrasts.

        Accepts a DataFrame (from :meth:`create_contrasts` or constructed
        manually) with the following layout:

        - **First column**: the groupby column (target values to compare)
        - **``reference`` column**: the control value in the groupby column
        - **Other columns**: split-by filters (e.g., cell type)

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

        Examples
        --------
        >>> distance = Distance(metric='edistance')

        >>> # Using create_contrasts helper
        >>> contrasts = Distance.create_contrasts(
        ...     adata, groupby="target_gene", selected_group="Non_target",
        ...     split_by="group_name",
        ... )
        >>> result = distance.contrast_distances(adata, contrasts=contrasts)

        >>> # Manual DataFrame construction
        >>> import pandas as pd
        >>> contrasts = pd.DataFrame({
        ...     "target_gene": ["Irf7", "Ski"],
        ...     "reference": ["Non_target", "Non_target"],
        ...     "group_name": ["CD4", "CD4"],
        ... })
        >>> result = distance.contrast_distances(adata, contrasts)
        """
        if not hasattr(self._metric_impl, "contrast_distances"):
            raise NotImplementedError(
                f"Metric '{self.metric}' does not support contrast_distances"
            )
        multi_gpu = self._check_multi_gpu_support(multi_gpu=multi_gpu)
        return self._metric_impl.contrast_distances(
            adata=adata,
            contrasts=contrasts,
            multi_gpu=multi_gpu,
        )

    def __repr__(self) -> str:
        """String representation of Distance object."""
        if self.layer_key is not None:
            return f"Distance(metric='{self.metric}', layer_key='{self.layer_key}')"
        return f"Distance(metric='{self.metric}', obsm_key='{self.obsm_key}')"
