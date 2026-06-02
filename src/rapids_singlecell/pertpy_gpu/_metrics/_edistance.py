from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Sequence

import cupy as cp
import numpy as np
import pandas as pd

from rapids_singlecell._cuda import _edistance_cuda as _ed
from rapids_singlecell._utils import (
    _calculate_blocks_per_pair,
    _split_pairs,
)
from rapids_singlecell.squidpy_gpu._utils import _assert_categorical_obs

from ._base_metric import BaseMetric, parse_device_ids

if TYPE_CHECKING:
    from anndata import AnnData


class EDistanceMetric(BaseMetric):
    """
    GPU-accelerated Energy Distance metric.

    Energy distance is a statistical distance between probability distributions
    that generalizes the Euclidean distance to distributions. It is particularly
    useful for comparing groups of cells in high-dimensional spaces.

    Parameters
    ----------
    layer_key
        Key in adata.layers for cell data. Mutually exclusive with obsm_key.
    obsm_key
        Key in adata.obsm for embeddings (default: 'X_pca')

    References
    ----------
    Székely, G. J., & Rizzo, M. L. (2013).
    Energy statistics: A class of statistics based on distances.
    Journal of Statistical Planning and Inference, 143(8), 1249-1272.
    """

    supports_multi_gpu: bool = True

    def __init__(
        self,
        layer_key: str | None = None,
        obsm_key: str | None = "X_pca",
    ):
        """Initialize energy distance metric."""
        super().__init__(layer_key=layer_key, obsm_key=obsm_key)

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
        """
        Compute pairwise energy distances between all cell groups.

        Returns a DataFrame containing pairwise distances. When bootstrap=True,
        returns a tuple of (distances, distances_var) DataFrames.

        The distances DataFrame contains:
        distances[a,b] = 2*d[a,b] - d[a] - d[b]

        When bootstrap=True, distances_var contains:
        distances_var[a,b] = 4*d_var[a,b] + d_var[a] + d_var[b]

        Parameters
        ----------
        adata
            Annotated data matrix
        groupby
            Key in adata.obs for grouping
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
        """
        _assert_categorical_obs(adata, key=groupby)

        embedding, cat_offsets, cell_indices, groups_list = self._subset_to_groups(
            adata, groupby, groups
        )
        k = len(groups_list)

        if not bootstrap:
            return self._prepare_edistance_df(
                embedding=embedding,
                cat_offsets=cat_offsets,
                cell_indices=cell_indices,
                k=k,
                groups_list=groups_list,
                groupby=groupby,
                multi_gpu=multi_gpu,
            )

        return self._prepare_edistance_df_bootstrap(
            embedding=embedding,
            cat_offsets=cat_offsets,
            cell_indices=cell_indices,
            k=k,
            groups_list=groups_list,
            groupby=groupby,
            n_bootstrap=n_bootstrap,
            random_state=random_state,
            multi_gpu=multi_gpu,
        )

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
        """
        Compute energy distances from selected reference group(s) to all other groups.

        Parameters
        ----------
        adata
            Annotated data matrix
        groupby
            Key in adata.obs for grouping cells
        selected_group
            Reference group(s) to compute distances from. Can be a single
            group name or a sequence of group names for multiple controls.
            When a single string is passed, returns a Series. When a sequence
            is passed, returns a DataFrame with one column per control.
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
            Series (single control) or DataFrame (multiple controls).
            If bootstrap=True, returns tuple of (distances, distances_var).
        """
        selected_groups, single_control, needed = self._resolve_onesided_inputs(
            adata, groupby, selected_group, groups
        )

        embedding, cat_offsets, cell_indices, groups_list = self._subset_to_groups(
            adata, groupby, needed
        )
        k = len(groups_list)
        group_map = {v: i for i, v in enumerate(groups_list)}
        selected_indices = [group_map[sg] for sg in selected_groups]

        device_ids = parse_device_ids(multi_gpu=multi_gpu)

        if bootstrap:
            cross_mean, diag_mean, cross_var, diag_var = self._onesided_means_bootstrap(
                embedding=embedding,
                cat_offsets=cat_offsets,
                cell_indices=cell_indices,
                k=k,
                selected_indices=selected_indices,
                n_bootstrap=n_bootstrap,
                random_state=random_state,
                device_ids=device_ids,
            )

            # Compute energy distances for each control:
            # e[s,b] = 2*d[s,b] - d[s,s] - d[b,b]
            ed_cols = {}
            var_cols = {}
            for i, (sg, si) in enumerate(zip(selected_groups, selected_indices)):
                ed_row = 2 * cross_mean[i, :] - diag_mean[si] - diag_mean
                ed_row[si] = 0.0
                ed_cols[sg] = ed_row.get()

                var_row = 4 * cross_var[i, :] + diag_var[si] + diag_var
                var_row[si] = 0.0
                var_cols[sg] = var_row.get()

            distances = pd.DataFrame(ed_cols, index=groups_list)
            distances.index.name = groupby
            distances.columns.name = "selected_group"

            variances = pd.DataFrame(var_cols, index=groups_list)
            variances.index.name = groupby
            variances.columns.name = "selected_group"

            if single_control:
                sg = selected_groups[0]
                return distances[sg], variances[sg]
            return distances, variances

        # Non-bootstrap path
        cross_means, diag_means = self._onesided_means(
            embedding,
            cat_offsets,
            cell_indices,
            k,
            selected_indices=selected_indices,
            device_ids=device_ids,
        )

        # Compute energy distances for each control:
        # e[s,b] = 2*d[s,b] - d[s,s] - d[b,b]
        # cross_means[i, j] = mean dist from selected[i] to group j
        # diag_means[j] = mean within-group dist for group j
        ed_cols = {}
        for i, (sg, si) in enumerate(zip(selected_groups, selected_indices)):
            ed_row = 2 * cross_means[i, :] - diag_means[si] - diag_means
            ed_row[si] = 0.0
            ed_cols[sg] = ed_row.get()

        df = pd.DataFrame(ed_cols, index=groups_list)
        df.index.name = groupby
        df.columns.name = "selected_group"

        if single_control:
            return df[selected_groups[0]]
        return df

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
    ) -> tuple[float, float]:
        """
        Compute bootstrap mean and variance for energy distance between two groups.

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
        # Compute pairwise with bootstrap
        df, df_var = self.pairwise(
            adata=adata,
            groupby=groupby,
            groups=[group_a, group_b],
            bootstrap=True,
            n_bootstrap=n_bootstrap,
            random_state=random_state,
            multi_gpu=multi_gpu,
        )

        mean = df.loc[group_a, group_b]
        variance = df_var.loc[group_a, group_b]

        return float(mean), float(variance)

    def contrast_distances(
        self,
        adata: AnnData,
        contrasts: pd.DataFrame,
        *,
        multi_gpu: bool | list[int] | str | None = None,
    ) -> pd.DataFrame:
        """
        Compute energy distances for contrasts.

        Parameters
        ----------
        adata
            Annotated data matrix
        contrasts
            DataFrame with a groupby column (first), a ``reference``
            column, and optional split columns.
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
            Copy of the input DataFrame with an added ``edistance`` column.
        """
        groupby, split_by = self._parse_contrasts(adata, contrasts)

        embedding_raw = self._get_embedding(adata)
        device_ids = parse_device_ids(multi_gpu=multi_gpu)

        all_cols = [groupby, *split_by]

        # Single groupby to get cell indices for all combinations
        grouped = adata.obs.groupby(all_cols, observed=True)
        group_indices = grouped.indices

        # Build conditions using numpy arrays (avoid per-row _asdict)
        target_vals = contrasts[groupby].values
        ref_vals = contrasts["reference"].values
        split_arrays = [contrasts[col].values for col in split_by]

        cond_to_idx: dict[tuple, int] = {}
        contrast_pairs: list[tuple[int, int]] = []

        for i in range(len(contrasts)):
            if split_by:
                split_vals = tuple(arr[i] for arr in split_arrays)
                target_key = (target_vals[i], *split_vals)
                ref_key = (ref_vals[i], *split_vals)
            else:
                target_key = (target_vals[i],)
                ref_key = (ref_vals[i],)

            for key in (target_key, ref_key):
                if key not in cond_to_idx:
                    cond_to_idx[key] = len(cond_to_idx)

            contrast_pairs.append((cond_to_idx[target_key], cond_to_idx[ref_key]))
        k = len(cond_to_idx)
        # Look up cell indices from the groupby
        group_cells: list[np.ndarray] = [None] * k  # type: ignore[list-item]
        for key, idx in cond_to_idx.items():
            lookup_key = key[0] if len(key) == 1 else key
            cell_idx = group_indices.get(lookup_key)
            group_cells[idx] = (
                cell_idx if cell_idx is not None else np.array([], dtype=np.intp)
            )
        # Build cat_offsets and cell_indices, subsetting the embedding
        # to only the referenced cells for memory efficiency
        offsets = [0]
        all_cell_idx = []
        for cells in group_cells:
            all_cell_idx.append(cells)
            offsets.append(offsets[-1] + len(cells))

        cat_offsets = cp.array(offsets, dtype=cp.int32)
        original_indices = np.concatenate(all_cell_idx)

        # Subset before GPU transfer when not all cells are referenced
        if len(original_indices) < int(len(embedding_raw) * 0.7):
            embedding = cp.asarray(embedding_raw[original_indices])
            cell_indices = cp.arange(len(original_indices), dtype=cp.int32)
        else:
            embedding = cp.asarray(embedding_raw)
            cell_indices = cp.array(original_indices, dtype=cp.int32)

        group_sizes = cp.diff(cat_offsets).astype(cp.int64)
        group_sizes_cpu = group_sizes.get()
        # Build deduplicated pairs
        pair_to_flat: dict[tuple[int, int], int] = {}
        for idx_a, idx_b in contrast_pairs:
            cross = (min(idx_a, idx_b), max(idx_a, idx_b))
            if cross not in pair_to_flat:
                pair_to_flat[cross] = len(pair_to_flat)
            if group_sizes_cpu[idx_a] >= 2 and (idx_a, idx_a) not in pair_to_flat:
                pair_to_flat[(idx_a, idx_a)] = len(pair_to_flat)
            if group_sizes_cpu[idx_b] >= 2 and (idx_b, idx_b) not in pair_to_flat:
                pair_to_flat[(idx_b, idx_b)] = len(pair_to_flat)

        n_pairs = len(pair_to_flat)

        if n_pairs == 0:
            result = contrasts.copy()
            result["edistance"] = 0.0
            return result

        pairs = sorted(pair_to_flat.keys(), key=lambda p: pair_to_flat[p])
        pair_left = cp.array([p[0] for p in pairs], dtype=cp.int32)
        pair_right = cp.array([p[1] for p in pairs], dtype=cp.int32)

        flat_sums = self._launch_distance_kernel(
            embedding,
            cat_offsets,
            cell_indices,
            pair_left=pair_left,
            pair_right=pair_right,
            device_ids=device_ids,
        )

        # Vectorized normalization
        is_diag = pair_left == pair_right
        sizes_l = group_sizes[pair_left.astype(cp.intp)]
        sizes_r = group_sizes[pair_right.astype(cp.intp)]
        flat_norms = cp.where(
            is_diag,
            cp.maximum(sizes_l * (sizes_l - 1) // 2, 1),
            sizes_l * sizes_r,
        ).astype(embedding.dtype)
        flat_means = flat_sums / flat_norms
        flat_means_cpu = flat_means.get()

        # Extract edistances
        edistances = np.empty(len(contrast_pairs), dtype=np.float64)
        for i, (idx_a, idx_b) in enumerate(contrast_pairs):
            if idx_a == idx_b:
                edistances[i] = 0.0
                continue
            cross = (min(idx_a, idx_b), max(idx_a, idx_b))
            d_cross = flat_means_cpu[pair_to_flat[cross]]
            diag_a = pair_to_flat.get((idx_a, idx_a))
            d_aa = flat_means_cpu[diag_a] if diag_a is not None else 0.0
            diag_b = pair_to_flat.get((idx_b, idx_b))
            d_bb = flat_means_cpu[diag_b] if diag_b is not None else 0.0
            edistances[i] = 2 * d_cross - d_aa - d_bb

        result = contrasts.copy()
        result["edistance"] = edistances
        return result

    # Helper methods

    def compute_distance(
        self,
        X: np.ndarray | cp.ndarray,
        Y: np.ndarray | cp.ndarray,
    ) -> float:
        """
        Compute energy distance between two arrays directly.

        Parameters
        ----------
        X
            First array of shape (n_samples_x, n_features)
        Y
            Second array of shape (n_samples_y, n_features)

        Returns
        -------
        float
            Energy distance between X and Y
        """
        # Convert to cupy arrays, preserving dtype
        X_gpu = cp.asarray(X)
        Y_gpu = cp.asarray(Y)

        if len(X_gpu) == 0 or len(Y_gpu) == 0:
            raise ValueError("Neither X nor Y can be empty.")

        # Compute mean pairwise distances
        d_xy = self._mean_pairwise_distance(X_gpu, Y_gpu)
        d_xx = self._mean_pairwise_distance_within(X_gpu)
        d_yy = self._mean_pairwise_distance_within(Y_gpu)

        # Energy distance formula
        return float(2 * d_xy - d_xx - d_yy)

    def bootstrap_arrays(
        self,
        X: np.ndarray | cp.ndarray,
        Y: np.ndarray | cp.ndarray,
        *,
        n_bootstrap: int = 100,
        random_state: int = 0,
    ) -> tuple[float, float]:
        """
        Compute bootstrap mean and variance for energy distance between arrays.

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
        mean
            Bootstrap mean distance
        variance
            Bootstrap variance
        """
        # Convert to cupy arrays, preserving dtype
        X_gpu = cp.asarray(X)
        Y_gpu = cp.asarray(Y)

        if len(X_gpu) == 0 or len(Y_gpu) == 0:
            raise ValueError("Neither X nor Y can be empty.")

        rng = np.random.default_rng(random_state)

        distances = []
        for _ in range(n_bootstrap):
            # Bootstrap sample indices
            X_idx = rng.choice(len(X_gpu), size=len(X_gpu), replace=True)
            Y_idx = rng.choice(len(Y_gpu), size=len(Y_gpu), replace=True)

            X_boot = X_gpu[X_idx]
            Y_boot = Y_gpu[Y_idx]

            # Compute distance for this bootstrap sample
            d_xy = self._mean_pairwise_distance(X_boot, Y_boot)
            d_xx = self._mean_pairwise_distance_within(X_boot)
            d_yy = self._mean_pairwise_distance_within(Y_boot)

            distance = 2 * d_xy - d_xx - d_yy
            distances.append(float(distance))

        return float(np.mean(distances)), float(np.var(distances))

    def _mean_pairwise_distance(
        self,
        X: cp.ndarray,
        Y: cp.ndarray,
    ) -> float:
        """Compute mean pairwise Euclidean distance between X and Y."""
        # Use cdist-like computation on GPU
        # X: (n, d), Y: (m, d) -> dist[i,j] = ||X[i] - Y[j]||

        # Compute squared distances using broadcasting
        # ||x - y||^2 = ||x||^2 + ||y||^2 - 2*x.y
        X_sq = cp.sum(X * X, axis=1, keepdims=True)  # (n, 1)
        Y_sq = cp.sum(Y * Y, axis=1, keepdims=True)  # (m, 1)
        XY = X @ Y.T  # (n, m)

        dist_sq = X_sq + Y_sq.T - 2 * XY
        dist_sq = cp.maximum(dist_sq, 0)  # Numerical stability
        distances = cp.sqrt(dist_sq)

        return float(cp.mean(distances))

    def _mean_pairwise_distance_within(self, X: cp.ndarray) -> float:
        """Compute mean pairwise Euclidean distance within X (upper triangle only)."""
        n = len(X)
        if n < 2:
            return 0.0

        # Compute all pairwise distances
        X_sq = cp.sum(X * X, axis=1, keepdims=True)
        dist_sq = X_sq + X_sq.T - 2 * (X @ X.T)
        dist_sq = cp.maximum(dist_sq, 0)
        distances = cp.sqrt(dist_sq)

        # Get upper triangle (excluding diagonal)
        triu_indices = cp.triu_indices(n, k=1)
        upper_distances = distances[triu_indices]

        return float(cp.mean(upper_distances))

    # Internal methods

    def _launch_distance_kernel(
        self,
        embedding: cp.ndarray,
        cat_offsets: cp.ndarray,
        cell_indices: cp.ndarray,
        *,
        pair_left: cp.ndarray,
        pair_right: cp.ndarray,
        device_ids: list[int],
    ) -> cp.ndarray:
        """Launch the edistance kernel across GPUs and return raw flat sums.

        This is the shared kernel launch logic used by all distance methods.
        Output is always a flat array of shape (n_pairs,) indexed by pair_id.

        Parameters
        ----------
        embedding
            Cell embeddings on GPU 0
        cat_offsets
            Category offsets on GPU 0
        cell_indices
            Cell indices on GPU 0
        pair_left
            Left group indices for each pair
        pair_right
            Right group indices for each pair
        device_ids
            List of GPU device IDs to use

        Returns
        -------
        cp.ndarray
            Raw distance sums of shape (n_pairs,), NOT normalized.
        """
        n_devices = len(device_ids)
        n_total_pairs = len(pair_left)
        _, n_features = embedding.shape
        group_sizes = cp.diff(cat_offsets).astype(cp.int64)

        # Split pairs across devices with load balancing
        pair_chunks = _split_pairs(pair_left, pair_right, n_devices, group_sizes)

        # Track which flat indices each device handles
        chunk_offsets = []
        offset = 0
        for chunk_left, _ in pair_chunks:
            chunk_offsets.append(offset)
            offset += len(chunk_left)

        # Phase 1: Create streams and start async data transfer to all devices
        streams = {}
        device_data = []

        for i, device_id in enumerate(device_ids):
            chunk_left, chunk_right = pair_chunks[i]
            if len(chunk_left) == 0:
                device_data.append(None)
                continue

            n_chunk_pairs = len(chunk_left)
            with cp.cuda.Device(device_id):
                streams[device_id] = cp.cuda.Stream(non_blocking=True)

                with streams[device_id]:
                    if device_id == device_ids[0]:
                        dev_emb = embedding
                        dev_off = cat_offsets
                        dev_idx = cell_indices
                    else:
                        dev_emb = cp.asarray(embedding)
                        dev_off = cp.asarray(cat_offsets)
                        dev_idx = cp.asarray(cell_indices)

                    device_data.append(
                        {
                            "emb": dev_emb,
                            "off": dev_off,
                            "idx": dev_idx,
                            "pair_left": cp.asarray(chunk_left),
                            "pair_right": cp.asarray(chunk_right),
                            "sums": cp.zeros(n_chunk_pairs, dtype=embedding.dtype),
                            "n_pairs": n_chunk_pairs,
                            "device_id": device_id,
                        }
                    )

        # Phase 2: Synchronize data transfers, then launch kernels
        for data in device_data:
            if data is None:
                continue

            device_id = data["device_id"]
            with cp.cuda.Device(device_id):
                streams[device_id].synchronize()

                is_double = embedding.dtype == np.float64
                config = _ed.get_kernel_config(n_features, is_double)
                if config is None:
                    raise RuntimeError(
                        "Insufficient shared memory for edistance kernel"
                    )
                cell_tile, feat_tile, block_size, shared_mem = config
                blocks_per_pair = _calculate_blocks_per_pair(data["n_pairs"])

                _ed.compute_distances(
                    data["emb"],
                    data["off"],
                    data["idx"],
                    data["pair_left"],
                    data["pair_right"],
                    data["sums"],
                    data["n_pairs"],
                    n_features,
                    blocks_per_pair,
                    cell_tile,
                    feat_tile,
                    block_size,
                    shared_mem,
                    cp.cuda.get_current_stream().ptr,
                )

        # Phase 3: Synchronize all devices
        for data in device_data:
            if data is not None:
                with cp.cuda.Device(data["device_id"]):
                    cp.cuda.Stream.null.synchronize()

        # Phase 4: Aggregate on GPU 0
        with cp.cuda.Device(device_ids[0]):
            total_sums = cp.zeros(n_total_pairs, dtype=embedding.dtype)
            for i, data in enumerate(device_data):
                if data is not None:
                    sums = cp.asarray(data["sums"])
                    start = chunk_offsets[i]
                    total_sums[start : start + len(sums)] = sums

        return total_sums

    def _pairwise_means(
        self,
        embedding: cp.ndarray,
        cat_offsets: cp.ndarray,
        cell_indices: cp.ndarray,
        k: int,
        device_ids: list[int],
    ) -> cp.ndarray:
        """Compute between-group mean distances for all group pairs.

        Uses flat kernel output and reconstructs a symmetric k×k matrix.

        Parameters
        ----------
        embedding
            Cell embeddings on GPU 0
        cat_offsets
            Category offsets on GPU 0
        cell_indices
            Cell indices on GPU 0
        k
            Number of groups
        device_ids
            List of GPU device IDs to use

        Returns
        -------
        cp.ndarray
            Matrix of mean pairwise distances (k x k)
        """
        group_sizes = cp.diff(cat_offsets).astype(cp.int64)

        # Build upper triangular indices, excluding diagonal for single-cell groups
        triu_indices = cp.triu_indices(k)
        pair_left = triu_indices[0].astype(cp.int32)
        pair_right = triu_indices[1].astype(cp.int32)

        is_diagonal = pair_left == pair_right
        has_pairs = group_sizes[pair_left] >= 2
        keep_mask = ~is_diagonal | has_pairs

        pair_left = pair_left[keep_mask]
        pair_right = pair_right[keep_mask]

        if len(pair_left) == 0:
            return cp.zeros((k, k), dtype=embedding.dtype)

        flat_sums = self._launch_distance_kernel(
            embedding,
            cat_offsets,
            cell_indices,
            pair_left=pair_left,
            pair_right=pair_right,
            device_ids=device_ids,
        )

        # Normalize flat sums
        flat_norms = cp.where(
            pair_left == pair_right,
            cp.maximum(group_sizes[pair_left] * (group_sizes[pair_left] - 1) // 2, 1),
            group_sizes[pair_left] * group_sizes[pair_right],
        ).astype(embedding.dtype)
        flat_means = flat_sums / flat_norms

        # Reconstruct symmetric k×k matrix from flat
        means = cp.zeros((k, k), dtype=embedding.dtype)
        means[pair_left.astype(cp.intp), pair_right.astype(cp.intp)] = flat_means
        means[pair_right.astype(cp.intp), pair_left.astype(cp.intp)] = flat_means

        return means

    def _onesided_means(
        self,
        embedding: cp.ndarray,
        cat_offsets: cp.ndarray,
        cell_indices: cp.ndarray,
        k: int,
        *,
        selected_indices: list[int],
        device_ids: list[int],
    ) -> tuple[cp.ndarray, cp.ndarray]:
        """Compute mean distances from selected group(s) to all groups.

        Uses flat kernel output to avoid O(k^2) memory allocation.

        Parameters
        ----------
        embedding
            Cell embeddings on GPU 0
        cat_offsets
            Category offsets on GPU 0
        cell_indices
            Cell indices on GPU 0
        k
            Number of groups
        selected_indices
            Indices of the selected (control) groups
        device_ids
            List of GPU device IDs to use

        Returns
        -------
        cross_means
            Array of shape (n_selected, k) where cross_means[i, j] is the
            mean distance from selected_indices[i] to group j.
        diag_means
            Array of shape (k,) with mean within-group distances.
        """
        group_sizes = cp.diff(cat_offsets).astype(cp.int64)
        group_sizes_cpu = group_sizes.get()
        n_selected = len(selected_indices)

        # Diagonal pairs first — one per group with >= 2 cells
        pair_list: list[tuple[int, int]] = []
        diag_flat: dict[int, int] = {}
        for j in range(k):
            if group_sizes_cpu[j] >= 2:
                diag_flat[j] = len(pair_list)
                pair_list.append((j, j))

        # Cross pairs grouped by selected group for L2 cache locality
        canon_to_flat: dict[tuple[int, int], int] = {}
        cross_flat: list[list[int]] = []
        for si in selected_indices:
            row = []
            for j in range(k):
                if si == j:
                    row.append(diag_flat.get(j, -1))
                else:
                    canon = (min(si, j), max(si, j))
                    idx = canon_to_flat.get(canon)
                    if idx is None:
                        idx = len(pair_list)
                        canon_to_flat[canon] = idx
                        pair_list.append((si, j))
                    row.append(idx)
            cross_flat.append(row)
        n_pairs = len(pair_list)
        if n_pairs == 0:
            return (
                cp.zeros((n_selected, k), dtype=embedding.dtype),
                cp.zeros(k, dtype=embedding.dtype),
            )

        pair_left = cp.array([p[0] for p in pair_list], dtype=cp.int32)
        pair_right = cp.array([p[1] for p in pair_list], dtype=cp.int32)

        flat_sums = self._launch_distance_kernel(
            embedding,
            cat_offsets,
            cell_indices,
            pair_left=pair_left,
            pair_right=pair_right,
            device_ids=device_ids,
        )

        # Vectorized normalization
        is_diag = pair_left == pair_right
        sizes_l = group_sizes[pair_left.astype(cp.intp)]
        sizes_r = group_sizes[pair_right.astype(cp.intp)]
        flat_norms = cp.where(
            is_diag,
            cp.maximum(sizes_l * (sizes_l - 1) // 2, 1),
            sizes_l * sizes_r,
        ).astype(embedding.dtype)
        flat_means = flat_sums / flat_norms

        # Vectorized reconstruction of cross_means (n_selected x k)
        cross_idx = cp.array(cross_flat, dtype=cp.int64)  # (n_selected, k)
        valid = cross_idx >= 0
        # Replace -1 with 0 for safe indexing, then mask
        safe_idx = cp.where(valid, cross_idx, 0)
        cross_means = cp.where(valid, flat_means[safe_idx], 0.0)

        # Vectorized reconstruction of diag_means (k,)
        diag_means = cp.zeros(k, dtype=embedding.dtype)
        if diag_flat:
            diag_j = cp.array(list(diag_flat.keys()), dtype=cp.intp)
            diag_idx = cp.array(list(diag_flat.values()), dtype=cp.int64)
            diag_means[diag_j] = flat_means[diag_idx]

        return cross_means, diag_means

    def _pairwise_means_bootstrap(
        self,
        embedding: cp.ndarray,
        *,
        cat_offsets: cp.ndarray,
        cell_indices: cp.ndarray,
        k: int,
        n_bootstrap: int,
        random_state: int,
        device_ids: list[int],
    ) -> tuple[cp.ndarray, cp.ndarray]:
        """Compute bootstrap statistics for pairwise distances.

        Each bootstrap iteration uses all GPUs for its pairwise computation.

        Parameters
        ----------
        embedding
            Cell embeddings on GPU 0
        cat_offsets
            Category offsets on GPU 0
        cell_indices
            Cell indices on GPU 0
        k
            Number of groups
        n_bootstrap
            Number of bootstrap iterations
        random_state
            Random seed for reproducibility
        device_ids
            List of GPU device IDs to use

        Returns
        -------
        tuple
            (means, variances) matrices (k x k each)
        """
        # Get group sizes for bootstrap sampling (on GPU 0)
        group_sizes = cp.diff(cat_offsets)

        # Run bootstrap iterations - each uses all GPUs for pairwise computation
        all_results = []
        for i in range(n_bootstrap):
            # Generate bootstrap sample on GPU 0
            boot_cat_offsets, boot_cell_indices = self._bootstrap_sample_cells(
                cat_offsets=cat_offsets,
                cell_indices=cell_indices,
                group_sizes_gpu=group_sizes,
                seed=random_state + i,
            )

            # Compute pairwise means using all GPUs
            pairwise_means = self._pairwise_means(
                embedding=embedding,
                cat_offsets=boot_cat_offsets,
                cell_indices=boot_cell_indices,
                k=k,
                device_ids=device_ids,
            )
            all_results.append(pairwise_means.get())

        # Compute statistics on first GPU
        with cp.cuda.Device(device_ids[0]):
            bootstrap_stack = cp.array(all_results)  # [n_bootstrap, k, k]
            means = cp.mean(bootstrap_stack, axis=0)
            variances = cp.var(bootstrap_stack, axis=0)

        return means, variances

    def _onesided_means_bootstrap(
        self,
        embedding: cp.ndarray,
        *,
        cat_offsets: cp.ndarray,
        cell_indices: cp.ndarray,
        k: int,
        selected_indices: list[int],
        n_bootstrap: int,
        random_state: int,
        device_ids: list[int],
    ) -> tuple[cp.ndarray, cp.ndarray, cp.ndarray, cp.ndarray]:
        """Compute bootstrap statistics for onesided distances.

        Each bootstrap iteration uses all GPUs for its onesided computation.

        Parameters
        ----------
        embedding
            Cell embeddings on GPU 0
        cat_offsets
            Category offsets on GPU 0
        cell_indices
            Cell indices on GPU 0
        k
            Number of groups
        selected_indices
            Indices of the selected (control) groups
        n_bootstrap
            Number of bootstrap iterations
        random_state
            Random seed for reproducibility
        device_ids
            List of GPU device IDs to use

        Returns
        -------
        cross_mean
            Mean of bootstrap cross_means, shape (n_selected, k)
        diag_mean
            Mean of bootstrap diag_means, shape (k,)
        cross_var
            Variance of bootstrap cross_means, shape (n_selected, k)
        diag_var
            Variance of bootstrap diag_means, shape (k,)
        """
        # Get group sizes for bootstrap sampling (on GPU 0)
        group_sizes = cp.diff(cat_offsets)

        # Run bootstrap iterations - each uses all GPUs for onesided computation
        all_cross = []
        all_diag = []
        for i in range(n_bootstrap):
            # Generate bootstrap sample on GPU 0
            boot_cat_offsets, boot_cell_indices = self._bootstrap_sample_cells(
                cat_offsets=cat_offsets,
                cell_indices=cell_indices,
                group_sizes_gpu=group_sizes,
                seed=random_state + i,
            )

            # Compute onesided means using all GPUs
            cross_means, diag_means = self._onesided_means(
                embedding=embedding,
                cat_offsets=boot_cat_offsets,
                cell_indices=boot_cell_indices,
                k=k,
                selected_indices=selected_indices,
                device_ids=device_ids,
            )
            all_cross.append(cross_means.get())
            all_diag.append(diag_means.get())

        # Compute statistics on first GPU
        with cp.cuda.Device(device_ids[0]):
            cross_stack = cp.array(all_cross)
            diag_stack = cp.array(all_diag)
            cross_mean = cp.mean(cross_stack, axis=0)
            diag_mean = cp.mean(diag_stack, axis=0)
            cross_var = cp.var(cross_stack, axis=0)
            diag_var = cp.var(diag_stack, axis=0)

        return cross_mean, diag_mean, cross_var, diag_var

    def _bootstrap_sample_cells(
        self,
        *,
        cat_offsets: cp.ndarray,
        cell_indices: cp.ndarray,
        group_sizes_gpu: cp.ndarray,
        seed: int,
    ) -> tuple[cp.ndarray, cp.ndarray]:
        """Generate bootstrap sample on the current device (fully parallel).

        Uses cupy random to generate indices directly on the GPU,
        avoiding cross-device transfers in multi-GPU scenarios.

        Parameters
        ----------
        cat_offsets
            Category offsets on current device
        cell_indices
            Cell indices on current device
        group_sizes_gpu
            Size of each group on current device
        seed
            Random seed for this bootstrap iteration

        Returns
        -------
        tuple
            (cat_offsets, new_cell_indices) on current device
            Note: cat_offsets unchanged since bootstrap preserves group sizes
        """
        rng = cp.random.default_rng(seed)
        total_cells = cell_indices.shape[0]

        if total_cells == 0:
            return cat_offsets, cell_indices

        # Generate random floats for all cells at once
        random_floats = rng.random(total_cells, dtype=cp.float32)

        # cp.repeat requires list for repeats - small transfer (k integers)
        group_sizes_list = group_sizes_gpu.get().tolist()

        # Expand group sizes to per-cell (each cell knows its group's size)
        cell_group_sizes = cp.repeat(group_sizes_gpu, group_sizes_list)

        # Scale random floats to local indices within each group
        bootstrap_local_idx = (random_floats * cell_group_sizes).astype(cp.int32)

        # Convert local indices to global indices by adding group offsets
        cell_group_offsets = cp.repeat(cat_offsets[:-1], group_sizes_list)
        bootstrap_global_idx = bootstrap_local_idx + cell_group_offsets

        # Gather bootstrap cells
        new_cell_indices = cell_indices[bootstrap_global_idx]

        return cat_offsets, new_cell_indices

    def _prepare_edistance_df_bootstrap(
        self,
        embedding: cp.ndarray,
        *,
        cat_offsets: cp.ndarray,
        cell_indices: cp.ndarray,
        k: int,
        groups_list: list[str],
        groupby: str,
        n_bootstrap: int = 100,
        random_state: int = 0,
        multi_gpu: bool | list[int] | str | None = None,
    ) -> tuple[pd.DataFrame, pd.DataFrame]:
        """Prepare bootstrap edistance DataFrames."""
        device_ids = parse_device_ids(multi_gpu=multi_gpu)
        pairwise_means_boot, pairwise_vars_boot = self._pairwise_means_bootstrap(
            embedding=embedding,
            cat_offsets=cat_offsets,
            cell_indices=cell_indices,
            k=k,
            n_bootstrap=n_bootstrap,
            random_state=random_state,
            device_ids=device_ids,
        )

        # Vectorized edistance: e[a,b] = 2*d[a,b] - d[a,a] - d[b,b]
        diag_means = cp.diag(pairwise_means_boot)
        edistance_means = (
            2 * pairwise_means_boot - diag_means[:, None] - diag_means[None, :]
        )
        cp.fill_diagonal(edistance_means, 0)

        # Vectorized variance computation (delta method approximation):
        # var[a, b] = 4 * var[a, b] + var[a, a] + var[b, b]
        diag_vars = cp.diag(pairwise_vars_boot)
        edistance_vars = (
            4 * pairwise_vars_boot + diag_vars[:, None] + diag_vars[None, :]
        )
        cp.fill_diagonal(edistance_vars, 0)

        df_mean = pd.DataFrame(
            edistance_means.get(), index=groups_list, columns=groups_list
        )
        df_mean.index.name = groupby
        df_mean.columns.name = groupby
        df_mean.name = "pairwise edistance"

        df_var = pd.DataFrame(
            edistance_vars.get(), index=groups_list, columns=groups_list
        )
        df_var.index.name = groupby
        df_var.columns.name = groupby
        df_var.name = "pairwise edistance variance"

        return df_mean, df_var

    def _prepare_edistance_df(
        self,
        embedding: cp.ndarray,
        *,
        cat_offsets: cp.ndarray,
        cell_indices: cp.ndarray,
        k: int,
        groups_list: list[str],
        groupby: str,
        multi_gpu: bool | list[int] | str | None = None,
    ) -> pd.DataFrame:
        """Prepare edistance DataFrame."""
        device_ids = parse_device_ids(multi_gpu=multi_gpu)
        pairwise_means = self._pairwise_means(
            embedding, cat_offsets, cell_indices, k, device_ids
        )

        # Vectorized edistance: e[a,b] = 2*d[a,b] - d[a,a] - d[b,b]
        diag = cp.diag(pairwise_means)
        edistance_matrix = 2 * pairwise_means - diag[:, None] - diag[None, :]
        cp.fill_diagonal(edistance_matrix, 0)  # Self-distance is 0

        df = pd.DataFrame(
            edistance_matrix.get(), index=groups_list, columns=groups_list
        )
        df.index.name = groupby
        df.columns.name = groupby
        df.name = "pairwise edistance"

        return df
