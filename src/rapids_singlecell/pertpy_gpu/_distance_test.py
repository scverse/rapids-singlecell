from __future__ import annotations

import warnings
from functools import partial
from math import isqrt
from typing import TYPE_CHECKING

import cupy as cp
import numpy as np
import pandas as pd
from statsmodels.stats.multitest import multipletests  # type: ignore[import-untyped]
from tqdm.auto import tqdm  # type: ignore[import-untyped]

from ._distance import Distance

if TYPE_CHECKING:
    from anndata import AnnData

    from ._distance import Metric


class DistanceTest:
    """Test each observed group against a reference by permuting their cell labels.

    Parameters
    ----------
    metric
        A :class:`~rapids_singlecell.ptg.Distance` metric that compares two groups.
    n_perms
        Number of label permutations per comparison.
    layer_key, obsm_key
        Data source, as in :class:`~rapids_singlecell.ptg.Distance`.
    padj_threshold
        Significance level for raw and adjusted p-values.
    correction
        Multiple-testing method accepted by statsmodels.
    cell_wise_metric
        cuML pairwise metric for the precomputed energy and mean pairwise tests.
        Defaults to Euclidean distance.
    max_distance_memory
        Maximum bytes for cached cell distances or one streamed distance tile.
        Defaults to half the available GPU memory. Labels, features, reductions
        and library workspace use additional memory.
    random_state
        An integer seed reproduces NumPy's label shuffles. Pass a generator
        such as ``cupy.random.default_rng(0)`` to sample labels on the GPU.
        An explicit NumPy or CuPy generator advances across calls.
    alpha
        Deprecated alias for `padj_threshold`.
    **kwargs
        Options forwarded to the distance metric. The precomputed E-test uses
        float64 for both ``precision="auto"`` and ``precision="float64"``.

    Notes
    -----
    Like pertpy, the p-value is the fraction of shuffled distances strictly
    larger than the observed distance, with a minimum of 1 / n_perms.
    Ties do not count, so a constant statistic also receives this minimum.
    For the single-reference test, the correction includes the reference at the minimum p-value, then its
    reported raw and adjusted p-values are both set to one. Nonfinite
    statistics produce missing p-values and are excluded from the correction.
    The single-reference E-test includes the diagonal when averaging within-group distances.
    The E-test and mean-pairwise test cache distances that fit and stream
    larger comparisons in tiles. Consecutive Euclidean E-contrasts can share
    a reference distance matrix or its row sums.
    """

    def __init__(
        self,
        metric: Metric,
        *,
        n_perms: int = 1000,
        layer_key: str | None = None,
        obsm_key: str | None = None,
        padj_threshold: float = 0.05,
        correction: str = "holm-sidak",
        cell_wise_metric: str | None = None,
        max_distance_memory: int | None = None,
        random_state: int | np.random.Generator | cp.random.Generator | None = 0,
        alpha: float | None = None,
        **kwargs,
    ):
        if metric == "classifier_cp":
            raise ValueError("classifier_cp requires a multiclass one-sided comparison.")
        if isinstance(n_perms, bool) or not isinstance(n_perms, int | np.integer) or n_perms < 1:
            raise ValueError("n_perms must be a positive integer.")
        if alpha is not None:
            warnings.warn("Use padj_threshold in place of alpha.", FutureWarning, stacklevel=2)
            padj_threshold = alpha
        if not 0 < padj_threshold < 1:
            raise ValueError("padj_threshold must be between 0 and 1.")
        if metric == "mean_pairwise" and cell_wise_metric is not None:
            kwargs["cell_wise_metric"] = cell_wise_metric
        if max_distance_memory is not None and (
            not isinstance(max_distance_memory, int | np.integer) or max_distance_memory < 8
        ):
            raise ValueError("max_distance_memory must be at least 8 bytes or None.")
        self.max_distance_memory = int(max_distance_memory) if max_distance_memory is not None else None
        self.distance = Distance(metric, layer_key=layer_key, obsm_key=obsm_key, **kwargs)
        if metric == "edistance" and kwargs.get("precision") in ("float16", "float32"):
            raise ValueError("E-distance permutation tests use float64. Use precision='auto' or 'float64'.")
        self.layer_key = self.distance.layer_key
        self.obsm_key = self.distance.obsm_key
        self.metric = metric
        self.n_perms = n_perms
        self.padj_threshold = padj_threshold
        self.correction = correction
        self.cell_wise_metric = cell_wise_metric or "euclidean"
        self.random_state = random_state

    def __call__(self, adata: AnnData, groupby: str, contrast: str, show_progressbar: bool = True) -> pd.DataFrame:  # noqa: FBT001, FBT002
        """Return distance, raw and adjusted p-values and significance flags for each group."""
        if self.metric in ("edistance", "mean_pairwise"):
            return self.test_precomputed(adata, groupby, contrast, show_progressbar)
        return self.test_xy(adata, groupby, contrast, show_progressbar)

    def test_xy(self, adata: AnnData, groupby: str, contrast: str, show_progressbar: bool = True) -> pd.DataFrame:  # noqa: FBT001, FBT002
        """Recompute the array distance for each shuffle using the metric's two-array arithmetic."""
        return self._test(adata, groupby, contrast, show_progressbar, self._xy_distances)

    def test_precomputed(self, adata: AnnData, groupby: str, contrast: str, verbose: bool = True) -> pd.DataFrame:  # noqa: FBT001, FBT002
        """Reuse cell distances across permutations, tiling comparisons that exceed the memory budget."""
        if self.metric not in ("edistance", "mean_pairwise"):
            raise ValueError(f"Metric {self.metric} does not accept precomputed distances.")
        return self._test(adata, groupby, contrast, verbose, self._precomputed_distances)

    def contrast_distances(
        self, adata: AnnData, contrasts: pd.DataFrame, *, show_progressbar: bool = True
    ) -> pd.DataFrame:
        """Test the distances in a table from :meth:`Distance.create_contrasts`.

        Each row shuffles the target and reference labels within its specified
        stratum, keeping both group sizes fixed. The E-distance statistic used
        here excludes within-group self-pairs, as in :meth:`Distance.contrast_distances`.
        A singleton group's within-group contribution is zero.

        Returns the input rows with the metric, raw and adjusted p-values, and
        significance flags. Correction spans the distinct, finite contrasts in
        the table. Repeated rows share a result. Self-comparisons have distance
        zero and p-values one. A missing group in a stratum yields NaNs.
        """
        from ._metrics._base_metric import BaseMetric

        if not contrasts.columns.is_unique:
            raise ValueError("Contrast column names must be unique.")
        groupby, splits = BaseMetric._parse_contrasts(adata, contrasts)
        reserved = {self.metric, "pvalue", "significant", "pvalue_adj", "significant_adj"}
        if conflicts := reserved.intersection(contrasts.columns):
            raise ValueError(f"Contrast columns conflict with output columns: {sorted(conflicts)}")

        columns = [groupby, "reference", *splits]
        requested = pd.MultiIndex.from_frame(contrasts[columns])
        distinct = requested.unique()
        inverse = distinct.get_indexer(requested)
        pairs = [
            ((target, *levels), (reference, *levels)) if splits else (target, reference)
            for target, reference, *levels in distinct
        ]
        compute = self._xy_distances
        if self.metric == "edistance":
            compute = partial(self._precomputed_distances, exclude_diagonal=True)
        elif self.metric == "mean_pairwise":
            compute = self._precomputed_distances
        group_rows = BaseMetric._contrast_indices(adata.obs, contrasts, groupby, splits)
        stats = self._test_pairs(adata, group_rows, pairs, compute, show_progressbar, correct_reference=False)
        result = contrasts.copy()
        for column, values in stats.items():
            result[self.metric if column == "distance" else column] = values.to_numpy()[inverse]
        return result

    def _test(self, adata, groupby, contrast, show_progressbar, compute):
        groups = adata.obs[groupby].dropna().unique()
        if contrast not in groups:
            raise ValueError(f"Contrast group {contrast!r} not found in adata.obs[{groupby!r}].")
        pairs = [(group, contrast) for group in groups]
        group_rows = adata.obs.groupby(groupby, sort=False, observed=True).indices
        result = self._test_pairs(adata, group_rows, pairs, compute, show_progressbar, correct_reference=True)
        result.index = pd.Index(groups)
        return result

    def _test_pairs(self, adata, group_rows, pairs, compute, show_progressbar, *, correct_reference):
        from cupyx.scipy import sparse as gpu_sparse

        from ._metrics._base_metric import _subset_data

        embedding = self.distance._metric_impl._get_embedding(adata) if pairs else None
        if gpu_sparse.issparse(embedding) and embedding.indptr.dtype != embedding.indices.dtype:
            # CuPy row indexing requires matching indptr and indices types.
            embedding = type(embedding)((embedding.data, embedding.indices, embedding.indptr), shape=embedding.shape)
        rng = (
            self.random_state
            if isinstance(self.random_state, cp.random.Generator)
            else np.random.default_rng(self.random_state)
        )
        distances = np.full(len(pairs), np.nan)
        pvalues = np.full(len(pairs), np.nan)
        is_reference = np.zeros(len(pairs), dtype=bool)
        share_reference = (
            compute != self._xy_distances
            and self.metric in ("edistance", "mean_pairwise")
            and self.cell_wise_metric in ("euclidean", "l2")
            and isinstance(embedding, np.ndarray | cp.ndarray)
            and embedding.dtype in (np.float32, np.float64)
        )
        available = self._available_memory() if share_reference else 0
        budget = min(available, self.max_distance_memory or available)
        statistic = "contrast" if self.metric == "edistance" and not correct_reference else self.metric
        cache = None
        for i, (target_key, ref_key) in enumerate(tqdm(pairs, disable=not show_progressbar)):
            if i and pairs[i - 1][1] != ref_key:
                cache = None
            target, reference = group_rows.get(target_key), group_rows.get(ref_key)
            if target is None or reference is None:
                continue
            if target_key == ref_key:
                distances[i] = 0
                is_reference[i] = True
                if correct_reference:
                    # pertpy includes the reference at the permutation floor in the correction.
                    pvalues[i] = 1 / self.n_perms
                continue
            # Merge the sorted row indices in observation order to preserve seeded shuffles.
            membership = np.zeros(len(reference) + len(target), dtype=bool)
            membership[np.searchsorted(reference, target) + np.arange(len(target))] = True
            oversized = 8 * len(membership) ** 2 > budget
            repeats = cache is not None or (i + 1 < len(pairs) and pairs[i + 1][1] == ref_key)
            big, small = max(len(target), len(reference)), min(len(target), len(reference))
            if (
                share_reference
                and big >= 512
                and small * 8 <= big
                and max(len(membership), embedding.shape[1]) <= np.iinfo(np.int32).max
                and len(membership) * embedding.shape[1] * 8 <= available
                and (not oversized or small**2 * embedding.shape[1] <= len(membership) ** 2)
                and (oversized or (statistic == "contrast" and len(reference) == big and repeats))
            ):
                from ._reference_energy import ReferenceEnergy

                if len(target) == big:
                    cache = None
                    engine = ReferenceEnergy(_subset_data(embedding, target), max_distance_memory=budget)
                    side, labels = reference, ~membership
                else:
                    if cache is None:
                        cache = ReferenceEnergy(_subset_data(embedding, reference), max_distance_memory=budget)
                    engine = cache
                    side, labels = target, membership
                # Draw in original target order, also when using the target as the large reference.
                batches = self._permutation_batches(membership, rng, bytes_per_row=32 * small, as_indices=True)
                scores = engine.scores(
                    _subset_data(embedding, side), labels, batches, self.n_perms, statistic=statistic
                )
                del engine
            else:
                cache = None
                rows = np.empty(len(membership), dtype=np.intp)
                rows[membership], rows[~membership] = target, reference
                scores = compute(_subset_data(embedding, rows), membership, rng)
            distances[i] = scores[0]
            if np.isfinite(scores).all():
                # Matrix reductions can differ by roundoff for theoretically tied labelings.
                threshold = scores[0] + 100 * np.finfo(float).eps * abs(scores[0])
                pvalues[i] = max(1, np.count_nonzero(scores[1:] > threshold)) / self.n_perms

        valid = np.isfinite(pvalues)
        rejected = np.zeros(len(pairs), dtype=bool)
        adjusted = np.full(len(pairs), np.nan)
        if valid.any():
            with np.errstate(divide="ignore"):
                rejected[valid], adjusted[valid], _, _ = multipletests(
                    pvalues[valid], alpha=self.padj_threshold, method=self.correction
                )
        pvalues[is_reference] = adjusted[is_reference] = 1.0
        rejected[is_reference] = False
        return pd.DataFrame(
            {
                "distance": distances,
                "pvalue": pvalues,
                "significant": pvalues < self.padj_threshold,
                "pvalue_adj": adjusted,
                "significant_adj": rejected,
            }
        )

    def _gather_small_group(self, n, k):
        return self.n_perms >= 64 and n >= 512 and k**2 <= 8 * n

    def _permutation_batches(self, membership, rng, *, bytes_per_row, max_batch=1024, as_indices=False):
        """Yield label shuffles, optionally as sorted indices of the smaller side."""
        gpu = isinstance(rng, cp.random.Generator)
        n = len(membership)
        n_group = int(membership.sum())
        k = min(n_group, n - n_group)
        sample_indices = gpu and n >= 8192 and k <= n // 16
        if gpu and as_indices:
            # CuPy's stream depends on draw shape. Keep the batches used for dense label masks.
            mask_bytes = 16 * n + 8 * k**2 if self._gather_small_group(n, k) else 25 * n
            bytes_per_row = max(bytes_per_row, mask_bytes)
        if sample_indices:
            observed = cp.asarray(np.flatnonzero(membership if k == n_group else ~membership))[None, :]
            bytes_per_row = max(bytes_per_row, n + 96 * k)
        elif gpu:
            bytes_per_row = max(bytes_per_row, 32 * n)
            membership = cp.asarray(membership)
        elif as_indices:
            bytes_per_row = max(bytes_per_row, 2 * n + 16 * k)
        xp = cp if gpu else np
        batch_size = max(1, min(max_batch, (64 << 20) // bytes_per_row))
        for start in range(0, self.n_perms + 1, batch_size):
            size = min(batch_size, self.n_perms + 1 - start)
            if sample_indices:
                rows = size - (start == 0)
                indices = self._small_group_indices(rng, rows, n, k) if rows else observed
                if start == 0 and rows:
                    indices = cp.concatenate((observed, indices))
                if as_indices:
                    yield start, cp.sort(indices, axis=1)
                else:
                    masks = cp.full((size, n), k != n_group)
                    cp.put_along_axis(masks, indices, k == n_group, axis=1)
                    yield start, masks
                continue
            masks = xp.broadcast_to(membership, (size, n)).copy()
            shuffled = masks[1:] if start == 0 else masks
            if gpu and len(shuffled):
                # With distinct boundary keys every k-subset is equally likely.
                while True:
                    keys = rng.integers(0, 2**64, size=shuffled.shape, dtype=cp.uint64)
                    ranked = cp.partition(keys, (k - 1, k), axis=1)
                    if cp.all(ranked[:, k - 1] != ranked[:, k]):
                        cutoff = ranked[:, k - 1, None]
                        select = cp.less_equal if k == n_group else cp.greater
                        select(keys, cutoff, out=shuffled)
                        break
            elif not gpu:
                rng.permuted(shuffled, axis=1, out=shuffled)
            if as_indices:
                masks = xp.nonzero(masks if k == n_group else ~masks)[1].reshape(size, k)
            yield start, cp.asarray(masks)

    @staticmethod
    def _small_group_indices(rng, rows, n, k):
        """The first k distinct indices in independent uniform draws form a uniform k-subset."""
        while True:
            draws = rng.integers(n, size=(rows, 2 * k), dtype=cp.int64)
            if k == 1:
                return draws[:, :1]
            order = cp.argsort(draws, axis=1)  # stable, so each value's first occurrence comes first
            sorted_draws = cp.take_along_axis(draws, order, axis=1)
            order[:, 1:] = cp.where(sorted_draws[:, 1:] == sorted_draws[:, :-1], 2 * k, order[:, 1:])
            first = cp.partition(order, k - 1, axis=1)[:, :k]
            if cp.all(first < 2 * k):
                return cp.take_along_axis(draws, first, axis=1)

    def _xy_distances(self, cells, membership, rng):
        from ._metrics._base_metric import _as_gpu_data, _is_sparse
        from ._metrics._group_metric import _as_dense
        from ._metrics._pseudobulk import PseudobulkMetric

        if self.metric != "edistance":
            cells = _as_gpu_data(cells)
        metric = self.distance._metric_impl
        pseudobulk = isinstance(metric, PseudobulkMetric)
        if (pseudobulk or (self.metric == "mmd" and metric.kernel == "linear")) and self._can_batch_means(cells):
            return self._mean_distances(cells, membership, rng)
        if self.metric in ("t_test", "sym_kldiv") and self._can_batch_means(cells):
            return self._moment_distances(cells, membership, rng)

        if self.metric == "edistance":
            # Its two-array path takes exact differences of int64/uint64 counts.
            cells = cp.asarray(cells.toarray() if _is_sparse(cells) else cells)
        elif not pseudobulk:
            cells = _as_dense(cells)
        elif cells.dtype not in (np.float32, np.float64):
            cells = cells.astype(cp.float64)
        # A biased MMD is w.T @ K @ w. Cache the kernel for small comparisons;
        # larger ones keep the metric's bounded-memory blockwise calculation.
        if self.metric == "mmd" and metric.kernel != "linear" and 8 * len(cells) ** 2 <= (256 << 20):
            cells = cp.ascontiguousarray(cells)
            gram = metric._kernel_values(cells, cells).astype(cp.float64, copy=False)
            gram -= gram[0, 0].copy()  # The weights sum to zero, including for constant kernels.
            return self._quadratic_distances(gram, membership, rng)
        scores = np.empty(self.n_perms + 1)
        for start, masks in self._permutation_batches(membership, rng, bytes_per_row=len(membership)):
            for i, mask in enumerate(masks, start):
                scores[i] = self.distance(cells[mask], cells[~mask])
        return scores

    @staticmethod
    def _can_batch_means(cells):
        values = cells if isinstance(cells, cp.ndarray) else cells.data
        # A zero mask times NaN or infinity would contaminate the other group.
        finite = cp.isfinite(values).all()
        if cells.dtype != cp.float64:
            return bool(finite)

        # Preserve scalar accumulation when roundoff in a large common offset
        # can change tied feature ranks or the ordering of near-zero distances.
        low, high = cells.min(axis=0), cells.max(axis=0)
        if not isinstance(cells, cp.ndarray):
            low, high = low.toarray(), high.toarray()
        scale = cp.maximum(cp.abs(low), cp.abs(high))
        exact_constant = (low == high) & (low == cp.trunc(low)) & (scale * cells.shape[0] < 2**53)
        return bool(finite & cp.all(exact_constant | (high - low >= np.sqrt(np.finfo(float).eps) * scale)))

    def _mean_distances(self, cells, membership, rng):
        n_group = int(membership.sum())
        n_ref = len(membership) - n_group
        features = cells.astype(cp.float64, copy=False).T
        if not isinstance(features, cp.ndarray):
            features = features.tocsr()
        scores = cp.empty(self.n_perms + 1, dtype=cp.float64)
        for start, masks in self._permutation_batches(
            membership, rng, bytes_per_row=9 * len(membership) + 64 * cells.shape[1]
        ):
            # Sum each side directly: subtracting a group from the pooled sum loses small reference values.
            left = (features @ masks.T.astype(cp.float64)).T / n_group
            right = (features @ (~masks.T).astype(cp.float64)).T / n_ref
            if self.metric == "mmd":
                batch = cp.square(left - right).sum(axis=1)
            else:
                batch = self.distance._metric_impl._distance_between_pairs(left, right)
            scores[start : start + len(batch)] = batch
        return scores.get()

    def _moment_distances(self, cells, membership, rng):
        n_x = int(membership.sum())
        n_y = len(membership) - n_x
        features = cells.astype(cp.float64, copy=False).T
        if isinstance(features, cp.ndarray):
            features = features - features.mean(axis=1, keepdims=True)
            squares = cp.square(features)
        else:
            # Leave implicit zeros sparse and coalesce duplicate indices before squaring.
            features = features.tocsr(copy=True)
            features.sum_duplicates()
            squares = features.power(2)

        def moments(mask, count):
            weights = mask.T.astype(cp.float64)
            mean = (features @ weights).T / count
            second = (squares @ weights).T / count
            var = second - mean**2
            # Use the direct two-pass variance if subtraction loses significant digits.
            unstable = ((second != 0) & (var <= np.sqrt(np.finfo(float).eps) * second)) | ~cp.isfinite(var)
            return mean, var, unstable.any(axis=1)

        scores = cp.empty(self.n_perms + 1, dtype=cp.float64)
        for start, masks in self._permutation_batches(
            membership, rng, bytes_per_row=9 * len(membership) + 96 * cells.shape[1]
        ):
            x, var_x, bad_x = moments(masks, n_x)
            y, var_y, bad_y = moments(~masks, n_y)
            batch = self.distance._metric_impl._from_moments(x - y, var_x, var_y, n_x, n_y)
            for i in cp.flatnonzero(bad_x | bad_y).get():
                batch[i] = self.distance(cells[masks[i]], cells[~masks[i]])
            scores[start : start + len(batch)] = batch
        return scores.get()

    @staticmethod
    def _available_memory():
        free, _ = cp.cuda.runtime.memGetInfo()
        pool = getattr(cp.cuda.get_allocator(), "__self__", None)
        if isinstance(pool, cp.cuda.MemoryPool):
            free += pool.free_bytes()
            if limit := pool.get_limit():
                free = min(free, limit - pool.used_bytes())
        return max(8, int(free) // 2)

    def _precomputed_distances(self, cells, membership, rng, *, exclude_diagonal=False):
        from ._pairwise_permutation import cell_distance_blocks, tiled_scores

        n = len(membership)
        read = cell_distance_blocks(cells, self.cell_wise_metric)
        available = self._available_memory()
        budget = min(available, self.max_distance_memory or available)
        dense = isinstance(cells, np.ndarray | cp.ndarray)
        if 8 * n**2 <= budget and (not dense or n * cells.shape[1] * 8 <= available):
            try:
                pairwise = read(0, n, 0, n)
            except MemoryError:
                # Library workspaces and other GPU users can reduce available memory.
                budget = min(budget, self._available_memory(), max(8, 2 * n**2))
            else:
                if exclude_diagonal:
                    return self._contrast_energy_distances(pairwise, membership, rng)
                return self._quadratic_distances(pairwise, membership, rng)

        small = min(int(membership.sum()), n - int(membership.sum()))
        gather = self._gather_small_group(n, small)
        # Keep the resident path's generator call shapes across the memory boundary.
        bytes_per_row = 16 * n + 8 * small**2 if gather else 25 * n
        batches = self._permutation_batches(membership, rng, bytes_per_row=bytes_per_row)
        return tiled_scores(
            read,
            membership,
            batches,
            n_perms=self.n_perms,
            block_size=max(
                1, min(4096, isqrt(budget // 8), (128 << 20) // max(1, 8 * cells.shape[1]) if dense else 4096)
            ),
            symmetric=self.cell_wise_metric
            in ("euclidean", "l2", "sqeuclidean", "cosine", "cityblock", "manhattan", "l1"),
            statistic="contrast" if exclude_diagonal else self.metric,
        )

    def _contrast_energy_distances(self, pairwise, membership, rng):
        from ._reference_energy import offdiagonal_energy

        n_target = int(membership.sum())
        small, large = sorted((n_target, len(membership) - n_target))
        # Removing a constant from all off-diagonal entries preserves the U-statistic
        # for nonsingleton groups and gives exactly tied scores for equidistant cells.
        offset = pairwise[0, 1].copy()
        pairwise -= offset
        cp.fill_diagonal(pairwise, 0)
        marginals = cp.stack((pairwise.sum(axis=1), pairwise.sum(axis=0)), axis=1)
        total = marginals[:, 0].sum()

        gather = self._gather_small_group(len(membership), small)
        bytes_per_row = 8 * small**2 + 32 * small if gather else 25 * len(membership)
        scores = cp.empty(self.n_perms + 1, dtype=cp.float64)
        for start, samples in self._permutation_batches(
            membership, rng, bytes_per_row=bytes_per_row, as_indices=gather
        ):
            if gather:
                indices = samples
                within = pairwise[indices[:, :, None], indices[:, None, :]].sum(axis=(1, 2))
                rows, cols = marginals[indices].sum(axis=1).T
            else:
                selected = samples if small == n_target else ~samples
                indicator = selected.astype(cp.float64)
                within = ((indicator @ pairwise) * indicator).sum(axis=1)
                rows, cols = (indicator @ marginals).T
            cross = (rows if small == n_target else cols) - within
            other = total - rows - cols + within
            scores[start : start + len(samples)] = offdiagonal_energy(cross, within, other, small, large, offset=offset)
        return scores.get()

    def _quadratic_distances(self, pairwise, membership, rng):
        n_group = int(membership.sum())
        n_reference = len(membership) - n_group
        small = min(n_group, n_reference)
        # Indexing the smaller group wins once its square fits in a few pooled rows.
        if self._gather_small_group(len(membership), small):
            return self._small_group_distances(pairwise, membership, rng, n_group=n_group, n_reference=n_reference)
        scores = cp.empty(self.n_perms + 1, dtype=cp.float64)
        for start, masks in self._permutation_batches(membership, rng, bytes_per_row=25 * len(membership)):
            if self.metric != "mean_pairwise":
                weights = cp.where(masks, 1.0 / n_group, -1.0 / n_reference)
                batch = ((weights @ pairwise) * weights).sum(axis=1)
                if self.metric == "edistance":
                    batch = -batch
            else:
                group = masks.astype(cp.float64)
                batch = ((group @ pairwise) * ~masks).sum(axis=1) / (n_group * n_reference)
            scores[start : start + len(batch)] = batch
        return scores.get()

    def _small_group_distances(self, pairwise, membership, rng, *, n_group, n_reference):
        small = min(n_group, n_reference)
        is_target = n_group <= n_reference
        if self.metric == "mean_pairwise":
            marginal = pairwise.sum(axis=1 if is_target else 0)
            marginal = cp.where(cp.isfinite(marginal).all(), marginal, cp.nan)
        else:
            # After double-centering, the weighted quadratic form uses only the
            # selected submatrix: w = ±(1/n_group + 1/n_reference) * centered_mask.
            rows, columns = pairwise.mean(axis=1), pairwise.mean(axis=0)
            pairwise -= rows[:, None]
            pairwise -= columns
            pairwise += rows.mean()

        scores = cp.empty(self.n_perms + 1, dtype=cp.float64)
        bytes_per_row = 8 * small**2 + 32 * small
        for start, indices in self._permutation_batches(membership, rng, bytes_per_row=bytes_per_row, as_indices=True):
            within = pairwise[indices[:, :, None], indices[:, None, :]].sum(axis=(1, 2))
            if self.metric == "mean_pairwise":
                batch = (marginal[indices].sum(axis=1) - within) / (n_group * n_reference)
            else:
                batch = within * (1 / n_group + 1 / n_reference) ** 2
                if self.metric == "edistance":
                    batch = -batch
            scores[start : start + len(batch)] = batch
        return scores.get()
