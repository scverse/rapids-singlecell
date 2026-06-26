from __future__ import annotations

import warnings
from typing import TYPE_CHECKING, Literal

import cupy as cp
import numpy as np

if TYPE_CHECKING:
    from anndata import AnnData

    from ._harmony import COLSUM_ALGO


def harmony_integrate(
    adata: AnnData,
    key: str | list[str],
    *,
    basis: str = "X_pca",
    adjusted_basis: str = "X_pca_harmony",
    dtype: type = np.float64,
    flavor: Literal["harmony2", "harmony1"] = "harmony2",
    n_clusters: int | None = None,
    max_iter_harmony: int = 10,
    max_iter_clustering: int = 200,
    tol_harmony: float = 1e-4,
    tol_clustering: float = 1e-5,
    sigma: float = 0.1,
    theta: float | int | list[float] | np.ndarray | cp.ndarray = 2.0,
    tau: int = 0,
    ridge_lambda: float = 1.0,
    alpha: float = 0.2,
    batch_prune_threshold: float | None = 1e-5,
    correction_method: Literal["fast", "original", "batched"] | None = None,
    colsum_algo: COLSUM_ALGO | None = None,
    block_proportion: float = 0.05,
    random_state: int = 0,
    verbose: bool = False,
) -> None:
    """Integrate different experiments using the Harmony algorithm :cite:p:`Korsunsky2019,Patikas2026`.

    This GPU-accelerated implementation is based on the harmony-pytorch package.
    As Harmony works by adjusting the principal components,
    this function should be run after performing PCA but before computing the neighbor graph.

    By default, the Harmony2 algorithm is used,
    which includes a stabilized diversity penalty,
    dynamic per-cluster-per-batch ridge regularization,
    and automatic batch pruning.
    To revert to the original Harmony behavior::

        rsc.pp.harmony_integrate(adata, key, flavor="harmony1")

    Parameters
    ----------
    adata
        The annotated data matrix.
    key
        The key(s) of the column(s) in ``adata.obs`` that differentiate(s) among experiments/batches.
        When multiple keys are provided, a combined batch variable is created from all columns.
    basis
        The name of the field in ``adata.obsm`` where the PCA table is stored.
    adjusted_basis
        The name of the field in ``adata.obsm`` where the adjusted PCA table will be stored.
    dtype
        The data type to use for Harmony computation.
        If you use 32-bit you may experience numerical instability.
    flavor
        Which version of the Harmony algorithm to use.
        ``"harmony2"`` (default) enables the stabilized diversity penalty,
        dynamic per-cluster-per-batch ridge regularization,
        and automatic batch pruning from :cite:p:`Patikas2026`.
        ``"harmony1"`` uses the original algorithm from :cite:p:`Korsunsky2019`.
    n_clusters
        Number of clusters used for soft k-means in the Harmony algorithm.
        If ``None``, uses ``min(100, N / 30)``.
        More clusters capture finer-grained structure but increase computation time.
    max_iter_harmony
        Maximum number of outer Harmony iterations
        (each consisting of a clustering step followed by a correction step).
    max_iter_clustering
        Maximum iterations for the clustering step within each Harmony iteration.
    tol_harmony
        Convergence tolerance for the Harmony objective function.
        The algorithm stops when the relative change in objective falls below this value.
    tol_clustering
        Convergence tolerance for the clustering step within each
        Harmony iteration.
    sigma
        Width of the soft-clustering kernel.
        Controls the entropy of cluster assignments:
        smaller values produce harder assignments (cells assigned to fewer clusters),
        while larger values produce softer assignments (cells spread across more clusters).
    theta
        Diversity penalty weight per batch variable.
        Controls how strongly Harmony encourages each cluster
        to contain a balanced representation of all batches.
        Higher values (e.g. ``4``) produce more aggressive mixing;
        lower values (e.g. ``0.5``) allow more batch-specific clusters.
        Set to ``0`` to disable batch correction entirely.
        A list can be provided to set different weights per batch variable.
    tau
        Discounting factor on ``theta``.
        When ``tau > 0``,
        the diversity penalty is down-weighted for batches with fewer cells,
        preventing over-correction of small batches.
        By default (``0``), there is no discounting.
    ridge_lambda
        Ridge regression regularization for the correction step.
        Larger values produce more conservative (smaller) corrections,
        preventing over-fitting.
        Only used with ``flavor="harmony1"``.
    alpha
        Scaling factor for the dynamic per-cluster-per-batch ridge regularization.
        The effective regularization for each cluster-batch pair is ``alpha * E_kb``
        where ``E_kb`` is the expected number of cells.
        Larger values produce more conservative corrections.
        Only used with ``flavor="harmony2"``.
    batch_prune_threshold
        Fraction threshold below which a batch-cluster pair is pruned (correction suppressed).
        When the fraction of a batch's cells assigned to a cluster (``O_kb / N_b``) falls below this threshold,
        that batch-cluster pair receives no correction, preventing spurious adjustments.
        Only used with ``flavor="harmony2"``.
        Set to ``None`` to disable pruning.
    correction_method
        Method for the correction step.
        ``"original"`` uses per-cluster ridge regression with explicit matrix inversion.
        ``"fast"`` uses a precomputed factorization that avoids the full inversion,
        which can be faster for datasets with many batches.
        ``"batched"`` processes all clusters simultaneously (fastest but requires more memory).
        If ``None`` (default), automatically selects ``"batched"`` unless
        the workspace would exceed 1 GB, in which case ``"fast"`` is used.
    colsum_algo
        Algorithm for column sums.
        If ``None``, chosen automatically.
        If ``"benchmark"``, benchmarks all algorithms.
    block_proportion
        Proportion of cells updated per clustering sub-iteration.
        Smaller values produce more stochastic updates.
        Larger values are faster but may converge to different solutions.
    random_state
        Random seed for reproducibility.
    verbose
        Whether to print benchmarking and convergence information.

    Returns
    -------
    Updates adata with the field ``adata.obsm[adjusted_basis]``,
    containing principal components adjusted by Harmony
    such that different experiments are integrated.
    """
    from ._harmony import harmonize

    # Resolve flavor into internal flags
    if flavor not in {"harmony1", "harmony2"}:
        raise ValueError(f"flavor must be 'harmony1' or 'harmony2', got {flavor!r}.")
    stabilized_penalty = flavor == "harmony2"
    dynamic_lambda = flavor == "harmony2"

    # Warn when flavor-incompatible parameters are explicitly set
    if flavor == "harmony2" and ridge_lambda != 1.0:
        warnings.warn(
            "ridge_lambda is ignored when flavor='harmony2'; "
            "use alpha to control regularization strength.",
            UserWarning,
            stacklevel=2,
        )
    if flavor == "harmony1":
        if alpha != 0.2:
            warnings.warn(
                "alpha is ignored when flavor='harmony1'; use ridge_lambda instead.",
                UserWarning,
                stacklevel=2,
            )
        if batch_prune_threshold != 1e-5:
            warnings.warn(
                "batch_prune_threshold is ignored when flavor='harmony1'.",
                UserWarning,
                stacklevel=2,
            )

    # Ensure the basis exists in adata.obsm
    if basis not in adata.obsm:
        raise ValueError(
            f"The specified basis '{basis}' is not available in adata.obsm. "
            f"Available bases: {list(adata.obsm.keys())}"
        )

    # Get the input data
    input_data = adata.obsm[basis]

    try:
        # Handle different array types
        if isinstance(input_data, np.ndarray):
            # NumPy array: convert directly to CuPy
            try:
                X = cp.array(input_data, dtype=dtype, order="C")
            except (cp.cuda.memory.OutOfMemoryError, MemoryError) as e:
                raise MemoryError(
                    "Not enough GPU memory to allocate array. "
                    "Try reducing the dataset size or using a GPU with more memory."
                ) from e
        elif isinstance(input_data, cp.ndarray):
            # CuPy array: ensure correct dtype and layout with a copy
            X = input_data.astype(dtype, order="C", copy=False)
        else:
            # Other array types: convert to NumPy first, then to CuPy
            try:
                # Try to convert to numpy (works for most array-like objects)
                np_array = np.array(input_data, dtype=dtype, order="C")
                X = cp.array(np_array, dtype=dtype, order="C")
            except Exception as e:
                raise TypeError(
                    f"Could not convert input of type {type(input_data).__name__} to CuPy array: {str(e)}"
                ) from e

        # Verify array is valid
        if cp.isnan(X).any():
            raise ValueError(
                "Input data contains NaN values. Please handle these before running harmony_integrate."
            )

    except Exception as e:
        raise RuntimeError(f"Error preparing data for Harmony: {str(e)}") from e

    harmony_out = harmonize(
        X,
        adata.obs,
        key,
        n_clusters=n_clusters,
        max_iter_harmony=max_iter_harmony,
        max_iter_clustering=max_iter_clustering,
        tol_harmony=tol_harmony,
        tol_clustering=tol_clustering,
        ridge_lambda=ridge_lambda,
        sigma=sigma,
        block_proportion=block_proportion,
        theta=theta,
        tau=tau,
        correction_method=correction_method,
        colsum_algo=colsum_algo,
        random_state=random_state,
        stabilized_penalty=stabilized_penalty,
        dynamic_lambda=dynamic_lambda,
        alpha=alpha,
        batch_prune_threshold=batch_prune_threshold,
        verbose=verbose,
    )

    adata.obsm[adjusted_basis] = harmony_out.get()
