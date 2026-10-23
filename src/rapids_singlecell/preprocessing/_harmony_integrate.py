from __future__ import annotations

import threading
import warnings
from typing import TYPE_CHECKING, Literal

import cupy as cp
import numpy as np

from rapids_singlecell._keys import _harmony_obsm_key, _resolve_obsm_key
from rapids_singlecell._utils._random import (
    RNGLike,
    SeedLike,
    _accepts_legacy_random_state,
)

if TYPE_CHECKING:
    from anndata import AnnData

    from ._harmony import COLSUM_ALGO


@_accepts_legacy_random_state(0)
def harmony_integrate(
    adata: AnnData,
    key: str | list[str],
    *,
    basis: str = "X_pca",
    adjusted_basis: str | None = None,
    dtype: type | str = np.float32,
    flavor: Literal["harmony2", "harmony1"] = "harmony2",
    n_clusters: int | None = None,
    max_iter_harmony: int = 10,
    max_iter_clustering: int | None = None,
    tol_harmony: float | None = None,
    tol_clustering: float | None = None,
    sigma: float = 0.1,
    theta: float | int | list[float] | np.ndarray | cp.ndarray = 2.0,
    tau: int = 0,
    ridge_lambda: float = 1.0,
    alpha: float = 0.2,
    batch_prune_threshold: float | None = 1e-5,
    correction_method: Literal["fast", "batched"] | None = None,
    colsum_algo: COLSUM_ALGO | None = None,
    block_proportion: float = 0.05,
    shuffle_chunk_size: int = 8,
    rng: SeedLike | RNGLike | None = None,
    verbose: bool = False,
) -> None:
    """Integrate different experiments using the Harmony algorithm :cite:p:`Korsunsky2019,Patikas2026`.

    This GPU-accelerated implementation was originally based on the
    harmony-pytorch package. Multiple batch variables now follow the
    per-covariate formulation described in the Harmony papers: each key is
    modeled separately instead of combining all keys into one joint category.
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
        Multiple keys are modeled as separate batch variables, with one active
        categorical level per variable and cell.
        To retain the joint-combination behavior of earlier releases, combine
        the desired columns into one categorical column and pass that single key.
    basis
        The name of the field in ``adata.obsm`` where the PCA table is stored.
        Either spelling of the PCA key resolves to whichever one the data
        actually uses, so the default works under any
        ``rapids_singlecell.settings.preset``.
    adjusted_basis
        The name of the field in ``adata.obsm`` where the adjusted PCA table will be
        stored. Defaults to ``basis`` suffixed with ``"_harmony"``, so it follows
        the naming of the basis it corrected (``"X_pca"`` gives
        ``"X_pca_harmony"``, ``"pca"`` gives ``"pca_harmony"``).
    dtype
        The data type used for initialization and Harmony computation.
        Defaults to ``numpy.float32``. Use ``numpy.float64`` when additional
        precision is needed; it can be slower on GPUs with reduced
        double-precision throughput. ``"bfloat16"`` computes in float32 but
        stores the soft cluster assignments in bfloat16, which is faster and
        halves their memory (about a quarter less GPU memory with 50
        components). It needs one batch key and the batched
        correction and falls back to float32 with a warning otherwise.
        ``None`` keeps the input's float32 or float64.
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
        If ``None``, uses the value of the reference implementation for the chosen
        ``flavor``: ``4`` for ``"harmony2"``, ``200`` for ``"harmony1"``.
    tol_harmony
        Convergence tolerance for the Harmony objective function.
        The algorithm stops when the relative change in objective falls below this value.
        If ``None``, ``1e-2`` for ``"harmony2"`` and ``1e-4`` for ``"harmony1"``.
    tol_clustering
        Convergence tolerance for the clustering step within each
        Harmony iteration.
        If ``None``, ``1e-3`` for ``"harmony2"`` and ``1e-5`` for ``"harmony1"``.
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
        Set to ``0`` to disable the diversity penalty for a batch variable.
        A scalar is applied to every key. A sequence may contain one value per
        key, expanded over that key's categorical levels, or one value per
        categorical level across all keys.
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
        Must be finite and greater than zero.
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
        Method for the correction step. Both solve the ridge regressions in
        closed form. ``"batched"`` corrects from exact right-hand sides, as
        many clusters per pass as fit in 1 GiB of workspace (clusters x
        batches x components), and normalizes in the same pass. ``"fast"``
        corrects one cluster at a time with the least workspace (float
        right-hand sides, a separate normalization pass, one GPU with
        ``multi_gpu``).
        With one key, ``None`` automatically selects ``"batched"`` unless one
        cluster's workspace would exceed 1 GiB, in which case ``"fast"`` is
        used.
        Multiple keys always use the exact general-design solve because the
        arrowhead optimization applies only to one batch variable; clusters are
        processed in workspace-bounded chunks when needed. For multiple keys,
        use ``None`` or ``"batched"``; passing ``"fast"`` emits a warning and
        uses the exact solve.
    colsum_algo
        No effect: column sums come from the fused clustering kernels. Kept
        for compatibility.
    block_proportion
        Proportion of cells updated per clustering sub-iteration.
        Smaller values produce more stochastic updates.
        Larger values are faster but may converge to different solutions.
        The blocks are drawn once per call and visited in a rotating order
        (the reference implementations redraw random blocks every iteration).
    shuffle_chunk_size
        Cells move between the update blocks in runs of this many neighbouring
        cells of the same batch (joint category with several keys), in random
        order within the batch, so the GPU writes contiguous rows. Every block
        takes an equal share of each stretch of about ``n_cells / 384``
        batch-sorted cells, so batches spanning several stretches are spread
        evenly; smaller batches are dealt in runs and can be spread less evenly
        than by independent random blocks (``1`` moves cells one at a time).
    rng
        Random seed or :class:`~numpy.random.Generator` for reproducibility.
        The superseded `random_state` argument is still accepted.
        Sequential calls with identical inputs, parameters, and integer seed
        are bitwise reproducible on the same hardware and software stack in
        float32 and float64. Initialization uses k-means++ and fixed-order
        Lloyd reductions. Results may differ across GPU architectures or
        CUDA versions; concurrent CUDA streams are outside this guarantee.
    verbose
        Whether to print the number of iterations until convergence.

    Returns
    -------
    Updates adata with the field ``adata.obsm[adjusted_basis]``,
    containing principal components adjusted by Harmony
    such that different experiments are integrated.
    """
    rng = np.random.default_rng(rng)
    bfloat16 = str(dtype) == "bfloat16" or getattr(dtype, "__name__", "") == "bfloat16"
    if bfloat16:
        dtype = np.float32
    if dtype is not None:
        dtype = np.dtype(dtype).newbyteorder("=")
        if dtype not in (np.float32, np.float64):
            raise ValueError(
                f"dtype must be float32, float64, 'bfloat16' or None, got {dtype}."
            )

    from ._harmony import _download, harmonize

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
            stacklevel=3,
        )
    if flavor == "harmony1":
        if alpha != 0.2:
            warnings.warn(
                "alpha is ignored when flavor='harmony1'; use ridge_lambda instead.",
                UserWarning,
                stacklevel=3,
            )
        if batch_prune_threshold != 1e-5:
            warnings.warn(
                "batch_prune_threshold is ignored when flavor='harmony1'.",
                UserWarning,
                stacklevel=3,
            )

    basis = _resolve_obsm_key(adata, basis)
    if adjusted_basis is None:
        adjusted_basis = _harmony_obsm_key(basis)

    # Ensure the basis exists in adata.obsm
    if basis not in adata.obsm:
        raise ValueError(
            f"The specified basis '{basis}' is not available in adata.obsm. "
            f"Available bases: {list(adata.obsm.keys())}"
        )

    # Get the input data
    input_data = adata.obsm[basis]
    if dtype is None:  # the input's float type, as in earlier releases
        if not hasattr(input_data, "dtype"):  # e.g. a pandas DataFrame
            input_data = np.asarray(input_data)
        dtype = np.dtype(input_data.dtype).newbyteorder("=")
        if dtype not in (np.float32, np.float64):
            dtype = np.dtype(np.float32)

    try:
        # Handle different array types
        if isinstance(input_data, np.ndarray):
            # NumPy array: Harmony uploads it in chunks, so no unsorted device
            # copy is kept. Harmony rejects NaN and infinite values.
            X = np.ascontiguousarray(input_data, dtype=dtype)
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

    except Exception as e:
        raise RuntimeError(f"Error preparing data for Harmony: {str(e)}") from e

    # Fault in the host output while the GPU works; copying into touched pages
    # is about twice as fast as into a fresh allocation.
    host_out = np.empty(X.shape, dtype=X.dtype)
    prefault = threading.Thread(target=host_out.fill, args=(0,), daemon=True)
    prefault.start()
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
        shuffle_chunk_size=shuffle_chunk_size,
        theta=theta,
        tau=tau,
        correction_method=correction_method,
        colsum_algo=colsum_algo,
        rng=rng,
        stabilized_penalty=stabilized_penalty,
        dynamic_lambda=dynamic_lambda,
        alpha=alpha,
        batch_prune_threshold=batch_prune_threshold,
        verbose=verbose,
        bfloat16=bfloat16,
    )

    prefault.join()
    _download(harmony_out, host_out)
    adata.obsm[adjusted_basis] = host_out
