from __future__ import annotations

import warnings
from typing import TYPE_CHECKING, Literal

import cupy as cp
import numpy as np
import pandas as pd
from scverse_misc import Deprecation, deprecated_arg

from rapids_singlecell._keys import _harmony_obsm_key, _resolve_obsm_key
from rapids_singlecell._utils._random import (
    RNGLike,
    SeedLike,
    _accepts_legacy_random_state,
)
from rapids_singlecell.get._out import DEPR_KEY_ADDED, DEPR_USE, _dest, _out_key, _resolve_obs
from rapids_singlecell.get._utils import AdRef, MultiAcc, _get_vec, _resolve_ref, _to_numpy_1d

if TYPE_CHECKING:
    from collections.abc import Sequence

    from anndata import AnnData

    from ._harmony import COLSUM_ALGO


@_accepts_legacy_random_state(0)
@deprecated_arg("key", Deprecation("0.19.0", "Use `batch` instead."))
@deprecated_arg("basis", DEPR_USE)
@deprecated_arg("adjusted_basis", DEPR_KEY_ADDED)
def harmony_integrate(
    adata: AnnData,
    batch: AdRef | str | Sequence[AdRef | str] | None = None,
    *,
    use: MultiAcc | str | None = None,
    out: MultiAcc | str | bool = False,
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
    rng: SeedLike | RNGLike | None = None,
    multi_gpu: bool | list[int] | str | None = False,
    verbose: bool = False,
    # deprecated
    key: str | list[str] | None = None,
    basis: str | None = None,
    adjusted_basis: str | None = None,
) -> np.ndarray | None:
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

        rsc.pp.harmony_integrate(adata, batch, flavor="harmony1")

    Parameters
    ----------
    adata
        The annotated data matrix.
    batch
        The annotation(s) that differentiate(s) among experiments/batches,
        e.g. `A.obs['batch']` or `[A.obs['batch'], A.obs['donor']]`.
        Strings name `.obs` columns, or are resolved by :meth:`anndata.acc.AdAcc.resolve`
        (e.g. `'obs.batch'`) if :attr:`rapids_singlecell.settings.preset` is `ScanpyV2Preview`.
        Multiple batch variables are modeled separately, with one active
        categorical level per variable and cell.
        To retain the joint-combination behavior of earlier releases, combine
        the desired columns into one categorical column and pass that.
    use
        Which embedding to adjust, e.g. `A.obsm['X_pca']` (or `'obsm.X_pca'`).
        By default, the one :func:`~rapids_singlecell.pp.pca` wrote under either preset's name.
    out
        Where to write the adjusted embedding, e.g. `A.obsm['pca_harmony']`.
        If `True`, it is returned instead.
        If `False`, it is written to `.obsm[k + '_harmony']` for `use=A.obsm[k]`.
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
        Cells join them in runs of up to 8 cells of the same batch, in random
        order within the batch, so a batch of fewer than
        ``8 / block_proportion`` cells is spread over fewer blocks.
    rng
        Random seed or :class:`~numpy.random.Generator` for reproducibility.
        The superseded `random_state` argument is still accepted.
        Sequential calls with identical inputs, parameters, and integer seed
        are bitwise reproducible on the same hardware and software stack in
        float32 and float64. Initialization uses k-means++ and fixed-order
        Lloyd reductions. Results may differ across GPU architectures or
        CUDA versions; concurrent CUDA streams are outside this guarantee.
    multi_gpu
        GPUs to split the cells across: ``True`` for all visible GPUs, a list
        or comma-separated string of device ids, or ``False``/``None`` for the
        current GPU. The result is bitwise identical to a run on one GPU of the
        same model. The fast correction runs on one GPU; it is also chosen
        for ``correction_method=None`` when one cluster's batched correction
        would need more than 1 GiB of workspace.
    verbose
        Whether to print the number of iterations until convergence.

    Returns
    -------
    The adjusted embedding if `out=True`, else `None`.
    It contains the principal components adjusted by Harmony,
    such that different experiments are integrated.
    """
    rng = np.random.default_rng(rng)
    bfloat16 = str(dtype) == "bfloat16" or getattr(dtype, "__name__", "") == "bfloat16"
    if bfloat16:
        dtype = np.float32
    if dtype is not None:
        try:
            dtype = np.dtype(dtype).newbyteorder("=")
        except TypeError:  # an unknown name, rejected below
            pass
        if dtype not in (np.float32, np.float64):
            raise ValueError(f"dtype must be float32, float64, 'bfloat16' or None, got {dtype}.")

    from rapids_singlecell._utils._multi_gpu import resolve_stream_devices

    from ._harmony import _SKIP, harmonize

    # Resolve flavor into internal flags
    if flavor not in {"harmony1", "harmony2"}:
        raise ValueError(f"flavor must be 'harmony1' or 'harmony2', got {flavor!r}.")
    stabilized_penalty = flavor == "harmony2"
    dynamic_lambda = flavor == "harmony2"

    # Warn when flavor-incompatible parameters are explicitly set
    if flavor == "harmony2" and ridge_lambda != 1.0:
        warnings.warn(
            "ridge_lambda is ignored when flavor='harmony2'; use alpha to control regularization strength.",
            UserWarning,
            skip_file_prefixes=_SKIP,
        )
    if flavor == "harmony1":
        if alpha != 0.2:
            warnings.warn(
                "alpha is ignored when flavor='harmony1'; use ridge_lambda instead.",
                UserWarning,
                skip_file_prefixes=_SKIP,
            )
        if batch_prune_threshold != 1e-5:
            warnings.warn(
                "batch_prune_threshold is ignored when flavor='harmony1'.",
                UserWarning,
                skip_file_prefixes=_SKIP,
            )

    if key is not None:
        if batch is not None:
            raise TypeError("Pass either `batch` or `key`, not both.")
        batch = key
    if batch is None:
        raise TypeError("harmony_integrate() missing required argument: 'batch'")
    if use is not None:
        if basis is not None:
            raise TypeError("Pass either `use` or `basis`, not both.")
        basis = _dest(_resolve_obs(use, kinds=(MultiAcc,)))["obsm"]
    else:
        basis = _resolve_obsm_key(adata, basis or "X_pca")
    dest = _out_key(out, adjusted_basis, kind="obsm", default=_harmony_obsm_key(basis))

    # Ensure the basis exists in adata.obsm
    if basis not in adata.obsm:
        raise ValueError(
            f"The specified basis '{basis}' is not available in adata.obsm. Available bases: {list(adata.obsm.keys())}"
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
            # NumPy array: Harmony uploads it in chunks that convert the dtype
            # and layout, so neither a host copy nor an unsorted device copy
            # is kept. Harmony rejects NaN and infinite values.
            X = input_data
        elif isinstance(input_data, cp.ndarray):
            # CuPy array: ensure correct dtype and layout, on its GPU
            with cp.cuda.Device(input_data.device.id):
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

    refs = _resolve_ref([batch] if isinstance(batch, AdRef | str) else batch)
    batch_df = pd.DataFrame(
        {
            str(r): v.array if isinstance(v, pd.Series) else _to_numpy_1d(v)
            for r, v in zip(refs, _get_vec(adata, refs, dim="obs"), strict=True)
        },
        index=adata.obs_names,
    )
    devices = resolve_stream_devices(multi_gpu=multi_gpu)
    host_out = np.empty(X.shape, dtype=dtype)
    with cp.cuda.Device(devices[0]):
        harmonize(
            X,
            batch_df,
            list(batch_df.columns),
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
            rng=rng,
            stabilized_penalty=stabilized_penalty,
            dynamic_lambda=dynamic_lambda,
            alpha=alpha,
            batch_prune_threshold=batch_prune_threshold,
            verbose=verbose,
            bfloat16=bfloat16,
            dtype=dtype,
            devices=devices,
            out=host_out,
        )
    if dest is True:
        return host_out
    adata.obsm[dest] = host_out
    return None
