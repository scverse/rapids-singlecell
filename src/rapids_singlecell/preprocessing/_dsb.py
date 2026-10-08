"""dsb normalization of CITE-seq protein (ADT) counts :cite:p:`Mule2022`.

GPU port of the dsb R package (CC0), following ``DSBNormalizeProtein`` and
``ModelNegativeADTnorm``. The R functions dsb calls were reimplemented from
published descriptions and calibrated by black-box runs only (no mclust, limma
or R stats source was consulted):

* ``mclust::Mclust(x, G = 2)``: see ``_mixture_fits`` (Fraley & Raftery 2002;
  Scrucca et al. 2016).
* ``limma::removeBatchEffect(x, covariates = v)`` (Ritchie et al. 2015):
  per-protein least squares on ``[1, v]``, subtracting slope times the
  *uncentered* ``v``.
* ``prcomp(scale. = TRUE)`` and ``quantile`` (type 7): standard definitions.
"""

from __future__ import annotations

import re
import warnings
from typing import TYPE_CHECKING, Literal, NamedTuple

import cupy as cp
import numpy as np
from cupyx.scipy import sparse as cpsparse
from scipy import sparse

from rapids_singlecell._compat import DaskArray
from rapids_singlecell.get import _get_obs_rep, _set_obs_rep

if TYPE_CHECKING:
    from collections.abc import Sequence

    from anndata import AnnData

    from rapids_singlecell._utils._random import RNGLike, SeedLike

_SUBSET = 2000  # Mclust starts longer vectors from a random subset this size
# EM iteration cap (R has none); only a guard against fits that never converge
_MAX_ITER = {np.dtype(np.float64): 100_000, np.dtype(np.float32): 10_000}
_ISOTYPE_PATTERN = re.compile(r"sotype|Iso|iso|control|CTRL|ctrl|Ctrl|ontrol")


def dsb(
    adata: AnnData,
    adata_empty: AnnData | None = None,
    *,
    isotype_controls: Sequence[str] | None = None,
    denoise_counts: bool = True,
    pseudocount: float | None = None,
    scale_factor: Literal["standardize", "mean_subtract"] = "standardize",
    fast_km: bool = False,
    quantile_clipping: bool = False,
    quantile_clip: tuple[float, float] = (0.001, 0.9995),
    layer: str | None = None,
    key_added: str | None = None,
    dtype: type = np.float64,
    rng: SeedLike | RNGLike | None = None,
    copy: bool = False,
) -> AnnData | None:
    """\
    Normalize and denoise protein (ADT) counts with dsb :cite:p:`Mule2022`.

    With empty droplets (`adata_empty`) this mirrors ``dsb::DSBNormalizeProtein``:

    1. *Ambient correction*: ``log(counts + pseudocount)`` of every protein is
       standardized with the mean and standard deviation of that protein in
       the empty droplets (or only centered, `scale_factor='mean_subtract'`).
    2. *Technical denoising* (`denoise_counts`): the first-component mean of a
       two-component Gaussian mixture fit to each cell's values is its
       background level. With the isotype controls it defines a technical
       component (first principal component of the standardized isotype values
       and background means), which is regressed out of every protein.

    Without `adata_empty` this mirrors ``dsb::ModelNegativeADTnorm``: step 1
    subtracts, per protein, the first-component mean of a mixture fit across
    cells.

    The mixture fits reproduce ``mclust::Mclust(x, G = 2)``. Mclust starts
    vectors longer than 2000 values from a random subset of 2000, which affects
    the per-protein fits of the negative model with more than 2000 cells: these
    match R only in distribution and depend on `rng`.

    Parameters
    ----------
    adata
        Cells-by-proteins raw ADT counts of cell-containing droplets.
    adata_empty
        Raw ADT counts of empty droplets with the same proteins. If `None`,
        the background is estimated from the cells.
    isotype_controls
        Names of isotype control proteins in `adata.var_names` for step 2. If
        `None`, only the per-cell background mean is regressed out.
    denoise_counts
        Run step 2. If `False`, `isotype_controls` is ignored.
    pseudocount
        Added before log transformation. Defaults to 10 with `adata_empty` and
        1 without, as in dsb.
    scale_factor
        Step 1 with empty droplets: `'standardize'` or `'mean_subtract'`.
    fast_km
        Use the lower center of a two-cluster k-means as per-cell background.
        R's ``kmeans(nstart = 5)`` depends on random starts; here the exact
        optimum of the one-dimensional problem is computed.
    quantile_clipping
        Clip every protein to its `quantile_clip` quantiles (type 7).
    quantile_clip
        Lower and upper quantiles for `quantile_clipping`.
    layer
        Layer with the counts in `adata` and `adata_empty`; `.X` if `None`.
    key_added
        Store the result in `adata.layers[key_added]` instead of overwriting
        `layer` (or `.X`).
    dtype
        Floating point type of all computations and the output. `float64`
        matches R to ~1e-10. `float32` is several times faster and usually
        agrees to ~1e-4, but for about 1 in 10,000 cells with very slowly
        converging fits, rounding changes the EM stopping iteration and that
        cell's background estimate.

        Where R would fail, rsc warns and continues: a cell whose mixture fit
        is singular in both models (R errors) gets the mean of its values
        below the start split as background, and EM stops after 100,000
        (float64) or 10,000 (float32) iterations (R has no limit).
    rng
        Random number generator for the subsampled fits of long vectors.
    copy
        Return a modified copy instead of updating `adata` in place.

    Returns
    -------
    Returns `adata` if `copy=True`, else updates it and sets:

    `.X` or `.layers[layer | key_added]`
        dsb normalized values (dense, `dtype`, on the device of the input).
    `.var['dsb_ambient_mean']`
        Per-protein background mean of ``log(counts + pseudocount)``: empty
        droplet mean, or first mixture component without `adata_empty`.
    `.var['dsb_ambient_sd']`
        Empty droplet standard deviation (only with `adata_empty`).
    `.obs['dsb_cell_background_mean']`
        Per-cell background mean of step 2 (only with `denoise_counts`).
    `.obs['dsb_technical_component']`
        Covariate regressed out in step 2 (only with `denoise_counts`). Its sign
        is arbitrary in R; here it increases with the background mean.
    """
    if scale_factor not in ("standardize", "mean_subtract"):
        raise ValueError("`scale_factor` must be 'standardize' or 'mean_subtract'.")
    dtype = np.dtype(dtype)
    if dtype not in (np.float32, np.float64):
        raise ValueError("`dtype` must be float32 or float64.")
    rng = np.random.default_rng(rng)
    proteins = adata.var_names
    X_in = _get_obs_rep(adata, layer=layer)
    on_gpu = isinstance(X_in, cp.ndarray) or cpsparse.issparse(X_in)
    if pseudocount is None:
        pseudocount = 1 if adata_empty is None else 10
    adt_log = cp.log(_dense_gpu(X_in, dtype) + dtype.type(pseudocount))

    iso_idx = None
    if denoise_counts and isotype_controls is not None:
        missing = [p for p in isotype_controls if p not in proteins]
        if missing:
            raise ValueError(
                f"isotype controls not found in `adata.var_names`: {missing}"
            )
        iso_idx = proteins.get_indexer(list(isotype_controls))
    elif denoise_counts:
        detected = [p for p in proteins if _ISOTYPE_PATTERN.search(p)]
        warnings.warn(
            "Denoising without `isotype_controls` is not recommended when "
            f"isotype controls are available. Potential isotype controls: {detected}",
            UserWarning,
            stacklevel=2,
        )

    var_stats = {}
    if adata_empty is not None:
        empty = _get_obs_rep(_match_proteins(adata_empty, proteins), layer=layer)
        empty_log = cp.log(_dense_gpu(empty, dtype) + dtype.type(pseudocount))
        mean, sd = empty_log.mean(axis=0), empty_log.std(axis=0, ddof=1)
        var_stats = {"dsb_ambient_mean": mean, "dsb_ambient_sd": sd}
        if scale_factor == "standardize":
            low = cp.asnumpy(sd < 0.05)
            if low.any():
                warnings.warn(
                    "Proteins with low background log-count sd < 0.05, check "
                    f"their raw and normalized distributions: {list(proteins[low])}",
                    UserWarning,
                    stacklevel=2,
                )
            norm = (adt_log - mean) / sd
        else:
            norm = adt_log - mean
    else:
        fit = _mixture_fits(cp.ascontiguousarray(adt_log.T), rng)
        mean = fit.params[:, 0].copy()
        failed = cp.asnumpy(fit.status < 0)
        if failed.any():
            warnings.warn(
                f"The background could not be fit for proteins "
                f"{list(proteins[failed])}; they are only log transformed.",
                UserWarning,
                stacklevel=2,
            )
            mean[cp.asarray(failed)] = 0
        var_stats = {"dsb_ambient_mean": mean}
        norm = adt_log - mean
    del adt_log
    norm = cp.ascontiguousarray(norm)

    obs_stats = {}
    if denoise_counts:
        if fast_km:
            background = _kmeans_low_centers(norm)
        else:
            fit = _mixture_fits(norm, rng)
            failed = fit.status < 0
            if bool(failed.any()):
                warnings.warn(
                    f"The mixture model could not be fit for {int(failed.sum())} "
                    "cells; using the mean of their lower values instead.",
                    UserWarning,
                    stacklevel=2,
                )
            background = cp.where(failed, _lower_mean(norm, fit.q), fit.params[:, 0])
        technical = background
        if iso_idx is not None:
            technical = _first_pc_scores(
                cp.concatenate([norm[:, iso_idx], background[:, None]], axis=1)
            )
        # limma::removeBatchEffect: subtract the slope times the uncentered v;
        # a constant covariate leaves the data unchanged
        vc = technical - technical.mean()
        if float(vc @ vc) > 0:
            norm -= technical[:, None] * ((vc @ norm) / (vc @ vc))
        else:
            warnings.warn(
                "The technical component is constant; step 2 leaves the data "
                "unchanged.",
                UserWarning,
                stacklevel=2,
            )
        obs_stats = {
            "dsb_cell_background_mean": background,
            "dsb_technical_component": technical,
        }

    if quantile_clipping:
        q = cp.quantile(norm, cp.asarray(quantile_clip, dtype=dtype), axis=0)
        norm = cp.clip(norm, q[0], q[1])

    if copy:
        adata = adata.copy()
    out = norm if on_gpu else cp.asnumpy(norm)
    if key_added is None:
        _set_obs_rep(adata, out, layer=layer)
    else:
        adata.layers[key_added] = out
    for key, val in var_stats.items():
        adata.var[key] = cp.asnumpy(val)
    for key, val in obs_stats.items():
        adata.obs[key] = cp.asnumpy(val)
    return adata if copy else None


def _dense_gpu(X, dtype: np.dtype) -> cp.ndarray:
    if isinstance(X, DaskArray):
        raise TypeError("dsb does not support dask arrays.")
    # move the narrower of source and target dtype, cast on the GPU otherwise
    # (cupyx sparse needs floats, so integer scipy matrices cast on the host)
    narrow = X.dtype.itemsize > dtype.itemsize or X.dtype.kind != "f"
    if sparse.issparse(X):
        X = cpsparse.csr_matrix(X.astype(dtype) if narrow else X)
    if cpsparse.issparse(X):
        return X.astype(dtype).toarray()
    if X.dtype.itemsize > dtype.itemsize:
        return cp.array(X, dtype=dtype, order="C")
    return cp.asarray(X).astype(dtype, order="C", copy=False)


def _match_proteins(adata_empty: AnnData, proteins) -> AnnData:
    names = adata_empty.var_names
    if names.equals(proteins):
        return adata_empty
    if len(names) != len(proteins) or set(names) != set(proteins):
        diff = sorted(set(names).symmetric_difference(proteins))
        raise ValueError(
            f"`adata` and `adata_empty` have different proteins: {diff[:20]}"
        )
    warnings.warn(
        "Reordered the proteins of `adata_empty` to match `adata`.",
        UserWarning,
        stacklevel=3,
    )
    return adata_empty[:, proteins]


class _MixtureFit(NamedTuple):
    params: cp.ndarray  # (n, 6): mean0, mean1, var0, var1, pro0, loglik
    status: cp.ndarray  # 0 = E, 1 = V, -1 = no fit; +2 = hit the EM cap
    q: cp.ndarray  # start split (of the subset for long vectors)


def _mixture_fits(M: cp.ndarray, rng: np.random.Generator) -> _MixtureFit:
    """Fit ``Mclust(x, G = 2)`` to every row of `M`.

    Behaviour of mclust 6.1.3, established by black-box runs only:

    * Start: hard split ``x < q``, ``q`` the type 7 quantile at the fraction
      ``j / k`` with the smallest ``k >= 2`` for which ``min < q < max`` (the
      median for untied data). ``mean[1]`` is the component started on the
      lower class.
    * EM (M-step, E-step) until ``|l_k - l_{k-1}| / (1 + |l_k|) < 1e-5``; a
      model with a variance ``<= eps`` is dropped. One more M-step is applied
      when ``sum((pro - colMeans(z))^2) > sqrt(eps)``. The larger
      ``2 loglik - npar log(n)`` of models E and V wins.
    * Vectors longer than 2000 values: split and first M-step on a random
      subset of 2000, EM on all values, extra M-step always applied.
    """
    from rapids_singlecell._cuda import _dsb_cuda as _dsb

    n_fits, n = M.shape
    params = cp.empty((n_fits, 6), dtype=M.dtype)
    status = cp.empty(n_fits, dtype=cp.int32)
    q = cp.empty(n_fits, dtype=M.dtype)
    kw = {"max_iter": _MAX_ITER[M.dtype], "stream": cp.cuda.get_current_stream().ptr}
    if n <= _SUBSET:
        _dsb.mix2_warp(M, params, status, q, params[:0], **kw)
    else:
        idx = np.stack([rng.choice(n, _SUBSET, replace=False) for _ in range(n_fits)])
        sub = cp.take_along_axis(M, cp.asarray(idx), axis=1)
        init = cp.empty((n_fits, 2, 6), dtype=M.dtype)
        _dsb.mix2_warp(sub, params, status, q, init, **kw)
        nbytes = _dsb.mix2_coop_workspace(n_fits, n, M.dtype == np.float64)
        work = cp.empty(nbytes, dtype=cp.uint8)
        _dsb.mix2_coop(M, init, params, status, work, **kw)
    if n_capped := int((status >= 2).sum()):
        warnings.warn(
            f"EM stopped at the iteration limit ({kw['max_iter']}) without "
            f"converging for {n_capped} fits; these may differ from R.",
            UserWarning,
            stacklevel=3,
        )
    return _MixtureFit(params, status, q)


def _lower_mean(M: cp.ndarray, q: cp.ndarray) -> cp.ndarray:
    """Mean of the values below the start split (fallback background)."""
    below = M < q[:, None]
    mean = cp.where(below, M, 0).sum(axis=1) / cp.maximum(below.sum(axis=1), 1)
    return cp.where(below.any(axis=1), mean, M.mean(axis=1)).astype(M.dtype)


def _kmeans_low_centers(M: cp.ndarray) -> cp.ndarray:
    """Lower center of the optimal 1-D two-means split of every row."""
    n = M.shape[1]
    S = cp.sort(M, axis=1).astype(cp.float64)
    left = cp.cumsum(S, axis=1)[:, :-1]  # sums of the s smallest, s = 1..n-1
    s = cp.arange(1, n, dtype=cp.float64)
    # minimizing the within-cluster sum of squares maximizes this; splits
    # inside runs of tied values are not real partitions
    score = left**2 / s + (S.sum(axis=1, keepdims=True) - left) ** 2 / (n - s)
    score = cp.where(S[:, :-1] < S[:, 1:], score, -cp.inf)
    best = cp.argmax(score, axis=1)
    low = cp.take_along_axis(left, best[:, None], axis=1)[:, 0] / (best + 1)
    return cp.where(S[:, 0] == S[:, -1], S[:, 0], low).astype(M.dtype)


def _first_pc_scores(A: cp.ndarray) -> cp.ndarray:
    """First principal component scores of the standardized columns, oriented
    along the last column (the background mean)."""
    Z = (A - A.mean(axis=0)) / A.std(axis=0, ddof=1)
    v = cp.linalg.eigh(Z.T @ Z)[1][:, -1]
    return Z @ (v if float(v[-1]) >= 0 else -v)
