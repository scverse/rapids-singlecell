"""dsb normalization of CITE-seq protein (ADT) counts :cite:p:`Mule2022`.

GPU port of the dsb R package. ``mclust::Mclust``, ``limma::removeBatchEffect``
and ``prcomp`` were reimplemented from published descriptions and calibrated
against R as a black box.
"""

from __future__ import annotations

import re
import warnings
from typing import TYPE_CHECKING, Literal, NamedTuple, cast, get_args

import cupy as cp
import numpy as np
from cupyx.scipy import sparse as cpsparse
from scanpy._utils import view_to_actual
from scipy import sparse

from rapids_singlecell._compat import DaskArray
from rapids_singlecell.get import _get_obs_rep, _set_obs_rep

from ._utils import _sparse_to_dense

if TYPE_CHECKING:
    from collections.abc import Callable, Iterable

    import pandas as pd
    from anndata import AnnData
    from numpy.typing import DTypeLike

    from rapids_singlecell._utils._random import RNGLike, SeedLike

_SUBSET = 2000
_MAX_ITER = 100_000
# float32 fits whose decisions are this close (relative) to a threshold are refit in float64
_MARGIN = {np.dtype(np.float64): 0.0, np.dtype(np.float32): 2e-3}
_CHUNK = 65_536
_ISOTYPE_PATTERN = re.compile(r"sotype|Iso|iso|control|CTRL|ctrl|Ctrl|ontrol")  # codespell:ignore ontrol
_ScaleFactor = Literal["standardize", "mean_subtract"]
_VAR_KEYS = ("dsb_ambient_mean", "dsb_ambient_sd")
_OBS_KEYS = ("dsb_cell_background_mean", "dsb_technical_component")


def dsb(
    adata: AnnData,
    adata_empty: AnnData | None = None,
    *,
    isotype_controls: str | Iterable[str] | None = None,
    use_isotype_controls: bool = True,
    denoise_counts: bool = True,
    pseudocount: float | None = None,
    scale_factor: _ScaleFactor = "standardize",
    fast_km: bool = False,
    quantile_clip: tuple[float, float] | None = None,
    layer: str | None = None,
    key_added: str | None = None,
    dtype: DTypeLike = np.float32,
    rng: SeedLike | RNGLike | None = None,
    copy: bool = False,
) -> AnnData | None:
    """Normalize and denoise protein (ADT) counts with dsb :cite:p:`Mule2022`.

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

    Parameters
    ----------
    adata
        Cells-by-proteins raw ADT counts of cell-containing droplets.
    adata_empty
        Raw ADT counts of empty droplets with the same proteins (may be backed).
        If `None`, the background is estimated from the cells.
    isotype_controls
        Name(s) of isotype control proteins in `adata.var_names` for step 2.
    use_isotype_controls
        Use `isotype_controls` in step 2, as recommended by dsb. If `False`, only
        the per-cell background mean is regressed out (as in limma >= 3.66).
    denoise_counts
        Run step 2.
    pseudocount
        Added before log transformation. Defaults to 10 with `adata_empty` and
        1 without, as in dsb.
    scale_factor
        Step 1 with empty droplets: `'standardize'` or `'mean_subtract'`.
    fast_km
        Use the lower center of the optimal two-cluster k-means split as
        per-cell background.
    quantile_clip
        Clip every protein to these lower and upper quantiles (type 7) of its
        normalized values; dsb's `quantile.clipping` uses `(0.001, 0.9995)`.
    layer
        Layer with the counts in `adata` and `adata_empty`; `.X` if `None`.
    key_added
        Store the result in `adata.layers[key_added]` instead of overwriting
        `layer` (or `.X`).
    dtype
        Floating point type of the computations and the output (unlike
        ``scanpy.pp.dsb``, where it only sets the output). `float64` reproduces
        R. `float32` is faster and stays within ~1e-5 of `float64`; mixture
        fits whose decisions fall within float32 noise are refit in float64.

        Where R would fail on a model fit, rsc warns and continues: a cell
        whose mixture fit is singular in both models gets the lower k-means
        center as background, and EM stops after 100,000 iterations.
    rng
        Random number generator for the mixture fits of vectors longer than
        2000 values, which start from a random subset.
    copy
        Return a modified copy instead of updating `adata` in place.

    Returns
    -------
    Returns `adata` if `copy=True`, else updates it and sets the following.

    `.X` or `.layers[layer | key_added]`
        dsb normalized values (dense, `dtype`, on the device of the input).
    `.var['dsb_ambient_mean']`
        Per-protein background mean of ``log(counts + pseudocount)`` (0 for
        proteins whose fit failed).
    `.var['dsb_ambient_sd']`
        Empty droplet standard deviation (only with `adata_empty`).
    `.obs['dsb_cell_background_mean']`
        Per-cell background mean of step 2 (only with `denoise_counts`).
    `.obs['dsb_technical_component']`
        Covariate regressed out in step 2 (only with `denoise_counts`).
    """
    if scale_factor not in (vals := get_args(_ScaleFactor)):
        raise ValueError(f"`scale_factor` must be one of {list(vals)}, got {scale_factor!r}.")
    if adata_empty is None and scale_factor != "standardize":
        raise ValueError("`scale_factor` only applies with `adata_empty`.")
    if fast_km and not denoise_counts:
        raise ValueError("`fast_km` only applies with `denoise_counts=True`.")
    dtype = np.dtype(dtype)
    if dtype not in (np.float32, np.float64):
        raise ValueError("`dtype` must be float32 or float64.")
    if pseudocount is None:
        pseudocount = 1 if adata_empty is None else 10
    if not pseudocount > 0:
        raise ValueError(f"`pseudocount` must be positive, got {pseudocount}.")
    if quantile_clip is not None and not 0 <= quantile_clip[0] < quantile_clip[1] <= 1:
        raise ValueError(f"`quantile_clip` must be increasing quantiles in [0, 1], got {quantile_clip}.")
    proteins = adata.var_names
    if not proteins.is_unique:
        raise ValueError(f"`adata.var_names` are not unique: {list(proteins[proteins.duplicated()])}")
    iso_idx = _isotype_indices(
        proteins, isotype_controls, use_isotype_controls=use_isotype_controls, required=denoise_counts
    )
    if adata.n_obs < 2 or (denoise_counts and adata.n_vars < 2):
        raise ValueError("dsb needs at least 2 cells, and 2 proteins with `denoise_counts`.")
    empty_order = None
    if adata_empty is not None:
        empty_order = _match_proteins(adata_empty.var_names, proteins)
        if adata_empty.n_obs < 2:
            raise ValueError("`adata_empty` needs at least 2 empty droplets.")
        if layer is not None and layer not in adata_empty.layers:
            raise ValueError(f"`adata_empty` has no layer {layer!r}.")
    rng = np.random.default_rng(rng)

    if copy:
        adata = adata.copy()
    else:
        view_to_actual(adata)
    adata.var = cast("pd.DataFrame", adata.var).drop(columns=[k for k in _VAR_KEYS if k in adata.var])
    adata.obs = cast("pd.DataFrame", adata.obs).drop(columns=[k for k in _OBS_KEYS if k in adata.obs])
    X_in = _get_obs_rep(adata, layer=layer)
    on_gpu = isinstance(X_in, cp.ndarray) or cpsparse.issparse(X_in)
    counts = _counts(X_in, "adata", dtype)
    scale = None
    if adata_empty is not None:
        # statistics in the order of `adata_empty`: reordering a backed matrix would load it
        mean, sd, constant = _empty_stats(_get_obs_rep(adata_empty, layer=layer), pseudocount)
        if empty_order is not None:
            mean, sd, constant = mean[cp.asarray(empty_order)], sd[cp.asarray(empty_order)], constant[empty_order]
        var_stats = {"dsb_ambient_mean": mean, "dsb_ambient_sd": sd}
        if scale_factor == "standardize":
            if constant.any():
                raise ValueError(
                    "Proteins with the same count in every empty droplet cannot be "
                    f"standardized; remove them from both objects: {list(proteins[constant])}"
                )
            if (low := sd < 0.05).any():
                warnings.warn(
                    "Proteins with low background log-count sd < 0.05, check "
                    f"their raw and normalized distributions: {list(proteins[cp.asnumpy(low)])}",
                    UserWarning,
                    stacklevel=2,
                )
            scale = sd
    else:
        mean = _mixture_fits(
            counts.T, rng, compress=True, transform=lambda v: cp.log(v.astype(np.float64) + pseudocount)
        ).params[:, 0]
        if (failed := cp.asnumpy(cp.isnan(mean))).any():
            warnings.warn(
                f"The background could not be fit for proteins {list(proteins[failed])}; they are only log transformed.",
                UserWarning,
                stacklevel=2,
            )
            mean[cp.asarray(failed)] = 0
        var_stats = {"dsb_ambient_mean": mean}

    def step1(out: cp.ndarray) -> cp.ndarray:
        out -= mean
        if scale is not None:
            out /= scale
        return out

    def exact_cells(rows) -> cp.ndarray:
        x = X_in[cp.asarray(rows) if on_gpu else rows]
        return step1(_log_counts(x, pseudocount, "adata", np.dtype(np.float64)))

    if dtype == np.float64:
        counts += pseudocount
        norm = step1(cp.log(counts, out=counts))
    else:
        # rounded once from float64, so values can only merge into ties (see warp_threshold)
        for r in range(0, counts.shape[0], _CHUNK):
            chunk = counts[r : r + _CHUNK].astype(np.float64)
            chunk += pseudocount
            counts[r : r + _CHUNK] = step1(cp.log(chunk, out=chunk))
        norm = counts
    del counts

    obs_stats = {}
    if denoise_counts:
        if fast_km:
            background = _kmeans_low_centers(norm, exact_cells)
        else:
            background = _mixture_fits(norm, rng, exact=exact_cells).params[:, 0].copy()
            if (n_failed := int((failed_cells := cp.isnan(background)).sum())) > 0:
                warnings.warn(
                    f"The mixture model could not be fit for {n_failed} "
                    "cells; using the lower k-means center of their values instead.",
                    UserWarning,
                    stacklevel=2,
                )
                background[failed_cells] = _kmeans_low_centers(norm[failed_cells])
        technical = background.astype(np.float64)
        if iso_idx is not None:
            noise = cp.concatenate([norm[:, iso_idx], background[:, None]], axis=1)
            constant = cp.asnumpy(noise.min(axis=0) == noise.max(axis=0))
            if constant.any():
                names = [*proteins[iso_idx], "cell background mean"]
                raise ValueError(
                    "Cannot standardize the technical component, constant across "
                    f"cells: {[n for n, c in zip(names, constant, strict=True) if c]}"
                )
            technical = _first_pc_scores(noise)
        if float(technical.max()) > float(technical.min()):
            vc = technical - technical.mean()
            slopes = sum(
                (norm[r : r + _CHUNK] * vc[r : r + _CHUNK, None].astype(dtype)).sum(axis=0, dtype=np.float64)
                for r in range(0, norm.shape[0], _CHUNK)
            ) / (vc @ vc)
            norm -= vc[:, None].astype(dtype) * slopes.astype(dtype)
        else:
            warnings.warn(
                "The technical component is constant; step 2 leaves the data unchanged.", UserWarning, stacklevel=2
            )
        obs_stats = {"dsb_cell_background_mean": background, "dsb_technical_component": technical}

    if quantile_clip is not None:
        q = cp.quantile(norm, cp.asarray(quantile_clip, dtype=dtype), axis=0)
        cp.clip(norm, q[0], q[1], out=norm)

    _set_obs_rep(adata, norm if on_gpu else cp.asnumpy(norm), layer=layer if key_added is None else key_added)
    for key, val in var_stats.items():
        adata.var[key] = cp.asnumpy(val).astype(dtype, copy=False)
    for key, val in obs_stats.items():
        adata.obs[key] = cp.asnumpy(val).astype(dtype, copy=False)
    return adata if copy else None


def _isotype_indices(proteins: pd.Index, isotype_controls, *, use_isotype_controls: bool, required: bool):
    if isinstance(isotype_controls, str):
        isotype_controls = [isotype_controls]
    elif isotype_controls is not None:
        isotype_controls = list(isotype_controls)
    if isotype_controls:
        if not use_isotype_controls:
            raise ValueError("Pass `isotype_controls` or `use_isotype_controls=False`, not both.")
        if missing := [p for p in isotype_controls if p not in proteins]:
            raise ValueError(f"isotype controls not found in `adata.var_names`: {missing}")
        return proteins.get_indexer(isotype_controls)
    if required and use_isotype_controls:
        detected = [p for p in proteins if _ISOTYPE_PATTERN.search(p)]
        raise ValueError(
            "Set `isotype_controls` to the names of the isotype controls, or "
            "`use_isotype_controls=False` to regress out only the per-cell "
            f"background (not recommended). Potential isotype controls: {detected}"
        )
    return None


def _counts(x, name: str, dtype: np.dtype) -> cp.ndarray:
    """Checked raw counts as a new dense C-ordered device array of `dtype`."""
    if x is None:
        raise ValueError(f"`{name}.X` is None; pass the `layer` with the counts.")
    if isinstance(x, DaskArray):
        raise TypeError("dsb does not support dask arrays.")
    if sparse.issparse(x):
        x = cpsparse.csr_matrix(sparse.csr_matrix(x, dtype=dtype))
    if cpsparse.issparse(x):
        out = _sparse_to_dense(x.tocsr(), order="C").astype(dtype, copy=False)
    else:
        out = cp.array(x, dtype=dtype, order="C")
    if not bool(cp.isfinite(out).all() and (out >= 0).all()):
        raise ValueError(f"`{name}` must contain raw counts: finite and non-negative values.")
    return out


def _log_counts(x, pseudocount: float, name: str, dtype: np.dtype) -> cp.ndarray:
    out = _counts(x, name, dtype)
    out += dtype.type(pseudocount)
    return cp.log(out, out=out)


def _empty_stats(x, pseudocount: float) -> tuple[cp.ndarray, cp.ndarray, np.ndarray]:
    """Per-protein float64 mean, sd and constancy of the log counts, in row chunks."""
    if x is None:
        raise ValueError("`adata_empty.X` is None; pass the `layer` with the counts.")
    n, n_vars = 0, x.shape[1]
    mean, m2 = cp.zeros(n_vars), cp.zeros(n_vars)
    lo, hi = cp.full(n_vars, np.inf), cp.full(n_vars, -np.inf)
    for start in range(0, x.shape[0], _CHUNK):
        chunk = _log_counts(x[start : start + _CHUNK], pseudocount, "adata_empty", np.dtype(np.float64))
        n_b, mean_b = chunk.shape[0], chunk.mean(axis=0)
        m2_b = ((chunk - mean_b) ** 2).sum(axis=0)
        # Chan et al.
        delta, n_ab = mean_b - mean, n + n_b
        mean = mean + delta * (n_b / n_ab)
        m2 = m2 + m2_b + delta**2 * (n * n_b / n_ab)
        n = n_ab
        lo, hi = cp.minimum(lo, chunk.min(axis=0)), cp.maximum(hi, chunk.max(axis=0))
        del chunk
    return mean, cp.sqrt(m2 / (n - 1)), cp.asnumpy(lo == hi)


def _match_proteins(names: pd.Index, proteins: pd.Index) -> np.ndarray | None:
    """Positions of `proteins` in `names`; None if in the same order."""
    if names.equals(proteins):
        return None
    if not names.is_unique or len(names) != len(proteins) or set(names) != set(proteins):
        diff = sorted(set(names).symmetric_difference(proteins))
        dup = list(names[names.duplicated()])
        raise ValueError(
            "`adata` and `adata_empty` have different proteins: "
            f"{diff[:20]}{f', duplicated: {dup[:20]}' if dup else ''}"
        )
    warnings.warn("Reordered the proteins of `adata_empty` to match `adata`.", UserWarning, stacklevel=3)
    return names.get_indexer(proteins)


class _MixtureFit(NamedTuple):
    """`params` (n, 6): mean0, mean1, var0, var1, pro0, pro1 (NaN = no fit); `capped`: hit the EM cap."""

    params: cp.ndarray
    capped: cp.ndarray


def _mixture_fits(
    M: cp.ndarray,
    rng: np.random.Generator,
    *,
    max_iter: int = _MAX_ITER,
    compress: bool = False,
    exact: Callable[[np.ndarray], cp.ndarray] | None = None,
    transform: Callable[[cp.ndarray], cp.ndarray] | None = None,
) -> _MixtureFit:
    """Fit ``Mclust(x, G = 2)`` to every row of `M`.

    Behaviour of mclust 6.1.3, established by black-box runs:

    * Start: hard split ``x < q``, ``q`` the type 7 quantile at the fraction
      ``j / k`` with the smallest ``k >= 2`` for which ``min < q < max``.
    * EM until ``|l_k - l_{k-1}| / (1 + |l_k|) < 1e-5``; a model with a
      variance ``<= eps`` is dropped. One more M-step when
      ``sum((pro - colMeans(z))^2) > sqrt(eps)``. Models E and V by BIC.
    * Vectors longer than 2000 values: split and first M-step on a random
      subset of 2000, extra M-step always applied.

    `compress` fits the distinct values of each row weighted by multiplicity,
    after the increasing map `transform`. Marginal float32 fits are refit
    from ``exact(rows)`` (float64).
    """
    n_fits, n = M.shape
    sub = None
    if n > _SUBSET:
        sub = np.array([rng.choice(n, _SUBSET, replace=False) for _ in range(n_fits)], dtype=np.intp).reshape(
            n_fits, _SUBSET
        )
    params, flags = _fit_rows(M, sub, max_iter=max_iter, compress=compress, transform=transform)
    if (replay := np.flatnonzero(cp.asnumpy(flags & 2))).size:
        rows = exact(replay) if exact is not None else M[cp.asarray(replay)].astype(np.float64)
        sub_r = None if sub is None else sub[replay]
        params_r, flags_r = _fit_rows(rows, sub_r, max_iter=max_iter, compress=compress, transform=transform)
        params[cp.asarray(replay)] = params_r.astype(M.dtype)
        flags[cp.asarray(replay)] = flags_r
    capped = (flags & 1).astype(bool)
    if n_capped := int(capped.sum()):
        warnings.warn(
            f"EM stopped at the iteration limit ({max_iter}) without "
            f"converging for {n_capped} fits; these may differ from R.",
            UserWarning,
            stacklevel=3,
        )
    return _MixtureFit(params, capped)


def _fit_rows(
    M: cp.ndarray,
    sub: np.ndarray | None,
    *,
    max_iter: int,
    compress: bool,
    transform: Callable[[cp.ndarray], cp.ndarray] | None = None,
) -> tuple[cp.ndarray, cp.ndarray]:
    """Params and flags (see mix2_kernel) of the rows of `M`, started from their subsets `sub`."""
    from rapids_singlecell._cuda import _dsb_cuda as _dsb

    f = transform if transform is not None else (lambda v: v)
    n_fits = M.shape[0]
    dtype = f(M[:0, :0]).dtype
    params = cp.empty((n_fits, 6), dtype=dtype)
    flags = cp.zeros(n_fits, dtype=cp.uint8)
    none = cp.empty(0, dtype=dtype)
    kw = {"max_iter": max_iter, "margin": _MARGIN[dtype], "stream": cp.cuda.get_current_stream().ptr}
    if n_fits == 0:
        return params, flags
    if sub is None and not compress:
        _dsb.mix2(cp.ascontiguousarray(f(M)), none, none, none, params, flags, subset=False, **kw)
        return params, flags
    start = M if sub is None else cp.take_along_axis(M, cp.asarray(sub), axis=1)
    init = cp.empty((n_fits, 2, 6), dtype=dtype)
    _dsb.mix2(cp.ascontiguousarray(f(start)), none, none, init, none, flags, subset=False, **kw)
    if compress and (unique := _unique_counts(M)) is not None:
        x, w = f(unique[0]), unique[1].astype(dtype, copy=False)
    else:
        x, w = cp.ascontiguousarray(f(M)), none
    _dsb.mix2(x, w, init, none, params, flags, subset=sub is not None, **kw)
    return params, flags


def _unique_counts(M: cp.ndarray) -> tuple[cp.ndarray, cp.ndarray] | None:
    """Distinct values of every row and their multiplicities, padded with the row maximum at weight 0.

    None if the rows have too many distinct values for this to pay off.
    """
    n_rows, n = M.shape
    vals = cp.unique(M)
    if n_rows * vals.size > M.size:
        return None
    idx = cp.searchsorted(vals, M)
    idx += (cp.arange(n_rows) * vals.size)[:, None]
    counts = cp.bincount(idx.ravel(), minlength=n_rows * vals.size).reshape(n_rows, vals.size)
    del idx
    present = counts > 0
    row, k = cp.nonzero(present)
    col = (cp.cumsum(present, axis=1) - 1)[row, k]
    values = cp.full((n_rows, int(present.sum(axis=1).max())), vals[-1], dtype=M.dtype)
    weights = cp.zeros(values.shape, dtype=M.dtype)
    values[row, col] = vals[k]
    weights[row, col] = counts[row, k]
    return values, weights


def _kmeans_low_centers(M: cp.ndarray, exact: Callable[[np.ndarray], cp.ndarray] | None = None) -> cp.ndarray:
    """Lower center of the optimal 1-D two-means split of every row.

    float32 rows whose two best splits score within 1e-6 (relative) are redone from ``exact(rows)``.
    """
    n = M.shape[1]
    S = cp.sort(M, axis=1).astype(cp.float64)
    left = cp.cumsum(S, axis=1)[:, :-1]
    s = cp.arange(1, n, dtype=cp.float64)
    # maximizing this minimizes the within-cluster sum of squares; splits inside ties are skipped
    score = left**2 / s + (S.sum(axis=1, keepdims=True) - left) ** 2 / (n - s)
    score = cp.where(S[:, :-1] < S[:, 1:], score, -cp.inf)
    best = cp.argmax(score, axis=1)
    low = cp.take_along_axis(left, best[:, None], axis=1)[:, 0] / (best + 1)
    low = cp.where(S[:, 0] == S[:, -1], S[:, 0], low).astype(M.dtype)
    if exact is not None and M.dtype != np.float64 and n > 2:
        top = cp.take_along_axis(score, best[:, None], axis=1)[:, 0]
        cp.put_along_axis(score, best[:, None], -cp.inf, axis=1)
        tie = cp.asnumpy(top - score.max(axis=1) <= 1e-6 * cp.abs(top))
        if (rows := np.flatnonzero(tie)).size:
            low[cp.asarray(rows)] = _kmeans_low_centers(exact(rows)).astype(M.dtype)
    return low


def _first_pc_scores(A: cp.ndarray) -> cp.ndarray:
    """float64 first principal component scores of the standardized columns, oriented along the last."""
    from ._pca import _run_cuml_pca
    from ._utils import _get_mean_var

    A = A.astype(np.float64)
    mean, var = _get_mean_var(A)
    pca_func, scores = _run_cuml_pca((A - mean) / cp.sqrt(var), 1, svd_solver="covariance_eigh")
    scores = cp.asarray(scores[:, 0])
    return scores if float(pca_func.components_[0, -1]) >= 0 else -scores
