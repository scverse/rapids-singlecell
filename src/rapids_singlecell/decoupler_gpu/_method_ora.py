from __future__ import annotations

import math
from functools import lru_cache
from typing import Literal

import cupy as cp
import numba as nb
import numpy as np
from anndata import AnnData

from rapids_singlecell._cuda import _ora_cuda
from rapids_singlecell.decoupler_gpu._helper._docs import docs
from rapids_singlecell.decoupler_gpu._helper._log import _log
from rapids_singlecell.decoupler_gpu._helper._Method import Method, MethodMeta


@nb.njit(cache=True)
def _loggamma_range(start, stop, fraction):
    values = np.empty(stop - start + 1, dtype=np.float64)
    for i in range(values.size):
        values[i] = math.lgamma(start + i + fraction + 1.0)
    return values


@lru_cache(maxsize=8)
def _host_loggamma(count, max_size, background):
    """Reuse decoupler's CPU libm values; ULP differences affect Fisher ties."""
    whole = math.floor(background)
    fraction = background - whole
    start = max(-1 if fraction else 0, whole - count - max_size)
    front = _loggamma_range(0, max(count, max_size), 0.0)
    tail = _loggamma_range(start, whole, fraction)
    return np.concatenate((front, tail)), front.size - start


def _fisher(a, selected, size, background, *, alternative="two-sided", compat=False):
    """Shared Fisher tests for lists, masks, and ranked fallback tables."""
    a = cp.asarray(a, dtype=cp.int32, order="C")
    sizes = cp.asarray(size, dtype=cp.int32).reshape(-1)
    counts = cp.asarray(selected, dtype=cp.int32).reshape(-1)
    gamma, tail_offset = (
        _host_loggamma(int(selected), int(sizes.max()), float(background))
        if compat
        else (cp.empty(0, dtype=cp.float64), 0)
    )
    pv = cp.empty(a.shape, dtype=cp.float64)
    _ora_cuda.fisher(
        a,
        counts=counts,
        sizes=sizes,
        background=float(background),
        pv=pv,
        alternative={"two-sided": 0, "greater": 1, "less": 2}[alternative],
        gamma=cp.asarray(gamma),
        tail_offset=tail_offset,
        stream=cp.cuda.get_current_stream().ptr,
    )
    return pv


def _log_odds(a, b, c, d, correction):
    return _fused_log_odds(a, b, c, d, np.float64(correction))


@cp.fuse
def _fused_log_odds(a, b, c, d, correction):
    xp = cp.get_array_module(a)
    ratio = ((a + correction) * (d + correction)) / (
        (b + correction) * (c + correction)
    )
    return xp.where(ratio == 0, 0, xp.log(ratio))


@cp.fuse
def _invalid_overlap(a, counts, sizes, background):
    return cp.any(background - sizes - counts + a < 0)


def _select_ora(mat, n_up, n_bm):
    nobs, nvar = mat.shape
    selected = cp.empty((nvar, nobs), dtype=cp.uint8)
    _ora_cuda.select(
        cp.ascontiguousarray(mat, dtype=cp.float32),
        n_up,
        n_bm,
        selected,
        stream=cp.cuda.get_current_stream().ptr,
    )
    return selected


def _ora_tables(offsets, count, background, correction, *, limit=2_000_000):
    """Reuse Fisher results across equal-sized terms and overlap counts."""
    sizes, groups = cp.unique(offsets, return_inverse=True)
    width = min(count, int(sizes.max())) + 1
    if sizes.size * width > min(limit, 2_000_000):
        return None
    a = cp.broadcast_to(cp.arange(width, dtype=cp.int32), (sizes.size, width))
    b, c = sizes[:, None] - a, count - a
    d = float(background) - sizes[:, None] - c
    pv = _fisher(a, count, sizes, background, compat=True)
    return _log_odds(a, b, c, d, correction), pv, groups.astype(cp.int32)


def _validate_ora(nvar, n_up, n_bm, n_bg, ha_corr):
    """Validate and round the numbers of top- and bottom-ranked features."""
    if n_up is None:
        n_up = max(int(np.ceil(0.05 * nvar)), 2)
    if n_bg is None:
        n_bg = 0
    for name, value, positive in (
        ("n_up", n_up, True),
        ("n_bm", n_bm, False),
        ("n_bg", n_bg, False),
    ):
        assert isinstance(value, int | float) and np.isfinite(value), (
            f"{name} must be finite and numeric"
        )
        assert value > 0 if positive else value >= 0, (
            f"{name} must be {'> 0' if positive else '>= 0'}"
        )
    n_up, n_bm = int(np.ceil(n_up)), int(np.ceil(n_bm))
    assert n_up + n_bm <= nvar, (
        f"For nvar={nvar}, n_up={n_up} and n_bm={n_bm} overlap, decrease any of them"
    )
    assert n_bg == 0 or n_bg >= n_up + n_bm, (
        "n_bg must be at least the number of selected features n_up + n_bm"
    )
    assert isinstance(ha_corr, int | float) and np.isfinite(ha_corr), (
        "ha_corr must be finite and numeric"
    )
    n_bg = nvar if n_bg == 0 else n_bg
    if max(nvar, n_bg) > np.iinfo(np.int32).max:
        raise ValueError("ORA feature counts and n_bg must fit in int32")
    return n_up, n_bm, n_bg


@docs.dedent
def _func_ora(
    mat: cp.ndarray,
    *,
    cnct: cp.ndarray,
    starts: cp.ndarray,
    offsets: cp.ndarray,
    n_up: int | float | None = None,
    n_bm: int | float = 0,
    n_bg: int | float | None = 20_000,
    ha_corr: int | float = 0.5,
    value_cutoff: float | None = None,
    value_direction: Literal["greater", "less"] = "greater",
    alternative: Literal["two-sided", "greater", "less"] = "two-sided",
    verbose: bool = False,
) -> tuple[np.ndarray, np.ndarray]:
    r"""
    Over Representation Analysis (ORA).

    Tests feature-set overlaps with Fisher's exact test and corrected log odds.
    Input matrices use float32; Fisher calculations use float64.

    %(yestest)s

    %(params)s
    n_up
        Number of highest-ranked features to select. Defaults to the top 5%%
        of features, with a minimum of two. Fractional values are rounded up.
    n_bm
        Number of lowest-ranked features to select (default zero), rounded up.
        The sum of ``n_up`` and ``n_bm`` must not exceed the number of features.
    n_bg
        Background size (default 20,000), including in value-cutoff mode.
        ``None`` or zero uses all processed genes. Must accommodate the union
        of selected genes and each feature set.
    ha_corr
        Correction added to contingency-table entries for log odds ratios
        (default 0.5). Does not affect p-values; cutoffs require a nonnegative value.
    value_cutoff
        Optional strict value threshold replacing ranks. Cannot combine with
        ``n_up`` or nonzero ``n_bm``. Use 0.5 for binary DEG labels, or
        ``value_cutoff=0.05, value_direction='less'`` for adjusted p-values.

        Supply comparisons as rows and all tested genes as columns, including
        unannotated genes. Zero rows/columns are retained regardless of ``empty``;
        comparisons with no selected genes have p-values of one. Values must
        be finite; exclude untested genes. Use ``n_bg=None`` for this universe.
    value_direction
        Select values strictly ``'greater'`` (default) or ``'less'`` than
        ``value_cutoff``. Equality is excluded, independently of the Fisher tail.
    alternative
        Fisher alternative: ``'two-sided'`` (default), ``'greater'`` for
        enrichment, or ``'less'`` for depletion.

    %(returns)s

    """
    mat = cp.asarray(mat, dtype=cp.float32)
    nobs, nvar = mat.shape
    if alternative not in ("two-sided", "greater", "less"):
        raise ValueError("alternative must be 'two-sided', 'greater', or 'less'")
    if value_cutoff is not None:
        if not isinstance(value_cutoff, int | float) or not np.isfinite(value_cutoff):
            raise ValueError("value_cutoff must be finite and numeric")
        if value_direction not in ("greater", "less"):
            raise ValueError("value_direction must be 'greater' or 'less'")
        if n_up is not None or n_bm != 0:
            raise ValueError(
                "value_cutoff cannot be combined with n_up or nonzero n_bm"
            )
        # Cutoffs allow zero or all genes to be selected.
        if n_bg is None:
            n_bg = 0
        if not (
            isinstance(n_bg, int | float)
            and np.isfinite(n_bg)
            and n_bg >= 0
            and n_bg == int(n_bg)
        ):
            raise ValueError("n_bg must be a nonnegative integer or None")
        if not (
            isinstance(ha_corr, int | float) and np.isfinite(ha_corr) and ha_corr >= 0
        ):
            raise ValueError("ha_corr must be finite, numeric and nonnegative")
        if not bool(cp.isfinite(mat).all()):
            raise ValueError(
                "ORA value_cutoff input must contain only finite values; "
                "exclude untested genes before making DEG labels"
            )
        mask = (
            mat > value_cutoff if value_direction == "greater" else mat < value_cutoff
        )
        selected = cp.ascontiguousarray(mask.T, dtype=cp.uint8)
        n_bg = int(n_bg) if n_bg else nvar
        count = None
    else:
        n_up, n_bm, n_bg = _validate_ora(nvar, n_up, n_bm, n_bg, ha_corr)
        selected = _select_ora(mat, n_up, n_bm)
        count = n_up + n_bm
    if max(nvar, n_bg, cnct.size, starts.size) > np.iinfo(np.int32).max:
        raise ValueError("ORA feature counts, background and indices must fit in int32")
    _log(
        f"ora - testing {starts.size} terms across {nobs} observations with n_bg={n_bg}",
        level="info",
        verbose=verbose,
    )
    shape = (starts.size, nobs)
    kernel_args = {
        "cnct": cnct,
        "starts": starts,
        "offsets": offsets,
        "stream": cp.cuda.get_current_stream().ptr,
    }
    ranked = count is not None and alternative == "two-sided"
    tables = None
    if ranked:
        tables = _ora_tables(offsets, count, n_bg, ha_corr, limit=starts.size * nobs)
    if tables is not None:
        score_table, p_table, groups = tables
        es, pv = cp.empty(shape, dtype=cp.float64), cp.empty(shape, dtype=cp.float64)
        invalid = cp.zeros(1, dtype=cp.int32)
        _ora_cuda.ora_lookup(
            selected,
            **kernel_args,
            groups=groups,
            score_table=score_table,
            p_table=p_table,
            count=count,
            background=float(n_bg),
            es=es,
            pv=pv,
            invalid=invalid,
        )
        invalid = bool(invalid[0])
    else:
        a = cp.empty(shape, dtype=cp.int32)
        _ora_cuda.overlap(selected, **kernel_args, overlaps=a)
        counts = count if count is not None else selected.sum(axis=0, dtype=cp.int32)
        invalid = bool(_invalid_overlap(a, counts, offsets[:, None], np.float64(n_bg)))
    if invalid:
        raise ValueError(
            "n_bg is too small for the union of selected features and a term"
        )
    if tables is None:
        b, c = offsets[:, None] - a, counts - a
        d = float(n_bg) - offsets[:, None] - c
        es = _log_odds(a, b, c, d, ha_corr)
        pv = _fisher(a, counts, offsets, n_bg, alternative=alternative, compat=ranked)
    return es.T.get(), pv.T.get()


_ora = MethodMeta(
    name="ora",
    desc="Over Representation Analysis (ORA)",
    func=_func_ora,
    stype="categorical",
    adj=False,
    weight=False,
    test=True,
    limits=(-np.inf, +np.inf),
    reference="https://doi.org/10.2307/2340521",
)


class _OraMethod(Method):
    """Keep decoupler's public positional call contract and batching default."""

    def __call__(  # noqa: PLR0917 - Preserve decoupler's positional API.
        self,
        data,
        net,
        tmin: int | float = 5,
        raw: bool = False,  # noqa: FBT001, FBT002
        empty: bool = True,  # noqa: FBT001, FBT002
        bsize: int | float = 250_000,
        verbose: bool = False,  # noqa: FBT001, FBT002
        **kwargs,
    ):
        layer = kwargs.get("layer")
        assert layer is None or isinstance(layer, str), "layer must be str or None"
        assert isinstance(raw, bool), "raw must be bool"
        if isinstance(data, AnnData) and raw:
            assert data.raw is not None, "Received raw=True, but data.raw is empty"
            kwargs["layer"] = None  # Match decoupler's raw precedence.
        if kwargs.get("value_cutoff") is not None:
            empty = False  # Keep non-DEGs and comparisons without selected genes.
        return super().__call__(
            data,
            net,
            tmin=tmin,
            raw=raw,
            empty=empty,
            bsize=bsize,
            verbose=verbose,
            **kwargs,
        )


ora = _OraMethod(_method=_ora)
