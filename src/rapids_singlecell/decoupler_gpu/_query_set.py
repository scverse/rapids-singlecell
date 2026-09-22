from __future__ import annotations

from typing import TYPE_CHECKING, Literal

import numpy as np
import pandas as pd
from scipy.stats import false_discovery_control

from rapids_singlecell.decoupler_gpu._helper._log import _log
from rapids_singlecell.decoupler_gpu._helper._net import prune
from rapids_singlecell.decoupler_gpu._method_ora import _fisher, _log_odds

if TYPE_CHECKING:
    from collections.abc import Iterable


def query_set(  # noqa: PLR0917 - Preserve decoupler's positional API.
    features: Iterable[str],
    net: pd.DataFrame,
    alternative: Literal["greater", "less", "two-sided"] = "greater",
    n_bg: int | float | None = 20_000,
    ha_corr: int | float = 0.5,
    tmin: int | float = 5,
    verbose: bool = False,  # noqa: FBT001, FBT002
    *,
    background: Iterable[str] | None = None,
) -> pd.DataFrame:
    """Test a list of features for overlap with each feature set in a network.

    Parameters
    ----------
    features
        Selected names; deduplicated, including names absent from the network.
    net
        ``source``/``target`` sets; weights ignored, duplicate pairs rejected.
    alternative
        Fisher test alternative: ``'greater'`` for enrichment, ``'less'`` for
        depletion, or ``'two-sided'`` for either direction.
    n_bg
        Integer-truncated background size; must fit each query-set union.
        ``None`` or zero uses the union of query and pruned network targets.
    background
        Background names, deduplicated, containing all selected features. Its size
        replaces ``n_bg``; other network targets are removed before ``tmin``.
    ha_corr
        Haldane-Anscombe correction for log odds ratios; p-values are unaffected.
    tmin
        Minimum targets per source within the background, before query overlap.
    verbose
        Whether to log progress.

    Returns
    -------
    DataFrame with columns ``source``, ``stat`` (natural log corrected odds
    ratio), ``pval`` (Fisher p-value), and ``padj`` (Benjamini-Hochberg adjusted
    p-value), sorted by ``padj`` and then ``pval``. Includes zero-overlap sets.
    """
    assert hasattr(features, "__iter__") and not isinstance(features, str | bytes), (
        "features must be an iterable collection such as a list"
    )
    features = set(features)
    assert alternative in ("greater", "less", "two-sided"), (
        "alternative must be 'greater', 'less', or 'two-sided'"
    )
    assert n_bg is None or (
        isinstance(n_bg, int | float) and np.isfinite(n_bg) and n_bg >= 0
    ), "n_bg must be finite, numeric and >= 0, or None"
    assert isinstance(ha_corr, int | float) and np.isfinite(ha_corr), (
        "ha_corr must be finite and numeric"
    )
    if background is not None:
        if not hasattr(background, "__iter__") or isinstance(background, str | bytes):
            raise ValueError("background must be an iterable of feature names")
        background = set(background)
        if not background or pd.Index(list(background)).hasnans:
            raise ValueError("background must contain non-missing feature names")
        if not features.issubset(background):
            raise ValueError("All selected features must belong to background")
    net = prune(
        features=None if background is None else list(background),
        net=net,
        tmin=tmin,
        verbose=verbose,
    )
    # Preserve source encounter order for tied results, as in decoupler.
    codes, sources = pd.factorize(net["source"], sort=False)
    sizes = np.bincount(codes)
    hit = net["target"].isin(features).to_numpy()
    a = np.bincount(codes[hit], minlength=len(sources))
    count = len(features)
    if background is not None:
        background_size = len(background)
    elif n_bg:
        background_size = int(n_bg)
    else:
        background_size = len(features.union(net["target"]))
    if max(background_size, count, int(sizes.max())) > np.iinfo(np.int32).max:
        raise ValueError("query_set feature counts and n_bg must fit in int32")
    b, c = sizes - a, count - a
    d = background_size - sizes - c
    if np.any(d < 0):
        source = sources[np.flatnonzero(d < 0)[0]]
        raise ValueError(
            f"n_bg={n_bg} is too small for the union of the query and source={source}; "
            "increase n_bg or use n_bg=None"
        )
    _log(
        f"query_set - testing {count} selected features against {len(sources)} sets",
        level="info",
        verbose=verbose,
    )
    pv = _fisher(a, count, sizes, background_size, alternative=alternative).get()
    # Match decoupler's scalar odds-ratio calculation for zero corrections.
    if np.any(((b + ha_corr) * (c + ha_corr)) == 0):
        raise ZeroDivisionError("division by zero")
    with np.errstate(divide="ignore", invalid="ignore"):
        stat = _log_odds(a, b, c, d, ha_corr)
    result = pd.DataFrame({"source": sources.tolist(), "stat": stat, "pval": pv})
    result["padj"] = false_discovery_control(pv, method="bh")
    return result.sort_values(["padj", "pval"]).reset_index(drop=True)
