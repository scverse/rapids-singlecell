# VIPER calculations adapted from decoupler 2.2.0; see LICENSE-decoupler.
from __future__ import annotations

import cupy as cp
import numpy as np
from cupyx.scipy.special import erfc, ndtri

from rapids_singlecell._cuda import _gsea_cuda as _gs
from rapids_singlecell._cuda import _viper_cuda as _vp
from rapids_singlecell.decoupler_gpu._helper._docs import docs
from rapids_singlecell.decoupler_gpu._helper._log import _log
from rapids_singlecell.decoupler_gpu._helper._Method import Method, MethodMeta

_PLEIOTROPY_BYTES = 1024**3


def _buffer(cache, name, shape, dtype=cp.float64):
    """Reuse flat storage so changing bucket widths remain C-contiguous."""
    size = shape[0] * shape[1]
    if name not in cache or cache[name].size < size:
        cache[name] = cp.empty(size, dtype=dtype)
    return cache[name][:size].reshape(shape)


def _unsigned_quantiles(quantiles):
    magnitude = cp.empty_like(quantiles)
    _vp.unsigned_quantiles(
        quantiles, magnitude, stream=cp.cuda.get_current_stream().ptr
    )
    return magnitude


def _sparse_network(adj):
    sources, targets = (cp.ascontiguousarray(x) for x in cp.nonzero(adj.T))
    starts = cp.searchsorted(sources, cp.arange(adj.shape[1] + 1))
    weights = cp.asarray(adj[targets, sources], dtype=cp.float64)
    return sources, targets, starts, weights


def _rankdata(mat: cp.ndarray, *, cache=None) -> cp.ndarray:
    """Rank each row with average ranks for ties, starting at one."""
    cache = {} if cache is None else cache
    mat = cp.ascontiguousarray(mat)
    if mat.dtype == cp.float32:
        order = _buffer(cache, "order", mat.shape, cp.int32)
        _gs.rank(
            mat,
            order,
            _buffer(cache, "sort_ranks", mat.shape, cp.int32),
            stream=cp.cuda.get_current_stream().ptr,
        )
        values = cp.take_along_axis(mat, order[:, ::-1], axis=1)
    else:
        values = cp.sort(mat, axis=1)
    ranks = _buffer(cache, "ranks", mat.shape)
    _vp.average_ranks(mat, values, ranks, stream=cp.cuda.get_current_stream().ptr)
    return ranks


def _get_inter_pvals(nes, mat, net, significant, n_targets, *, sparse_net, cache):
    """Rank and score active focal sources together on the GPU."""
    stream = cp.cuda.get_current_stream().ptr
    nsrc = nes.shape[1]
    _, edge_targets, starts, _ = sparse_net
    if "network" not in cache:
        support = (net != 0).astype(cp.float32 if net.shape[0] <= 2**24 else cp.float64)
        counts = (support.T @ support).astype(cp.float64)
        del support
        connected = counts > n_targets
        cp.fill_diagonal(connected, val=False)
        sizes = cp.diff(starts)
        cache["network"] = counts, connected.astype(cp.float32), sizes, int(sizes.max())
    counts, connected, sizes, max_size = cache["network"]
    active = significant & (significant.astype(cp.float32) @ connected > 0)
    rows, sources = cp.nonzero(active)
    pvals = _buffer(cache, "pvals", (rows.size, nsrc))
    pvals.fill(cp.nan)
    if not rows.size:
        return rows, sources, pvals
    # Bucket focal target counts so large regulons cannot pad every task.
    for exponent in range((max_size - 1).bit_length() + 1):
        width = 1 << exponent
        task = cp.flatnonzero((sizes[sources] > width // 2) & (sizes[sources] <= width))
        if not task.size:
            continue
        task_rows, task_sources = rows[task], sources[task]
        values = _buffer(cache, "values", (task.size, width), mat.dtype)
        _vp.gather_overlap(
            mat,
            edge_targets,
            starts,
            sizes,
            task_rows,
            task_sources,
            values,
            stream=stream,
        )
        ranks = _rankdata(values, cache=cache)
        tail = _buffer(cache, "tail", ranks.shape)
        signed = _buffer(cache, "signed", ranks.shape)
        _vp.transform_overlap(
            ranks,
            sizes,
            task_rows,
            task_sources,
            nes,
            tail,
            signed,
            stream=stream,
        )
        _vp.magnitude_overlap(tail, tail.max(axis=1), tail, stream=stream)
        scores = _buffer(cache, "scores", (task.size, nsrc))
        _vp.overlap_score(
            net,
            edge_targets,
            starts,
            sizes,
            task_rows,
            task_sources,
            signed,
            tail,
            significant,
            counts,
            scores,
            n_targets,
            stream=stream,
        )
        pvals[task] = scores
    return rows, sources, pvals


def _pleiotropy(
    nes: cp.ndarray,
    mat: cp.ndarray,
    net: cp.ndarray,
    quantiles: cp.ndarray,
    signed: cp.ndarray,
    *,
    reg_sign: float,
    n_targets: int,
    penalty: float,
    sparse_net: tuple,
    cache: dict,
) -> cp.ndarray:
    significant = cp.abs(nes) > reg_sign
    observations, sources, pvals = _get_inter_pvals(
        nes, mat, net, significant, n_targets, sparse_net=sparse_net, cache=cache
    )
    nsrc = net.shape[1]
    task, right = cp.nonzero(~cp.isnan(pvals) & (sources[:, None] < cp.arange(nsrc)))
    rows, left = observations[task], sources[task]
    if not rows.size:
        return nes
    lookup = cp.full(nes.shape, -1, dtype=cp.int64)
    lookup[observations, sources] = cp.arange(observations.size)
    reverse = pvals[lookup[rows, right], left]
    keep = ~cp.isnan(reverse)
    difference = cp.log10(reverse[keep]) - cp.log10(pvals[task[keep], right[keep]])
    rows, left, right = rows[keep], left[keep], right[keep]
    counts = cp.bincount(
        cp.concatenate((rows * nsrc + left, rows * nsrc + right)), minlength=nes.size
    ).reshape(nes.shape)
    first = difference > 0
    losers = cp.where(first, left, right)
    winners = cp.where(first, right, left)
    factors = (1 + cp.abs(difference)) ** (penalty / counts[rows, losers])
    magnitude = _unsigned_quantiles(quantiles)
    magnitude += (1 - cp.max(magnitude, axis=1, keepdims=True)) / 2
    return _correct(
        signed,
        ndtri(magnitude),
        net,
        nes,
        rows=rows,
        losers=losers,
        winners=winners,
        factors=factors,
        sparse_net=sparse_net,
        cache=cache,
    )


def _correct(
    signed,
    magnitude,
    adj,
    nes,
    *,
    rows,
    losers,
    winners,
    factors,
    sparse_net=None,
    cache=None,
):
    """Correct ordered pairs with storage proportional to network edges."""
    nobs, nvar = signed.shape
    cache = {} if cache is None else cache
    nsrc = adj.shape[1]
    sources, targets, starts, weights = (
        _sparse_network(adj) if sparse_net is None else sparse_net
    )
    nedge = sources.size
    if "penalty" not in cache:
        index = cp.full((nsrc, nvar), -1, dtype=cp.int32)
        index[sources, targets] = cp.arange(nedge, dtype=cp.int32)
        features = cp.flatnonzero(cp.bincount(targets, minlength=nvar) > 1)
        cache["penalty"] = index, features
    index, features = cache["penalty"]
    offsets = cp.searchsorted(rows, cp.arange(nobs + 1))
    likelihood = _buffer(cache, "likelihood", (nobs, nedge))
    likelihood.fill(1)
    losers, winners = losers.astype(cp.int32), winners.astype(cp.int32)
    stream = cp.cuda.get_current_stream().ptr
    _vp.penalize(
        index,
        offsets,
        losers,
        winners,
        factors,
        features,
        likelihood,
        stream=stream,
    )
    affected = _buffer(cache, "affected", nes.shape, cp.bool_)
    affected.fill(0)
    affected[rows, losers] = True
    result = _buffer(cache, "result", nes.shape)
    _vp.rescore(
        signed,
        magnitude,
        weights,
        targets,
        starts,
        likelihood,
        affected,
        nes,
        result,
        stream=stream,
    )
    return result


@docs.dedent
def _func_viper(
    mat: cp.ndarray,
    adj: cp.ndarray,
    *,
    pleiotropy: bool = True,
    reg_sign: float = 0.05,
    n_targets: int = 10,
    penalty: int | float = 20,
    verbose: bool = False,
) -> tuple[np.ndarray, np.ndarray]:
    r"""
    Virtual Inference of Protein-activity by Enriched Regulon analysis (VIPER).

    Rank molecular features within each observation and transform their ranks
    to normal quantiles. Combine signed and unsigned enrichment using the
    network weights to obtain normalized activity scores. Two-sided normal
    p-values are adjusted by Benjamini-Hochberg across sources.

    By default, pleiotropy correction penalizes shared targets of significantly
    enriched regulators and recomputes their activity scores.

    %(yestest)s

    %(params)s

    pleiotropy
        Whether to correct for pleiotropic regulation.
    reg_sign
        P-value threshold for selecting significant regulators for pleiotropy
        correction. Must lie strictly between zero and one.
    n_targets
        Shared-target count must be strictly greater than this nonnegative
        integer to apply pleiotropy correction, matching decoupler.
    penalty
        Positive penalty exponent for pleiotropic interactions.

    %(returns)s

    Notes
    -----
    Ties receive average ranks. As in decoupler, the unsigned rank transform
    uses a batch-wide adjustment. When all observations have tied extreme
    values, scores can depend on ``bsize`` or Dask chunk boundaries.

    Scoring and pleiotropy correction run entirely on the GPU. Signed sums
    within floating-point roundoff of zero are treated as zero when an
    unsigned contribution could amplify their sign. Such cancellation cases
    can differ from decoupler's CPU-dependent rounding.

    Example
    -------
    .. code-block:: python

        import decoupler as dc
        import rapids_singlecell as rsc

        adata, net = dc.ds.toy()
        rsc.dcg.viper(adata, net, tmin=3)
    """
    assert isinstance(pleiotropy, bool), "pleiotropy must be bool"
    assert (
        isinstance(reg_sign, int | float) and np.isfinite(reg_sign) and 0 < reg_sign < 1
    ), "reg_sign must be numeric and between 0 and 1"
    assert isinstance(n_targets, int) and n_targets >= 0, (
        "n_targets must be an integer and >= 0"
    )
    assert isinstance(penalty, int | float) and np.isfinite(penalty) and penalty > 0, (
        "penalty must be numeric and > 0"
    )
    mat, adj = cp.ascontiguousarray(mat), cp.ascontiguousarray(adj)
    if bool(cp.any(~cp.isfinite(adj))) or bool(cp.any(cp.all(adj == 0, axis=0))):
        raise ValueError("Each source must have finite weights and a nonzero target")
    _log(
        f"viper - calculating {adj.shape[1]} scores across {mat.shape[0]} observations",
        level="info",
        verbose=verbose,
    )
    quantiles = _rankdata(mat) / (mat.shape[1] + 1)
    magnitude = _unsigned_quantiles(quantiles)
    magnitude += (1 - cp.max(magnitude)) / 2
    sparse_net = _sparse_network(adj)
    _, targets, starts, weights = sparse_net
    counts = cp.diff(starts)
    signed = ndtri(quantiles)
    shape = (mat.shape[0], adj.shape[1])
    sum1, sum2 = (cp.empty(shape, dtype=cp.float64) for _ in range(2))
    _vp.initial_score(
        signed,
        ndtri(magnitude),
        weights,
        targets,
        starts,
        sum1,
        sum2,
        stream=cp.cuda.get_current_stream().ptr,
    )
    nes = (cp.abs(sum1) + cp.maximum(sum2, 0)) * cp.sign(sum1) * cp.sqrt(counts)
    if pleiotropy:
        _log(
            "viper - refining scores based on pleiotropy", level="info", verbose=verbose
        )
        threshold = float(ndtri(cp.asarray(1 - reg_sign / 2)))
        active = int(cp.max(cp.count_nonzero(cp.abs(nes) > threshold, axis=1)))
        # Bound overlap and likelihood scratch without changing the initial
        # batch-wide rank shift. Shadow rescoring is independent per row.
        # Retained rank buffers include padding to the next power of two.
        scratch = 64 * (active * adj.shape[1] + mat.shape[1]) + 96 * weights.size
        step = max(1, _PLEIOTROPY_BYTES // max(1, scratch))
        # Per-call state: network data and bounded scratch share this stream.
        cache = {}
        for start in range(0, mat.shape[0], step):
            stop = start + step
            nes[start:stop] = _pleiotropy(
                nes[start:stop],
                mat[start:stop],
                adj,
                quantiles[start:stop],
                signed[start:stop],
                reg_sign=threshold,
                n_targets=n_targets,
                penalty=penalty,
                sparse_net=sparse_net,
                cache=cache,
            )
    return nes.get(), erfc(cp.abs(nes) / cp.sqrt(2.0)).get()


viper = Method(
    _method=MethodMeta(
        name="viper",
        desc="Virtual Inference of Protein-activity by Enriched Regulon analysis (VIPER)",
        func=_func_viper,
        stype="numerical",
        adj=True,
        weight=True,
        test=True,
        limits=(-np.inf, +np.inf),
        reference="https://doi.org/10.1038/ng.3593",
    )
)
