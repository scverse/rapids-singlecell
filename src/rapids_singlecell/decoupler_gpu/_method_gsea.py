from __future__ import annotations

import os
from threading import Lock

import cupy as cp
import numpy as np

from rapids_singlecell._cuda import _gsea_cuda as _gs
from rapids_singlecell.decoupler_gpu._helper._docs import docs
from rapids_singlecell.decoupler_gpu._helper._Method import Method, MethodMeta


def _permutations(nvar, times, seed, size=128):
    """Yield NumPy-exact membership permutations in bounded chunks."""
    rng = np.random.default_rng(seed)
    identity = np.arange(nvar, dtype=np.int32)
    for start in range(0, times, size):
        rows = np.tile(identity, (min(size, times - start), 1))
        # Seed zero keeps the identity permutations, matching decoupler.
        if seed:
            rng.permuted(rows, axis=1, out=rows)
        yield rows


class _PermutationCache:
    """One invocation's bounded, stream-safe transposed permutation cache."""

    def __init__(self, max_bytes=256 * 1024**2):
        self.max_bytes = max_bytes
        self.entry = None
        self.lock = Lock()

    def __getstate__(self):
        return {"max_bytes": self.max_bytes}

    def __setstate__(self, state):
        self.__init__(**state)

    def run(self, nvar, times, seed, consume, *, forward=False, size=128):
        chunk_size = max(1, min(size, 64 * 1024**2 // max(1, 8 * nvar)))
        key = (os.getpid(), cp.cuda.runtime.getDevice(), nvar, times, seed, forward)
        stream = cp.cuda.get_current_stream()
        # Inverse indices have shape (features, permutations);
        # forward indices have shape (permutations, features).
        with self.lock:
            if self.entry is not None:
                cached_key, inverse_transposed, forward_permutations, ready = self.entry
                stream.wait_event(ready)
                if cached_key == key:
                    step = chunk_size if forward else times
                    for start in range(0, times, step):
                        consume(
                            inverse_transposed[:, start : start + step],
                            None
                            if forward_permutations is None
                            else forward_permutations[start : start + step],
                        )
                    ready.record(stream)
                    return
            self.entry = None
            cache_fits = 4 * times * nvar * (1 + forward) <= self.max_bytes
            inverse_transposed = (
                cp.empty((nvar, times), dtype=cp.int32) if cache_fits else None
            )
            forward_permutations = (
                cp.empty((times, nvar), dtype=cp.int32)
                if cache_fits and forward
                else None
            )
            start = 0
            for host_permutations in _permutations(nvar, times, seed, chunk_size):
                stop = start + len(host_permutations)
                forward_chunk = cp.asarray(host_permutations)
                inverse_chunk = cp.empty_like(forward_chunk)
                inverse_chunk[
                    cp.arange(len(host_permutations))[:, None], forward_chunk
                ] = cp.arange(nvar, dtype=cp.int32)
                if cache_fits:
                    inverse_transposed[:, start:stop] = inverse_chunk.T
                    inverse_transposed_chunk = inverse_transposed[:, start:stop]
                    if forward:
                        forward_permutations[start:stop] = forward_chunk
                else:
                    inverse_transposed_chunk = cp.ascontiguousarray(inverse_chunk.T)
                consume(inverse_transposed_chunk, forward_chunk if forward else None)
                start = stop
            if cache_fits:
                ready = cp.cuda.Event(disable_timing=True)
                ready.record(stream)
                self.entry = key, inverse_transposed, forward_permutations, ready


@docs.dedent
def _func_gsea(
    mat,
    *,
    cnct,
    starts,
    offsets,
    times=1000,
    seed=42,
    verbose=False,
    _permutation_cache=None,
):
    r"""
    Gene Set Enrichment Analysis (GSEA).

    Rank features with decoupler's Numba tie ordering and compute signed,
    absolute-value-weighted running-sum extrema. Network weights are ignored.
    With ``times > 1``, membership permutations give same-sign normalized scores
    and empirical p-values; otherwise return raw scores and p-values of one.
    Zero-weight sets score zero. Sensitive comparisons replay the float64 walk.
    ``seed=0`` uses identity permutations, matching decoupler. A bounded 256 MiB
    device cache shares permutations across observations within each invocation.

    %(yestest)s

    %(params)s
    %(times)s
    %(seed)s

    %(returns)s
    """
    for name, value in (("times", times), ("seed", seed)):
        assert isinstance(value, int | float) and np.isfinite(value) and value >= 0, (
            f"{name} must be numeric and >= 0"
        )
    times, seed = int(times), int(seed)
    mat = cp.ascontiguousarray(mat, dtype=cp.float32)
    nobs, nvar = mat.shape
    set_sizes, set_starts = offsets.get(), starts.get()
    if np.any((set_sizes <= 0) | (set_sizes > nvar)):
        raise ValueError("Feature sets must contain between 1 and n_features targets")
    order = cp.empty(mat.shape, dtype=cp.int32)
    ranks = cp.empty_like(order)
    stream = cp.cuda.get_current_stream().ptr
    _gs.rank(mat, order, ranks, stream=stream)
    values = cp.ascontiguousarray(cp.take_along_axis(mat, order, axis=1))
    value_bits = values.view(cp.int32)
    # Sparse rows retain the positive prefix; general inputs retain every rank.
    hit_limits = cp.sum(value_bits > 0, axis=1, dtype=cp.int32)
    skip_zero_weights = bool(
        cp.all(
            ((value_bits[:, -1] >= 0) | (value_bits[:, -1] == -2147483648))
            & (hit_limits * 2 < nvar)
        )
    )
    if not skip_zero_weights:
        hit_limits.fill(nvar)
    groups, previous_capacity = [], 0
    for capacity in (8, 16, 32, 64, 128, 256, 512, 1024, 2048):
        source_ids = np.flatnonzero(
            (set_sizes > previous_capacity)
            & (set_sizes <= capacity)
            & (set_sizes < nvar)
        )
        if source_ids.size:
            groups.append((capacity, cp.asarray(source_ids, dtype=cp.int32)))
        previous_capacity = capacity
    large_sources = np.flatnonzero((set_sizes > 2048) & (set_sizes < nvar))
    large_sources_gpu = cp.asarray(large_sources, dtype=cp.int32)
    large_membership = cp.zeros((len(large_sources), nvar), dtype=cp.bool_)
    for i, source in enumerate(large_sources):
        large_membership[
            i, cnct[set_starts[source] : set_starts[source] + set_sizes[source]]
        ] = True
    es = cp.zeros((nobs, len(set_sizes)), dtype=cp.float64)
    same_count = cp.zeros(es.shape, dtype=cp.int64)
    extreme_count = cp.zeros_like(same_count)
    null_sum = cp.zeros_like(es)

    def score_sets(inverse_transposed, **kwargs):
        for capacity, sources in groups:
            _gs.sparse(
                values,
                ranks,
                cnct,
                starts,
                offsets,
                sources,
                hit_limits,
                inverse_transposed,
                es,
                capacity=capacity,
                stream=stream,
                **kwargs,
            )

    def score_large_sets(forward_permutations):
        result = cp.empty(
            (nobs, len(forward_permutations), len(large_sources)), dtype=cp.float64
        )
        _gs.dense(
            values, order, large_membership, forward_permutations, result, stream=stream
        )
        return result

    identity = cp.arange(nvar, dtype=cp.int32)[None, :]
    score_sets(identity.T)
    if large_sources.size:
        es[:, large_sources_gpu] = score_large_sets(identity)[:, 0, :]
    pv = cp.ones_like(es)
    if times > 1:

        def consume(inverse_transposed, forward_permutations):
            score_sets(
                inverse_transposed,
                null_sum=null_sum,
                same_count=same_count,
                extreme_count=extreme_count,
            )
            if large_sources.size:
                null_scores = score_large_sets(forward_permutations)
                reference = es[:, None, large_sources_gpu]
                same_sign = cp.where(reference >= 0, null_scores >= 0, null_scores < 0)
                same_count[:, large_sources_gpu] += same_sign.sum(axis=1)
                extreme_count[:, large_sources_gpu] += (
                    same_sign
                    & cp.where(
                        reference >= 0,
                        null_scores >= reference,
                        null_scores <= reference,
                    )
                ).sum(axis=1)
                null_sum[:, large_sources_gpu] += cp.where(
                    same_sign, cp.abs(null_scores), 0
                ).sum(axis=1)

        cache = _permutation_cache or _PermutationCache()
        size = max(1, min(128, 64 * 1024**2 // max(1, 8 * nobs * len(large_sources))))
        cache.run(
            nvar, times, seed, consume, forward=bool(large_sources.size), size=size
        )
        valid = (same_count > 0) & (null_sum > 0)
        es = cp.where(valid, es * same_count / cp.where(valid, null_sum, 1), 0)
        pv = cp.where(valid, extreme_count / cp.maximum(same_count, 1), 1)
    full_sources = cp.asarray(np.flatnonzero(set_sizes == nvar), dtype=cp.int32)
    if full_sources.size:
        es[:, full_sources] = cp.any(value_bits & 0x7FFFFFFF, axis=1)[:, None]
    return es.get(), pv.get()


class GseaMethod(Method):
    """GSEA with a default batch size of 100 observations."""

    def __call__(self, data, net, *, bsize=100, **kwargs):
        kwargs.setdefault("_permutation_cache", _PermutationCache())
        return super().__call__(data, net, bsize=bsize, **kwargs)


gsea = GseaMethod(
    _method=MethodMeta(
        name="gsea",
        desc="Gene Set Enrichment Analysis (GSEA)",
        func=_func_gsea,
        stype="numerical",
        adj=False,
        weight=False,
        test=True,
        limits=(-np.inf, +np.inf),
        reference="https://doi.org/10.1073/pnas.0506580102",
    )
)
