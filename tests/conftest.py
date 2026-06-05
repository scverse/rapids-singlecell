"""Session-wide test configuration.

Scanpy's bundled dataset loaders (``pbmc3k``, ``pbmc3k_processed``,
``pbmc68k_reduced``, ``paul15``) re-parse their h5ad/h5 from disk on *every*
call. Across this suite they are loaded ~100 times (``pbmc68k_reduced`` ~43x,
``pbmc3k`` ~50x, ``paul15`` ~4x), which is pure host-CPU/IO overhead — and
disproportionately expensive on the slow CI host.

We memoize each loader once per session and hand every caller an independent
``.copy()`` so existing tests can keep mutating their AnnData in place without
leaking state to other tests. Callers don't need to change: the loaders are
patched in place on ``scanpy.datasets`` before any test module is imported, so
both ``sc.datasets.pbmc3k()`` and ``from scanpy.datasets import pbmc3k`` pick up
the cached version.

Only the deterministic disk-backed loaders are cached. ``sc.datasets.blobs`` is
intentionally left alone (it synthesizes data per-call with varying parameters).
"""

from __future__ import annotations

import functools

import scanpy as sc

_CACHED_LOADERS = ("pbmc3k", "pbmc3k_processed", "pbmc68k_reduced", "paul15")


def _memoize_loader(loader):
    cache = {}

    @functools.wraps(loader)
    def wrapper(*args, **kwargs):
        try:
            key = (args, tuple(sorted(kwargs.items())))
            hash(key)
        except TypeError:
            # Unhashable arguments (not used in this suite) -> don't cache.
            return loader(*args, **kwargs)
        if key not in cache:
            cache[key] = loader(*args, **kwargs)
        return cache[key].copy()

    return wrapper


for _name in _CACHED_LOADERS:
    setattr(sc.datasets, _name, _memoize_loader(getattr(sc.datasets, _name)))
