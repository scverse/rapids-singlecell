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

import cupy as cp
import pytest
import scanpy as sc

_CACHED_LOADERS = ("pbmc3k", "pbmc3k_processed", "pbmc68k_reduced", "paul15")


def pytest_configure(config):
    config.addinivalue_line("markers", "two_gpus: needs two GPUs (skipped with fewer)")


def pytest_collection_modifyitems(items):
    if cp.cuda.runtime.getDeviceCount() < 2:
        for item in items:
            if "two_gpus" in item.keywords:
                item.add_marker(pytest.mark.skip(reason="requires two GPUs"))


@pytest.fixture
def cupy_pool():
    """A fresh memory pool as CuPy's allocator during the test."""
    allocator, pool = cp.cuda.get_allocator(), cp.cuda.MemoryPool()
    cp.cuda.set_allocator(pool.malloc)
    try:
        yield pool
    finally:
        cp.cuda.set_allocator(allocator)
        for device in range(cp.cuda.runtime.getDeviceCount()):
            with cp.cuda.Device(device):
                pool.free_all_blocks()


@pytest.fixture(params=[False, True], ids=["host", "peer"])
def peer(request, monkeypatch):
    """Copies between GPUs 0 and 1 forced through the host, or by peer copies (skipped where none are validated)."""
    from rapids_singlecell._utils import _multi_gpu

    if not request.param:
        monkeypatch.setattr(_multi_gpu, "_peer_copy_works", lambda *_: False)
    elif not _multi_gpu._peer_copies_work([0, 1]):
        pytest.skip("no validated peer copies")
    return request.param


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
