from __future__ import annotations

import pytest
import zarr

_KEYS = ("codec_pipeline.path", "buffer", "ndbuffer")


@pytest.fixture(autouse=True)
def restore_zarr_config():
    """`rapids_singlecell.io.enable()` (e.g. via `read_lazy`) changes zarr's global config."""
    before = {k: zarr.config.get(k) for k in _KEYS}
    yield
    zarr.config.set(before)
