"""\
A zarr codec pipeline that reads chunks into GPU memory with KvikIO and decodes them with nvCOMP.

zarr loads this module through a `zarr.codec_pipeline` entry point whenever it looks up a
codec pipeline, so it stays separate from (and much lighter to import than) `rapids_singlecell`,
and KvikIO, nvCOMP and CuPy are only imported when chunks are read into GPU memory.
Use it through :mod:`rapids_singlecell.io`.
"""

from __future__ import annotations

from ._pipeline import KvikioCodecPipeline, release_memory

__all__ = ["KvikioCodecPipeline", "release_memory"]
