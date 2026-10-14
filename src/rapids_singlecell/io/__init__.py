"""\
Reading data for out-of-core analysis on the GPU.

With KvikIO and nvCOMP installed (``pip install 'rapids-singlecell[io-cu13]'``),
zarr chunks are read straight into GPU memory and decompressed there.
"""

from __future__ import annotations

from rapids_singlecell_zarr import KvikioCodecPipeline, release_memory

from ._check import check
from ._config import enable
from ._convert import convert_zarr
from ._read import read_lazy

__all__ = [
    "KvikioCodecPipeline",
    "check",
    "convert_zarr",
    "enable",
    "read_lazy",
    "release_memory",
]
