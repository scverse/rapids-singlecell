"""Reading data for out-of-core analysis on the GPU.

With `zarr-kvikio <https://github.com/scverse/zarr-kvikio>`_, KvikIO and nvCOMP installed
(``pip install 'zarr-kvikio[cuda13]'``, or the `io` extra), zarr chunks are read straight into GPU memory
and decompressed there.
"""

from __future__ import annotations

from ._check import check
from ._config import enable, release_memory
from ._convert import convert_zarr
from ._read import read_lazy

__all__ = ["check", "convert_zarr", "enable", "read_lazy", "release_memory"]
