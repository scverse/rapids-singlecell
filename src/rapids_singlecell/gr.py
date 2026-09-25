from __future__ import annotations

import sys

from .squidpy_gpu import *
from .squidpy_gpu import neighbors

# Importable like ``squidpy.gr.neighbors``.
sys.modules[f"{__name__}.neighbors"] = neighbors
