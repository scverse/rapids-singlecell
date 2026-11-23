"""Setting up Dask clusters for out-of-core analysis on GPUs."""

from __future__ import annotations

# registers the Dask sizes of cupyx CSR/CSC matrices
from . import _sizeof
from ._cluster import start_cluster

__all__ = ["start_cluster"]
