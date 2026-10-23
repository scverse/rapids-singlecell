"""Setting up Dask clusters for out-of-core analysis on GPUs."""

from __future__ import annotations

from ._cluster import start_cluster

__all__ = ["start_cluster"]
