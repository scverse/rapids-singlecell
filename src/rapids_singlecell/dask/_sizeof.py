"""Sizes of cupyx CSR and CSC matrices for Dask.

Dask registers none (``sizeof`` reports 48 bytes for any of them), so its scheduler moves GPU
chunks of gigabytes between workers as if they were free, e.g. when balancing work across GPUs.
"""

from __future__ import annotations

from dask.sizeof import sizeof


def register() -> None:
    import cupyx.scipy.sparse as cpx

    # the classes themselves (not a base class): Dask caches the lookup of classes it already sized
    @sizeof.register((cpx.csr_matrix, cpx.csc_matrix))  # the sparse formats AnnData stores
    def _sizeof_cupyx_sparse(x) -> int:
        return int(x.data.nbytes + x.indices.nbytes + x.indptr.nbytes)


register()
