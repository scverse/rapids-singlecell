"""Bookkeeping for the shared-memory minor-axis reductions in ``_cuda/minor_tiles.cuh``."""

from __future__ import annotations


def _known_unsorted(X) -> bool:
    """
    Whether ``X`` is already known to have unsorted indices within its rows.

    Reads cupyx's cached flag without triggering its check kernel. An unknown
    flag returns ``False`` so the tiled kernel detects the order itself.
    """
    return getattr(X, "_has_canonical_format", None) is False


def _minor_reduce(X, kernel, *args, **kwargs) -> None:
    """
    Run a minor-axis reduction binding on ``X``.

    The binding tries the tile sweep unless ``X`` is known to be unsorted and
    reports when it detected unsorted rows; that is remembered on ``X`` so the
    next call goes straight to the atomic kernel.
    """
    if kernel(*args, assume_unsorted=_known_unsorted(X), **kwargs):
        X.has_canonical_format = False
