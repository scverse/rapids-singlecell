from __future__ import annotations

import cuml.internals.logger as logger

from . import dcg, get, gr, pp, ptg, tl
from ._version import __version__


def _detect_duplicate_installation():
    """Warn if multiple rapids_singlecell variants are installed."""
    import importlib.metadata
    import warnings

    known = (
        "rapids-singlecell",
        "rapids-singlecell-cu12",
        "rapids-singlecell-cu13",
    )
    installed = []
    for pkg in known:
        try:
            importlib.metadata.distribution(pkg)
            installed.append(pkg)
        except importlib.metadata.PackageNotFoundError:
            pass

    if len(installed) > 1:
        pkg_list = ", ".join(sorted(installed))
        warnings.warn(
            f"\n"
            f"Multiple rapids_singlecell packages are installed: {pkg_list}\n"
            f"Please uninstall all versions and reinstall only one:\n"
            f"  pip uninstall {' '.join(sorted(installed))}\n",
            stacklevel=2,
        )


_detect_duplicate_installation()
logger.set_level(2)
