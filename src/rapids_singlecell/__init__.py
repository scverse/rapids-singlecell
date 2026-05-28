from __future__ import annotations

from . import dcg, get, gr, logging, pp, ptg, tl
from ._settings import Verbosity, settings
from ._version import __version__

__all__ = [
    "Verbosity",
    "__version__",
    "dcg",
    "get",
    "gr",
    "logging",
    "pp",
    "ptg",
    "settings",
    "tl",
]


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
