"""
CUDA extensions for rapids-singlecell (built via scikit-build-core/nanobind).

These modules provide GPU-accelerated kernels for various single-cell analysis
operations. Each module is compiled from CUDA source files and exposed through
nanobind bindings.

On systems without compiled extensions (e.g., docs builds), a genuinely absent
module resolves to None so that module-level imports don't raise ImportError. A
module that is present but fails to load (ABI/toolkit mismatch, missing shared
library) is re-raised with context rather than silently swallowed.
"""

from __future__ import annotations

import importlib

__all__ = [
    "_aggr_cuda",
    "_aucell_cuda",
    "_autocorr_cuda",
    "_bbknn_cuda",
    "_cooc_cuda",
    "_edistance_cuda",
    "_guide_assignment_cuda",
    "_gmm_cuda",
    "_harmony_clustering_cuda",
    "_harmony_colsum_cuda",
    "_harmony_correction_batched_cuda",
    "_harmony_correction_cuda",
    "_harmony_kmeans_cuda",
    "_harmony_normalize_cuda",
    "_harmony_outer_cuda",
    "_harmony_pen_cuda",
    "_harmony_scatter_cuda",
    "_hvg_cuda",
    "_kde_cuda",
    "_ligrec_cuda",
    "_mean_var_cuda",
    "_mixscale_cuda",
    "_nanmean_cuda",
    "_nn_descent_cuda",
    "_norm_cuda",
    "_pr_cuda",
    "_pv_cuda",
    "_qc_cuda",
    "_qc_dask_cuda",
    "_rank_stats_cuda",
    "_scale_cuda",
    "_sparse2dense_cuda",
    "_spca_cuda",
    "_wilcoxon_binned_cuda",
    "_wilcoxon_cuda",
    "_wilcoxon_sparse_cuda",
]


def _preload_rapids_runtime_libs() -> None:
    """Pre-load ``librmm`` / ``rapids_logger`` so the extensions' ``DT_NEEDED``
    soname deps resolve regardless of import order (the editable-install
    ``RUNPATH`` is unreliable). Best-effort: absent wheels (docs builds) skip.
    """
    for mod in ("librmm", "rapids_logger"):
        try:
            importlib.import_module(mod).load_library()
        except (ImportError, OSError, AttributeError, RuntimeError):
            pass


_preload_rapids_runtime_libs()


def __getattr__(name: str):
    if name in __all__:
        try:
            return importlib.import_module(f".{name}", __name__)
        except ModuleNotFoundError:
            # Extension genuinely absent (e.g. docs builds, no-GPU installs):
            # degrade to None so module-level imports don't raise.
            return None
        except ImportError as exc:
            # Extension present but failed to load (ABI/toolkit mismatch, a
            # missing shared library, the rmm symbol-ordering issue, ...).
            # Surface it with context instead of silently returning None and
            # crashing later with a cryptic ``'NoneType' has no attribute ...``.
            msg = (
                f"Failed to load compiled CUDA extension {name!r}: {exc}. "
                "Ensure a matching rapids-singlecell-cuXX wheel (and librmm) is "
                "installed for your CUDA version."
            )
            raise ImportError(msg) from exc
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
