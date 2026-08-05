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
    "_jaccard_cuda",
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
    "_rank_stream_cuda",
    "_scale_cuda",
    "_sparse2dense_cuda",
    "_spca_cuda",
    "_wilcoxon_binned_cuda",
    "_wilcoxon_cuda",
    "_wilcoxon_sparse_cuda",
]


def _preload_rapids_runtime_libs() -> None:
    """Pre-load RAPIDS runtime libs so extension ``DT_NEEDED`` deps resolve."""
    for mod in ("librmm", "rapids_logger"):
        try:
            importlib.import_module(mod).load_library()
        except (ImportError, OSError, AttributeError, RuntimeError):
            pass


_preload_rapids_runtime_libs()

# Modules whose CUDA kernels use device scratch. They allocate through a
# CuPy-backed allocator injected here, so temporaries land on the caller's
# current device resource (RMM pool / UVM aware) without linking librmm.
_SCRATCH_MODULES = frozenset(
    {"_wilcoxon_cuda", "_wilcoxon_sparse_cuda", "_rank_stream_cuda"}
)
_scratch_allocator = None


def _get_scratch_allocator():
    """(alloc, free) backed by CuPy's current allocator; shared process-wide."""
    global _scratch_allocator
    if _scratch_allocator is None:
        import cupy as cp

        live = {}

        def _alloc(nbytes: int) -> int:
            mem = cp.cuda.alloc(int(nbytes))
            live[int(mem.ptr)] = mem
            return int(mem.ptr)

        def _free(ptr: int) -> None:
            live.pop(int(ptr), None)

        _scratch_allocator = (_alloc, _free)
    return _scratch_allocator


def __getattr__(name: str):
    if name in __all__:
        try:
            mod = importlib.import_module(f".{name}", __name__)
        except ModuleNotFoundError:
            # Extension genuinely absent (docs/no-GPU): degrade to None.
            return None
        except ImportError as exc:
            # Present but failed to load: surface ABI/toolkit/lib errors now.
            # Returning None would cause a later cryptic attribute error.
            msg = (
                f"Failed to load compiled CUDA extension {name!r}: {exc}. "
                "Ensure a matching rapids-singlecell-cuXX wheel (and librmm) is "
                "installed for your CUDA version."
            )
            raise ImportError(msg) from exc
        if name in _SCRATCH_MODULES:
            mod._set_scratch_allocator(*_get_scratch_allocator())
        return mod
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
