"""Native CUDA kernels provided by the Rust/cuda-oxide/PyO3 backend.

The extension loads without creating a CUDA context. Documentation builds may
omit it; a present but incompatible extension always raises its import error.
"""

from __future__ import annotations

import importlib
import sys

__all__ = [
    "_aggr_cuda",
    "_aucell_cuda",
    "_autocorr_cuda",
    "_bbknn_cuda",
    "_cooc_cuda",
    "_edistance_cuda",
    "_elementwise_cuda",
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
    "_pseudobulk_cuda",
    "_pv_cuda",
    "_qc_cuda",
    "_qc_dask_cuda",
    "_rank_stats_cuda",
    "_rank_stream_cuda",
    "_scale_cuda",
    "_sinkhorn_cuda",
    "_sparse2dense_cuda",
    "_spca_cuda",
    "_wilcoxon_binned_cuda",
    "_wilcoxon_cuda",
    "_wilcoxon_sparse_cuda",
]


def _register_rust_backend() -> None:
    fullname = f"{__name__}._rust_cuda"
    try:
        backend = importlib.import_module("._rust_cuda", __name__)
    except ModuleNotFoundError as exc:
        if exc.name != fullname:
            raise ImportError(f"Failed to load the Rust CUDA backend: {exc}") from exc
        # No native extension is intentional in documentation builds. Block
        # stale legacy binaries left behind by an older editable installation.
        for name in __all__:
            globals()[name] = None
            sys.modules[f"{__name__}.{name}"] = None
        return
    except ImportError as exc:
        raise ImportError(f"Failed to load the Rust CUDA backend: {exc}") from exc

    exported = set(backend.__all__)
    expected = set(__all__)
    if exported != expected:
        missing = sorted(expected - exported)
        unknown = sorted(exported - expected)
        raise ImportError(
            "Rust CUDA backend module mismatch: "
            f"missing={missing}, unknown={unknown}. Rebuild or reinstall the package."
        )

    # Validate the complete backend before publishing any submodule. Both
    # attribute and dotted imports must resolve to this exact native extension.
    modules = {name: getattr(backend, name) for name in __all__}
    if any(
        getattr(module, "__backend__", None) != "rust" for module in modules.values()
    ):
        raise ImportError("Rust CUDA backend contains an invalid native submodule")
    for name, module in modules.items():
        module.__name__ = f"{__name__}.{name}"
        module.__package__ = __name__
        module.__file__ = backend.__file__
        sys.modules[module.__name__] = module
        globals()[name] = module


_register_rust_backend()
