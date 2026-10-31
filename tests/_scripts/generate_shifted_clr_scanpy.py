"""Regenerate ``tests/_data/shifted_clr_scanpy.npz`` with scanpy's ``normalize_shifted_clr``.

Needs a scanpy that contains scverse/scanpy#4160 (merged 2026-10-22, commit 9039259). Run::

    python tests/_scripts/generate_shifted_clr_scanpy.py
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import scanpy as sc
from anndata import AnnData
from scanpy.preprocessing._normalization import _estimate_nb_overdispersion

rng = np.random.default_rng(0)
# RNA-like overdispersed counts with an empty cell and an empty gene
mu = rng.lognormal(0.5, 1.2, size=30)
counts = rng.negative_binomial(2, 2 / (2 + mu), size=(40, 30)).astype(np.float64)
counts[3] = 0
counts[:, 7] = 0

out = {"counts": counts, "alpha_estimated": np.float64(_estimate_nb_overdispersion(counts))}
for name, kwargs in {"estimated": {}, "alpha_0.3": {"alpha": 0.3}}.items():
    adata = AnnData(counts.copy())
    sc.pp.normalize_shifted_clr(adata, **kwargs)
    out[name] = np.asarray(adata.X)

np.savez_compressed(Path(__file__).parents[1] / "_data/shifted_clr_scanpy.npz", **out)
print(f"scanpy {sc.__version__}: alpha_estimated={out['alpha_estimated']:.6g}")
