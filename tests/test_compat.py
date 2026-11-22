from __future__ import annotations

import subprocess
import sys
from inspect import signature

import anndata as ad
import numpy as np
import pytest
from numpy.testing import assert_array_equal

import rapids_singlecell as rsc
from rapids_singlecell._compat import _rng_kwargs
from rapids_singlecell._utils._random import _LegacyRng


def _legacy_paga_init(*, random_state) -> np.ndarray:
    """Model Scanpy <1.13 reseeding NumPy's global RNG from a scalar."""
    return np.random.RandomState(random_state).random(20)


def test_rng_kwargs_preserves_legacy_seed() -> None:
    expected = np.random.RandomState(0).random(20)

    rng = _LegacyRng(0)
    actual = _legacy_paga_init(**_rng_kwargs(_legacy_paga_init, rng))

    assert_array_equal(actual, expected)
    assert_array_equal(rng.random(20), expected)


# AnnData itself imports `anndata.acc` to slice, so nothing here subsets `adata`.
_NO_ACC = """
import anndata as ad, scanpy
sys.modules["anndata.acc"] = None
import numpy as np, pytest, rapids_singlecell as rsc

rng = np.random.default_rng(0)
means = rng.gamma(2.0, 1.0, (3, 50)) + 0.2
adata = ad.AnnData(rng.poisson(means[np.repeat(np.arange(3), 40)]).astype(np.float32))
adata.obs["batch"] = np.tile(["a", "b"], 60)
adata.layers["counts"] = adata.X.copy()
rsc.get.anndata_to_GPU(adata, convert_all=True)
rsc.pp.normalize_total(adata, target_sum=1e4)
rsc.pp.log1p(adata)
rsc.pp.highly_variable_genes(adata, n_top_genes=30)
rsc.pp.scale(adata, max_value=10)
rsc.pp.pca(adata, n_comps=10)
rsc.pp.neighbors(adata, n_neighbors=10)
rsc.tl.umap(adata)
rsc.tl.leiden(adata)
rsc.tl.rank_genes_groups(adata, "leiden", method="wilcoxon")
rsc.pp.highly_variable_genes(adata, n_top_genes=30, flavor="seurat_v3", layer="counts", batch_key="batch")
rsc.pp.normalize_total(adata, target_sum=1e4, layer="counts")
rsc.pp.pca(adata, n_comps=5, layer="counts", key_added="pca2")
rsc.pp.neighbors(adata, use_rep="pca2", key_added="nn2")
rsc.tl.umap(adata, neighbors_key="nn2", key_added="umap2")
rsc.tl.leiden(adata, neighbors_key="nn2", key_added="leiden2")
rsc.tl.rank_genes_groups(adata, "leiden2", method="wilcoxon", layer="counts", key_added="rgg2")
rsc.pp.dsb(adata, denoise_counts=False, layer="counts", key_added="dsb")
assert set(adata.obsm) == {"X_pca", "X_umap", "pca2", "umap2"} and "dsb" in adata.layers
assert rsc.preprocessing._scrublet._ref_to_uns("batch", None) == "batch"  # what scrublet(batch_key=...) stores
for call in (
    lambda: rsc.pp.normalize_total(adata, out="layers.x"),
    lambda: rsc.pp.pca(adata, out="obsm.x"),
    lambda: rsc.tl.leiden(adata, out="obs.x"),
    lambda: rsc.pp.pca(adata, mask="var.highly_variable"),
    lambda: rsc.pp.neighbors(adata, use="obsm.X_pca"),
):
    with pytest.raises(ImportError, match="version of `anndata` that provides `anndata.acc`"):
        call()
print(rsc.__file__)
"""


def test_default_and_legacy_paths_without_anndata_acc() -> None:
    """anndata 0.12 has no `anndata.acc`: only accessor and string requests may need it."""
    # The child gets this process' import path and finders, so it imports the same `rapids_singlecell`.
    finders = [type(finder).__name__ for finder in sys.meta_path]
    setup = (
        f"import sys\nsys.path[:] = {sys.path!r}\n"
        f"sys.meta_path[:] = [f for f in sys.meta_path if type(f).__name__ in {finders!r}]\n"
    )
    run = subprocess.run([sys.executable, "-c", setup + _NO_ACC], capture_output=True, text=True, timeout=50)
    assert (run.returncode, run.stdout.splitlines()[-1:]) == (0, [rsc.__file__]), run.stderr


def test_string_specs_without_resolve_vec(monkeypatch) -> None:
    """anndata 0.13.0 has `anndata.acc`, but its `A.resolve` has no `vec` parameter."""
    acc = pytest.importorskip("anndata.acc").A
    if "vec" in signature(resolve := acc.resolve).parameters:
        monkeypatch.setattr(
            type(acc), "resolve", lambda self, spec, *, strict=True: resolve(spec, strict=strict, vec=True)
        )
    rng = np.random.default_rng(0)
    adata = ad.AnnData(rng.poisson(rng.gamma(2.0, 1.0, 40) + 0.2, (100, 40)).astype(np.float32))
    adata.obs["batch"] = np.tile(["a", "b"], 50)
    rsc.get.anndata_to_GPU(adata)
    rsc.pp.log1p(adata)
    rsc.pp.highly_variable_genes(adata, n_top_genes=20, batch="obs.batch")
    rsc.pp.pca(adata, n_comps=5, mask="var.highly_variable")
    rsc.tl.kmeans(adata, n_clusters=2, n_pcs=5, out="obs.x")

    assert "highly_variable_nbatches" in adata.var
    assert_array_equal(adata.varm["PCs"][~adata.var["highly_variable"].to_numpy()], 0)
    assert adata.obs["x"].nunique() == 2
