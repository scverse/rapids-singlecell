from __future__ import annotations

import anndata
import cupy as cp
import numpy as np
import pandas as pd
import pytest
from scipy import sparse

import rapids_singlecell as rsc


@pytest.fixture
def mixscale_adata() -> anndata.AnnData:
    """Synthetic screen with strong and weak knockouts for mixscale scoring.

    Mirrors pertpy's mixscale fixture: 100 non-targeting controls, then for
    GeneA 100 strong-KO and 100 weak-KO cells (effect on the first 20 genes),
    and 50 GeneB cells (moderate effect on genes 20-39). The planted gradient
    lets a continuous score separate strong from weak knockdowns.
    """
    rng = np.random.default_rng(42)
    n_genes = 200
    n_cells = 350
    X = rng.standard_normal((n_cells, n_genes)).astype(np.float32)
    X[100:200, :20] -= 3.0  # GeneA strong KO
    X[200:300, :20] -= 1.0  # GeneA weak KO
    X[300:350, 20:40] -= 2.0  # GeneB moderate
    labels = ["NT"] * 100 + ["GeneA"] * 200 + ["GeneB"] * 50
    obs = pd.DataFrame(
        {"gene_target": labels}, index=[f"Cell_{i}" for i in range(n_cells)]
    )
    var = pd.DataFrame(index=[f"Gene_{i}" for i in range(n_genes)])
    adata = anndata.AnnData(X=X, obs=obs, var=var)
    adata.layers["X_pert"] = adata.X.copy()
    return adata


def test_mixscale_basic_scoring(mixscale_adata):
    """mixscale runs, writes a float score column, and controls score 0."""
    adata = mixscale_adata
    rsc.ptg.Mixscale().mixscale(
        adata, pert_key="gene_target", control="NT", test_method="t-test"
    )
    assert "mixscale_score" in adata.obs
    assert adata.obs["mixscale_score"].dtype.kind == "f"
    nt_scores = adata.obs.loc[adata.obs["gene_target"] == "NT", "mixscale_score"]
    assert (nt_scores == 0).all()


def test_mixscale_perturbed_cells_nonzero(mixscale_adata):
    """Perturbed cells receive non-zero scores."""
    adata = mixscale_adata
    rsc.ptg.Mixscale().mixscale(
        adata, pert_key="gene_target", control="NT", test_method="t-test"
    )
    ko = adata.obs.loc[adata.obs["gene_target"] == "GeneA", "mixscale_score"]
    assert ko.abs().mean() > 0


def test_mixscale_strong_vs_weak(mixscale_adata):
    """Strongly perturbed cells get larger absolute scores than weak ones."""
    adata = mixscale_adata
    rsc.ptg.Mixscale().mixscale(
        adata, pert_key="gene_target", control="NT", test_method="t-test"
    )
    scores = adata.obs["mixscale_score"].to_numpy()
    # Strong and weak share the 'GeneA' label (one model is fit over both), so
    # require a clear margin rather than a bare inequality.
    strong_mean = np.abs(scores[100:200]).mean()  # GeneA strong KO
    weak_mean = np.abs(scores[200:300]).mean()  # GeneA weak KO
    assert strong_mean > 1.5 * weak_mean


def test_mixscale_multiple_perturbations(mixscale_adata):
    """Every target gene with enough DE genes is scored."""
    adata = mixscale_adata
    rsc.ptg.Mixscale().mixscale(
        adata, pert_key="gene_target", control="NT", test_method="t-test"
    )
    for gene in ("GeneA", "GeneB"):
        s = adata.obs.loc[adata.obs["gene_target"] == gene, "mixscale_score"]
        assert s.abs().mean() > 0


def test_mixscale_custom_column_name(mixscale_adata):
    adata = mixscale_adata
    rsc.ptg.Mixscale().mixscale(
        adata,
        pert_key="gene_target",
        control="NT",
        test_method="t-test",
        new_class_name="my_score",
    )
    assert "my_score" in adata.obs
    assert "mixscale_score" not in adata.obs


def test_mixscale_copy_mode(mixscale_adata):
    adata = mixscale_adata
    result = rsc.ptg.Mixscale().mixscale(
        adata, pert_key="gene_target", control="NT", test_method="t-test", copy=True
    )
    assert result is not None
    assert result is not adata
    assert "mixscale_score" in result.obs
    assert "mixscale_score" not in adata.obs


def test_mixscale_requires_signature(mixscale_adata):
    del mixscale_adata.layers["X_pert"]
    with pytest.raises(KeyError, match="X_pert"):
        rsc.ptg.Mixscale().mixscale(
            mixscale_adata, pert_key="gene_target", control="NT", test_method="t-test"
        )


def test_mixscale_sparse_input(mixscale_adata):
    """Sparse layers score the same as dense (matrix is densified internally)."""
    adata = mixscale_adata
    adata.layers["X_pert"] = sparse.csr_matrix(adata.layers["X_pert"])
    rsc.ptg.Mixscale().mixscale(
        adata, pert_key="gene_target", control="NT", test_method="t-test"
    )
    pert = adata.obs.loc[adata.obs["gene_target"] != "NT", "mixscale_score"]
    assert not np.isnan(pert.to_numpy()).any()


def test_mixscale_matches_pertpy(mixscape_adata):
    """Continuous scores match pertpy's mixscale to numerical precision.

    Skips until the installed pertpy ships ``Mixscale.mixscale`` (PR #945).
    """
    pt = pytest.importorskip("pertpy")
    if not hasattr(pt.tl, "Mixscale"):
        pytest.skip("installed pertpy has no Mixscale")

    ad_rsc = mixscape_adata
    ad_rsc.layers["X_pert"] = ad_rsc.X.copy()
    ad_pt = ad_rsc.copy()

    rsc.ptg.Mixscale().mixscale(
        ad_rsc, pert_key="gene_target", control="NT", test_method="t-test"
    )
    pt.tl.Mixscale().mixscale(
        ad_pt,
        pert_key="gene_target",
        control="NT",
        layer="X_pert",
        test_method="t-test",
    )
    np.testing.assert_allclose(
        ad_rsc.obs["mixscale_score"].to_numpy(dtype=float),
        ad_pt.obs["mixscale_score"].to_numpy(dtype=float),
        rtol=1e-5,
        atol=1e-5,
    )


def _ref_mixscale_gene(block, is_guide, is_nt, *, do_scale):
    """NumPy reference for one gene's Mixscale score (pertpy's formula): z-score
    each column, project onto the (guide-mean − control-mean) direction, then
    standardize each guide cell's projection against the control distribution."""
    x = block.astype(np.float64)
    n, k = x.shape
    if do_scale:
        cmean = x.mean(0)
        sd = x.std(0, ddof=1)
        sd = np.where(sd == 0.0, 1.0, sd)
    else:
        cmean = np.zeros(k)
        sd = np.ones(k)
    vec = (x[is_guide].mean(0) - x[is_nt].mean(0)) / sd
    dotvv = float(vec @ vec)
    pvec = ((x - cmean) / sd) @ vec / max(dotvv, 1e-12)
    nt = pvec[is_nt]
    nt_mean = nt.mean()
    nt_std = nt.std(ddof=0) or 1.0
    out = np.zeros(n)
    out[is_guide] = (pvec[is_guide] - nt_mean) / nt_std
    return out


@pytest.mark.parametrize("dtype", [np.float32, np.float64])
def test_mixscale_matches_numpy_reference(dtype):
    """The batched scoring kernel matches an independent NumPy implementation of
    pertpy's Mixscale formula, per gene, at float32 and float64. This is the
    executed numerical-parity check (test_mixscale_matches_pertpy only runs once
    pertpy ships Mixscale.mixscale)."""
    from rapids_singlecell.pertpy_gpu._mixscale import _project_scores_batched

    rng = np.random.default_rng(0)
    n_obs, n_vars = 600, 40
    X = rng.normal(0.0, 1.0, (n_obs, n_vars)).astype(dtype)
    ctrl = np.arange(0, 200)  # control cells, shared across genes
    genes = {
        0: (np.arange(200, 350), np.array([1, 4, 7, 9, 12], dtype=np.int32)),
        1: (np.arange(350, 600), np.array([2, 5, 8, 20, 31, 33], dtype=np.int32)),
    }
    for gi, (guide, cols) in genes.items():
        X[np.ix_(guide, cols)] += 2.0 + gi  # plant a clear perturbation direction

    gene_jobs: list[dict] = []
    ref = np.zeros(n_obs)
    for guide, cols in genes.values():
        rows = np.concatenate([ctrl, guide]).astype(np.int32)
        is_guide = np.concatenate(
            [np.zeros(ctrl.size, bool), np.ones(guide.size, bool)]
        )
        nt_in_all = ~is_guide
        gene_jobs.append(
            {
                "row_ids": rows,
                "col_ids": cols,
                "is_guide": is_guide,
                "nt_in_all": nt_in_all,
            }
        )
        block_scores = _ref_mixscale_gene(
            X[rows][:, cols], is_guide, nt_in_all, do_scale=True
        )
        ref[guide] = block_scores[is_guide]

    Xg = cp.asarray(X)
    scores = cp.zeros(n_obs, dtype=Xg.dtype)
    _project_scores_batched(Xg, gene_jobs, scores, do_scale=True)
    atol = 2e-3 if dtype == np.float32 else 1e-9
    np.testing.assert_allclose(cp.asnumpy(scores), ref, atol=atol)


def test_mixscale_no_scale(mixscale_adata):
    """The scale=False path runs and produces finite, non-zero perturbed scores."""
    adata = mixscale_adata
    rsc.ptg.Mixscale().mixscale(
        adata, pert_key="gene_target", control="NT", test_method="t-test", scale=False
    )
    assert not np.isnan(adata.obs["mixscale_score"].to_numpy()).any()
    ko = adata.obs.loc[adata.obs["gene_target"] == "GeneA", "mixscale_score"]
    assert ko.abs().mean() > 0


def test_mixscale_float32_matches_float64():
    """float32 scores match the float64 result (in-kernel 64-bit accumulation)."""
    rng = np.random.default_rng(0)
    n_genes, n = 200, 120
    X = rng.standard_normal((2 * n, n_genes)).astype(np.float64)
    X[n:, :40] -= 2.0
    obs = pd.DataFrame(
        {"pert": ["NT"] * n + ["g"] * n}, index=[str(i) for i in range(2 * n)]
    )
    var = pd.DataFrame(index=[f"g{i}" for i in range(n_genes)])
    ad64 = anndata.AnnData(X=X, obs=obs, var=var)
    ad64.layers["X_pert"] = ad64.X.copy()
    ad32 = anndata.AnnData(X=X.astype(np.float32), obs=obs.copy(), var=var.copy())
    ad32.layers["X_pert"] = ad32.X.copy()
    for ad_ in (ad64, ad32):
        rsc.ptg.Mixscale().mixscale(
            ad_, pert_key="pert", control="NT", test_method="t-test"
        )
    np.testing.assert_allclose(
        ad32.obs["mixscale_score"].to_numpy(dtype=float),
        ad64.obs["mixscale_score"].to_numpy(dtype=float),
        rtol=1e-4,
        atol=1e-5,
    )


def test_mixscale_large_de_set_shared_mem_optin():
    """Many DE genes per perturbation exercise the >48KB dynamic shared-mem
    opt-in path (3 * k * 8 bytes exceeds the 48KB default for k > ~1700)."""
    rng = np.random.default_rng(0)
    n_genes, n = 2500, 120
    X = rng.standard_normal((2 * n, n_genes)).astype(np.float32)
    X[n:] -= 2.0  # every gene differentially expressed in the guide group
    obs = pd.DataFrame(
        {"pert": ["NT"] * n + ["g"] * n}, index=[str(i) for i in range(2 * n)]
    )
    var = pd.DataFrame(index=[f"g{i}" for i in range(n_genes)])
    adata = anndata.AnnData(X=X, obs=obs, var=var)
    adata.layers["X_pert"] = adata.X.copy()
    rsc.ptg.Mixscale().mixscale(
        adata,
        pert_key="pert",
        control="NT",
        test_method="t-test",
        max_de_genes=n_genes,
    )
    g = adata.obs.loc[adata.obs["pert"] == "g", "mixscale_score"].to_numpy()
    assert g.shape[0] == n
    assert not np.isnan(g).any()
    assert np.abs(g).mean() > 0
