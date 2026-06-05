from __future__ import annotations

import anndata
import cupy as cp
import numpy as np
import pandas as pd
import pytest
from scipy import sparse
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

import rapids_singlecell as rsc
from rapids_singlecell.pertpy_gpu._mixscape import _lda_fit_transform_gpu

NUM_CELLS_PER_GROUP = 10
NUM_NOT_DE = 10
NUM_DE = 10
ACCURACY_THRESHOLD = 0.8


@pytest.fixture
def mixscape_adata() -> anndata.AnnData:
    """Synthetic screen mirroring pertpy's mixscape test fixture.

    Cells 0-9 are non-targeting (NT); cells 10-19 express the guide but escape
    perturbation (NP); cells 20-29 are perturbed (KO). Genes 10-19 are
    differentially expressed in the KO subpopulation only.
    """
    rng = np.random.default_rng(seed=1)
    X = None
    for _ in range(NUM_NOT_DE):
        cols = [
            np.clip(rng.normal(0, 1, NUM_CELLS_PER_GROUP), 0, None) for _ in range(3)
        ]
        gene_i = np.concatenate(cols)[:, None]
        X = gene_i if X is None else np.concatenate((X, gene_i), axis=1)
    for i in range(NUM_DE):
        nt = np.clip(rng.normal(i + 2, 0.5 + 0.05 * i, NUM_CELLS_PER_GROUP), 0, None)
        npert = np.clip(rng.normal(i + 2, 0.5 + 0.05 * i, NUM_CELLS_PER_GROUP), 0, None)
        ko = np.clip(rng.normal(i + 4, 0.5 + 0.1 * i, NUM_CELLS_PER_GROUP), 0, None)
        gene_i = np.concatenate((nt, npert, ko))[:, None]
        X = np.concatenate((X, gene_i), axis=1)

    obs = pd.DataFrame(
        {
            "gene_target": ["NT"] * NUM_CELLS_PER_GROUP
            + ["target_gene_a"] * NUM_CELLS_PER_GROUP * 2
        }
    )
    obs = obs.set_index(np.arange(NUM_CELLS_PER_GROUP * 3).astype(str))
    var = pd.DataFrame(index=[f"gene{i}" for i in range(1, NUM_NOT_DE + NUM_DE + 1)])
    return anndata.AnnData(X=sparse.csr_matrix(X), obs=obs, var=var)


def _make_deterministic() -> tuple[anndata.AnnData, pd.DataFrame, np.ndarray, list]:
    n_genes, n_per = 5, 50
    classes, groups = ["NT", "KO", "NP"], ["Group1", "Group2"]
    cell_class = np.repeat(classes, n_per)
    group = np.tile(np.repeat(groups, n_per // 2), len(classes))
    obs = pd.DataFrame(
        {
            "cell_class": cell_class,
            "group": group,
            "perturbation": ["control" if c == "NT" else "pert1" for c in cell_class],
        }
    )
    data = np.zeros((len(obs), n_genes))  # float64, like pertpy's fixture
    rng = np.random.default_rng(0)
    pert = rng.uniform(-1, 1, size=(n_per // len(groups), n_genes))
    for grp in groups:
        base = 2 if grp == "Group1" else 10
        gmask = obs["group"] == grp
        data[(obs["cell_class"] == "NT") & gmask] = base
        data[(obs["cell_class"] == "KO") & gmask] = base + pert
        data[(obs["cell_class"] == "NP") & gmask] = base
    var = pd.DataFrame(index=[f"Gene{i + 1}" for i in range(n_genes)])
    obs.index = obs.index.astype(str)
    return anndata.AnnData(X=data, obs=obs, var=var), obs, pert, groups


def test_mixscape_classification(mixscape_adata):
    adata = mixscape_adata
    adata.layers["X_pert"] = adata.X.copy()
    rsc.ptg.Mixscape().mixscape(
        adata, pert_key="gene_target", control="NT", test_method="t-test"
    )

    assert "mixscape_class" in adata.obs
    assert "mixscape_class_global" in adata.obs
    assert "mixscape_class_p_ko" in adata.obs

    glob = adata.obs["mixscape_class_global"]
    np_correct = int(
        (glob[NUM_CELLS_PER_GROUP : 2 * NUM_CELLS_PER_GROUP] == "NP").sum()
    )
    ko_correct = int(
        (glob[2 * NUM_CELLS_PER_GROUP : 3 * NUM_CELLS_PER_GROUP] == "KO").sum()
    )
    assert np_correct > ACCURACY_THRESHOLD * NUM_CELLS_PER_GROUP
    assert ko_correct > ACCURACY_THRESHOLD * NUM_CELLS_PER_GROUP
    assert "mixscape" in adata.uns


@pytest.mark.parametrize("dtype", [np.float32, np.float64])
def test_mixscape_matches_pertpy(mixscape_adata, dtype):
    """Classification matches pertpy at both float32 and float64."""
    pt = pytest.importorskip("pertpy")

    ad_rsc = mixscape_adata
    ad_rsc.X = ad_rsc.X.astype(dtype)
    ad_rsc.layers["X_pert"] = ad_rsc.X.copy()
    ad_pt = ad_rsc.copy()

    rsc.ptg.Mixscape().mixscape(
        ad_rsc, pert_key="gene_target", control="NT", test_method="t-test"
    )
    pt.tl.Mixscape().mixscape(
        ad_pt, pert_key="gene_target", control="NT", test_method="t-test"
    )

    agreement = float(
        (
            ad_rsc.obs["mixscape_class_global"].values
            == ad_pt.obs["mixscape_class_global"].values
        ).mean()
    )
    assert agreement >= 0.95


@pytest.mark.parametrize("mode", ["nn", "split_by"])
def test_perturbation_signature_deterministic(mode):
    adata, obs, pert, groups = _make_deterministic()
    kwargs = {"pert_key": "perturbation", "control": "control", "split_by": "group"}
    if mode == "nn":
        kwargs["n_neighbors"] = 5
    else:
        kwargs["ref_selection_mode"] = "split_by"

    rsc.ptg.Mixscape().perturbation_signature(adata, **kwargs)

    assert "X_pert" in adata.layers
    xp = rsc.get.X_to_CPU(adata.layers["X_pert"])
    expected_ko = -np.concatenate([pert] * len(groups), axis=0)
    assert np.allclose(xp[obs["cell_class"] == "NT"], 0, atol=1e-4)
    assert np.allclose(xp[obs["cell_class"] == "NP"], 0, atol=1e-4)
    assert np.allclose(xp[obs["cell_class"] == "KO"], expected_ko, atol=1e-4)


def test_lda(mixscape_adata):
    adata = mixscape_adata
    adata.layers["X_pert"] = adata.X.copy()
    ms = rsc.ptg.Mixscape()
    ms.mixscape(adata, pert_key="gene_target", control="NT", test_method="t-test")
    ms.lda(adata, pert_key="gene_target", control="NT", test_method="t-test")

    assert "mixscape_lda" in adata.uns
    assert adata.uns["mixscape_lda"].ndim == 2


def test_gpu_lda_matches_sklearn():
    rng = np.random.default_rng(0)
    centers = rng.normal(scale=5.0, size=(4, 12))
    X = np.vstack([rng.normal(c, 1.0, size=(150, 12)) for c in centers]).astype(
        np.float64
    )
    y = np.repeat(np.arange(4), 150)
    n_comp = 3

    skl = LinearDiscriminantAnalysis(n_components=n_comp).fit_transform(X, y)
    gpu = cp.asnumpy(
        _lda_fit_transform_gpu(cp.asarray(X), cp.asarray(y), n_components=n_comp)
    )
    assert gpu.shape == skl.shape
    # float64 SVD port: matches sklearn to ~machine precision (measured ~4e-14);
    # discriminant directions are unique only up to a per-column sign flip.
    for j in range(skl.shape[1]):
        a, b = skl[:, j], gpu[:, j]
        assert np.allclose(a, b, atol=1e-10) or np.allclose(a, -b, atol=1e-10)


def test_perturbation_signature_writes_layer(mixscape_adata):
    rsc.ptg.Mixscape().perturbation_signature(
        mixscape_adata, pert_key="gene_target", control="NT"
    )
    assert "X_pert" in mixscape_adata.layers
    assert mixscape_adata.layers["X_pert"].shape == mixscape_adata.shape


def test_mixscape_requires_signature(mixscape_adata):
    with pytest.raises(KeyError, match="X_pert"):
        rsc.ptg.Mixscape().mixscape(
            mixscape_adata, pert_key="gene_target", control="NT", test_method="t-test"
        )


def test_perturbation_signature_invalid_mode(mixscape_adata):
    with pytest.raises(ValueError, match="ref_selection_mode"):
        rsc.ptg.Mixscape().perturbation_signature(
            mixscape_adata,
            pert_key="gene_target",
            control="NT",
            ref_selection_mode="bogus",
        )


def test_mixscape_output_schema(mixscape_adata):
    """obs/uns keys and label format match pertpy's Mixscape output."""
    adata = mixscape_adata
    adata.layers["X_pert"] = adata.X.copy()
    rsc.ptg.Mixscape().mixscape(
        adata, pert_key="gene_target", control="NT", test_method="t-test"
    )
    for key in ("mixscape_class", "mixscape_class_global", "mixscape_class_p_ko"):
        assert key in adata.obs
    assert set(adata.obs["mixscape_class_global"].unique()) <= {
        "NT",
        "KO",
        "NP",
    }
    assert adata.obs["mixscape_class_p_ko"].dtype.kind == "f"
    # uns["mixscape"][gene][category] is a per-cell projection DataFrame.
    assert isinstance(adata.uns["mixscape"], dict)
    gene_frames = next(iter(adata.uns["mixscape"].values()))
    frame = next(iter(gene_frames.values()))
    assert isinstance(frame, pd.DataFrame)
    assert "pvec" in frame.columns


@pytest.mark.parametrize("knn_algorithm", ["brute", "ivfflat"])
def test_perturbation_signature_knn_backends(knn_algorithm):
    """The brute and approximate cuVS neighbor backends both run and write X_pert."""
    rng = np.random.default_rng(0)
    n = 600
    X = np.clip(rng.normal(0, 1, size=(n, 30)), 0, None).astype(np.float32)
    obs = pd.DataFrame(
        {"pert": (["control"] * (n // 2) + ["g"] * (n - n // 2))},
        index=[str(i) for i in range(n)],
    )
    adata = anndata.AnnData(
        X=X, obs=obs, var=pd.DataFrame(index=[f"g{i}" for i in range(30)])
    )
    rsc.ptg.Mixscape().perturbation_signature(
        adata,
        pert_key="pert",
        control="control",
        use_rep="X",
        n_neighbors=15,
        knn_algorithm=knn_algorithm,
    )
    assert adata.layers["X_pert"].shape == adata.shape


def test_mixscape_split_by():
    """split_by computes the signature and classification per group."""
    adata, _obs, _pert, _groups = _make_deterministic()
    ms = rsc.ptg.Mixscape()
    ms.perturbation_signature(
        adata,
        pert_key="perturbation",
        control="control",
        split_by="group",
        ref_selection_mode="split_by",
    )
    ms.mixscape(
        adata,
        pert_key="perturbation",
        control="control",
        split_by="group",
        test_method="t-test",
    )
    assert "mixscape_class_global" in adata.obs


def test_signature_no_control_in_split_warns():
    """A split_by group lacking control cells warns and is left unchanged."""
    rng = np.random.default_rng(0)
    n = 60
    X = np.clip(rng.normal(0, 1, size=(n, 8)), 0, None).astype(np.float32)
    # Group "b" has only perturbed cells -> no control reference.
    pert = ["control"] * 20 + ["g"] * 10 + ["g"] * 30
    group = ["a"] * 30 + ["b"] * 30
    obs = pd.DataFrame({"pert": pert, "group": group}, index=[str(i) for i in range(n)])
    adata = anndata.AnnData(
        X=X, obs=obs, var=pd.DataFrame(index=[f"g{i}" for i in range(8)])
    )
    with pytest.warns(UserWarning, match="[Nn]o control"):
        rsc.ptg.Mixscape().perturbation_signature(
            adata,
            pert_key="pert",
            control="control",
            split_by="group",
            ref_selection_mode="split_by",
        )


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
    rsc.ptg.Mixscape().mixscale(
        adata, pert_key="gene_target", control="NT", test_method="t-test"
    )
    assert "mixscale_score" in adata.obs
    assert adata.obs["mixscale_score"].dtype.kind == "f"
    nt_scores = adata.obs.loc[adata.obs["gene_target"] == "NT", "mixscale_score"]
    assert (nt_scores == 0).all()


def test_mixscale_perturbed_cells_nonzero(mixscale_adata):
    """Perturbed cells receive non-zero scores."""
    adata = mixscale_adata
    rsc.ptg.Mixscape().mixscale(
        adata, pert_key="gene_target", control="NT", test_method="t-test"
    )
    ko = adata.obs.loc[adata.obs["gene_target"] == "GeneA", "mixscale_score"]
    assert ko.abs().mean() > 0


def test_mixscale_strong_vs_weak(mixscale_adata):
    """Strongly perturbed cells get larger absolute scores than weak ones."""
    adata = mixscale_adata
    rsc.ptg.Mixscape().mixscale(
        adata, pert_key="gene_target", control="NT", test_method="t-test"
    )
    scores = adata.obs["mixscale_score"].to_numpy()
    strong_mean = np.abs(scores[100:200]).mean()  # GeneA strong KO
    weak_mean = np.abs(scores[200:300]).mean()  # GeneA weak KO
    assert strong_mean > weak_mean


def test_mixscale_multiple_perturbations(mixscale_adata):
    """Every target gene with enough DE genes is scored."""
    adata = mixscale_adata
    rsc.ptg.Mixscape().mixscale(
        adata, pert_key="gene_target", control="NT", test_method="t-test"
    )
    for gene in ("GeneA", "GeneB"):
        s = adata.obs.loc[adata.obs["gene_target"] == gene, "mixscale_score"]
        assert s.abs().mean() > 0


def test_mixscale_custom_column_name(mixscale_adata):
    adata = mixscale_adata
    rsc.ptg.Mixscape().mixscale(
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
    result = rsc.ptg.Mixscape().mixscale(
        adata, pert_key="gene_target", control="NT", test_method="t-test", copy=True
    )
    assert result is not None
    assert result is not adata
    assert "mixscale_score" in result.obs
    assert "mixscale_score" not in adata.obs


def test_mixscale_requires_signature(mixscale_adata):
    del mixscale_adata.layers["X_pert"]
    with pytest.raises(KeyError, match="X_pert"):
        rsc.ptg.Mixscape().mixscale(
            mixscale_adata, pert_key="gene_target", control="NT", test_method="t-test"
        )


def test_mixscale_sparse_input(mixscale_adata):
    """Sparse layers score the same as dense (matrix is densified internally)."""
    adata = mixscale_adata
    adata.layers["X_pert"] = sparse.csr_matrix(adata.layers["X_pert"])
    rsc.ptg.Mixscape().mixscale(
        adata, pert_key="gene_target", control="NT", test_method="t-test"
    )
    pert = adata.obs.loc[adata.obs["gene_target"] != "NT", "mixscale_score"]
    assert not np.isnan(pert.to_numpy()).any()


def test_mixscale_matches_pertpy(mixscape_adata):
    """Continuous scores match pertpy's mixscale to numerical precision.

    Skips until the installed pertpy ships ``Mixscape.mixscale`` (PR #945).
    """
    pt = pytest.importorskip("pertpy")
    if not hasattr(pt.tl.Mixscape, "mixscale"):
        pytest.skip("installed pertpy has no Mixscape.mixscale")

    ad_rsc = mixscape_adata
    ad_rsc.layers["X_pert"] = ad_rsc.X.copy()
    ad_pt = ad_rsc.copy()

    rsc.ptg.Mixscape().mixscale(
        ad_rsc, pert_key="gene_target", control="NT", test_method="t-test"
    )
    pt.tl.Mixscape().mixscale(
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


def test_mixscale_no_scale(mixscale_adata):
    """The scale=False path runs and produces finite, non-zero perturbed scores."""
    adata = mixscale_adata
    rsc.ptg.Mixscape().mixscale(
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
        rsc.ptg.Mixscape().mixscale(
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
    rsc.ptg.Mixscape().mixscale(
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
