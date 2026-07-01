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
ACCURACY_THRESHOLD = 0.8


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


def test_lda_multiclass():
    """End-to-end ``lda()`` with more than two perturbation classes: the
    embedding must have ``n_classes - 1`` (>=2) discriminant components, which
    the single-target ``test_lda`` fixture (n_components == 1) cannot exercise.
    """
    rng = np.random.default_rng(0)
    n_ctrl, n_ko, n_genes = 80, 60, 40
    de_blocks = {  # each target knocks out a distinct, non-overlapping gene block
        "gene_a": slice(0, 10),
        "gene_b": slice(10, 20),
        "gene_c": slice(20, 30),
    }
    blocks = [rng.normal(1.0, 0.3, (n_ctrl, n_genes))]
    labels = ["NT"] * n_ctrl
    for gene, sl in de_blocks.items():
        ko = rng.normal(1.0, 0.3, (n_ko, n_genes))
        ko[:, sl] += 5.0  # strong, gene-specific knockout signal
        blocks.append(ko)
        labels += [gene] * n_ko
    X = np.vstack(blocks).astype(np.float32)
    obs = pd.DataFrame(
        {"gene_target": labels}, index=[str(i) for i in range(X.shape[0])]
    )
    var = pd.DataFrame(index=[f"gene{i}" for i in range(n_genes)])
    adata = anndata.AnnData(X=sparse.csr_matrix(X), obs=obs, var=var)
    adata.layers["X_pert"] = adata.X.copy()

    ms = rsc.ptg.Mixscape()
    ms.mixscape(adata, pert_key="gene_target", control="NT", test_method="t-test")
    ms.lda(
        adata,
        pert_key="gene_target",
        control="NT",
        test_method="t-test",
        min_de_genes=3,
    )

    emb = adata.uns["mixscape_lda"]
    assert emb.ndim == 2
    assert emb.shape[1] >= 2  # >2 classes -> multiple discriminant components


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
    classes = set(adata.obs["mixscape_class_global"].unique())
    assert classes <= {"NT", "KO", "NP"}
    # classification must actually separate cells, not collapse to all-NT
    assert {"KO", "NP"} & classes
    assert adata.obs["mixscape_class_p_ko"].dtype.kind == "f"
    # uns["mixscape"][gene][category] is a per-cell projection DataFrame.
    assert isinstance(adata.uns["mixscape"], dict)
    gene_frames = next(iter(adata.uns["mixscape"].values()))
    frame = next(iter(gene_frames.values()))
    assert isinstance(frame, pd.DataFrame)
    assert "pvec" in frame.columns


@pytest.mark.parametrize("knn_algorithm", ["brute", "ivfflat", "cagra", "ivfpq"])
def test_perturbation_signature_knn_backends(knn_algorithm):
    """Every advertised neighbor backend runs and writes a finite X_pert."""
    rng = np.random.default_rng(0)
    n = 2000
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
    xp = rsc.get.X_to_CPU(adata.layers["X_pert"])
    assert xp.shape == adata.shape
    assert np.all(np.isfinite(xp))


def test_mixscape_split_by():
    """split_by conditions per group: with strong, group-specific baselines, the
    planted KO cells are predominantly called KO within each group."""
    rng = np.random.default_rng(0)
    n_per, n_genes = 50, 60
    blocks, pert, group, is_ko = [], [], [], []
    for grp, base in (("A", 0.0), ("B", 8.0)):  # very different baselines per group
        nt = rng.normal(base, 0.5, (n_per, n_genes))
        ko = rng.normal(base, 0.5, (n_per, n_genes))
        ko[:, :20] += 5.0  # strong knockout signal on 20 genes
        npe = rng.normal(base, 0.5, (n_per, n_genes))  # escaped: no signal
        blocks += [nt, ko, npe]
        pert += ["control"] * n_per + ["geneX"] * (2 * n_per)
        group += [grp] * (3 * n_per)
        is_ko += [False] * n_per + [True] * n_per + [False] * n_per
    X = np.vstack(blocks).astype(np.float32)
    obs = pd.DataFrame(
        {"perturbation": pert, "group": group},
        index=[str(i) for i in range(X.shape[0])],
    )
    adata = anndata.AnnData(
        X=X, obs=obs, var=pd.DataFrame(index=[f"g{i}" for i in range(n_genes)])
    )

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
    glob = adata.obs["mixscape_class_global"].to_numpy()
    is_ko = np.array(is_ko)
    group = np.array(group)
    for grp in ("A", "B"):
        ko_cells = is_ko & (group == grp)
        assert (glob[ko_cells] == "KO").mean() > 0.7


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
