from __future__ import annotations

import anndata
import cupy as cp
import dask
import dask.array as da
import numpy as np
import pandas as pd
import pytest
from cupyx.scipy.sparse import csc_matrix as cp_csc_matrix
from cupyx.scipy.sparse import csr_matrix as cp_csr_matrix
from dask.callbacks import Callback
from scipy import sparse
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

import rapids_singlecell as rsc
from rapids_singlecell.pertpy_gpu import _base
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
    rsc.ptg.Mixscape().mixscape(adata, pert_key="gene_target", control="NT", test_method="t-test")

    assert "mixscape_class" in adata.obs
    assert "mixscape_class_global" in adata.obs
    assert "mixscape_class_p_ko" in adata.obs

    glob = adata.obs["mixscape_class_global"]
    np_correct = int((glob[NUM_CELLS_PER_GROUP : 2 * NUM_CELLS_PER_GROUP] == "NP").sum())
    ko_correct = int((glob[2 * NUM_CELLS_PER_GROUP : 3 * NUM_CELLS_PER_GROUP] == "KO").sum())
    assert np_correct > ACCURACY_THRESHOLD * NUM_CELLS_PER_GROUP
    assert ko_correct > ACCURACY_THRESHOLD * NUM_CELLS_PER_GROUP
    assert "mixscape" in adata.uns


def test_mixscape_min_cells_is_independent_of_min_de_genes():
    rng = np.random.default_rng(7)
    n_cells, n_genes = 6, 20
    adata = anndata.AnnData(
        X=np.vstack([rng.normal(1.0, 0.1, (n_cells, n_genes)), rng.normal(5.0, 0.1, (n_cells, n_genes))]),
        obs=pd.DataFrame(
            {"gene_target": ["NT"] * n_cells + ["target_gene"] * n_cells},
            index=[f"cell_{i}" for i in range(2 * n_cells)],
        ),
        var=pd.DataFrame(index=[f"gene_{i}" for i in range(n_genes)]),
    )
    adata.layers["X_pert"] = adata.X.copy()
    target = (adata.obs["gene_target"] == "target_gene").to_numpy()

    result = rsc.ptg.Mixscape().mixscape(
        adata, pert_key="gene_target", control="NT", test_method="t-test", min_de_genes=10, min_cells=3, copy=True
    )
    assert (result.obs.loc[target, "mixscape_class_global"] == "KO").all()

    result = rsc.ptg.Mixscape().mixscape(
        adata, pert_key="gene_target", control="NT", test_method="t-test", min_de_genes=1, min_cells=7, copy=True
    )
    assert (result.obs.loc[target, "mixscape_class_global"] == "NP").all()


@pytest.mark.parametrize("dtype", [np.float32, np.float64])
def test_mixscape_matches_pertpy(mixscape_adata, dtype):
    """Classification matches pertpy at both float32 and float64."""
    pt = pytest.importorskip("pertpy")

    ad_rsc = mixscape_adata
    ad_rsc.X = ad_rsc.X.astype(dtype)
    ad_rsc.layers["X_pert"] = ad_rsc.X.copy()
    ad_pt = ad_rsc.copy()

    rsc.ptg.Mixscape().mixscape(ad_rsc, pert_key="gene_target", control="NT", test_method="t-test")
    pt.tl.Mixscape().mixscape(ad_pt, pert_key="gene_target", control="NT", test_method="t-test")

    agreement = float((ad_rsc.obs["mixscape_class_global"].values == ad_pt.obs["mixscape_class_global"].values).mean())
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


def test_sparse_nn_signature_averages_controls_before_log():
    counts = np.array([[0, 8, 0], [2, 0, 0], [0, 0, 3]])
    logged = np.log1p(counts)
    adata = anndata.AnnData(sparse.csr_matrix(logged))
    adata.obs["target"] = ["NT", "NT", "A"]
    rsc.ptg.Mixscape().perturbation_signature(adata, "target", "NT", use_rep="X", n_neighbors=2)
    expected = np.log1p(counts[:2].mean(axis=0)) - logged
    np.testing.assert_allclose(adata.layers["X_pert"].get(), expected, atol=1e-12)


@pytest.fixture
def perturbation_screen():
    rng = np.random.default_rng(42)
    targets = np.tile(["NT", "A", "B"], 64)
    splits = np.repeat(["s1", "s2"], len(targets) // 2)
    X = np.clip(rng.normal(0.0, 0.6, (len(targets), 64)), 0, None)
    X[splits == "s2", :16] *= 2
    responders = rng.random(len(targets)) < 0.7
    X[(targets == "A") & responders, :8] += 3
    X[(targets == "B") & responders, 8:16] += 3
    obs = pd.DataFrame({"target": targets, "split": splits}, index=np.arange(len(targets)).astype(str))
    return anndata.AnnData(X, obs=obs)


@pytest.mark.parametrize("method", ["mixscape", "mixscale"])
@pytest.mark.parametrize(
    ("signature_kwargs", "sparse_format"),
    [
        ({"use_rep": "X", "n_neighbors": 5, "n_dims": None}, sparse.csr_matrix),
        ({"ref_selection_mode": "split_by"}, sparse.csc_matrix),
    ],
    ids=["nn-csr", "split-by-csc"],
)
def test_score_from_sparse_counts(method, signature_kwargs, sparse_format, perturbation_screen, monkeypatch):
    reference = perturbation_screen
    input_matrix = sparse_format(reference.X)
    adata = anndata.AnnData(input_matrix, obs=reference.obs.copy())
    analyzer = rsc.ptg.Mixscape() if method == "mixscape" else rsc.ptg.Mixscale()
    score = getattr(analyzer, method)
    kwargs = {"pert_key": "target", "control": "NT", "split_by": "split"}
    analyzer.perturbation_signature(reference, **kwargs, **signature_kwargs)
    score(reference, **kwargs)

    def guard_toarray(toarray):
        def check_shape(matrix, *args, **kwargs):
            assert matrix.shape != adata.shape
            return toarray(matrix, *args, **kwargs)

        return check_shape

    for cls in (cp_csr_matrix, cp_csc_matrix):
        monkeypatch.setattr(cls, "toarray", guard_toarray(cls.toarray))
    score(adata, **kwargs, signature_kwargs=signature_kwargs)
    assert adata.X is input_matrix
    assert "X_pert" not in adata.layers
    _assert_scores_match(adata, reference, method)


@pytest.mark.parametrize("method", ["mixscape", "mixscale"])
@pytest.mark.parametrize(
    ("mode", "array_type", "chunk_cols"),
    [
        ("raw", sparse.csr_matrix, -1),
        ("pca", sparse.csc_matrix, 7),
        ("split_by", np.asarray, -1),
        ("nn", sparse.csc_matrix, 7),
        ("stored", sparse.csr_matrix, -1),
    ],
    ids=["raw-nn-scipy-csr", "masked-pca-tiled-coo", "split-by-numpy", "nn-tiled-scipy-csc", "stored-cupy-csr"],
)
def test_score_from_dask_chunks(perturbation_screen, method, mode, array_type, chunk_cols):
    def chunk_array(values):
        chunks = da.from_array(array_type(values), chunks=(31, chunk_cols))
        if mode == "pca":
            # Dask joins can yield COO chunks even with CSR metadata.
            return chunks.map_blocks(sparse.coo_matrix, meta=sparse.csr_matrix((0, 0), dtype=values.dtype))
        return rsc.get.X_to_GPU(chunks) if mode == "stored" else chunks

    reference = perturbation_screen
    expression = chunk_array(reference.X)
    adata = anndata.AnnData(expression, obs=reference.obs.copy())
    signature = {"ref_selection_mode": "split_by"}
    if mode == "nn":
        embedding = np.random.default_rng(1).normal(size=(adata.n_obs, 4)).astype(np.float32)
        reference.obsm["embedding"] = adata.obsm["embedding"] = embedding
        signature = {"use_rep": "embedding", "n_neighbors": 5}
    elif mode == "raw":
        signature = {"use_rep": "X", "n_neighbors": 5, "n_dims": adata.n_vars - 1}
    elif mode == "pca":
        reference.var["highly_variable"] = adata.var["highly_variable"] = np.arange(adata.n_vars) % 3 != 0
        signature = {"n_pcs": 4, "n_neighbors": 5}
    analyzer = rsc.ptg.Mixscape() if method == "mixscape" else rsc.ptg.Mixscale()
    score = getattr(analyzer, method)
    kwargs = {"pert_key": "target", "control": "NT", "split_by": "split"}
    analyzer.perturbation_signature(reference, **kwargs, **signature)
    score(reference, **kwargs, test_method="t-test")
    options = {"signature_kwargs": signature}
    if mode == "stored":
        stored = chunk_array(reference.layers["X_pert"].get())
        adata.layers["custom"] = stored
        options = {"layer": "custom"}

    def check_chunk(key, result, dask_graph, state, worker):
        assert getattr(result, "shape", None) not in (adata.shape, (adata.n_obs, adata.n_vars - 1))

    with dask.config.set(scheduler="synchronous"), Callback(posttask=check_chunk):
        score(adata, **kwargs, test_method="t-test", **options)
    assert adata.X is expression
    assert "X_pert" not in adata.layers
    _assert_scores_match(adata, reference, method)


def test_dask_marker_batches_share_source_reads(monkeypatch):
    values = np.arange(48, dtype=np.float32).reshape(8, 6)
    reads = np.zeros(4, dtype=int)

    @dask.delayed
    def read_chunk(i):
        reads[i] += 1
        return sparse.csr_matrix(values[2 * i : 2 * i + 2])

    meta = sparse.csr_matrix((0, 0), dtype=values.dtype)
    expression = da.concatenate([da.from_delayed(read_chunk(i), (2, 6), values.dtype, meta=meta) for i in range(4)])
    jobs = [
        {"row_ids": np.array(rows), "col_ids": np.array(cols)}
        for rows, cols in [([0, 2, 4], [5, 0]), ([1, 3, 5], [2, 4]), ([0, 3, 4], [3, 1])]
    ]
    monkeypatch.setattr(_base, "_MARKER_BLOCK_BYTES", 2 * 3 * 2 * values.itemsize)
    with dask.config.set(scheduler="synchronous"):
        for job, count in zip(_base._with_marker_blocks(expression, jobs), [1, 1, 2], strict=True):
            np.testing.assert_array_equal(job["dat"].get(), values[np.ix_(job["row_ids"], job["col_ids"])])
            np.testing.assert_array_equal(reads, [count, count, count, 0])


@pytest.mark.parametrize(
    "n_cells,budget,gpu,targets,expected_reads",
    [
        (32, 1600, False, "ABC", [2, 2, 2, 1]),  # Share all three panels.
        (32, 128, False, "ABC", [4, 4, 2, 1]),  # Every sparse stage exceeds the budget.
        (17, 1600, True, "ABC", [3, 3, 2]),  # The third panel would stage the full matrix.
        (32, 1600, False, "A", [1, 1, 0, 0]),  # No other target can share this read.
    ],
)
def test_wide_sparse_dask_marker_batches(n_cells, budget, gpu, targets, expected_reads, *, monkeypatch):
    labels = np.array("NT A B NT NT C NT A NT B NT C NT A B NT C".split() + ["other"] * (n_cells - 17))
    values = np.eye(n_cells, 64, dtype=np.float32)
    values[np.arange(n_cells), np.arange(n_cells) + 32] = np.arange(n_cells) + 2
    reads = np.zeros((len(expected_reads), 2), dtype=int)
    csr = cp_csr_matrix if gpu else sparse.csr_matrix
    meta = csr((0, 0), dtype=values.dtype)

    @dask.delayed
    def read_tile(i, j):
        reads[i, j] += 1
        tile = values[8 * i : 8 * i + 8, 32 * j : 32 * j + 32]
        tile = csr(cp.asarray(tile) if gpu else tile)
        return tile.asformat(("csr", "csc", "coo")[i % 3])

    expression = da.block(
        [
            [da.from_delayed(read_tile(i, j), (min(8, n_cells - 8 * i), 32), values.dtype, meta=meta) for j in range(2)]
            for i in range(len(expected_reads))
        ]
    ).map_blocks(lambda tile: tile, meta=meta)  # da.block drops sparse CuPy metadata.
    adata = anndata.AnnData(
        expression, obs=pd.DataFrame({"target": labels, "batch": "all"}, index=np.arange(n_cells).astype(str))
    )
    signature = _base._PerturbationSignature(adata, "target", "NT", split_by="batch", ref_selection_mode="split_by")
    jobs = []
    for i, target in enumerate(targets):
        rows = np.flatnonzero((labels == target) | (labels == "NT"))
        cols = np.roll(np.arange(54, dtype=np.int32)[::-1], i)
        jobs.append({"split": 0, "row_ids": rows, "col_ids": cols, "is_guide": labels[rows] == target})

    monkeypatch.setattr(_base, "_MARKER_BLOCK_BYTES", budget)
    with dask.config.set(scheduler="synchronous"):
        for job in _base._with_marker_blocks(signature, jobs):
            rows, cols = job["row_ids"], job["col_ids"]
            expected = values[labels == "NT"][:, cols].mean(axis=0) - values[np.ix_(rows, cols)]
            np.testing.assert_array_equal(job.pop("dat").get(), expected)
    np.testing.assert_array_equal(reads, np.broadcast_to(np.asarray(expected_reads)[:, None], reads.shape))


def _assert_scores_match(adata, reference, method):
    if method == "mixscape":
        for column in ("mixscape_class", "mixscape_class_global"):
            np.testing.assert_array_equal(adata.obs[column], reference.obs[column])
        assert {"KO", "NP"} <= set(adata.obs["mixscape_class_global"])
        column = "mixscape_class_p_ko"
    else:
        column = "mixscale_score"
    for target in ("A", "B"):
        assert adata.obs.loc[adata.obs["target"] == target, column].abs().max() > 0.5
    np.testing.assert_allclose(adata.obs[column], reference.obs[column], atol=1e-6, rtol=1e-6)


@pytest.mark.parametrize("analyzer, method", [(rsc.ptg.Mixscape, "mixscape"), (rsc.ptg.Mixscale, "mixscale")])
def test_signature_options_conflict_with_layer(mixscape_adata, analyzer, method):
    mixscape_adata.layers["custom"] = mixscape_adata.X.copy()
    with pytest.raises(ValueError, match="layer"):
        getattr(analyzer(), method)(mixscape_adata, "gene_target", "NT", layer="custom", signature_kwargs={})


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
    X = np.vstack([rng.normal(c, 1.0, size=(150, 12)) for c in centers]).astype(np.float64)
    y = np.repeat(np.arange(4), 150)
    n_comp = 3

    skl = LinearDiscriminantAnalysis(n_components=n_comp).fit_transform(X, y)
    gpu = cp.asnumpy(_lda_fit_transform_gpu(cp.asarray(X), cp.asarray(y), n_components=n_comp))
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
    obs = pd.DataFrame({"gene_target": labels}, index=[str(i) for i in range(X.shape[0])])
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
    rsc.ptg.Mixscape().perturbation_signature(mixscape_adata, pert_key="gene_target", control="NT")
    assert "X_pert" in mixscape_adata.layers
    assert mixscape_adata.layers["X_pert"].shape == mixscape_adata.shape


@pytest.mark.parametrize("analyzer", [rsc.ptg.Mixscape, rsc.ptg.Mixscale])
@pytest.mark.parametrize("preset", list(rsc.Preset))
@pytest.mark.parametrize("pca_keys", [("X_pca",), ("pca",), ("X_pca", "pca")])
def test_perturbation_signature_recomputes_short_pca(mixscape_adata, analyzer, preset, pca_keys):
    adata = mixscape_adata
    selected_key = pca_keys[0]
    loadings = "PCs" if selected_key == "X_pca" else selected_key

    def pca(adata, n_comps, key):
        adata.obsm[key], adata.varm[loadings if key == selected_key else key] = rsc.pp.pca(
            adata, n_comps=n_comps, out=True
        )

    with rsc.settings.override(preset=preset, N_PCS=3):
        pca(adata, 2, selected_key)
        if len(pca_keys) > 1:
            pca(adata, 6, "pca")
        expected = adata.copy()
        pca(expected, 4, selected_key)

        for target in (adata, expected):
            analyzer().perturbation_signature(target, pert_key="gene_target", control="NT", n_pcs=4)

    assert adata.obsm[selected_key].shape[1] == 4
    assert set(adata.obsm) == set(pca_keys)
    if len(pca_keys) > 1:
        np.testing.assert_array_equal(adata.obsm["pca"], expected.obsm["pca"])
    cp.testing.assert_allclose(adata.layers["X_pert"], expected.layers["X_pert"])


@pytest.mark.filterwarnings("error::FutureWarning")
@pytest.mark.parametrize("preset", list(rsc.Preset))
@pytest.mark.parametrize("stored", list(rsc.Preset))
def test_perturbation_signature_recompute_updates_uns(mixscape_adata, preset, stored):
    adata = mixscape_adata
    with rsc.settings.override(preset=stored):
        rsc.pp.pca(adata, n_comps=2)
    with rsc.settings.override(preset=preset, N_PCS=3):
        rsc.ptg.Mixscape().perturbation_signature(adata, pert_key="gene_target", control="NT", n_pcs=4)
    varm = "PCs" if stored is rsc.Preset.ScanpyV1 else "pca"
    assert adata.varm[varm].shape[1] == len(adata.uns["pca"]["variance"]) == 4


@pytest.mark.parametrize("use_rep", ["X_pca", "pca", "custom"])
def test_perturbation_signature_explicit_rep_is_not_recomputed(mixscape_adata, use_rep):
    adata = mixscape_adata
    adata.obsm[use_rep] = np.zeros((adata.n_obs, 2), dtype=np.float32)
    with (
        rsc.settings.override(N_PCS=3),
        pytest.raises(ValueError, match="does not have enough Dimensions"),
    ):
        rsc.ptg.Mixscape().perturbation_signature(adata, pert_key="gene_target", control="NT", use_rep=use_rep, n_pcs=4)
    assert adata.obsm[use_rep].shape[1] == 2
    assert "pca" not in adata.uns


def test_mixscape_requires_signature(mixscape_adata):
    with pytest.raises(KeyError, match="X_pert"):
        rsc.ptg.Mixscape().mixscape(mixscape_adata, pert_key="gene_target", control="NT", test_method="t-test")


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
    rsc.ptg.Mixscape().mixscape(adata, pert_key="gene_target", control="NT", test_method="t-test")
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
    adata = anndata.AnnData(X=X, obs=obs, var=pd.DataFrame(index=[f"g{i}" for i in range(30)]))
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


def test_perturbation_signature_incomplete_neighbors_raise():
    # probing one of 64 lists cannot fill 100 neighbor slots
    adata = anndata.AnnData(np.clip(np.random.default_rng(0).normal(size=(600, 30)), 0, None).astype(np.float32))
    adata.obs["pert"] = ["control"] * 500 + ["g"] * 100
    kwargs = {"n_neighbors": 100, "knn_algorithm": "ivfflat", "knn_kwargs": {"n_lists": 64, "n_probes": 1}}
    with pytest.raises(ValueError, match="n_probes"):
        rsc.ptg.Mixscape().perturbation_signature(adata, pert_key="pert", control="control", use_rep="X", **kwargs)
    assert "X_pert" not in adata.layers


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
    adata = anndata.AnnData(X=X, obs=obs, var=pd.DataFrame(index=[f"g{i}" for i in range(n_genes)]))

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
    adata = anndata.AnnData(X=X, obs=obs, var=pd.DataFrame(index=[f"g{i}" for i in range(8)]))
    with pytest.warns(UserWarning, match="[Nn]o control"):
        rsc.ptg.Mixscape().perturbation_signature(
            adata,
            pert_key="pert",
            control="control",
            split_by="group",
            ref_selection_mode="split_by",
        )
