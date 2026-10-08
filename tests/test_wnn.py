from __future__ import annotations

from pathlib import Path

import cupy as cp
import numpy as np
import pytest
import scanpy as sc
from anndata import AnnData, read_h5ad

import rapids_singlecell as rsc
from rapids_singlecell._cuda import _wnn_cuda
from rapids_singlecell.preprocessing._neighbors import _wnn


def _ref_jaccard(nn, prune=0.0):  # Seurat's ComputeSNN
    A = np.zeros((len(nn), len(nn)))
    A[np.arange(len(nn))[:, None], nn] = 1
    shared = A @ A.T
    J = shared / (2 * nn.shape[1] - shared)
    return np.where(J < prune, 0, J)


def _ref_bandwidth(D, nn, nearest):  # ComputeSNNwidth + SNN_SmallestNonzero_Dist
    sigma = np.empty(len(nn))
    for i, row in enumerate(_ref_jaccard(nn)):
        partners = np.flatnonzero(row)
        n_i = min(nn.shape[1], partners.size)
        thresh = np.sort(row[partners])[n_i - 1]
        d = np.maximum(D[i, partners[row[partners] <= thresh]] - nearest[i], 0)
        sigma[i] = np.sort(d)[::-1][:n_i].mean()
    return sigma


def _seurat_wnn_reference(reps, *, k, knn_range, cross=1e-4, smooth=False):
    """Straight NumPy port of Seurat's FindMultiModalNeighbors (exact kNN)."""
    E = [x / np.linalg.norm(x, axis=1, keepdims=True) for x in reps]
    n, n_mod = E[0].shape[0], len(E)
    rows = np.arange(n)
    D = [np.linalg.norm(e[:, None, :] - e[None, :, :], axis=2) for e in E]
    order = [np.argsort(d, axis=1, kind="stable") for d in D]
    nn = [o[:, :k] for o in order]
    nearest = [D[m][rows, nn[m][:, 1]] for m in range(n_mod)]
    sigma = [_ref_bandwidth(D[m], nn[m], nearest[m]) for m in range(n_mod)]

    def impute(m, r):  # PredictAssay + impute_dist
        pred = E[m][nn[r][:, 1:]].mean(axis=1)
        return np.maximum(np.linalg.norm(E[m] - pred, axis=1) - nearest[m], 0)

    exp_scores = np.zeros((n_mod, n))
    for m in range(n_mod):
        within = np.exp(-impute(m, m) / sigma[m])
        for r in range(n_mod):
            if r == m:
                continue
            score = np.clip(within / (np.exp(-impute(m, r) / sigma[m]) + cross), 0, 200)
            if smooth:
                score = score[nn[m][:, 1:]].mean(axis=1)
            exp_scores[m] += np.exp(score)
    weights = exp_scores / exp_scores.sum(axis=0)

    idx = np.empty((n, k), dtype=int)
    sim = np.empty((n, k))
    for i in range(n):  # MultiModalNN
        cand = np.concatenate([o[i, 1:knn_range] for o in order])
        cand = np.array(list(dict.fromkeys(cand)))
        s = sum(
            weights[m, i]
            * np.exp(-np.maximum(D[m][i, cand] - nearest[m][i], 0) / sigma[m][i])
            for m in range(n_mod)
        )
        best = np.argsort(-s, kind="stable")[:k]
        idx[i], sim[i] = cand[best], s[best]
    snn = _ref_jaccard(idx, prune=1 / 15)
    return weights, idx, np.sqrt(np.clip((1 - sim) / 2, 0, 1)), snn


def _assert_matches(adata, names, ref):
    weights, idx, dist, snn = ref
    w = adata.obs[[f"{name}_weight" for name in names]].to_numpy()
    np.testing.assert_allclose(w, np.transpose(weights), rtol=1e-4, atol=1e-5)
    D = adata.obsp["distances"]
    # rows keep the neighbors in decreasing similarity, as Seurat does
    np.testing.assert_array_equal(D.indices.reshape(idx.shape), idx)
    np.testing.assert_allclose(D.data.reshape(idx.shape), dist, atol=1e-5)
    np.testing.assert_allclose(adata.obsp["snn"].toarray(), snn, atol=1e-7)


def _multimodal_adata(n=400):
    rng = np.random.default_rng(0)
    labels = rng.integers(0, 4, n)
    adata = AnnData(np.zeros((n, 1), dtype=np.float32))
    for key, d, n_groups in (("X_pca", 12, 4), ("X_apca", 8, 2), ("X_lsi", 6, 3)):
        centers = rng.normal(size=(4, d)) * 3
        X = centers[labels % n_groups] + rng.normal(size=(n, d))
        adata.obsm[key] = X.astype(np.float32)
    return adata


@pytest.mark.parametrize("smooth", [False, True])
@pytest.mark.parametrize("n_mod", [2, 3])
def test_wnn_matches_seurat_reference(smooth, n_mod):
    adata = _multimodal_adata()
    keys = ["X_pca", "X_apca", "X_lsi"][:n_mod]
    # NumPy integers, e.g. from a parameter sweep, are accepted
    rsc.pp.wnn(adata, keys, n_neighbors=np.int64(10), knn_range=40, smooth=smooth)
    reps = [adata.obsm[key].astype(np.float64) for key in keys]
    ref = _seurat_wnn_reference(reps, k=10, knn_range=40, smooth=smooth)
    _assert_matches(adata, keys, ref)


def test_wnn_matches_seurat_fixture():
    """Seurat 5.5.1 with exact RANN search, see tests/_scripts/generate_wnn_seurat.py."""
    ref = np.load(Path(__file__).parent / "_data" / "wnn_seurat_5.5.1.npz")
    adata = AnnData(np.zeros((200, 1), dtype=np.float32))
    adata.obsm["X_pca"], adata.obsm["X_apca"] = ref["pca"], ref["apca"]
    rsc.pp.wnn(adata, {"RNA": "X_pca", "ADT": "X_apca"}, knn_range=40)
    ref = ref["weights"].T, ref["indices"], ref["distances"], ref["snn"]
    _assert_matches(adata, ["RNA", "ADT"], ref)


@pytest.mark.parametrize(
    ("k", "max_keys"), [(30, None), (30, 1024), (70, None), (150, None)]
)
def test_wnn_snn_kernels(monkeypatch, k, max_keys):
    """SNN bandwidths and graph match Seurat, with the partners in shared memory
    (by row length) or sorted globally in chunks (bit-identical). k=150 has more
    histogram bins than threads and rows longer than a whole chunk."""
    E = _multimodal_adata(n=600).obsm["X_pca"].astype(np.float64)
    E /= np.linalg.norm(E, axis=1, keepdims=True)
    D = np.linalg.norm(E[:, None] - E[None], axis=2)
    nn = np.argsort(D, axis=1, kind="stable")[:, :k]
    nearest = D[np.arange(len(D)), nn[:, 1]]
    knn, emb = cp.asarray(nn, dtype=cp.int32), cp.asarray(E, dtype=cp.float32)

    def run():
        near = cp.asarray(nearest, dtype=cp.float32)
        sigma = _wnn._snn_bandwidth(knn, emb, near, s_nn=k).get()
        return sigma, _wnn._snn_graph(knn, prune=1 / 15)

    monkeypatch.setattr(_wnn, "_SNN_CHUNK_PAIRS", 1 << 14)
    if max_keys is not None:
        monkeypatch.setattr(_wnn, "_snn_max_keys", lambda s, key_bytes: max_keys)
    sigma, snn = run()
    np.testing.assert_allclose(snn.toarray(), _ref_jaccard(nn, 1 / 15), atol=1e-7)
    np.testing.assert_allclose(
        sigma, _ref_bandwidth(D, nn, nearest), rtol=1e-4, atol=1e-6
    )
    monkeypatch.setattr(_wnn, "_snn_max_keys", lambda s, key_bytes: 0)
    ref_sigma, ref_snn = run()
    np.testing.assert_array_equal(sigma, ref_sigma)
    for attr in ("indptr", "indices", "data"):
        np.testing.assert_array_equal(getattr(snn, attr), getattr(ref_snn, attr))


def test_wnn_outputs_h5ad_and_downstream(tmp_path):
    adata = _multimodal_adata()
    adata.X = None  # the layout must use an existing reduction
    use_rep = {"RNA": "X_pca", "ADT": "X_apca"}
    out = rsc.pp.wnn(
        adata,
        use_rep,
        n_pcs=[10, None],
        n_neighbors=15,
        knn_range=50,
        key_added="wnn",
        copy=True,
    )
    assert "wnn" not in adata.uns
    params = out.uns["wnn"]["params"]
    assert (params["use_rep"], params["n_pcs"]) == ("X_pca", 10)
    assert params["modalities"] == {
        "RNA": {"use_rep": "X_pca", "n_pcs": 10},
        "ADT": {"use_rep": "X_apca", "n_pcs": 8},
    }
    w = out.obs[["RNA_weight", "ADT_weight"]].to_numpy()
    np.testing.assert_allclose(w.sum(axis=1), 1, rtol=1e-6)
    # the RNA-like modality separates more groups, so it should dominate
    assert w[:, 0].mean() > w[:, 1].mean()
    C = out.obsp["wnn_connectivities"]
    assert abs(C - C.T).max() < 1e-6
    assert C.diagonal().max() == 0
    out.write_h5ad(tmp_path / "wnn.h5ad")
    loaded = read_h5ad(tmp_path / "wnn.h5ad")
    assert loaded.uns["wnn"]["params"]["modalities"] == params["modalities"]
    for key in ("wnn_distances", "wnn_connectivities", "wnn_snn"):
        assert (loaded.obsp[key] != out.obsp[key]).nnz == 0
    rsc.tl.leiden(out, neighbors_key="wnn")
    rsc.tl.leiden(out, obsp="wnn_snn", key_added="leiden_snn")
    rsc.tl.umap(out, neighbors_key="wnn")
    sc.tl.umap(loaded, neighbors_key="wnn", maxiter=10, random_state=0)
    assert np.isfinite(loaded.obsm["X_umap"]).all()
    # ingest would silently map through the first modality only
    with pytest.raises(NotImplementedError, match="multimodal"):
        rsc.tl.ingest(out[:50].copy(), out, obs="leiden", neighbors_key="wnn")


@pytest.mark.parametrize("method", ["gauss", "jaccard"])
def test_wnn_connectivity_methods(method):
    adata = _multimodal_adata()
    keys = ["X_pca", "X_apca"]
    rsc.pp.wnn(adata, keys, n_pcs=5, knn_range=50, method=method, weight_key=None)
    params = adata.uns["neighbors"]["params"]
    assert (params["method"], params["modalities"]["X_apca"]["n_pcs"]) == (method, 5)
    assert adata.obs.columns.empty
    C = adata.obsp["connectivities"]
    assert C.nnz > 0
    assert abs(C - C.T).max() < 1e-6


def test_wnn_mudata():
    md = pytest.importorskip("mudata")
    adata = _multimodal_adata()
    mods = {
        name: AnnData(adata.X, obs=adata.obs, obsm={"X_pca": adata.obsm[key]})
        for name, key in (("rna", "X_pca"), ("prot", "X_apca"))
    }
    mdata = md.MuData(mods)
    with pytest.raises(TypeError, match="mapping"):
        rsc.pp.wnn(mdata, ["X_pca", "X_pca"])
    rsc.pp.wnn(mdata, {"rna": "X_pca", "prot": "X_pca"}, n_pcs=[12, None], knn_range=50)
    rsc.pp.wnn(adata, ["X_pca", "X_apca"], knn_range=50)
    np.testing.assert_allclose(
        mdata.obs["rna_weight"], adata.obs["X_pca_weight"], rtol=1e-6
    )
    assert (mdata.obsp["distances"] != adata.obsp["distances"]).nnz == 0
    params = mdata.uns["neighbors"]["params"]
    assert params["modalities"] == {
        "rna": {"use_rep": "X_pca", "n_pcs": 12},
        "prot": {"use_rep": "X_pca", "n_pcs": 8},
    }
    # muon's schema, read by `mu.tl.umap`
    assert params["use_rep"] == {"rna": "X_pca", "prot": "X_pca"}
    assert params["n_pcs"] == {"rna": 12, "prot": 8}


def test_wnn_duplicated_cells():
    # the cell moves to the front of its neighbors, or replaces the last one
    knn = cp.array([[0, 1, 2], [0, 1, 2], [0, 1, 3], [1, 2, 0]], dtype=cp.int32)
    expected = [[0, 1, 2], [1, 0, 2], [2, 0, 1], [3, 1, 2]]
    np.testing.assert_array_equal(_wnn._ensure_self_first(knn).get(), expected)
    adata = _multimodal_adata(n=200)
    for key in ("X_pca", "X_apca"):
        adata.obsm[key][100:110] = adata.obsm[key][100]
    rsc.pp.wnn(adata, ["X_pca", "X_apca"], knn_range=50)
    assert np.isfinite(adata.obs[["X_pca_weight", "X_apca_weight"]]).all().all()
    D = adata.obsp["distances"]
    assert np.all(D.indices.reshape(adata.n_obs, -1) != np.arange(adata.n_obs)[:, None])


def _no_kernel(*args, **kwargs):
    pytest.fail("Incomplete neighbors reached the WNN CUDA kernels")


@pytest.mark.parametrize(
    ("bad_id", "bad_dist"),
    [(np.iinfo(np.int64).max, 1.0), (2**32 + 5, 1.0), (5, np.inf)],
)
def test_wnn_rejects_invalid_neighbors(monkeypatch, bad_id, bad_dist):
    def incomplete_knn(X, Y, k, **kwargs):
        idx = (cp.arange(X.shape[0])[:, None] + cp.arange(k)) % X.shape[0]
        dist = cp.tile(cp.arange(k, dtype=cp.float32), (X.shape[0], 1))
        idx = idx.astype(cp.int64)
        idx[0, -1], dist[0, -1] = bad_id, bad_dist
        return idx, dist

    adata = _multimodal_adata(n=40)
    monkeypatch.setitem(_wnn.KNN_ALGORITHMS, "brute", incomplete_knn)
    monkeypatch.setattr(_wnn_cuda, "impute_dist", _no_kernel)
    with pytest.raises(ValueError, match="incomplete"):
        rsc.pp.wnn(adata, ["X_pca", "X_apca"], n_neighbors=4, knn_range=24)
    assert not adata.obsp
    assert "neighbors" not in adata.uns


def test_wnn_incomplete_ivf_search_can_be_retried(monkeypatch):
    adata = _multimodal_adata(n=500)
    kwargs = {"use_rep": ["X_pca", "X_apca"], "algorithm": "ivfflat"}
    with monkeypatch.context() as patch:
        patch.setattr(_wnn_cuda, "impute_dist", _no_kernel)
        with pytest.raises(ValueError, match="n_probes"):
            rsc.pp.wnn(adata, **kwargs, algorithm_kwds={"n_lists": 64, "n_probes": 1})
    # searching all lists repairs the candidates in the same CUDA context
    rsc.pp.wnn(adata, **kwargs, algorithm_kwds={"n_lists": 64, "n_probes": 64})
    D = adata.obsp["distances"]
    assert np.all(np.diff(D.indptr) == 20)
    assert np.isfinite(D.data).all()


@pytest.mark.parametrize(
    ("kwargs", "match"),
    [
        ({"use_rep": ["X_pca"]}, "two modalities"),
        ({"use_rep": "X_pca"}, "one representation per modality"),
        ({"knn_range": 10, "n_neighbors": 20}, "larger than"),
        ({"knn_range": 1000}, "at most the number of cells"),
        ({"n_pcs": [5]}, "one entry per modality"),
        ({"cross_constant": [1e-4]}, "one entry per modality"),
        ({"algorithm": "kd_tree"}, "Invalid algorithm"),
        ({"use_rep": ["X_pca", "X_pca"]}, "unique strings"),
        ({"weight_key": "weight"}, "distinct column"),
    ],
)
def test_wnn_errors(kwargs, match):
    adata = _multimodal_adata()
    kwargs = {"use_rep": ["X_pca", "X_apca"], "knn_range": 50} | kwargs
    with pytest.raises((ValueError, TypeError), match=match):
        rsc.pp.wnn(adata, **kwargs)
    assert not adata.obsp and "neighbors" not in adata.uns
