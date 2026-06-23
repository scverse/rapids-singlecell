from __future__ import annotations

import cupy as cp
import cupyx.scipy.sparse as cpsp
import numpy as np
import pandas as pd
import pytest
import scanpy as sc
import scipy.sparse as sp
from scipy.stats import mannwhitneyu

import rapids_singlecell as rsc


def _to_format(X_dense, fmt):
    if fmt == "numpy_dense":
        return np.asarray(X_dense)
    if fmt == "scipy_csr":
        return sp.csr_matrix(X_dense)
    if fmt == "scipy_csc":
        return sp.csc_matrix(X_dense)
    if fmt == "cupy_dense":
        return cp.asarray(X_dense)
    if fmt == "cupy_csr":
        return cpsp.csr_matrix(cp.asarray(X_dense))
    if fmt == "cupy_csc":
        return cpsp.csc_matrix(cp.asarray(X_dense))
    raise ValueError(f"Unknown format: {fmt}")


def _make_nonnegative(adata):
    adata.X = np.abs(np.asarray(adata.X)).astype(np.float32)
    return adata


# The optimized sparse Wilcoxon paths inject implicit zeros analytically as a tie
# at the column minimum, which is valid only for nonnegative data. With negatives
# present they must NOT be used; instead the ranking falls back to the dense
# full-sort path (correct for any sign), so the result matches running the same
# method on the dense matrix. (t-test/t-test_overestim_var/logreg never need this
# and accept signed sparse data directly -- e.g. mixscape's LDA t-test.)
@pytest.mark.parametrize(
    "method",
    ["wilcoxon", "wilcoxon_binned"],
)
@pytest.mark.parametrize("fmt", ["scipy_csr", "scipy_csc", "cupy_csr", "cupy_csc"])
def test_rank_genes_groups_sparse_negative_values_fallback(method, fmt):
    X = np.array(
        [
            [-1.0, 0.0, 2.0],
            [0.0, 1.0, 0.0],
            [2.0, 0.0, 1.0],
            [0.0, 3.0, 0.0],
            [-2.0, 1.0, 0.0],
            [1.0, 0.0, 3.0],
        ],
        dtype=np.float64,
    )
    obs = pd.DataFrame({"group": pd.Categorical(list("aaabbb"), categories=["a", "b"])})
    var = pd.DataFrame(index=["g0", "g1", "g2"])

    sparse_adata = sc.AnnData(X=_to_format(X, fmt), obs=obs.copy(), var=var.copy())
    dense_fmt = "cupy_dense" if fmt.startswith("cupy") else "numpy_dense"
    dense_adata = sc.AnnData(X=_to_format(X, dense_fmt), obs=obs.copy(), var=var.copy())

    rsc.tl.rank_genes_groups(sparse_adata, "group", method=method, use_raw=False)
    rsc.tl.rank_genes_groups(dense_adata, "group", method=method, use_raw=False)

    # Sparse-with-negatives falls back to the dense ranking -> identical result.
    sp_scores = sparse_adata.uns["rank_genes_groups"]["scores"]
    dn_scores = dense_adata.uns["rank_genes_groups"]["scores"]
    for group in sp_scores.dtype.names:
        np.testing.assert_allclose(
            np.asarray(sp_scores[group], dtype=float),
            np.asarray(dn_scores[group], dtype=float),
            rtol=1e-13,
            atol=1e-13,
        )


@pytest.mark.parametrize("layout", ["csr", "csc"])
@pytest.mark.parametrize("reference", ["rest", "1"])
def test_device_sparse_int64_indptr_matches_scanpy(layout, reference):
    # Real int64 indptr only occurs at nnz > 2^31 (unallocatable in CI). cupy
    # >= 14.1 preserves an explicitly promoted int64 indptr, so a small matrix
    # promoted to int64 drives the *_i64 device kernels through the public API.
    rng = np.random.default_rng(0)
    dense = np.abs(rng.standard_normal((150, 8))).astype(np.float32)
    dense[dense < 0.5] = 0.0
    obs = pd.DataFrame({"group": pd.Categorical([f"{i % 3}" for i in range(150)])})
    var = pd.DataFrame(index=[f"g{j}" for j in range(8)])

    ctor = cpsp.csr_matrix if layout == "csr" else cpsp.csc_matrix
    mat = ctor(cp.asarray(dense))
    mat.indptr = mat.indptr.astype(cp.int64)
    mat.indices = mat.indices.astype(cp.int64)
    assert mat.indptr.dtype == cp.int64

    adata = sc.AnnData(X=mat, obs=obs.copy(), var=var.copy())
    adata_cpu = sc.AnnData(X=dense.copy(), obs=obs.copy(), var=var.copy())
    kw = {
        "method": "wilcoxon",
        "use_raw": False,
        "reference": reference,
        "tie_correct": True,
        "n_genes": 8,
    }
    rsc.tl.rank_genes_groups(adata, "group", **kw)
    sc.tl.rank_genes_groups(adata_cpu, "group", **kw)
    g = adata.uns["rank_genes_groups"]
    c = adata_cpu.uns["rank_genes_groups"]
    for field in ("scores", "pvals", "pvals_adj"):
        for grp in g[field].dtype.names:
            np.testing.assert_allclose(
                np.asarray(g[field][grp], dtype=float),
                np.asarray(c[field][grp], dtype=float),
                rtol=1e-13,
                atol=1e-15,
                equal_nan=True,
            )


def test_rank_genes_groups_structured_results_get_df_and_h5ad_match_scanpy(tmp_path):
    np.random.seed(42)
    adata_rsc = sc.datasets.blobs(n_variables=6, n_centers=3, n_observations=120)
    _make_nonnegative(adata_rsc)
    adata_rsc.obs["blobs"] = adata_rsc.obs["blobs"].astype("category")
    adata_rsc.X = sp.csr_matrix(adata_rsc.X)
    adata_cpu = adata_rsc.copy()
    adata_cpu.X = adata_cpu.X.toarray()

    kw = {
        "groupby": "blobs",
        "method": "wilcoxon",
        "reference": "1",
        "use_raw": False,
        "tie_correct": True,
        "n_genes": 4,
    }
    rsc.tl.rank_genes_groups(adata_rsc, **kw)
    sc.tl.rank_genes_groups(adata_cpu, **kw)

    rsc_result = adata_rsc.uns["rank_genes_groups"]
    assert isinstance(rsc_result["names"], np.ndarray)
    assert rsc_result["names"].dtype.names == ("0", "2")
    assert tuple(rsc_result["names"][0]) == tuple(
        adata_cpu.uns["rank_genes_groups"]["names"][0]
    )
    np.testing.assert_array_equal(
        rsc_result["names"].copy(),
        np.asarray(rsc_result["names"]),
    )

    h5ad_path = tmp_path / "rank_genes_groups.h5ad"
    adata_rsc.write_h5ad(h5ad_path)
    adata_rsc = sc.read_h5ad(h5ad_path)

    rsc_df = sc.get.rank_genes_groups_df(adata_rsc, group=None)
    scanpy_df = sc.get.rank_genes_groups_df(adata_cpu, group=None)
    pd.testing.assert_frame_equal(rsc_df, scanpy_df)


def test_rank_genes_groups_return_format_removed():
    adata = sc.datasets.blobs(n_variables=3, n_centers=2, n_observations=20)
    _make_nonnegative(adata)
    adata.obs["blobs"] = adata.obs["blobs"].astype("category")

    with pytest.raises(TypeError, match="return_format has been removed"):
        rsc.tl.rank_genes_groups(
            adata,
            "blobs",
            method="wilcoxon",
            use_raw=False,
            return_format="arrays",
        )


@pytest.mark.parametrize("reference", ["rest", "b"])
@pytest.mark.parametrize("fmt", ["numpy_dense", "scipy_csr", "cupy_csr"])
def test_rank_genes_groups_wilcoxon_return_u_values(reference, fmt):
    X = np.array(
        [
            [5.0, 0.0, 1.0, 2.0],
            [4.0, 0.0, 1.0, 2.0],
            [1.0, 3.0, 2.0, 2.0],
            [0.0, 2.0, 2.0, 2.0],
            [2.0, 1.0, 0.0, 3.0],
            [3.0, 1.0, 0.0, 3.0],
        ],
        dtype=np.float32,
    )
    labels = np.array(["a", "a", "b", "b", "c", "c"])
    adata = sc.AnnData(
        X=_to_format(X, fmt),
        obs=pd.DataFrame({"group": pd.Categorical(labels)}),
        var=pd.DataFrame(index=[f"g{i}" for i in range(X.shape[1])]),
    )

    rsc.tl.rank_genes_groups(
        adata,
        "group",
        groups=["a"],
        reference=reference,
        method="wilcoxon",
        use_raw=False,
        tie_correct=True,
        use_continuity=True,
        return_u_values=True,
        n_genes=adata.n_vars,
    )

    result = adata.uns["rank_genes_groups"]
    assert result["params"]["return_u_values"] is True
    assert result["scores"].dtype["a"] == np.dtype("float64")

    df = sc.get.rank_genes_groups_df(adata, group="a").sort_values("names")
    mask_group = labels == "a"
    mask_ref = labels != "a" if reference == "rest" else labels == reference
    expected = np.array(
        [
            mannwhitneyu(
                X[mask_group, gene],
                X[mask_ref, gene],
                alternative="two-sided",
            ).statistic
            for gene in range(X.shape[1])
        ],
        dtype=np.float64,
    )

    gene_to_idx = {name: idx for idx, name in enumerate(adata.var_names)}
    expected_sorted = np.array([expected[gene_to_idx[name]] for name in df["names"]])
    np.testing.assert_allclose(df["scores"].to_numpy(), expected_sorted)


def test_rank_genes_groups_wilcoxon_dense_edge_cases_match_scipy():
    X = np.array(
        [
            [1.0, 5.0, 0.0, 2.0, 1.0],
            [2.0, 5.0, 0.0, 2.0, 1.0],
            [3.0, 5.0, 1.0, 2.0, 1.0],
            [4.0, 5.0, 1.0, 3.0, 2.0],
            [5.0, 5.0, 1.0, 3.0, 2.0],
            [6.0, 5.0, 2.0, 3.0, 2.0],
            [7.0, 5.0, 2.0, 4.0, 3.0],
            [8.0, 5.0, 2.0, 4.0, 3.0],
        ],
        dtype=np.float32,
    )
    labels = np.array(["a", "a", "a", "a", "b", "b", "b", "b"])
    adata = sc.AnnData(
        X=X,
        obs=pd.DataFrame({"group": pd.Categorical(labels)}),
        var=pd.DataFrame(index=["no_ties", "all_ties", "zero_ties", "mixed", "pairs"]),
    )
    rsc.tl.rank_genes_groups(
        adata,
        "group",
        groups=["a"],
        reference="b",
        method="wilcoxon",
        use_raw=False,
        tie_correct=True,
        use_continuity=True,
        return_u_values=True,
        n_genes=adata.n_vars,
    )

    df = sc.get.rank_genes_groups_df(adata, group="a").sort_values("names")
    expected_u = {}
    for idx, name in enumerate(adata.var_names):
        result = mannwhitneyu(
            X[labels == "a", idx],
            X[labels == "b", idx],
            alternative="two-sided",
            method="asymptotic",
            use_continuity=True,
        )
        expected_u[name] = result.statistic

    np.testing.assert_allclose(
        df["scores"].to_numpy(),
        np.array([expected_u[name] for name in df["names"]]),
        rtol=1e-13,
        atol=1e-15,
    )
    assert np.isfinite(df["pvals"]).all()


def test_rank_genes_groups_return_u_values_requires_wilcoxon():
    adata = sc.datasets.blobs(n_variables=3, n_centers=2, n_observations=20)
    _make_nonnegative(adata)
    adata.obs["blobs"] = adata.obs["blobs"].astype("category")

    with pytest.raises(ValueError, match="only supported for method='wilcoxon'"):
        rsc.tl.rank_genes_groups(
            adata,
            "blobs",
            method="t-test",
            use_raw=False,
            return_u_values=True,
        )


@pytest.mark.parametrize("reference", ["rest", "1"])
@pytest.mark.parametrize("tie_correct", [True, False])
@pytest.mark.parametrize("sparse", [True, False])
def test_rank_genes_groups_wilcoxon_matches_scanpy(reference, tie_correct, sparse):
    """Test wilcoxon matches scanpy output across configurations."""
    np.random.seed(42)
    adata_gpu = sc.datasets.blobs(n_variables=6, n_centers=3, n_observations=200)
    _make_nonnegative(adata_gpu)
    adata_gpu.obs["blobs"] = adata_gpu.obs["blobs"].astype("category")

    if sparse:
        adata_gpu.X = sp.csr_matrix(adata_gpu.X)

    adata_cpu = adata_gpu.copy()

    rsc.tl.rank_genes_groups(
        adata_gpu,
        "blobs",
        method="wilcoxon",
        use_raw=False,
        n_genes=3,
        reference=reference,
        corr_method="benjamini-hochberg",
        tie_correct=tie_correct,
    )
    sc.tl.rank_genes_groups(
        adata_cpu,
        "blobs",
        method="wilcoxon",
        use_raw=False,
        n_genes=3,
        reference=reference,
        tie_correct=tie_correct,
    )

    gpu_result = adata_gpu.uns["rank_genes_groups"]
    cpu_result = adata_cpu.uns["rank_genes_groups"]

    assert gpu_result["names"].dtype.names == cpu_result["names"].dtype.names
    for group in gpu_result["names"].dtype.names:
        assert list(gpu_result["names"][group]) == list(cpu_result["names"][group])

    for field in ("scores", "logfoldchanges", "pvals", "pvals_adj"):
        gpu_field = gpu_result[field]
        cpu_field = cpu_result[field]
        rtol = 1e-13
        assert gpu_field.dtype.names == cpu_field.dtype.names
        for group in gpu_field.dtype.names:
            gpu_values = np.asarray(gpu_field[group], dtype=float)
            cpu_values = np.asarray(cpu_field[group], dtype=float)
            atol = 1e-15
            np.testing.assert_allclose(gpu_values, cpu_values, rtol=rtol, atol=atol)

    params = gpu_result["params"]
    assert params["use_raw"] is False
    assert params["corr_method"] == "benjamini-hochberg"
    assert params["tie_correct"] is tie_correct
    assert params["layer"] is None
    assert params["reference"] == reference


def test_rank_genes_groups_wilcoxon_dense_ovr_ties_match_scanpy():
    rng = np.random.default_rng(16)
    X = rng.integers(0, 40, size=(128, 7)).astype(np.float32)
    labels = rng.integers(0, 7, size=128).astype(str)
    adata_gpu = sc.AnnData(
        X=X.copy(),
        obs=pd.DataFrame({"group": pd.Categorical(labels)}),
        var=pd.DataFrame(index=[f"g{i}" for i in range(X.shape[1])]),
    )
    adata_cpu = adata_gpu.copy()

    kw = {
        "groupby": "group",
        "method": "wilcoxon",
        "reference": "rest",
        "use_raw": False,
        "tie_correct": True,
        "n_genes": adata_gpu.n_vars,
    }
    rsc.tl.rank_genes_groups(adata_gpu, **kw)
    sc.tl.rank_genes_groups(adata_cpu, **kw)

    gpu_result = adata_gpu.uns["rank_genes_groups"]
    cpu_result = adata_cpu.uns["rank_genes_groups"]
    for group in gpu_result["scores"].dtype.names:
        assert list(gpu_result["names"][group]) == list(cpu_result["names"][group])
        np.testing.assert_allclose(
            gpu_result["scores"][group], cpu_result["scores"][group], rtol=1e-13
        )
        np.testing.assert_allclose(
            gpu_result["pvals"][group], cpu_result["pvals"][group], rtol=1e-13
        )


@pytest.mark.parametrize("reference", ["rest", "1"])
def test_rank_genes_groups_wilcoxon_honors_layer_and_use_raw(reference):
    """Test that layer parameter is respected."""
    np.random.seed(42)
    base = sc.datasets.blobs(n_variables=5, n_centers=3, n_observations=150)
    _make_nonnegative(base)
    base.obs["blobs"] = base.obs["blobs"].astype("category")
    base.layers["signal"] = base.X.copy()

    ref_adata = base.copy()
    rsc.tl.rank_genes_groups(
        ref_adata, "blobs", method="wilcoxon", use_raw=False, reference=reference
    )
    reference_names = ref_adata.uns["rank_genes_groups"]["names"].copy()

    rng = np.random.default_rng(0)
    perturbed_matrix = base.X.copy()
    perturbed_matrix[rng.integers(0, 2, perturbed_matrix.shape, dtype=bool)] = 0.0

    layered = base.copy()
    layered.X = perturbed_matrix
    rsc.tl.rank_genes_groups(
        layered,
        "blobs",
        method="wilcoxon",
        layer="signal",
        use_raw=False,
        reference=reference,
    )
    layered_names = layered.uns["rank_genes_groups"]["names"].copy()

    no_layer = base.copy()
    no_layer.X = perturbed_matrix
    rsc.tl.rank_genes_groups(
        no_layer, "blobs", method="wilcoxon", use_raw=False, reference=reference
    )
    no_layer_names = no_layer.uns["rank_genes_groups"]["names"].copy()

    assert layered_names.dtype.names == reference_names.dtype.names
    for group in reference_names.dtype.names:
        assert tuple(layered_names[group]) == tuple(reference_names[group])
    differences = [
        tuple(no_layer_names[group]) != tuple(reference_names[group])
        for group in reference_names.dtype.names
    ]
    assert any(differences)


@pytest.mark.parametrize("reference", ["rest", "1"])
def test_rank_genes_groups_wilcoxon_subset_and_bonferroni(reference):
    """Test group subsetting and bonferroni correction."""
    np.random.seed(42)
    adata = sc.datasets.blobs(n_variables=5, n_centers=4, n_observations=150)
    _make_nonnegative(adata)
    adata.obs["blobs"] = adata.obs["blobs"].astype("category")

    groups = ["0", "1", "2"] if reference != "rest" else ["0", "2"]

    rsc.tl.rank_genes_groups(
        adata,
        "blobs",
        method="wilcoxon",
        groups=groups,
        reference=reference,
        use_raw=False,
        n_genes=2,
        corr_method="bonferroni",
    )

    result = adata.uns["rank_genes_groups"]
    expected_groups = tuple(g for g in groups if g != reference)
    assert result["scores"].dtype.names == expected_groups
    assert result["names"].dtype.names == expected_groups
    for group in result["names"].dtype.names:
        observed = np.asarray(result["names"][group])
        assert observed.size == 2
    for group in result["pvals_adj"].dtype.names:
        adjusted = np.asarray(result["pvals_adj"][group])
        assert np.all(adjusted <= 1.0)


def test_rank_genes_groups_wilcoxon_skip_empty_groups_filters_singletons():
    np.random.seed(42)
    adata = sc.datasets.blobs(n_variables=5, n_centers=2, n_observations=21)
    _make_nonnegative(adata)
    adata.obs["target"] = pd.Categorical(
        ["ref"] * 10 + ["valid"] * 10 + ["singleton"],
        categories=["ref", "valid", "singleton", "empty"],
    )

    rsc.tl.rank_genes_groups(
        adata,
        "target",
        method="wilcoxon",
        reference="ref",
        use_raw=False,
        n_genes=3,
        skip_empty_groups=True,
    )

    result = adata.uns["rank_genes_groups"]
    assert result["names"].dtype.names == ("valid",)
    assert result["scores"].dtype.names == ("valid",)


def test_rank_genes_groups_wilcoxon_skip_empty_groups_all_tests_filtered():
    np.random.seed(42)
    adata = sc.datasets.blobs(n_variables=5, n_centers=2, n_observations=11)
    _make_nonnegative(adata)
    adata.obs["target"] = pd.Categorical(
        ["ref"] * 10 + ["singleton"],
        categories=["ref", "singleton", "empty"],
    )

    rsc.tl.rank_genes_groups(
        adata,
        "target",
        method="wilcoxon",
        reference="ref",
        use_raw=False,
        skip_empty_groups=True,
    )

    result = adata.uns["rank_genes_groups"]
    assert "names" not in result
    assert result["params"]["reference"] == "ref"


@pytest.mark.parametrize(
    "fmt",
    [
        pytest.param("scipy_csr", id="host_csr"),
        pytest.param("scipy_csc", id="host_csc"),
        pytest.param("cupy_dense", id="device_dense"),
    ],
)
def test_wilcoxon_subset_rest_stats_match_scanpy(fmt):
    """groups=... with reference='rest' must use all other cells for stats."""
    np.random.seed(42)
    adata_gpu = sc.datasets.blobs(n_variables=6, n_centers=4, n_observations=160)
    _make_nonnegative(adata_gpu)
    adata_gpu.obs["blobs"] = adata_gpu.obs["blobs"].astype("category")
    adata_cpu = adata_gpu.copy()
    adata_gpu.X = _to_format(adata_gpu.X, fmt)

    kw = {
        "groupby": "blobs",
        "method": "wilcoxon",
        "use_raw": False,
        "groups": ["0", "2"],
        "reference": "rest",
        "pts": True,
        "n_genes": 6,
    }
    rsc.tl.rank_genes_groups(adata_gpu, **kw)
    sc.tl.rank_genes_groups(adata_cpu, **kw)

    gpu_result = adata_gpu.uns["rank_genes_groups"]
    cpu_result = adata_cpu.uns["rank_genes_groups"]
    for field in ("scores", "logfoldchanges", "pvals", "pvals_adj"):
        rtol = 1e-13
        atol = 1e-15
        for group in gpu_result[field].dtype.names:
            np.testing.assert_allclose(
                np.asarray(gpu_result[field][group], dtype=float),
                np.asarray(cpu_result[field][group], dtype=float),
                rtol=rtol,
                atol=atol,
                equal_nan=True,
            )

    for key in ("pts", "pts_rest"):
        gpu_pts = gpu_result[key]
        cpu_pts = cpu_result[key]
        for col in gpu_pts.columns:
            np.testing.assert_allclose(
                gpu_pts[col].values, cpu_pts[col].values, rtol=1e-13, atol=1e-15
            )


@pytest.mark.parametrize("reference", ["rest", "1"])
@pytest.mark.parametrize("fmt", ["scipy_csr", "scipy_csc"])
def test_wilcoxon_zero_nnz_host_sparse_does_not_crash(reference, fmt):
    obs = pd.DataFrame(
        {
            "group": pd.Categorical(
                ["0"] * 4 + ["1"] * 4 + ["2"] * 4,
                categories=["0", "1", "2"],
            )
        }
    )
    adata = sc.AnnData(
        X=_to_format(np.zeros((12, 5), dtype=np.float32), fmt),
        obs=obs,
        var=pd.DataFrame(index=[f"g{i}" for i in range(5)]),
    )

    rsc.tl.rank_genes_groups(
        adata,
        "group",
        method="wilcoxon",
        use_raw=False,
        reference=reference,
        pts=True,
    )

    result = adata.uns["rank_genes_groups"]
    for field in ("scores", "pvals"):
        for group in result[field].dtype.names:
            assert np.all(np.isfinite(np.asarray(result[field][group], dtype=float)))


def test_wilcoxon_ovo_host_csr_unsorted_indices_match_sorted():
    rng = np.random.default_rng(42)
    dense = rng.poisson(1.0, size=(80, 12)).astype(np.float32)
    dense[rng.random(dense.shape) < 0.55] = 0
    sorted_csr = sp.csr_matrix(dense)
    unsorted_csr = sorted_csr.copy()
    for row in range(unsorted_csr.shape[0]):
        start, stop = unsorted_csr.indptr[row : row + 2]
        order = np.arange(stop - start)[::-1]
        unsorted_csr.indices[start:stop] = unsorted_csr.indices[start:stop][order]
        unsorted_csr.data[start:stop] = unsorted_csr.data[start:stop][order]
    unsorted_csr.has_sorted_indices = False

    obs = pd.DataFrame(
        {
            "group": pd.Categorical(
                ["ref"] * 20 + ["a"] * 20 + ["b"] * 20 + ["c"] * 20,
                categories=["ref", "a", "b", "c"],
            )
        }
    )
    var = pd.DataFrame(index=[f"g{i}" for i in range(dense.shape[1])])
    sorted_adata = sc.AnnData(X=sorted_csr, obs=obs.copy(), var=var.copy())
    unsorted_adata = sc.AnnData(X=unsorted_csr, obs=obs.copy(), var=var.copy())

    kw = {
        "groupby": "group",
        "method": "wilcoxon",
        "reference": "ref",
        "use_raw": False,
        "tie_correct": True,
        "n_genes": dense.shape[1],
    }
    rsc.tl.rank_genes_groups(sorted_adata, **kw)
    rsc.tl.rank_genes_groups(unsorted_adata, **kw)

    sorted_result = sorted_adata.uns["rank_genes_groups"]
    unsorted_result = unsorted_adata.uns["rank_genes_groups"]
    for field in ("scores", "logfoldchanges", "pvals", "pvals_adj"):
        for group in sorted_result[field].dtype.names:
            np.testing.assert_allclose(
                np.asarray(unsorted_result[field][group], dtype=float),
                np.asarray(sorted_result[field][group], dtype=float),
                rtol=1e-13,
                atol=1e-15,
                equal_nan=True,
            )


@pytest.mark.parametrize("reference", ["rest", "1"])
@pytest.mark.parametrize(
    "fmt",
    [
        "numpy_dense",
        "scipy_csr",
        "scipy_csc",
        "cupy_dense",
        "cupy_csr",
        "cupy_csc",
    ],
)
def test_wilcoxon_all_public_formats_match_scanpy(reference, fmt):
    np.random.seed(42)
    adata_gpu = sc.datasets.blobs(n_variables=5, n_centers=3, n_observations=120)
    _make_nonnegative(adata_gpu)
    adata_gpu.obs["blobs"] = adata_gpu.obs["blobs"].astype("category")
    adata_cpu = adata_gpu.copy()
    adata_gpu.X = _to_format(adata_gpu.X, fmt)

    kw = {
        "groupby": "blobs",
        "method": "wilcoxon",
        "use_raw": False,
        "reference": reference,
        "tie_correct": True,
        "n_genes": 5,
    }
    rsc.tl.rank_genes_groups(adata_gpu, **kw)
    sc.tl.rank_genes_groups(adata_cpu, **kw)

    gpu_result = adata_gpu.uns["rank_genes_groups"]
    cpu_result = adata_cpu.uns["rank_genes_groups"]
    for field in ("scores", "logfoldchanges", "pvals", "pvals_adj"):
        rtol = 1e-13
        atol = 1e-15
        for group in gpu_result[field].dtype.names:
            np.testing.assert_allclose(
                np.asarray(gpu_result[field][group], dtype=float),
                np.asarray(cpu_result[field][group], dtype=float),
                rtol=rtol,
                atol=atol,
                equal_nan=True,
            )


def _make_sized_groups_adata(group_sizes, n_genes, seed=0):
    """AnnData with exact per-group sizes (drives OVO tier selection by max size)."""
    rng = np.random.default_rng(seed)
    n_obs = int(sum(group_sizes))
    X = np.abs(rng.standard_normal((n_obs, n_genes))).astype(np.float32)
    X[X < 0.3] = 0.0  # zeros create tie groups, exercising tie correction
    labels = np.concatenate(
        [np.full(sz, f"g{i}", dtype=object) for i, sz in enumerate(group_sizes)]
    )
    obs = pd.DataFrame({"group": pd.Categorical(labels)})
    var = pd.DataFrame(index=[f"gene_{j}" for j in range(n_genes)])
    adata = sc.AnnData(X=X, obs=obs, var=var)
    adata.uns["log1p"] = {"base": None}
    return adata


# Tier thresholds (wilcoxon_fast_common.cuh): tier0<=32, tier0_64<=64,
# tier2<=512, tier1(fused smem sort)<=2500, tier3(CUB segmented sort)>2500.
# Group sizes in the standard blobs datasets are <=~70, so tier1/tier3 are
# otherwise never exercised. These force a single large test group.
@pytest.mark.parametrize(
    "fmt", ["numpy_dense", "scipy_csr", "scipy_csc", "cupy_csr", "cupy_csc"]
)
@pytest.mark.parametrize("tie_correct", [False, True])
@pytest.mark.parametrize("big", [700, 3000], ids=["tier1_fused", "tier3_cub"])
def test_wilcoxon_ovo_large_group_tiers_match_scanpy(fmt, tie_correct, big):
    # g0 = reference, g1 = the large test group that drives tier selection.
    adata_gpu = _make_sized_groups_adata([60, big, 45], n_genes=6, seed=1)
    adata_cpu = adata_gpu.copy()
    adata_gpu.X = _to_format(adata_gpu.X, fmt)

    kw = {
        "groupby": "group",
        "method": "wilcoxon",
        "use_raw": False,
        "reference": "g0",
        "tie_correct": tie_correct,
        "n_genes": 6,
    }
    rsc.tl.rank_genes_groups(adata_gpu, **kw)
    sc.tl.rank_genes_groups(adata_cpu, **kw)

    gpu = adata_gpu.uns["rank_genes_groups"]
    cpu = adata_cpu.uns["rank_genes_groups"]
    for field in ("scores", "pvals"):
        for group in gpu[field].dtype.names:
            np.testing.assert_allclose(
                np.asarray(gpu[field][group], dtype=float),
                np.asarray(cpu[field][group], dtype=float),
                rtol=1e-13,
                atol=1e-15,
                equal_nan=True,
            )


@pytest.mark.parametrize(
    "fmt", ["numpy_dense", "scipy_csr", "scipy_csc", "cupy_csr", "cupy_csc"]
)
def test_wilcoxon_ovo_mixed_tier_sizes_match_scanpy(fmt):
    # Groups spanning tier0 (20), tier0_64 (50) and tier2 (300) co-launched with
    # tie_correct=True, pinning the skip_le boundaries and the ref_tie_sums gate.
    adata_gpu = _make_sized_groups_adata([80, 20, 50, 300], n_genes=6, seed=2)
    adata_cpu = adata_gpu.copy()
    adata_gpu.X = _to_format(adata_gpu.X, fmt)

    kw = {
        "groupby": "group",
        "method": "wilcoxon",
        "use_raw": False,
        "reference": "g0",
        "tie_correct": True,
        "n_genes": 6,
    }
    rsc.tl.rank_genes_groups(adata_gpu, **kw)
    sc.tl.rank_genes_groups(adata_cpu, **kw)

    gpu = adata_gpu.uns["rank_genes_groups"]
    cpu = adata_cpu.uns["rank_genes_groups"]
    for field in ("scores", "pvals"):
        for group in gpu[field].dtype.names:
            np.testing.assert_allclose(
                np.asarray(gpu[field][group], dtype=float),
                np.asarray(cpu[field][group], dtype=float),
                rtol=1e-13,
                atol=1e-15,
                equal_nan=True,
            )


# n_groups > ~3056 makes the per-block smem for the sparse-OVR accumulator
# ((2*n_groups+32) doubles) exceed the 48KB static limit, so sparse_ovr_smem_config
# (and the dense ovr_smem_config) fall back to the global-memory accumulator.
# This is the perturbation regime (thousands of guides vs rest). scanpy's
# 3000+-group DataFrame build is O(n_groups^2) and too slow for an in-suite
# parity check; gmem-vs-scanpy parity is verified out-of-band (<=2e-15). Here we
# guard that every storage format (incl. the dense reference kernel) agrees at
# gmem scale, with and without tie correction.
@pytest.mark.parametrize("tie_correct", [False, True])
def test_wilcoxon_ovr_many_groups_gmem_formats_agree(tie_correct):
    adata = _make_sized_groups_adata([26] * 3100, n_genes=6, seed=3)
    ref = None
    for fmt in ("numpy_dense", "scipy_csr", "scipy_csc", "cupy_csr", "cupy_csc"):
        a = adata.copy()
        a.X = _to_format(adata.X, fmt)
        rsc.tl.rank_genes_groups(
            a,
            "group",
            method="wilcoxon",
            use_raw=False,
            reference="rest",
            tie_correct=tie_correct,
            n_genes=6,
        )
        r = a.uns["rank_genes_groups"]
        cur = {
            field: np.vstack(
                [np.asarray(r[field][n], dtype=float) for n in r[field].dtype.names]
            )
            for field in ("scores", "pvals")
        }
        if ref is None:
            ref = cur
            continue
        for field in ("scores", "pvals"):
            np.testing.assert_allclose(
                cur[field], ref[field], rtol=1e-13, atol=1e-15, equal_nan=True
            )


# Regression guard for the host-dense OVR streaming launcher: in gmem rank mode
# the rank kernel atomicAdds onto its per-stream rank-sum buffer without
# self-zeroing, and those buffers are reused round-robin -- so the launcher must
# zero them per sub-batch. This only bites past the DENSE gmem flip
# (n_groups > ~6112) AND with enough genes (> N_STREAMS*sub_batch_cols = 256)
# that the per-stream buffers actually wrap; the formats-agree gmem test above
# uses n_genes=6 (one batch, fresh buffer) and cannot see it. Compares the host
# (numpy) dense path against the device (cupy) dense + sparse paths, which zero
# the gmem buffer correctly.
@pytest.mark.filterwarnings("ignore::RuntimeWarning")  # 6200 tiny groups warn
def test_wilcoxon_ovr_dense_gmem_host_streaming_buffer_reuse():
    adata = _make_sized_groups_adata([2] * 6200, n_genes=400, seed=7)
    ref = None
    for fmt in ("cupy_dense", "numpy_dense", "cupy_csr"):
        a = adata.copy()
        a.X = _to_format(adata.X, fmt)
        rsc.tl.rank_genes_groups(
            a,
            "group",
            method="wilcoxon",
            use_raw=False,
            reference="rest",
            tie_correct=True,
        )
        r = a.uns["rank_genes_groups"]
        cur = {
            field: np.vstack(
                [np.asarray(r[field][n], dtype=float) for n in r[field].dtype.names]
            )
            for field in ("scores", "pvals")
        }
        if ref is None:
            ref = cur
            continue
        for field in ("scores", "pvals"):
            np.testing.assert_allclose(
                cur[field], ref[field], rtol=1e-13, atol=1e-15, equal_nan=True
            )


# Host-dense OVR has only float32/float64 nanobind overloads; integer/bool/uint/
# float16 numpy must be cast to float32 (mirrors the sparse path) rather than
# raising a TypeError.
@pytest.mark.parametrize(
    "data_dtype", [np.int32, np.int64, np.uint16, np.float16, bool]
)
def test_wilcoxon_dense_nonfloat_data_matches_float32(data_dtype):
    rng = np.random.default_rng(5)
    n_obs, n_genes = 120, 8
    counts = rng.integers(0, 5, size=(n_obs, n_genes))
    if data_dtype is bool:
        counts = counts > 2
    typed = np.ascontiguousarray(counts.astype(data_dtype))
    f32 = np.ascontiguousarray(counts.astype(np.float32))
    labels = np.array([f"{i % 3}" for i in range(n_obs)])
    obs = pd.DataFrame({"group": pd.Categorical(labels)})
    var = pd.DataFrame(index=[f"g{j}" for j in range(n_genes)])

    def run(arr):
        adata = sc.AnnData(X=arr, obs=obs.copy(), var=var.copy())
        adata.uns["log1p"] = {"base": None}
        rsc.tl.rank_genes_groups(
            adata,
            "group",
            method="wilcoxon",
            use_raw=False,
            reference="rest",
            tie_correct=True,
            n_genes=n_genes,
        )
        return adata.uns["rank_genes_groups"]

    r_typed = run(typed)
    r_f32 = run(f32)
    for grp in r_typed["scores"].dtype.names:
        np.testing.assert_array_equal(
            np.asarray(r_typed["scores"][grp], dtype=float),
            np.asarray(r_f32["scores"][grp], dtype=float),
        )


# F-contiguous host-dense numpy hits the f_order=True branch of the host-
# streaming launcher: float32 -> the reinterpret-cast fast path (no cast kernel),
# float64 -> dense_block_to_f32_kernel's identity branch. Every numpy_dense
# fixture elsewhere is C-order, so this is the only coverage of that branch.
# AnnData preserves F-order, so an F-contiguous X reaches the path; result must
# match the C-order run on identical data.
@pytest.mark.parametrize("dtype", [np.float32, np.float64])
def test_wilcoxon_ovr_fortran_order_host_dense_matches_c_order(dtype):
    rng = np.random.default_rng(11)
    X = np.abs(rng.standard_normal((300, 40))).astype(dtype)
    X[X < 0.3] = 0.0
    labels = rng.integers(0, 5, 300)
    obs = pd.DataFrame({"group": pd.Categorical([f"g{c}" for c in labels])})
    var = pd.DataFrame(index=[f"g{j}" for j in range(40)])

    def run(arr):
        adata = sc.AnnData(X=arr, obs=obs.copy(), var=var.copy())
        adata.uns["log1p"] = {"base": None}
        rsc.tl.rank_genes_groups(
            adata,
            "group",
            method="wilcoxon",
            use_raw=False,
            reference="rest",
            tie_correct=True,
        )
        return adata.uns["rank_genes_groups"]

    xf = np.asfortranarray(X)
    assert xf.flags.f_contiguous
    r_f = run(xf)
    r_c = run(np.ascontiguousarray(X))
    for field in ("scores", "pvals", "logfoldchanges"):
        for grp in r_f[field].dtype.names:
            np.testing.assert_array_equal(
                np.asarray(r_f[field][grp], dtype=float),
                np.asarray(r_c[field][grp], dtype=float),
            )


# Regression guard for a shared-memory OOB write in the host sparse OVR
# cast-and-accumulate kernel: it placed the per-group nnz accumulator at a fixed
# 2*n_groups smem offset, but cast_accumulate_smem_config packs only the enabled
# arrays -- and the host OVR path runs with sq-sums OFF, nnz ON (pts=True). The
# overrun was benign at tiny n_groups (it landed in rounded smem slack, and the
# write/read used the same wrong offset so values stayed self-consistent) but
# caused an illegal memory access once n_groups grew past ~25. n_groups=50 +
# pts=True is the faulting regime, with the smem (non-gmem) accumulator still
# selected. Covers both host sparse formats (the ones that crashed) plus the
# dense/device formats for full parity.
@pytest.mark.parametrize(
    "fmt", ["numpy_dense", "scipy_csr", "scipy_csc", "cupy_csr", "cupy_csc"]
)
def test_wilcoxon_ovr_pts_many_groups_match_scanpy(fmt):
    adata_gpu = _make_sized_groups_adata([40] * 50, n_genes=8, seed=4)
    adata_cpu = adata_gpu.copy()
    adata_gpu.X = _to_format(adata_gpu.X, fmt)

    kw = {
        "groupby": "group",
        "method": "wilcoxon",
        "use_raw": False,
        "reference": "rest",
        "tie_correct": True,
        "pts": True,
        "n_genes": 8,
    }
    rsc.tl.rank_genes_groups(adata_gpu, **kw)
    sc.tl.rank_genes_groups(adata_cpu, **kw)

    gpu = adata_gpu.uns["rank_genes_groups"]
    cpu = adata_cpu.uns["rank_genes_groups"]
    for field in ("scores", "pvals"):
        for group in gpu[field].dtype.names:
            np.testing.assert_allclose(
                np.asarray(gpu[field][group], dtype=float),
                np.asarray(cpu[field][group], dtype=float),
                rtol=1e-13,
                atol=1e-15,
                equal_nan=True,
            )
    gpu_pts, cpu_pts = gpu["pts"], cpu["pts"]
    assert list(gpu_pts.columns) == list(cpu_pts.columns)
    for col in gpu_pts.columns:
        np.testing.assert_allclose(
            gpu_pts[col].values, cpu_pts[col].values, rtol=1e-13, atol=1e-15
        )


# Companion to test_wilcoxon_ovr_many_groups_gmem_formats_agree, with pts=True:
# at gmem scale (n_groups > ~3056) the global cast-accumulate and the
# analytic-zero rank kernel both drive the per-group nnz path. scanpy's
# 3000+-group build is too slow for an in-suite parity check, so we assert every
# storage format agrees, including the pts fraction-expressing matrix.
def test_wilcoxon_ovr_many_groups_gmem_pts_formats_agree():
    adata = _make_sized_groups_adata([26] * 3100, n_genes=6, seed=5)
    ref = None
    for fmt in ("numpy_dense", "scipy_csr", "scipy_csc", "cupy_csr", "cupy_csc"):
        a = adata.copy()
        a.X = _to_format(adata.X, fmt)
        rsc.tl.rank_genes_groups(
            a,
            "group",
            method="wilcoxon",
            use_raw=False,
            reference="rest",
            tie_correct=True,
            pts=True,
            n_genes=6,
        )
        r = a.uns["rank_genes_groups"]
        cur = {
            field: np.vstack(
                [np.asarray(r[field][n], dtype=float) for n in r[field].dtype.names]
            )
            for field in ("scores", "pvals")
        }
        cur["pts"] = r["pts"].values
        if ref is None:
            ref = cur
            continue
        for field in ("scores", "pvals", "pts"):
            np.testing.assert_allclose(
                cur[field], ref[field], rtol=1e-13, atol=1e-15, equal_nan=True
            )


@pytest.mark.parametrize(
    ("groups", "reference"),
    [
        (["0"], "rest"),
        (["0", "2"], "rest"),
        (["0"], "1"),
        (["0", "2"], "1"),
    ],
)
@pytest.mark.parametrize("tie_correct", [False, True])
def test_rank_genes_groups_wilcoxon_subset_matches_scanpy(
    groups, reference, tie_correct
):
    np.random.seed(42)
    adata_gpu = sc.datasets.blobs(n_variables=8, n_centers=5, n_observations=200)
    adata_gpu.obs["blobs"] = adata_gpu.obs["blobs"].astype("category")
    adata_cpu = adata_gpu.copy()

    rsc.tl.rank_genes_groups(
        adata_gpu,
        "blobs",
        method="wilcoxon",
        groups=groups,
        reference=reference,
        use_raw=False,
        tie_correct=tie_correct,
    )
    sc.tl.rank_genes_groups(
        adata_cpu,
        "blobs",
        method="wilcoxon",
        groups=groups,
        reference=reference,
        use_raw=False,
        tie_correct=tie_correct,
    )

    gpu_result = adata_gpu.uns["rank_genes_groups"]
    cpu_result = adata_cpu.uns["rank_genes_groups"]

    assert gpu_result["names"].dtype.names == cpu_result["names"].dtype.names
    for group in gpu_result["names"].dtype.names:
        gpu_names = list(gpu_result["names"][group])
        cpu_names = list(cpu_result["names"][group])
        for field in ("scores", "logfoldchanges", "pvals", "pvals_adj"):
            gpu_map = dict(
                zip(gpu_names, np.asarray(gpu_result[field][group], dtype=float))
            )
            cpu_map = dict(
                zip(cpu_names, np.asarray(cpu_result[field][group], dtype=float))
            )
            for gene, gpu_val in gpu_map.items():
                np.testing.assert_allclose(
                    gpu_val,
                    cpu_map[gene],
                    rtol=1e-6,
                    atol=1e-8,
                    err_msg=f"{field} mismatch for gene {gene} group {group}",
                )


@pytest.mark.parametrize(
    "reference_before,reference_after",
    [("rest", "rest"), ("1", "One")],
)
def test_rank_genes_groups_wilcoxon_with_renamed_categories(
    reference_before, reference_after
):
    """Test with renamed category labels."""
    np.random.seed(42)
    adata = sc.datasets.blobs(n_variables=4, n_centers=3, n_observations=200)
    _make_nonnegative(adata)
    adata.obs["blobs"] = adata.obs["blobs"].astype("category")

    # First run with original category names
    rsc.tl.rank_genes_groups(
        adata, "blobs", method="wilcoxon", reference=reference_before
    )
    names = adata.uns["rank_genes_groups"]["names"]
    expected_groups = ("0", "1", "2") if reference_before == "rest" else ("0", "2")
    assert names.dtype.names == expected_groups
    first_run = tuple(names[0])

    adata.rename_categories("blobs", ["Zero", "One", "Two"])
    assert tuple(adata.uns["rank_genes_groups"]["names"][0]) == first_run

    # Second run with renamed category names
    rsc.tl.rank_genes_groups(
        adata, "blobs", method="wilcoxon", reference=reference_after
    )
    renamed_names = adata.uns["rank_genes_groups"]["names"]
    assert tuple(renamed_names[0]) == first_run
    expected_renamed = (
        ("Zero", "One", "Two") if reference_after == "rest" else ("Zero", "Two")
    )
    assert renamed_names.dtype.names == expected_renamed


@pytest.mark.parametrize("reference", ["rest", "1"])
def test_rank_genes_groups_wilcoxon_with_unsorted_groups(reference):
    """Group order sets the output column order (matching scanpy); the per-group
    statistics themselves are order-independent."""
    np.random.seed(42)
    adata = sc.datasets.blobs(n_variables=6, n_centers=4, n_observations=180)
    _make_nonnegative(adata)
    adata.obs["blobs"] = adata.obs["blobs"].astype("category")
    bdata = adata.copy()

    groups = ["0", "1", "2", "3"] if reference != "rest" else ["0", "2", "3"]
    groups_reversed = list(reversed(groups))

    rsc.tl.rank_genes_groups(
        adata, "blobs", method="wilcoxon", groups=groups, reference=reference
    )
    rsc.tl.rank_genes_groups(
        bdata, "blobs", method="wilcoxon", groups=groups_reversed, reference=reference
    )

    # Column order echoes the user-provided group order (reference excluded).
    assert adata.uns["rank_genes_groups"]["names"].dtype.names == tuple(
        g for g in groups if g != reference
    )
    assert bdata.uns["rank_genes_groups"]["names"].dtype.names == tuple(
        g for g in groups_reversed if g != reference
    )

    # Pick a group that's not the reference for comparison
    test_group = "3" if reference != "3" else "0"
    for field in ("scores", "logfoldchanges", "pvals", "pvals_adj"):
        np.testing.assert_allclose(
            np.asarray(adata.uns["rank_genes_groups"][field][test_group], dtype=float),
            np.asarray(bdata.uns["rank_genes_groups"][field][test_group], dtype=float),
            rtol=1e-13,
            atol=1e-15,
            equal_nan=True,
        )

    assert tuple(adata.uns["rank_genes_groups"]["names"][test_group]) == tuple(
        bdata.uns["rank_genes_groups"]["names"][test_group]
    )


@pytest.mark.parametrize("reference", ["rest", "1"])
def test_rank_genes_groups_wilcoxon_pts(reference):
    """Test that pts (fraction of cells expressing) is computed correctly."""
    np.random.seed(42)
    adata_gpu = sc.datasets.blobs(n_variables=6, n_centers=3, n_observations=200)
    _make_nonnegative(adata_gpu)
    adata_gpu.obs["blobs"] = adata_gpu.obs["blobs"].astype("category")
    adata_cpu = adata_gpu.copy()

    # Run with pts=True
    rsc.tl.rank_genes_groups(
        adata_gpu,
        "blobs",
        method="wilcoxon",
        use_raw=False,
        pts=True,
        tie_correct=False,
        reference=reference,
    )
    sc.tl.rank_genes_groups(
        adata_cpu,
        "blobs",
        method="wilcoxon",
        use_raw=False,
        pts=True,
        tie_correct=False,
        reference=reference,
    )

    gpu_result = adata_gpu.uns["rank_genes_groups"]
    cpu_result = adata_cpu.uns["rank_genes_groups"]

    # Check pts DataFrame exists and has correct structure
    assert "pts" in gpu_result
    assert "pts" in cpu_result

    # Check pts values match scanpy
    gpu_pts = gpu_result["pts"]
    cpu_pts = cpu_result["pts"]
    assert list(gpu_pts.columns) == list(cpu_pts.columns)
    assert list(gpu_pts.index) == list(cpu_pts.index)

    for col in gpu_pts.columns:
        np.testing.assert_allclose(
            gpu_pts[col].values, cpu_pts[col].values, rtol=1e-13, atol=1e-15
        )

    # pts_rest only exists when reference='rest'
    if reference == "rest":
        assert "pts_rest" in gpu_result
        assert "pts_rest" in cpu_result

        gpu_pts_rest = gpu_result["pts_rest"]
        cpu_pts_rest = cpu_result["pts_rest"]

        for col in gpu_pts_rest.columns:
            np.testing.assert_allclose(
                gpu_pts_rest[col].values,
                cpu_pts_rest[col].values,
                rtol=1e-13,
                atol=1e-15,
            )


# ============================================================================
# Ground-truth validation against scipy.stats.mannwhitneyu
# ============================================================================


def _make_perturbation_adata(
    n_control: int = 200,
    n_treatment: int = 150,
    n_genes: int = 500,
    n_de_genes: int = 50,
    seed: int = 42,
):
    """Two-group perturbation AnnData with count-based log1p data (many ties)."""
    rng = np.random.default_rng(seed)
    n_cells = n_control + n_treatment

    gene_means = rng.gamma(shape=2.0, scale=5.0, size=n_genes)
    X = rng.poisson(lam=gene_means[None, :], size=(n_cells, n_genes)).astype(np.float32)
    for g in range(n_de_genes):
        X[n_control:, g] = rng.poisson(
            lam=gene_means[g] * 1.5, size=n_treatment
        ).astype(np.float32)

    obs = pd.DataFrame(
        {
            "group": pd.Categorical(
                ["control"] * n_control + ["treatment"] * n_treatment,
                categories=["control", "treatment"],
            ),
        },
        index=[f"cell_{i}" for i in range(n_cells)],
    )
    var = pd.DataFrame(index=[f"gene_{i}" for i in range(n_genes)])
    adata = sc.AnnData(X=X, obs=obs, var=var)
    sc.pp.normalize_total(adata, target_sum=1e4)
    sc.pp.log1p(adata)
    return adata


def _scipy_mannwhitneyu_pvals(adata, *, group, reference, groupby="group"):
    """Per-gene two-sided Mann-Whitney U p-values via scipy (ground truth)."""
    X = adata.X.toarray() if sp.issparse(adata.X) else np.array(adata.X)
    X = X.astype(np.float64)
    mask_g = (adata.obs[groupby] == group).values
    mask_r = (adata.obs[groupby] == reference).values
    return np.array(
        [
            mannwhitneyu(X[mask_g, i], X[mask_r, i], alternative="two-sided").pvalue
            for i in range(X.shape[1])
        ]
    )


@pytest.fixture
def perturbation_adata():
    return _make_perturbation_adata()


class TestWilcoxonAgainstScipy:
    """Validate rsc wilcoxon p-values against scipy.stats.mannwhitneyu."""

    def test_with_continuity_matches_scipy(self, perturbation_adata):
        """use_continuity + tie_correct matches scipy mannwhitneyu to machine eps."""
        adata = perturbation_adata.copy()
        rsc.tl.rank_genes_groups(
            adata,
            "group",
            groups=["treatment"],
            reference="control",
            method="wilcoxon",
            use_raw=False,
            tie_correct=True,
            use_continuity=True,
        )

        rsc_df = (
            sc.get.rank_genes_groups_df(adata, group="treatment")
            .sort_values("names")
            .reset_index(drop=True)
        )
        scipy_pvals = _scipy_mannwhitneyu_pvals(
            perturbation_adata, group="treatment", reference="control"
        )
        # Align scipy pvals to the same gene order
        gene_to_idx = {g: i for i, g in enumerate(perturbation_adata.var_names)}
        scipy_sorted = np.array([scipy_pvals[gene_to_idx[g]] for g in rsc_df["names"]])

        np.testing.assert_allclose(
            rsc_df["pvals"].values, scipy_sorted, rtol=1e-13, atol=1e-15
        )

    def test_without_continuity_close_to_scipy(self, perturbation_adata):
        """Without continuity correction the gap is only the 0.5 adjustment term."""
        adata = perturbation_adata.copy()
        rsc.tl.rank_genes_groups(
            adata,
            "group",
            groups=["treatment"],
            reference="control",
            method="wilcoxon",
            use_raw=False,
            tie_correct=True,
            use_continuity=False,
        )

        rsc_df = (
            sc.get.rank_genes_groups_df(adata, group="treatment")
            .sort_values("names")
            .reset_index(drop=True)
        )
        scipy_pvals = _scipy_mannwhitneyu_pvals(
            perturbation_adata, group="treatment", reference="control"
        )
        gene_to_idx = {g: i for i, g in enumerate(perturbation_adata.var_names)}
        scipy_sorted = np.array([scipy_pvals[gene_to_idx[g]] for g in rsc_df["names"]])

        np.testing.assert_allclose(
            rsc_df["pvals"].values, scipy_sorted, rtol=1e-2, atol=1e-15
        )

    @pytest.mark.parametrize("sparse", [True, False])
    def test_sparse_matches_dense(self, perturbation_adata, sparse):
        """Sparse and dense wilcoxon give identical results."""
        adata_dense = perturbation_adata.copy()
        adata_sparse = perturbation_adata.copy()
        adata_sparse.X = sp.csr_matrix(adata_sparse.X)

        kw = {
            "groupby": "group",
            "groups": ["treatment"],
            "reference": "control",
            "method": "wilcoxon",
            "use_raw": False,
            "tie_correct": True,
        }
        rsc.tl.rank_genes_groups(adata_dense, **kw)
        rsc.tl.rank_genes_groups(adata_sparse, **kw)

        dense_df = (
            sc.get.rank_genes_groups_df(adata_dense, group="treatment")
            .sort_values("names")
            .reset_index(drop=True)
        )
        sparse_df = (
            sc.get.rank_genes_groups_df(adata_sparse, group="treatment")
            .sort_values("names")
            .reset_index(drop=True)
        )
        np.testing.assert_array_equal(
            dense_df["scores"].values, sparse_df["scores"].values
        )
        np.testing.assert_array_equal(
            dense_df["pvals"].values, sparse_df["pvals"].values
        )


def _make_count_adata(seed=0, n_obs=120, n_genes=6, n_groups=3):
    # Integer-valued counts as float64: float32-exact, zeros create ties.
    rng = np.random.default_rng(seed)
    X = rng.integers(0, 8, size=(n_obs, n_genes)).astype(np.float64)
    X[X < 2] = 0.0  # extra zeros -> implicit-zero tie blocks
    labels = np.array([f"{i % n_groups}" for i in range(n_obs)])
    obs = pd.DataFrame({"group": pd.Categorical(labels)})
    var = pd.DataFrame(index=[f"g{j}" for j in range(n_genes)])
    adata = sc.AnnData(X=X, obs=obs, var=var)
    adata.uns["log1p"] = {"base": None}
    return adata


@pytest.mark.parametrize("fmt", ["scipy_csr", "scipy_csc"])
@pytest.mark.parametrize("reference", ["rest", "1"])
def test_wilcoxon_host_sparse_float64_data_matches_scanpy(fmt, reference):
    # float64 host-sparse data exercises the *_f64 kernel bindings.
    adata = _make_count_adata(seed=3)
    adata_cpu = adata.copy()
    mat = sp.csr_matrix(adata.X) if fmt == "scipy_csr" else sp.csc_matrix(adata.X)
    assert mat.dtype == np.float64
    adata.X = mat

    kw = {
        "groupby": "group",
        "method": "wilcoxon",
        "use_raw": False,
        "reference": reference,
        "tie_correct": True,
        "n_genes": adata.n_vars,
    }
    rsc.tl.rank_genes_groups(adata, **kw)
    sc.tl.rank_genes_groups(adata_cpu, **kw)
    g = adata.uns["rank_genes_groups"]
    c = adata_cpu.uns["rank_genes_groups"]
    for field in ("scores", "pvals", "pvals_adj"):
        for grp in g[field].dtype.names:
            np.testing.assert_allclose(
                np.asarray(g[field][grp], dtype=float),
                np.asarray(c[field][grp], dtype=float),
                rtol=1e-13,
                atol=1e-15,
                equal_nan=True,
            )


@pytest.mark.parametrize("fmt", ["scipy_csr", "scipy_csc"])
@pytest.mark.parametrize("data_dtype", [np.int32, np.int64, np.uint16, bool])
def test_wilcoxon_sparse_integer_bool_data_matches_float32(fmt, data_dtype):
    # Integer/bool data hits the cast-to-float32 branch; must match float32.
    rng = np.random.default_rng(5)
    n_obs, n_genes = 100, 6
    counts = rng.integers(0, 5, size=(n_obs, n_genes))
    if data_dtype is bool:
        counts = counts > 2
    typed = counts.astype(data_dtype)
    f32 = counts.astype(np.float32)
    labels = np.array([f"{i % 3}" for i in range(n_obs)])
    obs = pd.DataFrame({"group": pd.Categorical(labels)})
    var = pd.DataFrame(index=[f"g{j}" for j in range(n_genes)])

    def run(arr):
        adata = sc.AnnData(X=_to_format(arr, fmt), obs=obs.copy(), var=var.copy())
        adata.uns["log1p"] = {"base": None}
        rsc.tl.rank_genes_groups(
            adata,
            "group",
            method="wilcoxon",
            use_raw=False,
            reference="rest",
            tie_correct=True,
            n_genes=n_genes,
        )
        return adata.uns["rank_genes_groups"]

    r_typed = run(typed)
    r_f32 = run(f32)
    for grp in r_typed["scores"].dtype.names:
        np.testing.assert_array_equal(
            np.asarray(r_typed["scores"][grp], dtype=float),
            np.asarray(r_f32["scores"][grp], dtype=float),
        )


@pytest.mark.parametrize("fmt", ["scipy_csr", "scipy_csc"])
def test_wilcoxon_sparse_float16_data_raises(fmt):
    # Unsupported float16 sparse data is rejected with a TypeError.
    rng = np.random.default_rng(0)
    dense = np.abs(rng.standard_normal((40, 4))).astype(np.float32)
    mat = sp.csr_matrix(dense) if fmt == "scipy_csr" else sp.csc_matrix(dense)
    mat.data = mat.data.astype(np.float16)
    assert mat.data.dtype == np.float16
    adata = sc.AnnData(
        X=mat,
        obs=pd.DataFrame({"group": pd.Categorical([f"{i % 2}" for i in range(40)])}),
        var=pd.DataFrame(index=[f"g{j}" for j in range(4)]),
    )
    with pytest.raises(TypeError, match="float32"):
        rsc.tl.rank_genes_groups(adata, "group", method="wilcoxon", use_raw=False)


@pytest.mark.parametrize("reference", ["rest", "b"])
@pytest.mark.parametrize("fmt", ["scipy_csc", "cupy_csc", "cupy_dense"])
def test_rank_genes_groups_wilcoxon_return_u_values_more_formats(reference, fmt):
    # U-value + continuity epilogue on CSC (host/device) and device-dense.
    X = np.array(
        [
            [5.0, 0.0, 1.0, 2.0],
            [4.0, 0.0, 1.0, 2.0],
            [1.0, 3.0, 2.0, 2.0],
            [0.0, 2.0, 2.0, 2.0],
            [2.0, 1.0, 0.0, 3.0],
            [3.0, 1.0, 0.0, 3.0],
        ],
        dtype=np.float32,
    )
    labels = np.array(["a", "a", "b", "b", "c", "c"])
    adata = sc.AnnData(
        X=_to_format(X, fmt),
        obs=pd.DataFrame({"group": pd.Categorical(labels)}),
        var=pd.DataFrame(index=[f"g{i}" for i in range(X.shape[1])]),
    )

    rsc.tl.rank_genes_groups(
        adata,
        "group",
        groups=["a"],
        reference=reference,
        method="wilcoxon",
        use_raw=False,
        tie_correct=True,
        use_continuity=True,
        return_u_values=True,
        n_genes=adata.n_vars,
    )

    result = adata.uns["rank_genes_groups"]
    assert result["params"]["return_u_values"] is True
    assert result["scores"].dtype["a"] == np.dtype("float64")

    df = sc.get.rank_genes_groups_df(adata, group="a").sort_values("names")
    mask_group = labels == "a"
    mask_ref = labels != "a" if reference == "rest" else labels == reference
    expected = np.array(
        [
            mannwhitneyu(
                X[mask_group, gene],
                X[mask_ref, gene],
                alternative="two-sided",
            ).statistic
            for gene in range(X.shape[1])
        ],
        dtype=np.float64,
    )
    gene_to_idx = {name: idx for idx, name in enumerate(adata.var_names)}
    expected_sorted = np.array([expected[gene_to_idx[name]] for name in df["names"]])
    np.testing.assert_allclose(df["scores"].to_numpy(), expected_sorted)


@pytest.mark.parametrize("fmt", ["scipy_csr", "scipy_csc", "cupy_csr", "cupy_csc"])
def test_rank_genes_groups_sparse_negative_values_fallback_ovo(fmt):
    # Sparse-negative dense fallback on the OVO (with-reference) path.
    X = np.array(
        [
            [-1.0, 0.0, 2.0],
            [0.0, 1.0, 0.0],
            [2.0, 0.0, 1.0],
            [0.0, 3.0, 0.0],
            [-2.0, 1.0, 0.0],
            [1.0, 0.0, 3.0],
        ],
        dtype=np.float64,
    )
    obs = pd.DataFrame({"group": pd.Categorical(list("aaabbb"), categories=["a", "b"])})
    var = pd.DataFrame(index=["g0", "g1", "g2"])

    sparse_adata = sc.AnnData(X=_to_format(X, fmt), obs=obs.copy(), var=var.copy())
    dense_fmt = "cupy_dense" if fmt.startswith("cupy") else "numpy_dense"
    dense_adata = sc.AnnData(X=_to_format(X, dense_fmt), obs=obs.copy(), var=var.copy())

    kw = {"groupby": "group", "method": "wilcoxon", "use_raw": False, "reference": "b"}
    rsc.tl.rank_genes_groups(sparse_adata, **kw)
    rsc.tl.rank_genes_groups(dense_adata, **kw)

    sp_scores = sparse_adata.uns["rank_genes_groups"]["scores"]
    dn_scores = dense_adata.uns["rank_genes_groups"]["scores"]
    for group in sp_scores.dtype.names:
        np.testing.assert_allclose(
            np.asarray(sp_scores[group], dtype=float),
            np.asarray(dn_scores[group], dtype=float),
            rtol=1e-13,
            atol=1e-13,
        )


@pytest.mark.parametrize("reference", ["rest", "2"])
def test_wilcoxon_group_subset_column_order_matches_scanpy(reference):
    """Output column order must echo the user's ``groups=`` list (scanpy parity),
    not be re-sorted to category order."""
    np.random.seed(0)
    adata = sc.datasets.blobs(n_variables=6, n_centers=4, n_observations=180)
    adata.obs["blobs"] = adata.obs["blobs"].astype("category")
    _make_nonnegative(adata)
    bdata = adata.copy()

    # Deliberately out-of-category-order subset.
    groups = ["3", "1"] if reference != "rest" else ["3", "1", "0"]
    rsc.tl.rank_genes_groups(
        adata,
        "blobs",
        method="wilcoxon",
        use_raw=False,
        groups=groups,
        reference=reference,
    )
    sc.tl.rank_genes_groups(
        bdata,
        "blobs",
        method="wilcoxon",
        use_raw=False,
        groups=groups,
        reference=reference,
    )
    assert (
        adata.uns["rank_genes_groups"]["names"].dtype.names
        == bdata.uns["rank_genes_groups"]["names"].dtype.names
    )


def test_wilcoxon_host_sparse_negative_chunked_stats_match_scanpy():
    """Host scipy-sparse with negatives takes the dense fallback, whose group
    means/vars/pts run through the group_chunk_stats kernel (multi-chunk). Those
    means (-> logfoldchanges) and pts must match scanpy.
    """
    rng = np.random.default_rng(0)
    n_obs, n_vars = 200, 24
    X = (rng.random((n_obs, n_vars)) * 5.0).astype(np.float64)
    X[X < 1.5] = 0.0  # structural zeros so pts < 1
    X[rng.random((n_obs, n_vars)) < 0.01] = -0.5  # a few negatives -> fallback
    obs = pd.DataFrame({"group": pd.Categorical([f"{i % 3}" for i in range(n_obs)])})
    var = pd.DataFrame(index=[f"g{i}" for i in range(n_vars)])

    gpu = sc.AnnData(X=sp.csr_matrix(X), obs=obs.copy(), var=var.copy())
    cpu = sc.AnnData(X=X.copy(), obs=obs.copy(), var=var.copy())

    rsc.tl.rank_genes_groups(
        gpu,
        "group",
        method="wilcoxon",
        use_raw=False,
        reference="rest",
        pts=True,
        n_genes=n_vars,
        chunk_size=8,  # < n_vars -> multiple chunks
    )
    sc.tl.rank_genes_groups(
        cpu, "group", method="wilcoxon", use_raw=False, reference="rest", pts=True
    )
    g = gpu.uns["rank_genes_groups"]
    c = cpu.uns["rank_genes_groups"]
    for group in g["names"].dtype.names:
        g_lfc = dict(
            zip(g["names"][group], np.asarray(g["logfoldchanges"][group], float))
        )
        c_lfc = dict(
            zip(c["names"][group], np.asarray(c["logfoldchanges"][group], float))
        )
        for gene, val in g_lfc.items():
            np.testing.assert_allclose(
                val, c_lfc[gene], rtol=1e-12, atol=1e-13, equal_nan=True
            )
    for frame in ("pts", "pts_rest"):
        for col in c[frame].columns:
            np.testing.assert_allclose(
                g[frame].loc[c[frame].index, col].values,
                c[frame][col].values,
                rtol=1e-12,
                atol=1e-13,
            )


def test_wilcoxon_fdr_ties_nan_match_scanpy():
    """BH FDR must match scanpy on heavily-tied / constant / all-zero genes,
    locking in that the GPU argsort tie-break is inert for adjusted p-values.
    Integer data is float32-exact, so ranking is bit-identical to scanpy and the
    comparison isolates the FDR step.
    """
    rng = np.random.default_rng(1)
    n_obs, n_vars = 240, 30
    X = rng.integers(0, 3, size=(n_obs, n_vars)).astype(np.float64)  # heavy ties
    X[:, 0] = 1.0  # constant gene -> identical p across groups
    X[:, 1] = 0.0  # all-zero gene
    obs = pd.DataFrame({"group": pd.Categorical([f"{i % 3}" for i in range(n_obs)])})
    var = pd.DataFrame(index=[f"g{i}" for i in range(n_vars)])

    gpu = sc.AnnData(X=cp.asarray(X), obs=obs.copy(), var=var.copy())  # GPU FDR path
    cpu = sc.AnnData(X=X.copy(), obs=obs.copy(), var=var.copy())

    rsc.tl.rank_genes_groups(
        gpu, "group", method="wilcoxon", use_raw=False, tie_correct=True
    )
    sc.tl.rank_genes_groups(
        cpu, "group", method="wilcoxon", use_raw=False, tie_correct=True
    )
    g = gpu.uns["rank_genes_groups"]
    c = cpu.uns["rank_genes_groups"]
    for group in g["names"].dtype.names:
        g_adj = dict(zip(g["names"][group], np.asarray(g["pvals_adj"][group], float)))
        c_adj = dict(zip(c["names"][group], np.asarray(c["pvals_adj"][group], float)))
        for gene, val in g_adj.items():
            np.testing.assert_allclose(
                val, c_adj[gene], rtol=1e-12, atol=1e-13, equal_nan=True
            )


def _promote_host_index_dtypes(X, *, indptr64, indices64):
    """Copy a host scipy CSR/CSC matrix with promoted index-array dtypes.

    scipy couples indptr/indices to one dtype via get_index_dtype, so the
    decoupled (i64 indptr / i32 indices) combination only arises by explicit
    promotion -- which is exactly what drives the templated host kernels.
    """
    X = X.copy()
    if indptr64:
        X.indptr = X.indptr.astype(np.int64)
    if indices64:
        X.indices = X.indices.astype(np.int64)
    return X


@pytest.mark.parametrize("reference", ["rest", "1"])  # OVR vs OVO host paths
@pytest.mark.parametrize(
    ("layout", "data_dtype", "indices64"),
    [
        ("csr", np.float32, False),  # *_i64
        ("csr", np.float32, True),  # *_i64_idx64
        ("csr", np.float64, False),  # *_f64_i64
        ("csr", np.float64, True),  # *_f64_i64_idx64
        ("csc", np.float32, False),  # *_i64        (CSC has no idx64 template)
        ("csc", np.float64, False),  # *_f64_i64
    ],
)
def test_host_sparse_int64_templates_match_int32(
    reference, layout, data_dtype, indices64
):
    """Exercise the host-sparse int64-indptr / int64-indices kernel templates
    (the 12 ``*_i64`` / ``*_idx64`` / ``*_f64_i64`` host bindings the suite
    otherwise never reaches). These differ from the validated int32 host path
    only in index dtype, so they must be bit-identical to it. Real int64 indices
    only occur at nnz > 2^31 (unallocatable in CI), so we promote a small
    matrix's index arrays explicitly and keep it host-resident (scipy sparse +
    method='wilcoxon' is not moved to GPU)."""
    rng = np.random.default_rng(0)
    dense = (rng.random((150, 8)) * 4.0).astype(np.float64)
    dense[dense < 1.5] = 0.0  # nonnegative + structural zeros -> sparse fast path
    obs = pd.DataFrame({"group": pd.Categorical([f"{i % 3}" for i in range(150)])})
    var = pd.DataFrame(index=[f"g{i}" for i in range(8)])

    maker = sp.csr_matrix if layout == "csr" else sp.csc_matrix
    base = maker(dense.astype(data_dtype))

    a32 = sc.AnnData(X=base.copy(), obs=obs.copy(), var=var.copy())
    a64 = sc.AnnData(
        X=_promote_host_index_dtypes(base, indptr64=True, indices64=indices64),
        obs=obs.copy(),
        var=var.copy(),
    )
    kw = {
        "method": "wilcoxon",
        "use_raw": False,
        "reference": reference,
        "tie_correct": True,
    }
    rsc.tl.rank_genes_groups(a32, "group", **kw)
    rsc.tl.rank_genes_groups(a64, "group", **kw)

    r32, r64 = a32.uns["rank_genes_groups"], a64.uns["rank_genes_groups"]
    assert r64["names"].dtype.names == r32["names"].dtype.names
    for fld in ("scores", "pvals", "pvals_adj", "logfoldchanges"):
        for grp in r32[fld].dtype.names:
            np.testing.assert_array_equal(
                np.asarray(r64[fld][grp]), np.asarray(r32[fld][grp])
            )


def _anndata_with_group_sizes(sizes, *, n_genes=6, seed=0):
    """Dense AnnData whose per-group cell counts are exactly ``sizes``.

    The OVO tier dispatch picks the rank kernel by *test-group* size
    (WARP<=32, SMALL 33-64, MEDIUM 65-512, LARGE 513-2500, HUGE>2500), so
    engineered group sizes drive specific bands. Integer data is float32-exact,
    so ranking is bit-identical to scanpy.
    """
    rng = np.random.default_rng(seed)
    labels = []
    for name, n in sizes.items():
        labels += [name] * n
    X = rng.integers(0, 6, size=(len(labels), n_genes)).astype(np.float64)
    obs = pd.DataFrame({"group": pd.Categorical(labels)})
    var = pd.DataFrame(index=[f"g{i}" for i in range(n_genes)])
    return sc.AnnData(X=X, obs=obs, var=var)


def _assert_ovo_matches_scanpy(adata, reference):
    bdata = adata.copy()
    kw = {
        "method": "wilcoxon",
        "use_raw": False,
        "reference": reference,
        "tie_correct": True,
    }
    rsc.tl.rank_genes_groups(adata, "group", **kw)
    sc.tl.rank_genes_groups(bdata, "group", **kw)
    g, c = adata.uns["rank_genes_groups"], bdata.uns["rank_genes_groups"]
    for fld in ("scores", "pvals", "pvals_adj"):
        for grp in g[fld].dtype.names:
            gm = dict(zip(g["names"][grp], np.asarray(g[fld][grp], float)))
            cm = dict(zip(c["names"][grp], np.asarray(c[fld][grp], float)))
            for gene, val in gm.items():
                np.testing.assert_allclose(
                    val, cm[gene], rtol=1e-12, atol=1e-13, equal_nan=True
                )


def test_ovo_tier_bands_warp_small_medium_large_match_scanpy():
    """OVO dense-tiered path must hit the WARP/SMALL/MEDIUM/LARGE rank kernels
    (test-group sizes 20/50/300/1000, all <= 2500) and match scanpy."""
    adata = _anndata_with_group_sizes(
        {"ref": 40, "warp": 20, "small": 50, "medium": 300, "large": 1000}, seed=1
    )
    _assert_ovo_matches_scanpy(adata, reference="ref")


def test_ovo_tier_band_huge_match_scanpy():
    """OVO dense-tiered path must hit the HUGE band (CUB segmented sort, a
    test-group > 2500 cells) and match scanpy."""
    adata = _anndata_with_group_sizes({"ref": 40, "huge": 3000}, seed=2)
    _assert_ovo_matches_scanpy(adata, reference="ref")


@pytest.mark.filterwarnings("ignore::RuntimeWarning")  # 6200 tiny groups warn
def test_ovr_dense_gmem_branch_matches_scipy():
    """The DENSE OVR global-memory accumulator (use_gmem) engages only when the
    per-block group accumulators exceed the 48 KB MaxSharedMemoryPerBlock limit
    -- n_groups > 6112 (= 49152 / 8 - 32). No other test reaches it (the
    >3056-group gmem tests only flip the *sparse* accumulator). n_groups=6200
    deterministically routes through dense gmem; a scanpy oracle here costs ~30s
    (its per-group Python loop), so we validate a sample of groups against scipy
    mannwhitneyu with rsc's exact settings (tie-corrected asymptotic, no
    continuity)."""
    from scipy.stats import mannwhitneyu

    n_groups, n_genes = 6200, 4  # > 6112 -> dense gmem accumulator
    rng = np.random.default_rng(3)
    labels = np.repeat(np.arange(n_groups), 2)  # 2 cells per group
    X = rng.integers(0, 6, size=(labels.size, n_genes)).astype(np.float64)
    obs = pd.DataFrame({"group": pd.Categorical([str(x) for x in labels])})
    var = pd.DataFrame(index=[f"g{i}" for i in range(n_genes)])
    adata = sc.AnnData(X=X, obs=obs, var=var)

    rsc.tl.rank_genes_groups(
        adata, "group", method="wilcoxon", use_raw=False, tie_correct=True
    )
    res = adata.uns["rank_genes_groups"]
    for grp in ("0", "1", "250", "1000", "3057", "6112", "6199"):
        gp = dict(zip(res["names"][grp], np.asarray(res["pvals"][grp], float)))
        mask = labels == int(grp)
        for gi, gene in enumerate(var.index):
            _, p = mannwhitneyu(
                X[mask, gi],
                X[~mask, gi],
                use_continuity=False,
                alternative="two-sided",
                method="asymptotic",
            )
            np.testing.assert_allclose(
                gp[gene],
                p,
                rtol=1e-10,
                atol=1e-12,
                equal_nan=True,
                err_msg=f"group {grp} gene {gene}",
            )


def test_skip_empty_groups_vs_rest_drops_singleton():
    """skip_empty_groups=True with reference='rest' silently drops <2-cell
    groups (covers the reference=='rest' branch of _select_groups, which the
    existing reference='ref' skip tests miss)."""
    adata = _anndata_with_group_sizes({"a": 10, "b": 10, "c": 1}, seed=4)
    rsc.tl.rank_genes_groups(
        adata, "group", method="wilcoxon", use_raw=False, skip_empty_groups=True
    )
    names = set(adata.uns["rank_genes_groups"]["names"].dtype.names)
    assert names == {"a", "b"}  # singleton "c" dropped, no error


def test_skip_empty_groups_reference_too_small_raises():
    """skip_empty_groups=True with a <2-cell reference raises a clear error."""
    adata = _anndata_with_group_sizes({"a": 10, "b": 10, "c": 1}, seed=4)
    with pytest.raises(ValueError, match="reference = c has fewer than two samples"):
        rsc.tl.rank_genes_groups(
            adata,
            "group",
            method="wilcoxon",
            use_raw=False,
            reference="c",
            skip_empty_groups=True,
        )


def test_skip_empty_groups_none_remain_raises():
    """skip_empty_groups=True raises when no group has >=2 cells (vs-rest)."""
    adata = _anndata_with_group_sizes({"a": 1, "b": 1, "c": 1}, seed=4)
    with pytest.raises(ValueError, match="No groups with at least two samples remain"):
        rsc.tl.rank_genes_groups(
            adata, "group", method="wilcoxon", use_raw=False, skip_empty_groups=True
        )


@pytest.mark.parametrize(
    "fmt", ["numpy_dense", "scipy_csr", "scipy_csc", "cupy_csr", "cupy_csc"]
)
def test_ovr_tie_correct_false_tie_heavy_matches_scanpy(fmt):
    """OVR (reference='rest') tie_correct=False on TIE-HEAVY data must match
    scanpy across every storage format. The pre-existing tie_correct=False oracle
    uses tie-free blobs (tie_corr ~= 1), so a wrong uncorrected variance would
    pass; integer data with many ties stresses the omitted tie term on each path
    (dense, host-sparse CSR/CSC, device-sparse CSR/CSC)."""
    rng = np.random.default_rng(7)
    n_obs, n_genes = 180, 8
    dense = rng.integers(0, 5, size=(n_obs, n_genes)).astype(np.float64)  # ties
    dense[dense < 1.0] = 0.0  # nonnegative + structural zeros -> sparse fast path
    obs = pd.DataFrame({"group": pd.Categorical([f"{i % 3}" for i in range(n_obs)])})
    var = pd.DataFrame(index=[f"g{i}" for i in range(n_genes)])

    gpu = sc.AnnData(X=_to_format(dense, fmt), obs=obs.copy(), var=var.copy())
    cpu = sc.AnnData(X=dense.copy(), obs=obs.copy(), var=var.copy())
    kw = {
        "method": "wilcoxon",
        "use_raw": False,
        "reference": "rest",
        "tie_correct": False,
    }
    rsc.tl.rank_genes_groups(gpu, "group", **kw)
    sc.tl.rank_genes_groups(cpu, "group", **kw)
    g, c = gpu.uns["rank_genes_groups"], cpu.uns["rank_genes_groups"]
    for fld in ("scores", "pvals"):
        for grp in g[fld].dtype.names:
            gm = dict(zip(g["names"][grp], np.asarray(g[fld][grp], float)))
            cm = dict(zip(c["names"][grp], np.asarray(c[fld][grp], float)))
            for gene, val in gm.items():
                np.testing.assert_allclose(
                    val, cm[gene], rtol=1e-12, atol=1e-13, equal_nan=True
                )


@pytest.mark.parametrize(
    "fmt", ["numpy_dense", "scipy_csr", "scipy_csc", "cupy_csr", "cupy_csc"]
)
@pytest.mark.parametrize("reference", ["rest", "1"])  # OVR and OVO epilogues
def test_use_continuity_matches_scipy(fmt, reference):
    """use_continuity=True is validated only on dense-OVO elsewhere. Check the
    continuity epilogue composes correctly with each path's rank_sums (OVR + OVO,
    every format) vs scipy.mannwhitneyu(use_continuity=True, asymptotic).

    Groups OVERLAP (no separation) on purpose: that keeps |R-E[R]| moderate so
    the 0.5 continuity term MATERIALLY changes p -- a missing continuity
    correction would then fail the scipy oracle (non-vacuous). Because U and E[U]
    are multiples of 0.5, rsc's clamp (max(|d|-0.5,0)) and scipy's shift agree
    exactly. tie_correct=True matches scipy's always-on asymptotic tie term."""
    from scipy.stats import mannwhitneyu

    rng = np.random.default_rng(8)
    n_obs, n_genes = 150, 6
    # Overlapping groups (same distribution) -> moderate |R-E[R]| -> continuity
    # is material. Integer values give ties (exercises the tie term too).
    dense = rng.integers(0, 4, size=(n_obs, n_genes)).astype(np.float64)
    labels = np.array([str(i % 3) for i in range(n_obs)])
    obs = pd.DataFrame({"group": pd.Categorical(labels)})
    var = pd.DataFrame(index=[f"g{i}" for i in range(n_genes)])

    gpu = sc.AnnData(X=_to_format(dense, fmt), obs=obs.copy(), var=var.copy())
    rsc.tl.rank_genes_groups(
        gpu,
        "group",
        method="wilcoxon",
        use_raw=False,
        reference=reference,
        tie_correct=True,
        use_continuity=True,
        n_genes=n_genes,
    )
    res = gpu.uns["rank_genes_groups"]
    for grp in res["names"].dtype.names:
        gm = dict(zip(res["names"][grp], np.asarray(res["pvals"][grp], float)))
        mask_g = labels == grp
        mask_r = (labels != grp) if reference == "rest" else (labels == reference)
        for gi, gene in enumerate(var.index):
            _, p = mannwhitneyu(
                dense[mask_g, gi],
                dense[mask_r, gi],
                use_continuity=True,
                alternative="two-sided",
                method="asymptotic",
            )
            np.testing.assert_allclose(
                gm[gene], p, rtol=1e-10, atol=1e-12, equal_nan=True
            )


# ---------------------------------------------------------------------------
# Entry-point / init validation (rank_genes_groups + _RankGenes + _select_groups)
# ---------------------------------------------------------------------------


def test_rank_genes_groups_default_method_is_ttest():
    """Omitting method= defaults to t-test (rank_genes_groups path)."""
    adata = _anndata_with_group_sizes({"0": 10, "1": 10}, seed=5)
    rsc.tl.rank_genes_groups(adata, "group", use_raw=False)
    assert adata.uns["rank_genes_groups"]["params"]["method"] == "t-test"


@pytest.mark.parametrize(
    ("override", "exc", "match"),
    [
        ({"method": "nope"}, ValueError, "method must be one of"),
        ({"corr_method": "foo"}, ValueError, "corr_method must be either"),
        ({"chunk_size": 0}, ValueError, "chunk_size must be a positive integer"),
        ({"chunk_size": -4}, ValueError, "chunk_size must be a positive integer"),
        ({"groups": "0"}, ValueError, "Specify a sequence of groups"),
        ({"reference": "ZZ"}, ValueError, "needs to be one of groupby"),
    ],
)
def test_rank_genes_groups_invalid_args_raise(override, exc, match):
    """Public-API argument validation raises (covers __init__/_core guards)."""
    adata = _anndata_with_group_sizes({"0": 10, "1": 10}, seed=5)
    kwargs = {"method": "wilcoxon", "use_raw": False, **override}
    with pytest.raises(exc, match=match):
        rsc.tl.rank_genes_groups(adata, "group", **kwargs)


def test_rank_genes_groups_mask_var_missing_key_raises():
    adata = _anndata_with_group_sizes({"0": 10, "1": 10}, seed=5)
    with pytest.raises(KeyError, match="not found in adata.var"):
        rsc.tl.rank_genes_groups(
            adata, "group", method="wilcoxon", use_raw=False, mask_var="nope"
        )


def test_rank_genes_groups_mask_var_wrong_shape_raises():
    adata = _anndata_with_group_sizes({"0": 10, "1": 10}, seed=5)
    with pytest.raises(ValueError, match="mask_var has wrong shape"):
        rsc.tl.rank_genes_groups(
            adata,
            "group",
            method="wilcoxon",
            use_raw=False,
            mask_var=np.ones(adata.n_vars + 3, dtype=bool),
        )


def test_rank_genes_groups_layer_and_use_raw_conflict_raises():
    adata = _anndata_with_group_sizes({"0": 10, "1": 10}, seed=5)
    adata.layers["L"] = adata.X.copy()
    with pytest.raises(ValueError, match="Cannot specify .layer. and have"):
        rsc.tl.rank_genes_groups(
            adata, "group", method="wilcoxon", layer="L", use_raw=True
        )


def test_rank_genes_groups_use_raw_without_raw_raises():
    adata = _anndata_with_group_sizes({"0": 10, "1": 10}, seed=5)
    with pytest.raises(ValueError, match="is empty"):
        rsc.tl.rank_genes_groups(adata, "group", method="wilcoxon", use_raw=True)


def test_singleton_group_without_skip_raises():
    """Non-skip path: a <2-cell group raises in _select_groups (line 131-135)."""
    adata = _anndata_with_group_sizes({"a": 10, "b": 10, "c": 1}, seed=5)
    with pytest.raises(ValueError, match="fewer than two samples"):
        rsc.tl.rank_genes_groups(adata, "group", method="wilcoxon", use_raw=False)


@pytest.mark.parametrize("use_raw", [None, True])
def test_rank_genes_groups_reads_raw_matches_scanpy(use_raw):
    """use_raw=None (raw present) and use_raw=True both read adata.raw. X is
    overwritten with rank-scrambling noise so a path that wrongly read .X would
    diverge from scanpy (non-vacuous)."""
    adata = _anndata_with_group_sizes({"0": 30, "1": 30, "2": 30}, seed=6)
    adata.raw = adata.copy()  # raw holds the real signal
    rng = np.random.default_rng(99)
    adata.X = rng.integers(0, 6, size=adata.shape).astype(np.float64)  # noise in .X
    bdata = adata.copy()
    kw = {"method": "wilcoxon", "use_raw": use_raw, "tie_correct": True}
    rsc.tl.rank_genes_groups(adata, "group", **kw)
    sc.tl.rank_genes_groups(bdata, "group", **kw)
    g, c = adata.uns["rank_genes_groups"], bdata.uns["rank_genes_groups"]
    for grp in g["scores"].dtype.names:
        gm = dict(zip(g["names"][grp], np.asarray(g["scores"][grp], float)))
        cm = dict(zip(c["names"][grp], np.asarray(c["scores"][grp], float)))
        for gene, val in gm.items():
            np.testing.assert_allclose(
                val, cm[gene], rtol=1e-12, atol=1e-13, equal_nan=True
            )


@pytest.mark.parametrize("reference", ["rest", "1"])  # OVR (_core) + OVO (_wilcoxon)
@pytest.mark.parametrize("fmt", ["numpy_dense", "scipy_csr"])
def test_log1p_base_logfoldchanges_match_scanpy(reference, fmt):
    """A non-default log1p base changes expm1 in the logfoldchange computation
    (_core.py:115 + the OVO host-sparse fast path _wilcoxon.py:232-234)."""
    rng = np.random.default_rng(7)
    dense = rng.integers(1, 6, size=(120, 6)).astype(np.float64)  # nonneg, finite lfc
    obs = pd.DataFrame({"group": pd.Categorical([f"{i % 3}" for i in range(120)])})
    var = pd.DataFrame(index=[f"g{i}" for i in range(6)])
    gpu = sc.AnnData(X=_to_format(dense, fmt), obs=obs.copy(), var=var.copy())
    cpu = sc.AnnData(X=dense.copy(), obs=obs.copy(), var=var.copy())
    gpu.uns["log1p"] = {"base": 2.0}
    cpu.uns["log1p"] = {"base": 2.0}
    kw = {
        "method": "wilcoxon",
        "use_raw": False,
        "reference": reference,
        "tie_correct": True,
    }
    rsc.tl.rank_genes_groups(gpu, "group", **kw)
    sc.tl.rank_genes_groups(cpu, "group", **kw)
    g, c = gpu.uns["rank_genes_groups"], cpu.uns["rank_genes_groups"]
    for grp in g["logfoldchanges"].dtype.names:
        gm = dict(zip(g["names"][grp], np.asarray(g["logfoldchanges"][grp], float)))
        cm = dict(zip(c["names"][grp], np.asarray(c["logfoldchanges"][grp], float)))
        for gene, val in gm.items():
            np.testing.assert_allclose(
                val, cm[gene], rtol=1e-6, atol=1e-6, equal_nan=True
            )


# ---------------------------------------------------------------------------
# OVO / OVR parity & dispatch gaps
# ---------------------------------------------------------------------------


def test_ovo_dense_fallback_pts_match_scanpy():
    """OVO sparse-negative dense fallback computes pts via _fill_ovo_chunk_stats
    (ref + group branches). Validate pts vs scanpy on the dense equivalent."""
    rng = np.random.default_rng(11)
    dense = (rng.random((120, 8)) * 5.0).astype(np.float64)
    dense[dense < 1.5] = 0.0
    dense[rng.random(dense.shape) < 0.01] = -0.5  # negatives -> dense fallback
    obs = pd.DataFrame(
        {"group": pd.Categorical(["a" if i % 2 else "b" for i in range(120)])}
    )
    var = pd.DataFrame(index=[f"g{i}" for i in range(8)])
    gpu = sc.AnnData(X=sp.csr_matrix(dense), obs=obs.copy(), var=var.copy())
    cpu = sc.AnnData(X=dense.copy(), obs=obs.copy(), var=var.copy())
    kw = {
        "method": "wilcoxon",
        "use_raw": False,
        "reference": "b",
        "pts": True,
        "n_genes": 8,
    }
    rsc.tl.rank_genes_groups(gpu, "group", **kw)
    sc.tl.rank_genes_groups(cpu, "group", **kw)
    g, c = gpu.uns["rank_genes_groups"], cpu.uns["rank_genes_groups"]
    for col in c["pts"].columns:
        np.testing.assert_allclose(
            g["pts"].loc[c["pts"].index, col].values,
            c["pts"][col].values,
            rtol=1e-12,
            atol=1e-13,
        )


@pytest.mark.parametrize("fmt", ["numpy_dense", "cupy_csr"])  # CPU + GPU FDR epilogues
def test_bonferroni_matches_scanpy(fmt):
    """Bonferroni correction (CPU _core.py:584 via dense, GPU :630-631 via the
    cupy OVO result path) must match scanpy, not just be <=1 (the prior tests
    only asserted the tautological clamp)."""
    rng = np.random.default_rng(12)
    dense = rng.integers(0, 5, size=(150, 6)).astype(np.float64)
    dense[dense < 1.0] = 0.0
    obs = pd.DataFrame({"group": pd.Categorical([f"{i % 3}" for i in range(150)])})
    var = pd.DataFrame(index=[f"g{i}" for i in range(6)])
    gpu = sc.AnnData(X=_to_format(dense, fmt), obs=obs.copy(), var=var.copy())
    cpu = sc.AnnData(X=dense.copy(), obs=obs.copy(), var=var.copy())
    kw = {
        "method": "wilcoxon",
        "use_raw": False,
        "reference": "1",
        "corr_method": "bonferroni",
        "tie_correct": True,
        "n_genes": 6,
    }
    rsc.tl.rank_genes_groups(gpu, "group", **kw)
    sc.tl.rank_genes_groups(cpu, "group", **kw)
    g, c = gpu.uns["rank_genes_groups"], cpu.uns["rank_genes_groups"]
    for fld in ("scores", "pvals", "pvals_adj"):
        for grp in g[fld].dtype.names:
            gm = dict(zip(g["names"][grp], np.asarray(g[fld][grp], float)))
            cm = dict(zip(c["names"][grp], np.asarray(c[fld][grp], float)))
            for gene, val in gm.items():
                np.testing.assert_allclose(
                    val, cm[gene], rtol=1e-12, atol=1e-13, equal_nan=True
                )


def test_ovr_all_empty_csc_totals_runs():
    """All-zero host CSC + a groups= subset (leaves an unselected category) +
    reference='rest' + pts=True exercises the empty-column totals branch."""
    dense = np.zeros((20, 5), dtype=np.float64)
    obs = pd.DataFrame({"group": pd.Categorical([f"{i % 3}" for i in range(20)])})
    var = pd.DataFrame(index=[f"g{i}" for i in range(5)])
    adata = sc.AnnData(X=sp.csc_matrix(dense), obs=obs, var=var)
    rsc.tl.rank_genes_groups(
        adata,
        "group",
        method="wilcoxon",
        use_raw=False,
        groups=["0", "1"],
        reference="rest",
        pts=True,
    )
    res = adata.uns["rank_genes_groups"]
    for grp in res["scores"].dtype.names:
        assert np.all(np.isfinite(np.asarray(res["scores"][grp], float)))
    assert "pts_rest" in res


@pytest.mark.parametrize("fmt", ["scipy_csr", "scipy_csc", "cupy_csr", "cupy_csc"])
def test_ovr_fully_dense_column_match_scanpy(fmt):
    """A column with no structural zeros (nnz==n_rows) hits the total_zero==0
    branch of the sparse OVR accumulate kernel. Validate vs scanpy."""
    rng = np.random.default_rng(13)
    dense = rng.integers(0, 5, size=(90, 4)).astype(np.float64)
    dense[dense < 1.0] = 0.0
    dense[:, 0] = rng.integers(1, 6, size=90)  # column 0 strictly positive -> no zeros
    obs = pd.DataFrame({"group": pd.Categorical([f"{i % 3}" for i in range(90)])})
    var = pd.DataFrame(index=[f"g{i}" for i in range(4)])
    gpu = sc.AnnData(X=_to_format(dense, fmt), obs=obs.copy(), var=var.copy())
    cpu = sc.AnnData(X=dense.copy(), obs=obs.copy(), var=var.copy())
    kw = {
        "method": "wilcoxon",
        "use_raw": False,
        "reference": "rest",
        "tie_correct": True,
    }
    rsc.tl.rank_genes_groups(gpu, "group", **kw)
    sc.tl.rank_genes_groups(cpu, "group", **kw)
    g, c = gpu.uns["rank_genes_groups"], cpu.uns["rank_genes_groups"]
    for fld in ("scores", "pvals"):
        for grp in g[fld].dtype.names:
            gm = dict(zip(g["names"][grp], np.asarray(g[fld][grp], float)))
            cm = dict(zip(c["names"][grp], np.asarray(c[fld][grp], float)))
            for gene, val in gm.items():
                np.testing.assert_allclose(
                    val, cm[gene], rtol=1e-13, atol=1e-15, equal_nan=True
                )


@pytest.mark.parametrize("fmt", ["cupy_csr", "cupy_csc"])
def test_ovr_device_sparse_subset_match_scanpy(fmt):
    """Device-sparse OVR with a groups= subset exercises the sentinel-group skip
    in the device sparse kernels. Validate vs scanpy on the dense copy."""
    rng = np.random.default_rng(14)
    dense = rng.integers(0, 6, size=(160, 6)).astype(np.float64)
    dense[dense < 1.0] = 0.0
    obs = pd.DataFrame({"group": pd.Categorical([f"{i % 4}" for i in range(160)])})
    var = pd.DataFrame(index=[f"g{i}" for i in range(6)])
    gpu = sc.AnnData(X=_to_format(dense, fmt), obs=obs.copy(), var=var.copy())
    cpu = sc.AnnData(X=dense.copy(), obs=obs.copy(), var=var.copy())
    kw = {
        "method": "wilcoxon",
        "use_raw": False,
        "groups": ["0", "2"],
        "reference": "rest",
        "tie_correct": True,
    }
    rsc.tl.rank_genes_groups(gpu, "group", **kw)
    sc.tl.rank_genes_groups(cpu, "group", **kw)
    g, c = gpu.uns["rank_genes_groups"], cpu.uns["rank_genes_groups"]
    for fld in ("scores", "pvals"):
        for grp in g[fld].dtype.names:
            gm = dict(zip(g["names"][grp], np.asarray(g[fld][grp], float)))
            cm = dict(zip(c["names"][grp], np.asarray(c[fld][grp], float)))
            for gene, val in gm.items():
                np.testing.assert_allclose(
                    val, cm[gene], rtol=1e-13, atol=1e-15, equal_nan=True
                )


@pytest.mark.parametrize("reference", ["rest", "1"])
def test_host_csc_int64_indices_cast_matches_int32(reference):
    """Host CSC has no *_idx64 template, so int64 indices are cast to int32
    (_wilcoxon.py:355->357). Result must be bit-identical to the int32 input."""
    rng = np.random.default_rng(15)
    dense = rng.integers(0, 5, size=(120, 6)).astype(np.float64)
    dense[dense < 1.0] = 0.0
    obs = pd.DataFrame({"group": pd.Categorical([f"{i % 3}" for i in range(120)])})
    var = pd.DataFrame(index=[f"g{i}" for i in range(6)])
    base = sp.csc_matrix(dense)
    a32 = sc.AnnData(X=base.copy(), obs=obs.copy(), var=var.copy())
    m64 = base.copy()
    m64.indices = m64.indices.astype(np.int64)  # keep indptr int32
    a64 = sc.AnnData(X=m64, obs=obs.copy(), var=var.copy())
    kw = {
        "method": "wilcoxon",
        "use_raw": False,
        "reference": reference,
        "tie_correct": True,
    }
    rsc.tl.rank_genes_groups(a32, "group", **kw)
    rsc.tl.rank_genes_groups(a64, "group", **kw)
    r32, r64 = a32.uns["rank_genes_groups"], a64.uns["rank_genes_groups"]
    for fld in ("scores", "pvals"):
        for grp in r32[fld].dtype.names:
            np.testing.assert_array_equal(
                np.asarray(r64[fld][grp]), np.asarray(r32[fld][grp])
            )


def test_device_sparse_float16_raises():
    """A cupy sparse matrix with float16 data raises a clear TypeError (the
    device counterpart of the host float16 guard)."""
    rng = np.random.default_rng(16)
    dense = np.abs(rng.standard_normal((60, 4))).astype(np.float32)
    dense[dense < 0.5] = 0.0
    mat = cpsp.csr_matrix(cp.asarray(dense))
    mat.data = mat.data.astype(cp.float16)
    adata = sc.AnnData(
        X=mat,
        obs=pd.DataFrame({"group": pd.Categorical([f"{i % 3}" for i in range(60)])}),
        var=pd.DataFrame(index=[f"g{i}" for i in range(4)]),
    )
    with pytest.raises(TypeError, match="float32"):
        rsc.tl.rank_genes_groups(adata, "group", method="wilcoxon", use_raw=False)
