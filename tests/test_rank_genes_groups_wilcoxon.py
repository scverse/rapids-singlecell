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


@pytest.mark.parametrize(
    "method",
    ["t-test", "t-test_overestim_var", "wilcoxon", "wilcoxon_binned", "logreg"],
)
@pytest.mark.parametrize("fmt", ["scipy_csr", "scipy_csc", "cupy_csr", "cupy_csc"])
def test_rank_genes_groups_sparse_negative_values_raise(method, fmt):
    X = np.array(
        [
            [-1.0, 0.0, 2.0],
            [0.0, 1.0, 0.0],
            [2.0, 0.0, 1.0],
            [0.0, 3.0, 0.0],
        ],
        dtype=np.float32,
    )
    adata = sc.AnnData(
        X=_to_format(X, fmt),
        obs=pd.DataFrame(
            {"group": pd.Categorical(["a", "a", "b", "b"], categories=["a", "b"])}
        ),
        var=pd.DataFrame(index=["g0", "g1", "g2"]),
    )

    with pytest.raises(ValueError, match="Sparse input contains negative values"):
        rsc.tl.rank_genes_groups(adata, "group", method=method, use_raw=False)


@pytest.mark.parametrize("fmt", ["numpy_dense", "scipy_csr", "cupy_dense", "cupy_csr"])
def test_rank_genes_groups_complex_values_raise(fmt):
    X = np.array(
        [
            [1.0 + 0.0j, 0.0, 2.0],
            [0.0, 1.0, 0.0],
            [2.0, 0.0, 1.0],
            [0.0, 3.0, 0.0],
        ],
        dtype=np.complex64,
    )
    adata = sc.AnnData(
        X=_to_format(X, fmt),
        obs=pd.DataFrame(
            {"group": pd.Categorical(["a", "a", "b", "b"], categories=["a", "b"])}
        ),
        var=pd.DataFrame(index=["g0", "g1", "g2"]),
    )

    with pytest.raises(TypeError, match="complex expression values"):
        rsc.tl.rank_genes_groups(adata, "group", method="wilcoxon", use_raw=False)


def test_device_sparse_int64_indptr_selects_i64_kernel():
    from rapids_singlecell._cuda import _wilcoxon_sparse_cuda as _wcs
    from rapids_singlecell.tools._rank_genes_groups._wilcoxon import (
        _device_sparse_arrays_f32,
        _device_sparse_fn,
    )

    class FakeSparse:
        data = cp.asarray([1.0], dtype=cp.float32)
        indices = cp.asarray([0], dtype=cp.int32)
        indptr = cp.asarray([0, np.iinfo(np.int32).max + 1], dtype=cp.int64)

    # int64 indptr is preserved (no truncation, no dense fallback); the row
    # indices stay int32 because cells/genes are always < 2^31.
    data, indices, indptr = _device_sparse_arrays_f32(FakeSparse())
    assert indptr.dtype == cp.int64
    assert indices.dtype == cp.int32
    assert data.dtype == cp.float32

    # Dispatch routes an int64 indptr to the *_i64 kernel binding, int32 to base.
    assert (
        _device_sparse_fn(_wcs, "ovr_sparse_csc_device", indptr)
        is _wcs.ovr_sparse_csc_device_i64
    )
    assert (
        _device_sparse_fn(_wcs, "ovr_sparse_csc_device", indices)
        is _wcs.ovr_sparse_csc_device
    )


@pytest.mark.parametrize("layout", ["csc", "csr"])
def test_device_ovr_sparse_i64_indptr_matches_i32(layout):
    # cupyx coerces small matrices to int32 indptr, so int64 support is only
    # reachable for nnz > 2^31. Exercise the int64-templated kernels directly
    # with a hand-built int64 indptr and assert bit-parity with the int32 path.
    from rapids_singlecell._cuda import _wilcoxon_sparse_cuda as _wcs

    rng = np.random.default_rng(0)
    n_rows, n_cols, n_groups = 120, 10, 4
    dense = np.abs(rng.standard_normal((n_rows, n_cols))).astype(np.float32)
    dense[dense < 0.6] = 0.0
    mat = sp.csc_matrix(dense) if layout == "csc" else sp.csr_matrix(dense)
    mat.sort_indices()
    gcodes = rng.integers(0, n_groups, n_rows).astype(np.int32)
    gsizes = np.bincount(gcodes, minlength=n_groups).astype(np.float64)

    data = cp.asarray(mat.data, dtype=cp.float32)
    indices = cp.asarray(mat.indices, dtype=cp.int32)
    g = cp.asarray(gcodes)
    gs = cp.asarray(gsizes)
    base = getattr(_wcs, f"ovr_sparse_{layout}_device")
    i64 = getattr(_wcs, f"ovr_sparse_{layout}_device_i64")

    def run(indptr_dtype, fn):
        indptr = cp.asarray(mat.indptr, dtype=indptr_dtype)
        rs = cp.empty((n_groups, n_cols), dtype=cp.float64)
        tc = cp.ones(n_cols, dtype=cp.float64)
        fn(
            data,
            indices,
            indptr,
            g,
            gs,
            rs,
            tc,
            n_rows=n_rows,
            n_cols=n_cols,
            n_groups=n_groups,
            compute_tie_corr=True,
            sub_batch_cols=64,
        )
        cp.cuda.get_current_stream().synchronize()
        return rs.get(), tc.get()

    rs32, tc32 = run(cp.int32, base)
    rs64, tc64 = run(cp.int64, i64)
    np.testing.assert_array_equal(rs32, rs64)
    np.testing.assert_array_equal(tc32, tc64)


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
@pytest.mark.parametrize("pre_load", [False, True])
def test_wilcoxon_all_public_formats_match_scanpy(reference, fmt, pre_load):
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
    rsc.tl.rank_genes_groups(adata_gpu, **kw, pre_load=pre_load)
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
@pytest.mark.parametrize("pre_load", [False, True])
def test_rank_genes_groups_wilcoxon_subset_matches_scanpy(
    groups, reference, tie_correct, pre_load
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
        pre_load=pre_load,
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
    """Test that group order doesn't affect results."""
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

    expected_groups = {g for g in groups if g != reference}
    assert set(adata.uns["rank_genes_groups"]["names"].dtype.names) == expected_groups
    assert set(bdata.uns["rank_genes_groups"]["names"].dtype.names) == expected_groups

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
@pytest.mark.parametrize("pre_load", [True, False])
def test_rank_genes_groups_wilcoxon_pts(reference, pre_load):
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
        pre_load=pre_load,
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
