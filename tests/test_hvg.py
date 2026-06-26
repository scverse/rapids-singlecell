from __future__ import annotations

from pathlib import Path

import cupy as cp
import cupyx as cpx
import numpy as np
import pandas as pd
import pytest
import scanpy as sc
from anndata import AnnData
from scipy.sparse import csr_matrix

import rapids_singlecell as rsc

FILE = Path(__file__).parent / Path("_scripts/seurat_hvg.csv")
FILE_V3 = Path(__file__).parent / Path("_scripts/seurat_hvg_v3.csv.gz")
FILE_V3_BATCH = Path(__file__).parent / Path("_scripts/seurat_hvg_v3_batch.csv")
FILE_POISSON = Path(__file__).parent / Path("_data/scvi-poisson-ref.parquet")


@pytest.mark.parametrize("dtype", [cp.float32, cp.float64])
@pytest.mark.parametrize("sparse", [True, False])
def test_highly_variable_genes_basic(dtype, sparse):
    cudata = sc.datasets.blobs()
    cudata.X = cudata.X.astype(dtype)
    if sparse:
        cudata.X = csr_matrix(cudata.X)
        cudata.X = cpx.scipy.sparse.csr_matrix(cudata.X)
    else:
        cudata.X = cp.array(cudata.X)
    np.random.seed(0)
    cp.random.seed(0)
    cudata.obs["batch"] = np.random.binomial(3, 0.5, size=(cudata.n_obs))
    cudata.obs["batch"] = cudata.obs["batch"].astype("category")
    rsc.pp.highly_variable_genes(cudata, batch_key="batch")
    assert "highly_variable_nbatches" in cudata.var.columns
    assert "highly_variable_intersection" in cudata.var.columns

    cudata = sc.datasets.blobs()
    cudata.X = cudata.X.astype(dtype)
    if sparse:
        cudata.X = csr_matrix(cudata.X)
        cudata.X = cpx.scipy.sparse.csr_matrix(cudata.X)
    else:
        cudata.X = cp.array(cudata.X)
    batch = np.random.binomial(4, 0.5, size=(cudata.n_obs))
    cudata.obs["batch"] = batch
    cudata.obs["batch"] = cudata.obs["batch"].astype("category")
    rsc.pp.highly_variable_genes(cudata, batch_key="batch", n_top_genes=3)
    assert "highly_variable_nbatches" in cudata.var.columns
    assert cudata.var["highly_variable"].sum() == 3
    highly_var_first_layer = cudata.var["highly_variable"].copy()

    cudata = sc.datasets.blobs()
    cudata.X = cudata.X.astype(dtype)
    if sparse:
        cudata.X = csr_matrix(cudata.X)
        cudata.X = cpx.scipy.sparse.csr_matrix(cudata.X)
    else:
        cudata.X = cp.array(cudata.X)
    if sparse:
        new_layer = cudata.X.toarray()
    else:
        new_layer = cudata.X.copy()
    cp.random.shuffle(new_layer)
    if sparse:
        cudata.layers["test_layer"] = cpx.scipy.sparse.csr_matrix(new_layer)
    else:
        cudata.layers["test_layer"] = new_layer
    cudata.obs["batch"] = batch
    cudata.obs["batch"] = cudata.obs["batch"].astype("category")
    rsc.pp.highly_variable_genes(
        cudata, batch_key="batch", n_top_genes=3, layer="test_layer"
    )
    assert "highly_variable_nbatches" in cudata.var.columns
    assert cudata.var["highly_variable"].sum() == 3
    assert (highly_var_first_layer != cudata.var["highly_variable"]).any()

    rsc.pp.highly_variable_genes(cudata)
    no_batch_hvg = cudata.var.highly_variable.copy()
    assert no_batch_hvg.any()
    cudata.obs["batch"] = "batch"
    cudata.obs["batch"] = cudata.obs["batch"].astype("category")
    rsc.pp.highly_variable_genes(cudata, batch_key="batch")
    assert np.all(no_batch_hvg == cudata.var.highly_variable)
    assert np.all(cudata.var.highly_variable_intersection == cudata.var.highly_variable)

    cudata.obs["batch"] = "a"
    cudata.obs.batch.loc[::2] = "b"
    rsc.pp.highly_variable_genes(cudata, batch_key="batch")
    assert cudata.var["highly_variable"].any()

    colnames = [
        "means",
        "dispersions",
        "dispersions_norm",
        "highly_variable_nbatches",
        "highly_variable_intersection",
        "highly_variable",
    ]

    assert np.all(np.isin(colnames, cudata.var.columns))


@pytest.mark.parametrize("dtype", ["float32", "float64"])
def test_highly_variable_genes_compare_to_seurat(dtype):
    seurat_hvg_info = pd.read_csv(FILE, sep=" ")

    pbmc = sc.datasets.pbmc68k_reduced()
    pbmc.X = pbmc.raw.X
    pbmc.var_names_make_unique()
    pbmc.X = cpx.scipy.sparse.csr_matrix(pbmc.X, dtype=dtype)
    pbmc.X.sort_indices()
    rsc.pp.normalize_total(pbmc, target_sum=1e4)
    rsc.pp.log1p(pbmc)
    rsc.pp.highly_variable_genes(
        pbmc, flavor="seurat", min_mean=0.0125, max_mean=3, min_disp=0.5
    )

    np.testing.assert_array_equal(
        seurat_hvg_info["highly_variable"], pbmc.var["highly_variable"]
    )

    # (still) Not equal to tolerance rtol=2e-05, atol=2e-05
    # np.testing.assert_allclose(4, 3.9999, rtol=2e-05, atol=2e-05)
    np.testing.assert_allclose(
        seurat_hvg_info["means"],
        pbmc.var["means"],
        rtol=2e-05,
        atol=2e-05,
    )
    np.testing.assert_allclose(
        seurat_hvg_info["dispersions"],
        pbmc.var["dispersions"],
        rtol=2e-05,
        atol=2e-05,
    )
    np.testing.assert_allclose(
        seurat_hvg_info["dispersions_norm"],
        pbmc.var["dispersions_norm"],
        rtol=2e-05,
        atol=2e-05,
    )


def test_compare_to_seurat_v3():
    ### test without batch
    seurat_hvg_info = pd.read_csv(FILE_V3)

    pbmc = sc.datasets.pbmc3k()
    sc.pp.filter_cells(pbmc, min_genes=200)  # this doesn't do anything btw
    sc.pp.filter_genes(pbmc, min_cells=3)

    pbmc_dense = pbmc.copy()
    pbmc_dense.X = pbmc_dense.X.toarray()

    rsc.get.anndata_to_GPU(pbmc)
    rsc.get.anndata_to_GPU(pbmc_dense)

    rsc.pp.highly_variable_genes(pbmc, n_top_genes=1000, flavor="seurat_v3")
    rsc.pp.highly_variable_genes(pbmc_dense, n_top_genes=1000, flavor="seurat_v3")

    np.testing.assert_allclose(
        seurat_hvg_info["variance"],
        pbmc.var["variances"],
        rtol=2e-05,
        atol=2e-05,
    )
    np.testing.assert_allclose(
        seurat_hvg_info["variance"],
        pbmc_dense.var["variances"],
        rtol=2e-05,
        atol=2e-05,
    )
    ### test with batch
    # introduce a dummy "technical covariate"; this is used in Seurat's SelectIntegrationFeatures
    pbmc.obs["dummy_tech"] = (
        "source_" + pd.array([*range(1, 6), 5]).repeat(500).astype("string")
    )[: pbmc.n_obs]

    pbmc_paper = pbmc.copy()

    rsc.pp.highly_variable_genes(
        pbmc_paper,
        n_top_genes=2000,
        flavor="seurat_v3_paper",
        batch_key="dummy_tech",
    )

    rsc.pp.highly_variable_genes(
        pbmc,
        n_top_genes=2000,
        flavor="seurat_v3",
        batch_key="dummy_tech",
    )

    seurat_v3_paper = pbmc_paper.var
    seurat_v3 = pbmc.var

    seurat_hvg_info_batch = pd.read_csv(FILE_V3_BATCH)
    seu = pd.Index(seurat_hvg_info_batch["x"].to_numpy())

    gene_intersection_paper = seu.intersection(
        seurat_v3_paper[seurat_v3_paper["highly_variable"]].index
    )
    gene_intersection_impl = seu.intersection(
        seurat_v3[seurat_v3["highly_variable"]].index
    )
    assert len(gene_intersection_paper) / 2000 > 0.95
    assert len(gene_intersection_impl) / 2000 < 0.95


@pytest.mark.parametrize("dtype", ["float32", "float64"])
@pytest.mark.parametrize("sparse", ["csr", "csc", "dense"])
def test_seurat_v3_mean_var_output_with_batchkey(dtype, sparse):
    pbmc = sc.datasets.pbmc3k()
    pbmc.var_names_make_unique()
    if sparse == "csr":
        pbmc.X = cpx.scipy.sparse.csr_matrix(pbmc.X, dtype=dtype)
    elif sparse == "csc":
        pbmc.X = cpx.scipy.sparse.csr_matrix(pbmc.X, dtype=dtype).tocsc()
    elif sparse == "dense":
        pbmc.X = cpx.scipy.sparse.csr_matrix(pbmc.X, dtype=dtype).toarray()
    n_cells = pbmc.shape[0]
    batch = np.zeros((n_cells), dtype=int)
    batch[1500:] = 1
    pbmc.obs["batch"] = batch

    # true_mean, true_var = _get_mean_var(pbmc.X)
    if sparse == "csr" or sparse == "csc":
        X = pbmc.X.toarray().astype(cp.float64)
    else:
        X = pbmc.X.copy().astype(cp.float64)

    true_mean = cp.mean(X, axis=0)
    true_var = cp.var(X, axis=0, dtype=np.float64, ddof=1)

    rsc.pp.highly_variable_genes(
        pbmc, batch_key="batch", flavor="seurat_v3", n_top_genes=4000
    )
    cp.testing.assert_allclose(true_mean, pbmc.var["means"], rtol=2e-05, atol=2e-05)
    cp.testing.assert_allclose(true_var, pbmc.var["variances"], rtol=2e-05, atol=2e-05)


@pytest.mark.parametrize("dtype", ["float32", "float64"])
def test_cellranger_n_top_genes_warning(dtype):
    X = cp.random.poisson(2, (100, 30), dtype=dtype)
    cudata = AnnData(X=cpx.scipy.sparse.csr_matrix(X, dtype=dtype))
    rsc.pp.normalize_total(cudata, target_sum=1e4)
    rsc.pp.log1p(cudata)

    with pytest.warns(
        UserWarning,
        match="`n_top_genes` > number of normalized dispersions, returning all genes with normalized dispersions.",
    ):
        rsc.pp.highly_variable_genes(cudata, n_top_genes=1000, flavor="cell_ranger")


def _check_pearson_hvg_columns(output_df, n_top_genes):
    assert output_df["residual_variances"].dtype.kind == "f"

    assert output_df["highly_variable"].values.dtype is np.dtype("bool")
    assert np.sum(output_df["highly_variable"]) == n_top_genes

    assert np.nanmax(output_df["highly_variable_rank"].values) <= n_top_genes - 1


@pytest.mark.parametrize("clip", [None, np.inf, 30])
@pytest.mark.parametrize("theta", [100, np.inf])
@pytest.mark.parametrize("n_top_genes", [100, 200])
@pytest.mark.parametrize("dtype", ["float32", "float64"])
@pytest.mark.parametrize("sparse", [True, False])
def test_highly_variable_genes_pearson_residuals_general(
    clip, theta, n_top_genes, dtype, sparse
):
    cudata = sc.datasets.pbmc3k().copy()
    cudata = cudata[:1000, :500]
    np.random.seed(42)
    cudata.obs["batch"] = np.random.randint(0, 3, size=cudata.shape[0])
    sc.pp.filter_genes(cudata, min_cells=1)
    # cleanup var
    del cudata.var
    if sparse:
        cudata.X = cpx.scipy.sparse.csr_matrix(cudata.X, dtype=dtype)
    else:
        cudata.X = cp.array(cudata.X.toarray(), dtype=dtype)
    # compute reference output
    residuals_reference = rsc.pp.normalize_pearson_residuals(
        cudata, clip=clip, theta=theta, inplace=False
    )
    residual_variances_reference = cp.var(residuals_reference, axis=0)

    # compute output to be tested
    rsc.pp.highly_variable_genes(
        cudata,
        flavor="pearson_residuals",
        n_top_genes=n_top_genes,
        clip=clip,
        theta=theta,
    )

    # check output is complete
    for key in [
        "highly_variable",
        "means",
        "variances",
        "residual_variances",
        "highly_variable_rank",
    ]:
        assert key in cudata.var.columns

    print(cudata.var["residual_variances"].values.shape)
    print(residual_variances_reference.shape)
    print(
        f"cudata.var['residual_variances'].values: {cudata.var['residual_variances'].values}"
    )
    print(f"residual_variances_reference: {residual_variances_reference}")
    assert np.allclose(
        cudata.var["residual_variances"].values, residual_variances_reference
    )

    # check hvg flag - verify HVG selection is reasonable
    # (genes marked as HVG should have high residual variance)
    hvg_idx = np.where(cudata.var["highly_variable"])[0]
    topn_idx = np.sort(
        np.argsort(-cudata.var["residual_variances"].values)[:n_top_genes]
    )
    # Allow for minor ranking differences due to numerical precision
    # when genes have nearly identical variances
    if dtype == "float32":
        overlap = len(set(hvg_idx) & set(topn_idx))
        assert overlap >= n_top_genes - int(n_top_genes * 0.01), (
            f"HVG overlap {overlap}/{n_top_genes} too low"
        )
    else:
        assert np.all(hvg_idx == topn_idx)
    # check ranks
    assert np.nanmin(cudata.var["highly_variable_rank"].values) == 0

    # more general checks on ranks, hvg flag and residual variance
    _check_pearson_hvg_columns(cudata.var, n_top_genes)


@pytest.mark.parametrize("n_top_genes", [100, 200])
@pytest.mark.parametrize("dtype", ["float32", "float64"])
def test_highly_variable_genes_pearson_residuals_batch(n_top_genes, dtype):
    cudata = sc.datasets.pbmc3k().copy()
    cudata = cudata[:1000, :500].copy()
    np.random.seed(42)
    cudata.obs["batch"] = np.random.randint(0, 3, size=cudata.shape[0])
    sc.pp.filter_genes(cudata, min_cells=1)
    # cleanup var
    del cudata.var

    cudata.X = cpx.scipy.sparse.csr_matrix(cudata.X, dtype=dtype)
    n_genes = cudata.shape[1]

    rsc.pp.highly_variable_genes(
        cudata,
        flavor="pearson_residuals",
        n_top_genes=n_top_genes,
        batch_key="batch",
    )

    # check output is complete
    for key in [
        "highly_variable",
        "means",
        "variances",
        "residual_variances",
        "highly_variable_rank",
        "highly_variable_nbatches",
        "highly_variable_intersection",
    ]:
        assert key in cudata.var.keys()

    # general checks on ranks, hvg flag and residual variance
    _check_pearson_hvg_columns(cudata.var, n_top_genes)

    # check intersection flag
    nbatches = len(np.unique(cudata.obs["batch"]))
    assert cudata.var["highly_variable_intersection"].values.dtype is np.dtype("bool")
    assert np.sum(cudata.var["highly_variable_intersection"]) <= n_top_genes * nbatches
    assert np.all(
        cudata.var["highly_variable"][cudata.var.highly_variable_intersection]
    )

    # check ranks (with batch_key these are the median of within-batch ranks)
    assert cudata.var["highly_variable_rank"].dtype.kind == "f"

    # check nbatches
    assert cudata.var["highly_variable_nbatches"].values.dtype is np.dtype("int")
    assert np.min(cudata.var["highly_variable_nbatches"].values) >= 0
    assert np.max(cudata.var["highly_variable_nbatches"].values) <= nbatches

    assert len(cudata.var) == n_genes


def test_pearson_residuals_batch_order_invariant():
    """HVG ranking must not depend on alphabetical batch-label order."""
    rng = np.random.default_rng(0)
    n_big, n_small, n_genes = 5000, 200, 200
    counts = (rng.random((n_big + n_small, n_genes)) < 0.05).astype(np.int32)
    counts *= rng.integers(1, 31, size=counts.shape, dtype=np.int32)
    X = csr_matrix(counts.astype(np.float32))

    a1 = AnnData(X=cpx.scipy.sparse.csr_matrix(X.copy()))
    a1.obs["batch"] = np.array(["A"] * n_big + ["B"] * n_small)
    a1.obs["batch"] = a1.obs["batch"].astype("category")
    rsc.pp.highly_variable_genes(
        a1,
        flavor="pearson_residuals",
        n_top_genes=100,
        batch_key="batch",
        check_values=False,
    )

    a2 = AnnData(X=cpx.scipy.sparse.csr_matrix(X.copy()))
    a2.obs["batch"] = np.array(["B"] * n_big + ["A"] * n_small)
    a2.obs["batch"] = a2.obs["batch"].astype("category")
    rsc.pp.highly_variable_genes(
        a2,
        flavor="pearson_residuals",
        n_top_genes=100,
        batch_key="batch",
        check_values=False,
    )

    np.testing.assert_allclose(
        a1.var["residual_variances"].to_numpy(),
        a2.var["residual_variances"].to_numpy(),
        atol=1e-5,
    )


@pytest.mark.parametrize("dtype", ["float32", "float64"])
@pytest.mark.parametrize("sparse", [True, False])
def test_poisson_gene_selection_compare_to_scvi(dtype, sparse):
    """Test poisson_gene_selection against scvi-tools reference output.

    Note: scvi uses Monte Carlo sampling while we use the exact analytical formula
    P(Bern(p_obs) > Bern(p_exp)) = p_obs * (1 - p_exp). This leads to slightly
    different prob_zero_enrichment values and rankings, but the underlying
    observed/expected fractions should match closely.
    """
    ref = pd.read_parquet(FILE_POISSON)

    pbmc = sc.datasets.pbmc3k()
    sc.pp.filter_genes(pbmc, min_counts=5)
    if sparse:
        pbmc.X = cpx.scipy.sparse.csr_matrix(pbmc.X, dtype=dtype)
    else:
        pbmc.X = cp.array(pbmc.X.toarray(), dtype=dtype)

    n_top_genes = 2000
    rsc.pp.highly_variable_genes(
        pbmc,
        flavor="poisson_gene_selection",
        n_top_genes=n_top_genes,
        check_values=False,
    )

    # Check all expected columns are present
    for key in [
        "highly_variable",
        "observed_fraction_zeros",
        "expected_fraction_zeros",
        "prob_zero_enriched_nbatches",
        "prob_zero_enrichment",
        "prob_zero_enrichment_rank",
    ]:
        assert key in pbmc.var.columns, f"Missing column: {key}"

    # Compare observed_fraction_zeros - these should match exactly
    # (deterministic calculation from count data)
    np.testing.assert_allclose(
        ref["observed_fraction_zeros"].values,
        pbmc.var["observed_fraction_zeros"].values,
        rtol=1e-5,
        atol=1e-6,
        err_msg="observed_fraction_zeros mismatch",
    )

    # Compare expected_fraction_zeros - should match closely
    # (deterministic Poisson model calculation)
    np.testing.assert_allclose(
        ref["expected_fraction_zeros"].values,
        pbmc.var["expected_fraction_zeros"].values,
        rtol=1e-4,
        atol=1e-6,
        err_msg="expected_fraction_zeros mismatch",
    )

    # Compare HVG overlap - scvi uses Monte Carlo sampling so rankings may differ
    # at boundaries, but the overlap should be high (>95%)
    ref_hvg = set(ref[ref["highly_variable"]].index)
    our_hvg = set(pbmc.var[pbmc.var["highly_variable"]].index)
    overlap = len(ref_hvg & our_hvg)
    overlap_pct = overlap / n_top_genes
    assert overlap_pct >= 0.95, (
        f"HVG overlap {overlap_pct:.1%} is too low (expected >= 95%)"
    )

    # Check prob_zero_enriched_nbatches (should be 1 for HVGs, 0 for non-HVGs in single batch)
    hvg_mask = pbmc.var["highly_variable"]
    assert (pbmc.var.loc[hvg_mask, "prob_zero_enriched_nbatches"] == 1).all()
    assert (pbmc.var.loc[~hvg_mask, "prob_zero_enriched_nbatches"] == 0).all()

    # Check that exactly n_top_genes are selected
    assert pbmc.var["highly_variable"].sum() == n_top_genes

    # Check uns is set correctly
    assert pbmc.uns["hvg"]["flavor"] == "poisson_zeros"
