from __future__ import annotations

from contextlib import nullcontext
from pathlib import Path
from typing import TYPE_CHECKING

import cupy as cp
import numpy as np
import pytest
import scanpy as sc
from anndata import AnnData
from scanpy.datasets import paul15
from scipy.sparse import csc_matrix, csr_matrix

import rapids_singlecell as rsc

if TYPE_CHECKING:
    from numpy.typing import NDArray


HERE = Path(__file__).parent / "_data"


def _create_random_gene_names(n_genes, name_length) -> NDArray[np.str_]:
    """
    creates a bunch of random gene names (just CAPS letters)
    """
    return np.array(
        [
            "".join(map(chr, np.random.randint(65, 90, name_length)))
            for _ in range(n_genes)
        ]
    )


def _create_sparse_nan_matrix(rows, cols, percent_zero, percent_nan):
    """
    creates a sparse matrix, with certain amounts of NaN and Zeros
    """
    A = np.random.randint(0, 1000, rows * cols).reshape((rows, cols)).astype("float32")
    maskzero = np.random.rand(rows, cols) < percent_zero
    masknan = np.random.rand(rows, cols) < percent_nan
    if np.any(maskzero):
        A[maskzero] = 0
    if np.any(masknan):
        A[masknan] = np.nan
    S = csr_matrix(A)
    return S


def _create_adata(n_obs, n_var, p_zero, p_nan):
    """
    creates an AnnData with random data, sparseness and some NaN values
    """
    X = _create_sparse_nan_matrix(n_obs, n_var, p_zero, p_nan)
    adata = AnnData(X)
    gene_names = _create_random_gene_names(n_var, name_length=6)
    adata.var_names = gene_names
    return adata


@pytest.mark.parametrize("array_type", ["csr", "csc", "default"])
def test_score_with_reference(array_type):
    # Checks if score_genes output agrees with scanpy
    adata = paul15()
    if array_type != "default":
        if array_type == "csr":
            adata.X = csr_matrix(adata.X)
        elif array_type == "csc":
            adata.X = csc_matrix(adata.X)
    adata.X = adata.X.astype(np.float64)
    sc.pp.normalize_total(adata, target_sum=10000)

    sc.tl.score_genes(adata, gene_list=adata.var_names[:100], score_name="Test_cpu")
    rsc.tl.score_genes(adata, gene_list=adata.var_names[:100], score_name="Test_gpu")
    np.testing.assert_allclose(
        adata.obs["Test_cpu"].to_numpy(), adata.obs["Test_gpu"].to_numpy()
    )


def test_add_score():
    """
    check the dtype of the scores
    check that non-existing genes get ignored
    """
    adata = _create_adata(100, 1000, p_zero=0, p_nan=0).copy()

    sc.pp.normalize_total(adata, target_sum=1e4)
    sc.pp.log1p(adata)

    # the actual genes names are all 6letters
    # create some non-estinsting names with 7 letters:
    non_existing_genes = _create_random_gene_names(n_genes=3, name_length=7)
    some_genes = np.r_[
        np.unique(np.random.choice(adata.var_names, 10)), np.unique(non_existing_genes)
    ]
    rsc.tl.score_genes(adata, some_genes, score_name="Test")
    assert adata.obs["Test"].dtype == "float64"


@pytest.mark.parametrize("array_type", ["csr", "csc"])
def test_sparse_nanmean(array_type):
    """Needs to be fixed"""
    from rapids_singlecell.tools._utils import _nan_mean

    R, C = 100, 50

    # sparse matrix with nan
    S = _create_sparse_nan_matrix(R, C, percent_zero=0.3, percent_nan=0.3)
    S = rsc.get.X_to_GPU(S)
    S = S.astype(cp.float64)
    A = S.toarray()
    if array_type == "csc":
        S = S.tocsc()
    cp.testing.assert_allclose(_nan_mean(A, 1).ravel(), (_nan_mean(S, 1)).ravel())
    cp.testing.assert_allclose(_nan_mean(A, 0).ravel(), (_nan_mean(S, 0)).ravel())

    # sparse matrix, no NaN
    S = _create_sparse_nan_matrix(R, C, percent_zero=0.3, percent_nan=0)
    S = rsc.get.X_to_GPU(S)
    S = S.astype(cp.float64)
    A = S.toarray()
    if array_type == "csc":
        S = S.tocsc()
    # col/col sum
    cp.testing.assert_allclose(cp.mean(A, 0).ravel(), (_nan_mean(S, 0)).ravel())
    cp.testing.assert_allclose(cp.mean(A, 1).ravel(), (_nan_mean(S, 1)).ravel())


def test_score_genes_sparse_vs_dense():
    """
    score_genes() should give the same result for dense and sparse matrices
    """
    adata_sparse = _create_adata(100, 1000, p_zero=0.3, p_nan=0.3)

    adata_dense = adata_sparse.copy()
    adata_dense.X = adata_dense.X.toarray()

    gene_set = adata_dense.var_names[:10]

    rsc.tl.score_genes(adata_sparse, gene_list=gene_set, score_name="Test")
    rsc.tl.score_genes(adata_dense, gene_list=gene_set, score_name="Test")

    np.testing.assert_allclose(
        adata_sparse.obs["Test"].values, adata_dense.obs["Test"].values
    )


def test_score_genes_deplete():
    """
    deplete some cells from a set of genes.
    their score should be <0 since the sum of markers is 0 and
    the sum of random genes is >=0

    check that for both sparse and dense matrices
    """
    adata_sparse = _create_adata(100, 1000, p_zero=0.3, p_nan=0.3)

    adata_dense = adata_sparse.copy()
    adata_dense.X = adata_dense.X.toarray()

    # here's an arbitrary gene set
    gene_set = adata_dense.var_names[:10]

    gene_idx = np.where(adata_dense.var_names.isin(gene_set))[0]
    for adata in [adata_sparse, adata_dense]:
        # deplete these genes in 50 cells,
        ix_obs = np.random.choice(adata.shape[0], 50)
        adata.X[np.ix_(ix_obs, gene_idx)] = 0

        rsc.tl.score_genes(adata, gene_list=gene_set, score_name="Test")
        scores = adata.obs["Test"].values

        np.testing.assert_array_less(scores[ix_obs], 0)


def test_missing_genes():
    adata = _create_adata(100, 1000, p_zero=0, p_nan=0)
    # These genes have a different length of name
    non_extant_genes = _create_random_gene_names(n_genes=3, name_length=7)

    with pytest.raises(ValueError):
        rsc.tl.score_genes(adata, non_extant_genes)


def test_one_gene():
    # https://github.com/scverse/scanpy/issues/1395
    adata = _create_adata(100, 1000, p_zero=0, p_nan=0)
    rsc.tl.score_genes(adata, [adata.var_names[0]])


def test_use_raw_None():
    adata = _create_adata(100, 1000, p_zero=0, p_nan=0)
    adata_raw = adata.copy()
    adata_raw.var_names = [str(i) for i in range(adata_raw.n_vars)]
    adata.raw = adata_raw

    rsc.tl.score_genes(adata, adata_raw.var_names[:3], use_raw=None)


def test_layer():
    adata = _create_adata(100, 1000, p_zero=0, p_nan=0)

    sc.pp.normalize_total(adata, target_sum=1e4)
    sc.pp.log1p(adata)

    # score X
    gene_set = adata.var_names[:10]
    rsc.tl.score_genes(adata, gene_set, score_name="X_score")
    # score layer (`del` makes sure it actually uses the layer)
    adata.layers["test"] = adata.X.copy()
    adata.raw = adata

    del adata.X
    rsc.tl.score_genes(adata, gene_set, score_name="test_score", layer="test")

    np.testing.assert_array_equal(adata.obs["X_score"], adata.obs["test_score"])


@pytest.mark.parametrize("gene_pool", [[], ["foo", "bar"]])
def test_invalid_gene_pool(gene_pool):
    adata = _create_adata(100, 1000, p_zero=0, p_nan=0)

    with pytest.raises(ValueError, match="reference set"):
        rsc.tl.score_genes(adata, adata.var_names[:3], gene_pool=gene_pool)


def test_no_control_gene():
    np.random.seed(0)
    adata = _create_adata(100, 1, p_zero=0, p_nan=0)

    with pytest.raises(RuntimeError, match="No control genes found"):
        rsc.tl.score_genes(adata, adata.var_names[:1], ctrl_size=1)


@pytest.mark.parametrize("ctrl_as_ref", [True, False])
def test_gene_list_is_control(ctrl_as_ref: bool):
    np.random.seed(0)
    adata = sc.datasets.blobs(n_variables=10, n_observations=100, n_centers=20)
    with (
        pytest.raises(RuntimeError, match=r"No control genes found in any cut")
        if ctrl_as_ref
        else nullcontext()
    ):
        rsc.tl.score_genes(
            adata, gene_list="3", ctrl_size=1, n_bins=5, ctrl_as_ref=ctrl_as_ref
        )
