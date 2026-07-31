from __future__ import annotations

import itertools

import cupy as cp
import numpy as np
import pytest
import scanpy.external as sce
from anndata import AnnData
from bbknn.matrix import trimming as trimming_cpu
from scanpy.datasets import pbmc68k_reduced
from scanpy.neighbors._connectivity import gauss as scanpy_gauss
from scanpy.neighbors._connectivity import jaccard as scanpy_jaccard
from scipy import sparse as sc_sparse

from rapids_singlecell.get import X_to_GPU
from rapids_singlecell.pp import bbknn, neighbors
from rapids_singlecell.preprocessing._neighbors._helper import _trimming as trimming_gpu
from rapids_singlecell.preprocessing._neighbors._neighbors import (
    _get_connectivities_gauss,
    _get_connectivities_jaccard,
)

# the input data
X = np.array([[1, 0], [3, 0], [5, 6], [0, 4]])

distances_euclidean = [
    [0.0, 2.0, 0.0, 4.123105525970459],
    [2.0, 0.0, 0.0, 5.0],
    [0.0, 6.324555397033691, 0.0, 5.385164737701416],
    [4.123105525970459, 5.0, 0.0, 0.0],
]

connectivities_umap = [
    [0.0, 1.0, 0.0, 1.0],
    [1.0, 0.0, 0.5849691143165735, 0.8277419907567016],
    [0.0, 0.5849691143165735, 0.0, 1.0],
    [1.0, 0.8277419907567016, 1.0, 0.0],
]


@pytest.mark.parametrize("algo", ["brute", "ivfflat", "cagra"])
def test_umap_connectivities_euclidean(algo):
    adata = AnnData(X=X)
    neighbors(adata, n_neighbors=3, algorithm=algo)
    assert np.allclose(adata.obsp["distances"].toarray(), distances_euclidean)
    assert np.allclose(adata.obsp["connectivities"].toarray(), connectivities_umap)


@pytest.mark.parametrize("algo", ["brute", "ivfflat", "cagra", "ivfpq", "nn_descent"])
def test_algo(algo):
    adata = pbmc68k_reduced()
    neighbors(adata, n_neighbors=5, algorithm=algo)


def test_nn_descent_intermediate_graph_degree():
    adata = pbmc68k_reduced()
    neighbors(
        adata,
        n_neighbors=5,
        algorithm="nn_descent",
        algorithm_kwds={"intermediate_graph_degree": 10},
    )


@pytest.mark.parametrize("algo", ["ivfflat", "ivfpq"])
def test_ivf_algorithm_kwds(algo):
    adata = pbmc68k_reduced()
    neighbors(
        adata,
        n_neighbors=5,
        algorithm=algo,
        algorithm_kwds={"n_lists": 10, "n_probes": 10},
    )


@pytest.mark.parametrize("algo", ["nn_descent", "ivfpq"])
def test_indices_approx_nn(algo):
    adata = pbmc68k_reduced()
    brute_data = adata.copy()
    neighbors(adata, n_neighbors=5, algorithm=algo)
    neighbors(brute_data, n_neighbors=5, algorithm="brute")
    counter = 0
    for (rsc_start, rsc_stop), (b_start, b_stop) in zip(
        itertools.pairwise(adata.obsp["distances"].indptr),
        itertools.pairwise(brute_data.obsp["distances"].indptr),
    ):
        counter += len(
            np.intersect1d(
                adata.obsp["distances"].indices[rsc_start:rsc_stop],
                brute_data.obsp["distances"].indices[b_start:b_stop],
            )
        )
    assert counter / b_stop > 0.9


KEY = "test"


def test_neighbors_key_added():
    adata = pbmc68k_reduced()
    del adata.obsp
    neighbors(adata, n_neighbors=5, random_state=0)
    neighbors(adata, n_neighbors=5, random_state=0, key_added=KEY)

    conns_key = adata.uns[KEY]["connectivities_key"]
    dists_key = adata.uns[KEY]["distances_key"]

    assert adata.uns["neighbors"]["params"] == adata.uns[KEY]["params"]
    assert np.allclose(
        adata.obsp["connectivities"].toarray(), adata.obsp[conns_key].toarray()
    )
    assert np.allclose(
        adata.obsp["distances"].toarray(), adata.obsp[dists_key].toarray()
    )


def test_bbknn():
    """
    Test the bbknn function against the scanpy implementation. We want more than a 90% overlap between the two.
    This is calculated by the number of shared indices between the two sparse distance matrices divided by the number of indices in the original implementation.
    """

    adata = pbmc68k_reduced()
    sce.pp.bbknn(
        adata,
        n_pcs=15,
        batch_key="phase",
        computation="pynndescent",
        metric="euclidean",
    )
    bbknn(adata, n_pcs=15, batch_key="phase", algorithm="brute", key_added="rapids")
    counter = 0
    for (rsc_start, rsc_stop), (b_start, b_stop) in zip(
        itertools.pairwise(adata.obsp["rapids_distances"].indptr),
        itertools.pairwise(adata.obsp["distances"].indptr),
    ):
        counter += len(
            np.intersect1d(
                adata.obsp["rapids_distances"].indices[rsc_start:rsc_stop],
                adata.obsp["distances"].indices[b_start:b_stop],
            )
        )
    assert counter / b_stop > 0.9


def test_bbknn_distances_sorted_per_row():
    # fuzzy_simplicial_set uses the first non-zero distance per row as rho;
    # unsorted per-batch columns break sigma estimation and collapse weights.
    adata = pbmc68k_reduced()
    bbknn(adata, n_pcs=15, batch_key="phase", algorithm="brute")
    dists = adata.obsp["distances"]
    for start, stop in itertools.pairwise(dists.indptr):
        row = dists.data[start:stop]
        assert np.all(np.diff(row) >= 0), "bbknn distance rows must be sorted ascending"


def test_bbknn_connectivities_not_collapsed():
    # Regression: before the per-row sort fix, mean connectivity on this
    # dataset was ~0.85 with most weights pinned near 1.0. With sorted input
    # the distribution spreads out properly.
    adata = pbmc68k_reduced()
    bbknn(adata, n_pcs=15, batch_key="phase", algorithm="brute")
    weights = adata.obsp["connectivities"].data
    # Regression bounds chosen vs pre-fix behaviour (mean ~0.60, >0.99 frac ~0.28
    # on this dataset). Anything close to those values indicates sigma estimation
    # has broken again. Healthy distribution on the same dataset is mean ~0.50.
    assert weights.mean() < 0.7
    assert (weights > 0.99).mean() < 0.5


def test_bbknn_trim_default_matches_upstream():
    # bbknn upstream defaults trim = 10 * total_neighbors
    # (= 10 * neighbors_within_batch * n_batches).
    adata = pbmc68k_reduced()
    n_batches = adata.obs["phase"].nunique()
    neighbors_within_batch = 3
    bbknn(
        adata,
        n_pcs=15,
        batch_key="phase",
        algorithm="brute",
        neighbors_within_batch=neighbors_within_batch,
    )
    assert (
        adata.uns["neighbors"]["params"]["trim"]
        == 10 * neighbors_within_batch * n_batches
    )


@pytest.mark.parametrize("trim", [5, 240])
def test_trimming(trim):
    # trim=5: typical case.
    # trim=240: exercises the kernel's adaptive block-size path. A static
    # BLOCK_SIZE=64 would request 60 KB of dynamic shared memory and fail to
    # launch (default per-block cap is ~48 KB).
    adata = pbmc68k_reduced()
    cnts_gpu = X_to_GPU(adata.obsp["connectivities"]).astype(np.float32)
    cnts_cpu = adata.obsp["connectivities"].astype(np.float32)

    cnts_cpu = trimming_cpu(cnts_cpu, trim)
    cnts_gpu = trimming_gpu(cnts_gpu, trim)

    cp.testing.assert_array_equal(cnts_cpu.data, cnts_gpu.data)
    cp.testing.assert_array_equal(cnts_cpu.indices, cnts_gpu.indices)
    cp.testing.assert_array_equal(cnts_cpu.indptr, cnts_gpu.indptr)


@pytest.mark.parametrize("trim", [5, 50, 240])
@pytest.mark.parametrize("kernel", ["thread", "sorted"])
def test_trimming_kernels_agree(trim, kernel):
    # Both trim kernels must produce identical results to the CPU reference
    # (bbknn.matrix.trimming) on the same input. The "thread" kernel keeps a
    # per-thread top-k in shared memory; the "sorted" kernel does one block
    # per row with BlockRadixSort.
    adata = pbmc68k_reduced()
    cnts_gpu = X_to_GPU(adata.obsp["connectivities"]).astype(np.float32)
    cnts_cpu = adata.obsp["connectivities"].astype(np.float32)

    cnts_cpu = trimming_cpu(cnts_cpu, trim)
    cnts_gpu = trimming_gpu(cnts_gpu, trim, kernel=kernel)

    cp.testing.assert_array_equal(cnts_cpu.data, cnts_gpu.data)
    cp.testing.assert_array_equal(cnts_cpu.indices, cnts_gpu.indices)
    cp.testing.assert_array_equal(cnts_cpu.indptr, cnts_gpu.indptr)


class TestConnectivityMethods:
    """Tests for gauss and jaccard connectivity methods."""

    @pytest.fixture
    def knn_data(self):
        """Compute KNN and return indices/distances on both CPU and GPU."""
        from rapids_singlecell.preprocessing._neighbors._algorithms._brute import (
            _brute_knn,
        )
        from rapids_singlecell.preprocessing._neighbors._helper import (
            _check_neighbors_X,
            _fix_self_distances,
        )
        from rapids_singlecell.tools._utils import _choose_representation

        adata = pbmc68k_reduced()
        n_neighbors = 15
        X = _choose_representation(adata, use_rep=None, n_pcs=None)
        X_contiguous = _check_neighbors_X(X, "brute")
        knn_indices, knn_dist = _brute_knn(
            X_contiguous,
            X_contiguous,
            k=n_neighbors,
            metric="euclidean",
            metric_kwds={},
            algorithm_kwds={},
        )
        knn_dist = _fix_self_distances(knn_dist, "euclidean")
        n_obs = adata.shape[0]

        return {
            "knn_indices_cpu": knn_indices.get(),
            "knn_dist_cpu": knn_dist.get(),
            "knn_indices_gpu": knn_indices,
            "knn_dist_gpu": knn_dist,
            "n_obs": n_obs,
            "n_neighbors": n_neighbors,
        }

    def test_gauss_shapes(self, knn_data):
        W = _get_connectivities_gauss(
            knn_indices=knn_data["knn_indices_gpu"],
            knn_dist=knn_data["knn_dist_gpu"],
            n_obs=knn_data["n_obs"],
        )
        n = knn_data["n_obs"]
        assert W.shape == (n, n)
        # Symmetric
        diff = W - W.T
        assert abs(diff).max() < 1e-6
        # No self-loops (diagonal should be zero)
        diag = W.diagonal()
        assert cp.allclose(diag, 0.0)

    def test_gauss_matches_scanpy(self, knn_data):
        # GPU
        W_gpu = _get_connectivities_gauss(
            knn_indices=knn_data["knn_indices_gpu"],
            knn_dist=knn_data["knn_dist_gpu"],
            n_obs=knn_data["n_obs"],
        )
        W_gpu = W_gpu.get()

        # Scanpy: build sparse distance matrix from KNN arrays
        knn_indices = knn_data["knn_indices_cpu"]
        knn_dist = knn_data["knn_dist_cpu"]
        n_obs = knn_data["n_obs"]
        n_neighbors = knn_data["n_neighbors"]

        indptr = np.arange(0, n_obs * n_neighbors + 1, n_neighbors)
        dist_sparse = sc_sparse.csr_matrix(
            (knn_dist.ravel().astype(np.float64), knn_indices.ravel(), indptr),
            shape=(n_obs, n_obs),
        )
        dist_sparse.eliminate_zeros()
        W_sc = scanpy_gauss(dist_sparse, n_neighbors, knn=True)

        # Compare off-diagonal entries (scanpy includes self-loops on diagonal)
        W_gpu_dense = W_gpu.toarray()
        W_sc_dense = W_sc.toarray()
        np.fill_diagonal(W_sc_dense, 0.0)
        np.testing.assert_allclose(W_gpu_dense, W_sc_dense, atol=1e-5)

    def test_jaccard_shapes(self, knn_data):
        W = _get_connectivities_jaccard(
            knn_indices=knn_data["knn_indices_gpu"],
            n_obs=knn_data["n_obs"],
            n_neighbors=knn_data["n_neighbors"],
        )
        n = knn_data["n_obs"]
        assert W.shape == (n, n)
        # Symmetric
        diff = W - W.T
        assert abs(diff).max() < 1e-6

    def test_jaccard_matches_scanpy(self, knn_data):
        # GPU
        W_gpu = _get_connectivities_jaccard(
            knn_indices=knn_data["knn_indices_gpu"],
            n_obs=knn_data["n_obs"],
            n_neighbors=knn_data["n_neighbors"],
        )
        W_gpu = W_gpu.get()

        # Scanpy
        W_sc = scanpy_jaccard(
            knn_data["knn_indices_cpu"],
            n_obs=knn_data["n_obs"],
            n_neighbors=knn_data["n_neighbors"],
        )

        W_gpu_dense = W_gpu.toarray()
        W_sc_dense = W_sc.toarray()
        np.testing.assert_allclose(W_gpu_dense, W_sc_dense, atol=1e-6)

    @pytest.mark.parametrize("method", ["gauss", "jaccard"])
    def test_method_parameter(self, method):
        adata = pbmc68k_reduced()
        neighbors(adata, n_neighbors=15, method=method)
        assert "connectivities" in adata.obsp
        assert "distances" in adata.obsp
        conn = sc_sparse.csr_matrix(adata.obsp["connectivities"])
        assert conn.shape == (700, 700)
        assert conn.nnz > 0
