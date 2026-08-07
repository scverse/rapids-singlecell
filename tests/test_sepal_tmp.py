"""Tests for sepal GPU implementation."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from anndata import AnnData
from scipy import sparse

import rapids_singlecell as rsc


@pytest.fixture
def synthetic_spatial_data():
    """Create synthetic spatial data for testing sepal."""
    # Create a small 3x3 grid (9 cells)
    n_cells = 9
    n_genes = 5

    # Spatial coordinates (3x3 grid)
    spatial_coords = np.array(
        [[0, 0], [1, 0], [2, 0], [0, 1], [1, 1], [2, 1], [0, 2], [1, 2], [2, 2]],
        dtype=np.float32,
    )

    # Create connectivity matrix (4-neighbors for rectangular grid)
    connectivity = np.zeros((n_cells, n_cells), dtype=np.float32)
    for i in range(n_cells):
        row, col = i // 3, i % 3
        # Add connections to neighbors
        for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
            nr, nc = row + dr, col + dc
            if 0 <= nr < 3 and 0 <= nc < 3:
                j = nr * 3 + nc
                connectivity[i, j] = 1.0

    # Create expression data with some spatial pattern
    expression = np.random.poisson(5, (n_cells, n_genes)).astype(np.float32)
    # Add spatial gradient to first gene
    expression[:, 0] += np.arange(n_cells) * 2

    # Gene names
    gene_names = [f"Gene_{i}" for i in range(n_genes)]

    adata = AnnData(X=sparse.csr_matrix(expression), obsm={"spatial": spatial_coords})
    adata.var_names = gene_names
    adata.obsp["spatial_connectivities"] = sparse.csr_matrix(connectivity)

    return adata


@pytest.fixture
def synthetic_hex_data():
    """Create synthetic hexagonal grid data for testing sepal."""
    # Create a small hexagonal grid (7 cells in center + 6 neighbors)
    n_cells = 7
    n_genes = 3

    # Spatial coordinates (hexagonal pattern)
    spatial_coords = np.array(
        [
            [0, 0],  # center
            [1, 0],  # right
            [0.5, 0.866],
            [0.5, -0.866],  # top, bottom
            [-0.5, 0.866],
            [-0.5, -0.866],  # top-left, bottom-left
            [-1, 0],  # left
        ],
        dtype=np.float32,
    )

    # Create connectivity matrix (6-neighbors for hexagonal grid)
    connectivity = np.zeros((n_cells, n_cells), dtype=np.float32)
    # Center connects to all others
    for i in range(1, n_cells):
        connectivity[0, i] = 1.0
        connectivity[i, 0] = 1.0

    # Create expression data
    expression = np.random.poisson(3, (n_cells, n_genes)).astype(np.float32)
    gene_names = [f"HexGene_{i}" for i in range(n_genes)]

    adata = AnnData(X=sparse.csr_matrix(expression), obsm={"spatial": spatial_coords})
    adata.var_names = gene_names
    adata.obsp["spatial_connectivities"] = sparse.csr_matrix(connectivity)

    return adata


def test_sepal_rectangular_grid(synthetic_spatial_data):
    """Test sepal on rectangular grid (4-neighbors)."""
    adata = synthetic_spatial_data.copy()

    # Run sepal with small number of iterations for testing
    result = rsc.gr.sepal(
        adata,
        max_neighs=4,
        n_iter=100,  # Small number for testing
        copy=True,
    )

    # Check result type and shape
    assert isinstance(result, pd.DataFrame)
    assert result.shape == (5, 1)  # 5 genes, 1 score column
    assert "sepal_score" in result.columns

    # Check no NaN values
    assert not result["sepal_score"].isna().any()

    # Check scores are sorted descending
    assert result["sepal_score"].is_monotonic_decreasing

    # Check gene names match
    assert list(result.index) == [f"Gene_{i}" for i in range(5)]


def test_sepal_hexagonal_grid(synthetic_hex_data):
    """Test sepal on hexagonal grid (6-neighbors)."""
    adata = synthetic_hex_data.copy()

    # Run sepal with small number of iterations for testing
    result = rsc.gr.sepal(
        adata,
        max_neighs=6,
        n_iter=50,  # Small number for testing
        copy=True,
    )

    # Check result type and shape
    assert isinstance(result, pd.DataFrame)
    assert result.shape == (3, 1)  # 3 genes, 1 score column
    assert "sepal_score" in result.columns

    # Check no NaN values
    assert not result["sepal_score"].isna().any()

    # Check scores are sorted descending
    assert result["sepal_score"].is_monotonic_decreasing


def test_sepal_inplace_storage(synthetic_spatial_data):
    """Test sepal stores results in adata.uns when copy=False."""
    adata = synthetic_spatial_data.copy()

    # Run sepal in-place
    result = rsc.gr.sepal(adata, max_neighs=4, n_iter=50, copy=False)

    # Should return None
    assert result is None

    # Check results stored in adata.uns
    assert "sepal_score" in adata.uns
    stored_result = adata.uns["sepal_score"]

    # Check stored result
    assert isinstance(stored_result, pd.DataFrame)
    assert stored_result.shape == (5, 1)
    assert "sepal_score" in stored_result.columns


def test_sepal_gene_selection(synthetic_spatial_data):
    """Test sepal with specific gene selection."""
    adata = synthetic_spatial_data.copy()

    # Select only first 2 genes
    selected_genes = ["Gene_0", "Gene_1"]

    result = rsc.gr.sepal(
        adata, max_neighs=4, genes=selected_genes, n_iter=50, copy=True
    )

    # Check only selected genes are in result
    assert result.shape == (2, 1)
    assert list(result.index) == selected_genes


def test_sepal_validation_errors(synthetic_spatial_data):
    """Test sepal input validation."""
    adata = synthetic_spatial_data.copy()

    # Test invalid max_neighs
    with pytest.raises(
        ValueError, match="Expected `max_neighs` to be either `4` or `6`"
    ):
        rsc.gr.sepal(adata, max_neighs=5, copy=True)

    # Test missing connectivity
    adata.obsp.pop("spatial_connectivities")
    with pytest.raises(KeyError, match="Connectivity matrix"):
        rsc.gr.sepal(adata, max_neighs=4, copy=True)

    # Test missing spatial coordinates
    adata = synthetic_spatial_data.copy()
    adata.obsm.pop("spatial")
    with pytest.raises(KeyError, match="Spatial coordinates"):
        rsc.gr.sepal(adata, max_neighs=4, copy=True)


def test_sepal_connectivity_mismatch(synthetic_spatial_data):
    """Test sepal with connectivity that doesn't match max_neighs."""
    adata = synthetic_spatial_data.copy()

    # Modify connectivity to have 6 neighbors for some cells
    connectivity = adata.obsp["spatial_connectivities"].toarray()
    connectivity[0, 5] = 1.0  # Add extra connection
    connectivity[5, 0] = 1.0
    adata.obsp["spatial_connectivities"] = sparse.csr_matrix(connectivity)

    # Should raise error when max_neighs=4 but some cells have 5 neighbors
    with pytest.raises(
        ValueError, match="Expected `max_neighs=4`, found node with `5` neighbors"
    ):
        rsc.gr.sepal(adata, max_neighs=4, copy=True)
