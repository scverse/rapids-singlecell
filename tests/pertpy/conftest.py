from __future__ import annotations

import anndata
import numpy as np
import pandas as pd
import pytest
from scipy import sparse

NUM_CELLS_PER_GROUP = 10
NUM_NOT_DE = 10
NUM_DE = 10


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
