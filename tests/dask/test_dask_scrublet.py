from __future__ import annotations

import cupy as cp
import numpy as np
import pytest
from cupyx.scipy import sparse as cusparse
from scanpy.datasets import paul15, pbmc3k

import rapids_singlecell as rsc
from rapids_singlecell._testing import (
    as_dense_cupy_dask_array,
    as_sparse_cupy_dask_array,
)


@pytest.mark.parametrize("data_kind", ["sparse", "dense"])
def test_dask_scrublet(data_kind):
    if data_kind == "sparse":
        adata_1 = pbmc3k()[200:400].copy()
        adata_2 = pbmc3k()[200:400].copy()
        adata_2.X = cusparse.csr_matrix(adata_2.X.astype(np.float64))
        adata_1.X = as_sparse_cupy_dask_array(adata_1.X.astype(np.float64))
    elif data_kind == "dense":
        adata_1 = paul15()[200:400].copy()
        adata_2 = paul15()[200:400].copy()
        adata_2.X = cp.array(adata_2.X.astype(np.float64))
        adata_1.X = as_dense_cupy_dask_array(adata_1.X.astype(np.float64))
    else:
        raise ValueError(f"Unknown data_kind {data_kind}")

    batch = np.random.randint(0, 2, size=adata_1.shape[0])
    adata_1.obs["batch"] = batch
    adata_2.obs["batch"] = batch
    rsc.pp.scrublet(adata_1, batch_key="batch", verbose=False)

    # sort adata_2 to compare results
    batch_codes = adata_2.obs["batch"].astype("category").cat.codes
    order = np.argsort(batch_codes)
    adata_2 = adata_2[order]

    rsc.pp.scrublet(adata_2, batch_key="batch", verbose=False)
    adata_2 = adata_2[np.argsort(order)]

    np.testing.assert_allclose(
        adata_1.obs["doublet_score"], adata_2.obs["doublet_score"]
    )
    np.testing.assert_array_equal(
        adata_1.obs["predicted_doublet"], adata_2.obs["predicted_doublet"]
    )
