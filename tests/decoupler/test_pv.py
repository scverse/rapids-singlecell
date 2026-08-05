from __future__ import annotations

import numpy as np
import pytest
import scipy.stats as sts

import rapids_singlecell.decoupler_gpu as dc
from rapids_singlecell.decoupler_gpu._helper._pv import fdr_bh_axis1


@pytest.mark.parametrize("if_gpu", [True, False])
def test_func_mlm(
    adata,
    net,
    if_gpu,
):
    dc.mlm(data=adata, net=net, tmin=3)
    dc_pv_df = adata.obsm["padj_mlm"]
    dc_pv = dc_pv_df.values.astype(np.float64)
    adj = fdr_bh_axis1(dc_pv, if_gpu=if_gpu)
    np.testing.assert_allclose(
        adj, sts.false_discovery_control(dc_pv, axis=1, method="bh")
    )
