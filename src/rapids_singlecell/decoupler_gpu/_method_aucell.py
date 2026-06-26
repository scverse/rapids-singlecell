from __future__ import annotations

import cupy as cp
import numpy as np

from rapids_singlecell._cuda import _aucell_cuda as _au
from rapids_singlecell.decoupler_gpu._helper._docs import docs
from rapids_singlecell.decoupler_gpu._helper._log import _log
from rapids_singlecell.decoupler_gpu._helper._Method import Method, MethodMeta
from rapids_singlecell.decoupler_gpu._helper._run import _run


def rank_rows_desc(x: cp.ndarray) -> cp.ndarray:
    order = cp.argsort(-x, axis=1)
    n = x.shape[1]
    ranks = cp.empty_like(order, dtype=cp.int32)
    row_idx = cp.arange(x.shape[0])[:, None]
    ranks[row_idx, order] = cp.arange(1, n + 1, dtype=cp.int32)
    return ranks


def _auc(row, cnct, *, starts, offsets, n_up, n_fsets, max_aucs):
    # Cast dtypes to what the kernel expects
    ranks = rank_rows_desc(row)

    max_aucs = max_aucs.astype(cp.float32, copy=False)

    R, C = ranks.shape
    es = cp.zeros((R, n_fsets), dtype=cp.float32)

    _au.auc(
        ranks,
        R=R,
        C=C,
        cnct=cnct,
        starts=starts,
        lens=offsets,
        n_sets=n_fsets,
        n_up=n_up,
        max_aucs=max_aucs,
        es=es,
        stream=cp.cuda.get_current_stream().ptr,
    )
    return es


def _validate_n_up(
    nvar: int,
    n_up: int | float | None = None,
) -> int:
    assert isinstance(n_up, int | float) or n_up is None, (
        "n_up must be numerical or None"
    )
    if n_up is None:
        n_up = np.ceil(0.05 * nvar)
        n_up = int(np.clip(n_up, a_min=2, a_max=nvar))
    else:
        n_up = int(np.ceil(n_up))
    assert nvar >= n_up > 1, (
        f"For nvar={nvar}, n_up={n_up} must be between 1 and {nvar}"
    )
    return n_up


@docs.dedent
def _func_aucell(
    mat: cp.ndarray,
    *,
    cnct: cp.ndarray,
    starts: cp.ndarray,
    offsets: cp.ndarray,
    n_up: int | float | None = None,
    verbose: bool = False,
) -> tuple[np.ndarray, None]:
    r"""
    Area Under the Curve for set enrichment within single cells (AUCell).

    Given a ranked list of features per observation, AUCell calculates the AUC by measuring how early the features in
    the set appear in this ranking. Specifically, the enrichment score :math:`ES` is:

    .. math::

       {ES}_{i, F} = \int_0^1 {RecoveryCurve}_{i, F}(r_i) \, dr

    Where:

    - :math:`i` is the observation
    - :math:`F` is the feature set
    - :math:`{RecoveryCurve}_{i, F}(r_i)` is the proportion of features from :math:`F` recovered in the top :math:`r_i`-fraction of the ranked list for observation :math:`i`

    %(notest)s

    %(params)s
    n_up
        Number of features to include in the AUC calculation.
        If ``None``, the top 5% of features based on their magnitude are selected.

    %(returns)s

    Example
    -------
    .. code-block:: python

        import decoupler as dc

        adata, net = dc.ds.toy()
        rsc.dcg.aucell(adata, net, tmin=3)
    """
    nobs, nvar = mat.shape
    nsrc = starts.size
    n_up = _validate_n_up(nvar, n_up)
    m = f"aucell - calculating {nsrc} AUCs for {nvar} targets across {nobs} observations, categorizing features at rank={n_up}"
    _log(m, level="info", verbose=verbose)
    k = cp.minimum(offsets, n_up - 1)
    max_aucs = (k * (k - 1) / 2 + (n_up - k) * k).astype(cp.int32)
    es = _auc(
        row=mat,
        cnct=cnct,
        starts=starts,
        offsets=offsets,
        n_up=n_up,
        n_fsets=nsrc,
        max_aucs=max_aucs,
    )
    return es.get(), None


class AucellMethod(Method):
    """Custom Method class for aucell with bsize=100 as default."""

    def __call__(
        self,
        data,
        net,
        *,
        tmin: int | float = 5,
        raw: bool = False,
        empty: bool = True,
        bsize: int | float = 100,  # Default batch size of 100
        verbose: bool = False,
        pre_load: bool = False,
        n_up: int | float | None = None,
        **kwargs,
    ):
        return _run(
            name=self.name,
            func=self.func,
            adj=self.adj,
            test=self.test,
            data=data,
            net=net,
            tmin=tmin,
            raw=raw,
            empty=empty,
            bsize=bsize,
            verbose=verbose,
            pre_load=pre_load,
            n_up=n_up,
            **kwargs,
        )


_aucell = MethodMeta(
    name="aucell",
    desc="AUCell",
    func=_func_aucell,
    stype="categorical",
    adj=False,
    weight=False,
    test=False,
    limits=(0, 1),
    reference="https://doi.org/10.1038/nmeth.4463",
)
aucell = AucellMethod(_method=_aucell)
