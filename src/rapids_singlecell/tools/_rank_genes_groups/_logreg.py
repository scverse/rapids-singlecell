from __future__ import annotations

from typing import TYPE_CHECKING

import cupy as cp
import numpy as np

from rapids_singlecell._compat import DaskArray, _meta_dense

if TYPE_CHECKING:
    from numpy.typing import NDArray

    from ._core import _RankGenes


def logreg(rg: _RankGenes, **kwds) -> list[tuple[int, NDArray, None]]:
    """Compute logistic regression scores."""
    if len(rg.groups_order) == 1:
        msg = "Cannot perform logistic regression on a single cluster."
        raise ValueError(msg)

    n_groups = len(rg.groups_order)
    selected = rg.group_codes < n_groups
    X = rg.X[selected, :]
    codes = rg.group_codes[selected]

    # Encode the multinomial class labels in canonical (original category) order
    # rather than in `groups_order` order. groups_order echoes the user's
    # `groups=` argument (see _select_groups), but cuML's softmax solver is not
    # invariant to a class-index permutation, so without this the fitted scores
    # would depend on the order groups are listed in. canon_label[i] is the
    # class index used for groups_order[i]; coef_ rows are mapped back below.
    cat_order = {str(c): i for i, c in enumerate(rg.labels.cat.categories)}
    canon_key = np.array([cat_order[str(g)] for g in rg.groups_order])
    canon_label = np.empty(n_groups, dtype=np.int64)
    canon_label[np.argsort(canon_key, kind="stable")] = np.arange(n_groups)
    relabel = cp.asarray(canon_label) if isinstance(codes, cp.ndarray) else canon_label
    grouping_logreg = relabel[codes].astype(X.dtype)

    if isinstance(X, DaskArray):
        import dask.array as da
        from cuml.dask.linear_model import LogisticRegression

        grouping_logreg = da.from_array(
            grouping_logreg,
            chunks=(X.chunks[0]),
            meta=_meta_dense(grouping_logreg.dtype),
        )
    else:
        from cuml.linear_model import LogisticRegression

    clf = LogisticRegression(**kwds)
    clf.fit(X, grouping_logreg)
    scores_all = cp.array(clf.coef_)
    if n_groups == scores_all.shape[1]:
        scores_all = scores_all.T

    results: list[tuple[int, NDArray, None]] = []
    for igroup in range(n_groups):
        if n_groups <= 2:
            scores = scores_all[0].get()
        else:
            # coef_ rows are in canonical class order; map back to groups_order.
            scores = scores_all[int(canon_label[igroup])].get()

        results.append((igroup, scores, None))

        if n_groups <= 2:
            break

    return results
