from __future__ import annotations

import cupy as cp
import numpy as np

from rapids_singlecell._cuda import _wilcoxon_cuda as _wc

from ._wilcoxon import _finish_wilcoxon, _log1p_scale, _logfoldchanges_from_means


def rank_panel(values, codes, indptr, rg, options):
    """Rank one full-height gene panel, including cells excluded from the target groups."""
    sizes = np.asarray(rg.group_sizes)
    outside = rg.n_obs - int(sizes.sum(dtype=np.int64))
    all_sizes = np.append(sizes, outside) if outside else sizes
    ref = rg.ireference
    shape = (len(indptr) - 1, len(all_sizes))
    if _wc is None or outside < 0 or (shape[0] - 1).bit_length() + max(1, (shape[1] - 1).bit_length()) > 32:
        return None
    sums = cp.empty(shape, cp.float64)
    ranks = cp.empty_like(sums)
    ties = cp.empty(shape[0] if ref is None else shape, cp.float64)
    scale = 0.0 if rg.mean_in_log_space else _log1p_scale(rg)
    if not _wc.rank_panel(
        values,
        codes,
        cp.asarray(indptr, cp.int32),
        cp.asarray(all_sizes, cp.int32),
        sums,
        ranks,
        ties,
        rg.n_obs if ref is None else int(sizes[ref]),
        -1 if ref is None else int(ref),
        options["tie_correct"],
        scale,
        cp.cuda.get_current_stream().ptr,
    ):
        return None

    groups = np.arange(len(sizes), dtype=np.intp)
    targets = slice(len(sizes)) if ref is None else groups[groups != ref]
    n = cp.asarray(sizes[targets], cp.float64)[:, None]
    other = rg.n_obs - n if ref is None else int(sizes[ref])
    selected = sums[:, targets].T
    reference = (sums.sum(axis=1) - selected) / other if ref is None else sums[:, ref] / other
    variance = (ties if ref is None else ties[:, targets].T) * n * other * ((n + other + 1) / 12)
    return _finish_wilcoxon(
        ranks[:, targets].T,
        n * (n + other + 1) / 2,
        variance,
        n[:, 0],
        groups[targets],
        use_continuity=options["use_continuity"],
        return_u_values=options["return_u_values"],
        logfoldchanges_gpu=_logfoldchanges_from_means(rg, selected / n, reference),
    )
