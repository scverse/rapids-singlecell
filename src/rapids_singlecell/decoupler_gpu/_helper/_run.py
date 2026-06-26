from __future__ import annotations

from typing import TYPE_CHECKING

import cupy as cp
import cupyx.scipy.sparse as csps
import numpy as np
import pandas as pd
import scipy.sparse as sps
from anndata import AnnData
from tqdm.auto import tqdm

from rapids_singlecell._compat import DaskArray
from rapids_singlecell.decoupler_gpu._helper._data import extract
from rapids_singlecell.decoupler_gpu._helper._log import _log
from rapids_singlecell.decoupler_gpu._helper._net import adjmat, idxmat, prune
from rapids_singlecell.decoupler_gpu._helper._pv import fdr_bh_axis1
from rapids_singlecell.preprocessing._utils import _sparse_to_dense

if TYPE_CHECKING:
    from collections.abc import Callable

    from rapids_singlecell.decoupler_gpu._helper._data import DataType


def _return(
    name: str,
    data: DataType,
    es: pd.DataFrame,
    pv: pd.DataFrame,
    *,
    verbose: bool = False,
) -> tuple[pd.DataFrame, pd.DataFrame] | AnnData | None:
    if isinstance(data, AnnData):
        if data.obs_names.size != es.index.size:
            m = "Provided AnnData contains empty observations, returning repaired object"
            _log(m, level="warn", verbose=verbose)
            data = data[es.index, :].copy()
            data.obsm[f"score_{name}"] = es
            if pv is not None:
                data.obsm[f"padj_{name}"] = pv
            return data
        else:
            data.obsm[f"score_{name}"] = es
            if pv is not None:
                data.obsm[f"padj_{name}"] = pv
            return None
    else:
        return es, pv


def _get_batch(mat, srt, end):
    if sps.issparse(mat):
        bmat = csps.csr_matrix(mat[srt:end])
        bmat = _sparse_to_dense(bmat)
    elif csps.issparse(mat):
        bmat = _sparse_to_dense(mat[srt:end])
    elif isinstance(mat, np.ndarray):
        bmat = cp.array(mat[srt:end, :])
    elif isinstance(mat, cp.ndarray):
        bmat = mat[srt:end, :]
    else:
        bmat, msk_col = mat
        bmat = bmat[srt:end, :]
        if sps.issparse(bmat):
            bmat = csps.csr_matrix(bmat)
            bmat = _sparse_to_dense(bmat)
        else:
            bmat = cp.array(bmat)
        bmat = bmat[:, msk_col]
    return bmat.astype(cp.float32)


def _mat_to_array(mat):
    if sps.issparse(mat):
        mat = csps.csr_matrix(mat)
        mat = _sparse_to_dense(mat)
    elif csps.issparse(mat):
        mat = _sparse_to_dense(mat)
    elif isinstance(mat, np.ndarray):
        mat = cp.array(mat)
    elif isinstance(mat, cp.ndarray):
        mat = mat
    else:
        raise ValueError(f"Unsupported matrix type: {type(mat)}")
    return mat.astype(cp.float32)


def _run_dask_adj(
    func: Callable,
    mat: DaskArray,
    adjm_cpu: np.ndarray,
    *,
    test: bool,
    verbose: bool = False,
    **kwargs,
) -> tuple[np.ndarray, np.ndarray | None]:
    """Dask execution for adj=True methods - each chunk = one batch."""
    import dask

    n_sources = adjm_cpu.shape[1]

    # Wrap adjm_cpu in delayed to avoid serializing it per-task
    adjm_delayed = dask.delayed(adjm_cpu, pure=True)

    def process_chunk(chunk, adjm, block_info=None):
        # Convert to GPU inside chunk
        adjm_gpu = cp.array(adjm, dtype=cp.float32)
        chunk_gpu = _mat_to_array(chunk) if not isinstance(chunk, cp.ndarray) else chunk
        es, pv = func(chunk_gpu, adjm_gpu, verbose=False, **kwargs)
        # Free GPU memory
        del chunk_gpu, adjm_gpu
        cp.get_default_memory_pool().free_all_blocks()
        if test and pv is not None:
            return np.hstack([es, pv])
        return es

    # Create delayed tasks for each row block
    # Rechunk to ensure all features are in one chunk per row block
    if len(mat.chunks[1]) > 1:
        mat = mat.rechunk({1: -1})

    tasks = []
    for i in range(len(mat.chunks[0])):
        block = mat.blocks[i, 0]
        task = dask.delayed(process_chunk)(block, adjm_delayed)
        tasks.append(task)

    _log("dask - computing batches", level="info", verbose=verbose)
    results = dask.compute(*tasks)
    computed = np.vstack(results)

    if test:
        return computed[:, :n_sources], computed[:, n_sources:]
    return computed, None


def _run_dask_idx(
    func: Callable,
    mat: DaskArray,
    cnct_cpu: np.ndarray,
    starts_cpu: np.ndarray,
    offsets_cpu: np.ndarray,
    *,
    verbose: bool = False,
    **kwargs,
) -> tuple[np.ndarray, None]:
    """Dask execution for adj=False methods (AUCell) - each chunk = one batch."""
    import dask

    # Wrap arrays in delayed to avoid serializing per-task
    cnct_delayed = dask.delayed(cnct_cpu, pure=True)
    starts_delayed = dask.delayed(starts_cpu, pure=True)
    offsets_delayed = dask.delayed(offsets_cpu, pure=True)

    def process_chunk(chunk, cnct, starts, offsets):
        # Convert to GPU inside chunk
        cnct_gpu = cp.array(cnct, dtype=cp.int32)
        starts_gpu = cp.array(starts, dtype=cp.int32)
        offsets_gpu = cp.array(offsets, dtype=cp.int32)
        chunk_gpu = _mat_to_array(chunk) if not isinstance(chunk, cp.ndarray) else chunk
        es, _ = func(
            chunk_gpu,
            cnct=cnct_gpu,
            starts=starts_gpu,
            offsets=offsets_gpu,
            verbose=False,
            **kwargs,
        )
        # Free GPU memory
        del chunk_gpu, cnct_gpu, starts_gpu, offsets_gpu
        cp.get_default_memory_pool().free_all_blocks()
        return es

    # Create delayed tasks for each row block
    # Rechunk to ensure all features are in one chunk per row block
    if len(mat.chunks[1]) > 1:
        mat = mat.rechunk({1: -1})

    tasks = []
    for i in range(len(mat.chunks[0])):
        block = mat.blocks[i, 0]
        task = dask.delayed(process_chunk)(
            block, cnct_delayed, starts_delayed, offsets_delayed
        )
        tasks.append(task)

    _log("dask - computing batches", level="info", verbose=verbose)
    results = dask.compute(*tasks)
    computed = np.vstack(results)
    return computed, None


def _run(
    name: str,
    func: Callable,
    *,
    adj: bool,
    test: bool,
    data: DataType,
    net: pd.DataFrame,
    tmin: int | float = 5,
    layer: str | None = None,
    raw: bool = False,
    empty: bool = True,
    bsize: int | float = 5000,
    verbose: bool = False,
    pre_load: bool = False,
    adj_pv_gpu: bool = False,
    **kwargs,
) -> tuple[pd.DataFrame, pd.DataFrame] | AnnData | None:
    _log(f"{name} - Running {name}", level="info", verbose=verbose)
    # Process data
    mat, obs, var = extract(
        data,
        layer=layer,
        raw=raw,
        empty=empty,
        verbose=verbose,
        bsize=bsize,
        pre_load=pre_load,
    )
    issparse = sps.issparse(mat) or csps.issparse(mat)
    isbacked = isinstance(mat, tuple)
    is_dask = isinstance(mat, DaskArray) or (
        isinstance(mat, tuple) and isinstance(mat[0], DaskArray)
    )
    # Process net
    net = prune(features=var, net=net, tmin=tmin, verbose=verbose)
    # Handle stat type
    if adj:
        sources, targets, adjm = adjmat(features=var, net=net, verbose=verbose)
        # Handle Dask input (auto-detect)
        if is_dask:
            _log(f"{name} - using Dask execution", level="info", verbose=verbose)
            adjm_cpu = adjm  # Keep as numpy for Dask path
            es, pv = _run_dask_adj(
                func, mat, adjm_cpu, test=test, verbose=verbose, **kwargs
            )
            es = pd.DataFrame(es, index=obs, columns=sources)
        # Handle batches for sparse/backed data
        elif issparse or isbacked:
            adjm = cp.array(adjm, dtype=cp.float32)
            nbatch = int(np.ceil(obs.size / bsize))
            es, pv = [], []
            for i in tqdm(range(nbatch), disable=not verbose):
                if i == 0 and verbose:
                    batch_verbose = True
                else:
                    batch_verbose = False
                srt, end = i * bsize, i * bsize + bsize
                bmat = _get_batch(mat, srt, end)
                bes, bpv = func(bmat, adjm, verbose=batch_verbose, **kwargs)
                es.append(bes)
                pv.append(bpv)
            es = np.vstack(es)
            es = pd.DataFrame(es, index=obs, columns=sources)
        else:
            adjm = cp.array(adjm, dtype=cp.float32)
            mat = _mat_to_array(mat)
            es, pv = func(mat, adjm, verbose=verbose, **kwargs)
            es = pd.DataFrame(es, index=obs, columns=sources)
    else:
        sources, cnct, starts, offsets = idxmat(features=var, net=net, verbose=verbose)
        # Handle Dask input (auto-detect)
        if is_dask:
            _log(f"{name} - using Dask execution", level="info", verbose=verbose)
            es, pv = _run_dask_idx(
                func, mat, cnct, starts, offsets, verbose=verbose, **kwargs
            )
            es = pd.DataFrame(es, index=obs, columns=sources)
        else:
            cnct = cp.array(cnct, dtype=cp.int32)
            starts = cp.array(starts, dtype=cp.int32)
            offsets = cp.array(offsets, dtype=cp.int32)
            nbatch = int(np.ceil(obs.size / bsize))
            es, pv = [], []
            for i in tqdm(range(nbatch), disable=not verbose):
                srt, end = i * bsize, i * bsize + bsize
                bmat = _get_batch(mat, srt, end)
                if i == 0 and verbose:
                    batch_verbose = True
                else:
                    batch_verbose = False
                bes, bpv = func(
                    bmat,
                    cnct=cnct,
                    starts=starts,
                    offsets=offsets,
                    verbose=batch_verbose,
                    **kwargs,
                )
                es.append(bes)
                pv.append(bpv)
            es = np.vstack(es)
            es = pd.DataFrame(es, index=obs, columns=sources)
    # Handle pvals and FDR correction
    if test:
        pv = np.vstack(pv)
        pv = pd.DataFrame(pv, index=obs, columns=sources)
        if name != "mlm":
            _log(f"{name} - adjusting p-values by FDR", level="info", verbose=verbose)
            pv.loc[:, :] = fdr_bh_axis1(pv.values, if_gpu=adj_pv_gpu)
    else:
        pv = None
    _log(f"{name} - done", level="info", verbose=verbose)
    return _return(name, data, es, pv, verbose=verbose)
