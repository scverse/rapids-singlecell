from __future__ import annotations

import math
import warnings
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Literal

if TYPE_CHECKING:
    from collections.abc import Sequence

import cupy as cp
import cupyx.scipy.sparse as cpsp
import numpy as np
import pandas as pd

from rapids_singlecell._cuda import _edistance_cuda as _ed
from rapids_singlecell._utils import _concat_on_device, _copy_to_device, _split_pairs
from rapids_singlecell.squidpy_gpu._utils import _assert_categorical_obs

from ._base_metric import BaseMetric, _is_sparse, parse_device_ids

if TYPE_CHECKING:
    from anndata import AnnData


# Largest int32 indptr value; beyond this the CSR offsets need int64.
_INT32_MAX = np.iinfo(np.int32).max


# Per precision: kernel, A-tile rows, features per step, dtype of the norms, shared-memory element.
_MATH = {
    "float64": (_ed.pair_sums_f64, 64, 2, np.float64, np.float64),
    "float32": (_ed.pair_sums_f32, 128, 4, np.float32, np.float32),
    "float16": (_ed.pair_sums_f16, 128, 16, np.float32, np.float16),
}
_TILE_B = 128  # B-tile rows
_PAIR_CHUNK = 1 << 20  # pairs per kernel launch
_BLOCKS_PER_SM = 64  # blocks per multiprocessor a launch aims for
_SAMPLE_ROWS = 1 << 14  # rows the center is estimated from
_TRIM = 16.0  # sampled rows whose largest |value| is more than this many times the median row's are left out
_CAST_VALUES = 1 << 24  # values of cast two-array input held in float32 or float64 at a time, at most


def _float_dtype(dtype) -> np.dtype:
    """The dtype the kernels read data of dtype in: float32 or float64 in native byte order."""
    dtype = np.dtype(dtype)
    if dtype.kind not in "biuf":  # complex, strings, dates, objects
        raise ValueError(f"Unsupported dtype {dtype}")
    dtype = dtype.newbyteorder("=")
    return dtype if dtype in (np.float32, np.float64) else np.dtype(np.float32)


def _rows(a) -> cp.ndarray:
    """Dense rows on the device in float32 or float64; the kernels read them C-contiguous."""
    a = a.astype(_float_dtype(a.dtype), copy=False)  # converted where the array lives, then uploaded
    return cp.asarray(a, order="C")


def _device(a) -> cp.ndarray:
    """A two-array input on the device in the dtype it came in; raises unless that is boolean, integer or float."""
    a = cp.asarray(a)
    _float_dtype(a.dtype)
    return a


def _center(X, precision: str, rows=None) -> tuple[cp.ndarray, float]:
    """Shift ``nm`` and scale ``sc`` of the rows ``x * sc + nm`` the kernels compute on.

    The kernels get a distance from ``|a|² + |b|² - 2ab``, which needs centered
    rows to be accurate, and float16 needs them scaled into its range. The mean
    is taken over a sample of rows without its outlier rows; ``sc`` is the power
    of two (exact to undo) that puts the largest centered sample value in
    ``[2**10, 2**11)`` and ``nm = -mean * sc``. The sample is at most
    ``_SAMPLE_ROWS`` strided rows (2**25 values) of the cells a call reads, so
    a distance does not depend on other cells: the rows of ``X``, or those the
    int32 device array ``rows`` lists when a call reads part of a dense ``X``.

    The kernel reads the sampled rows where they are: C-contiguous float32 or
    float64 device rows, or a device CSR matrix with sorted columns (in its own
    index dtype), whose stored values give the ``nm`` and ``sc`` of the same
    dense rows.
    """
    n, d = X.shape[0] if rows is None else rows.size, X.shape[1]
    dtype = X.dtype
    step = max(1, -(-n // max(1, min(_SAMPLE_ROWS, 2**25 // max(d, 1)))))
    m = -(-n // step)
    stream = cp.cuda.get_current_stream()
    r, keep = cp.zeros(m, dtype=dtype), cp.zeros(m, dtype=bool)
    total, ext = cp.zeros(d, dtype=cp.float64), cp.zeros((2, d), dtype=dtype)
    if cpsp.issparse(X):
        reader = (*map(cp.ascontiguousarray, (X.indptr, X.indices, X.data)), step)
    else:
        reader = (X, rows, step)

    def run(what, shift=0):
        _ed.center(*reader, what, shift, keep, ext, r, total, stream=stream.ptr)
        stream.synchronize()  # the rows stay referenced until the kernel has finished

    run(0)  # r = the largest |value| of each row
    # Left out: rows with a non-finite value or above _TRIM times the median row (picked on the host)
    big = r.get()
    ref = big[(big > 0) & (big < np.inf)]
    kept = (big < np.inf) & (big / _TRIM <= (np.median(ref) if len(ref) else 0.0))
    keep.set(kept)
    n_kept = int(kept.sum())
    # The column sums add integers, exact in float64 in any order: dense and CSR input get the same mean
    shift = min(np.finfo(dtype).nmant, 52 - _SAMPLE_ROWS.bit_length())
    shift -= math.frexp(float(big[kept].max()))[1] if n_kept else 0
    run(1, shift)  # total = the column sums of rint(value * 2**shift) over the kept rows, ext = their column extremes
    total /= max(n_kept, 1)
    mu = cp.ldexp(total, -shift, out=total).astype(dtype, copy=False)
    top = cp.zeros(2, dtype=dtype)  # the largest |value - mu| of the kept rows: at a column's largest or smallest value
    if n_kept:
        _ed.center(ext, mu, 1, 0, 0, keep, ext, top, total, stream=stream.ptr)
    top = float(top.max())  # reading it waits for the kernel
    total = ext = r = keep = None  # not held while nm is built
    work = np.result_type(dtype, _MATH[precision][3])  # the kernels center in it: sc has to stay finite there
    sc = math.ldexp(1.0, min(10 - math.floor(math.log2(top)), np.finfo(work).maxexp - 2)) if 0 < top < math.inf else 1.0
    nm = mu.astype(work, copy=False)  # -mu * sc in place, with +0.0 for a zero mean
    nm *= sc
    cp.subtract(0.0, nm, out=nm)
    if precision == "float16":  # a zero then centers to an exact float16
        snap = nm.astype(np.float32).astype(np.float16)
        nm = cp.where(cp.isfinite(snap), snap.astype(nm.dtype), nm)
    return nm, sc


@dataclass(eq=False)
class _Source:
    """The cells on GPU as the edistance kernels read them.

    ``arrays`` is ``(X,)`` for C-contiguous dense rows or ``(indptr, indices, data)``
    for CSR with sorted columns (int32 ``indices``, int32 or int64 ``indptr``), in
    float32 or float64; only with ``duplicates`` (CSR that is not canonical) do
    the kernels sum the entries of a column. ``nm`` and ``sc`` come from
    :func:`_center`. The copies and row norms on each device depend only on the
    cells, so they are computed once and reused by every launch (e.g. each
    bootstrap iteration).
    """

    arrays: tuple[cp.ndarray, ...]
    shape: tuple[int, int]
    precision: str
    nm: cp.ndarray
    sc: float
    duplicates: bool = False
    _devices: dict[int, tuple] = field(default_factory=dict)

    @property
    def dtype(self) -> np.dtype:
        return self.arrays[-1].dtype

    def on_device(self, device_id: int) -> None:
        """Copy the cells to a device and compute their row norms there, once."""
        if device_id in self._devices:
            return
        fn, _, step, acc, elem = _MATH[self.precision]
        n, d = self.shape
        kp = -(-d // step) * step
        with cp.cuda.Device(device_id):
            arrays = tuple(_copy_to_device(a, device_id) for a in self.arrays)
            nm = _copy_to_device(self.nm, device_id)
            if len(arrays) == 1:
                at = arrays[0].data.ptr | (d * arrays[0].itemsize)
                args = (*arrays, min(16, at & -at), nm, self.sc, kp)
            else:  # the row of an all-zero cell: nm in the shared-memory element, nm itself where that is its dtype
                zero = nm.astype(acc, copy=False).astype(elem, copy=False)
                zero = zero.view(cp.uint16) if elem is np.float16 else zero
                args = (*arrays, nm, self.sc, zero, self.duplicates, kp)
            norms, unused = cp.full(n, -1, dtype=acc), cp.zeros(1, dtype=cp.int32)
            stream = cp.cuda.get_current_stream()
            fn(*args, norms, *[unused] * 5, 0, cp.zeros(1), norms_only=True, stream=stream.ptr)
            stream.synchronize()
            if self.precision == "float16" and n and float(norms[0]) < 0:  # the kernels did not run
                raise ValueError("precision='float16' needs compute capability 7.5 or newer; use precision='float32'.")
            # Beyond the float16 range a norm is inf; their sum tells without a warning for NaN cells
            overflow = self.precision == "float16" and bool(cp.isinf(cp.nansum(norms)))
            n_sm = cp.cuda.runtime.deviceGetAttribute(cp.cuda.runtime.cudaDevAttrMultiProcessorCount, device_id)
        self._devices[device_id] = args, norms, n_sm, overflow

    def pair_sums(
        self,
        device_id: int,
        cat_offsets: cp.ndarray,
        cell_indices: cp.ndarray,
        pair_left: cp.ndarray,
        pair_right: cp.ndarray,
    ) -> cp.ndarray:
        """Distance sums of the pairs on one device, at most ``_PAIR_CHUNK`` pairs per kernel launch."""
        fn, tile_a = _MATH[self.precision][:2]
        args, norms, n_sm, overflow = self._devices[device_id]
        with cp.cuda.Device(device_id):
            stream = cp.cuda.get_current_stream()
            # Only the cells of this call count: their norms are read 2**16 at a time
            if overflow and any(
                bool(cp.isinf(cp.nansum(norms[cell_indices[i : i + (1 << 16)]])))
                for i in range(0, cell_indices.size, 1 << 16)
            ):
                raise ValueError("The centered and scaled data exceeds the float16 range; use precision='float32'.")
            groups = (norms, cat_offsets, cell_indices)
            sizes = cp.diff(cat_offsets).astype(cp.int64)
            out = cp.empty(pair_left.size, dtype=self.dtype)
            for start in range(0, pair_left.size, _PAIR_CHUNK):
                left, right = pair_left[start : start + _PAIR_CHUNK], pair_right[start : start + _PAIR_CHUNK]
                # Blocks per pair follow its tiles (the triangle within a group): at least 1, at most its A tiles
                tiles_a = -(-cp.maximum(sizes[left], sizes[right]) // tile_a)
                tiles_b = -(-cp.minimum(sizes[left], sizes[right]) // _TILE_B)
                work = cp.where(left == right, tiles_a * (tiles_b + 1) // 2, tiles_a * tiles_b)
                del tiles_b
                per_block = max(1, -(-int(work.sum()) // (_BLOCKS_PER_SM * n_sm)))
                n_blocks = cp.minimum(cp.maximum(-(-work // per_block), 1), cp.maximum(tiles_a, 1))
                del tiles_a, work
                block_offsets = cp.zeros(left.size + 1, dtype=cp.int32)
                cp.cumsum(n_blocks, dtype=cp.int32, out=block_offsets[1:])
                sums = cp.zeros(left.size, dtype=cp.float64)
                grid = int(block_offsets[-1])
                fn(*args, *groups, left, right, block_offsets, grid, sums, norms_only=False, stream=stream.ptr)
                out[start : start + left.size] = sums / self.sc
                stream.synchronize()  # the chunk's arrays stay referenced until its kernel has finished
                del n_blocks, block_offsets, sums  # and are freed before the next chunk's
        return out


def _sort_indices(X):
    """Sort the columns of CSR ``X`` in place; a copy of it when its buffers are strided or not aligned."""
    if not X.has_sorted_indices:
        if X.indptr.dtype != X.indices.dtype:  # CuPy sorts int64 indptr with int32 indices into a wrong order
            raise ValueError(f"unsorted CSR needs one index dtype, got {X.indptr.dtype} and {X.indices.dtype}")
        # CuPy's sort reads the buffers without their strides and needs the values aligned to 16 bytes: such a
        # matrix is copied into C-contiguous buffers of its own and the caller's stays untouched
        if not all(a.flags.c_contiguous for a in (X.data, X.indices, X.indptr)) or X.data.data.ptr % 16:
            X = X.copy()
        X.sort_indices()
    return X


def _build_csr_source(embedding_raw, selector):
    """Move a (host or device) sparse matrix to GPU and subset its rows.

    ``selector`` is a boolean mask, an integer row-index array, or ``None``
    (keep all rows). Rows are kept in ``selector`` order so the caller's
    ``cell_indices`` line up. The result is CSR with sorted columns: a device
    matrix that is used whole is sorted in place, a sorted one is not touched;
    duplicate entries stay. The ``has_sorted_indices`` and
    ``has_canonical_format`` flags a matrix has cached are trusted.
    """
    from rapids_singlecell.get._anndata import X_to_GPU

    # cupyx CSR only stores float/complex/bool data, so cast unsupported dtypes
    # (e.g. integer raw counts) to float32 BEFORE the GPU transfer. The kernel
    # itself only has float32/float64 specializations.
    if embedding_raw.dtype != (dtype := _float_dtype(embedding_raw.dtype)):
        embedding_raw = embedding_raw.astype(dtype)

    X = X_to_GPU(embedding_raw)
    if not cpsp.isspmatrix_csr(X):
        X = X.tocsr()
    if selector is not None:
        rows = selector
        if getattr(rows, "dtype", None) == np.bool_:
            rows = np.flatnonzero(rows)
        X = X[cp.asarray(rows)]
        if not cpsp.isspmatrix_csr(X):
            X = X.tocsr()
    return _sort_indices(X)


def _materialize_source(embedding_raw, selector, precision: str, rows=None) -> _Source:
    """Materialize the subset on GPU as a :class:`_Source`: C-contiguous rows (dense input) or sorted CSR (sparse).

    ``selector`` is a boolean mask, an integer row-index array, or ``None``;
    ``rows`` are the rows a call reads of a dense array it loads whole.
    """
    dtype = _float_dtype(embedding_raw.dtype)
    if precision == "auto":
        precision = "float64" if dtype == np.float64 else "float32"
    if _is_sparse(embedding_raw):
        X = _build_csr_source(embedding_raw, selector)
        center = _center(X, precision)  # of the loaded cells, before the copies below
        # nm with zeros up to whole shared-memory windows: a column index of invalid input in the last window reads a 0
        width = 128 // np.dtype(_MATH[precision][4]).itemsize
        center = cp.pad(center[0], (0, -X.shape[1] % width)), center[1]
        indptr = X.indptr.astype(cp.int64 if X.nnz > _INT32_MAX else cp.int32, copy=False)
        arrays = tuple(cp.ascontiguousarray(a) for a in (indptr, X.indices.astype(cp.int32, copy=False), X.data))
    else:
        if getattr(selector, "dtype", None) == np.bool_:  # CuPy scans every element of the array for a row mask
            selector = np.flatnonzero(selector)
        X = _rows(embedding_raw if selector is None else embedding_raw[selector])
        center = _center(X, precision, rows)
        arrays = (X,)
    # CuPy caches the flag: asked before the sort it would stay False. It cannot scan indptr and indices of
    # different dtypes: such a matrix takes the summing loop.
    duplicates = cpsp.issparse(X) and (X.indptr.dtype != X.indices.dtype or not X.has_canonical_format)
    return _Source(arrays, X.shape, precision, *center, duplicates)


class EDistanceMetric(BaseMetric):
    """
    GPU-accelerated Energy Distance metric.

    Energy distance is a statistical distance between probability distributions
    that generalizes the Euclidean distance to distributions. It is particularly
    useful for comparing groups of cells in high-dimensional spaces.

    Parameters
    ----------
    layer_key
        Key in adata.layers for cell data. Mutually exclusive with obsm_key.
    obsm_key
        Key in adata.obsm for embeddings (default: 'X_pca')
    precision
        Arithmetic of the cell-to-cell distances; see
        :class:`~rapids_singlecell.ptg.Distance`.

    References
    ----------
    Székely, G. J., & Rizzo, M. L. (2013).
    Energy statistics: A class of statistics based on distances.
    Journal of Statistical Planning and Inference, 143(8), 1249-1272.
    """

    supports_multi_gpu: bool = True

    def __init__(
        self,
        layer_key: str | None = None,
        obsm_key: str | None = "X_pca",
        precision: Literal["auto", "float32", "float64", "float16"] = "auto",
    ):
        """Initialize energy distance metric."""
        super().__init__(layer_key=layer_key, obsm_key=obsm_key)
        if not isinstance(precision, str) or precision not in ("auto", *_MATH):
            raise ValueError(f"precision must be 'auto', 'float32', 'float64' or 'float16', got {precision!r}.")
        self.precision = precision

    def _load_source(
        self,
        adata: AnnData,
        groupby: str,
        needed_groups: Sequence[str] | None,
    ) -> tuple[_Source, cp.ndarray, cp.ndarray, list[str]]:
        """Load the cell data for ``groupby``, dense or sparse.

        Dense and sparse (CSR/CSC) input both return a :class:`_Source`; sparse
        rows are densified by the kernel on the fly. Both share the category
        mapping built by :meth:`_subset_indices`, so the downstream
        means/bootstrap/contrast math is layout-agnostic.
        """
        embedding_raw = self._get_embedding(adata)
        mask, cat_offsets, cell_indices, groups_list = self._subset_indices(adata, groupby, needed_groups)
        source = _materialize_source(embedding_raw, mask, self.precision)
        return source, cat_offsets, cell_indices, groups_list

    def pairwise(
        self,
        adata: AnnData,
        groupby: str,
        *,
        groups: Sequence[str] | None = None,
        bootstrap: bool = False,
        n_bootstrap: int = 100,
        random_state: int = 0,
        multi_gpu: bool | list[int] | str | None = None,
    ) -> pd.DataFrame | tuple[pd.DataFrame, pd.DataFrame]:
        """
        Compute pairwise energy distances between all cell groups.

        Returns a DataFrame containing pairwise distances. When bootstrap=True,
        returns a tuple of (distances, distances_var) DataFrames.

        The distances DataFrame contains:
        distances[a,b] = 2*d[a,b] - d[a] - d[b]

        When bootstrap=True, distances_var contains:
        distances_var[a,b] = 4*d_var[a,b] + d_var[a] + d_var[b]

        Parameters
        ----------
        adata
            Annotated data matrix
        groupby
            Key in adata.obs for grouping
        groups
            Specific groups to compute (if None, use all)
        bootstrap
            Whether to compute bootstrap variance estimates
        n_bootstrap
            Number of bootstrap iterations (if bootstrap=True)
        random_state
            Random seed for reproducibility
        multi_gpu
            GPU selection:
            - None: Use all GPUs if metric supports it, else GPU 0 (default)
            - True: Use all available GPUs
            - False: Use only the current GPU
            - list[int]: Use specific GPU IDs (e.g., [0, 2])
            - str: Comma-separated GPU IDs (e.g., "0,2")

        Returns
        -------
        result
            DataFrame with pairwise distances. If bootstrap=True, returns
            tuple of (distances, distances_var) DataFrames.
        """
        _assert_categorical_obs(adata, key=groupby)

        embedding, cat_offsets, cell_indices, groups_list = self._load_source(adata, groupby, groups)
        k = len(groups_list)

        if not bootstrap:
            return self._prepare_edistance_df(
                embedding=embedding,
                cat_offsets=cat_offsets,
                cell_indices=cell_indices,
                k=k,
                groups_list=groups_list,
                groupby=groupby,
                multi_gpu=multi_gpu,
            )

        return self._prepare_edistance_df_bootstrap(
            embedding=embedding,
            cat_offsets=cat_offsets,
            cell_indices=cell_indices,
            k=k,
            groups_list=groups_list,
            groupby=groupby,
            n_bootstrap=n_bootstrap,
            random_state=random_state,
            multi_gpu=multi_gpu,
        )

    def onesided_distances(
        self,
        adata: AnnData,
        groupby: str,
        selected_group: str | Sequence[str],
        *,
        groups: Sequence[str] | None = None,
        bootstrap: bool = False,
        n_bootstrap: int = 100,
        random_state: int = 0,
        multi_gpu: bool | list[int] | str | None = None,
    ) -> pd.Series | pd.DataFrame | tuple[pd.Series, pd.Series] | tuple[pd.DataFrame, pd.DataFrame]:
        """
        Compute energy distances from selected reference group(s) to all other groups.

        Parameters
        ----------
        adata
            Annotated data matrix
        groupby
            Key in adata.obs for grouping cells
        selected_group
            Reference group(s) to compute distances from. Can be a single
            group name or a sequence of group names for multiple controls.
            When a single string is passed, returns a Series. When a sequence
            is passed, returns a DataFrame with one column per control.
        groups
            Specific groups to compute distances to (if None, use all)
        bootstrap
            Whether to compute bootstrap variance estimates
        n_bootstrap
            Number of bootstrap iterations (if bootstrap=True)
        random_state
            Random seed for reproducibility
        multi_gpu
            GPU selection:
            - None: Use all GPUs if metric supports it, else GPU 0 (default)
            - True: Use all available GPUs
            - False: Use only the current GPU
            - list[int]: Use specific GPU IDs (e.g., [0, 2])
            - str: Comma-separated GPU IDs (e.g., "0,2")

        Returns
        -------
        distances
            Series (single control) or DataFrame (multiple controls).
            If bootstrap=True, returns tuple of (distances, distances_var).
        """
        selected_groups, single_control, needed = self._resolve_onesided_inputs(adata, groupby, selected_group, groups)

        embedding, cat_offsets, cell_indices, groups_list = self._load_source(adata, groupby, needed)
        k = len(groups_list)
        group_map = {v: i for i, v in enumerate(groups_list)}
        selected_indices = [group_map[sg] for sg in selected_groups]

        device_ids = parse_device_ids(multi_gpu=multi_gpu)

        if bootstrap:
            cross_mean, diag_mean, cross_var, diag_var = self._onesided_means_bootstrap(
                embedding=embedding,
                cat_offsets=cat_offsets,
                cell_indices=cell_indices,
                k=k,
                selected_indices=selected_indices,
                n_bootstrap=n_bootstrap,
                random_state=random_state,
                device_ids=device_ids,
            )

            # Compute energy distances for each control:
            # e[s,b] = 2*d[s,b] - d[s,s] - d[b,b]
            ed_cols = {}
            var_cols = {}
            for i, (sg, si) in enumerate(zip(selected_groups, selected_indices, strict=False)):
                ed_row = 2 * cross_mean[i, :] - diag_mean[si] - diag_mean
                ed_row[si] = 0.0
                ed_cols[sg] = ed_row.get()

                var_row = 4 * cross_var[i, :] + diag_var[si] + diag_var
                var_row[si] = 0.0
                var_cols[sg] = var_row.get()

            distances = pd.DataFrame(ed_cols, index=groups_list)
            distances.index.name = groupby
            distances.columns.name = "selected_group"

            variances = pd.DataFrame(var_cols, index=groups_list)
            variances.index.name = groupby
            variances.columns.name = "selected_group"

            if single_control:
                sg = selected_groups[0]
                return distances[sg], variances[sg]
            return distances, variances

        # Non-bootstrap path
        cross_means, diag_means = self._onesided_means(
            embedding,
            cat_offsets,
            cell_indices,
            k,
            selected_indices=selected_indices,
            device_ids=device_ids,
        )

        # Compute energy distances for each control:
        # e[s,b] = 2*d[s,b] - d[s,s] - d[b,b]
        # cross_means[i, j] = mean dist from selected[i] to group j
        # diag_means[j] = mean within-group dist for group j
        ed_cols = {}
        for i, (sg, si) in enumerate(zip(selected_groups, selected_indices, strict=False)):
            ed_row = 2 * cross_means[i, :] - diag_means[si] - diag_means
            ed_row[si] = 0.0
            ed_cols[sg] = ed_row.get()

        df = pd.DataFrame(ed_cols, index=groups_list)
        df.index.name = groupby
        df.columns.name = "selected_group"

        if single_control:
            return df[selected_groups[0]]
        return df

    def bootstrap(
        self,
        adata: AnnData,
        groupby: str,
        group_a: str,
        group_b: str,
        *,
        n_bootstrap: int = 100,
        random_state: int = 0,
        multi_gpu: bool | list[int] | str | None = None,
    ) -> tuple[float, float]:
        """
        Compute bootstrap mean and variance for energy distance between two groups.

        Parameters
        ----------
        adata
            Annotated data matrix
        groupby
            Key in adata.obs for grouping cells
        group_a
            First group name
        group_b
            Second group name
        n_bootstrap
            Number of bootstrap iterations
        random_state
            Random seed for reproducibility
        multi_gpu
            GPU selection:
            - None: Use all GPUs if metric supports it, else GPU 0 (default)
            - True: Use all available GPUs
            - False: Use only the current GPU
            - list[int]: Use specific GPU IDs (e.g., [0, 2])
            - str: Comma-separated GPU IDs (e.g., "0,2")

        Returns
        -------
        mean
            Bootstrap mean distance
        variance
            Bootstrap variance
        """
        # Compute pairwise with bootstrap
        df, df_var = self.pairwise(
            adata=adata,
            groupby=groupby,
            groups=[group_a, group_b],
            bootstrap=True,
            n_bootstrap=n_bootstrap,
            random_state=random_state,
            multi_gpu=multi_gpu,
        )

        mean = df.loc[group_a, group_b]
        variance = df_var.loc[group_a, group_b]

        return float(mean), float(variance)

    def contrast_distances(
        self,
        adata: AnnData,
        contrasts: pd.DataFrame,
        *,
        multi_gpu: bool | list[int] | str | None = None,
    ) -> pd.DataFrame:
        """
        Compute energy distances for contrasts.

        Parameters
        ----------
        adata
            Annotated data matrix
        contrasts
            DataFrame with a groupby column (first), a ``reference``
            column, and optional split columns.
        multi_gpu
            GPU selection:
            - None: Use all GPUs if metric supports it, else GPU 0 (default)
            - True: Use all available GPUs
            - False: Use only the current GPU
            - list[int]: Use specific GPU IDs (e.g., [0, 2])
            - str: Comma-separated GPU IDs (e.g., "0,2")

        Returns
        -------
        pd.DataFrame
            Copy of the input DataFrame with an added ``edistance`` column.
        """
        groupby, split_by = self._parse_contrasts(adata, contrasts)

        embedding_raw = self._get_embedding(adata)
        device_ids = parse_device_ids(multi_gpu=multi_gpu)

        all_cols = [groupby, *split_by]

        # Single groupby to get cell indices for all combinations
        grouped = adata.obs.groupby(all_cols, observed=True)
        group_indices = grouped.indices

        # Build conditions using numpy arrays (avoid per-row _asdict)
        target_vals = contrasts[groupby].values
        ref_vals = contrasts["reference"].values
        split_arrays = [contrasts[col].values for col in split_by]

        cond_to_idx: dict[tuple, int] = {}
        contrast_pairs: list[tuple[int, int]] = []

        for i in range(len(contrasts)):
            if split_by:
                split_vals = tuple(arr[i] for arr in split_arrays)
                target_key = (target_vals[i], *split_vals)
                ref_key = (ref_vals[i], *split_vals)
            else:
                target_key = (target_vals[i],)
                ref_key = (ref_vals[i],)

            for key in (target_key, ref_key):
                if key not in cond_to_idx:
                    cond_to_idx[key] = len(cond_to_idx)

            contrast_pairs.append((cond_to_idx[target_key], cond_to_idx[ref_key]))
        k = len(cond_to_idx)
        # Look up cell indices from the groupby
        group_cells: list[np.ndarray] = [None] * k  # type: ignore[list-item]
        missing: list[tuple] = []
        for key, idx in cond_to_idx.items():
            lookup_key = key[0] if len(key) == 1 else key
            cell_idx = group_indices.get(lookup_key)
            if cell_idx is None:
                missing.append(key)
            group_cells[idx] = cell_idx if cell_idx is not None else np.array([], dtype=np.intp)
        if missing:
            warnings.warn(
                f"No cells found for {missing}; their contrasts are NaN.",
                stacklevel=2,
            )
        # Build cat_offsets and cell_indices, subsetting the embedding
        # to only the referenced cells for memory efficiency
        offsets = [0]
        all_cell_idx = []
        for cells in group_cells:
            all_cell_idx.append(cells)
            offsets.append(offsets[-1] + len(cells))

        cat_offsets = cp.array(offsets, dtype=cp.int32)
        original_indices = np.concatenate(all_cell_idx)

        # Subset the referenced cells. For sparse input always row-subset into
        # CSR (the kernel densifies on the fly); for dense input subset when fewer
        # than 70 % of the cells are used, else load the whole array and read the
        # used cells through cell_indices.
        if _is_sparse(embedding_raw):
            embedding = _materialize_source(embedding_raw, original_indices, self.precision)
            cell_indices = cp.arange(len(original_indices), dtype=cp.int32)
        elif len(original_indices) < int(len(embedding_raw) * 0.7):
            embedding = _materialize_source(embedding_raw, original_indices, self.precision)
            cell_indices = cp.arange(len(original_indices), dtype=cp.int32)
        else:
            cell_indices = cp.array(original_indices, dtype=cp.int32)
            embedding = _materialize_source(embedding_raw, None, self.precision, cell_indices)

        group_sizes = cp.diff(cat_offsets).astype(cp.int64)
        group_sizes_cpu = group_sizes.get()
        # Build deduplicated pairs
        pair_to_flat: dict[tuple[int, int], int] = {}
        for idx_a, idx_b in contrast_pairs:
            cross = (min(idx_a, idx_b), max(idx_a, idx_b))
            if cross not in pair_to_flat:
                pair_to_flat[cross] = len(pair_to_flat)
            if group_sizes_cpu[idx_a] >= 2 and (idx_a, idx_a) not in pair_to_flat:
                pair_to_flat[(idx_a, idx_a)] = len(pair_to_flat)
            if group_sizes_cpu[idx_b] >= 2 and (idx_b, idx_b) not in pair_to_flat:
                pair_to_flat[(idx_b, idx_b)] = len(pair_to_flat)

        n_pairs = len(pair_to_flat)

        if n_pairs == 0:
            result = contrasts.copy()
            result["edistance"] = 0.0
            return result

        pairs = sorted(pair_to_flat.keys(), key=lambda p: pair_to_flat[p])
        pair_left = cp.array([p[0] for p in pairs], dtype=cp.int32)
        pair_right = cp.array([p[1] for p in pairs], dtype=cp.int32)

        flat_sums = self._launch_distance_kernel(
            embedding,
            cat_offsets,
            cell_indices,
            pair_left=pair_left,
            pair_right=pair_right,
            device_ids=device_ids,
        )

        # Vectorized normalization
        is_diag = pair_left == pair_right
        sizes_l = group_sizes[pair_left.astype(cp.intp)]
        sizes_r = group_sizes[pair_right.astype(cp.intp)]
        flat_norms = cp.where(
            is_diag,
            cp.maximum(sizes_l * (sizes_l - 1) // 2, 1),
            sizes_l * sizes_r,
        ).astype(embedding.dtype)
        flat_means = flat_sums / flat_norms
        flat_means_cpu = flat_means.get()

        # Extract edistances
        edistances = np.empty(len(contrast_pairs), dtype=flat_means_cpu.dtype)
        for i, (idx_a, idx_b) in enumerate(contrast_pairs):
            if idx_a == idx_b:
                edistances[i] = 0.0
                continue
            cross = (min(idx_a, idx_b), max(idx_a, idx_b))
            d_cross = flat_means_cpu[pair_to_flat[cross]]
            diag_a = pair_to_flat.get((idx_a, idx_a))
            d_aa = flat_means_cpu[diag_a] if diag_a is not None else 0.0
            diag_b = pair_to_flat.get((idx_b, idx_b))
            d_bb = flat_means_cpu[diag_b] if diag_b is not None else 0.0
            edistances[i] = 2 * d_cross - d_aa - d_bb

        result = contrasts.copy()
        result["edistance"] = edistances
        return result

    # Helper methods

    def compute_distance(
        self,
        X: np.ndarray | cp.ndarray,
        Y: np.ndarray | cp.ndarray,
    ) -> float:
        """
        Compute energy distance between two arrays directly.

        Parameters
        ----------
        X
            First array of shape (n_samples_x, n_features)
        Y
            Second array of shape (n_samples_y, n_features)

        Returns
        -------
        float
            Energy distance between X and Y
        """
        X_gpu, Y_gpu = (_device(a) for a in (X, Y))

        if len(X_gpu) == 0 or len(Y_gpu) == 0:
            raise ValueError("Neither X nor Y can be empty.")

        return self._energy(X_gpu, Y_gpu)

    def bootstrap_arrays(
        self,
        X: np.ndarray | cp.ndarray,
        Y: np.ndarray | cp.ndarray,
        *,
        n_bootstrap: int = 100,
        random_state: int = 0,
    ) -> tuple[float, float]:
        """
        Compute bootstrap mean and variance for energy distance between arrays.

        Parameters
        ----------
        X
            First array of shape (n_samples_x, n_features)
        Y
            Second array of shape (n_samples_y, n_features)
        n_bootstrap
            Number of bootstrap iterations
        random_state
            Random seed for reproducibility

        Returns
        -------
        mean
            Bootstrap mean distance
        variance
            Bootstrap variance
        """
        X_gpu, Y_gpu = (_device(a) for a in (X, Y))

        if len(X_gpu) == 0 or len(Y_gpu) == 0:
            raise ValueError("Neither X nor Y can be empty.")

        rng = np.random.default_rng(random_state)

        distances = []
        for _ in range(n_bootstrap):
            # Bootstrap sample indices
            X_idx = rng.choice(len(X_gpu), size=len(X_gpu), replace=True)
            Y_idx = rng.choice(len(Y_gpu), size=len(Y_gpu), replace=True)
            distances.append(self._energy(X_gpu[X_idx], Y_gpu[Y_idx]))

        return float(np.mean(distances)), float(np.var(distances))

    def _energy(self, X: cp.ndarray, Y: cp.ndarray) -> float:
        """Energy distance of two device arrays.

        Two float32 or float64 arrays take the float calls. Otherwise both arrays
        are read one block of columns at a time in float32 (float64 next to a
        float64 array). The blocks add up the row norms and the three Gram
        matrices. A block holds what the larger array has in bytes beyond the
        Gram matrices and one product, or as many bytes as these where that is
        more; next to a float64 array that second size is cut to what the float
        calls hold beyond the Gram matrices. A block has at most
        ``_CAST_VALUES`` values and at least one column.
        """
        if all(a.dtype in (np.float32, np.float64) for a in (X, Y)):
            d_xy = self._mean_pairwise_distance(X, Y)
            d_xx = self._mean_pairwise_distance_within(X)
            d_yy = self._mean_pairwise_distance_within(Y)
            return float(2 * d_xy - d_xx - d_yy)
        if X.ndim != 2 or Y.ndim != 2 or X.shape[1] != Y.shape[1]:
            raise ValueError("X and Y need two dimensions and the same number of features.")
        dtype = np.result_type(_float_dtype(X.dtype), _float_dtype(Y.dtype))
        n, m = len(X), len(Y)
        grams = (n * n + n * m + m * m + max(n, m) ** 2) * dtype.itemsize  # the Gram matrices and one product
        # on the input as it came the float calls hold 45 B per pair of cells of a group, 33 B in float16: stay below
        held = max((32 if a.dtype == np.float16 else 44) * len(a) ** 2 for a in (X, Y))
        room = max(max(X.nbytes, Y.nbytes) - grams, min(grams, held - grams))  # bytes of a block
        step = max(1, min(_CAST_VALUES, room // dtype.itemsize) // (n + m + 1))  # + 1: CuPy copies a single row
        gxx, gxy, gyy = (cp.zeros(shape, dtype=dtype) for shape in ((n, n), (n, m), (m, m)))
        nx, ny = cp.zeros(n, dtype=dtype), cp.zeros(m, dtype=dtype)
        for c in range(0, X.shape[1], step):
            xb, yb = (a[:, c : c + step].astype(dtype) for a in (X, Y))  # one compact copy of the block
            if n > 1:  # one cell has no pairs
                gxx += xb @ xb.T
            gxy += xb @ yb.T
            if m > 1:
                gyy += yb @ yb.T
            nx += cp.square(xb, out=xb).sum(axis=1)  # summed as the float calls sum the norms, in the block's memory
            ny += cp.square(yb, out=yb).sum(axis=1)
            del xb, yb  # before the next block is cast

        def mean_distance(g, a, b, pairs):  # the Gram matrix becomes the distances in place
            g -= a[:, None] / 2  # the halves first: a + b - 2 g overflows only where the squared distance does
            g -= b[None, :] / 2
            g *= -2
            cp.maximum(g, 0, out=g)
            if a is b:
                cp.fill_diagonal(g, 0)  # a cell against itself
            return float(cp.sqrt(g, out=g).sum()) / pairs if pairs else 0.0

        d_xy = mean_distance(gxy, nx, ny, n * m)
        d_xx = mean_distance(gxx, nx, nx, n * (n - 1))  # both triangles
        d_yy = mean_distance(gyy, ny, ny, m * (m - 1))
        return float(2 * d_xy - d_xx - d_yy)

    def _mean_pairwise_distance(
        self,
        X: cp.ndarray,
        Y: cp.ndarray,
    ) -> float:
        """Compute mean pairwise Euclidean distance between X and Y."""
        # Use cdist-like computation on GPU
        # X: (n, d), Y: (m, d) -> dist[i,j] = ||X[i] - Y[j]||

        # Compute squared distances using broadcasting
        # ||x - y||^2 = ||x||^2 + ||y||^2 - 2*x.y
        X_sq = cp.sum(X * X, axis=1, keepdims=True)  # (n, 1)
        Y_sq = cp.sum(Y * Y, axis=1, keepdims=True)  # (m, 1)
        XY = X @ Y.T  # (n, m)

        dist_sq = X_sq + Y_sq.T - 2 * XY
        dist_sq = cp.maximum(dist_sq, 0)  # Numerical stability
        distances = cp.sqrt(dist_sq)

        return float(cp.mean(distances))

    def _mean_pairwise_distance_within(self, X: cp.ndarray) -> float:
        """Compute mean pairwise Euclidean distance within X (upper triangle only)."""
        n = len(X)
        if n < 2:
            return 0.0

        # Compute all pairwise distances
        X_sq = cp.sum(X * X, axis=1, keepdims=True)
        dist_sq = X_sq + X_sq.T - 2 * (X @ X.T)
        dist_sq = cp.maximum(dist_sq, 0)
        distances = cp.sqrt(dist_sq)

        # Get upper triangle (excluding diagonal)
        triu_indices = cp.triu_indices(n, k=1)
        upper_distances = distances[triu_indices]

        return float(cp.mean(upper_distances))

    # Internal methods

    def _launch_distance_kernel(
        self,
        embedding: _Source,
        cat_offsets: cp.ndarray,
        cell_indices: cp.ndarray,
        *,
        pair_left: cp.ndarray,
        pair_right: cp.ndarray,
        device_ids: list[int],
    ) -> cp.ndarray:
        """Launch the edistance kernel across GPUs and return raw flat sums.

        This is the shared kernel launch logic used by all distance methods.
        Output is always a flat array of shape (n_pairs,) indexed by pair_id.

        Parameters
        ----------
        embedding
            Cells as a :class:`_Source`
        cat_offsets
            Category offsets on GPU 0
        cell_indices
            Cell indices on GPU 0
        pair_left
            Left group indices for each pair
        pair_right
            Right group indices for each pair
        device_ids
            List of GPU device IDs to use

        Returns
        -------
        cp.ndarray
            Raw distance sums of shape (n_pairs,), NOT normalized.
        """
        group_sizes = cp.diff(cat_offsets).astype(cp.int64)
        result_device = cat_offsets.device.id

        # Split pairs across devices with load balancing
        pair_chunks = _split_pairs(pair_left, pair_right, len(device_ids), group_sizes)

        # Every device's inputs are in place before the first one starts
        jobs = []
        for device_id, chunk in zip(device_ids, pair_chunks, strict=True):
            if len(chunk[0]):
                embedding.on_device(device_id)
                arrays = (cat_offsets, cell_indices, *chunk)
                jobs.append((device_id, *(_copy_to_device(a, device_id) for a in arrays)))
        if not jobs:
            with cp.cuda.Device(result_device):
                return cp.zeros(len(pair_left), dtype=embedding.dtype)

        # One thread per device: its chunks wait on the host without idling the others
        if len(jobs) == 1:
            sums = [embedding.pair_sums(*jobs[0])]
        else:
            with ThreadPoolExecutor(max_workers=len(jobs)) as executor:
                sums = list(executor.map(lambda job: embedding.pair_sums(*job), jobs))

        # Gather on the input device (chunks are consecutive pair ranges)
        return _concat_on_device(sums, result_device)

    def _pairwise_means(
        self,
        embedding: _Source,
        cat_offsets: cp.ndarray,
        cell_indices: cp.ndarray,
        k: int,
        device_ids: list[int],
    ) -> cp.ndarray:
        """Compute between-group mean distances for all group pairs.

        Uses flat kernel output and reconstructs a symmetric k×k matrix.

        Parameters
        ----------
        embedding
            Cells as a :class:`_Source`
        cat_offsets
            Category offsets on GPU 0
        cell_indices
            Cell indices on GPU 0
        k
            Number of groups
        device_ids
            List of GPU device IDs to use

        Returns
        -------
        cp.ndarray
            Matrix of mean pairwise distances (k x k)
        """
        group_sizes = cp.diff(cat_offsets).astype(cp.int64)

        # Build upper triangular indices, excluding diagonal for single-cell groups
        triu_indices = cp.triu_indices(k)
        pair_left = triu_indices[0].astype(cp.int32)
        pair_right = triu_indices[1].astype(cp.int32)

        is_diagonal = pair_left == pair_right
        has_pairs = group_sizes[pair_left] >= 2
        keep_mask = ~is_diagonal | has_pairs

        pair_left = pair_left[keep_mask]
        pair_right = pair_right[keep_mask]

        if len(pair_left) == 0:
            return cp.zeros((k, k), dtype=embedding.dtype)

        flat_sums = self._launch_distance_kernel(
            embedding,
            cat_offsets,
            cell_indices,
            pair_left=pair_left,
            pair_right=pair_right,
            device_ids=device_ids,
        )

        # Normalize flat sums
        flat_norms = cp.where(
            pair_left == pair_right,
            cp.maximum(group_sizes[pair_left] * (group_sizes[pair_left] - 1) // 2, 1),
            group_sizes[pair_left] * group_sizes[pair_right],
        ).astype(embedding.dtype)
        flat_means = flat_sums / flat_norms

        # Reconstruct symmetric k×k matrix from flat
        means = cp.zeros((k, k), dtype=embedding.dtype)
        means[pair_left.astype(cp.intp), pair_right.astype(cp.intp)] = flat_means
        means[pair_right.astype(cp.intp), pair_left.astype(cp.intp)] = flat_means

        return means

    def _onesided_means(
        self,
        embedding: _Source,
        cat_offsets: cp.ndarray,
        cell_indices: cp.ndarray,
        k: int,
        *,
        selected_indices: list[int],
        device_ids: list[int],
    ) -> tuple[cp.ndarray, cp.ndarray]:
        """Compute mean distances from selected group(s) to all groups.

        Uses flat kernel output to avoid O(k^2) memory allocation.

        Parameters
        ----------
        embedding
            Cells as a :class:`_Source`
        cat_offsets
            Category offsets on GPU 0
        cell_indices
            Cell indices on GPU 0
        k
            Number of groups
        selected_indices
            Indices of the selected (control) groups
        device_ids
            List of GPU device IDs to use

        Returns
        -------
        cross_means
            Array of shape (n_selected, k) where cross_means[i, j] is the
            mean distance from selected_indices[i] to group j.
        diag_means
            Array of shape (k,) with mean within-group distances.
        """
        group_sizes = cp.diff(cat_offsets).astype(cp.int64)
        group_sizes_cpu = group_sizes.get()
        n_selected = len(selected_indices)

        # Diagonal pairs first — one per group with >= 2 cells
        pair_list: list[tuple[int, int]] = []
        diag_flat: dict[int, int] = {}
        for j in range(k):
            if group_sizes_cpu[j] >= 2:
                diag_flat[j] = len(pair_list)
                pair_list.append((j, j))

        # Cross pairs grouped by selected group for L2 cache locality
        canon_to_flat: dict[tuple[int, int], int] = {}
        cross_flat: list[list[int]] = []
        for si in selected_indices:
            row = []
            for j in range(k):
                if si == j:
                    row.append(diag_flat.get(j, -1))
                else:
                    canon = (min(si, j), max(si, j))
                    idx = canon_to_flat.get(canon)
                    if idx is None:
                        idx = len(pair_list)
                        canon_to_flat[canon] = idx
                        pair_list.append((si, j))
                    row.append(idx)
            cross_flat.append(row)
        n_pairs = len(pair_list)
        if n_pairs == 0:
            return (
                cp.zeros((n_selected, k), dtype=embedding.dtype),
                cp.zeros(k, dtype=embedding.dtype),
            )

        pair_left = cp.array([p[0] for p in pair_list], dtype=cp.int32)
        pair_right = cp.array([p[1] for p in pair_list], dtype=cp.int32)

        flat_sums = self._launch_distance_kernel(
            embedding,
            cat_offsets,
            cell_indices,
            pair_left=pair_left,
            pair_right=pair_right,
            device_ids=device_ids,
        )

        # Vectorized normalization
        is_diag = pair_left == pair_right
        sizes_l = group_sizes[pair_left.astype(cp.intp)]
        sizes_r = group_sizes[pair_right.astype(cp.intp)]
        flat_norms = cp.where(
            is_diag,
            cp.maximum(sizes_l * (sizes_l - 1) // 2, 1),
            sizes_l * sizes_r,
        ).astype(embedding.dtype)
        flat_means = flat_sums / flat_norms

        # Vectorized reconstruction of cross_means (n_selected x k)
        cross_idx = cp.array(cross_flat, dtype=cp.int64)  # (n_selected, k)
        valid = cross_idx >= 0
        # Replace -1 with 0 for safe indexing, then mask
        safe_idx = cp.where(valid, cross_idx, 0)
        cross_means = cp.where(valid, flat_means[safe_idx], 0.0)

        # Vectorized reconstruction of diag_means (k,)
        diag_means = cp.zeros(k, dtype=embedding.dtype)
        if diag_flat:
            diag_j = cp.array(list(diag_flat.keys()), dtype=cp.intp)
            diag_idx = cp.array(list(diag_flat.values()), dtype=cp.int64)
            diag_means[diag_j] = flat_means[diag_idx]

        return cross_means, diag_means

    def _pairwise_means_bootstrap(
        self,
        embedding: _Source,
        *,
        cat_offsets: cp.ndarray,
        cell_indices: cp.ndarray,
        k: int,
        n_bootstrap: int,
        random_state: int,
        device_ids: list[int],
    ) -> tuple[cp.ndarray, cp.ndarray]:
        """Compute bootstrap statistics for pairwise distances.

        Each bootstrap iteration uses all GPUs for its pairwise computation.

        Parameters
        ----------
        embedding
            Cells as a :class:`_Source`
        cat_offsets
            Category offsets on GPU 0
        cell_indices
            Cell indices on GPU 0
        k
            Number of groups
        n_bootstrap
            Number of bootstrap iterations
        random_state
            Random seed for reproducibility
        device_ids
            List of GPU device IDs to use

        Returns
        -------
        tuple
            (means, variances) matrices (k x k each)
        """
        # Get group sizes for bootstrap sampling (on GPU 0)
        group_sizes = cp.diff(cat_offsets)

        # Run bootstrap iterations - each uses all GPUs for pairwise computation
        all_results = []
        for i in range(n_bootstrap):
            # Generate bootstrap sample on GPU 0
            boot_cat_offsets, boot_cell_indices = self._bootstrap_sample_cells(
                cat_offsets=cat_offsets,
                cell_indices=cell_indices,
                group_sizes_gpu=group_sizes,
                seed=random_state + i,
            )

            # Compute pairwise means using all GPUs
            pairwise_means = self._pairwise_means(
                embedding=embedding,
                cat_offsets=boot_cat_offsets,
                cell_indices=boot_cell_indices,
                k=k,
                device_ids=device_ids,
            )
            all_results.append(pairwise_means.get())

        # Compute statistics on the input device
        with cp.cuda.Device(cat_offsets.device.id):
            bootstrap_stack = cp.array(all_results)  # [n_bootstrap, k, k]
            means = cp.mean(bootstrap_stack, axis=0)
            variances = cp.var(bootstrap_stack, axis=0)

        return means, variances

    def _onesided_means_bootstrap(
        self,
        embedding: _Source,
        *,
        cat_offsets: cp.ndarray,
        cell_indices: cp.ndarray,
        k: int,
        selected_indices: list[int],
        n_bootstrap: int,
        random_state: int,
        device_ids: list[int],
    ) -> tuple[cp.ndarray, cp.ndarray, cp.ndarray, cp.ndarray]:
        """Compute bootstrap statistics for onesided distances.

        Each bootstrap iteration uses all GPUs for its onesided computation.

        Parameters
        ----------
        embedding
            Cells as a :class:`_Source`
        cat_offsets
            Category offsets on GPU 0
        cell_indices
            Cell indices on GPU 0
        k
            Number of groups
        selected_indices
            Indices of the selected (control) groups
        n_bootstrap
            Number of bootstrap iterations
        random_state
            Random seed for reproducibility
        device_ids
            List of GPU device IDs to use

        Returns
        -------
        cross_mean
            Mean of bootstrap cross_means, shape (n_selected, k)
        diag_mean
            Mean of bootstrap diag_means, shape (k,)
        cross_var
            Variance of bootstrap cross_means, shape (n_selected, k)
        diag_var
            Variance of bootstrap diag_means, shape (k,)
        """
        # Get group sizes for bootstrap sampling (on GPU 0)
        group_sizes = cp.diff(cat_offsets)

        # Run bootstrap iterations - each uses all GPUs for onesided computation
        all_cross = []
        all_diag = []
        for i in range(n_bootstrap):
            # Generate bootstrap sample on GPU 0
            boot_cat_offsets, boot_cell_indices = self._bootstrap_sample_cells(
                cat_offsets=cat_offsets,
                cell_indices=cell_indices,
                group_sizes_gpu=group_sizes,
                seed=random_state + i,
            )

            # Compute onesided means using all GPUs
            cross_means, diag_means = self._onesided_means(
                embedding=embedding,
                cat_offsets=boot_cat_offsets,
                cell_indices=boot_cell_indices,
                k=k,
                selected_indices=selected_indices,
                device_ids=device_ids,
            )
            all_cross.append(cross_means.get())
            all_diag.append(diag_means.get())

        # Compute statistics on the input device
        with cp.cuda.Device(cat_offsets.device.id):
            cross_stack = cp.array(all_cross)
            diag_stack = cp.array(all_diag)
            cross_mean = cp.mean(cross_stack, axis=0)
            diag_mean = cp.mean(diag_stack, axis=0)
            cross_var = cp.var(cross_stack, axis=0)
            diag_var = cp.var(diag_stack, axis=0)

        return cross_mean, diag_mean, cross_var, diag_var

    def _bootstrap_sample_cells(
        self,
        *,
        cat_offsets: cp.ndarray,
        cell_indices: cp.ndarray,
        group_sizes_gpu: cp.ndarray,
        seed: int,
    ) -> tuple[cp.ndarray, cp.ndarray]:
        """Generate bootstrap sample on the current device (fully parallel).

        Uses cupy random to generate indices directly on the GPU,
        avoiding cross-device transfers in multi-GPU scenarios.

        Parameters
        ----------
        cat_offsets
            Category offsets on current device
        cell_indices
            Cell indices on current device
        group_sizes_gpu
            Size of each group on current device
        seed
            Random seed for this bootstrap iteration

        Returns
        -------
        tuple
            (cat_offsets, new_cell_indices) on current device
            Note: cat_offsets unchanged since bootstrap preserves group sizes
        """
        rng = cp.random.default_rng(seed)
        total_cells = cell_indices.shape[0]

        if total_cells == 0:
            return cat_offsets, cell_indices

        # Generate random integers for all cells at once
        random_ints = rng.integers(0, cp.iinfo(cp.int64).max, size=total_cells, dtype=cp.int64)

        # cp.repeat requires list for repeats - small transfer (k integers)
        group_sizes_list = group_sizes_gpu.get().tolist()

        # Expand group sizes to per-cell (each cell knows its group's size)
        cell_group_sizes = cp.repeat(group_sizes_gpu, group_sizes_list)

        # Reduce to local indices within each group
        bootstrap_local_idx = random_ints % cell_group_sizes

        # Convert local indices to global indices by adding group offsets
        cell_group_offsets = cp.repeat(cat_offsets[:-1], group_sizes_list)
        bootstrap_global_idx = bootstrap_local_idx + cell_group_offsets

        # Gather bootstrap cells
        new_cell_indices = cell_indices[bootstrap_global_idx]

        return cat_offsets, new_cell_indices

    def _prepare_edistance_df_bootstrap(
        self,
        embedding: _Source,
        *,
        cat_offsets: cp.ndarray,
        cell_indices: cp.ndarray,
        k: int,
        groups_list: list[str],
        groupby: str,
        n_bootstrap: int = 100,
        random_state: int = 0,
        multi_gpu: bool | list[int] | str | None = None,
    ) -> tuple[pd.DataFrame, pd.DataFrame]:
        """Prepare bootstrap edistance DataFrames."""
        device_ids = parse_device_ids(multi_gpu=multi_gpu)
        pairwise_means_boot, pairwise_vars_boot = self._pairwise_means_bootstrap(
            embedding=embedding,
            cat_offsets=cat_offsets,
            cell_indices=cell_indices,
            k=k,
            n_bootstrap=n_bootstrap,
            random_state=random_state,
            device_ids=device_ids,
        )

        # Vectorized edistance: e[a,b] = 2*d[a,b] - d[a,a] - d[b,b]
        diag_means = cp.diag(pairwise_means_boot)
        edistance_means = 2 * pairwise_means_boot - diag_means[:, None] - diag_means[None, :]
        cp.fill_diagonal(edistance_means, 0)

        # Vectorized variance computation (delta method approximation):
        # var[a, b] = 4 * var[a, b] + var[a, a] + var[b, b]
        diag_vars = cp.diag(pairwise_vars_boot)
        edistance_vars = 4 * pairwise_vars_boot + diag_vars[:, None] + diag_vars[None, :]
        cp.fill_diagonal(edistance_vars, 0)

        df_mean = pd.DataFrame(edistance_means.get(), index=groups_list, columns=groups_list)
        df_mean.index.name = groupby
        df_mean.columns.name = groupby
        df_mean.name = "pairwise edistance"

        df_var = pd.DataFrame(edistance_vars.get(), index=groups_list, columns=groups_list)
        df_var.index.name = groupby
        df_var.columns.name = groupby
        df_var.name = "pairwise edistance variance"

        return df_mean, df_var

    def _prepare_edistance_df(
        self,
        embedding: _Source,
        *,
        cat_offsets: cp.ndarray,
        cell_indices: cp.ndarray,
        k: int,
        groups_list: list[str],
        groupby: str,
        multi_gpu: bool | list[int] | str | None = None,
    ) -> pd.DataFrame:
        """Prepare edistance DataFrame."""
        device_ids = parse_device_ids(multi_gpu=multi_gpu)
        pairwise_means = self._pairwise_means(embedding, cat_offsets, cell_indices, k, device_ids)

        # Vectorized edistance: e[a,b] = 2*d[a,b] - d[a,a] - d[b,b]
        diag = cp.diag(pairwise_means)
        edistance_matrix = 2 * pairwise_means - diag[:, None] - diag[None, :]
        cp.fill_diagonal(edistance_matrix, 0)  # Self-distance is 0

        df = pd.DataFrame(edistance_matrix.get(), index=groups_list, columns=groups_list)
        df.index.name = groupby
        df.columns.name = groupby
        df.name = "pairwise edistance"

        return df
