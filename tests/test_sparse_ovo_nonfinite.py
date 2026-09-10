"""Archived floating-search semantics for sparse OVO nonfinite values."""

from __future__ import annotations

import json
from pathlib import Path

import cupy as cp
import numpy as np
import pytest
import scipy.sparse as sparse

from rapids_singlecell._cuda import _wilcoxon_sparse_cuda as kernel

_ADAPTERS = [
    ("host_csr", "float32"),
    ("host_csr", "float64"),
    ("host_csc", "float32"),
    ("host_csc", "float64"),
    ("device_csr", "float32"),
    ("device_csc", "float32"),
]
_ORACLE = Path(__file__).parent / "_data" / "sparse_ovo_nonfinite.json"


def _nonfinite_source(dtype, geometry):
    # Both signs of signaling/quiet NaNs, infinities, signed zeros and subnormals.
    special = np.asarray(
        [
            0x7F800001,
            0x7FC00001,
            0x7FA00002,
            0xFFC00001,
            0xFF800001,
            0x3F800000,
            0,
            0x80000000,
            0xBF800000,
            0x7F800000,
            0x40000000,
            0xFF800000,
            0x7FFFFFFF,
            0xFFFFFFFF,
            0x80000001,
            1,
        ],
        np.uint32,
    )
    count, split = {"medium": (16, 3), "large": (516, 513), "huge": (2504, 2501)}[
        geometry
    ]
    reference = np.empty((5, 3), np.uint32)
    reference[:, 0] = special[:5]
    reference[:, 1] = np.asarray([-3, -1, 0, 1, 3], np.float32).view(np.uint32)
    reference[:, 2] = special[[4, 8, 6, 7, 0]]
    group = np.empty((count, 3), np.uint32)
    group[:, 0] = special[(np.arange(count) + 5) % len(special)]
    group[:, 1] = ((np.arange(count) * 31 % 7) - 3).astype(np.float32).view(np.uint32)
    group[:, 2] = special[(np.arange(count) * 7 + 3) % len(special)]
    # Keep one ordinary column next to NaN columns in the same batch.
    bits = np.concatenate((reference, group))
    rows, cols = bits.shape
    if dtype == "float64":
        single = bits.view(np.float32)
        nan = (bits & 0x7FFFFFFF) > 0x7F800000
        values = np.zeros(bits.shape, np.float64)
        values[~nan] = single[~nan]
        # Form double NaNs by integer operations, preserving signaling bits
        # until the archived/native float64-to-float32 rank conversion.
        values.view(np.uint64)[nan] = (
            ((bits[nan].astype(np.uint64) & 0x80000000) << 32)
            | np.uint64(0x7FF0000000000000)
            | ((bits[nan].astype(np.uint64) & 0x7FFFFF) << 29)
        )
    else:
        values = bits.view(np.float32)
    positions = np.arange(rows * cols).reshape(rows, cols)
    keep = ((bits & 0x7FFFFFFF) != 0) | (positions % 3 != 0)
    # Some zeros are implicit; retain other explicit signed zeros unchanged.
    offsets = np.r_[0, np.cumsum(keep.sum(axis=1))].astype(np.int32)
    source = sparse.csr_matrix(
        (values[keep], np.broadcast_to(np.arange(cols), bits.shape)[keep], offsets),
        shape=bits.shape,
    )
    return source, split


def _run_nonfinite_case(module, adapter, dtype, geometry, *, reverse, compute):
    source, split = _nonfinite_source(dtype, geometry)
    rows, cols = source.shape
    reference = np.arange(5, dtype=np.int32)
    members = np.arange(5, rows, dtype=np.int32)
    if reverse:
        reference = reference[::-1].copy()
        members = np.r_[members[:split][::-1], members[split:][::-1]]
    offsets = np.asarray([0, split, len(members)], np.int32)
    ref_map, grp_map = (np.full(rows, -1, np.int32) for _ in range(2))
    ref_map[reference], grp_map[members] = np.arange(5), np.arange(len(members))
    stats_codes = np.full(rows, -1, np.int32)
    stats_codes[reference] = 2
    stats_codes[members[:split]], stats_codes[members[split:]] = 0, 1
    ranks = cp.full((2, cols), -11, cp.float64)
    ties = cp.full((2, cols) if compute else (1,), -13, cp.float64)
    options = {"compute_tie_corr": compute, "sub_batch_cols": 2}
    if adapter.endswith("csc"):
        source = source.tocsc()
    if adapter.startswith("host"):
        statistics = [cp.empty((3, cols), cp.float64) for _ in range(2)]
        options["compute_nnz"] = True
        if adapter.endswith("csr"):
            arguments = [
                source.data,
                source.indices,
                source.indptr[:-1],
                source.indptr[1:],
                reference,
                members,
                offsets,
                ranks,
                ties,
                *statistics,
            ]
            options["n_cols"] = cols
        else:
            arguments = [
                source.data,
                source.indices,
                source.indptr,
                ref_map,
                grp_map,
                offsets,
                stats_codes,
                ranks,
                ties,
                *statistics,
            ]
            options.update(n_ref=len(reference), n_all_grp=len(members))
    else:
        arguments = [
            cp.asarray(source.data),
            cp.asarray(source.indices),
            cp.asarray(source.indptr),
            cp.asarray(ref_map if adapter.endswith("csc") else reference),
            cp.asarray(grp_map if adapter.endswith("csc") else members),
            cp.asarray(offsets),
            ranks,
            ties,
        ]
        options.update(n_ref=len(reference), n_all_grp=len(members))
    cp.cuda.get_current_stream().synchronize()
    location, format = adapter.split("_")
    with cp.cuda.Stream(non_blocking=True) as stream:
        getattr(module, f"ovo_streaming_{format}_{location}")(*arguments, **options)
    stream.synchronize()
    return {"ranks": ranks.get().tolist(), "ties": ties.get().tolist()}


@pytest.mark.parametrize(("adapter", "dtype"), _ADAPTERS)
@pytest.mark.parametrize("geometry", ["medium", "large", "huge"])
@pytest.mark.parametrize("reverse", [False, True])
@pytest.mark.parametrize("compute", [False, True])
def test_sparse_ovo_nonfinite_matches_archived_float_search(
    adapter, dtype, geometry, reverse, compute
):
    key = f"{adapter}/{dtype}/{geometry}/{int(reverse)}/{int(compute)}"
    expected = json.loads(_ORACLE.read_text())["cases"][key]
    actual = _run_nonfinite_case(
        kernel, adapter, dtype, geometry, reverse=reverse, compute=compute
    )
    np.testing.assert_array_equal(actual["ranks"], expected["ranks"])
    np.testing.assert_allclose(actual["ties"], expected["ties"], rtol=1e-15, atol=0)
