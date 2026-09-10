"""Empty host OVO populations preserve outputs after validating the call."""

from __future__ import annotations

import cupy as cp
import numpy as np
import pytest
import scipy.sparse as sparse

from rapids_singlecell._cuda import _wilcoxon_sparse_cuda as kernel


def _empty_case(format, empty, dtype, compute=True):
    rows, cols = 7, 0 if empty == "columns" else 5
    source = getattr(sparse, f"{format}_matrix")(
        np.arange(rows * cols, dtype=dtype).reshape(rows, cols)
    )
    references = np.asarray([] if empty == "reference" else [1, 0], np.int32)
    members = np.asarray([] if empty in {"members", "groups"} else [2, 3, 4], np.int32)
    offsets = np.asarray(
        [0] if empty == "groups" else [0, 0, 0] if empty == "members" else [0, 2, 3],
        np.int32,
    )
    groups = len(offsets) - 1
    # CSC statistics populations are independent of ranking populations, and
    # include rows excluded from ranking. Empty rankings must preserve these too.
    stats_rows = 4 if format == "csc" else groups + 1
    shapes = [(groups, cols), (groups, cols) if compute else (1,)]
    shapes += [(stats_rows, cols), (stats_rows, cols) if compute else (1,)]
    outputs, storage = [], []
    for seed, shape in enumerate(shapes, 17):
        size = int(np.prod(shape))
        values = np.arange(size + 7, dtype=np.float64) + seed
        values[-3:] = [-0.0, np.nan, np.inf]
        backing = cp.asarray(values)
        outputs.append(backing[:size].reshape(shape))
        storage.append((backing, values.view(np.uint64).copy()))
    options = {
        "compute_tie_corr": compute,
        "compute_nnz": compute,
        "sub_batch_cols": 2,
    }
    if format == "csr":
        arguments = [
            source.data,
            source.indices,
            source.indptr[:-1],
            source.indptr[1:],
            references,
            members,
            offsets,
            *outputs,
        ]
        options["n_cols"] = cols
    else:
        ref_map, group_map = (np.full(rows, -1, np.int32) for _ in range(2))
        ref_map[references] = np.arange(len(references))
        group_map[members] = np.arange(len(members))
        statistics = np.asarray([3, 0, -1, 2, 1, 3, 0], np.int32)
        arguments = [
            source.data,
            source.indices,
            source.indptr,
            ref_map,
            group_map,
            offsets,
            statistics,
            *outputs,
        ]
        options.update(n_ref=len(references), n_all_grp=len(members))
    return arguments, options, storage


def _assert_preserved(storage):
    for backing, expected in storage:
        np.testing.assert_array_equal(backing.view(cp.uint64).get(), expected)


@pytest.mark.parametrize("format", ["csr", "csc"])
@pytest.mark.parametrize("empty", ["reference", "members", "groups", "columns"])
@pytest.mark.parametrize("dtype", [np.float32, np.float64])
@pytest.mark.parametrize("compute", [False, True])
def test_empty_host_ovo_preserves_seeded_outputs(format, empty, dtype, compute):
    arguments, options, storage = _empty_case(format, empty, dtype, compute)
    cp.cuda.get_current_stream().synchronize()
    with cp.cuda.Stream(non_blocking=True) as stream:
        getattr(kernel, f"ovo_streaming_{format}_host")(*arguments, **options)
    stream.synchronize()
    # Include backing tails, so zero-sized output views also detect mutation.
    _assert_preserved(storage)


@pytest.mark.parametrize("format", ["csr", "csc"])
@pytest.mark.parametrize(
    "invalid",
    [
        "rank_dtype",
        "rank_shape",
        "tie_dtype",
        "sums_dtype",
        "counts_capacity",
        "overlap",
        "member_bounds",
        "offsets",
        "index_dtype",
    ],
)
def test_empty_host_ovo_still_validates_before_preserving_outputs(format, invalid):
    arguments, options, storage = _empty_case(format, "reference", np.float64)
    if invalid == "rank_dtype":
        arguments[7] = arguments[7].astype(cp.float32)
    elif invalid == "rank_shape":
        arguments[7] = arguments[7].reshape(-1)
    elif invalid == "tie_dtype":
        arguments[8] = arguments[8].astype(cp.float32)
    elif invalid == "sums_dtype":
        arguments[9] = arguments[9].astype(cp.float32)
    elif invalid == "counts_capacity":
        arguments[10] = arguments[10].ravel()[:-1]
    elif invalid == "overlap":
        arguments[8] = arguments[7]
    elif invalid == "member_bounds":
        arguments[5 if format == "csr" else 4] = np.asarray(
            [99, 3, 4] if format == "csr" else [-1, -1, 99, 1, 2, -1, -1],
            np.int32,
        )
    elif invalid == "offsets":
        arguments[6 if format == "csr" else 5] = np.asarray([0, 4, 3], np.int32)
    else:
        arguments[1] = arguments[1].astype(np.float64)
    cp.cuda.get_current_stream().synchronize()
    with cp.cuda.Stream(non_blocking=True) as stream:
        with pytest.raises((TypeError, ValueError)):
            getattr(kernel, f"ovo_streaming_{format}_host")(*arguments, **options)
        stream.synchronize()
    _assert_preserved(storage)
