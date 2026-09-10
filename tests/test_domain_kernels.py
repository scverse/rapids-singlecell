"""Boundary and precision checks for native domain kernel implementations."""

from __future__ import annotations

import cupy as cp
import numpy as np
import pytest
from cupyx.scipy import sparse
from scipy.special import logsumexp

from rapids_singlecell._cuda import (
    _aucell_cuda,
    _bbknn_cuda,
    _cooc_cuda,
    _edistance_cuda,
    _kde_cuda,
    _nn_descent_cuda,
    _pv_cuda,
)


@pytest.mark.parametrize("n_thresholds", [1, 257, 513, 1537])
@pytest.mark.parametrize("tuning_block", [None, 32, 96, 256, 512, 1024])
def test_cooccurrence_threshold_windows_and_large_counters(n_thresholds, tuning_block):
    rng = np.random.default_rng(14)
    counts_by_group = [2, 129, 5]
    offsets = np.r_[0, np.cumsum(counts_by_group)].astype(np.int32)
    xy = rng.integers(0, 13, size=(offsets[-1], 2)).astype(np.float32)
    thresholds = np.linspace(0, 300, n_thresholds, dtype=np.float32)
    left, right = np.triu_indices(len(counts_by_group))
    initial = np.uint64(2**40)
    counts = cp.full((3, 3, n_thresholds), initial, dtype=cp.uint64)
    cell_tile, _, block_size, shared_mem = _cooc_cuda.get_kernel_config(
        n_thresholds, len(xy), 3
    )
    if tuning_block is not None:
        block_size = tuning_block
    _cooc_cuda.count_csr_catpairs(
        cp.asarray(xy),
        thresholds=cp.asarray(thresholds),
        cat_offsets=cp.asarray(offsets),
        cell_indices=cp.arange(len(xy), dtype=cp.int32),
        pair_left=cp.asarray(left, dtype=cp.int32),
        pair_right=cp.asarray(right, dtype=cp.int32),
        counts=counts,
        num_pairs=len(left),
        k=3,
        l_val=n_thresholds,
        blocks_per_pair=3,
        cell_tile=cell_tile,
        block_size=block_size,
        shared_mem=shared_mem,
    )
    expected = np.full(counts.shape, initial, dtype=np.uint64)
    for a, b in zip(left, right, strict=True):
        delta = (
            xy[offsets[a] : offsets[a + 1], None]
            - xy[None, offsets[b] : offsets[b + 1]]
        )
        distance = (delta * delta).sum(axis=-1)
        if a == b:
            distance = distance[np.triu_indices(len(distance), 1)]
        expected[a, b] += (distance.reshape(-1, 1) <= thresholds).sum(
            axis=0, dtype=np.uint64
        )
    np.testing.assert_array_equal(counts.get(), expected)


@pytest.mark.parametrize(
    "x,y,threshold,expected",
    [
        (6.841277599334717, 1.6219831705093384, 49.43390655517578, 0),
        (6.036921977996826, 3.712001085281372, 50.2233772277832, 1),
    ],
)
@pytest.mark.parametrize("pairwise", [False, True])
def test_cooccurrence_preserves_fused_threshold_boundary(
    x, y, threshold, expected, pairwise
):
    # Archived CUDA contracts dx*dx + dy*dy into fma(dx,dx,rounded(dy*dy)).
    # These adjacent-float thresholds distinguish it from two rounded squares.
    spatial = cp.asarray([[0, 0], [x, y]], dtype=cp.float32)
    thresholds = cp.asarray([threshold], dtype=cp.float32)
    result = cp.zeros(2 if pairwise else 1, dtype=cp.uint64)
    if pairwise:
        _cooc_cuda.count_pairwise(
            spatial,
            thresholds=thresholds,
            labels=cp.zeros(2, dtype=cp.int32),
            result=result,
            n=2,
            k=1,
            l_val=1,
        )
    else:
        tile, _, block, shared = _cooc_cuda.get_kernel_config(1, 2, 1)
        _cooc_cuda.count_csr_catpairs(
            spatial,
            thresholds=thresholds,
            cat_offsets=cp.asarray([0, 2], dtype=cp.int32),
            cell_indices=cp.arange(2, dtype=cp.int32),
            pair_left=cp.zeros(1, dtype=cp.int32),
            pair_right=cp.zeros(1, dtype=cp.int32),
            counts=result,
            num_pairs=1,
            k=1,
            l_val=1,
            blocks_per_pair=1,
            cell_tile=tile,
            block_size=block,
            shared_mem=shared,
        )
    assert int(result.sum().get()) == expected


@pytest.mark.parametrize("dtype", [np.float32, np.float64])
def test_kde_stable_logsumexp_on_external_stream(dtype):
    rng = np.random.default_rng(20)
    xy = rng.normal(size=(257, 2)).astype(dtype)
    a, b, c = -150.0, 35.0, -90.0
    delta = xy[:, None].astype(np.float64) - xy[None].astype(np.float64)
    dx, dy = delta[..., 0], delta[..., 1]
    expected = logsumexp(a * dx * dx + b * dx * dy + c * dy * dy, axis=1)
    stream = cp.cuda.Stream(non_blocking=True)
    with stream:
        device_xy = cp.asarray(xy)
        output = cp.empty(len(xy), dtype=dtype)
        _kde_cuda.gaussian_kde_2d(
            device_xy, out=output, n=len(xy), a=a, b=b, c=c, stream=stream.ptr
        )
    stream.synchronize()
    tolerance = 2e-6 if dtype == np.float32 else 1e-12
    np.testing.assert_allclose(output.get(), expected, atol=tolerance, rtol=tolerance)


@pytest.mark.parametrize("n_features", [1, 33, 65])
@pytest.mark.parametrize("dtype", [np.float32, np.float64])
@pytest.mark.parametrize("sparse_input", [False, True])
@pytest.mark.parametrize("large_block", [False, True])
def test_energy_distance_feature_and_cell_tile_tails(
    n_features, dtype, sparse_input, large_block
):
    rng = np.random.default_rng(32)
    x = rng.normal(size=(64, n_features)).astype(dtype)
    offsets = np.array([0, 31, 64], dtype=np.int32)
    left, right = np.triu_indices(2)
    stream = cp.cuda.Stream(non_blocking=True)
    with stream:
        args = [
            cp.asarray(x),
            cp.asarray(offsets),
            cp.arange(len(x), dtype=cp.int32),
            cp.asarray(left, dtype=cp.int32),
            cp.asarray(right, dtype=cp.int32),
            cp.full(3, 7, dtype=dtype),
        ]
        cell_tile, feat_tile, block_size, shared_mem = (
            _edistance_cuda.get_kernel_config(n_features, dtype == np.float64)
        )
        if large_block:
            block_size = 1024
        if sparse_input:
            csr = sparse.csr_matrix(args[0])
            csr.sort_indices()
            operands = [csr.indptr, csr.indices, csr.data, *args[1:]]
            function = _edistance_cuda.compute_distances_sparse
        else:
            operands = args
            function = _edistance_cuda.compute_distances
        function(
            *operands,
            num_pairs=3,
            n_features=n_features,
            blocks_per_pair=3,
            cell_tile=cell_tile,
            feat_tile=feat_tile,
            block_size=block_size,
            shared_mem=shared_mem,
            stream=stream.ptr,
        )
    stream.synchronize()
    expected = []
    for a, b in zip(left, right, strict=True):
        delta = x[offsets[a] : offsets[a + 1], None].astype(np.float64) - x[
            None, offsets[b] : offsets[b + 1]
        ].astype(np.float64)
        distance = np.sqrt((delta * delta).sum(axis=-1))
        if a == b:
            distance = distance[np.triu_indices(len(distance), 1)]
        expected.append(7 + distance.sum())
    tolerance = 3e-7 if dtype == np.float32 else 1e-12
    np.testing.assert_allclose(args[-1].get(), expected, rtol=tolerance)


@pytest.mark.parametrize("dtype", [np.float32, np.float64])
@pytest.mark.parametrize("n_features", [1025, 3072, 3073])
@pytest.mark.parametrize("geary", [False, True])
def test_sparse_autocorrelation_feature_windows(dtype, n_features, geary):
    from rapids_singlecell._cuda import _autocorr_cuda

    rows = 7
    data = np.zeros((rows, n_features), dtype=dtype)
    columns = np.unique(np.minimum([0, 1023, 1024, 3071, 3072], n_features - 1))
    data[:, columns] = np.arange(1, rows + 1, dtype=dtype)[:, None]
    adjacency = np.zeros((rows, rows), dtype=dtype)
    for row in range(rows):
        adjacency[row, (row + 1) % rows] = 0.5
        adjacency[row, (row + 3) % rows] = 0.25
    mean = data.mean(axis=0)
    if geary:
        expected = (adjacency[:, :, None] * (data[:, None] - data[None]) ** 2).sum(
            axis=(0, 1)
        )
    else:
        centered = data - mean
        expected = (centered * (adjacency @ centered)).sum(axis=0)
    stream = cp.cuda.Stream(non_blocking=True)
    with stream:
        csr = sparse.csr_matrix(cp.asarray(data))
        graph = sparse.csr_matrix(cp.asarray(adjacency))
        output = cp.full(n_features, 7, dtype=dtype)
        kwargs = {
            "data_row_ptr": csr.indptr,
            "data_col_ind": csr.indices,
            "data_values": csr.data,
            "n_samples": rows,
            "n_features": n_features,
            "num": output,
            "stream": stream.ptr,
        }
        function = _autocorr_cuda.gearys_sparse
        if not geary:
            function = _autocorr_cuda.morans_sparse
            kwargs["mean_array"] = cp.asarray(mean)
        function(graph.indptr, graph.indices, graph.data, **kwargs)
    stream.synchronize()
    tolerance = 2e-6 if dtype == np.float32 else 1e-12
    np.testing.assert_allclose(
        output.get(), 7 + expected, rtol=tolerance, atol=tolerance
    )


@pytest.mark.parametrize("sparse_input", [False, True])
def test_energy_legacy_c32_tile_with_1024_threads(sparse_input):
    rng = np.random.default_rng(937)
    x = rng.normal(size=(70, 65)).astype(np.float32)
    x[np.abs(x) < 0.7] = 0
    stream = cp.cuda.Stream(non_blocking=True)
    with stream:
        device = cp.asarray(x)
        output = cp.zeros(1, dtype=cp.float32)
        metadata = [
            cp.asarray([0, 33, 70], dtype=cp.int32),
            cp.arange(70, dtype=cp.int32),
            cp.asarray([0], dtype=cp.int32),
            cp.asarray([1], dtype=cp.int32),
            output,
        ]
        if sparse_input:
            csr = sparse.csr_matrix(device)
            csr.sort_indices()
            operands = [csr.indptr, csr.indices, csr.data, *metadata]
            function = _edistance_cuda.compute_distances_sparse
        else:
            operands = [device, *metadata]
            function = _edistance_cuda.compute_distances
        function(
            *operands,
            num_pairs=1,
            n_features=65,
            blocks_per_pair=3,
            cell_tile=32,
            feat_tile=32,
            block_size=1024,
            shared_mem=4096,
            stream=stream.ptr,
        )
    stream.synchronize()
    delta = x[:33, None].astype(np.float64) - x[None, 33:].astype(np.float64)
    expected = np.sqrt((delta * delta).sum(axis=-1)).sum()
    np.testing.assert_allclose(output.get()[0], expected, rtol=3e-7)


@pytest.mark.parametrize("features", [3, 31, 513])
@pytest.mark.parametrize("metric", ["sqeuclidean", "inner", "cosine"])
def test_neighbor_distances_preserve_float32_accumulation(features, metric):
    rng = np.random.default_rng(81)
    values = rng.normal(size=(3, features)).astype(np.float32)
    values[0] = 0
    pairs = np.asarray([[1, 2], [0, 2], [1, 1]], dtype=np.uint32)
    result = cp.empty(pairs.shape, dtype=cp.float32)
    getattr(_nn_descent_cuda, metric)(
        cp.asarray(values),
        out=result,
        pairs=cp.asarray(pairs),
        n_samples=len(values),
        n_features=features,
        n_neighbors=pairs.shape[1],
    )

    def fused_dot(left, right):
        # Float64 exactly represents each float32 product for these bounded
        # inputs; round only after adding it to the float32 running sum.
        total = np.float32(0)
        for x, y in zip(left, right, strict=True):
            total = np.float32(np.float64(x) * np.float64(y) + np.float64(total))
        return total

    expected = np.empty(pairs.shape, dtype=np.float32)
    for row in range(len(values)):
        for column, neighbor in enumerate(pairs[row]):
            left, right = values[row], values[neighbor]
            if metric == "sqeuclidean":
                delta = left - right
                expected[row, column] = fused_dot(delta, delta)
            elif metric == "inner":
                expected[row, column] = fused_dot(left, right)
            else:
                norm_left = fused_dot(left, left)
                norm_right = fused_dot(right, right)
                inverse_left = 1 / np.sqrt(norm_left) if norm_left > 0 else 0
                inverse_right = 1 / np.sqrt(norm_right) if norm_right > 0 else 0
                expected[row, column] = (
                    1 - fused_dot(left, right) * inverse_left * inverse_right
                )
    if metric == "cosine":
        # CUDA's rsqrtf approximation differs by a few ulps from NumPy sqrt.
        np.testing.assert_allclose(result.get(), expected, rtol=3e-7, atol=3e-7)
    else:
        np.testing.assert_array_equal(result.get(), expected)


@pytest.mark.parametrize("inplace", [False, True])
@pytest.mark.parametrize("columns", [3, 4097])
def test_reverse_cumulative_minimum_preserves_nan_and_signed_zero(inplace, columns):
    values = np.asarray(
        [
            [4, 3, np.nan],
            [4, np.nan, 3],
            [np.nan, 4, 3],
            [2, 0.0, -0.0],
            [2, -0.0, 0.0],
            [np.inf, -np.inf, np.inf],
        ],
        dtype=np.float64,
    )
    values = np.tile(values, (1, (columns + 2) // 3))[:, :columns].copy()
    values[0, -1] = np.nan
    expected = values.copy()
    for row in expected:
        current = row[-1]
        for column in range(len(row) - 2, -1, -1):
            if row[column] < current:
                current = row[column]
            row[column] = current
    data = cp.asarray(values)
    output = data if inplace else cp.empty_like(data)
    _pv_cuda.rev_cummin64(data, out=output, n_rows=len(values), m=values.shape[1])
    actual = output.get()
    np.testing.assert_array_equal(actual, expected)
    np.testing.assert_array_equal(np.signbit(actual), np.signbit(expected))


@pytest.mark.parametrize("cutoff", [-2, 0, 3])
@pytest.mark.parametrize("repeats", [1, 257])
def test_aucell_signed_rank_cutoff(cutoff, repeats):
    ranks = np.asarray([[-3, -1, 0, 2, 5], [5, 2, 0, -1, -3]], dtype=np.int32)
    sets = [[0, 1, 2] * repeats, [1, 3, 4] * repeats]
    maxima = np.asarray([5, 7], dtype=np.float32)
    result = cp.empty((2, 2), dtype=cp.float32)
    _aucell_cuda.auc(
        cp.asarray(ranks),
        R=2,
        C=5,
        cnct=cp.asarray(np.concatenate(sets), dtype=cp.int32),
        starts=cp.asarray([0, 3 * repeats], dtype=cp.int32),
        lens=cp.asarray([3 * repeats, 3 * repeats], dtype=cp.int32),
        n_sets=2,
        n_up=cutoff,
        max_aucs=cp.asarray(maxima),
        es=result,
    )
    expected = np.asarray(
        [
            [
                sum(cutoff - int(row[gene]) for gene in genes if row[gene] <= cutoff)
                / maximum
                for genes, maximum in zip(sets, maxima, strict=True)
            ]
            for row in ranks
        ],
        dtype=np.float32,
    )
    np.testing.assert_array_equal(result.get(), expected)


@pytest.mark.parametrize("length", [127, 128, 129, 257])
def test_aucell_long_sets_preserve_row_tails_and_indirect_bounds(length):
    rng = np.random.default_rng(338)
    rows, columns, sets, cutoff = 33, 257, 5, 9
    ranks = rng.integers(-3, 18, (rows, columns), dtype=np.int32)
    genes = rng.integers(-1, columns + 2, sets * length, dtype=np.int32)
    starts = np.asarray([0, length, 2 * length, -1, len(genes) - 7], np.int32)
    lengths = np.asarray([length, length, length, 0, length], np.int32)
    maxima = np.arange(1, sets + 1, dtype=np.float32)
    expected = np.zeros((rows, sets), dtype=np.float32)
    for set_index, (start, count) in enumerate(zip(starts, lengths, strict=True)):
        if start < 0 or count <= 0:
            continue
        selected = genes[start : min(start + count, len(genes))]
        selected = selected[(selected >= 0) & (selected < columns)]
        values = ranks[:, selected].astype(np.int64)
        score = np.where(values <= cutoff, cutoff - values, 0).sum(axis=1)
        expected[:, set_index] = score.astype(np.float32) / maxima[set_index]
    output = cp.full(rows * sets + 5, -17, dtype=cp.float32)
    _aucell_cuda.auc(
        cp.asarray(ranks),
        R=rows,
        C=columns,
        cnct=cp.asarray(genes),
        starts=cp.asarray(starts),
        lens=cp.asarray(lengths),
        n_sets=sets,
        n_up=cutoff,
        max_aucs=cp.asarray(maxima),
        es=output,
    )
    np.testing.assert_array_equal(
        output[: rows * sets].get().reshape(rows, sets), expected
    )
    np.testing.assert_array_equal(output[rows * sets :].get(), -17)


def test_typed_reverse_minimum_rejects_short_or_overlapping_outputs():
    data = cp.arange(13, dtype=cp.float64)
    with pytest.raises(ValueError, match="allocation"):
        _pv_cuda.rev_cummin64(data[:12], out=cp.empty(11, cp.float64), n_rows=3, m=4)
    with pytest.raises(ValueError, match="overlap"):
        _pv_cuda.rev_cummin64(data[:12], out=data[1:], n_rows=3, m=4)


def test_typed_aucell_rejects_short_gene_set_metadata():
    with pytest.raises(ValueError, match="allocation"):
        _aucell_cuda.auc(
            cp.zeros((2, 5), cp.int32),
            R=2,
            C=5,
            cnct=cp.asarray([0, 1], dtype=cp.int32),
            starts=cp.asarray([0], dtype=cp.int32),
            lens=cp.asarray([1, 1], dtype=cp.int32),
            n_sets=2,
            n_up=3,
            max_aucs=cp.ones(2, cp.float32),
            es=cp.empty((2, 2), cp.float32),
        )


@pytest.mark.parametrize(
    "case",
    ["zeros", "zeros_wide", "negative_nan", "special", "negative_finite", 31, 33, 2048],
)
def test_bbknn_sorted_preserves_cub_bit_order_and_padding(case):
    # Independent stable-sort oracle, checked against the archived CUB kernel.
    # Keep signaling NaNs as integer bit patterns throughout the reference.
    if isinstance(case, int):
        bits = np.random.default_rng(case).integers(
            0, 2**32, (4, case), dtype=np.uint32
        )
    elif case in ("zeros", "zeros_wide"):
        bits = np.asarray(
            [[0x80000000, 0, 0x80000000, 0, 0, 0x80000000, 0, 0x80000000]],
            np.uint32,
        )
        if case == "zeros_wide":
            bits = np.tile(bits, (1, 256))
    elif case == "negative_nan":
        bits = np.asarray([[0xFFC00000 + i for i in range(1, 9)]], np.uint32)
    elif case == "special":
        bits = np.asarray(
            [
                [
                    0x7FC00002,
                    0x7F800001,
                    0xFFC00003,
                    0xFF800001,
                    0x7F800000,
                    0xFF800000,
                    0x80000000,
                    0,
                ]
            ],
            np.uint32,
        )
    else:
        bits = np.asarray([[-2, -9, -5, -7, -3, -4, -1, -6]], np.float32).view(
            np.uint32
        )
    rows, length = bits.shape
    padded = np.full((rows, 2048), 0xFF800000, np.uint32)
    padded[:, :length] = bits
    keys = np.where(
        padded & 0x80000000, np.bitwise_not(padded), padded ^ np.uint32(0x80000000)
    ).astype(np.uint32)
    keys[padded & 0x7FFFFFFF == 0] = 0x80000000
    order = np.argsort(~keys, axis=1, kind="stable")
    ordered = np.take_along_axis(padded, order, axis=1)
    data = cp.asarray(bits).view(cp.float32)
    indptr = cp.arange(rows + 1, dtype=cp.int32) * length
    output = cp.empty(rows, cp.float32)
    for trim in sorted({1, 2, min(7, length - 1), length // 2, length - 1, length}):
        _bbknn_cuda.find_top_k_per_row_sorted(
            data, indptr, n_rows=rows, trim=trim, vals=output
        )
        expected = ordered[:, trim - 1] if trim < length else np.zeros(rows, np.uint32)
        np.testing.assert_array_equal(output.get().view(np.uint32), expected)
