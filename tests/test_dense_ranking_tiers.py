"""Exact OVO tier boundaries, independent ties and bounded final column batches."""

from __future__ import annotations

import cupy as cp
import numpy as np
import pytest
from scipy.stats import rankdata, tiecorrect

from rapids_singlecell._cuda import _wilcoxon_cuda

pytestmark = pytest.mark.skipif(
    getattr(_wilcoxon_cuda, "__backend__", None) != "rust",
    reason="requires Rust backend",
)


@pytest.mark.parametrize("compute_ties", [False, True])
@pytest.mark.parametrize(
    "sizes",
    [(0, 1, 512), (513, 1024, 2500), (2501, 4097, 3001), (0, 512, 513, 2500, 2501)],
)
@pytest.mark.parametrize("distribution", ["continuous", "ties", "constant"])
def test_dense_ovo_tier_boundaries(sizes, compute_ties, distribution):
    rng = np.random.default_rng(83)
    reference = rng.normal(size=(1031, 5)).astype(np.float32)
    group = rng.normal(size=(sum(sizes), 5)).astype(np.float32)
    if distribution == "ties":
        reference = np.round(reference)
        group = np.round(group)
        # Exercise IEEE ordering and signed-zero equality in both GPU sorts.
        reference[:4, 0] = [-np.inf, np.inf, -0.0, 0.0]
        group[:4, 0] = [np.inf, -np.inf, 0.0, -0.0]
    elif distribution == "constant":
        reference.fill(0)
        group.fill(0)
    offsets = np.r_[0, np.cumsum(sizes)].astype(np.int64)
    expected = np.empty((len(sizes), 5))
    expected_ties = np.empty_like(expected)
    for g, size in enumerate(sizes):
        combined = np.concatenate((group[offsets[g] : offsets[g + 1]], reference))
        expected[g] = rankdata(combined, axis=0)[:size].sum(axis=0, dtype=np.float64)
        expected_ties[g] = [tiecorrect(col) for col in combined.T]
    stream = cp.cuda.Stream(non_blocking=True)
    with stream:
        ref = cp.asarray(reference, order="F")
        grp = cp.asarray(group, order="F")
        off = cp.asarray(offsets)
        result = cp.full(expected.shape, -1, dtype=np.float64)
        ties = cp.full_like(result, 19)
    _wilcoxon_cuda.ovo_rank_dense_tiered_unsorted_ref(
        ref,
        grp,
        off,
        result,
        ties,
        compute_tie_corr=compute_ties,
        sub_batch_cols=3,
        stream=stream.ptr,
    )
    np.testing.assert_array_equal(cp.asnumpy(result), expected)
    if compute_ties:
        np.testing.assert_allclose(cp.asnumpy(ties), expected_ties, atol=1e-14)
    else:
        cp.testing.assert_array_equal(ties, 19)


@pytest.mark.parametrize("dtype", [np.float32, np.float64])
@pytest.mark.parametrize("order", ["C", "F"])
def test_host_ovo_tier_statistics_preserve_original_precision(dtype, order):
    rng = np.random.default_rng(96)
    sizes = (1, 512, 513, 2500, 2501)
    reference_rows = 733
    data = np.array(
        rng.normal(size=(reference_rows + sum(sizes), 7)), dtype=dtype, order=order
    )
    data[::3] = 0
    # These differences collapse under float32 ranking but must survive LFC sums.
    data[1::3] += np.array(2**-30, dtype=dtype)
    ref_ids = np.arange(reference_rows, dtype=np.int64)
    grp_ids = np.arange(reference_rows, len(data), dtype=np.int64)
    offsets = np.r_[0, np.cumsum(sizes)].astype(np.int64)
    rank = cp.empty((len(sizes), 5), dtype=np.float64)
    ties = cp.empty_like(rank)
    sums = cp.empty((len(sizes) + 1, 5), dtype=np.float64)
    nnz = cp.empty_like(sums)
    _wilcoxon_cuda.ovo_rank_dense_host_streaming(
        data,
        ref_ids,
        grp_ids,
        offsets,
        rank,
        ties,
        sums,
        nnz,
        compute_tie_corr=True,
        compute_nnz=True,
        col_start=1,
        col_stop=6,
        sub_batch_cols=3,
    )
    windows = [
        data[grp_ids[offsets[g] : offsets[g + 1]], 1:6] for g in range(len(sizes))
    ]
    windows.append(data[ref_ids, 1:6])
    np.testing.assert_allclose(
        cp.asnumpy(sums),
        [x.sum(axis=0, dtype=np.float64) for x in windows],
        rtol=1e-12,
        atol=1e-12,
    )
    np.testing.assert_array_equal(
        cp.asnumpy(nnz), [np.count_nonzero(x, axis=0) for x in windows]
    )
    for g, size in enumerate(sizes):
        combined = np.concatenate((windows[g], windows[-1])).astype(np.float32)
        np.testing.assert_array_equal(
            cp.asnumpy(rank[g]),
            rankdata(combined, axis=0)[:size].sum(axis=0, dtype=np.float64),
        )
        np.testing.assert_allclose(
            cp.asnumpy(ties[g]), [tiecorrect(col) for col in combined.T], atol=1e-14
        )


@pytest.mark.parametrize(
    "rows,groups",
    [
        (1, 1),
        (257, 7),
        (10003, 512),
        (10003, 513),
        (10003, 4096),
        (10003, 6112),
        (10003, 6113),
    ],
)
@pytest.mark.parametrize("compute_ties", [False, True])
def test_ovr_tie_walk_chunk_boundaries_and_many_groups(rows, groups, compute_ties):
    rng = np.random.default_rng(37)
    data = np.round(rng.normal(size=(rows, 3))).astype(np.float32)
    codes = rng.integers(-1, groups + 1, rows, dtype=np.int32)
    ranks = cp.full((groups, 3), -1, dtype=np.float64)
    ties = cp.full(3, 23, dtype=np.float64)
    _wilcoxon_cuda.ovr_rank_dense_streaming(
        cp.asarray(data, order="F"),
        cp.asarray(codes),
        ranks,
        ties,
        compute_tie_corr=compute_ties,
        sub_batch_cols=2,
    )
    expected = np.zeros((groups, 3), dtype=np.float64)
    included = (codes >= 0) & (codes < groups)
    np.add.at(expected, codes[included], rankdata(data, axis=0)[included])
    np.testing.assert_array_equal(cp.asnumpy(ranks), expected)
    if compute_ties:
        np.testing.assert_allclose(
            cp.asnumpy(ties), [tiecorrect(col) for col in data.T], atol=1e-14
        )
    else:
        cp.testing.assert_array_equal(ties, 23)


@pytest.mark.parametrize("location", ["device", "host"])
@pytest.mark.parametrize("compute_ties", [False, True])
@pytest.mark.parametrize(
    "api,empty",
    [
        ("ovr", "rows"),
        ("ovr", "groups"),
        ("ovr", "columns"),
        ("ovo", "reference"),
        ("ovo", "rows"),
        ("ovo", "groups"),
        ("ovo", "columns"),
    ],
)
def test_dense_empty_whole_populations_preserve_seeded_outputs(
    api, empty, location, compute_ties
):
    """Archived native entry points make these validated calls a no-op."""
    groups = 0 if empty == "groups" else 2
    columns = 0 if empty == "columns" else 3
    rows = 0 if empty in ("rows", "groups") and api == "ovo" else 5
    if api == "ovr" and empty == "rows":
        rows = 0
    reference_rows = 0 if empty == "reference" else 7
    ranks = cp.full((groups, columns), 17, dtype=np.float64)
    ties = cp.full(
        (groups, columns) if api == "ovo" and compute_ties else columns,
        19,
        dtype=np.float64,
    )
    if api == "ovo" and not compute_ties:
        ties = cp.full(1, 19, dtype=np.float64)
    outputs = [ranks, ties]
    flags = {"compute_tie_corr": compute_ties, "sub_batch_cols": 2}
    if api == "ovr":
        data = np.ones((rows, columns), dtype=np.float32)
        codes = cp.zeros(rows, dtype=np.int32)
        if location == "device":
            _wilcoxon_cuda.ovr_rank_dense_streaming(
                cp.asarray(data, order="F"), codes, ranks, ties, **flags
            )
        else:
            stats = [cp.full_like(ranks, 23), cp.full_like(ranks, 29)]
            stats += [cp.full(columns, n, dtype=np.float64) for n in (31, 37)]
            outputs += stats
            _wilcoxon_cuda.ovr_rank_dense_host_streaming(
                data,
                codes,
                ranks,
                ties,
                *stats,
                compute_nnz=True,
                compute_totals=True,
                **flags,
            )
    else:
        offsets = np.linspace(0, rows, groups + 1, dtype=np.int32)
        if location == "device":
            _wilcoxon_cuda.ovo_rank_dense_tiered_unsorted_ref(
                cp.ones((reference_rows, columns), dtype=np.float32, order="F"),
                cp.ones((rows, columns), dtype=np.float32, order="F"),
                cp.asarray(offsets),
                ranks,
                ties,
                **flags,
            )
        else:
            data = np.ones((reference_rows + rows, columns), dtype=np.float32)
            stats = [
                cp.full((groups + 1, columns), n, dtype=np.float64) for n in (23, 29)
            ]
            outputs += stats
            _wilcoxon_cuda.ovo_rank_dense_host_streaming(
                data,
                np.arange(reference_rows, dtype=np.int32),
                np.arange(reference_rows, len(data), dtype=np.int32),
                offsets,
                ranks,
                ties,
                *stats,
                compute_nnz=True,
                **flags,
            )
    seeds = (17, 19, 23, 29, 31, 37)[: len(outputs)]
    for output, seed in zip(outputs, seeds, strict=True):
        cp.testing.assert_array_equal(output, seed)


@pytest.mark.parametrize("rows", [17, 1023, 1024, 1025, 4099, 32769])
def test_ovr_radix_stable_ieee_payloads_across_tiles(rows):
    rng = np.random.default_rng(126)
    bits = rng.integers(0, 2**32, (rows, 3), dtype=np.uint32)
    specials = np.array(
        [
            0,
            0x80000000,
            0x7F800000,
            0xFF800000,
            0x7FFFFFFF,
            0xFFFFFFFF,
            0x7FC00001,
            0xFFC00001,
            0x00000001,
            0x80000001,
        ],
        dtype=np.uint32,
    )
    bits.ravel()[: len(specials)] = specials
    data = bits.view(np.float32)
    codes = np.arange(rows, dtype=np.int32) % 11
    # CUB sorts NaNs by transformed IEEE bits, while signed zeros share a
    # stable key. Its tie walker treats each NaN as a distinct value.
    expected = np.zeros((11, 3), dtype=np.float64)
    for col in range(3):
        raw = bits[:, col]
        keys = np.where(raw & np.uint32(0x80000000), ~raw, raw ^ np.uint32(0x80000000))
        keys[data[:, col] == 0] = np.uint32(0x80000000)
        order = np.argsort(keys, kind="stable")
        first = 0
        while first < rows:
            end = first + 1
            while end < rows and data[order[end], col] == data[order[first], col]:
                end += 1
            for row in order[first:end]:
                expected[codes[row], col] += (first + end + 1) * 0.5
            first = end
    ranks = cp.empty_like(cp.asarray(expected))
    _wilcoxon_cuda.ovr_rank_dense_streaming(
        cp.asarray(data, order="F"),
        cp.asarray(codes),
        ranks,
        None,
        compute_tie_corr=False,
        sub_batch_cols=2,
    )
    np.testing.assert_array_equal(cp.asnumpy(ranks), expected)


@pytest.mark.parametrize("varying_bits", [0, 0xFF, 0xFF0000, 0xFFFF, 0x7FFFFF])
@pytest.mark.parametrize("ovo", [False, True])
def test_dense_sort_preserves_low_precision_float_distributions(varying_bits, ovo):
    # Float bit patterns with varying precision exercise skipped radix bytes,
    # both ping-pong destinations, and the independent huge-group sorter.
    rng = np.random.default_rng(92)
    rows = 32771
    bits = np.uint32(0x3F800000) | (
        rng.integers(0, 2**23, (rows, 3), dtype=np.uint32) & np.uint32(varying_bits)
    )
    data = bits.view(np.float32)
    data[::3] *= -1
    if ovo:
        sizes = (32773, 3)
        group = np.resize(data[::-1], (sum(sizes), 3)).copy()
        offsets = np.r_[0, np.cumsum(sizes)].astype(np.int64)
        rank = cp.empty((len(sizes), 3), dtype=np.float64)
        ties = cp.empty_like(rank)
        _wilcoxon_cuda.ovo_rank_dense_tiered_unsorted_ref(
            cp.asarray(data, order="F"),
            cp.asarray(group, order="F"),
            cp.asarray(offsets),
            rank,
            ties,
            compute_tie_corr=True,
            sub_batch_cols=2,
        )
        for g, size in enumerate(sizes):
            combined = np.concatenate((group[offsets[g] : offsets[g + 1]], data))
            np.testing.assert_array_equal(
                cp.asnumpy(rank[g]),
                rankdata(combined, axis=0)[:size].sum(axis=0, dtype=np.float64),
            )
            np.testing.assert_allclose(
                cp.asnumpy(ties[g]), [tiecorrect(col) for col in combined.T], atol=1e-14
            )
    else:
        codes = np.arange(rows, dtype=np.int32) % 7
        rank = cp.empty((7, 3), dtype=np.float64)
        ties = cp.empty(3, dtype=np.float64)
        _wilcoxon_cuda.ovr_rank_dense_streaming(
            cp.asarray(data, order="F"),
            cp.asarray(codes),
            rank,
            ties,
            compute_tie_corr=True,
            sub_batch_cols=2,
        )
        expected = rankdata(data, axis=0)
        np.testing.assert_array_equal(
            cp.asnumpy(rank),
            [expected[codes == g].sum(axis=0, dtype=np.float64) for g in range(7)],
        )
        np.testing.assert_allclose(
            cp.asnumpy(ties), [tiecorrect(col) for col in data.T], atol=1e-14
        )


@pytest.mark.parametrize(
    "rows,width", [(4999, 32), (5000, 31), (5000, 32), (8192, 32), (8193, 32)]
)
@pytest.mark.parametrize("distribution", ["continuous", "ties", "fixed_mantissa"])
def test_dense_reference_block_sort_boundaries(rows, width, distribution):
    rng = np.random.default_rng(419)
    columns = 67  # Complete block-sort windows followed by a radix tail.
    reference = rng.normal(size=(rows, columns)).astype(np.float32)
    if distribution == "ties":
        reference = np.round(reference)
    elif distribution == "fixed_mantissa":
        bits = np.uint32(0x3F800000) | (
            rng.integers(0, 256, reference.shape, dtype=np.uint32) << 12
        )
        bits[::3] |= np.uint32(0x80000000)
        reference = bits.view(np.float32)
    reference[:4] = np.array([-0.0, 0.0, -np.inf, np.inf], dtype=np.float32)[:, None]
    sizes = (17, 513)
    group = np.resize(reference[::-1], (sum(sizes), columns)).copy()
    offsets = np.r_[0, np.cumsum(sizes)].astype(np.int32)
    ranks = cp.empty((len(sizes), columns), dtype=np.float64)
    ties = cp.empty_like(ranks)
    _wilcoxon_cuda.ovo_rank_dense_tiered_unsorted_ref(
        cp.asarray(reference, order="F"),
        cp.asarray(group, order="F"),
        cp.asarray(offsets),
        ranks,
        ties,
        compute_tie_corr=True,
        sub_batch_cols=width,
    )
    for g, size in enumerate(sizes):
        combined = np.concatenate((group[offsets[g] : offsets[g + 1]], reference))
        np.testing.assert_array_equal(
            ranks[g].get(), rankdata(combined, axis=0)[:size].sum(axis=0)
        )
        np.testing.assert_allclose(
            ties[g].get(), [tiecorrect(col) for col in combined.T], atol=1e-14
        )
