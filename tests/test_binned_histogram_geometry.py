"""Binned histograms retain counts across warp and block traversal paths."""

from __future__ import annotations

import cupy as cp
import numpy as np
import pytest
import scipy.sparse as sp

from rapids_singlecell._cuda import _wilcoxon_binned_cuda as native


@pytest.mark.parametrize("shape", [(53, 257), (5001, 23)])
@pytest.mark.parametrize("dtype", [np.float32, np.float64])
@pytest.mark.parametrize("fmt", ["dense", "csr", "csc"])
def test_histograms_match_independent_bins_for_long_segments(shape, dtype, fmt):
    rng = np.random.default_rng(428)
    data = rng.normal(size=shape).astype(dtype)
    data[rng.random(shape) < 0.2] = 0
    codes = np.arange(shape[0], dtype=np.int32) % 5 - 1
    low, inverse, bins = -2.0, 3.0, 16
    expected = np.zeros((shape[1], 4, bins + 1), dtype=np.uint32)
    numbers = (
        np.clip(
            np.trunc((data.astype(np.float64) - low) * inverse), 0, bins - 1
        ).astype(np.int64)
        + 1
    )
    for column in range(shape[1]):
        valid = codes >= 0
        if fmt != "dense":
            valid &= data[:, column] != 0
        np.add.at(expected[column], (codes[valid], numbers[valid, column]), 1)
    with cp.cuda.Stream(non_blocking=True) as work:
        hist = cp.zeros(expected.shape, dtype=np.uint32)
        device_codes = cp.asarray(codes)
        kwargs = {
            "n_cells": shape[0],
            "n_genes": shape[1],
            "n_groups": 4,
            "n_bins": bins,
            "bin_low": low,
            "inv_bin_width": inverse,
            "stream": work.ptr,
        }
        if fmt == "dense":
            storage = (cp.asarray(data, order="F"), device_codes, hist)
        else:
            source = getattr(sp, f"{fmt}_matrix")(data)
            storage = (
                cp.asarray(source.data),
                cp.asarray(source.indices, dtype=np.int64),
                cp.asarray(source.indptr, dtype=np.int64),
                device_codes,
                hist,
            )
            kwargs["gene_start"] = 0
        getattr(native, f"{fmt}_hist")(*storage, **kwargs)
        observed = hist.get(stream=work)
    np.testing.assert_array_equal(observed, expected)


@pytest.mark.parametrize("dtype", [np.float32, np.float64])
def test_histogram_special_values_keep_original_bin_clamping(dtype):
    # Verified against the original CUDA conversion/clamping behavior, including
    # NaN, infinities, signed zero, and values beyond the signed integer range.
    values = np.array(
        [np.nan, np.inf, -np.inf, -1e30, 1e30, -0.0, 0.0, 2.0**32], dtype=dtype
    )
    data = cp.asarray(values).reshape((1, -1))
    codes = cp.zeros(1, dtype=np.int32)
    hist = cp.zeros((values.size, 1, 17), dtype=np.uint32)
    native.dense_hist(
        data,
        codes,
        hist,
        n_cells=1,
        n_genes=values.size,
        n_groups=1,
        n_bins=16,
        bin_low=-2.0,
        inv_bin_width=3.0,
    )
    expected = np.zeros(hist.shape, dtype=np.uint32)
    expected[np.arange(values.size), 0, [1, 16, 1, 1, 16, 7, 7, 16]] = 1
    np.testing.assert_array_equal(hist.get(), expected)
