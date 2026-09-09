"""Compatibility tests for the Jaccard CUDA extension and its Rust replacement."""

from __future__ import annotations

import cupy as cp
import numpy as np
import pytest

import rapids_singlecell._cuda._jaccard_cuda as _jaccard

rust_only = pytest.mark.skipif(
    getattr(_jaccard, "__backend__", None) != "rust",
    reason="Input validation is provided by the Rust binding",
)


def _reference(knn):
    """Count matching pairs after column zero, preserving duplicate entries."""
    n_obs, k = knn.shape
    result = np.zeros((n_obs, k), dtype=np.float32)
    for row in range(n_obs):
        for slot, neighbor in enumerate(knn[row]):
            if neighbor == row or not 0 <= neighbor < n_obs:
                continue
            count = np.count_nonzero(knn[row, 1:, None] == knn[neighbor, None, 1:])
            denominator = 2 * (k - 1) - count
            if denominator > 0:
                result[row, slot] = count / denominator
    return result.ravel()


@pytest.mark.parametrize(
    "neighbors",
    [
        pytest.param([[0, 1, 2], [1, 0, 2], [2, 0, 1]], id="self-first"),
        pytest.param([[2, 1, 2], [2, 0, 2], [1, 0, 1]], id="exclude-column-zero"),
        pytest.param([[0, 1, 2, 2], [1, 0, 2, 2], [2, 0, 1, 1]], id="duplicate-pairs"),
        pytest.param([[0, 1, 1, 1], [1, 1, 1, 1]], id="nonpositive-denominator"),
        pytest.param(
            [[0, -1, 3, 1], [1, 3, -1, 0], [2, 2, 1, 0]],
            id="self-and-invalid-neighbors",
        ),
        pytest.param([[0], [0], [2]], id="one-neighbor"),
        pytest.param(
            np.random.default_rng(17).integers(-1, 258, size=(257, 17)),
            id="arbitrary-graph-multiple-blocks",
        ),
    ],
)
def test_jaccard_shared_counts(neighbors):
    host_knn = np.asarray(neighbors, dtype=np.int32)
    knn = cp.asarray(host_knn)
    out = cp.full(knn.size, cp.nan, dtype=cp.float32)

    result = _jaccard.jaccard_shared_counts(
        knn,
        n_obs=knn.shape[0],
        k=knn.shape[1],
        jaccard_vals=out,
        stream=cp.cuda.get_current_stream().ptr,
    )

    assert result is None
    cp.testing.assert_allclose(out, _reference(host_knn), rtol=1e-6, atol=0)
    cp.testing.assert_array_equal(knn, host_knn)


@pytest.mark.parametrize("shape", [(0, 0), (0, 5), (3, 0)])
def test_jaccard_zero_work(shape):
    knn = cp.empty(shape, dtype=cp.int32)
    out = cp.empty(0, dtype=cp.float32)

    result = _jaccard.jaccard_shared_counts(
        knn, n_obs=shape[0], k=shape[1], jaccard_vals=out
    )

    assert result is None
    assert out.size == 0


def test_jaccard_uses_cupy_views_on_nondefault_stream():
    host_knn = np.asarray([[0, 1, 2], [1, 0, 2], [2, 0, 1]], dtype=np.int32)
    stream = cp.cuda.Stream(non_blocking=True)
    with stream:
        # Offset views exercise borrowed device pointers for both arguments.
        input_storage = cp.full(host_knn.size + 2, -123, dtype=cp.int32)
        knn = input_storage[1:-1].reshape(host_knn.shape)
        knn.set(host_knn, stream=stream)
        output_storage = cp.full(host_knn.size + 2, -123, dtype=cp.float32)
        out = output_storage[1:-1]
        _jaccard.jaccard_shared_counts(
            knn,
            n_obs=knn.shape[0],
            k=knn.shape[1],
            jaccard_vals=out,
            stream=stream.ptr,
        )
        # This copy must observe the launch on the caller's stream.
        observed = output_storage.copy()
    stream.synchronize()

    expected = np.concatenate(([-123], _reference(host_knn), [-123]))
    cp.testing.assert_allclose(observed, expected, rtol=1e-6, atol=0)
    cp.testing.assert_array_equal(knn, host_knn)


def test_jaccard_disjoint_views_in_one_allocation():
    host_knn = np.asarray([[0, 1, 2], [1, 0, 2], [2, 0, 1]], dtype=np.int32)
    storage = cp.empty(2 * host_knn.size, dtype=cp.int32)
    knn = storage[: host_knn.size].reshape(host_knn.shape)
    knn.set(host_knn)
    out = storage[host_knn.size :].view(cp.float32)

    _jaccard.jaccard_shared_counts(
        knn,
        n_obs=knn.shape[0],
        k=knn.shape[1],
        jaccard_vals=out,
        stream=cp.cuda.get_current_stream().ptr,
    )

    cp.testing.assert_allclose(out, _reference(host_knn), rtol=1e-6, atol=0)
    cp.testing.assert_array_equal(knn, host_knn)


def test_jaccard_managed_memory():
    device = cp.cuda.runtime.getDevice()
    if not cp.cuda.runtime.getDeviceProperties(device)["managedMemory"]:
        pytest.skip("CUDA device does not support managed memory")
    host_knn = np.asarray([[0, 1, 2], [1, 0, 2], [2, 0, 1]], dtype=np.int32)
    with cp.cuda.using_allocator(cp.cuda.malloc_managed), cp.cuda.Stream.null:
        knn = cp.asarray(host_knn)
        out = cp.full(knn.size, cp.nan, dtype=cp.float32)
        _jaccard.jaccard_shared_counts(
            knn, n_obs=knn.shape[0], k=knn.shape[1], jaccard_vals=out
        )
        cp.testing.assert_allclose(out, _reference(host_knn), rtol=1e-6, atol=0)
        cp.testing.assert_array_equal(knn, host_knn)


@rust_only
@pytest.mark.parametrize("output_offset", [0, 1])
def test_jaccard_rejects_overlapping_arrays(output_offset):
    storage = cp.zeros(10, dtype=cp.int32)
    knn = storage[:9].reshape(3, 3)
    out = storage[output_offset : output_offset + 9].view(cp.float32)

    with pytest.raises(ValueError):
        _jaccard.jaccard_shared_counts(knn, n_obs=3, k=3, jaccard_vals=out)


@rust_only
@pytest.mark.parametrize("unaligned_argument", ["knn", "jaccard_vals"])
def test_jaccard_rejects_unaligned_arrays(unaligned_argument):
    knn = cp.zeros((3, 3), dtype=cp.int32)
    out = cp.empty(9, dtype=cp.float32)
    unaligned_pointer = cp.cuda.alloc(knn.nbytes + 1) + 1
    if unaligned_argument == "knn":
        knn = cp.ndarray(knn.shape, dtype=knn.dtype, memptr=unaligned_pointer)
    else:
        out = cp.ndarray(out.shape, dtype=out.dtype, memptr=unaligned_pointer)

    with pytest.raises(ValueError):
        _jaccard.jaccard_shared_counts(knn, n_obs=3, k=3, jaccard_vals=out)


@rust_only
@pytest.mark.parametrize("oversized_argument", ["knn", "jaccard_vals"])
def test_jaccard_rejects_views_beyond_their_allocation(oversized_argument):
    knn = cp.zeros((32, 32), dtype=cp.int32)
    out = cp.empty(1024, dtype=cp.float32)
    if oversized_argument == "knn":
        storage = cp.empty(1, dtype=cp.int32)
        knn = cp.lib.stride_tricks.as_strided(
            storage, shape=knn.shape, strides=(32 * knn.itemsize, knn.itemsize)
        )
        oversized = knn
    else:
        storage = cp.empty(1, dtype=cp.float32)
        out = cp.lib.stride_tricks.as_strided(
            storage, shape=out.shape, strides=(out.itemsize,)
        )
        oversized = out
    assert oversized.flags.c_contiguous
    assert oversized.nbytes > oversized.data.mem.size

    with pytest.raises(ValueError):
        _jaccard.jaccard_shared_counts(knn, n_obs=32, k=32, jaccard_vals=out)


@rust_only
@pytest.mark.parametrize("input_device,output_device", [(0, 1), (1, 0), (1, 1)])
def test_jaccard_rejects_arrays_on_another_device(input_device, output_device):
    if cp.cuda.runtime.getDeviceCount() < 2:
        pytest.skip("Requires two CUDA devices")
    with cp.cuda.Device(input_device):
        knn = cp.zeros((3, 3), dtype=cp.int32)
    with cp.cuda.Device(output_device):
        out = cp.empty(9, dtype=cp.float32)

    with cp.cuda.Device(0):
        with pytest.raises(ValueError):
            _jaccard.jaccard_shared_counts(knn, n_obs=3, k=3, jaccard_vals=out)
        assert cp.cuda.runtime.getDevice() == 0


@rust_only
@pytest.mark.parametrize(
    "knn_dtype,out_dtype",
    [
        (cp.int64, cp.float32),
        (cp.float32, cp.float32),
        (cp.int32, cp.float64),
        (cp.int32, cp.int32),
    ],
)
def test_jaccard_rejects_incompatible_dtypes(knn_dtype, out_dtype):
    knn = cp.zeros((3, 3), dtype=knn_dtype)
    out = cp.empty(9, dtype=out_dtype)

    with pytest.raises(TypeError):
        _jaccard.jaccard_shared_counts(knn, n_obs=3, k=3, jaccard_vals=out)


@rust_only
@pytest.mark.parametrize("host_argument", ["knn", "jaccard_vals"])
def test_jaccard_rejects_host_arrays(host_argument):
    knn = cp.zeros((3, 3), dtype=cp.int32)
    out = cp.empty(9, dtype=cp.float32)
    if host_argument == "knn":
        knn = np.zeros((3, 3), dtype=np.int32)
    else:
        out = np.empty(9, dtype=np.float32)

    with pytest.raises(TypeError):
        _jaccard.jaccard_shared_counts(knn, n_obs=3, k=3, jaccard_vals=out)


@rust_only
@pytest.mark.parametrize("noncontiguous_argument", ["knn", "jaccard_vals"])
def test_jaccard_rejects_noncontiguous_arrays(noncontiguous_argument):
    knn = cp.zeros((3, 3), dtype=cp.int32)
    out = cp.empty(9, dtype=cp.float32)
    if noncontiguous_argument == "knn":
        knn = knn.T
    else:
        out = cp.empty(18, dtype=cp.float32)[::2]

    with pytest.raises(ValueError):
        _jaccard.jaccard_shared_counts(knn, n_obs=3, k=3, jaccard_vals=out)


@rust_only
@pytest.mark.parametrize(
    "knn_shape,out_shape,n_obs,k",
    [
        ((3, 3), (9,), 2, 3),
        ((3, 3), (9,), 3, 2),
        ((3, 3), (8,), 3, 3),
        ((3, 3), (10,), 3, 3),
        ((9,), (9,), 3, 3),
        ((3, 3), (3, 3), 3, 3),
        ((3, 3), (9,), -1, 3),
        ((3, 3), (9,), 3, -1),
        ((3, 3), (9,), 0, 3),
        ((3, 3), (9,), 3, 0),
    ],
)
def test_jaccard_rejects_inconsistent_shapes(knn_shape, out_shape, n_obs, k):
    knn = cp.zeros(knn_shape, dtype=cp.int32)
    out = cp.empty(out_shape, dtype=cp.float32)

    with pytest.raises(ValueError):
        _jaccard.jaccard_shared_counts(knn, n_obs=n_obs, k=k, jaccard_vals=out)
