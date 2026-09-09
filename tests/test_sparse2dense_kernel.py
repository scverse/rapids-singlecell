"""Compatibility tests for sparse scattering through the Rust bindings."""

from __future__ import annotations

import cupy as cp
import numpy as np
import pytest

import rapids_singlecell._cuda._sparse2dense_cuda as _s2d

rust_only = pytest.mark.skipif(
    getattr(_s2d, "__backend__", None) != "rust",
    reason="Input validation and malformed row guards are provided by Rust",
)

_ARRAY_NAMES = ("indptr", "index", "data", "out")
_EXPECTED = np.asarray([[1, 0, 5, 4], [0, 0, 0, 0], [0, 3, 0, 7]])


def _arguments(*, dtype=cp.float32, index_dtype=cp.int32, fmt="csr", order="C"):
    if fmt == "csr":
        indptr = [0, 4, 4, 7]
        index = [0, 2, 2, 3, 1, 1, 3]
        data = [1, 2, 3, 4, 5, -2, 7]
        major, minor = 3, 4
    else:
        indptr = [0, 1, 3, 5, 7]
        index = [0, 2, 2, 0, 0, 0, 2]
        data = [1, 5, -2, 2, 3, 4, 7]
        major, minor = 4, 3
    return {
        "indptr": cp.asarray(indptr, dtype=index_dtype),
        "index": cp.asarray(index, dtype=index_dtype),
        "data": cp.asarray(data, dtype=dtype),
        "out": cp.zeros((3, 4), dtype=dtype, order=order),
        "major": major,
        "minor": minor,
        "c_switch": (fmt == "csr") == (order == "C"),
        "max_nnz": 4,
        "stream": cp.cuda.get_current_stream().ptr,
    }


@pytest.mark.parametrize("dtype", [cp.float32, cp.float64])
@pytest.mark.parametrize("index_dtype", [cp.int32, cp.int64])
@pytest.mark.parametrize("fmt", ["csr", "csc"])
@pytest.mark.parametrize("order", ["C", "F"])
@pytest.mark.parametrize("initial_value", [0, 10])
def test_sparse2dense_accumulates_duplicates(
    dtype, index_dtype, fmt, order, initial_value
):
    args = _arguments(dtype=dtype, index_dtype=index_dtype, fmt=fmt, order=order)
    args["out"].fill(initial_value)
    original_inputs = {name: args[name].copy() for name in _ARRAY_NAMES[:-1]}

    result = _s2d.sparse2dense(**args)

    assert result is None
    cp.testing.assert_array_equal(args["out"], _EXPECTED + initial_value)
    for name, original in original_inputs.items():
        cp.testing.assert_array_equal(args[name], original)


@pytest.mark.parametrize("index_dtype", [cp.int32, cp.int64])
def test_sparse2dense_skips_invalid_minor_indices(index_dtype):
    args = _arguments(index_dtype=index_dtype)
    args["index"] = cp.asarray([-1, 2, 2, 4, 1, 1, 3], dtype=index_dtype)

    _s2d.sparse2dense(**args)

    cp.testing.assert_array_equal(
        args["out"], [[0, 0, 5, 0], [0, 0, 0, 0], [0, 3, 0, 7]]
    )


@pytest.mark.parametrize("shape", [(0, 0), (0, 4), (3, 0), (3, 4)])
@pytest.mark.parametrize("max_nnz", [0, 1])
def test_sparse2dense_zero_work(shape, max_nnz):
    out = cp.full(shape, 7, dtype=cp.float32)

    result = _s2d.sparse2dense(
        cp.zeros(shape[0] + 1, dtype=cp.int32),
        cp.empty(0, dtype=cp.int32),
        cp.empty(0, dtype=cp.float32),
        out=out,
        major=shape[0],
        minor=shape[1],
        c_switch=True,
        max_nnz=max_nnz,
        stream=cp.cuda.get_current_stream().ptr,
    )

    assert result is None
    cp.testing.assert_array_equal(out, cp.full(shape, 7, dtype=cp.float32))


def test_sparse2dense_zero_hint_is_noop():
    args = _arguments()
    args["max_nnz"] = 0
    args["out"].fill(7)

    _s2d.sparse2dense(**args)

    cp.testing.assert_array_equal(args["out"], cp.full((3, 4), 7, dtype=cp.float32))


def test_sparse2dense_default_stream():
    with cp.cuda.Stream.null:
        args = _arguments()
        args.pop("stream")

        _s2d.sparse2dense(**args)

        cp.testing.assert_array_equal(args["out"], _EXPECTED)


@pytest.mark.parametrize("max_nnz", [1, 2**40])
def test_sparse2dense_max_nnz_is_launch_hint(max_nnz):
    args = _arguments()
    args["max_nnz"] = max_nnz

    _s2d.sparse2dense(**args)

    cp.testing.assert_array_equal(args["out"], _EXPECTED)


@pytest.mark.parametrize("c_switch", [False, True])
def test_sparse2dense_multiple_blocks_and_long_rows(c_switch):
    rng = np.random.default_rng(18)
    major, minor = 257, 19
    lengths = rng.integers(0, 300, size=major)
    indptr = np.concatenate(([0], np.cumsum(lengths)))
    index = rng.integers(0, minor, size=indptr[-1], dtype=np.int64)
    data = rng.integers(-3, 4, size=index.size).astype(np.float64)
    expected = np.zeros((major, minor))
    for row, (start, stop) in enumerate(zip(indptr[:-1], indptr[1:], strict=True)):
        np.add.at(expected[row], index[start:stop], data[start:stop])
    out = cp.zeros(expected.shape, dtype=cp.float64, order="C" if c_switch else "F")

    _s2d.sparse2dense(
        cp.asarray(indptr),
        cp.asarray(index),
        cp.asarray(data),
        out=out,
        major=major,
        minor=minor,
        c_switch=c_switch,
        max_nnz=int(lengths.max()),
        stream=cp.cuda.get_current_stream().ptr,
    )

    cp.testing.assert_array_equal(out, expected)


@pytest.mark.parametrize("physical_shape", [(12,), (2, 6)])
def test_sparse2dense_output_shape_uses_element_count(physical_shape):
    args = _arguments()
    args["out"] = args["out"].reshape(physical_shape)

    _s2d.sparse2dense(**args)

    cp.testing.assert_array_equal(args["out"].ravel(), _EXPECTED.ravel())


def test_sparse2dense_borrowed_views_on_nondefault_stream():
    stream = cp.cuda.Stream(non_blocking=True)
    with stream:
        args = _arguments()
        allocations = {}
        for name in _ARRAY_NAMES:
            array = args[name]
            storage = cp.full(array.size + 2, -123, dtype=array.dtype)
            view = storage[1:-1].reshape(array.shape)
            view[...] = array
            args[name] = view
            allocations[name] = storage
        _s2d.sparse2dense(**args)
        observed = allocations["out"].copy()
    stream.synchronize()

    expected = np.concatenate(([-123], _EXPECTED.ravel(), [-123]))
    cp.testing.assert_array_equal(observed, expected)
    for storage in allocations.values():
        cp.testing.assert_array_equal(storage[[0, -1]], [-123, -123])


def test_sparse2dense_disjoint_views_in_one_allocation():
    args = _arguments()
    storage = cp.empty(args["data"].size + args["out"].size, dtype=cp.float32)
    data_size = args["data"].size
    storage[:data_size] = args["data"]
    args["data"] = storage[:data_size]
    args["out"] = storage[data_size:].reshape(3, 4)
    args["out"].fill(0)

    _s2d.sparse2dense(**args)

    cp.testing.assert_array_equal(args["out"], _EXPECTED)


@pytest.mark.parametrize("dtype", [cp.float32, cp.float64])
def test_sparse2dense_matches_native_atomic_subnormal_behavior(dtype):
    tiny = np.finfo(dtype).tiny
    data = cp.asarray([tiny / 2, tiny / 2], dtype=dtype)
    out = cp.zeros((1, 1), dtype=dtype)

    _s2d.sparse2dense(
        cp.asarray([0, 2], dtype=cp.int32),
        cp.asarray([0, 0], dtype=cp.int32),
        data,
        out=out,
        major=1,
        minor=1,
        c_switch=True,
        max_nnz=2,
        stream=cp.cuda.get_current_stream().ptr,
    )

    # Native float32 atomicAdd flushes subnormals; float64 preserves them.
    cp.testing.assert_array_equal(out, [[0 if dtype == cp.float32 else tiny]])


def test_sparse2dense_managed_memory():
    device = cp.cuda.runtime.getDevice()
    if not cp.cuda.runtime.getDeviceProperties(device)["managedMemory"]:
        pytest.skip("CUDA device does not support managed memory")
    with cp.cuda.using_allocator(cp.cuda.malloc_managed):
        args = _arguments(dtype=cp.float64, index_dtype=cp.int64)

        _s2d.sparse2dense(**args)

        cp.testing.assert_array_equal(args["out"], _EXPECTED)


@rust_only
@pytest.mark.parametrize("indptr", [[-1, 4, 4, 7], [0, 8, 4, 7], [0, 4, 4, 8]])
def test_sparse2dense_skips_invalid_compressed_rows(indptr):
    args = _arguments()
    args["indptr"] = cp.asarray(indptr, dtype=cp.int32)
    expected = _EXPECTED.copy()
    for row, (start, stop) in enumerate(zip(indptr[:-1], indptr[1:], strict=True)):
        if not 0 <= start <= stop <= args["data"].size:
            expected[row] = 0

    _s2d.sparse2dense(**args)

    cp.testing.assert_array_equal(args["out"], expected)


@rust_only
@pytest.mark.parametrize(
    "name,dtype",
    [
        ("indptr", cp.int64),
        ("index", cp.int64),
        ("index", cp.uint32),
        ("data", cp.float64),
        ("out", cp.float64),
        ("data", cp.int32),
    ],
)
def test_sparse2dense_rejects_incompatible_dtypes(name, dtype):
    args = _arguments()
    args[name] = args[name].astype(dtype)

    with pytest.raises(TypeError):
        _s2d.sparse2dense(**args)


@rust_only
@pytest.mark.parametrize("name", _ARRAY_NAMES)
def test_sparse2dense_rejects_host_arrays(name):
    args = _arguments()
    args[name] = cp.asnumpy(args[name])

    with pytest.raises(TypeError):
        _s2d.sparse2dense(**args)


@rust_only
@pytest.mark.parametrize("name", _ARRAY_NAMES)
def test_sparse2dense_rejects_noncontiguous_arrays(name):
    args = _arguments()
    original = args[name]
    args[name] = cp.empty(original.size * 2, dtype=original.dtype)[::2]

    with pytest.raises(ValueError):
        _s2d.sparse2dense(**args)


@rust_only
@pytest.mark.parametrize("name", ["indptr", "index", "data"])
def test_sparse2dense_requires_input_vectors(name):
    args = _arguments()
    args[name] = args[name].reshape(1, -1)

    with pytest.raises(ValueError):
        _s2d.sparse2dense(**args)


@rust_only
@pytest.mark.parametrize("name", _ARRAY_NAMES)
def test_sparse2dense_rejects_inconsistent_array_sizes(name):
    args = _arguments()
    args[name] = args[name].ravel()[:-1]

    with pytest.raises(ValueError):
        _s2d.sparse2dense(**args)


@rust_only
@pytest.mark.parametrize("name", ["major", "minor", "max_nnz"])
def test_sparse2dense_rejects_negative_dimensions_or_hint(name):
    args = _arguments()
    args[name] = -1

    with pytest.raises(ValueError):
        _s2d.sparse2dense(**args)


@rust_only
@pytest.mark.parametrize("name", _ARRAY_NAMES)
def test_sparse2dense_rejects_unaligned_arrays(name):
    args = _arguments()
    original = args[name]
    pointer = cp.cuda.alloc(original.nbytes + 1) + 1
    args[name] = cp.ndarray(original.shape, dtype=original.dtype, memptr=pointer)

    with pytest.raises(ValueError):
        _s2d.sparse2dense(**args)


@rust_only
@pytest.mark.parametrize("name", _ARRAY_NAMES)
def test_sparse2dense_rejects_views_beyond_allocation(name):
    args = _arguments()
    original = args[name]
    memory = cp.cuda.Memory(original.itemsize)
    storage = cp.ndarray(
        (1,), dtype=original.dtype, memptr=cp.cuda.MemoryPointer(memory, 0)
    )
    args[name] = cp.lib.stride_tricks.as_strided(
        storage, shape=original.shape, strides=original.strides
    )
    assert args[name].nbytes > args[name].data.mem.size

    with pytest.raises(ValueError):
        _s2d.sparse2dense(**args)


@rust_only
@pytest.mark.parametrize("name", ["indptr", "index", "data"])
def test_sparse2dense_rejects_output_overlapping_input(name):
    args = _arguments()
    storage = cp.zeros(12, dtype=cp.float32)
    original = args[name]
    args[name] = storage.view(original.dtype)[: original.size]
    args["out"] = storage.reshape(3, 4)

    with pytest.raises(ValueError):
        _s2d.sparse2dense(**args)


@rust_only
@pytest.mark.parametrize("name", _ARRAY_NAMES)
def test_sparse2dense_rejects_arrays_on_another_device(name):
    if cp.cuda.runtime.getDeviceCount() < 2:
        pytest.skip("Requires two CUDA devices")
    with cp.cuda.Device(0):
        args = _arguments()
    with cp.cuda.Device(1):
        args[name] = cp.asarray(cp.asnumpy(args[name]))

    with cp.cuda.Device(0):
        with pytest.raises(ValueError):
            _s2d.sparse2dense(**args)
        assert cp.cuda.runtime.getDevice() == 0
