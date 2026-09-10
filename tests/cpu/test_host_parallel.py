"""Native host staging tests that do not import CuPy or initialize CUDA."""

from __future__ import annotations

import importlib.util
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from threading import Barrier, Event, Thread

import numpy as np
import pytest


@pytest.fixture(scope="module")
def native():
    for name, module in tuple(sys.modules.items()):
        if (
            name.endswith("._rust_cuda")
            and getattr(module, "__backend__", None) == "rust"
        ):
            return module
    path = (
        Path(__file__).resolve().parents[2]
        / "src/rapids_singlecell/_cuda/_rust_cuda.abi3.so"
    )
    if not path.is_file():
        pytest.skip("native Rust extension must be built to test host parallelism")
    spec = importlib.util.spec_from_file_location("_rust_cuda", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def csr_example(index_dtype, pointer_dtype, rows=8193):
    # Duplicate columns, empty rows, and a nonzero first pointer exercise the
    # advancing lower bound and sliced CSR inputs independently of SciPy.
    row = np.array([0, 3, 3, 7, 11, 14, 20, 29, 31], dtype=index_dtype)
    indices = np.concatenate(
        [np.array([99], dtype=index_dtype), np.tile(row, rows - 1)]
    )
    indptr = np.arange(rows + 1, dtype=pointer_dtype) * row.size + 1
    indptr[19:] -= row.size
    cuts = np.array([0, 3, 3, 4, 15, 31, 32], dtype=np.int32)
    expected = np.stack(
        [
            [
                start + np.searchsorted(indices[start:stop], cut)
                for start, stop in zip(indptr[:-1], indptr[1:], strict=True)
            ]
            for cut in cuts
        ]
    )
    return indices, indptr, cuts, expected


@pytest.mark.parametrize("workers", [0, 1, 2, 4])
@pytest.mark.parametrize("index_dtype", [np.int32, np.int64])
@pytest.mark.parametrize("pointer_dtype", [np.int32, np.int64])
def test_parallel_boundaries_match_searchsorted(
    native, workers, index_dtype, pointer_dtype
):
    backend = native._wilcoxon_sparse_cuda
    indices, indptr, cuts, expected = csr_example(index_dtype, pointer_dtype)
    output = np.empty_like(expected, dtype=pointer_dtype)
    previous = backend._set_host_worker_limit(workers)
    try:
        backend.csr_row_boundaries_host(indices, indptr, cuts, output, n_cols=32)
    finally:
        backend._set_host_worker_limit(previous)
    np.testing.assert_array_equal(output, expected)


def test_caller_worker_limits_are_thread_local_and_shared_between_modules(native):
    barrier = Barrier(4)
    indices, indptr, cuts, expected = csr_example(np.int32, np.int64)

    def shard(workers):
        backend = native._wilcoxon_sparse_cuda
        previous = backend._set_host_worker_limit(workers)
        try:
            barrier.wait(timeout=10)
            output = np.empty_like(expected, dtype=np.int64)
            backend.csr_row_boundaries_host(indices, indptr, cuts, output, n_cols=32)
            assert native._rank_stream_cuda._set_host_worker_limit(workers) == workers
            np.testing.assert_array_equal(output, expected)
        finally:
            backend._set_host_worker_limit(previous)

    with ThreadPoolExecutor(max_workers=4) as pool:
        list(pool.map(shard, [1, 2, 3, 4]))


@pytest.mark.parametrize(
    "invalid", ["readonly", "noncontiguous", "bad_pointer", "bad_cut"]
)
def test_invalid_boundaries_leave_output_untouched(native, invalid):
    indices = np.array([0, 3, 5], dtype=np.int32)
    indptr = np.array([0, 1, 3], dtype=np.int32)
    cuts = np.array([0, 3, 6], dtype=np.int32)
    output = np.full((3, 2), 17, dtype=np.int32)
    if invalid == "readonly":
        output.flags.writeable = False
    elif invalid == "noncontiguous":
        output = np.full((2, 3), 17, dtype=np.int32).T
    elif invalid == "bad_pointer":
        indptr[-1] = 4
    else:
        cuts[-1] = 7
    with pytest.raises(ValueError):
        native._wilcoxon_sparse_cuda.csr_row_boundaries_host(
            indices, indptr, cuts, output, n_cols=6
        )
    np.testing.assert_array_equal(output, 17)


def test_boundary_search_detaches_from_interpreter(native):
    # A long switch interval prevents ordinary bytecode scheduling from making
    # this pass. The observer mutates the Python input once the native search
    # detaches; workers must retain the original owned snapshot throughout.
    backend = native._wilcoxon_sparse_cuda
    rows = 100_000
    indices = np.tile(np.arange(64, dtype=np.int32), rows)
    indptr = np.arange(rows + 1, dtype=np.int32) * 64
    cuts = np.arange(65, dtype=np.int32)
    output = np.empty((cuts.size, rows), dtype=np.int32)
    ready, start, progressed = Event(), Event(), Event()

    def observer():
        ready.set()
        start.wait(timeout=10)
        indices.fill(0)
        progressed.set()

    thread = Thread(target=observer)
    thread.start()
    ready.wait(timeout=10)
    interval = sys.getswitchinterval()
    previous = backend._set_host_worker_limit(1)
    try:
        sys.setswitchinterval(10)
        start.set()
        backend.csr_row_boundaries_host(indices, indptr, cuts, output, n_cols=64)
        assert progressed.is_set(), "native host search held the interpreter lock"
    finally:
        sys.setswitchinterval(interval)
        backend._set_host_worker_limit(previous)
        start.set()
        thread.join(timeout=10)
    np.testing.assert_array_equal(output[17], indptr[:-1] + 17)
