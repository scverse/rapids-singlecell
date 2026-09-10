"""Check an explicit Rust extension with the current CUDA 12 or 13 CuPy stack."""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
from pathlib import Path

import cupy as cp
import numpy as np
from cupy_backends.cuda.libs import cublas


def check_sparse_layouts(native):
    cases = 0
    for dtype in (cp.float32, cp.float64):
        for index_dtype in (cp.int32, cp.int64):
            for order in ("C", "F"):
                with cp.cuda.Stream(non_blocking=True) as stream:
                    output = cp.zeros((2, 2), dtype=dtype, order=order)
                    pointers = cp.asarray([0, 2, 3], dtype=index_dtype)
                    indices = cp.asarray([0, 1, 0], dtype=index_dtype)
                    values = cp.asarray([2, 4, 6], dtype=dtype)
                    native._sparse2dense_cuda.sparse2dense(
                        pointers,
                        indices,
                        values,
                        out=output,
                        major=2,
                        minor=2,
                        c_switch=order == "C",
                        max_nnz=2,
                        stream=stream.ptr,
                    )
                stream.synchronize()
                cp.testing.assert_array_equal(output, [[2, 4], [6, 0]])
                cases += 1
    return cases


def check_blas_capture(native, dtype, pointer_mode):
    rng = np.random.default_rng(48)
    cells, features, clusters, batches = 1031, 4, 2, 3
    categories = rng.integers(0, batches, cells, dtype=np.int32)
    data = (rng.normal(size=(cells, features)) + categories[:, None]).astype(dtype)
    weights = rng.uniform(0.1, 1, (cells, clusters)).astype(dtype)
    weights /= weights.sum(axis=1, keepdims=True)
    counts = np.asarray(
        [
            weights[categories == batch].sum(axis=0, dtype=np.float64)
            for batch in range(batches)
        ],
        dtype=dtype,
    )
    penalties = np.full((batches, clusters), 50_000, dtype=dtype)
    expected = data.astype(np.float64)
    for cluster in range(clusters):
        weight = weights[:, cluster].astype(np.float64)
        weighted = data * weight[:, None]
        rhs = np.vstack(
            [weighted.sum(axis=0)]
            + [weighted[categories == batch].sum(axis=0) for batch in range(batches)]
        )
        count = counts[:, cluster].astype(np.float64)
        gram = np.diag(np.r_[count.sum(), count + penalties[:, cluster]])
        gram[0, 1:], gram[1:, 0] = count, count
        coefficients = np.linalg.solve(gram, rhs)
        expected -= weight[:, None] * coefficients[1:][categories]
    order = np.argsort(categories, kind="stable").astype(np.int32)
    offsets = np.r_[0, np.cumsum(np.bincount(categories, minlength=batches))].astype(
        np.int32
    )
    with cp.cuda.Stream(non_blocking=True) as stream:
        x = cp.asarray(data)
        arguments = {
            "R": cp.asarray(weights),
            "O": cp.asarray(counts),
            "cats": cp.asarray(categories),
            "cat_offsets": cp.asarray(offsets),
            "cell_indices": cp.asarray(order),
            "lambda_kb": cp.asarray(penalties),
            "n_cells": cells,
            "n_pcs": features,
            "n_clusters": clusters,
            "n_batches": batches,
            "Z": cp.empty_like(x),
            "inv_mat": cp.empty((batches + 1, batches + 1), dtype=dtype),
            "R_col": cp.empty(cells, dtype=dtype),
            "Phi_t_diag_R_X": cp.empty((batches + 1, features), dtype=dtype),
            "W": cp.empty((batches + 1, features), dtype=dtype),
            "g_factor": cp.empty(batches, dtype=dtype),
            "g_P_row0": cp.empty(batches, dtype=dtype),
            "stream": stream.ptr,
            "handle": cp.cuda.device.get_cublas_handle(),
        }
    correction = native._harmony_correction_cuda.correction_fast
    correction(x, **arguments)
    stream.synchronize()
    handle = arguments["handle"]
    previous_mode = cublas.getPointerMode(handle)
    try:
        cublas.setPointerMode(handle, pointer_mode)
        with stream:
            stream.begin_capture()
            try:
                correction(x, **arguments)
            finally:
                graph = stream.end_capture()
            arguments["Z"].fill(cp.nan)
            graph.launch(stream)
        stream.synchronize()
        assert cublas.getPointerMode(handle) == pointer_mode
    finally:
        cublas.setPointerMode(handle, previous_mode)
    tolerance = 1e-5 if dtype == np.float32 else 1e-11
    np.testing.assert_allclose(
        arguments["Z"].get(), expected, atol=tolerance, rtol=tolerance
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--backend", required=True, type=Path, help="Built _rust_cuda.abi3.so"
    )
    parser.add_argument("--expect-cuda-major", type=int, choices=(12, 13))
    options = parser.parse_args()
    backend = options.backend.resolve(strict=True)
    runtime = cp.cuda.runtime.runtimeGetVersion()
    if (
        options.expect_cuda_major is not None
        and runtime // 1000 != options.expect_cuda_major
    ):
        parser.error(
            f"Expected CUDA {options.expect_cuda_major}, found runtime {runtime}"
        )
    spec = importlib.util.spec_from_file_location("_rust_cuda", backend)
    native = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(native)
    assert native.__backend__ == "rust"
    assert all(getattr(native, name).__backend__ == "rust" for name in native.__all__)
    sparse_cases = check_sparse_layouts(native)
    for dtype in (np.float32, np.float64):
        for pointer_mode in (0, 1):
            check_blas_capture(native, dtype, pointer_mode)
    device = cp.cuda.runtime.getDeviceProperties(cp.cuda.Device().id)["name"]
    print(
        json.dumps(
            {
                "backend": str(backend),
                "backend_sha256": hashlib.sha256(backend.read_bytes()).hexdigest(),
                "cupy": cp.__version__,
                "cuda_runtime": runtime,
                "cuda_driver": cp.cuda.runtime.driverGetVersion(),
                "cublas": cublas.getVersion(cp.cuda.device.get_cublas_handle()),
                "device": device.decode() if isinstance(device, bytes) else device,
                "sparse_dtype_index_layout_cases": sparse_cases,
                "blas_graph_pointer_mode_cases": 4,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
