from __future__ import annotations

import ctypes
import sys
from types import SimpleNamespace

import cupy as cp
import numpy as np
from cupy.cuda import cublas, runtime  # type: ignore[attr-defined]
from cupyx.scipy.sparse import isspmatrix_csr

from rapids_singlecell._cuda import _sparse2dense_cuda

from ._helper import _check_matrix_for_zero_genes


@cp.fuse()  # type: ignore[attr-defined]
def _min_nonzero_abs(values):
    return cp.min(cp.where(values == 0, np.inf, cp.abs(values)))


@cp.fuse()  # type: ignore[attr-defined]
def _max_abs(values):
    return cp.max(cp.abs(values))


def _mantissa_bits(values):
    low, high = cp.stack((_min_nonzero_abs(values), _max_abs(values))).get()
    if not np.isfinite(high):
        raise ValueError("Ozaki PCA requires finite input.")
    # Preserve every binary64 significand under a shared row/column exponent.
    # Include the sign bit and a guard bit at the upper end of the range.
    return 64 if high == 0 else max(64, 55 + int(np.frexp(high)[1] - np.frexp(low)[1]))


def _ozaki_option(library, handle, name, value):
    # CuPy exposes the BLAS routines but does not yet wrap the mantissa controls.
    status = getattr(library, name)(ctypes.c_void_p(handle), ctypes.c_int(value))
    if status:
        raise cublas.CUBLASError(status)


def _run_tiled_pca(X, n_comps, compute_precision):
    n_rows, n_cols = X.shape
    if not 1 <= n_comps <= min(n_rows, n_cols) or n_rows < 2:
        raise ValueError("Tiled PCA requires at least two observations and 1 <= n_comps <= min(X.shape).")
    sparse = isspmatrix_csr(X)
    if sparse:
        if not X.has_canonical_format:
            X = X.copy()
            X.sum_duplicates()
        _check_matrix_for_zero_genes(X)
        max_nnz = int(cp.diff(X.indptr).max())

    ozaki = compute_precision == "ozaki"
    if compute_precision in {"ozaki", "tf32"} and int(cp.cuda.Device().compute_capability) < 80:
        raise ValueError(f"{compute_precision} PCA requires an Ampere or newer GPU.")
    dtype = X.dtype
    blas_dtype = runtime.CUDA_R_64F if dtype == cp.float64 else runtime.CUDA_R_32F
    compute = {
        "float32": cublas.CUBLAS_COMPUTE_32F_PEDANTIC,
        "tf32": cublas.CUBLAS_COMPUTE_32F_FAST_TF32,
        "float64": cublas.CUBLAS_COMPUTE_64F_PEDANTIC,
        "ozaki": 79,  # CUBLAS_COMPUTE_64F_EMULATED_FIXEDPOINT
    }[compute_precision]
    scalars = np.array([0, 1], dtype=dtype)
    zero, one = scalars.ctypes.data, scalars.ctypes.data + scalars.itemsize
    handle = cublas.create()
    try:
        cublas.setStream(handle, cp.cuda.get_current_stream().ptr)
        if ozaki:
            if cublas.getVersion(handle) < 131000:
                raise RuntimeError("Ozaki PCA requires cuBLAS 13.10 or newer (CUDA Toolkit 13.5).")
            library = ctypes.CDLL("cublas64_13.dll" if sys.platform == "win32" else "libcublas.so.13")
            cublas.setMathMode(handle, 8)  # CUBLAS_FP64_EMULATED_FIXEDPOINT_MATH
            _ozaki_option(library, handle, "cublasSetEmulationStrategy", 2)  # EAGER
            _ozaki_option(library, handle, "cublasSetFixedPointEmulationMantissaBitOffset", 0)
            gram_bits = _mantissa_bits(X.data) if sparse else 0
            fixed_gram = 0 < gram_bits <= 96
            _ozaki_option(
                library, handle, "cublasSetFixedPointEmulationMaxMantissaBitCount", gram_bits if fixed_gram else 96
            )
            _ozaki_option(library, handle, "cublasSetFixedPointEmulationMantissaControl", int(fixed_gram))

        def gemm(left, right, result, shape, ops, *, beta):
            cublas.gemmEx(
                handle,
                *ops,
                *shape,
                one,
                left.data.ptr,
                blas_dtype,
                n_cols,
                right.data.ptr,
                blas_dtype,
                n_cols,
                beta,
                result.data.ptr,
                blas_dtype,
                shape[0],
                compute,
                cublas.CUBLAS_GEMM_DEFAULT,
            )

        def tiles(center=None):
            for start in range(0, n_rows, tile_size):
                rows = min(tile_size, n_rows - start)
                if sparse:
                    tile = cp.zeros((rows, n_cols), dtype=dtype)
                    _sparse2dense_cuda.sparse2dense(
                        X.indptr[start : start + rows + 1],
                        X.indices,
                        X.data,
                        out=tile,
                        major=rows,
                        minor=n_cols,
                        c_switch=True,
                        max_nnz=max_nnz,
                        stream=cp.cuda.get_current_stream().ptr,
                    )
                else:
                    tile = X[start : start + rows]
                if center is not None:
                    output = tile if sparse else cp.empty((rows, n_cols), dtype=dtype)
                    tile = cp.subtract(tile, center, out=output)
                yield start, tile

        tile_size = min(4096, max(1, (64 << 20) // (n_cols * dtype.itemsize)))
        gram = cp.zeros((n_cols, n_cols), dtype=dtype, order="F")

        def build_gram(center, *, collect_mean=False):
            for start, tile in tiles(center):
                beta = one if start else zero
                # A C-order cell-by-gene tile is a column-major gene-by-cell matrix for BLAS.
                if ozaki:
                    cublas.dsyrk(
                        handle,
                        cublas.CUBLAS_FILL_MODE_UPPER,
                        cublas.CUBLAS_OP_N,
                        n_cols,
                        len(tile),
                        one,
                        tile.data.ptr,
                        n_cols,
                        beta,
                        gram.data.ptr,
                        n_cols,
                    )
                else:
                    gemm(
                        tile,
                        tile,
                        gram,
                        (n_cols, n_cols, len(tile)),
                        (cublas.CUBLAS_OP_N, cublas.CUBLAS_OP_T),
                        beta=beta,
                    )
                if collect_mean:
                    mean[:] += tile.sum(axis=0, dtype=cp.float64)

        if sparse:
            mean = cp.zeros(n_cols, dtype=cp.float64)
            if ozaki:
                build_gram(None, collect_mean=True)
            else:
                for _, tile in tiles():
                    mean += tile.sum(axis=0, dtype=cp.float64)
            mean /= n_rows
        else:
            mean = X.mean(axis=0, dtype=cp.float64)
            if ozaki and not cp.isfinite(mean).all():
                raise ValueError("Ozaki PCA requires finite input.")

        center = mean
        if ozaki and sparse:
            # Use the sparse Gram only when covariance subtraction loses at most
            # two decimal digits on every diagonal. Otherwise explicitly center.
            second_moment = gram.diagonal() / n_rows
            if cp.all(cp.isfinite(second_moment) & (mean**2 <= 0.99 * second_moment)):
                center = None
            else:
                fixed_gram = False
                _ozaki_option(library, handle, "cublasSetFixedPointEmulationMantissaControl", 0)
        if center is not None:
            build_gram(center)
        gram /= n_rows if sparse else n_rows - 1
        if center is None:
            gram -= mean[:, None] * mean[None, :]
        values, vectors = cp.linalg.eigh(gram, UPLO="U")
        components = cp.ascontiguousarray(vectors[:, -n_comps:][:, ::-1].T)
        variance = values[-n_comps:][::-1]
        variance_ratio = variance / values.sum()
        if ozaki and fixed_gram:
            bits = max(gram_bits, _mantissa_bits(components))
            if bits <= 96:
                _ozaki_option(library, handle, "cublasSetFixedPointEmulationMaxMantissaBitCount", bits)
            else:
                _ozaki_option(library, handle, "cublasSetFixedPointEmulationMantissaControl", 0)

        offset = (components * mean).sum(axis=1) if center is None else None
        scores = cp.empty((n_rows, n_comps), dtype=dtype)
        for start, tile in tiles(center):
            block = scores[start : start + len(tile)]
            gemm(
                components,
                tile,
                block,
                (n_comps, len(tile), n_cols),
                (cublas.CUBLAS_OP_T, cublas.CUBLAS_OP_N),
                beta=zero,
            )
            if offset is not None:
                block -= offset
        model = SimpleNamespace(
            components_=components.get(),
            explained_variance_=variance.get(),
            explained_variance_ratio_=variance_ratio.get(),
        )
        return model, scores.get()
    finally:
        cublas.destroy(handle)
