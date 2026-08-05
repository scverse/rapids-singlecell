from __future__ import annotations

import cupy as cp
from cupyx.scipy.sparse import csc_matrix, csr_matrix

from rapids_singlecell.preprocessing._utils import _sparse_to_dense


def _make_small_csr(dtype=cp.float32):
    # 3x4
    indptr = cp.asarray([0, 2, 3, 4], dtype=cp.int32)
    indices = cp.asarray([0, 2, 1, 3], dtype=cp.int32)
    data = cp.asarray([1, 5, 2, 3], dtype=dtype)
    return csr_matrix((data, indices, indptr), shape=(3, 4))


def _make_small_csc(dtype=cp.float32):
    # 3x4; transpose of above to ensure different pattern
    indptr = cp.asarray([0, 1, 3, 3, 4], dtype=cp.int32)
    indices = cp.asarray([0, 0, 2, 1], dtype=cp.int32)
    data = cp.asarray([1, 5, 2, 3], dtype=dtype)
    return csc_matrix((data, indices, indptr), shape=(3, 4))


def test_sparse2dense_csr_c_order():
    X = _make_small_csr(cp.float32)
    got = _sparse_to_dense(X, order="C")
    exp = X.toarray()
    cp.testing.assert_array_equal(got, exp)


def test_sparse2dense_csr_f_order():
    X = _make_small_csr(cp.float64)
    got = _sparse_to_dense(X, order="F")
    exp = X.toarray()
    cp.testing.assert_array_equal(got, exp)


def test_sparse2dense_csc_c_order():
    X = _make_small_csc(cp.float32)
    got = _sparse_to_dense(X, order="C")
    exp = X.toarray()
    cp.testing.assert_array_equal(got, exp)


def test_sparse2dense_csc_f_order():
    X = _make_small_csc(cp.float64)
    got = _sparse_to_dense(X, order="F")
    exp = X.toarray()
    cp.testing.assert_array_equal(got, exp)


def test_sparse2dense_random_shapes_seeded():
    rs = cp.random.RandomState(123)
    for dtype in (cp.float32, cp.float64):
        for m, n in [(1, 1), (2, 3), (7, 5), (16, 16)]:
            dense = rs.rand(m, n).astype(dtype)
            dense[dense < 0.7] = 0  # sparsify
            csr = csr_matrix(dense)
            csc = csc_matrix(dense)
            got_csr_c = _sparse_to_dense(csr, order="C")
            got_csr_f = _sparse_to_dense(csr, order="F")
            got_csc_c = _sparse_to_dense(csc, order="C")
            got_csc_f = _sparse_to_dense(csc, order="F")
            exp = csr.toarray()
            cp.testing.assert_array_equal(got_csr_c, exp)
            cp.testing.assert_array_equal(got_csr_f, exp)
            cp.testing.assert_array_equal(got_csc_c, exp)
            cp.testing.assert_array_equal(got_csc_f, exp)
