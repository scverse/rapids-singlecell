from __future__ import annotations

import cupy as cp
import decoupler as dc_cpu
import numpy as np
import pandas as pd
import pytest
import scipy.sparse as sps
from anndata import AnnData

import rapids_singlecell.decoupler_gpu as dc
from rapids_singlecell.decoupler_gpu import _method_gsva as g


def _data(kcdf):
    rng = np.random.default_rng(820)
    mat = (
        rng.poisson(3, size=(9, 37)) if kcdf == "poisson" else rng.normal(size=(9, 37))
    ).astype(np.float32)
    frame = pd.DataFrame(mat, columns=[f"g{i}" for i in range(mat.shape[1])])
    sets = [np.arange(7), np.arange(5, 18), np.arange(20, 37, 2)]
    net = pd.DataFrame(
        {
            "source": np.repeat(["a", "b", "c"], list(map(len, sets))),
            "target": frame.columns[np.concatenate(sets)],
        }
    )
    return frame, net


def test_gsva_standard_registration_preserves_metadata():
    from rapids_singlecell.decoupler_gpu._helper._Method import Method

    assert type(dc.gsva) is Method
    assert type(dc.aucell) is Method
    assert dc.gsva._method is g._gsva
    assert dc.gsva.func is g._func_gsva
    assert dc.gsva.__doc__ == g._func_gsva.__doc__
    pd.testing.assert_frame_equal(dc.gsva.meta(), g._gsva.meta())


@pytest.mark.parametrize("kcdf", ["gaussian", "poisson", None])
@pytest.mark.parametrize("nonblocking", [False, True])
@pytest.mark.parametrize(
    ("maxdiff", "absrnk", "tau"),
    [(True, False, 1), (True, True, 2), (False, False, 0.5)],
)
def test_public_gsva_matches_decoupler(kcdf, nonblocking, maxdiff, absrnk, tau):
    frame, net = _data(kcdf)
    kwargs = {
        "tmin": 0,
        "kcdf": kcdf,
        "maxdiff": maxdiff,
        "absrnk": absrnk,
        "tau": tau,
    }
    expected, _ = dc_cpu.mt.gsva(frame, net, **kwargs)
    # A tiny bsize must not split the observation-global density transform.
    stream = cp.cuda.Stream(non_blocking=True) if nonblocking else cp.cuda.Stream.null
    with stream:
        actual, pvalue = dc.gsva(frame, net, bsize=2, **kwargs)
    assert pvalue is None
    pd.testing.assert_frame_equal(actual, expected, rtol=1e-6, atol=1e-6)


@pytest.mark.parametrize("kcdf", ["gaussian", "poisson", None])
@pytest.mark.parametrize("sparse", [False, True])
def test_dask_filters_empty_values_before_ranking(kcdf, sparse):
    import dask.array as da

    frame, net = _data("poisson")
    frame.iloc[0, :] = 0
    frame.iloc[:, 3] = 0
    expected, _ = dc_cpu.mt.gsva(frame, net, tmin=0, kcdf=kcdf)
    mat = frame.to_numpy()
    if sparse:
        mat = sps.csr_matrix(mat)
    adata = AnnData(
        da.from_array(mat, chunks=(2, 11)),
        obs=pd.DataFrame(index=frame.index.astype(str)),
        var=pd.DataFrame(index=frame.columns),
    )
    result = dc.gsva(adata, net, tmin=0, kcdf=kcdf, bsize=2)
    assert result is not None  # Removing empty observations returns a new AnnData.
    pd.testing.assert_frame_equal(result.obsm["score_gsva"], expected)
    assert adata.n_obs == frame.shape[0]


@pytest.mark.parametrize("value", [np.nan, np.inf, -np.inf])
@pytest.mark.parametrize("sparse", [False, True])
def test_dask_rejects_nonfinite_values(value, sparse):
    import dask.array as da

    frame, net = _data("gaussian")
    mat = frame.to_numpy().copy()
    mat[0, 0] = value
    if sparse:
        mat = sps.csr_matrix(mat)
    adata = AnnData(
        da.from_array(mat, chunks=(2, 11)), var=pd.DataFrame(index=frame.columns)
    )
    with pytest.raises(ValueError, match="non finite"):
        dc.gsva(adata, net, tmin=0, kcdf=None)


def test_sparse_anndata_and_dask_match_dense():
    import dask.array as da

    frame, net = _data("gaussian")
    dense, _ = dc.gsva(frame, net, tmin=0)

    adata = AnnData(
        sps.csr_matrix(frame.to_numpy()), var=pd.DataFrame(index=frame.columns)
    )
    dc.gsva(adata, net, tmin=0)
    pd.testing.assert_frame_equal(adata.obsm["score_gsva"], dense)

    dense_ecdf, _ = dc.gsva(frame, net, tmin=0, kcdf=None)
    dc.gsva(adata, net, tmin=0, kcdf=None)
    pd.testing.assert_frame_equal(adata.obsm["score_gsva"], dense_ecdf)

    adata.X = da.from_array(frame.to_numpy(), chunks=(2, 11))
    dc.gsva(adata, net, tmin=0)
    pd.testing.assert_frame_equal(adata.obsm["score_gsva"], dense)


@pytest.mark.parametrize("nobs", [1, 9])
@pytest.mark.parametrize("kcdf", ["gaussian", "poisson", None])
def test_sparse_conversion_preserves_duplicate_values(nobs, kcdf, monkeypatch):
    from cupyx.scipy.sparse import csr_matrix

    frame, _ = _data("poisson")
    mat = sps.csr_matrix(frame.iloc[:nobs].to_numpy())
    duplicated = csr_matrix(
        sps.csr_matrix(
            (np.repeat(mat.data / 2, 2), np.repeat(mat.indices, 2), mat.indptr * 2),
            shape=mat.shape,
        )
    )
    convert = g._sparse_to_dense

    def checked_convert(value):
        assert value.has_canonical_format
        return convert(value)

    monkeypatch.setattr(g, "_sparse_to_dense", checked_convert)
    kwargs = {
        "cnct": cp.arange(7, dtype=cp.int32),
        "starts": cp.asarray([0], dtype=cp.int32),
        "offsets": cp.asarray([7], dtype=cp.int32),
        "kcdf": kcdf,
    }
    cp.cuda.get_current_stream().synchronize()
    with cp.cuda.Stream(non_blocking=True):
        expected, _ = g._func_gsva(cp.asarray(mat.toarray()), **kwargs)
        actual, _ = g._func_gsva(duplicated, **kwargs)
    np.testing.assert_allclose(actual, expected, rtol=1e-6, atol=1e-6)
    assert not duplicated.has_canonical_format


@pytest.mark.parametrize("raw,layer", [(False, None), (True, None), (False, "signal")])
def test_anndata_representation_and_return(raw, layer):
    frame, net = _data("gaussian")
    adata = AnnData(
        sps.csr_matrix(frame.to_numpy()), var=pd.DataFrame(index=frame.columns)
    )
    adata.raw = adata.copy()
    adata = adata[:, :30].copy()
    adata.layers["signal"] = adata.X.copy()
    adata.X = -adata.X
    selected = frame if raw else frame.iloc[:, :30]
    if not raw and layer is None:
        selected = -selected
    expected, _ = dc_cpu.mt.gsva(selected, net, tmin=0, kcdf=None)

    result = dc.gsva(adata, net, raw=raw, layer=layer, tmin=0, kcdf=None, bsize=2)

    assert result is None
    assert "padj_gsva" not in adata.obsm
    pd.testing.assert_frame_equal(adata.obsm["score_gsva"], expected)
    assert adata.n_vars == 30
    assert adata.raw.n_vars == frame.shape[1]


@pytest.mark.parametrize("backed", [False, True])
@pytest.mark.parametrize("bsize", [1, 2, 9])
def test_sparse_anndata_batching_matches_decoupler(backed, bsize, tmp_path):
    from anndata import read_h5ad

    frame, net = _data("gaussian")
    adata = AnnData(
        sps.csr_matrix(frame.to_numpy()), var=pd.DataFrame(index=frame.columns)
    )
    if backed:
        path = tmp_path / "gsva.h5ad"
        adata.write_h5ad(path)
        adata = read_h5ad(path, backed="r")
        reference = read_h5ad(path, backed="r")
    else:
        reference = adata.copy()
    try:
        # In-memory input always uses the full density; backed input follows
        # Decoupler's per-batch density, including the final single row.
        reference_bsize = bsize if backed else adata.n_obs
        dc_cpu.mt.gsva(reference, net, tmin=0, kcdf=None, bsize=reference_bsize)
        assert dc.gsva(adata, net, tmin=0, kcdf=None, bsize=bsize) is None
        pd.testing.assert_frame_equal(
            adata.obsm["score_gsva"], reference.obsm["score_gsva"]
        )
        assert "padj_gsva" not in adata.obsm
    finally:
        if backed:
            adata.file.close()
            reference.file.close()


def test_sparse_ecdf_matches_dense_with_zeros_ties_and_negative_values():
    mat = np.array(
        [
            [0, -2, 0, 3, 1],
            [0, 0, 2, 3, 0],
            [-1, 0, 2, 0, 1],
            [0, -2, 0, 0, 0],
        ],
        dtype=np.float32,
    )
    frame = pd.DataFrame(mat, columns=[f"g{i}" for i in range(mat.shape[1])])
    net = pd.DataFrame(
        {
            "source": ["a", "a", "a", "b", "b", "b"],
            "target": ["g0", "g1", "g2", "g2", "g3", "g4"],
        }
    )
    expected, _ = dc_cpu.mt.gsva(frame, net, tmin=0, kcdf=None)
    adata = AnnData(sps.csr_matrix(mat), var=pd.DataFrame(index=frame.columns))
    dc.gsva(adata, net, tmin=0, kcdf=None)
    pd.testing.assert_frame_equal(adata.obsm["score_gsva"], expected)


@pytest.mark.parametrize(
    "mat",
    [
        sps.csr_matrix([[0, 0, 0], [0, 1, 2], [3, 2, 1]], dtype=np.float32),
        sps.csr_matrix((2, 3), dtype=np.float32),
        sps.csr_matrix(
            (
                np.array([-0.0, 1, 0, 1, 2], dtype=np.float32),
                [0, 1, 0, 1, 2],
                [0, 2, 5],
            ),
            shape=(2, 3),
        ),
        sps.csr_matrix(
            (
                np.array([1, 2, 2, 1, 3, 1], dtype=np.float32),
                [0, 0, 1, 0, 1, 2],
                [0, 3, 6],
            ),
            shape=(2, 3),
        ),
    ],
    ids=["empty_row", "all_zero", "signed_zero", "duplicate_indices"],
)
def test_sparse_ecdf_rank_edge_cases_match_decoupler(mat):
    from cupyx.scipy.sparse import csr_matrix

    expected = dc_cpu.mt._gsva._rankmat(dc_cpu.mt._gsva._density(mat.toarray(), None))
    targets = cp.arange(mat.shape[1], dtype=cp.int32)
    dos, srs, cnct = g._rank_sparse_ecdf(csr_matrix(mat), targets)
    np.testing.assert_array_equal(dos.get(), expected[0])
    np.testing.assert_array_equal(srs.get(), expected[1])
    np.testing.assert_array_equal(cnct.get(), targets.get())


@pytest.mark.parametrize("nvar", [8193, 65537])
def test_sparse_ecdf_outside_fast_path_matches_dense(nvar):
    from cupyx.scipy.sparse import csr_matrix

    rng = np.random.default_rng(31)
    if nvar == 8193:
        mat = sps.csr_matrix(rng.uniform(1, 2, size=(2, nvar)).astype(np.float32))
    else:
        mat = sps.csr_matrix(
            (np.ones(3, dtype=np.float32), [0, nvar - 1, nvar - 2], [0, 2, 3]),
            shape=(2, nvar),
        )
    kwargs = {
        "cnct": cp.asarray([0, 1, nvar - 1], dtype=cp.int32),
        "starts": cp.asarray([0], dtype=cp.int32),
        "offsets": cp.asarray([3], dtype=cp.int32),
        "kcdf": None,
    }
    expected, _ = g._func_gsva(cp.asarray(mat.toarray()), **kwargs)
    actual, _ = g._func_gsva(csr_matrix(mat), **kwargs)
    np.testing.assert_allclose(actual, expected, rtol=1e-6, atol=1e-6)


def test_rank_ties_and_single_observation_match_decoupler():
    mat = np.array([[1, 1, 2, 0], [0, 0, 0, 0]], dtype=np.float32)
    expected_dos, expected_srs = dc_cpu.mt._gsva._rankmat(mat)
    dos, srs = g._rankmat(cp.asarray(mat))
    np.testing.assert_array_equal(dos.get(), expected_dos)
    np.testing.assert_array_equal(srs.get(), expected_srs)


@pytest.mark.parametrize("tau", [0, 0.5, 1, 2])
def test_signed_symmetric_walk_matches_decoupler(tau):
    frame = pd.DataFrame(
        [np.arange(37, 0, -1, dtype=np.float32)],
        columns=[f"g{i}" for i in range(37)],
    )
    net = pd.DataFrame({"source": ["a"] * 3, "target": ["g10", "g16", "g26"]})
    kwargs = {"tmin": 0, "kcdf": None, "maxdiff": False, "tau": tau}
    expected, _ = dc_cpu.mt.gsva(frame, net, **kwargs)
    actual, _ = dc.gsva(frame, net, **kwargs)
    pd.testing.assert_frame_equal(actual, expected, rtol=1e-6, atol=1e-6)


@pytest.mark.parametrize("factor", [1, 11, 683], ids=["warp", "sorted", "fallback"])
@pytest.mark.parametrize("tau", [0, 0.5, 1, 2])
def test_signed_symmetric_walk_all_score_kernels(factor, tau):
    # Scaling the same symmetric walk dispatches sets of 3, 33, and 2049
    # targets. Compare the reference scorer directly to avoid quadratic ranking.
    nvar = 37 * factor
    targets = np.concatenate(
        [np.arange(i * factor, (i + 1) * factor, dtype=np.int32) for i in (10, 16, 26)]
    )
    dos = np.arange(1, nvar + 1, dtype=np.int32)[None, :]
    srs = np.abs(2 * dos - nvar - 2) // 2
    expected = dc_cpu.mt._gsva._ks_fset(dos, srs, targets + 1, False, False, tau)
    actual = g._score_sets(
        cp.asarray(dos),
        cp.asarray(srs),
        cp.asarray(targets),
        cp.asarray([0], dtype=cp.int32),
        cp.asarray([targets.size], dtype=cp.int32),
        maxdiff=False,
        absrnk=False,
        tau=tau,
    )
    np.testing.assert_allclose(actual.get()[:, 0], expected, rtol=1e-6, atol=1e-6)


@pytest.mark.parametrize(
    ("maxdiff", "absrnk"), [(True, False), (True, True), (False, False)]
)
def test_degenerate_sets_match_decoupler(maxdiff, absrnk):
    mat = np.arange(5, dtype=np.float32)[None, :]
    sets = [np.arange(5), np.array([2])]
    sizes = np.array(list(map(len, sets)), dtype=np.int32)
    cnct = np.concatenate(sets).astype(np.int32)
    starts = np.r_[0, np.cumsum(sizes)[:-1]].astype(np.int32)
    expected, _ = dc_cpu.mt._gsva._func_gsva(
        mat, cnct, starts, sizes, maxdiff=maxdiff, absrnk=absrnk
    )
    actual, _ = g._func_gsva(
        cp.asarray(mat),
        cnct=cp.asarray(cnct),
        starts=cp.asarray(starts),
        offsets=cp.asarray(sizes),
        maxdiff=maxdiff,
        absrnk=absrnk,
    )
    np.testing.assert_allclose(actual, expected, equal_nan=True)


def test_validation_matches_decoupler():
    mat = cp.ones((3, 4), dtype=cp.float32)
    with pytest.raises(AssertionError, match="kcdf must be"):
        g._density(mat, "bad")
    with pytest.raises(ZeroDivisionError):
        g._density(mat, "gaussian")
    with pytest.raises(AssertionError, match="input data must be integers"):
        g._density(mat + 0.1, "poisson")


@pytest.mark.parametrize("shift", [0, 0.5, 65, 4097])
def test_poisson_density_ties_preserve_default_scores(shift):
    mat = np.random.default_rng(61).poisson(3, (10, 20)).astype(np.float32) + shift
    frame = pd.DataFrame(mat, columns=[f"g{i}" for i in range(20)])
    net = pd.DataFrame(
        {"source": np.tile(["a", "b", "c", "d"], 5), "target": frame.columns}
    )
    expected, _ = dc_cpu.mt.gsva(frame, net, kcdf="poisson", tmin=0)
    actual, _ = dc.gsva(frame, net, kcdf="poisson", tmin=0)
    pd.testing.assert_frame_equal(actual, expected, rtol=1e-6, atol=1e-6)


def test_many_poisson_counts_preserve_default_scores():
    mat = np.concatenate(
        [
            np.random.default_rng(61).poisson(3, (10, 20)) + 4097,
            np.arange(300).reshape(10, 30),
        ],
        axis=1,
    ).astype(np.float32)
    frame = pd.DataFrame(mat, columns=[f"g{i}" for i in range(mat.shape[1])])
    net = pd.DataFrame(
        {
            "source": np.resize(["a", "b", "c", "d"], mat.shape[1]),
            "target": frame.columns,
        }
    )
    expected, _ = dc_cpu.mt.gsva(frame, net, kcdf="poisson", tmin=0)
    actual, _ = dc.gsva(frame, net, kcdf="poisson", tmin=0)
    pd.testing.assert_frame_equal(actual, expected, rtol=1e-6, atol=1e-6)


def test_poisson_cdf_tiles_preserve_density_order():
    mat = np.random.default_rng(61).poisson(3, (10, 20)).astype(np.float32)
    expected = dc_cpu.mt._gsva._density(mat.astype(np.float64), "poisson")
    actual = g._poisson_density_columns(cp.asarray(mat), max_entries=16).get()
    np.testing.assert_allclose(actual, expected, rtol=1e-14, atol=1e-14)
    expected_ranks = dc_cpu.mt._gsva._rankmat(expected)[0]
    actual_ranks = g._rankmat(cp.asarray(actual))[0].get()
    np.testing.assert_array_equal(actual_ranks, expected_ranks)


def test_large_poisson_counts_match_decoupler():
    frame, net = _data("poisson")
    frame.iloc[0, 0] = 1000
    expected, _ = dc_cpu.mt.gsva(frame, net, tmin=0, kcdf="poisson")
    actual, _ = dc.gsva(frame, net, tmin=0, kcdf="poisson")
    pd.testing.assert_frame_equal(actual, expected, rtol=1e-6, atol=1e-6)


def test_large_feature_sets_match_decoupler():
    rng = np.random.default_rng(12)
    mat = rng.normal(size=(4, 300)).astype(np.float32)
    sets = [
        rng.choice(mat.shape[1], size, replace=False) for size in (1, 127, 129, 250)
    ]
    sizes = np.array(list(map(len, sets)), dtype=np.int32)
    cnct = np.concatenate(sets).astype(np.int32)
    starts = np.r_[0, np.cumsum(sizes)[:-1]].astype(np.int32)
    expected, _ = dc_cpu.mt._gsva._func_gsva(
        mat, cnct, starts, sizes, kcdf=None, maxdiff=False, tau=0.5
    )
    actual, _ = g._func_gsva(
        cp.asarray(mat),
        cnct=cp.asarray(cnct),
        starts=cp.asarray(starts),
        offsets=cp.asarray(sizes),
        kcdf=None,
        maxdiff=False,
        tau=0.5,
    )
    np.testing.assert_allclose(actual, expected, rtol=1e-6, atol=1e-6)
