from __future__ import annotations

from pathlib import Path

import anndata as ad
import cupy as cp
import numpy as np
import pandas as pd
import pytest
from anndata import AnnData
from cupyx.scipy import sparse as cpsparse
from scipy import sparse

import rapids_singlecell as rsc
from rapids_singlecell.preprocessing._dsb import _MARGIN, _fit_rows, _mixture_fits

pytestmark = pytest.mark.filterwarnings("ignore:Proteins with low background")

# Reference values from the dsb R package, see tests/_scripts/dsb_reference.R
DATA = Path(__file__).parent / "_data" / "dsb"
ISOTYPES = [
    "MouseIgG1kappaisotype_PROT",
    "MouseIgG2akappaisotype_PROT",
    "Mouse IgG2bkIsotype_PROT",
    "RatIgG2bkIsotype_PROT",
]


def _read(name: str) -> pd.DataFrame:
    return pd.read_csv(DATA / f"{name}.csv.gz")


@pytest.fixture(scope="module")
def counts() -> tuple[AnnData, AnnData]:
    cells, empty = _read("cells"), _read("empty")

    def to_adata(df, prefix):
        obs = pd.DataFrame(index=[f"{prefix}{i}" for i in range(len(df))])
        return AnnData(df.to_numpy(dtype=np.float64), obs=obs, var=pd.DataFrame(index=df.columns))

    return to_adata(cells, "c"), to_adata(empty, "e")


# dtype -> max abs. difference to R (achieved on this data: float64 2e-13,
# float32 1.6e-6)
PRECISIONS = {np.float64: 1e-10, np.float32: 1e-5}


CONFIGS = {
    "default": ({"isotype_controls": ISOTYPES}, True),
    "meansub_clip": (
        {
            "isotype_controls": ISOTYPES,
            "pseudocount": 5,
            "scale_factor": "mean_subtract",
            "quantile_clip": (0.001, 0.9995),
        },
        True,
    ),
    "negative": ({"isotype_controls": ISOTYPES}, False),
    "no_isotypes": ({"use_isotype_controls": False}, True),
    "negative_no_isotypes": ({"use_isotype_controls": False}, False),
}


@pytest.mark.parametrize("dtype", list(PRECISIONS))
@pytest.mark.parametrize("config", list(CONFIGS))
def test_matches_r(counts, config, dtype):
    cells, empty = counts
    kwargs, with_empty = CONFIGS[config]
    res = rsc.pp.dsb(
        cells,
        empty if with_empty else None,
        dtype=dtype,
        copy=True,
        **kwargs,
    )
    assert res.X.dtype == dtype
    np.testing.assert_allclose(res.X, _read(config).to_numpy(), rtol=0, atol=PRECISIONS[dtype])


@pytest.mark.parametrize("dtype", list(PRECISIONS))
def test_stats(counts, dtype):
    cells, empty = counts
    res = rsc.pp.dsb(
        cells,
        empty,
        isotype_controls=ISOTYPES,
        dtype=dtype,
        copy=True,
    )
    ref = _read("default_technical_stats")
    tol = PRECISIONS[dtype]
    np.testing.assert_allclose(
        res.obs["dsb_cell_background_mean"],
        ref["cellwise_background_mean"],
        rtol=0,
        atol=tol,
    )
    # the sign of a principal component is arbitrary in R
    tech = res.obs["dsb_technical_component"].to_numpy()
    ref_tech = ref["dsb_technical_component"].to_numpy()
    sign = np.sign(tech @ ref_tech)
    np.testing.assert_allclose(tech, sign * ref_tech, rtol=0, atol=tol)
    assert np.corrcoef(tech, res.obs["dsb_cell_background_mean"])[0, 1] > 0

    log_empty = np.log(empty.X + 10)
    # float64 empty statistics, also for float32
    np.testing.assert_allclose(res.var["dsb_ambient_mean"], log_empty.mean(0), rtol=1e-7)
    np.testing.assert_allclose(res.var["dsb_ambient_sd"], log_empty.std(0, ddof=1), rtol=1e-7)


def test_no_denoising(counts):
    cells, empty = counts
    res = rsc.pp.dsb(cells, empty, denoise_counts=False, dtype=np.float64, copy=True)
    log_empty = np.log(empty.X + 10)
    expected = (np.log(cells.X + 10) - log_empty.mean(0)) / log_empty.std(0, ddof=1)
    np.testing.assert_allclose(res.X, expected, rtol=1e-12, atol=1e-12)
    assert "dsb_technical_component" not in res.obs


@pytest.mark.parametrize(
    "convert",
    [
        np.asarray,
        sparse.csr_matrix,
        sparse.csc_matrix,
        cp.asarray,
        lambda x: cpsparse.csr_matrix(cp.asarray(x)),
    ],
    ids=["numpy", "scipy_csr", "scipy_csc", "cupy", "cupy_csr"],
)
def test_input_types(counts, convert):
    cells, empty = counts
    expected = rsc.pp.dsb(cells, empty, isotype_controls=ISOTYPES, copy=True).X
    a = AnnData(convert(cells.X), obs=cells.obs, var=cells.var)
    e = AnnData(convert(empty.X), obs=empty.obs, var=empty.var)
    on_gpu = isinstance(a.X, cp.ndarray) or cpsparse.issparse(a.X)
    rsc.pp.dsb(a, e, isotype_controls=ISOTYPES)
    assert isinstance(a.X, cp.ndarray if on_gpu else np.ndarray)
    np.testing.assert_allclose(cp.asnumpy(a.X), expected, rtol=0, atol=1e-12)


@pytest.mark.parametrize("fmt", ["dense", "csr", "csc"])
def test_backed_empty(counts, tmp_path, monkeypatch, fmt):
    import rapids_singlecell.preprocessing._dsb as dsb_mod

    cells, empty = counts
    expected = rsc.pp.dsb(cells, empty, isotype_controls=ISOTYPES, copy=True).X
    x = empty.X if fmt == "dense" else getattr(sparse, f"{fmt}_matrix")(empty.X)
    AnnData(x, obs=empty.obs, var=empty.var).write_h5ad(tmp_path / "empty.h5ad")
    backed = ad.read_h5ad(tmp_path / "empty.h5ad", backed="r")
    monkeypatch.setattr(dsb_mod, "_CHUNK", 64)  # several chunks
    res = rsc.pp.dsb(cells, backed, isotype_controls=ISOTYPES, copy=True)
    np.testing.assert_allclose(res.X, expected, rtol=0, atol=1e-10)


def test_layer_and_key_added(counts):
    cells, empty = counts
    expected = rsc.pp.dsb(cells, empty, isotype_controls=ISOTYPES, copy=True).X
    a = cells.copy()
    a.layers["counts"] = a.X.copy()
    a.X = np.zeros_like(a.X)
    e = empty.copy()
    e.layers["counts"] = e.X.copy()
    e.X = np.zeros_like(e.X)
    rsc.pp.dsb(a, e, isotype_controls=ISOTYPES, layer="counts", key_added="dsb")
    np.testing.assert_allclose(a.layers["dsb"], expected, rtol=0, atol=1e-12)
    np.testing.assert_array_equal(a.layers["counts"], cells.X)
    assert not a.X.any()


def test_copy_leaves_input(counts):
    cells, empty = counts
    before = cells.X.copy()
    res = rsc.pp.dsb(cells, empty, isotype_controls=ISOTYPES, copy=True)
    assert res is not cells
    np.testing.assert_array_equal(cells.X, before)
    assert "dsb_ambient_mean" not in cells.var


def test_empty_protein_order(counts):
    cells, empty = counts
    expected = rsc.pp.dsb(cells, empty, isotype_controls=ISOTYPES, copy=True).X
    shuffled = empty[:, empty.var_names[::-1]].copy()
    with pytest.warns(UserWarning, match="Reordered"):
        res = rsc.pp.dsb(cells, shuffled, isotype_controls=ISOTYPES, copy=True)
    np.testing.assert_allclose(res.X, expected, rtol=0, atol=1e-12)
    with pytest.raises(ValueError, match="different proteins"):
        rsc.pp.dsb(cells, empty[:, 1:].copy(), isotype_controls=ISOTYPES, copy=True)


def test_invalid_arguments(counts):
    cells, empty = counts
    with pytest.raises(ValueError, match="isotype controls not found"):
        rsc.pp.dsb(cells, empty, isotype_controls=["nope"], copy=True)
    with pytest.raises(ValueError, match="scale_factor"):
        rsc.pp.dsb(cells, empty, scale_factor="scale", copy=True)
    with pytest.raises(ValueError, match="Set `isotype_controls`"):
        rsc.pp.dsb(cells, empty, copy=True)
    with pytest.raises(ValueError, match="not both"):
        rsc.pp.dsb(cells, empty, isotype_controls=ISOTYPES, use_isotype_controls=False)
    with pytest.raises(ValueError, match="only applies with `adata_empty`"):
        rsc.pp.dsb(cells, scale_factor="mean_subtract", isotype_controls=ISOTYPES)
    with pytest.raises(ValueError, match="fast_km"):
        rsc.pp.dsb(cells, empty, fast_km=True, denoise_counts=False)
    with pytest.raises(ValueError, match="pseudocount"):
        rsc.pp.dsb(cells, empty, isotype_controls=ISOTYPES, pseudocount=0)
    with pytest.raises(ValueError, match="quantile_clip"):
        rsc.pp.dsb(cells, empty, isotype_controls=ISOTYPES, quantile_clip=(0.5, 0.1))
    with pytest.raises(ValueError, match="at least 2 cells"):
        rsc.pp.dsb(cells[:1].copy(), empty, isotype_controls=ISOTYPES)
    with pytest.raises(ValueError, match="at least 2 empty"):
        rsc.pp.dsb(cells, empty[:1].copy(), isotype_controls=ISOTYPES)
    with pytest.raises(ValueError, match="no layer"):
        rsc.pp.dsb(cells, empty, isotype_controls=ISOTYPES, layer="raw")


def test_isotype_inputs(counts):
    cells, empty = counts
    expected = rsc.pp.dsb(cells, empty, isotype_controls=ISOTYPES[:1], copy=True).X
    res = rsc.pp.dsb(cells, empty, isotype_controls=ISOTYPES[0], copy=True)
    np.testing.assert_array_equal(res.X, expected)


@pytest.mark.parametrize("case", ["negative", "nan", "duplicates", "x_none"])
def test_invalid_counts(counts, case):
    cells, empty = counts
    a = cells.copy()
    if case == "duplicates":
        a.var_names = [a.var_names[0]] * a.n_vars
        match = "not unique"
    elif case == "x_none":
        a = AnnData(obs=cells.obs, var=cells.var, layers={"counts": cells.X})
        match = r"`adata.X` is None"
    else:
        a.X[0, 0] = -1 if case == "negative" else np.nan
        match = "raw counts"
    with pytest.raises(ValueError, match=match):
        rsc.pp.dsb(a, empty, isotype_controls=ISOTYPES)


def test_constant_inputs(counts):
    cells, empty = counts
    e = empty.copy()
    e.X[:, 0] = 3
    with pytest.raises(ValueError, match="same count in every empty"):
        rsc.pp.dsb(cells, e, isotype_controls=ISOTYPES)
    c = cells.copy()
    c.X[:] = c.X[0]
    with pytest.raises(ValueError, match="technical component, constant"):
        rsc.pp.dsb(c, empty, isotype_controls=ISOTYPES)


def test_stats_are_reset(counts):
    cells, empty = counts
    a = cells.copy()
    rsc.pp.dsb(a, empty, isotype_controls=ISOTYPES, key_added="x")
    rsc.pp.dsb(a, None, isotype_controls=ISOTYPES, key_added="x")
    assert "dsb_ambient_sd" not in a.var


def test_fast_km_is_optimal_two_means(counts):
    cells, empty = counts
    res = rsc.pp.dsb(cells, empty, isotype_controls=ISOTYPES, fast_km=True, dtype=np.float64, copy=True)
    step1 = rsc.pp.dsb(cells, empty, denoise_counts=False, dtype=np.float64, copy=True).X
    for x, low in zip(step1, res.obs["dsb_cell_background_mean"], strict=True):
        xs = np.sort(x)
        sse = [((xs[:s] - xs[:s].mean()) ** 2).sum() + ((xs[s:] - xs[s:].mean()) ** 2).sum() for s in range(1, len(xs))]
        best = int(np.argmin(sse)) + 1
        assert low == pytest.approx(xs[:best].mean(), abs=1e-12)


# --- mixture fits ----------------------------------------------------------


def _ref_threshold(x):
    """Type 7 quantile at the simplest j / k with min < q < max."""
    xs, n = np.sort(x), len(x)
    for k in range(2, 2 * n):
        for j in range(1, k):
            lo, rem = divmod((n - 1) * j, k)
            q = xs[lo] + rem / k * (xs[min(lo + 1, n - 1)] - xs[lo])
            if xs[0] < q < xs[-1]:
                return q
    return np.nan


def _ref_mclust2(x, sub=None, *, eps=np.finfo(np.float64).eps, iterations=False):
    """NumPy reference of the fits (see ``_mixture_fits``); `sub` is the
    random subset that starts long vectors. With `iterations`, also returns
    the EM iterations of models E and V and the index of the selected one."""

    def mstep(v, z, equal):
        nk = z.sum(0)
        mu = (z * v[:, None]).sum(0) / nk
        ss = (z * (v[:, None] - mu) ** 2).sum(0)
        return mu, np.full(2, ss.sum() / len(v)) if equal else ss / nk, nk / len(v)

    def estep(mu, s2, pro):
        lg = np.log(pro) - 0.5 * np.log(2 * np.pi * s2) - 0.5 * (x[:, None] - mu) ** 2 / s2
        ll = np.logaddexp(lg[:, 0], lg[:, 1])
        return np.exp(lg - ll[:, None]), ll.sum()

    def valid(p):
        return np.all(np.isfinite(p[0])) and np.all(p[1] > eps)

    start = x if sub is None else sub
    c = start < _ref_threshold(start)
    fits, its = [], []
    for equal, npar in ((True, 4), (False, 5)):
        p = mstep(start, np.stack([c, ~c], 1).astype(float), equal)
        skip, prev, ok, it = sub is not None, None, valid(p), 0
        while ok:
            z, ll = estep(*p)
            pn = mstep(x, z, equal)
            if skip:
                skip = False
            else:
                it += 1
                if prev is not None and abs(ll - prev) < 1e-5 * (1 + abs(ll)):
                    break
                prev = ll
            ok = valid(pn)
            p = pn if ok else p
        its.append(it if ok else -1)
        if not ok:
            continue
        if sub is not None or ((p[2] - pn[2]) ** 2).sum() > np.sqrt(eps):
            p = pn
        fits.append((2 * ll - npar * np.log(len(x)), p, npar - 4))
    best = max(fits, key=lambda f: f[0]) if fits else None
    if iterations:
        return best and best[1], its, best and best[2]
    return best and best[1]


def _random_rows(rng, n_rows):
    rows = []
    for i in range(n_rows):
        n = int(rng.integers(20, 150))
        kind = i % 3
        if kind == 0:
            x = np.r_[rng.normal(0, 1, n // 2), rng.normal(rng.uniform(1, 6), 2, n)]
        elif kind == 1:  # log counts with many ties
            x = np.log(rng.poisson(rng.uniform(0.05, 3), 2 * n) + 1.0)
        else:
            x = np.log(rng.poisson(rng.uniform(0.05, 30), 2 * n) + 10.0)
        rows.append(x[: 2 * n])
    return rows


def _assert_matches_ref(got, ref):
    np.testing.assert_allclose(got[:2], ref[0], rtol=1e-9, atol=1e-10)
    np.testing.assert_allclose(got[2:4], ref[1], rtol=1e-9, atol=1e-10)
    np.testing.assert_allclose(got[4:], ref[2], rtol=0, atol=1e-10)


@pytest.mark.parametrize("compress", [False, True])
def test_mixture_fits_match_reference(compress):
    rng = np.random.default_rng(0)
    for x in _random_rows(rng, 60):
        got = cp.asnumpy(_mixture_fits(cp.asarray(x[None, :]), rng, compress=compress).params[0])
        ref = _ref_mclust2(x)
        if ref is None:
            assert np.isnan(got).all()
        else:
            _assert_matches_ref(got, ref)


def _subset_rows(n=3000):
    gen = np.random.default_rng(1)
    return np.stack(
        [
            np.log(gen.poisson(gen.uniform(0.1, 5), n) + 1.0),
            np.r_[gen.normal(0, 1, 2000), gen.normal(4, 1.5, n - 2000)],
        ]
    )


@pytest.mark.parametrize("compress", [False, True])
def test_mixture_fits_subset_path(compress):
    # vectors longer than 2000 start from a random subset of 2000 values
    X = _subset_rows()
    fit = _mixture_fits(cp.asarray(X), np.random.default_rng(7), compress=compress)
    again = _mixture_fits(cp.asarray(X), np.random.default_rng(7), compress=compress)
    cp.testing.assert_array_equal(fit.params, again.params)
    assert not cp.asnumpy(fit.capped).any()
    sub_rng = np.random.default_rng(7)
    for x, got in zip(X, cp.asnumpy(fit.params), strict=True):
        _assert_matches_ref(got, _ref_mclust2(x, x[sub_rng.choice(x.size, 2000, replace=False)]))


@pytest.mark.parametrize("compress", [False, True])
def test_mixture_fits_subset_matches_r(compress):
    # R's own subsets, passed to Mclust as `initialization = list(subset = idx)`
    X = _read("subset_x").to_numpy().T
    sub = _read("subset_idx").to_numpy().T - 1
    ref = _read("subset_params").to_numpy()
    params, flags = _fit_rows(cp.asarray(X), sub, max_iter=100_000, compress=compress)
    np.testing.assert_allclose(cp.asnumpy(params), ref, rtol=0, atol=1e-10)
    assert not cp.asnumpy(flags).any()


def test_mixture_fits_no_rows():
    for n in (100, 3000):
        fit = _mixture_fits(cp.empty((0, n)), np.random.default_rng(0), compress=True)
        assert fit.params.shape == (0, 6)


@pytest.mark.parametrize("dtype", [np.float32, np.float64])
def test_mixture_fits_terminate_on_ties(dtype):
    # log(counts + 10) with mostly zero counts: heavy ties, near-singular fits
    rng = np.random.default_rng(3)
    counts = rng.poisson(0.05, (500, 87))
    counts[:, :3] = rng.poisson(50, (500, 3))
    X = cp.asarray(np.log(counts + 10.0), dtype=dtype)
    # per-row fits, and whole-column fits of the same data (subset path)
    for M, compress in ((X, False), (cp.tile(X, (5, 1)).T, True)):
        fit = _mixture_fits(M, rng, compress=compress)
        params = cp.asnumpy(fit.params)
        ok = ~np.isnan(params[:, 0])
        assert ok.any()
        assert np.isfinite(params[ok]).all()
        assert not cp.asnumpy(fit.capped).any()


def test_mixture_constant_rows():
    X = cp.asarray(np.r_[np.ones((2, 30)), np.arange(60.0).reshape(2, 30)])
    fit = _mixture_fits(X, np.random.default_rng(0))
    np.testing.assert_array_equal(np.isnan(cp.asnumpy(fit.params[:, 0])), [True, True, False, False])


@pytest.mark.parametrize("dtype", [np.float64, np.float32])
@pytest.mark.parametrize("n_cells", [20, 21, 33, 100])
@pytest.mark.parametrize("row", [0, 7])
def test_constant_technical_component(counts, n_cells, row, dtype):
    # limma::removeBatchEffect returns the data unchanged for a constant
    # covariate; identical cells give a constant background mean, whose mean
    # may differ from it by an ulp
    cells, empty = counts
    same = AnnData(np.repeat(cells.X[row : row + 1], n_cells, axis=0), var=cells.var)
    kw = {"dtype": dtype, "copy": True}
    expected = rsc.pp.dsb(same, empty, denoise_counts=False, **kw).X
    with pytest.warns(UserWarning, match="technical component is constant"):
        res = rsc.pp.dsb(same, empty, use_isotype_controls=False, **kw)
    np.testing.assert_array_equal(res.X, expected)


@pytest.mark.parametrize("n", [100, 3000], ids=["full", "subset"])
def test_iteration_cap_is_reported(n):
    # rows 0/1 need more than 2 EM iterations in both models / only in one
    X = _subset_rows()[:, :n]
    with pytest.warns(UserWarning, match="iteration limit"):
        fit = _mixture_fits(cp.asarray(X), np.random.default_rng(0), max_iter=2)
    assert cp.asnumpy(fit.capped).all()
    assert not cp.asnumpy(_mixture_fits(cp.asarray(X), np.random.default_rng(0)).capped).any()


def test_iteration_cap_of_either_model():
    # a cap hit only by the model that loses the BIC comparison is reported
    rng = np.random.default_rng(0)
    checked = 0
    for x in _random_rows(rng, 60):
        _, its, sel = _ref_mclust2(x, iterations=True)
        if sel is None or its[1 - sel] <= its[sel] + 1:
            continue
        _, flags = _fit_rows(cp.asarray(x[None, :]), None, max_iter=its[sel] + 1, compress=False)
        assert int(flags[0]) & 1
        checked += 1
    assert checked > 5


def test_view_is_made_actual(counts):
    cells, empty = counts
    view = cells[:50]
    with pytest.warns(UserWarning, match="Received a view"):
        rsc.pp.dsb(view, empty, isotype_controls=ISOTYPES)
    assert not view.is_view
    assert "dsb_technical_component" in view.obs


def test_binding_rejects_mistyped_outputs():
    from rapids_singlecell._cuda import _dsb_cuda as _dsb

    X = cp.asarray(np.random.default_rng(0).normal(size=(4, 50)))
    none = cp.empty(0)
    for params, capped in (
        (cp.empty((4, 6), dtype=cp.float32), cp.zeros(4, dtype=cp.uint8)),  # dtype
        (cp.empty((6, 4)).T, cp.zeros(4, dtype=cp.uint8)),  # not C-contiguous
    ):
        with pytest.raises(TypeError):
            _dsb.mix2(X, none, none, none, params, capped, subset=False, max_iter=10)
    with pytest.raises(ValueError, match="params must be"):
        _dsb.mix2(X, none, none, none, cp.empty((3, 6)), cp.zeros(4, dtype=cp.uint8), subset=False, max_iter=10)


@pytest.mark.parametrize("n", [100, 3000])
def test_compression_is_exact(n):
    # EM on distinct values weighted by multiplicity equals EM on all values;
    # rows of mostly distinct values (second case) skip the compression
    gen = np.random.default_rng(2)
    for X in (np.log(gen.poisson(gen.uniform(0.1, 5, (3, 1)), (3, n)) + 1.0), gen.normal(size=(3, n))):
        sub = np.stack([gen.choice(n, 2000, replace=False) for _ in range(3)]) if n > 2000 else None
        fits = [_fit_rows(cp.asarray(X), sub, max_iter=100_000, compress=c)[0] for c in (False, True)]
        cp.testing.assert_allclose(*fits, rtol=1e-12, atol=1e-12)


def test_float32_fits_match_float64(counts):
    # ADT-like cells: plain float32 EM moved Mclust's plateau stopping point in
    # about 1 of 200 such cells (background off by up to ~0.6)
    cells, empty = counts
    gen = np.random.default_rng(0)
    x = gen.poisson(cells.X[gen.integers(0, cells.n_obs, 3000)])
    log_empty = np.log(empty.X + 10)
    norm = (np.log(x + 10.0) - log_empty.mean(0)) / log_empty.std(0, ddof=1)
    p64 = cp.asnumpy(_mixture_fits(cp.asarray(norm), gen).params)
    p32 = cp.asnumpy(
        _mixture_fits(cp.asarray(norm, dtype=np.float32), gen, exact=lambda r: cp.asarray(norm[r])).params
    ).astype(np.float64)
    np.testing.assert_array_equal(p32[:, 2] == p32[:, 3], p64[:, 2] == p64[:, 3])  # E / V
    np.testing.assert_allclose(p32[:, 0], p64[:, 0], rtol=0, atol=1e-4)


def test_float32_ties_are_refit(counts, monkeypatch):
    # float64 values closer than float32 rounding merge into ties: here the
    # value just below each cell's median, which moves to the upper class of
    # the start split; without the refit, backgrounds are off by up to 1e-3
    # here (up to 1.3 on 10,000 cells of the benchmark data)
    cells, empty = counts
    log_empty = np.log(empty.X + 10)
    norm = (np.log(cells.X + 10) - log_empty.mean(0)) / log_empty.std(0, ddof=1)
    rows, order = np.arange(len(norm)), np.argsort(norm, axis=1)
    med = (norm.shape[1] - 1) // 2
    q = norm[rows, order[:, med]].astype(np.float32).astype(np.float64)
    norm[rows, order[:, med]] = q
    norm[rows, order[:, med - 1]] = np.nextafter(q, -np.inf)  # rounds to q
    x32 = cp.asarray(norm, dtype=np.float32)
    assert (x32[rows, order[:, med - 1]] == x32[rows, order[:, med]]).all()
    p64 = cp.asnumpy(_mixture_fits(cp.asarray(norm), None).params)
    _, flags = _fit_rows(x32, None, max_iter=100_000, compress=False)
    assert (cp.asnumpy(flags) & 2).all()
    p32 = cp.asnumpy(_mixture_fits(x32, None, exact=lambda r: cp.asarray(norm[r])).params)
    np.testing.assert_allclose(p32[:, 0], p64[:, 0], rtol=0, atol=1e-5)
    monkeypatch.setitem(_MARGIN, np.dtype(np.float32), 0.0)  # no refits
    p_no = cp.asnumpy(_mixture_fits(x32, None).params)
    assert np.abs(p_no[:, 0] - p64[:, 0]).max() > 1e-4
