from __future__ import annotations

from contextlib import nullcontext
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
import rapids_singlecell.preprocessing._dsb as dsb_mod
from rapids_singlecell.preprocessing._dsb import _MARGIN, _fit_rows, _mixture_fits

pytestmark = pytest.mark.filterwarnings("ignore:Proteins with low background")

# reference values from the dsb R package, see tests/_scripts/dsb_reference.R
DATA = Path(__file__).parent / "_data" / "dsb"
ISOTYPES = [
    "MouseIgG1kappaisotype_PROT",
    "MouseIgG2akappaisotype_PROT",
    "Mouse IgG2bkIsotype_PROT",
    "RatIgG2bkIsotype_PROT",
]


def _dsb(adata, *args, **kwargs):
    adata = adata.copy()
    rsc.pp.dsb(adata, *args, **kwargs)
    return adata


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


def _read(name: str) -> pd.DataFrame:
    return pd.read_csv(DATA / f"{name}.csv.gz")


@pytest.fixture(scope="module")
def counts() -> tuple[AnnData, AnnData]:
    def to_adata(df, prefix):
        obs = pd.DataFrame(index=[f"{prefix}{i}" for i in range(len(df))])
        return AnnData(df.to_numpy(dtype=np.float64), obs=obs, var=pd.DataFrame(index=df.columns))

    return to_adata(_read("cells"), "c"), to_adata(_read("empty"), "e")


def _step1(cells, empty):
    log_empty = np.log(empty.X + 10)
    return (np.log(cells.X + 10) - log_empty.mean(0)) / log_empty.std(0, ddof=1)


@pytest.mark.parametrize("dtype", list(PRECISIONS))
@pytest.mark.parametrize("config", list(CONFIGS))
def test_matches_r(counts, config, dtype):
    cells, empty = counts
    kwargs, with_empty = CONFIGS[config]
    res = _dsb(cells, empty if with_empty else None, dtype=dtype, **kwargs)
    assert res.X.dtype == dtype
    np.testing.assert_allclose(res.X, _read(config).to_numpy(), rtol=0, atol=PRECISIONS[dtype])
    if config == "default":
        ref, tol = _read("default_technical_stats"), PRECISIONS[dtype]
        bg, tech = res.obs["dsb_cell_background_mean"], res.obs["dsb_technical_component"].to_numpy()
        np.testing.assert_allclose(bg, ref["cellwise_background_mean"], rtol=0, atol=tol)
        ref_tech = ref["dsb_technical_component"].to_numpy()
        np.testing.assert_allclose(tech, np.sign(tech @ ref_tech) * ref_tech, rtol=0, atol=tol)
        log_empty = np.log(empty.X + 10)
        np.testing.assert_allclose(res.var["dsb_ambient_mean"], log_empty.mean(0), rtol=1e-7)
        np.testing.assert_allclose(res.var["dsb_ambient_sd"], log_empty.std(0, ddof=1), rtol=1e-7)


@pytest.mark.parametrize(
    "convert",
    [np.asarray, sparse.csr_matrix, sparse.csc_matrix, cp.asarray, lambda x: cpsparse.csr_matrix(cp.asarray(x))],
    ids=["numpy", "scipy_csr", "scipy_csc", "cupy", "cupy_csr"],
)
def test_input_types(counts, convert):
    cells, empty = counts
    expected = _dsb(cells, empty, isotype_controls=ISOTYPES).X
    a = AnnData(convert(cells.X), obs=cells.obs, var=cells.var)
    e = AnnData(convert(empty.X), obs=empty.obs, var=empty.var)
    on_gpu = isinstance(a.X, cp.ndarray) or cpsparse.issparse(a.X)
    rsc.pp.dsb(a, e, isotype_controls=ISOTYPES)
    assert isinstance(a.X, cp.ndarray if on_gpu else np.ndarray)
    np.testing.assert_allclose(cp.asnumpy(a.X), expected, rtol=0, atol=1e-12)


@pytest.mark.parametrize("reordered", [False, True])
@pytest.mark.parametrize("fmt", ["dense", "csr", "csc"])
def test_backed_empty(counts, tmp_path, monkeypatch, fmt, reordered):
    cells, empty = counts
    expected = _dsb(cells, empty, isotype_controls=ISOTYPES).X
    if reordered:
        empty = empty[:, empty.var_names[::-1]].copy()
    x = empty.X if fmt == "dense" else getattr(sparse, f"{fmt}_matrix")(empty.X)
    AnnData(x, obs=empty.obs, var=empty.var).write_h5ad(tmp_path / "empty.h5ad")
    backed = ad.read_h5ad(tmp_path / "empty.h5ad", backed="r")
    monkeypatch.setattr(dsb_mod, "_CHUNK", 64)
    # the backed matrix itself reaches the chunked statistics, not a loaded copy
    inputs, empty_stats = [], dsb_mod._empty_stats
    monkeypatch.setattr(dsb_mod, "_empty_stats", lambda x, *a: inputs.append(x) or empty_stats(x, *a))
    with pytest.warns(UserWarning, match="Reordered") if reordered else nullcontext():
        res = _dsb(cells, backed, isotype_controls=ISOTYPES)
    assert not isinstance(inputs[0], np.ndarray | sparse.spmatrix | sparse.sparray)
    np.testing.assert_array_equal(res.X, expected)


def test_use_and_out(counts):
    cells, empty = counts
    expected = _dsb(cells, empty, isotype_controls=ISOTYPES).X
    a, e = AnnData(np.zeros_like(cells.X), var=cells.var), AnnData(np.zeros_like(empty.X), var=empty.var)
    a.layers["counts"], e.layers["counts"] = cells.X.copy(), empty.X.copy()
    rsc.pp.dsb(a, e, isotype_controls=ISOTYPES, use="layers.counts", out="layers.dsb")
    np.testing.assert_allclose(a.layers["dsb"], expected, rtol=0, atol=1e-12)
    assert not a.X.any()
    res = rsc.pp.dsb(a, e, isotype_controls=ISOTYPES, use="layers.counts", out=True)
    np.testing.assert_allclose(res, expected, rtol=0, atol=1e-12)
    with pytest.warns(FutureWarning, match="key_added"):
        rsc.pp.dsb(a, e, isotype_controls=ISOTYPES, use="layers.counts", key_added="old")
    np.testing.assert_allclose(a.layers["old"], expected, rtol=0, atol=1e-12)


def test_view_is_made_actual(counts):
    cells, empty = counts
    view = cells[:50]
    with pytest.warns(UserWarning, match="Received a view"):
        rsc.pp.dsb(view, empty, isotype_controls=ISOTYPES)
    assert "dsb_technical_component" in view.obs


@pytest.mark.filterwarnings("ignore:Variable names are not unique:UserWarning")
def test_invalid_inputs(counts):
    c, e = counts
    iso = {"isotype_controls": ISOTYPES}
    neg, nan, dup, same_cells, same_empty = c.copy(), c.copy(), c.copy(), c.copy(), e.copy()
    neg.X[0, 0] = -1
    nan.X[0, 0] = np.nan
    dup.var_names = [c.var_names[0]] * c.n_vars
    same_cells.X[:] = c.X[0]
    same_empty.X[:, 0] = 3
    cases = [
        (c, e, {"isotype_controls": ["nope"]}, "isotype controls not found"),
        (c, e, {"scale_factor": "scale"}, "scale_factor"),
        (c, e, {}, "Set `isotype_controls`"),
        (c, e, {**iso, "use_isotype_controls": False}, "not both"),
        (c, None, {**iso, "scale_factor": "mean_subtract"}, "only applies"),
        (c, e, {"fast_km": True, "denoise_counts": False}, "fast_km"),
        (c, e, {**iso, "pseudocount": 0}, "pseudocount"),
        (c, e, {**iso, "quantile_clip": (0.5, 0.1)}, "quantile_clip"),
        (c, e, {**iso, "dtype": np.int32}, "dtype"),
        (c, e, {**iso, "use": "layers.raw"}, "no layer"),
        (c[:1].copy(), e, iso, "at least 2 cells"),
        (c, e[:1].copy(), iso, "at least 2 empty"),
        (c, e[:, 1:].copy(), iso, "different proteins"),
        (neg, e, iso, "raw counts"),
        (nan, e, iso, "raw counts"),
        (dup, e, {}, "not unique"),
        (c, same_empty, iso, "same count in every empty"),
        (same_cells, e, iso, "constant across cells"),
    ]
    for cells, empty, kwargs, match in cases:
        with pytest.raises(ValueError, match=match):
            _dsb(cells, empty, **kwargs)


def test_fast_km_is_optimal_two_means(counts):
    cells, empty = counts
    res = _dsb(cells, empty, isotype_controls=ISOTYPES, fast_km=True, dtype=np.float64)
    for x, low in zip(_step1(cells, empty), res.obs["dsb_cell_background_mean"], strict=True):
        xs = np.sort(x)
        sse = [((xs[:s] - xs[:s].mean()) ** 2).sum() + ((xs[s:] - xs[s:].mean()) ** 2).sum() for s in range(1, len(xs))]
        assert low == pytest.approx(xs[: int(np.argmin(sse)) + 1].mean(), abs=1e-12)


@pytest.mark.parametrize("dtype", list(PRECISIONS))
@pytest.mark.parametrize("n_cells", [20, 21, 33, 100])
def test_constant_technical_component(counts, n_cells, dtype):
    # removeBatchEffect leaves the data unchanged for a constant covariate,
    # here one whose mean differs from its values by an ulp
    cells, empty = counts
    same = AnnData(np.repeat(cells.X[:1], n_cells, axis=0), var=cells.var)
    expected = _dsb(same, empty, denoise_counts=False, dtype=dtype).X
    with pytest.warns(UserWarning, match="technical component is constant"):
        res = _dsb(same, empty, use_isotype_controls=False, dtype=dtype)
    np.testing.assert_array_equal(res.X, expected)


def _ref_threshold(x):
    xs, n = np.sort(x), len(x)
    for k in range(2, 2 * n):
        for j in range(1, k):
            lo, rem = divmod((n - 1) * j, k)
            q = xs[lo] + rem / k * (xs[min(lo + 1, n - 1)] - xs[lo])
            if xs[0] < q < xs[-1]:
                return q
    return np.nan


def _ref_mclust2(x, eps=np.finfo(np.float64).eps):
    """NumPy reference of `_mixture_fits`: params, EM iterations of E and V, selected model."""

    def mstep(z, equal):
        nk = z.sum(0)
        with np.errstate(invalid="ignore"):
            mu = (z * x[:, None]).sum(0) / nk
        ss = (z * (x[:, None] - mu) ** 2).sum(0)
        return mu, np.full(2, ss.sum() / len(x)) if equal else ss / nk, nk / len(x)

    def estep(mu, s2, pro):
        lg = np.log(pro) - 0.5 * np.log(2 * np.pi * s2) - 0.5 * (x[:, None] - mu) ** 2 / s2
        ll = np.logaddexp(lg[:, 0], lg[:, 1])
        return np.exp(lg - ll[:, None]), ll.sum()

    def valid(p):
        return np.all(np.isfinite(p[0])) and np.all(p[1] > eps)

    c = x < _ref_threshold(x)
    fits, its = [], []
    for equal, npar in ((True, 4), (False, 5)):
        p = mstep(np.stack([c, ~c], 1).astype(float), equal)
        prev, ok, it = None, valid(p), 0
        while ok:
            z, ll = estep(*p)
            pn = mstep(z, equal)
            it += 1
            if prev is not None and abs(ll - prev) < 1e-5 * (1 + abs(ll)):
                break
            prev, ok = ll, valid(pn)
            p = pn if ok else p
        its.append(it if ok else -1)
        if ok:
            if ((p[2] - pn[2]) ** 2).sum() > np.sqrt(eps):
                p = pn
            fits.append((2 * ll - npar * np.log(len(x)), p, npar - 4))
    best = max(fits, key=lambda f: f[0]) if fits else (None, None, None)
    return best[1], its, best[2]


def _random_rows(rng, n_rows):
    rows = []
    for i in range(n_rows):
        n = int(rng.integers(20, 150))
        if i % 3 == 0:
            x = np.r_[rng.normal(0, 1, n), rng.normal(rng.uniform(1, 6), 2, n)]
        else:  # log counts with many ties
            x = np.log(rng.poisson(rng.uniform(0.05, 3 if i % 3 == 1 else 30), 2 * n) + (1.0 if i % 3 == 1 else 10.0))
        rows.append(x)
    return [*rows, np.ones(30)]


@pytest.mark.parametrize("compress", [False, True])
def test_mixture_fits_match_reference(compress):
    rng = np.random.default_rng(0)
    for x in _random_rows(rng, 60):
        got = cp.asnumpy(_mixture_fits(cp.asarray(x[None, :]), rng, compress=compress).params[0])
        ref = _ref_mclust2(x)[0]
        if ref is None:
            assert np.isnan(got).all()
            continue
        np.testing.assert_allclose(got, np.concatenate(ref), rtol=1e-9, atol=1e-10)


@pytest.mark.parametrize("compress", [False, True])
def test_mixture_fits_subset_matches_r(compress):
    # vectors longer than 2000 start from a subset, here R's own
    X = _read("subset_x").to_numpy().T
    sub = _read("subset_idx").to_numpy().T - 1
    params, flags = _fit_rows(cp.asarray(X), sub, max_iter=100_000, compress=compress)
    np.testing.assert_allclose(cp.asnumpy(params), _read("subset_params").to_numpy(), rtol=0, atol=1e-10)
    assert not cp.asnumpy(flags).any()


@pytest.mark.parametrize("dtype", list(PRECISIONS))
def test_mixture_fits_terminate_on_ties(dtype):
    # log(counts + 10) with mostly zero counts: heavy ties, near-singular fits
    rng = np.random.default_rng(3)
    counts = rng.poisson(0.05, (500, 87))
    counts[:, :3] = rng.poisson(50, (500, 3))
    X = cp.asarray(np.log(counts + 10.0), dtype=dtype)
    for M, compress in ((X, False), (cp.tile(X, (5, 1)).T, True)):
        fit = _mixture_fits(M, rng, compress=compress)
        params = cp.asnumpy(fit.params)
        ok = ~np.isnan(params[:, 0])
        assert ok.any()
        assert np.isfinite(params[ok]).all()
        assert not cp.asnumpy(fit.capped).any()


@pytest.mark.parametrize("n", [100, 3000])
def test_iteration_cap_is_reported(n):
    gen = np.random.default_rng(1)
    X = np.stack(
        [np.log(gen.poisson(gen.uniform(0.1, 5), 3000) + 1.0), np.r_[gen.normal(0, 1, 2000), gen.normal(4, 1.5, 1000)]]
    )
    with pytest.warns(UserWarning, match="iteration limit"):
        fit = _mixture_fits(cp.asarray(X[:, :n]), np.random.default_rng(0), max_iter=2)
    assert cp.asnumpy(fit.capped).all()
    # also when only the model that loses the BIC comparison hits the cap
    checked = 0
    for x in _random_rows(np.random.default_rng(0), 60):
        _, its, sel = _ref_mclust2(x)
        if sel is not None and its[1 - sel] > its[sel] + 1:
            _, flags = _fit_rows(cp.asarray(x[None, :]), None, max_iter=its[sel] + 1, compress=False)
            assert int(flags[0]) & 1
            checked += 1
    assert checked > 5


def test_float32_fits_match_float64(counts):
    cells, empty = counts
    gen = np.random.default_rng(0)
    x = AnnData(gen.poisson(cells.X[gen.integers(0, cells.n_obs, 3000)]).astype(np.float64))
    norm = _step1(x, empty)
    p64 = cp.asnumpy(_mixture_fits(cp.asarray(norm), gen).params)
    p32 = cp.asnumpy(_mixture_fits(cp.asarray(norm, dtype=np.float32), gen, exact=lambda r: cp.asarray(norm[r])).params)
    np.testing.assert_array_equal(p32[:, 2] == p32[:, 3], p64[:, 2] == p64[:, 3])
    np.testing.assert_allclose(p32[:, 0], p64[:, 0], rtol=0, atol=1e-4)


def test_float32_ties_are_refit(counts, monkeypatch):
    # make the value below each cell's median round onto it in float32: the
    # start split changes unless the fit is redone in float64
    norm = _step1(*counts)
    rows, order = np.arange(len(norm)), np.argsort(norm, axis=1)
    med = (norm.shape[1] - 1) // 2
    q = norm[rows, order[:, med]].astype(np.float32).astype(np.float64)
    norm[rows, order[:, med]] = q
    norm[rows, order[:, med - 1]] = np.nextafter(q, -np.inf)
    x32 = cp.asarray(norm, dtype=np.float32)
    p64 = cp.asnumpy(_mixture_fits(cp.asarray(norm), None).params)
    assert (cp.asnumpy(_fit_rows(x32, None, max_iter=100_000, compress=False)[1]) & 2).all()
    p32 = cp.asnumpy(_mixture_fits(x32, None, exact=lambda r: cp.asarray(norm[r])).params)
    np.testing.assert_allclose(p32[:, 0], p64[:, 0], rtol=0, atol=1e-5)
    monkeypatch.setitem(_MARGIN, np.dtype(np.float32), 0.0)
    assert np.abs(cp.asnumpy(_mixture_fits(x32, None).params)[:, 0] - p64[:, 0]).max() > 1e-4
