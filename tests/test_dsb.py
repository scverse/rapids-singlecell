from __future__ import annotations

from pathlib import Path

import cupy as cp
import numpy as np
import pandas as pd
import pytest
from anndata import AnnData
from cupyx.scipy import sparse as cpsparse
from scipy import sparse

import rapids_singlecell as rsc
from rapids_singlecell.preprocessing._dsb import _mixture_fits

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
        return AnnData(
            df.to_numpy(dtype=np.float64), obs=obs, var=pd.DataFrame(index=df.columns)
        )

    return to_adata(cells, "c"), to_adata(empty, "e")


def _tol(dtype):
    return 1e-10 if dtype == np.float64 else 5e-2


CONFIGS = {
    "default": ({"isotype_controls": ISOTYPES}, True),
    "meansub_clip": (
        {
            "isotype_controls": ISOTYPES,
            "pseudocount": 5,
            "scale_factor": "mean_subtract",
            "quantile_clipping": True,
        },
        True,
    ),
    "negative": ({"isotype_controls": ISOTYPES}, False),
}


@pytest.mark.parametrize("dtype", [np.float64, np.float32])
@pytest.mark.parametrize("config", list(CONFIGS))
def test_matches_r(counts, config, dtype):
    cells, empty = counts
    kwargs, with_empty = CONFIGS[config]
    res = rsc.pp.dsb(
        cells, empty if with_empty else None, dtype=dtype, copy=True, **kwargs
    )
    assert res.X.dtype == dtype
    np.testing.assert_allclose(
        res.X, _read(config).to_numpy(), rtol=0, atol=_tol(dtype)
    )


@pytest.mark.parametrize("dtype", [np.float64, np.float32])
def test_stats(counts, dtype):
    cells, empty = counts
    res = rsc.pp.dsb(cells, empty, isotype_controls=ISOTYPES, dtype=dtype, copy=True)
    ref = _read("default_technical_stats")
    tol = _tol(dtype)
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
    np.testing.assert_allclose(
        res.var["dsb_ambient_mean"], log_empty.mean(0), rtol=1e-5
    )
    np.testing.assert_allclose(
        res.var["dsb_ambient_sd"], log_empty.std(0, ddof=1), rtol=1e-5
    )


def test_no_denoising(counts):
    cells, empty = counts
    res = rsc.pp.dsb(cells, empty, denoise_counts=False, copy=True)
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
    res = rsc.pp.dsb(cells, empty, copy=True)
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
        rsc.pp.dsb(cells, empty[:, 1:].copy(), copy=True)


def test_invalid_arguments(counts):
    cells, empty = counts
    with pytest.raises(ValueError, match="isotype controls not found"):
        rsc.pp.dsb(cells, empty, isotype_controls=["nope"], copy=True)
    with pytest.raises(ValueError, match="scale_factor"):
        rsc.pp.dsb(cells, empty, scale_factor="scale", copy=True)
    with pytest.warns(UserWarning, match="isotype controls"):
        rsc.pp.dsb(cells, empty, copy=True)


def test_fast_km_is_optimal_two_means(counts):
    cells, empty = counts
    res = rsc.pp.dsb(cells, empty, isotype_controls=ISOTYPES, fast_km=True, copy=True)
    step1 = rsc.pp.dsb(cells, empty, denoise_counts=False, copy=True).X
    for x, low in zip(step1, res.obs["dsb_cell_background_mean"]):
        xs = np.sort(x)
        sse = [
            ((xs[:s] - xs[:s].mean()) ** 2).sum()
            + ((xs[s:] - xs[s:].mean()) ** 2).sum()
            for s in range(1, len(xs))
        ]
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


def _ref_mclust2(x, sub=None, *, eps=np.finfo(np.float64).eps):
    """NumPy reference of the fits (see ``_mixture_fits``); `sub` is the
    random subset that starts long vectors."""

    def mstep(v, z, equal):
        nk = z.sum(0)
        mu = (z * v[:, None]).sum(0) / nk
        ss = (z * (v[:, None] - mu) ** 2).sum(0)
        return mu, np.full(2, ss.sum() / len(v)) if equal else ss / nk, nk / len(v)

    def estep(mu, s2, pro):
        lg = (
            np.log(pro)
            - 0.5 * np.log(2 * np.pi * s2)
            - 0.5 * (x[:, None] - mu) ** 2 / s2
        )
        ll = np.logaddexp(lg[:, 0], lg[:, 1])
        return np.exp(lg - ll[:, None]), ll.sum()

    def valid(p):
        return np.all(np.isfinite(p[0])) and np.all(p[1] > eps)

    start = x if sub is None else sub
    c = start < _ref_threshold(start)
    fits = []
    for equal, npar in ((True, 4), (False, 5)):
        p = mstep(start, np.stack([c, ~c], 1).astype(float), equal)
        skip, prev, ok = sub is not None, None, valid(p)
        while ok:
            z, ll = estep(*p)
            pn = mstep(x, z, equal)
            if skip:
                skip = False
            elif prev is not None and abs(ll - prev) < 1e-5 * (1 + abs(ll)):
                break
            else:
                prev = ll
            ok = valid(pn)
            p = pn if ok else p
        if not ok:
            continue
        if sub is not None or ((p[2] - pn[2]) ** 2).sum() > np.sqrt(eps):
            p = pn
        fits.append((2 * ll - npar * np.log(len(x)), p))
    return max(fits, key=lambda f: f[0])[1] if fits else None


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


def test_mixture_fits_match_reference():
    rng = np.random.default_rng(0)
    for x in _random_rows(rng, 60):
        fit = _mixture_fits(cp.asarray(x[None, :]), rng)
        assert float(fit.q[0]) == pytest.approx(_ref_threshold(x), abs=1e-12)
        ref = _ref_mclust2(x)
        if ref is None:
            assert int(fit.status[0]) == -1
            continue
        got = cp.asnumpy(fit.params[0])
        np.testing.assert_allclose(got[:2], ref[0], rtol=1e-9, atol=1e-10)
        np.testing.assert_allclose(got[2:4], ref[1], rtol=1e-9, atol=1e-10)
        assert got[4] == pytest.approx(ref[2][0], abs=1e-10)


def test_mixture_fits_subset_path():
    # vectors longer than 2000 start from a random subset of 2000 values
    n = 3000
    gen = np.random.default_rng(1)
    X = np.stack(
        [
            np.log(gen.poisson(gen.uniform(0.1, 5), n) + 1.0),
            np.r_[gen.normal(0, 1, 2000), gen.normal(4, 1.5, 1000)],
        ]
    )
    fit = _mixture_fits(cp.asarray(X), np.random.default_rng(7))
    again = _mixture_fits(cp.asarray(X), np.random.default_rng(7))
    cp.testing.assert_array_equal(fit.params, again.params)

    sub_rng = np.random.default_rng(7)
    for x, got in zip(X, cp.asnumpy(fit.params)):
        ref = _ref_mclust2(x, x[sub_rng.choice(n, 2000, replace=False)])
        np.testing.assert_allclose(got[:2], ref[0], rtol=1e-9, atol=1e-10)


@pytest.mark.parametrize("dtype", [np.float32, np.float64])
def test_mixture_fits_terminate_on_ties(dtype):
    # log(counts + 10) with mostly zero counts: heavy ties, near-singular fits
    rng = np.random.default_rng(3)
    counts = rng.poisson(0.05, (500, 87))
    counts[:, :3] = rng.poisson(50, (500, 3))
    X = cp.asarray(np.log(counts + 10.0), dtype=dtype)
    fit = _mixture_fits(X, rng)
    status = cp.asnumpy(fit.status)
    assert set(np.unique(status)) <= {-1, 0, 1}
    assert np.isfinite(cp.asnumpy(fit.params)[status >= 0]).all()
    # whole-column fits of the same data, including the subset path
    cols = cp.ascontiguousarray(cp.tile(X, (5, 1)).T)
    fit = _mixture_fits(cols, rng)
    assert cp.asnumpy(fit.status).shape == (87,)


def test_mixture_constant_rows():
    X = cp.asarray(np.r_[np.ones((2, 30)), np.arange(60.0).reshape(2, 30)])
    fit = _mixture_fits(X, np.random.default_rng(0))
    np.testing.assert_array_equal(cp.asnumpy(fit.status[:2]), [-1, -1])
    assert (cp.asnumpy(fit.status[2:]) >= 0).all()


def test_constant_technical_component(counts):
    # limma::removeBatchEffect returns the data unchanged for a constant
    # covariate; identical cells give a constant background mean
    cells, empty = counts
    same = AnnData(np.repeat(cells.X[:1], 20, axis=0), var=cells.var)
    expected = rsc.pp.dsb(same, empty, denoise_counts=False, copy=True).X
    with pytest.warns(UserWarning, match="technical component is constant"):
        res = rsc.pp.dsb(same, empty, isotype_controls=None, copy=True)
    np.testing.assert_array_equal(res.X, expected)


def test_iteration_cap_is_reported(monkeypatch):
    import rapids_singlecell.preprocessing._dsb as dsb_mod

    gen = np.random.default_rng(1)
    X = np.stack(
        [
            gen.normal(0, 1, 3000),
            np.r_[gen.normal(0, 1, 2000), gen.normal(4, 1.5, 1000)],
        ]
    )
    for M in (X[:, :100], X):  # warp and cooperative (subset) paths
        monkeypatch.setattr(dsb_mod, "_MAX_ITER", {np.dtype(np.float64): 2})
        with pytest.warns(UserWarning, match="iteration limit"):
            fit = _mixture_fits(cp.asarray(M), np.random.default_rng(0))
        assert (cp.asnumpy(fit.status) >= 2).all()
        monkeypatch.undo()
        fit = _mixture_fits(cp.asarray(M), np.random.default_rng(0))
        assert (cp.asnumpy(fit.status) < 2).all()
