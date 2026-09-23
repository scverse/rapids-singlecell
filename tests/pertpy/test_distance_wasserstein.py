from __future__ import annotations

import cupy as cp
import numpy as np
import pandas as pd
import pytest
from anndata import AnnData

from rapids_singlecell.pertpy_gpu import Distance, MeanVar
from rapids_singlecell.pertpy_gpu._metrics._sinkhorn import (
    finalize,
    make_state,
    run_async,
)
from rapids_singlecell.pertpy_gpu._metrics._wasserstein import (
    WassersteinMetric,
    _build_ragged_layout,
    _launch_build_cost,
)


def _make_grouped_adata(
    *,
    n_groups: int = 4,
    cells_per_group: int = 20,
    n_features: int = 8,
    shift: float = 1.0,
    seed: int = 0,
    dtype: type = np.float32,
) -> AnnData:
    rng = np.random.default_rng(seed)
    total = n_groups * cells_per_group
    X = rng.normal(size=(total, n_features)).astype(dtype)
    for g in range(n_groups):
        X[g * cells_per_group : (g + 1) * cells_per_group] += g * shift

    labels = [f"g{i}" for i in range(n_groups) for _ in range(cells_per_group)]
    obs = pd.DataFrame(
        {"group": pd.Categorical(labels, categories=[f"g{i}" for i in range(n_groups)])}
    )
    adata = AnnData(X.copy(), obs=obs)
    adata.obsm["X_pca"] = X.copy()
    return adata


def _cpu_sinkhorn(
    X: np.ndarray,
    Y: np.ndarray,
    *,
    eps: float,
    max_iter: int = 5000,
    tol: float = 1e-10,
) -> float:
    """Reference: numpy log-domain Sinkhorn, returning OTT-style reg_ot_cost."""
    n_a, n_b = X.shape[0], Y.shape[0]
    C = ((X[:, None, :] - Y[None, :, :]) ** 2).sum(-1)
    log_a = -np.log(n_a) * np.ones(n_a)
    log_b = -np.log(n_b) * np.ones(n_b)
    f = np.zeros(n_a)
    g = np.zeros(n_b)
    for _ in range(max_iter):
        # g update
        M = (-C + f[:, None]) / eps
        g_new = eps * (log_b - _logsumexp(M, axis=0))
        # f update
        M = (-C + g_new[None, :]) / eps
        f_new = eps * (log_a - _logsumexp(M, axis=1))
        if max(np.max(np.abs(f_new - f)), np.max(np.abs(g_new - g))) < tol:
            f = f_new
            g = g_new
            break
        f, g = f_new, g_new
    reg = float(np.sum(f) / n_a + np.sum(g) / n_b + eps * (np.log(n_a) + np.log(n_b)))
    return reg


def _logsumexp(x, axis):
    m = np.max(x, axis=axis, keepdims=True)
    return np.log(np.exp(x - m).sum(axis=axis)) + np.squeeze(m, axis=axis)


def _ragged_solve(pairs, *, eps=None, max_iter=5000, tol=1e-10, omega=1.0):
    """Solve a list of ``(X, Y)`` pairs through the ragged Sinkhorn path.

    Concatenates the pairs into one embedding (each pair is two groups), builds
    the flat no-padding layout the production code uses, and runs the solver.
    Returns ``(reg_ot_cost (n_pairs,), state)``. ``eps`` (scalar or per-pair)
    overrides the regularization; otherwise it is auto ``0.05 * std(C)``.
    """
    dtype = cp.asarray(pairs[0][0]).dtype
    blocks = []
    offs = [0]
    rows_h, cols_h, n_h, m_h = [], [], [], []
    g = 0
    for X, Y in pairs:
        Xc = cp.ascontiguousarray(cp.asarray(X, dtype=dtype))
        Yc = cp.ascontiguousarray(cp.asarray(Y, dtype=dtype))
        blocks += [Xc, Yc]
        na, nb = int(Xc.shape[0]), int(Yc.shape[0])
        start = offs[-1]
        offs += [start + na, start + na + nb]
        gx, gy = g, g + 1
        g += 2
        # Orient the smaller group as rows (matches the production path).
        if na <= nb:
            rows_h.append(gx), cols_h.append(gy), n_h.append(na), m_h.append(nb)
        else:
            rows_h.append(gy), cols_h.append(gx), n_h.append(nb), m_h.append(na)
    emb = cp.ascontiguousarray(cp.concatenate(blocks, axis=0))
    cat_offsets = cp.asarray(offs, dtype=cp.int32)
    cell_indices = cp.arange(offs[-1], dtype=cp.int32)
    layout = _build_ragged_layout(
        cat_offsets,
        cell_indices,
        np.asarray(rows_h, dtype=np.int64),
        np.asarray(cols_h, dtype=np.int64),
        n_h=np.asarray(n_h, dtype=np.int64),
        m_h=np.asarray(m_h, dtype=np.int64),
        dtype=dtype,
    )
    cost = cp.empty(layout["total_cost"], dtype=dtype)
    stream = cp.cuda.get_current_stream()
    _launch_build_cost(emb, layout, cost, stream.ptr)
    st = make_state(layout, cost, epsilon_scale=0.05)
    if eps is not None:
        st["eps"][:] = cp.asarray(eps, dtype=dtype)
    run_async(
        [{"dev": cp.cuda.runtime.getDevice(), "stream": stream, "state": st}],
        max_iter=max_iter,
        tol=tol,
        check_every=20,
        omega=omega,
    )
    return finalize(st), st


# ---------------------------------------------------------------------------
# Sinkhorn solver unit tests (ragged / no-padding layout)
# ---------------------------------------------------------------------------


def test_ragged_sinkhorn_matches_cpu_reference_single_pair() -> None:
    rng = np.random.default_rng(0)
    n_a, n_b, d = 25, 30, 6
    X = rng.normal(size=(n_a, d)).astype(np.float64)
    Y = rng.normal(size=(n_b, d)).astype(np.float64)
    Y[:] += 0.5

    eps = 0.1
    cpu_val = _cpu_sinkhorn(X, Y, eps=eps)
    out, _ = _ragged_solve([(X, Y)], eps=eps, max_iter=5000, tol=1e-10)
    np.testing.assert_allclose(float(out[0]), cpu_val, rtol=1e-6, atol=1e-6)


def test_ragged_sinkhorn_batch_matches_individual() -> None:
    """A ragged batch of heterogeneous pairs equals solving each pair alone.

    The ragged layout stores every pair flat with no padding, so this is the
    no-padding analogue of the old "padded == unpadded" check: differently sized
    pairs packed into one batch must not contaminate each other.
    """
    rng = np.random.default_rng(1)
    shapes = [(20, 30, 5), (40, 25, 5), (15, 50, 5), (35, 35, 5)]
    pairs = [
        (
            rng.normal(size=(n_a, d)).astype(np.float64),
            rng.normal(size=(n_b, d)).astype(np.float64),
        )
        for n_a, n_b, d in shapes
    ]

    batched, _ = _ragged_solve(pairs, max_iter=10000, tol=1e-10)
    singles = np.array(
        [float(_ragged_solve([p], max_iter=10000, tol=1e-10)[0][0]) for p in pairs]
    )
    np.testing.assert_allclose(batched.get(), singles, rtol=1e-7, atol=1e-7)


def test_ragged_sinkhorn_auto_epsilon_uses_std() -> None:
    """Auto-eps should equal scale * std(C), matching OTT-JAX convention."""
    rng = np.random.default_rng(2)
    X = rng.normal(size=(30, 6)).astype(np.float64)
    Y = rng.normal(size=(40, 6)).astype(np.float64)
    C = ((X[:, None, :] - Y[None, :, :]) ** 2).sum(-1)
    expected_eps = 0.05 * float(np.std(C))

    auto_val, st = _ragged_solve([(X, Y)], max_iter=5000, tol=1e-10)
    np.testing.assert_allclose(float(st["eps"][0]), expected_eps, rtol=1e-6)
    fixed_val, _ = _ragged_solve([(X, Y)], eps=expected_eps, max_iter=5000, tol=1e-10)
    np.testing.assert_allclose(float(auto_val[0]), float(fixed_val[0]), rtol=1e-10)


def test_ragged_sinkhorn_rejects_non_float_cost() -> None:
    """make_state guards the cost dtype (float32/float64 only)."""
    cat_offsets = cp.asarray([0, 8, 18], dtype=cp.int32)
    cell_indices = cp.arange(18, dtype=cp.int32)
    layout = _build_ragged_layout(
        cat_offsets,
        cell_indices,
        np.array([0], dtype=np.int64),
        np.array([1], dtype=np.int64),
        n_h=np.array([8], dtype=np.int64),
        m_h=np.array([10], dtype=np.int64),
        dtype=cp.float64,
    )
    int_cost = cp.ones(layout["total_cost"], dtype=cp.int32)
    with pytest.raises(TypeError):
        make_state(layout, int_cost, epsilon_scale=0.05)


# ---------------------------------------------------------------------------
# Reference-value tests through the public API at default config (the way
# users actually call it). Each value is checked against the numpy reference
# Sinkhorn run to convergence at the same auto-epsilon (0.05 * std(C)) the
# default path uses. rtol=1e-4 leaves ~20x margin over the observed error
# (~5e-6) while still catching a default-config regression (e.g. a looser
# default tol). float64 isolates algorithm accuracy from float32 rounding.
# ---------------------------------------------------------------------------


def _auto_eps(X: np.ndarray, Y: np.ndarray) -> float:
    """The per-pair auto-epsilon the default solver uses: 0.05 * std(C)."""
    C = ((X[:, None, :] - Y[None, :, :]) ** 2).sum(-1)
    return 0.05 * float(np.std(C))


def test_pairwise_matches_reference_default_config() -> None:
    n_groups, cpg = 4, 20
    adata = _make_grouped_adata(
        n_groups=n_groups, cells_per_group=cpg, n_features=8, seed=0, dtype=np.float64
    )
    df = Distance(metric="wasserstein").pairwise(adata, groupby="group")
    X = adata.obsm["X_pca"]
    for i in range(n_groups):
        for j in range(i + 1, n_groups):
            Xi = X[i * cpg : (i + 1) * cpg]
            Xj = X[j * cpg : (j + 1) * cpg]
            ref = _cpu_sinkhorn(Xi, Xj, eps=_auto_eps(Xi, Xj))
            np.testing.assert_allclose(
                df.loc[f"g{i}", f"g{j}"], ref, rtol=1e-4, atol=1e-4
            )


def test_onesided_matches_reference_default_config() -> None:
    n_groups, cpg = 4, 20
    adata = _make_grouped_adata(
        n_groups=n_groups, cells_per_group=cpg, n_features=8, seed=0, dtype=np.float64
    )
    s = Distance(metric="wasserstein").onesided_distances(adata, "group", "g0")
    X = adata.obsm["X_pca"]
    Xi = X[0:cpg]
    for j in range(1, n_groups):
        Xj = X[j * cpg : (j + 1) * cpg]
        ref = _cpu_sinkhorn(Xi, Xj, eps=_auto_eps(Xi, Xj))
        np.testing.assert_allclose(s[f"g{j}"], ref, rtol=1e-4, atol=1e-4)


def test_compute_distance_matches_reference_default_config() -> None:
    rng = np.random.default_rng(1)
    X = rng.normal(size=(25, 6)).astype(np.float64)
    Y = (rng.normal(size=(30, 6)) + 0.5).astype(np.float64)
    val = Distance(metric="wasserstein")(X, Y)  # public call -> compute_distance
    ref = _cpu_sinkhorn(X, Y, eps=_auto_eps(X, Y))
    np.testing.assert_allclose(val, ref, rtol=1e-4, atol=1e-4)


# ---------------------------------------------------------------------------
# True upstream parity: rapids_singlecell == pertpy (OTT-JAX) across the whole
# public API (pairwise, onesided, single call). Needs pertpy *and* its OTT-JAX
# backend, a separate extra since pertpy 1.3.0; skipped without either. Both
# use a squared-Euclidean cost and the same auto-epsilon (0.05 * std(C) is
# exactly OTT's PointCloud default), so values agree to ~1e-6. rtol=1e-4 leaves
# ~25x margin and still catches a divergent cost or epsilon (which differ by
# orders of magnitude).
# ---------------------------------------------------------------------------


def test_matches_pertpy_ott_parity() -> None:
    pytest.importorskip("pertpy")
    # pertpy alone is no longer enough. Since 1.3.0 the JAX stack sits behind
    # the "jax", "scgen" and "tcoda" extras, so pertpy can be installed while
    # its OTT backend is not -- and pertpy's wasserstein metric then raises
    # ImportError at call time rather than being absent. Skip on the backend
    # this parity actually needs.
    pytest.importorskip("ott")
    from pertpy.tools import Distance as PertpyDistance

    adata = _make_grouped_adata(
        n_groups=4, cells_per_group=20, n_features=8, seed=0, dtype=np.float64
    )
    ours = Distance(metric="wasserstein")
    upstream = PertpyDistance(metric="wasserstein", obsm_key="X_pca")

    # pairwise: full K x K matrix
    rp = ours.pairwise(adata, groupby="group")
    pp = upstream.pairwise(adata, groupby="group", show_progressbar=False).loc[
        rp.index, rp.columns
    ]
    np.testing.assert_allclose(rp.values, pp.values, rtol=1e-4, atol=1e-4)

    # onesided: distances from one group to the rest (compare the other groups)
    rs = ours.onesided_distances(adata, "group", "g0")
    ps = upstream.onesided_distances(
        adata, groupby="group", selected_group="g0", show_progressbar=False
    )
    others = [g for g in rs.index if g != "g0" and g in ps.index]
    np.testing.assert_allclose(
        rs.reindex(others).values, ps.reindex(others).values, rtol=1e-4, atol=1e-4
    )

    # single call (X, Y)
    X = adata.obsm["X_pca"][:20]
    Y = adata.obsm["X_pca"][20:40]
    np.testing.assert_allclose(ours(X, Y), float(upstream(X, Y)), rtol=1e-4, atol=1e-4)


# ---------------------------------------------------------------------------
# Distance(metric="wasserstein") API tests
# ---------------------------------------------------------------------------


def test_distance_wasserstein_initialization() -> None:
    distance = Distance(metric="wasserstein")
    assert distance.metric == "wasserstein"
    assert distance.obsm_key == "X_pca"


def test_relaxation_option() -> None:
    """Over-relaxation (omega > 1) reaches the same fixed point; bounds enforced.

    Also exercises Distance(**kwargs) forwarding of metric-specific options.
    """
    adata = _make_grouped_adata(
        n_groups=4,
        cells_per_group=30,
        n_features=8,
        shift=1.0,
        seed=3,
        dtype=np.float64,
    )
    base = Distance(metric="wasserstein").pairwise(adata, "group")  # omega = 1
    d_accel = Distance(metric="wasserstein", relaxation=1.5)  # forwarded via **kwargs
    assert d_accel._metric_impl.relaxation == 1.5
    accel = d_accel.pairwise(adata, "group")  # over-relaxed
    # SOR converges to the same fixed point (just in fewer iterations)
    np.testing.assert_allclose(accel.values, base.values, rtol=1e-3, atol=1e-3)
    # bounds: must be in [1, 2)
    with pytest.raises(ValueError):
        Distance(metric="wasserstein", relaxation=2.5)
    with pytest.raises(ValueError):
        Distance(metric="wasserstein", relaxation=0.5)


@pytest.mark.parametrize("omega", [1.2, 1.4, 1.6])
def test_relaxation_converges_across_omega(omega) -> None:
    """Over-relaxation across [1, ~1.7) reaches the same fixed point as omega=1."""
    adata = _make_grouped_adata(
        n_groups=4,
        cells_per_group=25,
        n_features=8,
        shift=1.0,
        seed=3,
        dtype=np.float64,
    )
    base = Distance(metric="wasserstein").pairwise(adata, "group")
    accel = Distance(metric="wasserstein", relaxation=omega).pairwise(adata, "group")
    np.testing.assert_allclose(accel.values, base.values, rtol=1e-5, atol=1e-5)


def test_high_omega_diverges_gracefully() -> None:
    """Too-large omega (past the divergence cliff) hits the iteration cap and
    warns, but stays finite and never crashes -- on both the pairwise and the
    bootstrap solve paths."""
    adata = _make_grouped_adata(
        n_groups=4,
        cells_per_group=25,
        n_features=8,
        shift=1.0,
        seed=3,
        dtype=np.float64,
    )
    d = Distance(metric="wasserstein", relaxation=1.99)
    with pytest.warns(RuntimeWarning, match="did not converge"):
        df = d.pairwise(adata, "group")
    assert np.all(np.isfinite(df.values))  # diverged but finite, no crash
    wm = WassersteinMetric(relaxation=1.99)
    with pytest.warns(RuntimeWarning, match="did not converge"):
        mean, var = wm.bootstrap(
            adata, "group", "g0", "g1", n_bootstrap=5, random_state=0
        )
    assert np.isfinite(mean) and var >= 0


def test_distance_wasserstein_pairwise_shape_and_diag() -> None:
    adata = _make_grouped_adata(n_groups=4, cells_per_group=15, n_features=6, seed=10)
    distance = Distance(metric="wasserstein")
    df = distance.pairwise(adata, groupby="group")

    assert isinstance(df, pd.DataFrame)
    assert df.shape == (4, 4)
    np.testing.assert_allclose(np.diag(df.values), 0.0, atol=1e-8)
    np.testing.assert_allclose(df.values, df.values.T, rtol=0, atol=1e-6)
    assert np.all(np.isfinite(df.values))


def test_distance_wasserstein_pairwise_monotonic_in_separation() -> None:
    """Pairs farther apart in feature space should have larger distance."""
    adata = _make_grouped_adata(
        n_groups=4, cells_per_group=20, n_features=8, shift=1.0, seed=21
    )
    distance = Distance(metric="wasserstein")
    df = distance.pairwise(adata, groupby="group")
    # g0-g3 (shift 3) > g0-g1 (shift 1)
    assert df.loc["g0", "g3"] > df.loc["g0", "g1"]
    assert df.loc["g0", "g2"] > df.loc["g0", "g1"]


def test_distance_wasserstein_compute_distance_matches_pairwise() -> None:
    adata = _make_grouped_adata(n_groups=3, cells_per_group=18, n_features=5, seed=33)
    distance = Distance(metric="wasserstein")
    df = distance.pairwise(adata, groupby="group")

    cells = adata.obsm["X_pca"]
    groups = adata.obs["group"].values
    A = cells[groups == "g0"]
    B = cells[groups == "g1"]
    direct = distance(A, B)
    # Sinkhorn convergence path differs slightly; allow small tol.
    np.testing.assert_allclose(direct, df.loc["g0", "g1"], rtol=5e-4, atol=1e-4)


def test_distance_wasserstein_onesided_single_control() -> None:
    adata = _make_grouped_adata(n_groups=4, cells_per_group=20, n_features=6, seed=4)
    distance = Distance(metric="wasserstein")
    s = distance.onesided_distances(adata, groupby="group", selected_group="g0")
    assert isinstance(s, pd.Series)
    assert len(s) == 4
    assert s.loc["g0"] == pytest.approx(0.0, abs=1e-8)
    assert np.all(s.values >= -1e-6)


def test_distance_wasserstein_onesided_multi_control_matches_single() -> None:
    adata = _make_grouped_adata(n_groups=4, cells_per_group=20, n_features=6, seed=5)
    distance = Distance(metric="wasserstein")
    df = distance.onesided_distances(
        adata, groupby="group", selected_group=["g0", "g2"]
    )
    assert isinstance(df, pd.DataFrame)
    assert df.shape == (4, 2)

    s0 = distance.onesided_distances(adata, groupby="group", selected_group="g0")
    s2 = distance.onesided_distances(adata, groupby="group", selected_group="g2")
    np.testing.assert_allclose(df["g0"].values, s0.values, rtol=1e-3, atol=1e-3)
    np.testing.assert_allclose(df["g2"].values, s2.values, rtol=1e-3, atol=1e-3)


def test_distance_wasserstein_contrast_distances() -> None:
    adata = _make_grouped_adata(n_groups=4, cells_per_group=15, n_features=5, seed=6)
    distance = Distance(metric="wasserstein")
    contrasts = Distance.create_contrasts(adata, groupby="group", selected_group="g0")
    result = distance.contrast_distances(adata, contrasts=contrasts)
    assert "wasserstein" in result.columns
    assert len(result) == 3
    assert np.all(np.isfinite(result["wasserstein"].values))
    assert np.all(result["wasserstein"].values > 0)


def test_distance_wasserstein_pairwise_subset_groups() -> None:
    adata = _make_grouped_adata(n_groups=5, cells_per_group=15, n_features=5, seed=7)
    distance = Distance(metric="wasserstein")
    df_sub = distance.pairwise(adata, groupby="group", groups=["g0", "g2", "g4"])
    assert df_sub.shape == (3, 3)
    assert list(df_sub.columns) == ["g0", "g2", "g4"]


# ---------------------------------------------------------------------------
# Bootstrap (resample cells with replacement). Wasserstein has no structural
# shortcut like edistance, so each resample is a full re-solve, batched on one
# GPU. Stochastic, so checks are structural (var>=0, reproducible, mean ~ point
# estimate) rather than exact-value -- same approach as the edistance suite.
# ---------------------------------------------------------------------------


def test_bootstrap_pairwise_variance_and_reproducible() -> None:
    adata = _make_grouped_adata(n_groups=3, cells_per_group=25, n_features=6, seed=1)
    distance = Distance(metric="wasserstein")
    mean1, var1 = distance.pairwise(
        adata, groupby="group", bootstrap=True, n_bootstrap=30, random_state=7
    )
    assert isinstance(mean1, pd.DataFrame) and isinstance(var1, pd.DataFrame)
    np.testing.assert_allclose(np.diag(mean1.values), 0.0, atol=1e-8)
    assert np.all(var1.values >= 0)
    assert np.all(var1.values[~np.eye(3, dtype=bool)] > 0)  # off-diagonal var > 0
    # same seed -> identical mean and variance
    mean2, var2 = distance.pairwise(
        adata, groupby="group", bootstrap=True, n_bootstrap=30, random_state=7
    )
    np.testing.assert_array_equal(mean1.values, mean2.values)
    np.testing.assert_array_equal(var1.values, var2.values)


def test_bootstrap_mean_close_to_point_estimate() -> None:
    adata = _make_grouped_adata(
        n_groups=3, cells_per_group=60, n_features=8, shift=1.0, seed=2
    )
    distance = Distance(metric="wasserstein")
    point = distance.pairwise(adata, groupby="group")
    boot_mean, _ = distance.pairwise(
        adata, groupby="group", bootstrap=True, n_bootstrap=100, random_state=2
    )
    np.testing.assert_allclose(boot_mean.values, point.values, rtol=0.3, atol=0.5)


def test_bootstrap_arrays_and_onesided() -> None:
    adata = _make_grouped_adata(n_groups=4, cells_per_group=25, n_features=6, seed=3)
    distance = Distance(metric="wasserstein")
    X = adata.obsm["X_pca"][:25]
    Y = adata.obsm["X_pca"][25:50]
    r = distance.bootstrap(X, Y, n_bootstrap=40, random_state=5)
    assert isinstance(r, MeanVar) and r.variance >= 0
    assert r == distance.bootstrap(X, Y, n_bootstrap=40, random_state=5)  # reproducible
    np.testing.assert_allclose(r.mean, distance(X, Y), rtol=0.3, atol=0.5)
    ms, vs = distance.onesided_distances(
        adata,
        groupby="group",
        selected_group="g0",
        bootstrap=True,
        n_bootstrap=30,
        random_state=5,
    )
    assert isinstance(ms, pd.Series) and isinstance(vs, pd.Series)
    assert ms.loc["g0"] == pytest.approx(0.0, abs=1e-8)
    assert np.all(vs.values >= 0)


def test_bootstrap_adata_method_and_multicontrol() -> None:
    adata = _make_grouped_adata(n_groups=4, cells_per_group=20, n_features=6, seed=4)
    # adata-based bootstrap(group_a, group_b) -> (mean, var) floats
    mean, var = WassersteinMetric().bootstrap(
        adata, "group", "g0", "g1", n_bootstrap=20, random_state=1
    )
    assert isinstance(mean, float) and isinstance(var, float) and var >= 0
    # multi-control onesided bootstrap -> (DataFrame, DataFrame)
    md, vd = Distance(metric="wasserstein").onesided_distances(
        adata, "group", ["g0", "g2"], bootstrap=True, n_bootstrap=20, random_state=1
    )
    assert isinstance(md, pd.DataFrame) and isinstance(vd, pd.DataFrame)
    assert list(md.columns) == ["g0", "g2"]
    assert np.all(vd.values >= 0)


def test_bootstrap_guards() -> None:
    """Non-positive n_bootstrap raises; an empty workload (no pairs) is a no-op."""
    distance = Distance(metric="wasserstein")
    rng = np.random.default_rng(0)
    X = rng.normal(size=(10, 5)).astype(np.float32)
    Y = rng.normal(size=(12, 5)).astype(np.float32)
    with pytest.raises(ValueError):
        distance.bootstrap(X, Y, n_bootstrap=0)
    # single-group adata -> no pairs -> bootstrap pairwise must not crash
    adata = _make_grouped_adata(n_groups=1, cells_per_group=12, seed=2)
    mean, var = distance.pairwise(adata, "group", bootstrap=True, n_bootstrap=5)
    assert mean.shape == (1, 1)
    assert float(mean.iloc[0, 0]) == 0.0 and float(var.iloc[0, 0]) == 0.0


# ---------------------------------------------------------------------------
# edist-parity coverage: dtypes, layer_key, output/axioms, groups filtering,
# contrasts, __call__, invalid-group. Block-size and triangle-inequality are
# intentionally not ported (Sinkhorn has no block-size dual path; entropic W2
# is not a metric).
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("dtype", [np.float32, np.float64])
def test_pairwise_correctness_dtypes(dtype) -> None:
    cpg = 15
    adata = _make_grouped_adata(
        n_groups=3, cells_per_group=cpg, n_features=6, seed=4, dtype=dtype
    )
    df = Distance(metric="wasserstein").pairwise(adata, "group")
    X = np.asarray(adata.obsm["X_pca"], dtype=np.float64)
    rtol = 1e-4 if dtype == np.float32 else 1e-5
    for i in range(3):
        for j in range(i + 1, 3):
            Xi, Xj = X[i * cpg : (i + 1) * cpg], X[j * cpg : (j + 1) * cpg]
            ref = _cpu_sinkhorn(Xi, Xj, eps=_auto_eps(Xi, Xj))
            np.testing.assert_allclose(df.iloc[i, j], ref, rtol=rtol, atol=1e-4)


def test_float32_matches_float64() -> None:
    """float32 and float64 pairwise agree within float32 precision.

    Same input values (float32), one run upcast to float64, so this isolates the
    solver dtype. Observed max rel diff ~2e-7 (auto-eps keeps it well
    conditioned); rtol=1e-5 leaves ~50x margin.
    """
    base = _make_grouped_adata(
        n_groups=5,
        cells_per_group=30,
        n_features=10,
        shift=1.5,
        seed=2,
        dtype=np.float32,
    )
    distance = Distance(metric="wasserstein")
    df32 = distance.pairwise(base, "group")
    base64 = base.copy()
    base64.obsm["X_pca"] = base.obsm["X_pca"].astype(np.float64)
    df64 = distance.pairwise(base64, "group")
    np.testing.assert_allclose(df32.values, df64.values, rtol=1e-5, atol=1e-5)


def test_high_dimensional_embedding() -> None:
    """pairwise with a large feature dim (D >> FEAT_TILE) builds the cost and
    matches the reference. The fused cost kernel streams features, so shared
    memory stays bounded regardless of D -- the previous full-D cache overflowed
    the per-block shared-memory limit for large D (e.g. raw-feature layers)."""
    D, cpg = 2048, 20
    adata = _make_grouped_adata(
        n_groups=3,
        cells_per_group=cpg,
        n_features=D,
        shift=0.3,
        seed=4,
        dtype=np.float64,
    )
    df = Distance(metric="wasserstein").pairwise(adata, "group")  # -> build_cost
    X = adata.obsm["X_pca"]
    for i in range(3):
        for j in range(i + 1, 3):
            Xi, Xj = X[i * cpg : (i + 1) * cpg], X[j * cpg : (j + 1) * cpg]
            ref = _cpu_sinkhorn(Xi, Xj, eps=_auto_eps(Xi, Xj))
            np.testing.assert_allclose(df.iloc[i, j], ref, rtol=1e-4, atol=1e-4)


def test_layer_key_matches_reference() -> None:
    rng = np.random.default_rng(7)
    ng, cpg, nf = 3, 15, 6
    data = rng.normal(size=(ng * cpg, nf)).astype(np.float64)
    labels = [f"g{i}" for i in range(ng) for _ in range(cpg)]
    obs = pd.DataFrame(
        {"group": pd.Categorical(labels, categories=[f"g{i}" for i in range(ng)])}
    )
    adata = AnnData(np.zeros((ng * cpg, nf), np.float32), obs=obs)
    adata.layers["counts"] = data
    df = Distance(metric="wasserstein", layer_key="counts").pairwise(adata, "group")
    Xi, Xj = data[:cpg], data[cpg : 2 * cpg]
    ref = _cpu_sinkhorn(Xi, Xj, eps=_auto_eps(Xi, Xj))
    np.testing.assert_allclose(df.loc["g0", "g1"], ref, rtol=1e-4, atol=1e-4)


def test_pairwise_output_format_and_axioms() -> None:
    adata = _make_grouped_adata(
        n_groups=4,
        cells_per_group=20,
        n_features=8,
        shift=1.5,
        seed=3,
        dtype=np.float64,
    )
    df = Distance(metric="wasserstein").pairwise(adata, "group")
    assert list(df.index) == list(df.columns) == [f"g{i}" for i in range(4)]
    assert df.index.name == "group" and df.columns.name == "group"
    np.testing.assert_allclose(np.diag(df.values), 0.0, atol=1e-8)  # definiteness
    np.testing.assert_allclose(df.values, df.values.T, atol=1e-6)  # symmetry
    assert np.all(df.values[~np.eye(4, dtype=bool)] > 0)  # positivity (distinct groups)


def test_groups_filter_values_match_full() -> None:
    adata = _make_grouped_adata(n_groups=5, cells_per_group=15, n_features=5, seed=7)
    distance = Distance(metric="wasserstein")
    full = distance.pairwise(adata, "group")
    sub = distance.pairwise(adata, "group", groups=["g0", "g2", "g4"])
    for a in ["g0", "g2", "g4"]:
        for b in ["g0", "g2", "g4"]:
            np.testing.assert_allclose(
                sub.loc[a, b], full.loc[a, b], rtol=1e-4, atol=1e-4
            )


def test_onesided_invalid_group_raises() -> None:
    adata = _make_grouped_adata(n_groups=3, cells_per_group=10)
    with pytest.raises(ValueError):
        Distance(metric="wasserstein").onesided_distances(
            adata, "group", selected_group="nope"
        )


def test_unused_categories_and_singletons() -> None:
    """Zero-cell (unused) categories are dropped; a single-cell group still
    works; requesting a group with no cells raises."""
    rng = np.random.default_rng(0)
    sizes = {"g0": 15, "g1": 1, "g2": 15}  # g1 is a singleton
    blocks, labels = [], []
    for i, (g, n) in enumerate(sizes.items()):
        blocks.append((rng.normal(size=(n, 6)) + i).astype(np.float32))
        labels += [g] * n
    X = np.vstack(blocks)
    # categorical also lists 'gZ', which has zero cells (unused)
    obs = pd.DataFrame(
        {"group": pd.Categorical(labels, categories=["g0", "g1", "g2", "gZ"])}
    )
    adata = AnnData(X.copy(), obs=obs)
    adata.obsm["X_pca"] = X.copy()
    distance = Distance(metric="wasserstein")
    df = distance.pairwise(adata, "group")
    assert list(df.index) == ["g0", "g1", "g2"]  # unused 'gZ' dropped
    assert np.all(np.isfinite(df.values))  # singleton 'g1' handled
    np.testing.assert_allclose(np.diag(df.values), 0.0, atol=1e-8)
    with pytest.raises(ValueError):  # requesting a zero-cell group
        distance.pairwise(adata, "group", groups=["g0", "gZ"])


def test_call_api_basic_and_cupy() -> None:
    adata = _make_grouped_adata(n_groups=3, cells_per_group=15, seed=8)
    distance = Distance(metric="wasserstein")
    X = adata.obsm["X_pca"][:15]
    Y = adata.obsm["X_pca"][15:30]
    d_np = distance(X, Y)
    assert isinstance(d_np, float) and np.isfinite(d_np)
    d_cp = distance(cp.asarray(X), cp.asarray(Y))
    np.testing.assert_allclose(d_np, d_cp, rtol=1e-5, atol=1e-5)


def test_call_and_bootstrap_input_handling() -> None:
    distance = Distance(metric="wasserstein")
    rng = np.random.default_rng(0)
    X = rng.normal(size=(20, 6)).astype(np.float32)
    Y = rng.normal(size=(25, 6)).astype(
        np.float64
    )  # different dtype -> cast internally
    assert np.isfinite(distance(X, Y))
    assert distance.bootstrap(X, Y, n_bootstrap=10, random_state=0).variance >= 0
    with pytest.raises(ValueError):  # empty input
        distance(X[:0], Y)
    with pytest.raises(ValueError):
        distance.bootstrap(X[:0], Y, n_bootstrap=5)


def test_contrast_self_distance_zero_and_no_split() -> None:
    rng = np.random.default_rng(9)
    n = 12
    emb = rng.normal(size=(n * 2, 5)).astype(np.float32)
    obs = pd.DataFrame({"treatment": pd.Categorical(["ctrl"] * n + ["drugA"] * n)})
    adata = AnnData(emb.copy(), obs=obs)
    adata.obsm["X_pca"] = emb.copy()
    distance = Distance(metric="wasserstein")
    contrasts = Distance.create_contrasts(
        adata, groupby="treatment", selected_group="ctrl"
    )
    res = distance.contrast_distances(adata, contrasts=contrasts)
    assert len(res) == 1 and np.all(np.isfinite(res["wasserstein"].values))
    self_c = pd.DataFrame({"treatment": ["ctrl"], "reference": ["ctrl"]})
    res0 = distance.contrast_distances(adata, contrasts=self_c)
    assert res0["wasserstein"].iloc[0] == pytest.approx(0.0, abs=1e-7)


def test_contrast_distances_with_split_by() -> None:
    rng = np.random.default_rng(9)
    n = 12
    emb = rng.normal(size=(n * 4, 5)).astype(np.float32)
    obs = pd.DataFrame(
        {
            "treatment": pd.Categorical(
                ["ctrl"] * n + ["drugA"] * n + ["ctrl"] * n + ["drugA"] * n
            ),
            "celltype": pd.Categorical(["T"] * n * 2 + ["B"] * n * 2),
        }
    )
    adata = AnnData(emb.copy(), obs=obs)
    adata.obsm["X_pca"] = emb.copy()
    distance = Distance(metric="wasserstein")
    contrasts = Distance.create_contrasts(
        adata, groupby="treatment", selected_group="ctrl", split_by="celltype"
    )
    res = distance.contrast_distances(adata, contrasts=contrasts)
    assert "wasserstein" in res.columns and len(res) == 2
    assert np.all(np.isfinite(res["wasserstein"].values))


requires_2_gpus = pytest.mark.skipif(
    cp.cuda.runtime.getDeviceCount() < 2,
    reason="multi-GPU test requires >= 2 GPUs",
)


@requires_2_gpus
def test_distance_wasserstein_multi_gpu_pairwise_matches_single() -> None:
    """Splitting pairs across GPUs must give the same result as one GPU.

    Convergence is decided per pair (inside the fused kernel), so the result is
    independent of how pairs are split across devices — bitwise identical here.
    """
    adata = _make_grouped_adata(n_groups=8, cells_per_group=40, n_features=10, seed=11)
    distance = Distance(metric="wasserstein")
    df1 = distance.pairwise(adata, groupby="group", multi_gpu=False)
    df2 = distance.pairwise(adata, groupby="group", multi_gpu=[0, 1])
    np.testing.assert_allclose(df2.values, df1.values, rtol=1e-5, atol=1e-5)


@requires_2_gpus
def test_distance_wasserstein_multi_gpu_onesided_matches_single() -> None:
    adata = _make_grouped_adata(n_groups=6, cells_per_group=40, n_features=8, seed=12)
    distance = Distance(metric="wasserstein")
    s1 = distance.onesided_distances(
        adata, groupby="group", selected_group="g0", multi_gpu=False
    )
    s2 = distance.onesided_distances(
        adata, groupby="group", selected_group="g0", multi_gpu=[0, 1]
    )
    np.testing.assert_allclose(s2.values, s1.values, rtol=1e-5, atol=1e-5)


@requires_2_gpus
def test_distance_wasserstein_multi_gpu_contrast_matches_single() -> None:
    adata = _make_grouped_adata(n_groups=6, cells_per_group=30, n_features=6, seed=13)
    distance = Distance(metric="wasserstein")
    contrasts = Distance.create_contrasts(adata, groupby="group", selected_group="g0")
    r1 = distance.contrast_distances(adata, contrasts=contrasts, multi_gpu=False)
    r2 = distance.contrast_distances(adata, contrasts=contrasts, multi_gpu=[0, 1])
    np.testing.assert_allclose(
        r2["wasserstein"].values, r1["wasserstein"].values, rtol=1e-5, atol=1e-5
    )
