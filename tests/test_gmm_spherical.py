"""Batched spherical-GMM kernel (the Mixscape mixture).

``spherical_gmm_fit_batched`` fits many independent two-component spherical
mixtures at once -- one CUDA block per segment, component 0 pinned -- and is the
GMM behind ``rsc.ptg.Mixscape`` (the fused projection+EM kernel shares the same
in-kernel EM and is covered end-to-end by ``tests/pertpy/test_mixscape.py``).
Parity is checked against an independent NumPy reference at float32 and float64.
"""

from __future__ import annotations

import cupy as cp
import numpy as np
import pytest

from rapids_singlecell.squidpy_gpu._gmm import spherical_gmm_fit_batched

_HALF_LOG2PI = 0.5 * np.log(2.0 * np.pi)
_WEIGHT_FLOOR = 1e-10


def _ref_fixed_spherical_em(pvec, m0, v0, m1, v1, *, max_iter, tol, reg_covar):
    """NumPy reference 2-component spherical EM with component 0 pinned.

    Mirrors ``mixscape_em_batched_kernel``: component 0 stays at ``(m0, v0)``,
    component 1 is free (init ``(m1, v1)``), uniform initial weights. Returns the
    component-1 posterior per cell plus the fitted ``(m1, v1, w1)``.
    """
    pvec = np.asarray(pvec, dtype=np.float64)
    n = pvec.shape[0]
    cv0 = max(v0, reg_covar)
    w1 = 0.5
    prev = -np.inf

    def _components(w1, m1, v1):
        c0 = np.log(max(1.0 - w1, _WEIGHT_FLOOR)) - 0.5 * np.log(cv0) - _HALF_LOG2PI
        c1 = np.log(max(w1, _WEIGHT_FLOOR)) - 0.5 * np.log(v1) - _HALF_LOG2PI
        lp0 = c0 - 0.5 * (pvec - m0) ** 2 / cv0
        lp1 = c1 - 0.5 * (pvec - m1) ** 2 / v1
        return lp0, lp1

    for _ in range(max_iter):
        lp0, lp1 = _components(w1, m1, v1)
        mx = np.maximum(lp0, lp1)
        ll = mx + np.log(np.exp(lp0 - mx) + np.exp(lp1 - mx))
        meanll = ll.mean()
        if abs(meanll - prev) < tol:
            break
        prev = meanll
        r1 = np.exp(lp1 - ll)
        n1 = r1.sum()
        inv = 1.0 / max(n1, 1e-12)
        w1 = n1 / n
        m1 = (r1 * pvec).sum() * inv
        v1 = max((r1 * pvec * pvec).sum() * inv - m1 * m1 + reg_covar, reg_covar)

    lp0, lp1 = _components(w1, m1, v1)
    # Stable posterior via the same logsumexp used in the EM loop above
    # (the naive 1/(1+exp(lp0-lp1)) overflows on degenerate segments).
    mx = np.maximum(lp0, lp1)
    ll = mx + np.log(np.exp(lp0 - mx) + np.exp(lp1 - mx))
    resp1 = np.exp(lp1 - ll)
    return resp1, m1, v1, w1


def _bimodal_genes(seed):
    """A ragged batch of bimodal segments (control mode at 0, perturbed shifted)."""
    rng = np.random.default_rng(seed)
    sizes = [500, 1500, 800, 2000, 600]
    pvecs, m0s, v0s, m1is, v1is = [], [], [], [], []
    for i, n in enumerate(sizes):
        ctrl = rng.normal(0.0, 1.0, n // 2)
        pert = rng.normal(4.0 + 0.5 * i, 1.2, n - n // 2)
        pvecs.append(np.concatenate([ctrl, pert]))
        m0s.append(0.0)
        v0s.append(1.0)
        m1is.append(4.0 + 0.5 * i)
        v1is.append(1.44)
    offsets = np.concatenate([[0], np.cumsum(sizes)])
    return pvecs, offsets, np.array(m0s), np.array(v0s), np.array(m1is), np.array(v1is)


@pytest.mark.parametrize("dtype", [np.float32, np.float64])
def test_batched_spherical_matches_numpy_reference(dtype):
    pvecs, offsets, m0s, v0s, m1is, v1is = _bimodal_genes(0)
    pvec_flat = np.concatenate(pvecs).astype(dtype)
    # Tight tol so kernel and reference converge to the same optimum.
    r1, m1, v1, w1 = spherical_gmm_fit_batched(
        cp.asarray(pvec_flat),
        cp.asarray(offsets.astype(np.int32)),
        m0=cp.asarray(m0s.astype(dtype)),
        v0=cp.asarray(v0s.astype(dtype)),
        m1_init=cp.asarray(m1is.astype(dtype)),
        v1_init=cp.asarray(v1is.astype(dtype)),
        max_iter=200,
        tol=1e-6,
        reg_covar=1e-6,
    )
    r1 = cp.asnumpy(r1)
    resp_atol = 2e-3 if dtype == np.float32 else 1e-6
    par_atol = 1e-2 if dtype == np.float32 else 1e-5
    for gi in range(len(pvecs)):
        s, e = int(offsets[gi]), int(offsets[gi + 1])
        ref_r1, ref_m1, ref_v1, ref_w1 = _ref_fixed_spherical_em(
            pvec_flat[s:e],
            float(m0s[gi]),
            float(v0s[gi]),
            float(m1is[gi]),
            float(v1is[gi]),
            max_iter=200,
            tol=1e-6,
            reg_covar=1e-6,
        )
        np.testing.assert_allclose(r1[s:e], ref_r1, atol=resp_atol)
        # KO classification (the posterior > 0.5 decision) must match exactly.
        assert np.array_equal(r1[s:e] > 0.5, ref_r1 > 0.5)
        assert abs(float(m1[gi]) - ref_m1) < par_atol
        assert abs(float(v1[gi]) - ref_v1) < par_atol
        assert abs(float(w1[gi]) - ref_w1) < par_atol


def test_batched_spherical_recovers_perturbed_mode():
    # Component 0 is pinned at the control mode (0); component 1 must move to the
    # shifted perturbed mode regardless of a deliberately poor init.
    pvecs, offsets, m0s, v0s, _m1is, _v1is = _bimodal_genes(1)
    pvec_flat = np.concatenate(pvecs).astype(np.float32)
    bad_m1 = np.full(len(pvecs), 0.5, dtype=np.float32)  # start near control
    bad_v1 = np.ones(len(pvecs), dtype=np.float32)
    _r1, m1, _v1, _w1 = spherical_gmm_fit_batched(
        cp.asarray(pvec_flat),
        cp.asarray(offsets.astype(np.int32)),
        m0=cp.asarray(m0s.astype(np.float32)),
        v0=cp.asarray(v0s.astype(np.float32)),
        m1_init=cp.asarray(bad_m1),
        v1_init=cp.asarray(bad_v1),
        max_iter=200,
        tol=1e-6,
        reg_covar=1e-6,
    )
    m1 = cp.asnumpy(m1)
    for gi in range(len(pvecs)):
        assert m1[gi] > 2.0  # recovered the perturbed mode, away from control


def test_batched_spherical_single_segment_shapes():
    pvec = np.concatenate([np.zeros(100, np.float32), np.full(100, 5.0, np.float32)])
    r1, m1, v1, w1 = spherical_gmm_fit_batched(
        cp.asarray(pvec),
        cp.asarray(np.array([0, pvec.shape[0]], np.int32)),
        m0=cp.asarray(np.array([0.0], np.float32)),
        v0=cp.asarray(np.array([1.0], np.float32)),
        m1_init=cp.asarray(np.array([5.0], np.float32)),
        v1_init=cp.asarray(np.array([1.0], np.float32)),
        max_iter=100,
    )
    assert r1.shape == (pvec.shape[0],)
    assert m1.shape == v1.shape == w1.shape == (1,)
    assert np.all((cp.asnumpy(r1) >= 0.0) & (cp.asnumpy(r1) <= 1.0))


@pytest.mark.parametrize("dtype", [np.float32, np.float64])
def test_batched_spherical_degenerate_segments(dtype):
    """Empty, tiny (n<=2) and zero-variance segments stay finite (the kernel's
    guarded/floored branches), and match the reference where it is defined."""
    seg0 = np.concatenate(
        [
            np.random.default_rng(3).normal(0, 1, 300),
            np.random.default_rng(4).normal(4, 1, 300),
        ]
    )
    seg1 = np.array([], dtype=float)  # empty (offsets[g]==offsets[g+1])
    seg2 = np.full(50, 2.0)  # zero variance (all identical)
    seg3 = np.array([0.0, 5.0])  # tiny (n=2)
    segs = [seg0, seg1, seg2, seg3]
    pvec = np.concatenate(segs).astype(dtype)
    sizes = [s.size for s in segs]
    offsets = np.concatenate([[0], np.cumsum(sizes)]).astype(np.int32)
    m0 = np.zeros(4, dtype)
    v0 = np.ones(4, dtype)
    m1i = np.array([4.0, 0.0, 2.0, 5.0], dtype)
    v1i = np.ones(4, dtype)

    r1, m1, v1, w1 = spherical_gmm_fit_batched(
        cp.asarray(pvec),
        cp.asarray(offsets),
        m0=cp.asarray(m0),
        v0=cp.asarray(v0),
        m1_init=cp.asarray(m1i),
        v1_init=cp.asarray(v1i),
        max_iter=100,
        tol=1e-6,
        reg_covar=1e-6,
    )
    r1 = cp.asnumpy(r1)
    # every output is finite and posteriors are valid probabilities
    assert np.all(np.isfinite(r1)) and np.all((r1 >= 0.0) & (r1 <= 1.0))
    for arr in (m1, v1, w1):
        assert np.all(np.isfinite(cp.asnumpy(arr)))
    # well-conditioned segments (bimodal + tiny) match the NumPy reference
    for gi in (0, 3):
        s, e = int(offsets[gi]), int(offsets[gi + 1])
        ref, *_ = _ref_fixed_spherical_em(
            pvec[s:e],
            float(m0[gi]),
            float(v0[gi]),
            float(m1i[gi]),
            float(v1i[gi]),
            max_iter=100,
            tol=1e-6,
            reg_covar=1e-6,
        )
        np.testing.assert_allclose(
            r1[s:e], ref, atol=2e-3 if dtype == np.float32 else 1e-6
        )
