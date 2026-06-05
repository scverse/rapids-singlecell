from __future__ import annotations

import cupy as cp
import numpy as np
import pandas as pd
import pytest
from anndata import AnnData
from cupyx.scipy.sparse import csc_matrix as gpu_csc
from cupyx.scipy.sparse import csr_matrix as gpu_csr

import rapids_singlecell as rsc
from rapids_singlecell.get import X_to_CPU
from rapids_singlecell.pertpy_gpu._guide_assignment import _fit_assign_cuda


@pytest.fixture
def guide_adata() -> AnnData:
    """Synthetic guide RNA dataset with clear bimodal signal.

    200 cells, 8 guides.  For each guide, ~30 % of cells have high counts
    (signal, drawn from Poisson(lambda=50)) and the rest have low counts
    (background, drawn from Poisson(lambda=2)).
    """
    rng = np.random.default_rng(42)
    n_cells = 200
    n_guides = 8
    n_signal = 60  # 30% signal

    X = np.zeros((n_cells, n_guides), dtype=np.float32)
    for g in range(n_guides):
        bg = rng.poisson(lam=2, size=n_cells - n_signal).astype(np.float32)
        sig = rng.poisson(lam=50, size=n_signal).astype(np.float32)
        X[:, g] = np.concatenate([bg, sig])
        rng.shuffle(X[:, g])

    var = pd.DataFrame(index=[f"guide_{i}" for i in range(n_guides)])
    obs = pd.DataFrame(index=[f"cell_{i}" for i in range(n_cells)])
    return AnnData(X=cp.array(X), obs=obs, var=var)


@pytest.fixture
def guide_adata_sparse(guide_adata: AnnData) -> AnnData:
    """Same data as guide_adata but stored as CuPy CSR sparse."""
    adata = guide_adata.copy()
    adata.X = gpu_csr(adata.X)
    return adata


# ------------------------------------------------------------------ #
#  assign_by_threshold
# ------------------------------------------------------------------ #


def test_assign_by_threshold(guide_adata: AnnData) -> None:
    ga = rsc.ptg.GuideAssignment()
    ga.assign_by_threshold(guide_adata, assignment_threshold=5)

    result = X_to_CPU(guide_adata.layers["assigned_guides"])
    X = X_to_CPU(guide_adata.X)

    expected = (X >= 5).astype(np.int8)
    np.testing.assert_array_equal(result, expected)


def test_assign_by_threshold_sparse(guide_adata_sparse: AnnData) -> None:
    ga = rsc.ptg.GuideAssignment()
    ga.assign_by_threshold(guide_adata_sparse, assignment_threshold=5)

    result = X_to_CPU(guide_adata_sparse.layers["assigned_guides"]).toarray()
    X = X_to_CPU(guide_adata_sparse.X).toarray()

    expected = (X >= 5).astype(np.int8)
    np.testing.assert_array_equal(result, expected)


def test_assign_by_threshold_sparse_csc(guide_adata: AnnData) -> None:
    adata = guide_adata.copy()
    adata.X = gpu_csc(adata.X)

    ga = rsc.ptg.GuideAssignment()
    ga.assign_by_threshold(adata, assignment_threshold=5)

    result = adata.layers["assigned_guides"]
    assert isinstance(result, gpu_csc)

    result = X_to_CPU(result).toarray()
    expected = (X_to_CPU(adata.X).toarray() >= 5).astype(np.int8)
    np.testing.assert_array_equal(result, expected)


# ------------------------------------------------------------------ #
#  assign_to_max_guide
# ------------------------------------------------------------------ #


def test_assign_to_max_guide(guide_adata: AnnData) -> None:
    ga = rsc.ptg.GuideAssignment()
    ga.assign_to_max_guide(guide_adata, assignment_threshold=5)

    assigned = guide_adata.obs["assigned_guide"]
    X = X_to_CPU(guide_adata.X)

    for i in range(guide_adata.n_obs):
        row = X[i]
        max_val = row.max()
        if max_val >= 5:
            expected_guide = guide_adata.var_names[int(row.argmax())]
            assert assigned.iloc[i] == expected_guide, (
                f"Cell {i}: expected {expected_guide}, got {assigned.iloc[i]}"
            )
        else:
            assert assigned.iloc[i] == "Negative", (
                f"Cell {i}: expected Negative, got {assigned.iloc[i]}"
            )


def test_assign_to_max_guide_sparse(guide_adata_sparse: AnnData) -> None:
    ga = rsc.ptg.GuideAssignment()
    ga.assign_to_max_guide(guide_adata_sparse, assignment_threshold=5)

    assigned = guide_adata_sparse.obs["assigned_guide"]
    X = X_to_CPU(guide_adata_sparse.X).toarray()

    for i in range(guide_adata_sparse.n_obs):
        row = X[i]
        max_val = row.max()
        if max_val >= 5:
            expected_guide = guide_adata_sparse.var_names[int(row.argmax())]
            assert assigned.iloc[i] == expected_guide
        else:
            assert assigned.iloc[i] == "Negative"


def test_assign_to_max_guide_below_threshold() -> None:
    """All counts below threshold → all Negative."""
    rng = np.random.default_rng(0)
    X = rng.poisson(lam=1, size=(50, 4)).astype(np.float32)
    adata = AnnData(
        X=cp.array(X),
        var=pd.DataFrame(index=[f"g{i}" for i in range(4)]),
        obs=pd.DataFrame(index=[f"c{i}" for i in range(50)]),
    )
    ga = rsc.ptg.GuideAssignment()
    ga.assign_to_max_guide(adata, assignment_threshold=100)
    assert (adata.obs["assigned_guide"] == "Negative").all()


# ------------------------------------------------------------------ #
#  assign_mixture_model
# ------------------------------------------------------------------ #


def test_mixture_model_separation(guide_adata: AnnData) -> None:
    """EM should separate the clearly bimodal signal."""
    ga = rsc.ptg.GuideAssignment()
    ga.assign_mixture_model(guide_adata, max_iter=100)

    # Check that at least some cells are assigned (not all negative)
    assigned = guide_adata.obs["assigned_guide"]
    n_assigned = (assigned != "negative").sum()
    assert n_assigned > 0, "No cells were assigned to any guide"

    # With such clear separation (Poisson(2) vs Poisson(50)),
    # most high-count cells should be positive
    X = X_to_CPU(guide_adata.X)

    # For each guide, cells with count >= 20 should mostly be assigned
    for g in range(X.shape[1]):
        high_count_cells = X[:, g] >= 20
        if high_count_cells.sum() == 0:
            continue
        # At least 80% of high-count cells should have this guide in assignment
        guide_name = guide_adata.var_names[g]
        has_guide = assigned.str.contains(guide_name, na=False)
        overlap = (high_count_cells & has_guide.values).sum()
        assert overlap / high_count_cells.sum() >= 0.8, (
            f"Guide {guide_name}: only {overlap}/{high_count_cells.sum()} "
            "high-count cells assigned"
        )


def test_mixture_model_stores_params(guide_adata: AnnData) -> None:
    ga = rsc.ptg.GuideAssignment()
    ga.assign_mixture_model(guide_adata)

    for col in [
        "poisson_rate",
        "gaussian_mean",
        "gaussian_std",
        "mix_probs_0",
        "mix_probs_1",
        "threshold",
        "weight_Poisson",
        "weight_Normal",
        "lambda",
        "mu",
        "scale",
    ]:
        assert col in guide_adata.var.columns, f"Missing column: {col}"

    # Parameters should be finite for all guides
    for col in ["poisson_rate", "gaussian_mean", "gaussian_std"]:
        vals = guide_adata.var[col].dropna()
        assert len(vals) > 0, f"No values for {col}"
        assert np.all(np.isfinite(vals)), f"Non-finite values in {col}"

    # Poisson rate should be < Gaussian mean (anti-flip)
    rates = guide_adata.var["poisson_rate"].dropna()
    means = guide_adata.var["gaussian_mean"].dropna()
    assert (rates < means).all(), "Poisson rate should be < Gaussian mean"

    # Crispat-compatible aliases should mirror the pertpy-style parameter names.
    np.testing.assert_allclose(
        guide_adata.var["weight_Poisson"].dropna(),
        guide_adata.var["mix_probs_0"].dropna(),
    )
    np.testing.assert_allclose(
        guide_adata.var["weight_Normal"].dropna(),
        guide_adata.var["mix_probs_1"].dropna(),
    )
    np.testing.assert_allclose(
        guide_adata.var["lambda"].dropna(),
        guide_adata.var["poisson_rate"].dropna(),
    )
    np.testing.assert_allclose(
        guide_adata.var["mu"].dropna(),
        guide_adata.var["gaussian_mean"].dropna(),
    )
    np.testing.assert_allclose(
        guide_adata.var["scale"].dropna(),
        guide_adata.var["gaussian_std"].dropna(),
    )
    assert guide_adata.var["threshold"].dropna().ge(1).all()


def test_mixture_model_overwrites_existing_var_columns(guide_adata: AnnData) -> None:
    guide_adata.var["threshold"] = pd.Categorical(["old"] * guide_adata.n_vars)
    guide_adata.var["lambda"] = "old"

    ga = rsc.ptg.GuideAssignment()
    ga.assign_mixture_model(guide_adata)

    assert pd.api.types.is_float_dtype(guide_adata.var["threshold"])
    assert pd.api.types.is_float_dtype(guide_adata.var["lambda"])
    assert guide_adata.var["threshold"].dropna().ge(1).all()
    assert np.isfinite(guide_adata.var["lambda"].dropna()).all()


def test_mixture_model_sparse_input(guide_adata_sparse: AnnData) -> None:
    ga = rsc.ptg.GuideAssignment()
    ga.assign_mixture_model(guide_adata_sparse)

    assigned = guide_adata_sparse.obs["assigned_guide"]
    n_assigned = (assigned != "negative").sum()
    assert n_assigned > 0


@pytest.mark.parametrize("layer_kind", ["dense", "csr", "csc"])
def test_mixture_model_layer_is_honored(guide_adata: AnnData, layer_kind: str) -> None:
    adata = guide_adata.copy()
    layer_counts = guide_adata.X.copy()
    if layer_kind == "csr":
        layer_counts = gpu_csr(layer_counts)
    elif layer_kind == "csc":
        layer_counts = gpu_csc(layer_counts)
    adata.layers["raw_guides"] = layer_counts
    adata.X = cp.zeros_like(guide_adata.X)

    ga = rsc.ptg.GuideAssignment()
    with pytest.warns(UserWarning, match="No guides"):
        x_result = ga.assign_mixture_model(adata, only_return_results=True)
    layer_result = ga.assign_mixture_model(
        adata, layer="raw_guides", only_return_results=True
    )

    assert (x_result == "negative").all()
    assert np.count_nonzero(layer_result != "negative") > 0


def test_mixture_model_only_return_results(guide_adata: AnnData) -> None:
    ga = rsc.ptg.GuideAssignment()
    obs_columns = guide_adata.obs.columns.copy()
    var_columns = guide_adata.var.columns.copy()
    result = ga.assign_mixture_model(guide_adata, only_return_results=True)

    assert result is not None
    assert len(result) == guide_adata.n_obs
    assert isinstance(result, np.ndarray)
    assert guide_adata.obs.columns.equals(obs_columns)
    assert guide_adata.var.columns.equals(var_columns)


@pytest.mark.parametrize(
    ("kwargs", "match"),
    [
        ({"posterior_threshold": 1.0}, "posterior_threshold"),
        ({"max_iter": 0}, "max_iter"),
        ({"tol": 0.0}, "tol"),
        ({"max_assignments_per_cell": 0}, "max_assignments_per_cell"),
    ],
)
def test_mixture_model_invalid_args_before_data_access(
    guide_adata: AnnData, kwargs: dict[str, float | int], match: str
) -> None:
    ga = rsc.ptg.GuideAssignment()
    with pytest.raises(ValueError, match=match):
        ga.assign_mixture_model(guide_adata, layer="missing_layer", **kwargs)


def test_mixture_model_cuda_assignments_are_bool(guide_adata: AnnData) -> None:
    from rapids_singlecell._cuda import _guide_assignment_cuda

    if _guide_assignment_cuda is None:
        pytest.skip("_guide_assignment_cuda extension is not available")

    assignments, *_ = _fit_assign_cuda(
        cp.ascontiguousarray(guide_adata.X.astype(cp.float32, copy=False)),
        max_iter=90,
        tol=1e-4,
        posterior_threshold=0.645,
    )

    assert assignments.dtype == cp.bool_


def test_mixture_model_cuda_c_and_f_layouts_match(guide_adata: AnnData) -> None:
    from rapids_singlecell._cuda import _guide_assignment_cuda

    if _guide_assignment_cuda is None:
        pytest.skip("_guide_assignment_cuda extension is not available")

    X_c = cp.ascontiguousarray(guide_adata.X.astype(cp.float32, copy=False))
    X_f = cp.asfortranarray(guide_adata.X.astype(cp.float32, copy=False))

    c_result = _fit_assign_cuda(
        X_c,
        max_iter=90,
        tol=1e-4,
        posterior_threshold=0.645,
    )
    f_result = _fit_assign_cuda(
        X_f,
        max_iter=90,
        tol=1e-4,
        posterior_threshold=0.645,
    )

    for c_value, f_value in zip(c_result[:-1], f_result[:-1], strict=True):
        cp.testing.assert_array_equal(c_value, f_value)
    assert c_result[-1] == f_result[-1]


def test_mixture_model_skip_low_count() -> None:
    """Guides with < 2 expressing cells should be skipped with a warning."""
    X = np.zeros((50, 3), dtype=np.float32)
    # guide 0: only 1 cell expressing
    X[0, 0] = 10.0
    # guide 1: no cells expressing
    # guide 2: good signal
    X[:20, 2] = np.random.default_rng(0).poisson(lam=50, size=20).astype(np.float32)

    adata = AnnData(
        X=cp.array(X),
        var=pd.DataFrame(index=["low", "empty", "good"]),
        obs=pd.DataFrame(index=[f"c{i}" for i in range(50)]),
    )

    ga = rsc.ptg.GuideAssignment()
    with pytest.warns(UserWarning, match="less than 2 cells"):
        ga.assign_mixture_model(adata)

    # "good" guide should have some assignments
    assigned = adata.obs["assigned_guide"]
    assert assigned.str.contains("good", na=False).any()


def test_mixture_model_skip_max_count_below_two() -> None:
    """Crispat skips guides whose non-zero counts never reach 2 UMIs."""
    X = np.zeros((50, 2), dtype=np.float32)
    X[:25, :] = 1.0

    adata = AnnData(
        X=cp.array(X),
        var=pd.DataFrame(index=["one_a", "one_b"]),
        obs=pd.DataFrame(index=[f"c{i}" for i in range(50)]),
    )

    ga = rsc.ptg.GuideAssignment()
    with pytest.warns(UserWarning, match="maximum UMI count is less than 2"):
        ga.assign_mixture_model(adata)

    assert (adata.obs["assigned_guide"] == "negative").all()


def test_multiple_guide_assignment() -> None:
    """Cells assigned to multiple guides get joined names."""
    rng = np.random.default_rng(99)
    n_cells = 100
    n_guides = 3

    # All cells get high counts for guide 0 and 1, low for guide 2
    X = np.zeros((n_cells, n_guides), dtype=np.float32)
    X[:, 0] = rng.poisson(lam=50, size=n_cells).astype(np.float32)
    X[:, 1] = rng.poisson(lam=50, size=n_cells).astype(np.float32)
    # guide 2: mix of low and high
    X[:30, 2] = rng.poisson(lam=50, size=30).astype(np.float32)
    X[30:, 2] = rng.poisson(lam=2, size=70).astype(np.float32)

    # Add some background cells
    bg_cells = rng.choice(n_cells, size=20, replace=False)
    X[bg_cells, 0] = rng.poisson(lam=2, size=20).astype(np.float32)
    X[bg_cells, 1] = rng.poisson(lam=2, size=20).astype(np.float32)

    adata = AnnData(
        X=cp.array(X),
        var=pd.DataFrame(index=["gA", "gB", "gC"]),
        obs=pd.DataFrame(index=[f"c{i}" for i in range(n_cells)]),
    )

    ga = rsc.ptg.GuideAssignment()
    ga.assign_mixture_model(adata)

    assigned = adata.obs["assigned_guide"]
    # Some cells should have multi-guide assignments (containing "+")
    has_multi = assigned.str.contains("+", na=False, regex=False)
    assert has_multi.any(), "Expected some cells with multiple guide assignments"


def test_multiple_guide_max_cap() -> None:
    """Cells exceeding max_assignments_per_cell get the multiple key."""
    rng = np.random.default_rng(7)
    n_cells = 50
    n_guides = 6

    # All cells get very high counts for all guides
    X = rng.poisson(lam=100, size=(n_cells, n_guides)).astype(np.float32)
    # Add some background cells for EM to separate
    X[:10, :] = rng.poisson(lam=2, size=(10, n_guides)).astype(np.float32)

    adata = AnnData(
        X=cp.array(X),
        var=pd.DataFrame(index=[f"g{i}" for i in range(n_guides)]),
        obs=pd.DataFrame(index=[f"c{i}" for i in range(n_cells)]),
    )

    ga = rsc.ptg.GuideAssignment()
    ga.assign_mixture_model(
        adata,
        max_assignments_per_cell=2,
        multiple_grna_assigned_key="too_many",
    )

    assigned = adata.obs["assigned_guide"]
    # Some cells should be capped
    assert (assigned == "too_many").any(), (
        "Expected some cells capped at max assignments"
    )
