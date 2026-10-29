from __future__ import annotations

import cupy as cp
import numpy as np
import pandas as pd
import pytest
from anndata import AnnData
from cupyx.scipy.sparse import csc_matrix as gpu_csc
from cupyx.scipy.sparse import csr_matrix as gpu_csr
from fast_array_utils.conv import to_dense

import rapids_singlecell as rsc
from rapids_singlecell.pertpy_gpu import _guide_assignment


def _synthetic_counts(
    rng: np.random.Generator,
    n_cells: int,
    n_guides: int,
    *,
    frac_pos: float,
    log2_mean: float,
    background: float,
) -> tuple[np.ndarray, np.ndarray]:
    X = np.zeros((n_cells, n_guides), dtype=np.float32)
    truth = np.zeros((n_cells, n_guides), dtype=bool)
    for g in range(n_guides):
        is_pos = rng.random(n_cells) < frac_pos
        X[is_pos, g] = np.clip(np.round(2 ** rng.normal(log2_mean, 0.6, is_pos.sum())), 1, None)
        X[~is_pos, g] = rng.poisson(background, (~is_pos).sum())
        truth[is_pos, g] = True
    return X, truth


def _adata(X: np.ndarray) -> AnnData:
    return AnnData(
        X=cp.asarray(X),
        obs=pd.DataFrame(index=[f"cell_{i}" for i in range(X.shape[0])]),
        var=pd.DataFrame(index=[f"guide_{i}" for i in range(X.shape[1])]),
    )


@pytest.fixture
def guide_adata() -> tuple[AnnData, np.ndarray]:
    """800 cells by 12 guides with 12 % positive cells per guide."""
    X, truth = _synthetic_counts(np.random.default_rng(0), 800, 12, frac_pos=0.12, log2_mean=5.0, background=0.8)
    return _adata(X), truth


@pytest.fixture
def tiny_adata() -> AnnData:
    return _adata(np.array([[6, 0, 2], [1, 5, 0], [0, 1, 7]], dtype=np.float32))


@pytest.mark.parametrize("storage", ["dense", "csr", "csc"])
def test_assign_by_threshold(tiny_adata: AnnData, storage: str) -> None:
    if storage != "dense":
        tiny_adata.X = (gpu_csr if storage == "csr" else gpu_csc)(tiny_adata.X)
    rsc.ptg.GuideAssignment().assign_by_threshold(tiny_adata, assignment_threshold=5)

    result = tiny_adata.layers["assigned_guides"]
    assert type(result) is type(tiny_adata.X)
    if storage == "dense":
        assert result.dtype == cp.int8
    else:
        assert result.nnz == 3
    np.testing.assert_array_equal(
        to_dense(result, to_cpu_memory=True), (to_dense(tiny_adata.X, to_cpu_memory=True) >= 5).astype(np.int8)
    )


@pytest.mark.parametrize("storage", ["dense", "csr", "csc"])
def test_assign_to_max_guide(tiny_adata: AnnData, storage: str) -> None:
    if storage != "dense":
        tiny_adata.X = (gpu_csr if storage == "csr" else gpu_csc)(tiny_adata.X)
    rsc.ptg.GuideAssignment().assign_to_max_guide(tiny_adata, assignment_threshold=6)
    assert tuple(tiny_adata.obs["assigned_guide"]) == ("guide_0", "Negative", "guide_2")


def test_mixture_model_recovers_known_positives(
    guide_adata: tuple[AnnData, np.ndarray],
) -> None:
    adata, truth = guide_adata
    rsc.ptg.GuideAssignment().assign_mixture_model(adata)

    thresholds = np.nan_to_num(adata.var["threshold"].to_numpy(), nan=np.inf)
    pred = to_dense(adata.X, to_cpu_memory=True) >= thresholds[None, :]
    assert (pred == truth).mean() > 0.98
    assert (pred & truth).sum() / truth.sum() > 0.9
    assert (pred & ~truth).sum() <= 0.005 * truth.size

    layer = adata.layers["assigned_guides"]
    assert type(layer) is type(adata.X)
    np.testing.assert_array_equal(to_dense(layer, to_cpu_memory=True), pred.astype(np.int8))
    for col in (
        "poisson_rate",
        "gaussian_mean",
        "gaussian_std",
        "mix_probs_0",
        "mix_probs_1",
        "threshold",
        "final_loss",
    ):
        assert np.isfinite(adata.var[col]).all()
    assert (adata.var["gaussian_mean"] > adata.var["poisson_rate"]).all()


def test_mixture_model_high_moi_keeps_component_order() -> None:
    """With half of the cells positive, the Gaussian must stay the on-target component."""
    X, truth = _synthetic_counts(np.random.default_rng(0), 2000, 3, frac_pos=0.5, log2_mean=6.0, background=1.0)
    adata = _adata(X)
    rsc.ptg.GuideAssignment().assign_mixture_model(adata)

    thresholds = adata.var["threshold"].to_numpy()
    assert np.all(thresholds > 1)
    assert (adata.var["gaussian_mean"] > adata.var["poisson_rate"]).all()
    assert ((X >= thresholds[None, :]) == truth).mean() > 0.99


def test_mixture_model_multiple_guides(guide_adata: tuple[AnnData, np.ndarray]) -> None:
    adata, _ = guide_adata
    rsc.ptg.GuideAssignment().assign_mixture_model(
        adata, max_assignments_per_cell=1, multiple_grna_assigned_key="too_many"
    )
    n_assigned = to_dense(adata.layers["assigned_guides"], to_cpu_memory=True).sum(axis=1)
    assigned = adata.obs["assigned_guide"].astype(str).to_numpy()
    assert (n_assigned > 1).any()
    assert (assigned[n_assigned == 0] == "negative").all()
    assert (assigned[n_assigned > 1] == "too_many").all()

    rsc.ptg.GuideAssignment().assign_mixture_model(adata, output_layer=None)
    joined = adata.obs["assigned_guide"].astype(str).to_numpy()[n_assigned == 2]
    assert all(len(label.split("+")) == 2 for label in joined)


def test_mixture_model_only_return_results(
    guide_adata: tuple[AnnData, np.ndarray],
) -> None:
    adata, _ = guide_adata
    result = rsc.ptg.GuideAssignment().assign_mixture_model(adata, only_return_results=True)

    assert isinstance(result, np.ndarray)
    assert result.shape == (adata.n_obs,)
    assert "assigned_guide" not in adata.obs
    assert "threshold" not in adata.var
    assert "assigned_guides" not in adata.layers


def test_mixture_model_layer_is_honored(
    guide_adata: tuple[AnnData, np.ndarray],
) -> None:
    adata, _ = guide_adata
    adata.layers["raw_guides"] = gpu_csc(adata.X)
    adata.X = cp.zeros_like(adata.X)

    ga = rsc.ptg.GuideAssignment()
    with pytest.warns(UserWarning, match="Skipping 12 guides"):
        x_result = ga.assign_mixture_model(adata, only_return_results=True)
    layer_result = ga.assign_mixture_model(adata, layer="raw_guides", only_return_results=True)

    assert (x_result == "negative").all()
    assert np.count_nonzero(layer_result != "negative") > 0


def test_mixture_model_skips_low_count_guides() -> None:
    X = np.zeros((200, 4), dtype=np.float32)
    X[0, 1] = 1
    X[:50, 2] = 1
    rng = np.random.default_rng(0)
    is_pos = rng.random(200) < 0.15
    X[is_pos, 3] = np.round(2 ** rng.normal(5.0, 0.6, is_pos.sum()))
    X[~is_pos, 3] = rng.poisson(0.8, (~is_pos).sum())
    adata = _adata(X)

    with pytest.warns(UserWarning, match="Skipping 3 guides .*: guide_0, guide_1, guide_2"):
        rsc.ptg.GuideAssignment().assign_mixture_model(adata)
    assert adata.var["threshold"].isna().tolist() == [True, True, True, False]


def test_mixture_model_rejects_negative_values_and_unknown_model(
    tiny_adata: AnnData,
) -> None:
    with pytest.raises(ValueError, match="Model not implemented"):
        rsc.ptg.GuideAssignment().assign_mixture_model(tiny_adata, model="not_a_model")
    tiny_adata.X[0, 0] = -1
    with pytest.raises(ValueError, match="negative values"):
        rsc.ptg.GuideAssignment().assign_mixture_model(tiny_adata)


@pytest.mark.parametrize("storage", ["dense", "csr", "csc"])
def test_mixture_model_rejects_negative_values_next_to_nan(tiny_adata: AnnData, storage: str) -> None:
    tiny_adata.X[0, 0] = -1
    tiny_adata.X[1, 1] = cp.nan
    if storage != "dense":
        tiny_adata.X = (gpu_csr if storage == "csr" else gpu_csc)(tiny_adata.X)
    with pytest.raises(ValueError, match="negative values"):
        rsc.ptg.GuideAssignment().assign_mixture_model(tiny_adata)


def _layouts(X: np.ndarray) -> dict:
    """The same counts as dense C/F, CSR, CSC, COO, int64 or mixed indices, float64 and unsorted CSR."""
    csr = gpu_csr(cp.asarray(X))
    wide = csr.copy()
    wide.indptr, wide.indices = (
        wide.indptr.astype(cp.int64),
        wide.indices.astype(cp.int64),
    )
    wide_csc = gpu_csc(cp.asarray(X, dtype=cp.float64))
    wide_csc.indptr, wide_csc.indices = (
        wide_csc.indptr.astype(cp.int64),
        wide_csc.indices.astype(cp.int64),
    )
    mixed = csr.copy()
    mixed.indptr = mixed.indptr.astype(cp.int64)
    indptr = cp.asnumpy(csr.indptr)
    reverse = np.concatenate([np.arange(start, end)[::-1] for start, end in zip(indptr[:-1], indptr[1:], strict=True)])
    unsorted = gpu_csr((csr.data[reverse], csr.indices[reverse], csr.indptr), shape=csr.shape)
    return {
        "dense C": cp.asarray(X),
        "dense F": cp.asfortranarray(cp.asarray(X)),
        "csr": csr,
        "csc": gpu_csc(cp.asarray(X)),
        "coo": csr.tocoo(),
        "int64 indices": wide,
        "int64 float64 csc": wide_csc,
        "mixed index dtypes": mixed,
        "float64": csr.astype(cp.float64),
        "unsorted csr": unsorted,
    }


def test_layouts_match(guide_adata: tuple[AnnData, np.ndarray], monkeypatch: pytest.MonkeyPatch) -> None:
    adata, _ = guide_adata
    X = cp.asnumpy(adata.X)
    ga = rsc.ptg.GuideAssignment()
    reference = adata.copy()
    ga.assign_mixture_model(reference)
    monkeypatch.setattr(_guide_assignment, "_HISTOGRAM_CAP", 8)
    monkeypatch.setattr(_guide_assignment, "_COLUMN_CHUNK", 7)
    for name, layout in _layouts(X).items():
        a = AnnData(X=layout, obs=adata.obs.copy(), var=adata.var.copy())
        ga.assign_mixture_model(a)
        if name != "coo":
            assert type(a.layers["assigned_guides"]) is type(layout)
        ga.assign_to_max_guide(a, assignment_threshold=3, obs_key="max_guide")
        ga.assign_by_threshold(a, assignment_threshold=3, output_layer="threshold")
        np.testing.assert_array_equal(a.var["threshold"], reference.var["threshold"], err_msg=name)
        np.testing.assert_array_equal(a.obs["assigned_guide"], reference.obs["assigned_guide"], err_msg=name)
        np.testing.assert_array_equal(
            to_dense(a.layers["assigned_guides"], to_cpu_memory=True),
            to_dense(reference.layers["assigned_guides"], to_cpu_memory=True),
            err_msg=name,
        )
        np.testing.assert_array_equal(to_dense(a.layers["threshold"], to_cpu_memory=True), X >= 3, err_msg=name)
        best = np.where(X.max(axis=1) >= 3, X.argmax(axis=1), -1)
        expected = np.append(adata.var_names, "Negative")[best]
        np.testing.assert_array_equal(a.obs["max_guide"], expected, err_msg=name)


@pytest.mark.parametrize("storage", ["dense", "csr", "csc"])
def test_assign_to_max_guide_nan_ties_and_empty_cells(storage: str) -> None:
    X = np.array([[np.nan, 5, 0], [2, 2, 0], [0, 0, 0], [-0.0, 0, 0]], np.float32)
    adata = _adata(X)
    if storage != "dense":
        adata.X = (gpu_csr if storage == "csr" else gpu_csc)(adata.X)
    rsc.ptg.GuideAssignment().assign_to_max_guide(adata, assignment_threshold=1)
    assert list(adata.obs["assigned_guide"]) == [
        "Negative",
        "guide_0",
        "Negative",
        "Negative",
    ]


@pytest.mark.parametrize("storage", ["dense", "csr", "csc"])
def test_assign_to_max_guide_without_guides(storage: str) -> None:
    adata = _adata(np.zeros((3, 0), np.float32))
    if storage != "dense":
        adata.X = (gpu_csr if storage == "csr" else gpu_csc)(adata.X)
    rsc.ptg.GuideAssignment().assign_to_max_guide(adata, assignment_threshold=1)
    assert list(adata.obs["assigned_guide"]) == ["Negative"] * 3


@pytest.mark.parametrize("storage", ["dense", "csr", "csc"])
def test_thresholds_compare_in_data_precision(storage: str) -> None:
    adata = _adata(np.array([[0.7, 0.2]], np.float32))
    if storage != "dense":
        adata.X = (gpu_csr if storage == "csr" else gpu_csc)(adata.X)
    rsc.ptg.GuideAssignment().assign_by_threshold(adata, assignment_threshold=0.7)
    rsc.ptg.GuideAssignment().assign_to_max_guide(adata, assignment_threshold=0.7)
    np.testing.assert_array_equal(to_dense(adata.layers["assigned_guides"], to_cpu_memory=True), [[1, 0]])
    assert list(adata.obs["assigned_guide"]) == ["guide_0"]


def test_mixture_model_rejects_too_many_starts_and_bad_shapes(
    tiny_adata: AnnData,
) -> None:
    with pytest.raises(ValueError, match="n_init_seeds"):
        rsc.ptg.GuideAssignment().assign_mixture_model(tiny_adata, n_init_seeds=70000)
    zeros = cp.zeros((2, 1), cp.float32)
    with pytest.raises(ValueError, match="shapes"):
        _guide_assignment._ga.fit_poisson_gauss(
            zeros,
            zeros,
            cp.zeros(2, cp.int32),
            cp.zeros(2, cp.int64),
            cp.zeros((3, 1, 15, 2), cp.float32),
            cp.zeros((6, 2, 4), cp.float32),
            cp.zeros((6, 2), cp.float32),
            cp.zeros((1, 6), cp.float32),
            n_iter=1,
        )


def _pertpy_labels(binary: np.ndarray, names: np.ndarray, **kwargs) -> pd.Categorical:
    """pertpy's per-cell labels of a dense binary assignment."""
    cap = kwargs.get("max_assignments_per_cell", 5)
    out = np.full(len(binary), kwargs.get("no_grna_assigned_key", "negative"), object)
    for i, row in enumerate(binary):
        guides = names[np.flatnonzero(row)]
        if 1 <= guides.size <= cap:
            out[i] = "+".join(guides)
    out[binary.sum(axis=1) > cap] = kwargs.get("multiple_grna_assigned_key", "multiple")
    return pd.Categorical(out)


@pytest.mark.parametrize(
    ("names", "kwargs"),
    [
        (None, {"no_grna_assigned_key": "none", "multiple_grna_assigned_key": "none"}),
        (None, {"multiple_grna_assigned_key": "guide_0+guide_1"}),
        (None, {"max_assignments_per_cell": 0}),
        (None, {"max_assignments_per_cell": 1}),
        (["a", "a", "b", "b", "c", "c", "d", "d", "e", "e", "f", "f"], {}),
    ],
)
def test_mixture_model_labels_match_pertpy(
    guide_adata: tuple[AnnData, np.ndarray], names: list | None, kwargs: dict
) -> None:
    adata, _ = guide_adata
    if names is not None:
        adata.var_names = names
    rsc.ptg.GuideAssignment().assign_mixture_model(adata, **kwargs)
    expected = _pertpy_labels(
        to_dense(adata.layers["assigned_guides"], to_cpu_memory=True) > 0,
        np.asarray(adata.var_names, dtype=object),
        **kwargs,
    )
    labels = adata.obs["assigned_guide"]
    assert list(labels.cat.categories) == list(expected.categories)
    np.testing.assert_array_equal(labels.astype(str), np.asarray(expected, dtype=str))
