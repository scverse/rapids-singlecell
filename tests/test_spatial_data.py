from __future__ import annotations

import subprocess
import sys

import numpy as np
import pandas as pd
import pytest
from anndata import AnnData
from scipy import sparse

import rapids_singlecell as rsc
from rapids_singlecell.squidpy_gpu._spatial_data import (
    _extract_adata,
    _resolve_spatial_data,
)

sd = pytest.importorskip("spatialdata")
from spatialdata.models import (  # noqa: E402
    Labels2DModel,
    PointsModel,
    ShapesModel,
    TableModel,
)
from spatialdata.transformations import Scale  # noqa: E402


@pytest.fixture
def sdata():
    rng = np.random.default_rng(18)
    points = {}
    for region in ("a", "b"):
        points[region] = PointsModel.parse(
            pd.DataFrame(rng.uniform(0, 5, (12, 2)), columns=["x", "y"]),
            transformations={"scaled": Scale([2, 3], axes=("x", "y"))},
        )
    table = AnnData(
        rng.uniform(1, 5, (24, 4)).astype(np.float32),
        obs=pd.DataFrame(
            {
                "region": pd.Categorical(np.repeat(["a", "b"], 12)),
                "instance": np.tile(np.arange(12), 2),
                "cluster": pd.Categorical(np.tile(["c1", "c2"], 12)),
            },
            index=[f"cell{i}" for i in range(24)],
        ),
    )
    table = TableModel.parse(
        table, region=["a", "b"], region_key="region", instance_key="instance"
    )
    return sd.SpatialData(points=points, tables={"cells": table, "other": table.copy()})


def _resolve(sdata, **kwargs):
    return _resolve_spatial_data(
        sdata,
        table_key="cells",
        elements_to_coordinate_systems={"a": "scaled", "b": "scaled"},
        spatial_key="coords",
        library_key=None,
        **kwargs,
    )


@pytest.mark.parametrize("shuffle", [False, True])
def test_centroids_follow_table_order(sdata, shuffle):
    if shuffle:
        sdata.tables["cells"] = sdata.tables["cells"][
            np.random.default_rng(0).permutation(24)
        ].copy()
    table, library = _resolve(sdata)
    assert library == "region"
    for i, row in enumerate(table.obs.itertuples()):
        point = sdata.points[row.region].compute().loc[row.instance, ["x", "y"]]
        np.testing.assert_allclose(table.obsm["coords"][i], point * [2, 3])
    assert "coords" not in sdata.tables["other"].obsm


@pytest.mark.parametrize("kind", ["labels", "shapes"])
def test_element_centroids(kind):
    if kind == "labels":
        element = Labels2DModel.parse(np.array([[0, 1, 1], [2, 2, 0]]))
        expected = [[1, 1.5], [2, 0.5]]
    else:
        import geopandas as gpd
        from shapely.geometry import Point

        element = ShapesModel.parse(
            gpd.GeoDataFrame(
                {"geometry": [Point(1, 2), Point(3, 4)], "radius": [1.0, 1.0]},
                index=[1, 2],
            )
        )
        expected = [[3, 4], [1, 2]]
    table = AnnData(
        np.zeros((2, 1)),
        obs=pd.DataFrame(
            {"region": pd.Categorical(["cells", "cells"]), "instance": [2, 1]},
            index=["a", "b"],
        ),
    )
    table = TableModel.parse(
        table, region="cells", region_key="region", instance_key="instance"
    )
    data = sd.SpatialData(**{kind: {"cells": element}}, tables={"table": table})
    result, _ = _resolve_spatial_data(
        data,
        table_key="table",
        elements_to_coordinate_systems={"cells": "global"},
        spatial_key="spatial",
        library_key=None,
    )
    np.testing.assert_allclose(result.obsm["spatial"], expected)


_MODES = [
    ("spatial_neighbors_knn", {"n_neighs": 3}),
    ("spatial_neighbors_radius", {"radius": 4}),
    ("spatial_neighbors_delaunay", {}),
    ("spatial_neighbors_grid", {"n_neighs": 3, "n_rings": 2}),
    ("spatial_neighbors_from_builder", {"builder": rsc.gr.neighbors.KNNBuilder(3)}),
]


@pytest.mark.parametrize("name,kwargs", _MODES)
@pytest.mark.parametrize("copy", [False, True])
def test_neighbors_spatialdata(sdata, name, kwargs, copy):
    table, _ = _resolve(sdata)
    reference = table.copy()
    del table.obsm["coords"]
    func = getattr(rsc.gr, name)
    expected = func(
        reference, spatial_key="coords", library_key="region", copy=True, **kwargs
    )
    actual = func(
        sdata,
        table_key="cells",
        elements_to_coordinate_systems={"a": "scaled", "b": "scaled"},
        spatial_key="coords",
        library_key="ignored_for_spatialdata",
        key_added="test",
        copy=copy,
        **kwargs,
    )
    if not copy:
        assert actual is None
        actual = (table.obsp["test_connectivities"], table.obsp["test_distances"])
        assert (
            table.uns["test_neighbors"]["connectivities_key"] == "test_connectivities"
        )
    else:
        assert not table.obsp
        assert "test_neighbors" not in table.uns
    for observed, wanted in zip(actual, expected, strict=True):
        np.testing.assert_allclose(observed.toarray(), wanted.toarray())
        assert observed[:12, 12:].nnz == observed[12:, :12].nnz == 0
    assert not sdata.tables["other"].obsp


_DOWNSTREAM = [
    ("spatial_autocorr", {"n_perms": None, "multi_gpu": False}),
    (
        "co_occurrence",
        {
            "cluster_key": "cluster",
            "interval": np.array([0, 2, 6, 20]),
            "multi_gpu": False,
        },
    ),
    ("ripley", {"cluster_key": "cluster", "mode": "G", "n_neigh": 1, "rng": 0}),
    ("interaction_matrix", {"cluster_key": "cluster", "weights": True}),
    ("nhood_enrichment", {"cluster_key": "cluster", "n_perms": 20, "seed": 0}),
    (
        "ligrec",
        {
            "cluster_key": "cluster",
            "interactions": [("0", "1")],
            "use_raw": False,
            "n_perms": 4,
        },
    ),
    (
        "calculate_niche_neighborhood",
        {"groups": "cluster", "resolutions": 0.5, "n_neighbors": 3},
    ),
    ("calculate_niche_utag", {"resolutions": 0.5, "n_neighbors": 3}),
    ("calculate_niche_cellcharter", {"n_components": 2, "distance": 1}),
    (
        "calculate_niche",
        {
            "flavor": "neighborhood",
            "groups": "cluster",
            "resolutions": 0.5,
            "n_neighbors": 3,
        },
    ),
]


@pytest.mark.parametrize("name,kwargs", _DOWNSTREAM)
@pytest.mark.parametrize("copy", [False, True])
def test_downstream_spatialdata(sdata, name, kwargs, copy):
    table, _ = _resolve(sdata)
    table.obsm["spatial"] = table.obsm["coords"].copy()
    table.obsp["spatial_connectivities"] = sparse.diags(
        [np.ones(23), np.ones(23)], [-1, 1], shape=(24, 24), format="csr"
    )
    reference = table.copy()
    func = getattr(rsc.gr, name)
    niche = name.startswith("calculate_niche")
    options = {**kwargs, **({"inplace": not copy} if niche else {"copy": copy})}
    actual = func(sdata, table_key="cells", **options)
    expected = func(reference, **options)
    if niche:
        a, b = (actual, expected) if copy else (table, reference)
        cols = [c for c in b.obs if "niche" in c]
        assert cols
        pd.testing.assert_frame_equal(a.obs[cols], b.obs[cols])
        if copy:
            assert not any("niche" in c for c in table.obs)
    elif name == "spatial_autocorr":
        pd.testing.assert_frame_equal(
            actual if copy else table.uns["moranI"],
            expected if copy else reference.uns["moranI"],
        )
    elif name == "ripley":
        a = actual if copy else table.uns["cluster_ripley_G"]
        b = expected if copy else reference.uns["cluster_ripley_G"]
        pd.testing.assert_frame_equal(a["G_stat"], b["G_stat"])
        pd.testing.assert_frame_equal(a["sims_stat"], b["sims_stat"])
    elif name == "co_occurrence":
        a = actual if copy else table.uns["cluster_co_occurrence"].values()
        b = expected if copy else reference.uns["cluster_co_occurrence"].values()
        for x, y in zip(a, b, strict=True):
            np.testing.assert_allclose(x, y)
    elif name == "interaction_matrix":
        a = actual if copy else table.uns["cluster_interactions"]
        b = expected if copy else reference.uns["cluster_interactions"]
        np.testing.assert_array_equal(a, b)
    elif name == "nhood_enrichment":
        a = actual if copy else table.uns["cluster_nhood_enrichment"].values()
        b = expected if copy else reference.uns["cluster_nhood_enrichment"].values()
        for x, y in zip(a, b, strict=True):
            np.testing.assert_array_equal(x, y)
    else:
        a = actual if copy else table.uns["cluster_ligrec"]
        b = expected if copy else reference.uns["cluster_ligrec"]
        # Permutations are random; means and structure must agree.
        pd.testing.assert_frame_equal(a["means"], b["means"])
        assert a["pvalues"].shape == b["pvalues"].shape
    assert not any("niche" in c for c in sdata.tables["other"].obs)
    assert set(sdata.tables["other"].uns) == {"spatialdata_attrs"}


@pytest.mark.parametrize("name,kwargs", _MODES + _DOWNSTREAM)
def test_table_required_everywhere(sdata, name, kwargs):
    with pytest.raises(TypeError, match="table_key"):
        getattr(rsc.gr, name)(sdata, **kwargs)
    with pytest.raises(ValueError, match="not found"):
        getattr(rsc.gr, name)(sdata, table_key="missing", **kwargs)


def test_invalid_spatial_mapping(sdata):
    for mapping, message in [
        (None, "requires"),
        ({"a": "scaled"}, "Missing coordinate"),
    ]:
        with pytest.raises(ValueError, match=message):
            _resolve_spatial_data(
                sdata,
                table_key="cells",
                elements_to_coordinate_systems=mapping,
                spatial_key="spatial",
                library_key=None,
            )
    sdata.tables["cells"].obs.loc["cell0", "instance"] = 99
    with pytest.raises(ValueError, match="instances are missing"):
        _resolve(sdata)
    assert "coords" not in sdata.tables["cells"].obsm


def test_anndata_needs_no_spatialdata():
    # A fresh interpreter catches eager imports anywhere in the package.
    subprocess.run(
        [
            sys.executable,
            "-c",
            """
import sys
class NoSpatialData:
    def find_spec(self, fullname, path=None, target=None):
        if fullname.split('.')[0] in ('spatialdata', 'squidpy'):
            raise AssertionError(f'Unexpected import: {fullname}')
sys.meta_path.insert(0, NoSpatialData())
import rapids_singlecell as rsc
import numpy as np
from anndata import AnnData
from rapids_singlecell.squidpy_gpu._spatial_data import _extract_adata, _resolve_spatial_data
adata = AnnData()
assert _extract_adata(adata) is adata
assert _resolve_spatial_data(adata, table_key=None, elements_to_coordinate_systems=None, spatial_key='spatial', library_key=None)[0] is adata
adata = AnnData(np.zeros((3, 1)))
adata.obsm['spatial'] = np.array([[0., 0.], [1., 0.], [3., 0.]])
result = rsc.gr.spatial_neighbors_knn(adata, n_neighs=1, copy=True)
assert result.connectivities.nnz == 3

""",
        ],
        check=True,
    )


def test_extract_table_identity(sdata):
    assert _extract_adata(sdata, table_key="cells") is sdata.tables["cells"]


def test_invalid_input_without_spatialdata(monkeypatch):
    # Invalid AnnData calls must still raise TypeError when the optional package is absent.
    import builtins

    original = builtins.__import__

    def without_spatialdata(name, *args, **kwargs):
        if name == "spatialdata":
            raise ModuleNotFoundError(
                "No module named 'spatialdata'", name="spatialdata"
            )
        return original(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", without_spatialdata)
    with pytest.raises(TypeError, match="Expected `adata`"):
        _extract_adata(None)
