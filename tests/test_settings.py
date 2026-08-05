from __future__ import annotations

import importlib
import inspect

import cupy as cp
import numpy as np
import pandas as pd
import pytest
from anndata import AnnData
from cupyx.scipy import sparse as cp_sparse
from scanpy.datasets import pbmc68k_reduced
from scipy import sparse

import rapids_singlecell as rsc


@pytest.mark.parametrize(
    ("preset", "expected"),
    [
        (
            rsc.Preset.ScanpyV1,
            {
                "hvg_flavor": "seurat",
                "hvg_return_df": False,
                "pca_key": None,
                "umap_key": None,
                "tsne_key": None,
                "diffmap_key": None,
                "draw_graph_key": None,
                "rank_method": "t-test",
                "mean_in_log_space": True,
                "rank_mask_var": None,
                "zero_center": True,
                "ctrl_as_ref": True,
            },
        ),
        (
            rsc.Preset.ScanpyV2Preview,
            {
                "hvg_flavor": "seurat_v3_paper",
                "hvg_return_df": True,
                "pca_key": "pca",
                "umap_key": "umap",
                "tsne_key": "tsne",
                "diffmap_key": "diffmap",
                "draw_graph_key": "graph_{layout}",
                "rank_method": "wilcoxon",
                "mean_in_log_space": False,
                "rank_mask_var": None,
                "zero_center": None,
                "ctrl_as_ref": False,
            },
        ),
    ],
)
def test_scanpy_preset_values(preset, expected):
    assert preset.highly_variable_genes.flavor == expected["hvg_flavor"]
    assert preset.highly_variable_genes.return_df is expected["hvg_return_df"]
    assert preset.pca.key_added == expected["pca_key"]
    assert preset.umap.key_added == expected["umap_key"]
    assert preset.tsne.key_added == expected["tsne_key"]
    assert preset.diffmap.key_added == expected["diffmap_key"]
    assert preset.draw_graph.key_added == expected["draw_graph_key"]
    assert preset.rank_genes_groups.method == expected["rank_method"]
    assert preset.rank_genes_groups.mean_in_log_space is expected["mean_in_log_space"]
    assert preset.rank_genes_groups.mask_var is expected["rank_mask_var"]
    assert preset.scale.zero_center is expected["zero_center"]
    assert preset.score_genes.ctrl_as_ref is expected["ctrl_as_ref"]


def test_settings_validation_override_and_reset():
    original = rsc.settings.preset
    try:
        rsc.settings.preset = "scanpy-v2-preview"
        assert rsc.settings.preset is rsc.Preset.ScanpyV2Preview

        with rsc.settings.override(preset="scanpy-v1"):
            assert rsc.settings.preset is rsc.Preset.ScanpyV1
        assert rsc.settings.preset is rsc.Preset.ScanpyV2Preview

        rsc.settings.preset = rsc.Preset.ScanpyV1
        with rsc.settings.preset.override(rsc.Preset.ScanpyV2Preview) as previous:
            assert previous is rsc.Preset.ScanpyV1
            assert rsc.settings.preset is rsc.Preset.ScanpyV2Preview
        assert rsc.settings.preset is rsc.Preset.ScanpyV1

        with pytest.raises(ValueError):
            rsc.settings.preset = "invalid-preset"

        rsc.settings.preset = rsc.Preset.ScanpyV2Preview
        with rsc.settings.reset("preset") as reset:
            assert reset == {"preset"}
            assert rsc.settings.preset is rsc.Preset.ScanpyV1
        assert rsc.settings.preset is rsc.Preset.ScanpyV2Preview

        rsc.settings.reset("preset")
        assert rsc.settings.preset is rsc.Preset.ScanpyV1
    finally:
        rsc.settings.preset = original


@pytest.mark.parametrize(
    ("preset", "key_obsm", "key_varm", "key_uns"),
    [
        (rsc.Preset.ScanpyV1, "X_pca", "PCs", "pca"),
        (rsc.Preset.ScanpyV2Preview, "pca", "pca", "pca"),
    ],
)
def test_pca_preset_storage_keys(preset, key_obsm, key_varm, key_uns):
    adata = AnnData(
        np.array(
            [
                [1, 0, 2, 1],
                [2, 1, 0, 1],
                [0, 2, 1, 3],
                [3, 1, 2, 0],
                [1, 3, 0, 2],
            ],
            dtype=np.float32,
        )
    )

    with rsc.settings.override(preset=preset):
        rsc.pp.pca(adata, n_comps=2)

    assert key_obsm in adata.obsm
    assert key_varm in adata.varm
    assert key_uns in adata.uns


@pytest.mark.parametrize(
    ("preset", "expected_mask_var"),
    [
        (rsc.Preset.ScanpyV1, "highly_variable"),
        (rsc.Preset.ScanpyV2Preview, "var.highly_variable"),
    ],
)
def test_pca_preset_default_mask_metadata(preset, expected_mask_var):
    adata = AnnData(np.arange(30, dtype=np.float32).reshape(6, 5))
    adata.var["highly_variable"] = [True, True, True, False, False]

    with rsc.settings.override(preset=preset):
        rsc.pp.pca(adata, n_comps=2)

    assert adata.uns["pca"]["params"]["mask_var"] == expected_mask_var


def test_n_pcs_setting_controls_pca_and_representation_selection():
    adata = AnnData(np.arange(80, dtype=np.float32).reshape(10, 8))

    with rsc.settings.override(N_PCS=3):
        rsc.pp.pca(adata)
        representation = rsc.pp.neighbors(adata, n_neighbors=3, copy=True)

    assert adata.obsm["X_pca"].shape == (10, 3)
    assert representation.uns["neighbors"]["params"].get("n_pcs") is None


def test_scanpy_v2_automatic_representation_uses_preview_pca_key():
    adata = AnnData(np.arange(600, dtype=np.float32).reshape(60, 10))

    with rsc.settings.override(preset=rsc.Preset.ScanpyV2Preview, N_PCS=3):
        rsc.pp.pca(adata)
        rsc.pp.neighbors(adata)

    assert "pca" in adata.obsm
    assert "neighbors" in adata.uns


def test_scanpy_v2_aggregate_legacy_selection_warning():
    adata = AnnData(
        np.arange(24, dtype=np.float32).reshape(6, 4),
        obs=pd.DataFrame(
            {"group": pd.Categorical(["a", "a", "a", "b", "b", "b"])},
            index=[f"cell_{i}" for i in range(6)],
        ),
    )
    adata.layers["counts"] = adata.X.copy()

    with rsc.settings.override(preset=rsc.Preset.ScanpyV2Preview):
        with pytest.warns(FutureWarning, match="`acc` will replace"):
            rsc.get.aggregate(adata, by="group", func="sum", layer="counts")


@pytest.mark.parametrize(
    ("embedding", "key"),
    [
        (rsc.tl.umap, "umap"),
        (rsc.tl.tsne, "tsne"),
        (rsc.tl.diffmap, "diffmap"),
    ],
)
def test_scanpy_v2_embedding_storage_keys(embedding, key):
    adata = pbmc68k_reduced()[:100, :100].copy()

    with rsc.settings.override(preset=rsc.Preset.ScanpyV2Preview):
        embedding(adata)

    assert key in adata.obsm
    assert key in adata.uns


def test_scanpy_v2_draw_graph_storage_keys():
    adata = pbmc68k_reduced()[:100, :100].copy()

    with rsc.settings.override(preset=rsc.Preset.ScanpyV2Preview):
        rsc.tl.draw_graph(adata)

    assert "graph_fa" in adata.obsm
    assert "graph_fa" in adata.uns


def test_scanpy_v2_hvg_default_reaches_implementation(monkeypatch):
    hvg_module = importlib.import_module("rapids_singlecell.preprocessing._hvg")
    called = {}

    def record_call(**kwargs):
        called.update(kwargs)

    monkeypatch.setattr(hvg_module, "_highly_variable_genes_seurat_v3", record_call)
    with rsc.settings.override(preset=rsc.Preset.ScanpyV2Preview):
        rsc.pp.highly_variable_genes(
            AnnData(np.ones((3, 2), dtype=np.float32)), n_top_genes=1
        )

    assert called["flavor"] == "seurat_v3_paper"


def test_scanpy_v2_rank_genes_groups_defaults():
    group = pd.Categorical(["a"] * 30 + ["b"] * 30)
    counts = np.vstack(
        [
            np.tile([4, 0, 2, 1], (30, 1)),
            np.tile([0, 4, 1, 2], (30, 1)),
        ]
    ).astype(np.float32)
    obs = pd.DataFrame({"group": group}, index=[f"cell_{i}" for i in range(len(group))])
    adata = AnnData(np.log1p(counts), obs=obs)

    with rsc.settings.override(preset=rsc.Preset.ScanpyV2Preview):
        rsc.tl.rank_genes_groups(adata, "group", method=None)

    params = adata.uns["rank_genes_groups"]["params"]
    assert params["method"] == "wilcoxon"
    assert params["mean_in_log_space"] is False


@pytest.mark.parametrize(
    ("preset", "expected_logfc"),
    [
        (rsc.Preset.ScanpyV1, np.log2((np.sqrt(5) - 1) / 8)),
        (rsc.Preset.ScanpyV2Preview, -2.0),
    ],
)
@pytest.mark.parametrize("method", ["t-test", "wilcoxon"])
@pytest.mark.parametrize("array_type", ["numpy", "scipy_csr", "cupy", "cupy_csr"])
def test_rank_genes_groups_mean_in_log_space_default(
    preset, expected_logfc, method, array_type
):
    n_genes = 5
    n_group = 30
    group_a = np.zeros((n_group, n_genes))
    group_a[n_group // 2 :] = np.log(5)
    group_b = np.full((n_group, n_genes), np.log(9))
    X = np.concatenate([group_a, group_b])
    X = {
        "numpy": lambda: X,
        "scipy_csr": lambda: sparse.csr_matrix(X),
        "cupy": lambda: cp.asarray(X),
        "cupy_csr": lambda: cp_sparse.csr_matrix(cp.asarray(X)),
    }[array_type]()
    group = pd.Categorical(["a"] * n_group + ["b"] * n_group)
    obs = pd.DataFrame(
        {"group": group}, index=[f"cell_{i}" for i in range(2 * n_group)]
    )
    adata = AnnData(X=X, obs=obs)

    with rsc.settings.override(preset=preset):
        rsc.tl.rank_genes_groups(
            adata,
            groupby="group",
            groups=["a"],
            reference="b",
            method=method,
        )

    logfoldchanges = adata.uns["rank_genes_groups"]["logfoldchanges"]["a"]
    np.testing.assert_allclose(logfoldchanges, expected_logfc)


def test_scanpy_v2_scale_requires_zero_center():
    adata = AnnData(np.ones((3, 2), dtype=np.float32))

    with (
        rsc.settings.override(preset=rsc.Preset.ScanpyV2Preview),
        pytest.raises(TypeError, match="zero_center"),
    ):
        rsc.pp.scale(adata)


def test_scverse_misc_deprecation_metadata():
    assert rsc.pp.pca.__scverse_misc_deprecated_arg__[0].arg == "use_highly_variable"
    assert "rank_genes_groups_logreg" in rsc.tl.rank_genes_groups_logreg.__deprecated__


def test_changing_default_repr_mentions_scanpy_2():
    default = inspect.signature(rsc.pp.pca).parameters["key_added"].default

    with rsc.settings.override(preset=rsc.Preset.ScanpyV1):
        assert "changes in 2.0" in repr(default)
    with rsc.settings.override(preset=rsc.Preset.ScanpyV2Preview):
        assert "changes in 2.0" not in repr(default)
