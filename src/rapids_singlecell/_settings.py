from __future__ import annotations

import enum
from contextlib import contextmanager
from dataclasses import dataclass
from typing import TYPE_CHECKING, Literal, NamedTuple, cast

from scverse_misc import Settings as BaseSettings

if TYPE_CHECKING:
    from collections.abc import Generator


type HVGFlavor = Literal[
    "seurat",
    "cell_ranger",
    "seurat_v3",
    "seurat_v3_paper",
    "pearson_residuals",
    "poisson_gene_selection",
]
type DETest = Literal[
    "logreg", "t-test", "t-test_overestim_var", "wilcoxon", "wilcoxon_binned"
]


class HVGPreset(NamedTuple):
    flavor: HVGFlavor
    return_df: bool


class BasicEmbeddingPreset(NamedTuple):
    key_added: str | None


class RankGenesGroupsPreset(NamedTuple):
    method: DETest
    mask_var: str | None
    mean_in_log_space: bool


class ScalePreset(NamedTuple):
    zero_center: bool | None


class ScoreGenesPreset(NamedTuple):
    ctrl_as_ref: bool


class Preset(enum.StrEnum):
    """Presets for :attr:`rapids_singlecell.settings.preset`.

    See properties below for details.
    """

    ScanpyV1 = "scanpy-v1"
    """Scanpy 1.*’s default settings."""

    ScanpyV2Preview = "scanpy-v2-preview"
    """Scanpy 2.*’s feature default settings. (Preview: subject to change!)"""

    @property
    def highly_variable_genes(self) -> HVGPreset:
        return {
            Preset.ScanpyV1: HVGPreset(flavor="seurat", return_df=False),
            Preset.ScanpyV2Preview: HVGPreset(flavor="seurat_v3_paper", return_df=True),
        }[self]

    @property
    def pca(self) -> BasicEmbeddingPreset:
        return self._embedding("pca")

    @property
    def umap(self) -> BasicEmbeddingPreset:
        return self._embedding("umap")

    @property
    def tsne(self) -> BasicEmbeddingPreset:
        return self._embedding("tsne")

    @property
    def diffmap(self) -> BasicEmbeddingPreset:
        return self._embedding("diffmap")

    @property
    def draw_graph(self) -> BasicEmbeddingPreset:
        return BasicEmbeddingPreset(
            key_added=None if self is Preset.ScanpyV1 else "graph_{layout}"
        )

    @property
    def rank_genes_groups(self) -> RankGenesGroupsPreset:
        return {
            Preset.ScanpyV1: RankGenesGroupsPreset(
                method="t-test", mask_var=None, mean_in_log_space=True
            ),
            Preset.ScanpyV2Preview: RankGenesGroupsPreset(
                method="wilcoxon", mask_var=None, mean_in_log_space=False
            ),
        }[self]

    @property
    def scale(self) -> ScalePreset:
        return ScalePreset(zero_center=True if self is Preset.ScanpyV1 else None)

    @property
    def score_genes(self) -> ScoreGenesPreset:
        return ScoreGenesPreset(ctrl_as_ref=self is Preset.ScanpyV1)

    def _embedding(self, name: str) -> BasicEmbeddingPreset:
        return BasicEmbeddingPreset(key_added=None if self is Preset.ScanpyV1 else name)

    @contextmanager
    def override(self, preset: Preset) -> Generator[Preset, None, None]:
        """Temporarily override :attr:`rapids_singlecell.settings.preset`."""
        with settings.override(preset=preset):
            yield self


class Settings(BaseSettings):
    """Validated global settings for rapids-singlecell."""

    preset: Preset = Preset.ScanpyV1
    """Preset to use."""

    N_PCS: int = 50
    """Default number of principal components to use."""


settings = Settings()


@dataclass(frozen=True)
class Default:
    """Marker for a function default resolved from :data:`settings`."""

    preset: tuple[str, str] | None = None
    repr: str | None = None

    def __post_init__(self) -> None:
        if self.preset is not None and self.repr is not None:
            raise TypeError("Cannot provide both preset and repr.")

    def resolve(self) -> object:
        if self.preset is None:
            raise TypeError("A default without a preset cannot be resolved.")
        return self._get_value(settings.preset)

    def _get_value(self, preset: Preset) -> object:
        if self.preset is None:
            raise TypeError("A default without a preset has no preset value.")
        section, field = self.preset
        return getattr(getattr(preset, section), field)

    def __repr__(self) -> str:
        if self.preset is None:
            return self.repr or "default"
        value = self.resolve()
        suffix = (
            " – changes in 2.0"
            if settings.preset is Preset.ScanpyV1
            and value != self._get_value(Preset.ScanpyV2Preview)
            else ""
        )
        return f"{value!r} (settings.preset={str(settings.preset)!r}{suffix})"


def resolve_default[T](value: T | Default) -> T:
    """Resolve a :class:`Default` marker against the active preset."""
    return cast("T", value.resolve()) if isinstance(value, Default) else value


__all__ = ["Preset", "settings"]
