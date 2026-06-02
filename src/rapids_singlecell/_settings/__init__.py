"""scverse-style settings for rapids_singlecell.

Built on :class:`scverse_misc.Settings` (a Pydantic ``BaseSettings``).
Setting :attr:`settings.verbosity` updates the rsc root logger and propagates
to :mod:`cuml.internals.logger` so both stay in sync.

Environment variables are read with the ``RAPIDS_SINGLECELL_`` prefix
(e.g. ``RAPIDS_SINGLECELL_VERBOSITY=debug``).
"""

from __future__ import annotations

from typing import Annotated

from pydantic import BeforeValidator, model_validator
from pydantic_settings import SettingsConfigDict
from scverse_misc import Settings

from .verbosity import Verbosity

__all__ = ["Verbosity", "settings"]


def _coerce_verbosity(value: object) -> Verbosity | object:
    if isinstance(value, str):
        try:
            return Verbosity[value.lower()]
        except KeyError as e:
            valid = ", ".join(Verbosity.__members__)
            msg = f"Cannot set verbosity to {value!r}. Valid names are: {valid}"
            raise ValueError(msg) from e
    return value


class _RscSettings(
    Settings,
    exported_object_name="settings",
    docstring_style="numpy",
):
    model_config = SettingsConfigDict(extra="ignore")

    verbosity: Annotated[Verbosity, BeforeValidator(_coerce_verbosity)] = (
        Verbosity.warning
    )
    """Logging verbosity. Accepts a :class:`Verbosity`, its int value, or its name."""

    @model_validator(mode="after")
    def _apply_logging_level(self):
        from rapids_singlecell.logging import _set_log_level

        _set_log_level(self.verbosity.level)
        return self


settings = _RscSettings()
