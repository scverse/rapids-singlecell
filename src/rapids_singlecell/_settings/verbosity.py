from __future__ import annotations

from contextlib import contextmanager
from enum import IntEnum
from logging import getLevelNamesMapping
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Generator
    from typing import Literal

    type _VerbosityName = Literal["error", "warning", "info", "hint", "debug"]


_VERBOSITY_TO_LOGLEVEL: dict[str, str] = {
    "error": "ERROR",
    "warning": "WARNING",
    "info": "INFO",
    "hint": "HINT",
    "debug": "DEBUG",
}


class Verbosity(IntEnum):
    """Logging verbosity levels for :attr:`rapids_singlecell.settings.verbosity`."""

    error = 0
    """Error (`0`)"""
    warning = 1
    """Warning (`1`)"""
    info = 2
    """Info (`2`)"""
    hint = 3
    """Hint (`3`)"""
    debug = 4
    """Debug (`4`)"""

    @property
    def level(self) -> int:
        """The :ref:`logging level <levels>` corresponding to this verbosity level."""
        return getLevelNamesMapping()[_VERBOSITY_TO_LOGLEVEL[self.name]]

    @contextmanager
    def override(
        self, verbosity: Verbosity | _VerbosityName | int
    ) -> Generator[Verbosity, None, None]:
        """Temporarily override verbosity.

        >>> import rapids_singlecell as rsc
        >>> rsc.settings.verbosity = rsc.Verbosity.info
        >>> with rsc.settings.verbosity.override(rsc.Verbosity.debug):
        ...     rsc.settings.verbosity
        <Verbosity.debug: 4>
        >>> rsc.settings.verbosity
        <Verbosity.info: 2>
        """
        from . import settings

        settings.verbosity = verbosity
        try:
            yield self
        finally:
            settings.verbosity = self
