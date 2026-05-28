from __future__ import annotations

from rapids_singlecell import logging as _rsc_logging


def _log(message: str, level: str = "info", *, verbose: bool = False) -> None:
    """Log a message via rapids_singlecell's root logger.

    Parameters
    ----------
    message
        The message to log.
    level
        The logging level (``"info"`` or ``"warn"``).
    verbose
        Whether to emit the log.
    """
    if not verbose:
        return
    if level.lower() == "warn":
        _rsc_logging.warning(message)
    else:
        _rsc_logging.info(message)
