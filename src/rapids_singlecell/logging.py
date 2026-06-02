"""Logging utilities, modelled on :mod:`scanpy.logging`."""

from __future__ import annotations

import logging
import sys
from datetime import UTC, datetime, timedelta
from functools import partial, update_wrapper
from logging import CRITICAL, DEBUG, ERROR, INFO, WARNING
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from typing import TextIO


__all__ = [
    "debug",
    "error",
    "hint",
    "info",
    "warning",
]

HINT = (INFO + DEBUG) // 2
logging.addLevelName(HINT, "HINT")


# Mapping from rsc Verbosity -> cuml log level (which is inverted: 0=trace, 6=off).
# Built lazily to avoid importing cuml at module import time.
def _cuml_level_for(verbosity_level: int) -> object:
    import cuml.internals.logger as _cuml_logger

    # stdlib levels: ERROR=40, WARNING=30, INFO=20, HINT=15, DEBUG=10.
    # Lower stdlib level == more verbose; cuml uses its own enum.
    if verbosity_level <= DEBUG:
        return _cuml_logger.level_enum.debug
    if verbosity_level <= INFO:
        return _cuml_logger.level_enum.info
    if verbosity_level <= WARNING:
        return _cuml_logger.level_enum.warn
    return _cuml_logger.level_enum.error


class _RsLogger(logging.Logger):
    """Standard-hierarchy logger with scanpy-style ``time``/``deep`` helpers.

    Unlike scanpy's isolated root logger, this lives in the normal
    :mod:`logging` hierarchy and *propagates* to the stdlib root, so rsc log
    records are visible to standard tooling (an application's logging config,
    pytest's ``caplog``) like any other library logger -- while the package's
    own :class:`_LogFormatter` handler still renders nicely formatted output.
    Verbosity is enforced on the handler, not the logger, so the logger stays
    permissive and capture tools see every record they ask for.
    """

    def log(
        self,
        level: int,
        msg: str,
        *,
        extra: dict | None = None,
        time: datetime | None = None,
        deep: str | None = None,
    ) -> datetime:
        from ._settings import settings

        now = datetime.now(UTC)
        time_passed: timedelta | None = None if time is None else now - time
        extra = {
            **(extra or {}),
            "deep": deep if settings.verbosity.level < level else None,
            "time_passed": time_passed,
        }
        super().log(level, msg, extra=extra)
        return now

    def critical(self, msg, *, time=None, deep=None, extra=None) -> datetime:
        return self.log(CRITICAL, msg, time=time, deep=deep, extra=extra)

    def error(self, msg, *, time=None, deep=None, extra=None) -> datetime:
        return self.log(ERROR, msg, time=time, deep=deep, extra=extra)

    def warning(self, msg, *, time=None, deep=None, extra=None) -> datetime:
        return self.log(WARNING, msg, time=time, deep=deep, extra=extra)

    def info(self, msg, *, time=None, deep=None, extra=None) -> datetime:
        return self.log(INFO, msg, time=time, deep=deep, extra=extra)

    def hint(self, msg, *, time=None, deep=None, extra=None) -> datetime:
        return self.log(HINT, msg, time=time, deep=deep, extra=extra)

    def debug(self, msg, *, time=None, deep=None, extra=None) -> datetime:
        return self.log(DEBUG, msg, time=time, deep=deep, extra=extra)


class _LogFormatter(logging.Formatter):
    def __init__(
        self, fmt="{levelname}: {message}", datefmt="%Y-%m-%d %H:%M", style="{"
    ):
        super().__init__(fmt, datefmt, style)

    def format(self, record: logging.LogRecord):
        format_orig = self._style._fmt
        if record.levelno == INFO:
            self._style._fmt = "{message}"
        elif record.levelno == HINT:
            self._style._fmt = "--> {message}"
        elif record.levelno == DEBUG:
            self._style._fmt = "    {message}"
        if record.time_passed is not None:
            if record.time_passed.microseconds:
                record.time_passed = timedelta(
                    seconds=int(record.time_passed.total_seconds())
                )
            if "{time_passed}" in record.msg:
                record.msg = record.msg.replace(
                    "{time_passed}", str(record.time_passed)
                )
            else:
                self._style._fmt += " ({time_passed})"
        if record.deep is not None:
            record.msg = f"{record.msg}: {record.deep}"
        result = logging.Formatter.format(self, record)
        self._style._fmt = format_orig
        return result


def _default_logfile() -> TextIO:
    import builtins

    in_ipython = getattr(builtins, "__IPYTHON__", False)
    return sys.stdout if in_ipython else sys.stderr


# The package's handler enforces verbosity; the logger stays permissive (see
# _RsLogger). This tracks the current verbosity for handlers installed before
# ``settings`` is initialised.
_HANDLER_LEVEL = WARNING


def _make_root_logger() -> _RsLogger:
    prev_cls = logging.getLoggerClass()
    logging.setLoggerClass(_RsLogger)
    try:
        logger = logging.getLogger("rapids_singlecell")
    finally:
        logging.setLoggerClass(prev_cls)
    # Permissive level: records are always created and propagate to the stdlib
    # root, so verbosity (enforced on the handler) never hides records from
    # ``caplog`` or an application's logging configuration.
    logger.setLevel(DEBUG)
    logger.propagate = True
    return logger


_root_logger = _make_root_logger()


def _install_handler(stream: TextIO) -> None:
    for handler in list(_root_logger.handlers):
        _root_logger.removeHandler(handler)
        handler.close()
    h = logging.StreamHandler(stream)
    h.setFormatter(_LogFormatter())
    h.setLevel(_HANDLER_LEVEL)
    _root_logger.addHandler(h)


def _set_log_level(level: int) -> None:
    """Apply ``level`` to the rsc handler(s) and propagate to cuml.

    The level gates the package's handler rather than the logger, so the logger
    stays permissive and standard capture tools (``caplog`` etc.) keep seeing
    every record while verbosity still controls what is written to the stream.
    """
    global _HANDLER_LEVEL
    _HANDLER_LEVEL = level
    for h in list(_root_logger.handlers):
        h.setLevel(level)
    try:
        import cuml.internals.logger as _cuml_logger

        _cuml_logger.set_level(_cuml_level_for(level))
    except ImportError:
        pass


_install_handler(_default_logfile())


def _copy_docs_and_signature(fn):
    return partial(update_wrapper, wrapped=fn, assigned=["__doc__", "__annotations__"])


def error(
    msg: str,
    *,
    time: datetime | None = None,
    deep: str | None = None,
    extra: dict | None = None,
) -> datetime:
    """Log message with specific level and return current time.

    Parameters
    ----------
    msg
        Message to display.
    time
        A time in the past. If passed, the difference from then to now is
        appended to ``msg`` as ``(HH:MM:SS)``. If ``msg`` contains
        ``{time_passed}``, the time difference is inserted at that position.
    deep
        If the current verbosity is higher than the log function's level,
        this gets displayed as well.
    extra
        Additional values you can specify in ``msg`` like ``{time_passed}``.

    """
    return _root_logger.error(msg, time=time, deep=deep, extra=extra)


@_copy_docs_and_signature(error)
def warning(msg, *, time=None, deep=None, extra=None) -> datetime:
    return _root_logger.warning(msg, time=time, deep=deep, extra=extra)


@_copy_docs_and_signature(error)
def info(msg, *, time=None, deep=None, extra=None) -> datetime:
    return _root_logger.info(msg, time=time, deep=deep, extra=extra)


@_copy_docs_and_signature(error)
def hint(msg, *, time=None, deep=None, extra=None) -> datetime:
    return _root_logger.hint(msg, time=time, deep=deep, extra=extra)


@_copy_docs_and_signature(error)
def debug(msg, *, time=None, deep=None, extra=None) -> datetime:
    return _root_logger.debug(msg, time=time, deep=deep, extra=extra)
