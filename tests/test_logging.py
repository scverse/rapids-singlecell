from __future__ import annotations

import sys
from datetime import datetime
from logging import StreamHandler
from typing import TYPE_CHECKING

import pytest

import rapids_singlecell as rsc
from rapids_singlecell import Verbosity
from rapids_singlecell import logging as log
from rapids_singlecell import settings as s

if TYPE_CHECKING:
    from collections.abc import Generator


@pytest.fixture(autouse=True)
def _reset_verbosity() -> Generator[None, None, None]:
    """Restore default verbosity after each test."""
    original = s.verbosity
    yield
    s.verbosity = original


@pytest.fixture(autouse=True)
def _restore_handler() -> Generator[None, None, None]:
    """Reinstate a fresh handler on the real stream after each test.

    Several tests point the handler at a captured stream (``capsys``) that is
    closed on teardown; reinstalling here avoids leaving a closed-stream
    handler that would raise ``I/O operation on closed file`` for later logs.
    """
    yield
    log._install_handler(log._default_logfile())


def test_defaults() -> None:
    """Default verbosity is warning and a StreamHandler enforces it.

    The logger itself stays permissive so records always propagate to the
    stdlib root (and are visible to ``caplog``/logging config); verbosity is
    enforced on the handler.
    """
    assert Verbosity.warning == 1

    [handler] = log._root_logger.handlers
    assert isinstance(handler, StreamHandler)
    assert handler.level == s.verbosity.level


def test_records(caplog: pytest.LogCaptureFixture) -> None:
    s.verbosity = Verbosity.debug
    log.error("0")
    log.warning("1")
    log.info("2")
    log.hint("3")
    log.debug("4")
    assert caplog.record_tuples == [
        ("rapids_singlecell", 40, "0"),
        ("rapids_singlecell", 30, "1"),
        ("rapids_singlecell", 20, "2"),
        ("rapids_singlecell", 15, "3"),
        ("rapids_singlecell", 10, "4"),
    ]


def test_formats(capsys: pytest.CaptureFixture) -> None:
    log._install_handler(sys.stderr)
    s.verbosity = Verbosity.debug
    log.error("0")
    assert capsys.readouterr().err == "ERROR: 0\n"
    log.warning("1")
    assert capsys.readouterr().err == "WARNING: 1\n"
    log.info("2")
    assert capsys.readouterr().err == "2\n"
    log.hint("3")
    assert capsys.readouterr().err == "--> 3\n"
    log.debug("4")
    assert capsys.readouterr().err == "    4\n"


def test_deep(capsys: pytest.CaptureFixture) -> None:
    log._install_handler(sys.stderr)
    s.verbosity = Verbosity.hint
    log.hint("0")
    assert capsys.readouterr().err == "--> 0\n"
    log.hint("1", deep="1!")
    assert capsys.readouterr().err == "--> 1\n"
    s.verbosity = Verbosity.debug
    log.hint("2")
    assert capsys.readouterr().err == "--> 2\n"
    log.hint("3", deep="3!")
    assert capsys.readouterr().err == "--> 3: 3!\n"


def test_timing(monkeypatch, capsys: pytest.CaptureFixture) -> None:
    counter = 0

    class IncTime:
        @staticmethod
        def now(tz):
            nonlocal counter
            counter += 1
            return datetime(2000, 1, 1, second=counter, microsecond=counter, tzinfo=tz)

    monkeypatch.setattr(log, "datetime", IncTime)
    log._install_handler(sys.stderr)
    s.verbosity = Verbosity.debug

    log.hint("1")
    assert counter == 1
    assert capsys.readouterr().err == "--> 1\n"

    start = log.info("2")
    assert counter == 2
    assert capsys.readouterr().err == "2\n"

    log.hint("3")
    assert counter == 3
    assert capsys.readouterr().err == "--> 3\n"

    log.info("4", time=start)
    assert counter == 4
    assert capsys.readouterr().err == "4 (0:00:02)\n"

    log.info("5 {time_passed}", time=start)
    assert counter == 5
    assert capsys.readouterr().err == "5 0:00:03\n"


def test_verbosity_propagates_to_cuml() -> None:
    """Setting rsc.settings.verbosity also updates cuml's logger level."""
    import cuml.internals.logger as cuml_log

    s.verbosity = Verbosity.error
    assert cuml_log.get_level() == cuml_log.level_enum.error

    s.verbosity = Verbosity.warning
    assert cuml_log.get_level() == cuml_log.level_enum.warn

    s.verbosity = Verbosity.info
    assert cuml_log.get_level() == cuml_log.level_enum.info

    s.verbosity = Verbosity.hint
    # cuml has no hint level; we map hint -> info
    assert cuml_log.get_level() == cuml_log.level_enum.info

    s.verbosity = Verbosity.debug
    assert cuml_log.get_level() == cuml_log.level_enum.debug


def test_verbosity_override() -> None:
    """Verbosity.override is a context manager that restores the prior level."""
    s.verbosity = Verbosity.info
    with s.verbosity.override(Verbosity.debug):
        assert s.verbosity == Verbosity.debug
    assert s.verbosity == Verbosity.info


def test_settings_override() -> None:
    """scverse_misc Settings.override restores the prior value."""
    s.verbosity = Verbosity.info
    with s.override(verbosity="error"):
        assert s.verbosity == Verbosity.error
    assert s.verbosity == Verbosity.info


def test_string_verbosity_coercion() -> None:
    """`settings.verbosity = "debug"` (string by name) is accepted."""
    s.verbosity = "debug"
    assert s.verbosity == Verbosity.debug
    s.verbosity = "warning"
    assert s.verbosity == Verbosity.warning


def test_invalid_verbosity_raises() -> None:
    """Unknown verbosity names raise a ValidationError-style error."""
    from pydantic import ValidationError

    with pytest.raises(ValidationError):
        s.verbosity = "nonexistent"


def test_module_exports() -> None:
    """The expected names are accessible from the rsc top level."""
    assert rsc.settings is s
    assert rsc.Verbosity is Verbosity
    assert rsc.logging is log
