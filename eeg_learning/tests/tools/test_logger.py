"""Test Suite for the logging wrapper"""

from __future__ import annotations

import io
import logging
import sys

import pytest

from eeg_learning.tools.logger import ROOT_NAME, Logger, get_logger, _owned_handlers


def owned_handlers():
    """Handlers this module installed, ignoring anything else on the logger.

    ``Logger.configure`` sets ``propagate = False`` on the package root, and
    pytest 9 attaches its four capture handlers (live-log, log-file, and two
    ``LogCaptureHandler``s) to every non-propagating logger so ``caplog`` keeps
    working. Counting raw ``.handlers`` therefore measures the test runner, not
    the code under test.
    """
    return _owned_handlers(logging.getLogger(ROOT_NAME))


@pytest.fixture(autouse=True)
def clean_root():
    # Handlers live on the shared package logger, so every test starts and ends
    # with a clean root or configuration leaks between tests.
    Logger.reset()
    yield
    Logger.reset()
    logging.getLogger(ROOT_NAME).setLevel(logging.NOTSET)


def test_names_are_reparented_under_the_package_root():
    assert Logger().name == ROOT_NAME
    assert Logger("io.dataset_builder").name == "eeg_learning.io.dataset_builder"
    # __name__ from inside the package is already qualified, so it is left alone.
    assert Logger("eeg_learning.models.factory").name == "eeg_learning.models.factory"


def test_first_logger_installs_a_console_handler_at_info():
    log = get_logger("io")

    root = logging.getLogger(ROOT_NAME)
    assert len(owned_handlers()) == 1
    assert root.level == logging.INFO
    assert log.is_enabled_for("INFO")
    assert not log.is_enabled_for(logging.DEBUG)


def test_repeat_configuration_does_not_stack_console_handlers():
    get_logger("io")
    get_logger("models")
    Logger.configure(level="DEBUG")

    root = logging.getLogger(ROOT_NAME)
    assert len(owned_handlers()) == 1  # keyed by destination, so refreshed not added
    assert root.level == logging.DEBUG


def test_messages_reach_the_configured_stream(capsys):
    log = Logger("io", level="DEBUG")

    log.info("loaded %d recordings", 12)
    log.debug("cache hit")

    err = capsys.readouterr().err
    assert "loaded 12 recordings" in err
    assert "eeg_learning.io" in err
    assert "cache hit" in err


def test_console_follows_stderr_rebound_after_setup():
    # Modules build their logger at import time, so the console handler exists
    # before pytest's capsys (or a notebook, or redirect_stderr) swaps
    # sys.stderr. Resolving the stream at emit time is what keeps that output
    # visible; pinning it silently dropped the CLI's error line under capsys.
    log = get_logger("io")  # handler installed against the current stderr
    replacement = io.StringIO()
    original, sys.stderr = sys.stderr, replacement
    try:
        log.warning("after rebinding")
    finally:
        sys.stderr = original

    assert "after rebinding" in replacement.getvalue()


def test_reconfiguring_keeps_writing_to_the_live_stderr():
    # What cli.main() does: a logger already exists from import, then the entry
    # point re-configures the format. The refreshed handler must still follow
    # the current stderr.
    log = get_logger("cli")
    Logger.configure(fmt="%(message)s")
    replacement = io.StringIO()
    original, sys.stderr = sys.stderr, replacement
    try:
        log.error("error: %s", "unknown backend 'nope'")
    finally:
        sys.stderr = original

    assert replacement.getvalue() == "error: unknown backend 'nope'\n"


def test_level_filters_below_threshold(capsys):
    log = Logger("io", level="WARNING")

    log.info("chatty")
    log.warning("worth reading")

    err = capsys.readouterr().err
    assert "chatty" not in err
    assert "worth reading" in err


def test_log_file_is_written_and_parents_created(tmp_path):
    log_file = tmp_path / "nested" / "run.log"

    Logger("pipeline", log_file=log_file).info("stage complete")
    Logger.reset()  # flush + close before reading back

    assert log_file.read_text().strip().endswith("eeg_learning.pipeline: stage complete")


def test_child_extends_the_parent_name():
    assert get_logger("io").child("labeling").name == "eeg_learning.io.labeling"


def test_set_level_is_local_to_the_logger():
    parent = get_logger("io")  # package root at INFO
    child = parent.child("labeling")
    child.set_level("ERROR")

    assert not child.is_enabled_for("WARNING")
    assert parent.is_enabled_for("WARNING")


def test_from_config_reads_the_logging_section(tmp_path):
    log_file = tmp_path / "run.log"
    config = {"logging": {"level": "DEBUG", "file": str(log_file), "console": False}}

    log = Logger.from_config(config, "pipeline")
    log.debug("verbose detail")
    Logger.reset()

    assert owned_handlers() == []  # console disabled, file closed by reset
    assert "verbose detail" in log_file.read_text()


@pytest.mark.parametrize("config", [{}, {"logging": {}}, {"logging": {"file": ""}}])
def test_from_config_tolerates_a_missing_section(config):
    # params.toml has no [logging] section on older configs; console/INFO default.
    Logger.from_config(config)

    root = logging.getLogger(ROOT_NAME)
    assert root.level == logging.INFO
    assert len(owned_handlers()) == 1


def test_unknown_level_is_rejected():
    with pytest.raises(ValueError, match="unknown log level"):
        Logger("io", level="LOUD")


def test_exception_records_the_traceback(capsys):
    log = get_logger("io")

    try:
        raise RuntimeError("edf read failed")
    except RuntimeError:
        log.exception("could not load recording")

    err = capsys.readouterr().err
    assert "could not load recording" in err
    assert "RuntimeError: edf read failed" in err


def test_package_output_does_not_propagate_to_the_interpreter_root(capsys):
    get_logger("io").info("once only")

    assert logging.getLogger(ROOT_NAME).propagate is False
    assert capsys.readouterr().err.count("once only") == 1
