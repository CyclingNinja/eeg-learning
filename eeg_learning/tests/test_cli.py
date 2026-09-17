"""Tests for the command-line entrypoint (eeg_learning/cli.py).

The heavy work (data loading, training) lives behind the backend layer and is
mocked here; these tests assert the CLI's own job: argv -> Job -> backend, and
result -> stdout / errors -> exit code.
"""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from eeg_learning import cli
from eeg_learning.api.backends import JobKind


@pytest.fixture
def fake_config():
    """A sentinel config dict returned by the patched loader."""
    return {"training": {}, "model": {}, "split": {}, "run": {}}


def test_build_parser_train_requires_paths():
    """`train` without the required path args exits (argparse error)."""
    parser = cli.build_parser()
    with pytest.raises(SystemExit):
        parser.parse_args(["train"])


def test_no_command_errors():
    """Invoking with no subcommand is an argparse error, not a crash."""
    parser = cli.build_parser()
    with pytest.raises(SystemExit):
        parser.parse_args([])


def test_cmd_train_builds_train_job_and_submits(fake_config):
    """cmd_train wires argv into a TRAIN Job and returns the backend result."""
    backend = MagicMock()
    handle = object()
    backend.submit.return_value = handle
    backend.result.return_value = {"model_id": "deep4_x", "model_path": "m.pt"}

    with (
        patch.object(cli, "load_config", return_value=fake_config) as load_config,
        patch.object(cli, "get_backend", return_value=backend) as get_backend,
    ):
        args = cli.build_parser().parse_args(
            [
                "train",
                "--windows-path",
                "/data/windows",
                "--output-dir",
                "/out",
                "--model-id",
                "deep4_x",
            ]
        )
        result = cli.cmd_train(args)

    load_config.assert_called_once_with()  # no --config -> packaged default
    get_backend.assert_called_once_with("local")  # default backend

    job = backend.submit.call_args.args[0]
    assert job.kind is JobKind.TRAIN
    assert job.config is fake_config
    assert job.windows_path == "/data/windows"
    assert job.output_dir == "/out"
    assert job.options == {"model_id": "deep4_x"}

    backend.result.assert_called_once_with(handle)
    assert result == {"model_id": "deep4_x", "model_path": "m.pt"}


def test_cmd_train_passes_explicit_config_path(fake_config):
    """A --config value is forwarded to the loader as a Path."""
    with (
        patch.object(cli, "load_config", return_value=fake_config) as load_config,
        patch.object(cli, "get_backend", return_value=MagicMock()),
    ):
        args = cli.build_parser().parse_args(
            [
                "train",
                "--windows-path",
                "w",
                "--output-dir",
                "o",
                "--config",
                "custom/params.toml",
            ]
        )
        cli.cmd_train(args)

    load_config.assert_called_once_with(Path("custom/params.toml"))


def test_main_prints_result_json_and_returns_zero(fake_config, capsys):
    """main writes the result dict as JSON to stdout and exits 0."""
    backend = MagicMock()
    backend.submit.return_value = object()
    backend.result.return_value = {"model_id": "abc"}

    with (
        patch.object(cli, "load_config", return_value=fake_config),
        patch.object(cli, "get_backend", return_value=backend),
    ):
        code = cli.main(["train", "--windows-path", "w", "--output-dir", "o"])

    assert code == 0
    assert json.loads(capsys.readouterr().out) == {"model_id": "abc"}


def test_main_unknown_backend_returns_one(fake_config, capsys):
    """An unknown backend surfaces as a clean stderr error and exit code 1."""
    with patch.object(cli, "load_config", return_value=fake_config):
        code = cli.main(["train", "--windows-path", "w", "--output-dir", "o", "--backend", "nope"])

    assert code == 1
    assert "error:" in capsys.readouterr().err


def test_main_planned_backend_returns_one(fake_config, capsys):
    """A planned-but-unbuilt backend (azureml) errors cleanly, exit code 1."""
    with patch.object(cli, "load_config", return_value=fake_config):
        code = cli.main(["train", "--windows-path", "w", "--output-dir", "o", "--backend", "azureml"])

    assert code == 1
    assert "error:" in capsys.readouterr().err
