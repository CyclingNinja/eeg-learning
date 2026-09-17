"""Command-line entrypoint for the EEG win-stack.

A thin wrapper over the API's backend layer: parse arguments, build a
:class:`~eeg_learning.api.backends.Job`, hand it to the selected
:class:`~eeg_learning.api.backends.Backend`, and report the result. All the real
work lives in :mod:`eeg_learning.api` — this module only translates argv into a
job and prints the outcome, so the same command a developer runs locally is the
command the remote backends invoke.

Invoke as ``python -m eeg_learning <command> ...`` or via the ``eeg-learning``
console script. Results print to stdout as JSON so callers (a shell, a remote
runner) can parse them; user-facing errors and any log output from the run go to
stderr, errors with a non-zero exit.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from eeg_learning.api.backends import Job, JobKind, get_backend
from eeg_learning.config import load as load_config
from eeg_learning.tools.logger import Logger, get_logger

log = get_logger(__name__)


def cmd_train(args: argparse.Namespace) -> dict:
    """Run the ``train`` command: build a TRAIN job and submit it to a backend.

    Returns the backend's JSON-serialisable result dict (model id + artifact
    paths). Loading of windowed data and everything downstream happens inside
    :func:`eeg_learning.api.jobs.run_training`, reached via the backend.
    """
    config = load_config(args.config) if args.config is not None else load_config()
    job = Job(
        kind=JobKind.TRAIN,
        config=config,
        windows_path=str(args.windows_path),
        output_dir=str(args.output_dir),
        options={"model_id": args.model_id},
    )
    backend = get_backend(args.backend)
    handle = backend.submit(job)
    return backend.result(handle)


def _add_train_parser(subparsers: argparse._SubParsersAction) -> None:
    """Register the ``train`` subcommand and its arguments."""
    parser = subparsers.add_parser(
        "train",
        help="Train a model from windowed data and save it as an artifact.",
        description="Train a model from pre-windowed data on disk and save a weights + manifest artifact pair.",
    )
    parser.add_argument(
        "--windows-path",
        required=True,
        type=Path,
        help="Directory of saved windowed data to train on.",
    )
    parser.add_argument(
        "--output-dir",
        required=True,
        type=Path,
        help="Directory the model artifact pair is written into.",
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=None,
        help="Path to a params.toml. Defaults to the packaged config.",
    )
    parser.add_argument(
        "--backend",
        default="local",
        help="Execution backend: 'local' (default); 'azureml'/'slurm' are planned.",
    )
    parser.add_argument(
        "--model-id",
        default=None,
        help="Identifier for the saved artifact. Defaults to '<name>_<timestamp>'.",
    )
    parser.set_defaults(func=cmd_train)


def build_parser() -> argparse.ArgumentParser:
    """Build the top-level argument parser with all subcommands attached."""
    parser = argparse.ArgumentParser(
        prog="eeg-learning",
        description="Train and (later) evaluate EEG window-stack models.",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)
    _add_train_parser(subparsers)
    return parser


def main(argv: list[str] | None = None) -> int:
    """Parse ``argv``, dispatch to the chosen command, and print its result.

    Returns a process exit code: ``0`` on success, ``1`` for a user-facing error
    (unknown or unimplemented backend). Errors from the job itself propagate as
    exceptions — the fail-loud behaviour expected of a local run.
    """
    parser = build_parser()
    args = parser.parse_args(argv)
    # Bare messages on stderr: for a CLI the text is the product, so timestamps
    # and logger names would just be noise in front of it.
    Logger.configure(fmt="%(message)s")
    try:
        result = args.func(args)
    except (NotImplementedError, ValueError) as exc:
        log.error("error: %s", exc)
        return 1
    json.dump(result, sys.stdout, indent=2)
    sys.stdout.write("\n")
    return 0
