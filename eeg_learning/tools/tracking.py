"""Cross-stage MLflow run threading for the DVC pipeline.

DVC launches every stage as its own process, so a single "experiment run"
cannot simply be held open across stages. Instead the ``train`` stage mints the
run and writes a small run token next to the model it produced; ``evaluate``
and ``decision`` read that token and resume the same run via
``mlflow.start_run(run_id=...)``.

The token lives inside ``output.saved_models_path``, which is a cached DVC
output of the train stage. That makes the run ID a property of the trained
model: when DVC restores a cached train stage it restores the token with it, so
a re-run of ``evaluate`` alone resumes the run that actually produced the model
being evaluated.

The token also records which ``.pt`` file train wrote, so downstream stages no
longer recover the model by globbing the directory and taking the newest entry.
That handoff works whether or not MLflow is reachable.
"""

from __future__ import annotations

import json
import os
from contextlib import contextmanager
from datetime import datetime
from pathlib import Path

import mlflow

from eeg_learning.tools.logger import get_logger

log = get_logger(__name__)

TOKEN_FILENAME = "run.json"

DEFAULT_TRACKING_URI = "mlruns"


class MLflowUnavailableError(RuntimeError):
    """Raised when tracking is required but the run could not be established."""


def _run_cfg(cfg):
    return cfg.get("run", {})


def is_required(cfg) -> bool:
    """Whether a tracking failure should abort the stage.

    Defaults to ``True``: MLflow is the system of record for run history, so a
    silently missing run is a hole in that record. Set ``run.mlflow_required``
    to ``false`` for offline work where the pipeline should still complete.
    """
    return bool(_run_cfg(cfg).get("mlflow_required", True))


def tracking_uri(cfg) -> str:
    return _run_cfg(cfg).get("mlflow_tracking_uri", DEFAULT_TRACKING_URI)


def experiment_name(cfg) -> str:
    """Resolve the experiment name the same way in every stage.

    Resuming a run raises if ``set_experiment`` selected a different experiment
    than the run belongs to, so this must be deterministic across stages. The
    ``_local`` suffix means toggling ``use_azure_artifacts`` between stages
    would otherwise break every resume.
    """
    run_cfg = _run_cfg(cfg)
    name = run_cfg["experiment_name"]
    return name if run_cfg.get("use_azure_artifacts", False) else f"{name}_local"


def token_path(cfg) -> Path:
    """Location of the run token, inside the train stage's model directory.

    Raises ``KeyError`` when the config has no model directory; callers that
    only want to *find* a token treat that the same as a missing one.
    """
    return Path(cfg["output"]["saved_models_path"]) / TOKEN_FILENAME


def write_token(cfg, run_id: str | None, model_path: str | Path) -> Path:
    """Record the minted run ID and the model train actually wrote.

    ``run_id`` is ``None`` when tracking is disabled or unavailable; the token
    is still written, because the ``model_path`` handoff does not depend on
    MLflow.
    """
    path = token_path(cfg)
    path.parent.mkdir(parents=True, exist_ok=True)
    token = {
        "run_id": run_id,
        "experiment": experiment_name(cfg),
        "model_path": str(model_path),
        "created": datetime.now().isoformat(timespec="seconds"),
    }
    path.write_text(json.dumps(token, indent=2))
    log.info("wrote run token %s (run_id=%s)", path, run_id)
    return path


def read_token(cfg) -> dict | None:
    """Load the run token written by the train stage, or ``None`` if absent."""
    try:
        path = token_path(cfg)
    except KeyError:
        log.warning("config has no output.saved_models_path; no run token to read")
        return None

    if not path.is_file():
        log.warning("no run token at %s", path)
        return None
    try:
        return json.loads(path.read_text())
    except (OSError, json.JSONDecodeError) as exc:
        log.warning("run token %s is unreadable: %s", path, exc)
        return None


def resolve_model_path(cfg) -> Path:
    """Path of the model produced by train, from the token where possible.

    Falls back to the newest ``.pt`` in the model directory so that models
    trained outside the DVC pipeline still evaluate. Logs which path was taken,
    since the fallback silently picking the wrong file is the failure mode this
    replaces.
    """
    token = read_token(cfg)
    if token and token.get("model_path"):
        candidate = Path(token["model_path"])
        if candidate.is_file():
            log.info("resolved model from run token: %s", candidate)
            return candidate
        log.warning("run token points at missing model %s; falling back to newest .pt", candidate)

    saved_models_path = Path(cfg["output"]["saved_models_path"])
    candidates = sorted(saved_models_path.glob("*.pt"), key=lambda p: p.stat().st_mtime)
    if not candidates:
        raise FileNotFoundError(f"No model checkpoint (*.pt) found in {saved_models_path}")
    log.info("resolved model by mtime fallback: %s", candidates[-1])
    return candidates[-1]


class Tracker:
    """Thin wrapper over the active MLflow run.

    Call sites stay free of ``try``/``except`` and of ``if mlflow_enabled``
    branches: when tracking is unavailable and not required, :class:`NullTracker`
    stands in with the same surface.
    """

    def __init__(self, run_id: str):
        self.run_id = run_id

    def log_params(self, params: dict) -> None:
        mlflow.log_params(params)

    def log_metrics(self, metrics: dict, step: int | None = None) -> None:
        mlflow.log_metrics(metrics, step=step)

    def log_artifact(self, local_path: str | Path, artifact_path: str | None = None) -> None:
        mlflow.log_artifact(str(local_path), artifact_path=artifact_path)

    def set_tags(self, tags: dict) -> None:
        mlflow.set_tags(tags)

    @contextmanager
    def nested_run(self, run_name: str | None = None):
        """Open a child run under this one, for per-repetition results.

        Repetitions are separate experiment runs sharing one parent, which is
        what MLflow nesting is for — as opposed to the aggregate scalars, which
        stay on the parent so DVC can pick them up as metrics.
        """
        with mlflow.start_run(nested=True, run_name=run_name) as child:
            yield Tracker(child.info.run_id)


class NullTracker:
    """No-op stand-in used when tracking is unavailable and not required."""

    run_id = None

    def log_params(self, params: dict) -> None:
        log.debug("tracking disabled; dropping %d params", len(params))

    def log_metrics(self, metrics: dict, step: int | None = None) -> None:
        log.debug("tracking disabled; dropping %d metrics", len(metrics))

    def log_artifact(self, local_path: str | Path, artifact_path: str | None = None) -> None:
        log.debug("tracking disabled; not logging artifact %s", local_path)

    def set_tags(self, tags: dict) -> None:
        log.debug("tracking disabled; dropping %d tags", len(tags))

    @contextmanager
    def nested_run(self, run_name: str | None = None):
        log.debug("tracking disabled; not opening nested run %s", run_name)
        yield self


def configure(cfg, experiment: str | None = None) -> str:
    """Point MLflow at the configured tracking store and experiment.

    ``experiment`` overrides the config-derived name; resuming stages pass the
    name recorded in the token so a config change between stages cannot cause
    the experiment-mismatch error from ``mlflow.start_run``.
    """
    mlflow.set_tracking_uri(tracking_uri(cfg))
    name = experiment or experiment_name(cfg)

    if mlflow.get_experiment_by_name(name) is None:
        if _run_cfg(cfg).get("use_azure_artifacts", False):
            mlflow.create_experiment(name, artifact_location=_run_cfg(cfg)["azure_artifact_root"])
        else:
            mlflow.create_experiment(name)

    mlflow.set_experiment(name)
    return name


def _handle_failure(cfg, message: str, exc: Exception):
    if is_required(cfg):
        raise MLflowUnavailableError(
            f"{message}. Set run.mlflow_required = false to continue without tracking."
        ) from exc
    log.warning("%s; continuing without tracking: %s", message, exc)
    return NullTracker()


@contextmanager
def start_run(cfg, *, resume: bool):
    """Open the pipeline's MLflow run, minting it or resuming the minted one.

    ``resume=False`` (the train stage) creates the run. ``resume=True``
    (evaluate, decision) reads the token and rejoins that same run, so all
    stages of one pipeline invocation land in a single run.
    """
    token = read_token(cfg) if resume else None

    if resume and not (token and token.get("run_id")):
        try:
            location = str(token_path(cfg))
        except KeyError:
            location = "<no output.saved_models_path configured>"
        yield _handle_failure(
            cfg,
            f"No MLflow run to resume from {location}",
            FileNotFoundError(location),
        )
        return

    try:
        experiment = token.get("experiment") if token else None
        configure(cfg, experiment=experiment)
        run_id = token["run_id"] if token else None
        active = mlflow.start_run(run_id=run_id)
    except Exception as exc:  # noqa: BLE001 - tracking store, auth, and Azure errors all land here
        verb = "resume" if resume else "start"
        yield _handle_failure(cfg, f"Could not {verb} MLflow run", exc)
        return

    run_id = active.info.run_id
    log.info("%s MLflow run %s in experiment %r", "resumed" if resume else "started", run_id, experiment_name(cfg))
    try:
        with active:
            yield Tracker(run_id)
    finally:
        # Resuming stages must not leave MLFLOW_RUN_ID set: mlflow consumes it
        # destructively, and a stale value would hijack the next start_run.
        os.environ.pop("MLFLOW_RUN_ID", None)
