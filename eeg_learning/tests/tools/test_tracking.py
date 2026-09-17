"""Tests for cross-stage MLflow run threading (eeg_learning/tools/tracking.py)."""

from __future__ import annotations

import os
import time
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from eeg_learning.tools import tracking


@pytest.fixture
def cfg(tmp_path):
    """Config with the model directory — and so the run token — under tmp_path."""
    return {
        "run": {
            "experiment_name": "eeg_learning",
            "use_azure_artifacts": False,
            "azure_artifact_root": "wasbs://bucket/experiments",
            "mlflow_tracking_uri": str(tmp_path / "mlruns"),
            "mlflow_required": True,
        },
        "output": {"saved_models_path": str(tmp_path / "saved_models" / "window_model")},
    }


class TestConfigAccessors:
    def test_required_defaults_to_true(self):
        assert tracking.is_required({"run": {}}) is True

    def test_required_honours_flag(self, cfg):
        cfg["run"]["mlflow_required"] = False
        assert tracking.is_required(cfg) is False

    def test_tracking_uri_falls_back_to_default(self):
        assert tracking.tracking_uri({"run": {}}) == tracking.DEFAULT_TRACKING_URI

    def test_experiment_name_suffixed_when_local(self, cfg):
        assert tracking.experiment_name(cfg) == "eeg_learning_local"

    def test_experiment_name_bare_when_azure(self, cfg):
        cfg["run"]["use_azure_artifacts"] = True
        assert tracking.experiment_name(cfg) == "eeg_learning"


class TestConfigure:
    """Ported from tests/pipeline/test_evaluate.py when configure_mlflow moved here."""

    def test_uses_local_experiment_when_azure_disabled(self):
        cfg = {"run": {"experiment_name": "test-exp", "use_azure_artifacts": False}}
        mlflow = MagicMock()
        mlflow.get_experiment_by_name.return_value = None
        with patch("eeg_learning.tools.tracking.mlflow", mlflow):
            tracking.configure(cfg)

        mlflow.set_tracking_uri.assert_called_once_with("mlruns")
        mlflow.create_experiment.assert_called_once_with("test-exp_local")
        mlflow.set_experiment.assert_called_once_with("test-exp_local")

    def test_uses_azure_artifact_when_enabled(self):
        cfg = {
            "run": {
                "experiment_name": "test-exp",
                "use_azure_artifacts": True,
                "azure_artifact_root": "wasbs://bucket/path",
            }
        }
        mlflow = MagicMock()
        mlflow.get_experiment_by_name.return_value = None
        with patch("eeg_learning.tools.tracking.mlflow", mlflow):
            tracking.configure(cfg)

        mlflow.create_experiment.assert_called_once_with("test-exp", artifact_location="wasbs://bucket/path")

    def test_explicit_experiment_overrides_config(self):
        cfg = {"run": {"experiment_name": "test-exp", "use_azure_artifacts": False}}
        mlflow = MagicMock()
        mlflow.get_experiment_by_name.return_value = object()
        with patch("eeg_learning.tools.tracking.mlflow", mlflow):
            assert tracking.configure(cfg, experiment="from-token") == "from-token"

        mlflow.set_experiment.assert_called_once_with("from-token")


class TestToken:
    def test_write_then_read_round_trips(self, cfg, tmp_path):
        model = tmp_path / "deep4_2026-08-13_14-15-27_params.pt"
        tracking.write_token(cfg, run_id="abc123", model_path=model)

        token = tracking.read_token(cfg)
        assert token["run_id"] == "abc123"
        assert token["model_path"] == str(model)
        assert token["experiment"] == "eeg_learning_local"

    def test_write_creates_parent_directory(self, cfg):
        path = tracking.write_token(cfg, run_id=None, model_path="model.pt")
        assert path.is_file()
        assert path.name == tracking.TOKEN_FILENAME

    def test_token_written_even_without_a_run(self, cfg):
        """The model_path handoff must not depend on MLflow being reachable."""
        tracking.write_token(cfg, run_id=None, model_path="model.pt")
        assert tracking.read_token(cfg)["run_id"] is None

    def test_read_returns_none_when_absent(self, cfg):
        assert tracking.read_token(cfg) is None

    def test_read_returns_none_when_config_has_no_model_dir(self):
        """A partial stage config is treated as 'no token', not a raw KeyError."""
        assert tracking.read_token({"run": {}, "output": {}}) is None

    def test_read_returns_none_on_corrupt_token(self, cfg):
        path = tracking.token_path(cfg)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("{not json")
        assert tracking.read_token(cfg) is None


class TestResolveModelPath:
    def test_prefers_the_token(self, cfg, tmp_path):
        recorded = Path(cfg["output"]["saved_models_path"]) / "recorded.pt"
        recorded.parent.mkdir(parents=True, exist_ok=True)
        recorded.touch()
        newer = recorded.parent / "newer.pt"
        newer.touch()
        # Make the untracked file unambiguously newer than the recorded one.
        os.utime(newer, (time.time() + 100, time.time() + 100))

        tracking.write_token(cfg, run_id="abc", model_path=recorded)
        assert tracking.resolve_model_path(cfg) == recorded

    def test_falls_back_to_newest_when_no_token(self, cfg):
        save_dir = Path(cfg["output"]["saved_models_path"])
        save_dir.mkdir(parents=True, exist_ok=True)
        older = save_dir / "older.pt"
        older.touch()
        newer = save_dir / "newer.pt"
        newer.touch()
        os.utime(newer, (time.time() + 100, time.time() + 100))

        assert tracking.resolve_model_path(cfg) == newer

    def test_falls_back_when_token_points_at_missing_model(self, cfg):
        save_dir = Path(cfg["output"]["saved_models_path"])
        save_dir.mkdir(parents=True, exist_ok=True)
        present = save_dir / "present.pt"
        present.touch()

        tracking.write_token(cfg, run_id="abc", model_path=save_dir / "vanished.pt")
        assert tracking.resolve_model_path(cfg) == present

    def test_raises_when_nothing_to_resolve(self, cfg):
        Path(cfg["output"]["saved_models_path"]).mkdir(parents=True, exist_ok=True)
        with pytest.raises(FileNotFoundError):
            tracking.resolve_model_path(cfg)


class TestStartRun:
    def test_mint_starts_a_fresh_run(self, cfg):
        mlflow = MagicMock()
        mlflow.start_run.return_value.info.run_id = "minted"
        with patch("eeg_learning.tools.tracking.mlflow", mlflow):
            with tracking.start_run(cfg, resume=False) as tracker:
                assert tracker.run_id == "minted"

        assert mlflow.start_run.call_args.kwargs["run_id"] is None

    def test_resume_reuses_the_token_run_id(self, cfg):
        tracking.write_token(cfg, run_id="minted", model_path="model.pt")
        mlflow = MagicMock()
        mlflow.start_run.return_value.info.run_id = "minted"
        with patch("eeg_learning.tools.tracking.mlflow", mlflow):
            with tracking.start_run(cfg, resume=True) as tracker:
                assert tracker.run_id == "minted"

        assert mlflow.start_run.call_args.kwargs["run_id"] == "minted"

    def test_resume_uses_the_experiment_recorded_in_the_token(self, cfg):
        """Guards the start_run experiment-mismatch trap when config drifts."""
        tracking.write_token(cfg, run_id="minted", model_path="model.pt")
        cfg["run"]["use_azure_artifacts"] = True  # would otherwise resolve differently

        mlflow = MagicMock()
        mlflow.start_run.return_value.info.run_id = "minted"
        with patch("eeg_learning.tools.tracking.mlflow", mlflow):
            with tracking.start_run(cfg, resume=True):
                pass

        mlflow.set_experiment.assert_called_once_with("eeg_learning_local")

    def test_resume_without_a_token_raises_when_required(self, cfg):
        with patch("eeg_learning.tools.tracking.mlflow", MagicMock()):
            with pytest.raises(tracking.MLflowUnavailableError):
                with tracking.start_run(cfg, resume=True):
                    pass

    def test_resume_without_a_token_no_ops_when_optional(self, cfg):
        cfg["run"]["mlflow_required"] = False
        with patch("eeg_learning.tools.tracking.mlflow", MagicMock()):
            with tracking.start_run(cfg, resume=True) as tracker:
                assert isinstance(tracker, tracking.NullTracker)
                tracker.log_metrics({"accuracy": 1.0})  # must not raise

    def test_resume_with_partial_config_no_ops_when_optional(self):
        """Stages with a trimmed config still run when tracking is not required."""
        cfg = {"run": {"mlflow_required": False}, "output": {}}
        with patch("eeg_learning.tools.tracking.mlflow", MagicMock()):
            with tracking.start_run(cfg, resume=True) as tracker:
                assert isinstance(tracker, tracking.NullTracker)

    def test_tracking_error_raises_when_required(self, cfg):
        mlflow = MagicMock()
        mlflow.start_run.side_effect = OSError("tracking server unreachable")
        with patch("eeg_learning.tools.tracking.mlflow", mlflow):
            with pytest.raises(tracking.MLflowUnavailableError):
                with tracking.start_run(cfg, resume=False):
                    pass

    def test_tracking_error_degrades_when_optional(self, cfg):
        cfg["run"]["mlflow_required"] = False
        mlflow = MagicMock()
        mlflow.start_run.side_effect = OSError("tracking server unreachable")
        with patch("eeg_learning.tools.tracking.mlflow", mlflow):
            with tracking.start_run(cfg, resume=False) as tracker:
                assert isinstance(tracker, tracking.NullTracker)

    def test_creates_experiment_only_when_missing(self, cfg):
        mlflow = MagicMock()
        mlflow.get_experiment_by_name.return_value = object()
        with patch("eeg_learning.tools.tracking.mlflow", mlflow):
            with tracking.start_run(cfg, resume=False):
                pass

        mlflow.create_experiment.assert_not_called()

    def test_azure_artifact_root_passed_on_creation(self, cfg):
        cfg["run"]["use_azure_artifacts"] = True
        mlflow = MagicMock()
        mlflow.get_experiment_by_name.return_value = None
        with patch("eeg_learning.tools.tracking.mlflow", mlflow):
            with tracking.start_run(cfg, resume=False):
                pass

        mlflow.create_experiment.assert_called_once_with("eeg_learning", artifact_location="wasbs://bucket/experiments")

    def test_tracking_uri_comes_from_config(self, cfg):
        mlflow = MagicMock()
        with patch("eeg_learning.tools.tracking.mlflow", mlflow):
            with tracking.start_run(cfg, resume=False):
                pass

        mlflow.set_tracking_uri.assert_called_once_with(cfg["run"]["mlflow_tracking_uri"])

    def test_stale_run_id_env_var_is_cleared(self, cfg, monkeypatch):
        monkeypatch.setenv("MLFLOW_RUN_ID", "stale")
        mlflow = MagicMock()
        with patch("eeg_learning.tools.tracking.mlflow", mlflow):
            with tracking.start_run(cfg, resume=False):
                pass

        assert "MLFLOW_RUN_ID" not in os.environ


class TestTrackers:
    def test_tracker_delegates_to_mlflow(self):
        mlflow = MagicMock()
        with patch("eeg_learning.tools.tracking.mlflow", mlflow):
            tracker = tracking.Tracker("abc")
            tracker.log_params({"a": 1})
            tracker.log_metrics({"b": 2.0})
            tracker.log_artifact(Path("model.pt"), artifact_path="model")
            tracker.set_tags({"c": "d"})

        mlflow.log_params.assert_called_once_with({"a": 1})
        mlflow.log_metrics.assert_called_once_with({"b": 2.0}, step=None)
        mlflow.log_artifact.assert_called_once_with("model.pt", artifact_path="model")
        mlflow.set_tags.assert_called_once_with({"c": "d"})

    def test_tracker_passes_step_through(self):
        """Per-epoch curves depend on step reaching mlflow."""
        mlflow = MagicMock()
        with patch("eeg_learning.tools.tracking.mlflow", mlflow):
            tracking.Tracker("abc").log_metrics({"train_loss": 0.5}, step=7)

        mlflow.log_metrics.assert_called_once_with({"train_loss": 0.5}, step=7)

    def test_nested_run_opens_a_child(self):
        mlflow = MagicMock()
        mlflow.start_run.return_value.__enter__.return_value.info.run_id = "child"
        with patch("eeg_learning.tools.tracking.mlflow", mlflow):
            with tracking.Tracker("parent").nested_run("rep-1") as child:
                assert child.run_id == "child"

        mlflow.start_run.assert_called_once_with(nested=True, run_name="rep-1")

    def test_null_tracker_nested_run_yields_itself(self):
        null = tracking.NullTracker()
        with null.nested_run("rep-1") as child:
            assert child is null
            child.log_metrics({"test_acc": 1.0})  # must not raise

    def test_null_tracker_matches_the_tracker_surface(self):
        methods = {name for name in vars(tracking.Tracker) if not name.startswith("_")}
        assert methods <= set(dir(tracking.NullTracker))
        assert tracking.NullTracker().run_id is None
