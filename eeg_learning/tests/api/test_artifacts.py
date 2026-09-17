"""Tests for ModelArtifact (eeg_learning/api/artifacts.py)."""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import patch

import pytest

from eeg_learning.api.artifacts import MANIFEST_FORMAT_VERSION, ModelArtifact


class DummyClassifier:
    """Stand-in for an EEGClassifier whose save_params writes a real file."""

    def __init__(self):
        self.saved_to = None

    def save_params(self, f_params):
        self.saved_to = f_params
        Path(f_params).write_text("weights")


@pytest.fixture
def build_kwargs():
    return {
        "n_channels": 19,
        "n_classes": 2,
        "input_window_samples": 6000,
        "drop_prob": 0.1,
        "final_conv_length": "auto",
        "n_filters_time": 25,
    }


class TestSave:
    def test_writes_weights_and_manifest_pair(self, tmp_path, build_kwargs):
        artifact = ModelArtifact.save(
            DummyClassifier(),
            model_id="deep4_run1",
            model_name="deep4",
            build_kwargs=build_kwargs,
            output_dir=tmp_path,
        )
        assert artifact.model_path == tmp_path / "deep4_run1.pt"
        assert artifact.manifest_path == tmp_path / "deep4_run1.json"
        assert artifact.model_path.exists()
        assert artifact.manifest_path.exists()

    def test_save_params_called_with_pt_path(self, tmp_path, build_kwargs):
        clf = DummyClassifier()
        ModelArtifact.save(
            clf,
            model_id="m",
            model_name="deep4",
            build_kwargs=build_kwargs,
            output_dir=tmp_path,
        )
        assert clf.saved_to == str(tmp_path / "m.pt")

    def test_manifest_contents(self, tmp_path, build_kwargs):
        artifact = ModelArtifact.save(
            DummyClassifier(),
            model_id="m",
            model_name="deep4",
            build_kwargs=build_kwargs,
            output_dir=tmp_path,
            training_config={"n_epochs": 30},
        )
        manifest = json.loads(artifact.manifest_path.read_text())
        assert manifest["format_version"] == MANIFEST_FORMAT_VERSION
        assert manifest["model_id"] == "m"
        assert manifest["weights_file"] == "m.pt"
        assert manifest["model"] == {"name": "deep4", "build_kwargs": build_kwargs}
        assert manifest["training"] == {"n_epochs": 30}
        assert "created_at" in manifest

    def test_creates_nested_output_dir(self, tmp_path, build_kwargs):
        out = tmp_path / "deep" / "nested"
        ModelArtifact.save(
            DummyClassifier(),
            model_id="m",
            model_name="deep4",
            build_kwargs=build_kwargs,
            output_dir=out,
        )
        assert (out / "m.pt").exists()

    def test_extra_merged_into_manifest(self, tmp_path, build_kwargs):
        artifact = ModelArtifact.save(
            DummyClassifier(),
            model_id="m",
            model_name="deep4",
            build_kwargs=build_kwargs,
            output_dir=tmp_path,
            extra={"git_sha": "abc123"},
        )
        assert artifact.manifest["git_sha"] == "abc123"

    def test_training_config_defaults_to_empty(self, tmp_path, build_kwargs):
        artifact = ModelArtifact.save(
            DummyClassifier(),
            model_id="m",
            model_name="deep4",
            build_kwargs=build_kwargs,
            output_dir=tmp_path,
        )
        assert artifact.manifest["training"] == {}


class TestLoad:
    def test_round_trip(self, tmp_path, build_kwargs):
        ModelArtifact.save(
            DummyClassifier(),
            model_id="m",
            model_name="deep4",
            build_kwargs=build_kwargs,
            output_dir=tmp_path,
            training_config={"n_epochs": 30},
        )
        reloaded = ModelArtifact.load("m", models_dir=tmp_path)
        assert reloaded.model_name == "deep4"
        assert reloaded.build_kwargs == build_kwargs
        assert reloaded.manifest["training"] == {"n_epochs": 30}
        assert reloaded.model_path == tmp_path / "m.pt"

    def test_missing_manifest_raises(self, tmp_path):
        with pytest.raises(FileNotFoundError, match="No manifest"):
            ModelArtifact.load("ghost", models_dir=tmp_path)

    def test_missing_weights_raises(self, tmp_path, build_kwargs):
        ModelArtifact.save(
            DummyClassifier(),
            model_id="m",
            model_name="deep4",
            build_kwargs=build_kwargs,
            output_dir=tmp_path,
        )
        (tmp_path / "m.pt").unlink()
        with pytest.raises(FileNotFoundError, match="missing weights"):
            ModelArtifact.load("m", models_dir=tmp_path)


class TestBuildModel:
    @patch("eeg_learning.api.artifacts.ModelFactory")
    def test_rebuilds_from_manifest(self, mock_factory, tmp_path, build_kwargs):
        artifact = ModelArtifact.save(
            DummyClassifier(),
            model_id="m",
            model_name="deep4",
            build_kwargs=build_kwargs,
            output_dir=tmp_path,
        )
        model = artifact.build_model()
        mock_factory.create.assert_called_once_with("deep4", **build_kwargs)
        assert model is mock_factory.create.return_value
