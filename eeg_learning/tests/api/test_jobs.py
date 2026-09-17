"""Tests for the training job orchestration (eeg_learning/api/jobs.py)."""

from __future__ import annotations

from dataclasses import asdict
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

from eeg_learning.api.jobs import (
    TrainResult,
    _model_build_kwargs,
    _training_config,
    run_training,
)
from eeg_learning.training.trainer import TrainingConfig


@pytest.fixture
def config():
    """Minimal resolved config covering the sections run_training reads."""
    return {
        "run": {"n_jobs": 1, "random_state": 87},
        "split": {
            "split_way": "train_on_tuab_tueg_test_on_tueg",
            "train_size": 0.8,
            "valid_size": 0.1,
            "test_size": 0.1,
            "shuffle": True,
        },
        "training": {
            "learning_rate": 0.001,
            "weight_decay": 0.0005,
            "batch_size": 1,
            "n_epochs": 30,
            "n_classes": 2,
            "test_on_eval": True,
            "earlystopping": True,
            "es_patience": 10,
            "es_threshold": 0.001,
            "checkpoint_dir": "",
        },
        "model": {
            "name": "deep4",
            "final_conv_length": "auto",
            "dropout": 0.1,
            "deep4": {"n_filters_time": 25, "n_filters_4": 200},
            "shallow": {"n_filters_time": 40},
        },
    }


EXPECTED_BUILD_KWARGS = {
    "n_channels": 19,
    "n_classes": 2,
    "input_window_samples": 6000,
    "drop_prob": 0.1,
    "final_conv_length": "auto",
    "n_filters_time": 25,
    "n_filters_4": 200,
}


class TestModelBuildKwargs:
    def test_applies_only_matching_subsection(self, config):
        kwargs = _model_build_kwargs(config["model"], n_classes=2, n_channels=19, window_len_samples=6000)
        assert kwargs == EXPECTED_BUILD_KWARGS
        # shallow's n_filters_time (40) must not leak in
        assert kwargs["n_filters_time"] == 25

    def test_no_matching_subsection_uses_base_only(self, config):
        config["model"]["name"] = "tcn"  # no [model.tcn] in this config
        kwargs = _model_build_kwargs(config["model"], n_classes=2, n_channels=19, window_len_samples=6000)
        assert kwargs == {
            "n_channels": 19,
            "n_classes": 2,
            "input_window_samples": 6000,
            "drop_prob": 0.1,
            "final_conv_length": "auto",
        }


class TestTrainingConfigMapping:
    def test_maps_fields_including_earlystopping(self, config):
        cfg = _training_config(config["training"])
        assert isinstance(cfg, TrainingConfig)
        assert cfg.learning_rate == 0.001
        assert cfg.weight_decay == 0.0005
        assert cfg.batch_size == 1
        assert cfg.n_epochs == 30
        assert cfg.early_stopping is True  # mapped from "earlystopping"
        assert cfg.es_patience == 10
        assert cfg.es_threshold == 0.001
        assert cfg.test_on_eval is True
        assert cfg.checkpoint_dir == ""


@pytest.fixture
def mocks(config):
    """Patch every heavy collaborator of run_training and wire return values."""
    with (
        patch("eeg_learning.api.jobs.torch.cuda.is_available", return_value=False),
        patch("eeg_learning.api.jobs.load_concat_dataset") as load,
        patch("eeg_learning.api.jobs.DatasetSplitter") as splitter,
        patch("eeg_learning.api.jobs.ModelFactory") as factory,
        patch("eeg_learning.api.jobs.Trainer") as trainer,
        patch("eeg_learning.api.jobs.ModelArtifact") as artifact,
    ):
        windows_ds = MagicMock(name="windows_ds")
        windows_ds.__getitem__.return_value.__getitem__.return_value.shape = (19, 6000)
        load.return_value = windows_ds

        train_set = MagicMock(name="train_set")
        valid_set = MagicMock(name="valid_set")
        test_set = MagicMock(name="test_set")
        splitter.return_value.split_data.return_value = (train_set, valid_set, test_set)

        model = MagicMock(name="model")
        factory.create.return_value = model

        classifier = MagicMock(name="classifier")
        trainer.return_value.fit.return_value = classifier

        art = SimpleNamespace(
            model_id="art_id",
            model_path="/m/art_id.pt",
            manifest_path="/m/art_id.json",
        )
        artifact.save.return_value = art

        yield SimpleNamespace(
            load=load,
            splitter=splitter,
            factory=factory,
            trainer=trainer,
            artifact=artifact,
            windows_ds=windows_ds,
            train_set=train_set,
            valid_set=valid_set,
            model=model,
            classifier=classifier,
            art=art,
        )


class TestRunTraining:
    def test_returns_train_result_from_artifact(self, config, mocks):
        result = run_training(config, windows_path="data/saved_windows", output_dir="out")
        assert isinstance(result, TrainResult)
        assert result.model_id == "art_id"
        assert result.model_path == "/m/art_id.pt"
        assert result.manifest_path == "/m/art_id.json"
        assert result.artifact is mocks.art

    def test_loads_windows_with_expected_args(self, config, mocks):
        run_training(config, windows_path="data/saved_windows", output_dir="out")
        mocks.load.assert_called_once_with(
            path="data/saved_windows", preload=False, target_name="pathological", n_jobs=1
        )

    def test_splitter_wiring(self, config, mocks):
        run_training(config, windows_path="w", output_dir="out")
        args, kwargs = mocks.splitter.call_args
        assert args == (mocks.windows_ds, 0.8, 0.1, 0.1, 87)
        assert kwargs == {"shuffle": True, "remove_attribute": None}
        mocks.splitter.return_value.split_data.assert_called_once_with("train_on_tuab_tueg_test_on_tueg")

    def test_model_built_with_selected_kwargs(self, config, mocks):
        run_training(config, windows_path="w", output_dir="out")
        args, kwargs = mocks.factory.create.call_args
        assert args == ("deep4",)
        assert kwargs == EXPECTED_BUILD_KWARGS

    def test_trainer_fits_model_with_train_and_valid(self, config, mocks):
        run_training(config, windows_path="w", output_dir="out")
        training_config = mocks.trainer.call_args.args[0]
        assert isinstance(training_config, TrainingConfig)
        assert training_config.n_epochs == 30
        assert training_config.early_stopping is True
        mocks.trainer.return_value.fit.assert_called_once_with(mocks.model, mocks.train_set, mocks.valid_set)

    def test_artifact_saved_with_classifier_and_recipe(self, config, mocks):
        run_training(config, windows_path="w", output_dir="out")
        args, kwargs = mocks.artifact.save.call_args
        assert args[0] is mocks.classifier
        assert kwargs["model_name"] == "deep4"
        assert kwargs["build_kwargs"] == EXPECTED_BUILD_KWARGS
        assert kwargs["output_dir"] == "out"
        assert kwargs["training_config"] == asdict(_training_config(config["training"]))

    def test_default_model_id_is_name_and_timestamp(self, config, mocks):
        with patch("eeg_learning.api.jobs.time.strftime", return_value="TS"):
            run_training(config, windows_path="w", output_dir="out")
        assert mocks.artifact.save.call_args.kwargs["model_id"] == "deep4_TS"

    def test_explicit_model_id_passed_through(self, config, mocks):
        run_training(config, windows_path="w", output_dir="out", model_id="custom")
        assert mocks.artifact.save.call_args.kwargs["model_id"] == "custom"
