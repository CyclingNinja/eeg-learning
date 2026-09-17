"""Tests for Trainer and TrainingConfig (eeg_learning/training/trainer.py)."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest
import torch

from eeg_learning.training.trainer import Trainer, TrainingConfig


@pytest.fixture
def model():
    """Stand-in for an instantiated braindecode model."""
    return MagicMock(name="model")


@pytest.fixture
def train_set():
    """Mock windowed dataset exposing get_metadata().target."""
    ds = MagicMock(name="train_set")
    ds.get_metadata.return_value.target = MagicMock(name="targets")
    return ds


@pytest.fixture
def valid_set():
    """Mock windowed validation dataset."""
    return MagicMock(name="valid_set")


class TestTrainingConfig:
    def test_defaults(self):
        cfg = TrainingConfig()
        assert cfg.learning_rate == 1e-3
        assert cfg.n_epochs == 30
        assert cfg.early_stopping is True
        assert cfg.test_on_eval is True
        assert cfg.device is None

    @patch("eeg_learning.training.trainer.torch.cuda.is_available", return_value=False)
    def test_resolve_device_cpu(self, mock_cuda):
        assert TrainingConfig().resolve_device() == "cpu"

    @patch("eeg_learning.training.trainer.torch.cuda.is_available", return_value=True)
    def test_resolve_device_cuda(self, mock_cuda):
        assert TrainingConfig().resolve_device() == "cuda"

    def test_resolve_device_explicit_overrides_autodetect(self):
        assert TrainingConfig(device="cpu").resolve_device() == "cpu"


class TestBuildCallbacks:
    @patch("eeg_learning.training.trainer.EarlyStopping")
    @patch("eeg_learning.training.trainer.Checkpoint")
    @patch("eeg_learning.training.trainer.LRScheduler")
    def test_includes_early_stopping(self, mock_lr, mock_cp, mock_es):
        callbacks = Trainer(TrainingConfig(n_epochs=30, early_stopping=True))._build_callbacks()
        assert callbacks[0] == "accuracy"
        assert callbacks[1][0] == "lr_scheduler"
        assert callbacks[2][0] == "cp"
        assert callbacks[3][0] == "es"
        assert len(callbacks) == 4

    @patch("eeg_learning.training.trainer.EarlyStopping")
    @patch("eeg_learning.training.trainer.Checkpoint")
    @patch("eeg_learning.training.trainer.LRScheduler")
    def test_lr_scheduler_t_max_is_epochs_minus_one(self, mock_lr, mock_cp, mock_es):
        Trainer(TrainingConfig(n_epochs=30))._build_callbacks()
        mock_lr.assert_called_once_with("CosineAnnealingLR", T_max=29)

    @patch("eeg_learning.training.trainer.EarlyStopping")
    @patch("eeg_learning.training.trainer.Checkpoint")
    @patch("eeg_learning.training.trainer.LRScheduler")
    def test_default_patience_is_third_of_epochs(self, mock_lr, mock_cp, mock_es):
        Trainer(TrainingConfig(n_epochs=30, early_stopping=True))._build_callbacks()
        _, kwargs = mock_es.call_args
        assert kwargs["patience"] == 10
        assert kwargs["threshold_mode"] == "rel"

    @patch("eeg_learning.training.trainer.EarlyStopping")
    @patch("eeg_learning.training.trainer.Checkpoint")
    @patch("eeg_learning.training.trainer.LRScheduler")
    def test_es_patience_override_respected(self, mock_lr, mock_cp, mock_es):
        Trainer(TrainingConfig(n_epochs=30, es_patience=5))._build_callbacks()
        _, kwargs = mock_es.call_args
        assert kwargs["patience"] == 5

    @patch("eeg_learning.training.trainer.EarlyStopping")
    @patch("eeg_learning.training.trainer.Checkpoint")
    @patch("eeg_learning.training.trainer.LRScheduler")
    def test_excludes_early_stopping_when_disabled(self, mock_lr, mock_cp, mock_es):
        callbacks = Trainer(TrainingConfig(early_stopping=False))._build_callbacks()
        mock_es.assert_not_called()
        assert len(callbacks) == 3


class TestBuildClassifier:
    @patch("eeg_learning.training.trainer.predefined_split")
    @patch("eeg_learning.training.trainer.weight_function", return_value=torch.tensor([1.0, 1.0]))
    @patch("eeg_learning.training.trainer.EEGClassifier")
    def test_passes_hyperparameters(self, mock_clf, mock_wf, mock_split, model, train_set, valid_set):
        cfg = TrainingConfig(learning_rate=0.002, weight_decay=0.01, batch_size=4, device="cpu")
        Trainer(cfg)._build_classifier(model, train_set, valid_set)
        args, kwargs = mock_clf.call_args
        assert args[0] is model
        assert kwargs["optimizer__lr"] == 0.002
        assert kwargs["optimizer__weight_decay"] == 0.01
        assert kwargs["batch_size"] == 4
        assert kwargs["device"] == "cpu"
        assert kwargs["optimizer"] is torch.optim.AdamW
        # braindecode 1.x models emit logits, so the trainer uses CrossEntropyLoss.
        assert isinstance(kwargs["criterion"], torch.nn.CrossEntropyLoss)

    @patch("eeg_learning.training.trainer.predefined_split", return_value="SPLIT")
    @patch("eeg_learning.training.trainer.weight_function", return_value=torch.tensor([1.0, 1.0]))
    @patch("eeg_learning.training.trainer.EEGClassifier")
    def test_train_split_used_when_test_on_eval(self, mock_clf, mock_wf, mock_split, model, train_set, valid_set):
        Trainer(TrainingConfig(test_on_eval=True, device="cpu"))._build_classifier(model, train_set, valid_set)
        _, kwargs = mock_clf.call_args
        assert kwargs["train_split"] == "SPLIT"
        mock_split.assert_called_once_with(valid_set)

    @patch("eeg_learning.training.trainer.predefined_split")
    @patch("eeg_learning.training.trainer.weight_function", return_value=torch.tensor([1.0, 1.0]))
    @patch("eeg_learning.training.trainer.EEGClassifier")
    def test_no_train_split_when_not_test_on_eval(self, mock_clf, mock_wf, mock_split, model, train_set, valid_set):
        Trainer(TrainingConfig(test_on_eval=False, device="cpu"))._build_classifier(model, train_set, valid_set)
        _, kwargs = mock_clf.call_args
        assert kwargs["train_split"] is None
        mock_split.assert_not_called()

    @patch("eeg_learning.training.trainer.predefined_split")
    @patch("eeg_learning.training.trainer.weight_function")
    @patch("eeg_learning.training.trainer.EEGClassifier")
    def test_unweighted_loss_without_train_set(self, mock_clf, mock_wf, mock_split, model):
        Trainer(TrainingConfig(device="cpu"))._build_classifier(model)
        mock_wf.assert_not_called()
        _, kwargs = mock_clf.call_args
        assert kwargs["criterion"].weight is None


class TestFit:
    @patch("eeg_learning.training.trainer.torch.cuda.is_available", return_value=False)
    def test_fit_trains_and_returns_classifier(self, mock_cuda, model, train_set, valid_set):
        trainer = Trainer(TrainingConfig(n_epochs=7, device="cpu"))
        mock_clf = MagicMock()
        with patch.object(trainer, "_build_classifier", return_value=mock_clf):
            result = trainer.fit(model, train_set, valid_set)
        mock_clf.fit.assert_called_once_with(train_set, y=None, epochs=7)
        assert result is mock_clf


class TestSaveLoad:
    def test_save_creates_parent_and_writes(self, tmp_path):
        clf = MagicMock()
        path = tmp_path / "models" / "params.pt"
        Trainer.save(clf, path)
        assert path.parent.exists()
        clf.save_params.assert_called_once_with(str(path))

    def test_load_initializes_then_loads(self, model):
        trainer = Trainer(TrainingConfig(device="cpu"))
        mock_clf = MagicMock()
        with patch.object(trainer, "_build_classifier", return_value=mock_clf):
            result = trainer.load(model, "/tmp/params.pt")
        mock_clf.initialize.assert_called_once()
        mock_clf.load_params.assert_called_once_with("/tmp/params.pt")
        assert result is mock_clf


class FakeHistory:
    """Minimal stand-in for a skorch ``History``.

    Supports the two access patterns :meth:`Trainer.save_history` uses:
    ``history[-1]`` (last-epoch row dict), ``history[:, col]`` (a column across
    epochs), and truthiness/length by epoch count.
    """

    def __init__(self, rows):
        self._rows = rows

    def __len__(self):
        return len(self._rows)

    def __getitem__(self, key):
        if isinstance(key, tuple):
            row_slice, column = key
            return [row[column] for row in self._rows[row_slice]]
        return self._rows[key]


def _read_csv(path):
    import csv

    with path.open(newline="") as history_file:
        return list(csv.reader(history_file))


class TestSaveHistory:
    def _classifier(self, rows):
        clf = MagicMock()
        clf.history = FakeHistory(rows)
        return clf

    def test_writes_header_and_one_row_per_epoch(self, tmp_path):
        clf = self._classifier(
            [
                {"epoch": 1, "train_loss": 0.9, "valid_loss": 1.0, "train_accuracy": 0.5, "valid_accuracy": 0.4},
                {"epoch": 2, "train_loss": 0.7, "valid_loss": 0.8, "train_accuracy": 0.6, "valid_accuracy": 0.55},
            ]
        )
        path = tmp_path / "history" / "result.csv"
        Trainer.save_history(clf, path)

        rows = _read_csv(path)
        assert rows[0] == ["epoch", "train_loss", "valid_loss", "train_accuracy", "valid_accuracy"]
        assert rows[1] == ["1", "0.9", "1.0", "0.5", "0.4"]
        assert rows[2] == ["2", "0.7", "0.8", "0.6", "0.55"]

    def test_creates_parent_directory(self, tmp_path):
        clf = self._classifier(
            [{"epoch": 1, "train_loss": 0.1, "valid_loss": 0.2, "train_accuracy": 0.9, "valid_accuracy": 0.8}]
        )
        path = tmp_path / "nested" / "dir" / "result.csv"
        Trainer.save_history(clf, path)
        assert path.exists()

    def test_omits_columns_not_recorded(self, tmp_path):
        # No validation split → skorch never logs valid_* keys.
        clf = self._classifier(
            [
                {"epoch": 1, "train_loss": 0.9, "train_accuracy": 0.5},
                {"epoch": 2, "train_loss": 0.7, "train_accuracy": 0.6},
            ]
        )
        path = tmp_path / "result.csv"
        Trainer.save_history(clf, path)

        rows = _read_csv(path)
        assert rows[0] == ["epoch", "train_loss", "train_accuracy"]
        assert rows[1] == ["1", "0.9", "0.5"]

    def test_empty_history_writes_header_only(self, tmp_path):
        clf = self._classifier([])
        path = tmp_path / "result.csv"
        Trainer.save_history(clf, path)

        rows = _read_csv(path)
        assert rows == [["epoch"]]


class TestHistoryRows:
    """The extraction shared by save_history and the train stage's MLflow curves."""

    def _classifier(self, rows):
        clf = MagicMock()
        clf.history = FakeHistory(rows)
        return clf

    def test_returns_one_pair_per_epoch(self):
        clf = self._classifier(
            [
                {"epoch": 1, "train_loss": 0.9, "valid_loss": 1.0, "train_accuracy": 0.5, "valid_accuracy": 0.4},
                {"epoch": 2, "train_loss": 0.7, "valid_loss": 0.8, "train_accuracy": 0.6, "valid_accuracy": 0.55},
            ]
        )
        present, rows = Trainer.history_rows(clf)

        assert present == ["train_loss", "valid_loss", "train_accuracy", "valid_accuracy"]
        assert rows[0] == (1, {"train_loss": 0.9, "valid_loss": 1.0, "train_accuracy": 0.5, "valid_accuracy": 0.4})
        assert rows[1][0] == 2

    def test_omits_columns_not_recorded(self):
        clf = self._classifier([{"epoch": 1, "train_loss": 0.9, "train_accuracy": 0.5}])
        present, rows = Trainer.history_rows(clf)

        assert present == ["train_loss", "train_accuracy"]
        assert rows == [(1, {"train_loss": 0.9, "train_accuracy": 0.5})]

    def test_empty_history_yields_nothing(self):
        present, rows = Trainer.history_rows(self._classifier([]))
        assert present == []
        assert rows == []
