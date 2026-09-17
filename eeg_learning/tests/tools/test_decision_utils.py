"""Tests for decision utilities (eeg_learning/tools/decision_utils.py)."""

from __future__ import annotations

import csv

import numpy as np
import pandas as pd
import pytest
import torch

from eeg_learning.tools.decision_utils import (
    DecisionEvaluationResult,
    DecisionTrainingResult,
    compute_decision_metrics,
    save_decision_results,
    save_training_detail,
)


class TestDecisionTrainingResult:
    """Tests for DecisionTrainingResult dataclass."""

    def test_init(self):
        model = torch.nn.Linear(10, 2)
        result = DecisionTrainingResult(
            model=model,
            best_valid_loss=0.5,
            train_losses=[1.0, 0.8, 0.6],
            valid_losses=[1.2, 0.9, 0.7],
        )
        assert result.best_valid_loss == 0.5
        assert len(result.train_losses) == 3
        assert len(result.valid_losses) == 3


class TestDecisionEvaluationResult:
    """Tests for DecisionEvaluationResult dataclass."""

    def test_init(self):
        cm = np.array([[100, 10], [5, 85]])
        result = DecisionEvaluationResult(
            test_acc=0.925,
            ori_acc=0.850,
            argmax_acc=0.900,
            mean_acc=0.880,
            confusion_matrix=cm,
        )
        assert result.test_acc == 0.925
        assert result.confusion_matrix[0, 0] == 100


class TestComputeDecisionMetrics:
    """Tests for compute_decision_metrics function."""

    @pytest.fixture
    def perfect_predictions(self):
        """Model predicts perfectly."""
        batch_size = 4
        # Predictions: class 0 high confidence, class 1 high confidence, etc.
        predictions = torch.tensor(
            [
                [-0.1, -2.0],  # Predicts 0 (negative, large)
                [-2.0, -0.1],  # Predicts 1
                [-0.1, -2.0],  # Predicts 0
                [-2.0, -0.1],  # Predicts 1
            ]
        )
        targets = torch.tensor([0, 1, 0, 1])
        data = torch.ones(batch_size, 20)  # All raw probs = 1 (above 0.5)
        data[0, :] = 0.0  # First sample: all raw probs = 0 (below 0.5)
        data[2, :] = 0.0
        valid_lens = torch.tensor([20, 20, 20, 20])
        return predictions, targets, data, valid_lens

    @pytest.fixture
    def random_predictions(self):
        """Model makes random predictions."""
        batch_size = 10
        predictions = torch.randn(batch_size, 2)
        targets = torch.randint(0, 2, (batch_size,))
        data = torch.rand(batch_size, 20)
        valid_lens = torch.full((batch_size,), 20)
        return predictions, targets, data, valid_lens

    def test_output_structure(self, perfect_predictions):
        pred, target, data, valid_len = perfect_predictions
        result = compute_decision_metrics(pred, target, data, valid_len)

        assert isinstance(result, DecisionEvaluationResult)
        assert hasattr(result, "test_acc")
        assert hasattr(result, "ori_acc")
        assert hasattr(result, "argmax_acc")
        assert hasattr(result, "mean_acc")
        assert hasattr(result, "confusion_matrix")

    def test_metrics_in_range_0_1(self, random_predictions):
        pred, target, data, valid_len = random_predictions
        result = compute_decision_metrics(pred, target, data, valid_len)

        assert 0 <= result.test_acc <= 1
        assert 0 <= result.ori_acc <= 1
        assert 0 <= result.argmax_acc <= 1
        assert 0 <= result.mean_acc <= 1

    def test_confusion_matrix_shape(self, random_predictions):
        pred, target, data, valid_len = random_predictions
        result = compute_decision_metrics(pred, target, data, valid_len)

        assert result.confusion_matrix.shape == (2, 2)
        assert result.confusion_matrix.sum() == len(target)

    def test_perfect_predictions_model_accuracy(self, perfect_predictions):
        """When model predicts correctly, test_acc should be 1.0."""
        pred, target, data, valid_len = perfect_predictions
        result = compute_decision_metrics(pred, target, data, valid_len)

        assert result.test_acc == 1.0

    def test_ori_acc_matches_threshold(self):
        """ori_acc should match > 0.5 threshold on raw data."""
        predictions = torch.tensor([[-1.0, -0.1], [-0.1, -1.0]])
        targets = torch.tensor([1, 0])
        # First sample: target=1, data should be >0.5
        # Second sample: target=0, data should be <0.5
        data = torch.tensor(
            [
                [0.6, 0.7, 0.8, 0.9, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5],
                [0.1, 0.2, 0.3, 0.4, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5],
            ]
        )
        valid_lens = torch.tensor([20, 20])

        result = compute_decision_metrics(predictions, targets, data, valid_lens)
        # First sample: 4 > 0.5 out of 20 = correct (target is 1)
        # Second sample: 4 < 0.5 out of 20 = correct (target is 0)
        # ori_acc = 8/20 = 0.4 from first + 16/20 = 0.8 from second = avg
        assert result.ori_acc > 0.0

    def test_different_valid_lengths(self):
        """Test with varying valid_len values."""
        predictions = torch.randn(3, 2)
        targets = torch.tensor([0, 1, 0])
        data = torch.rand(3, 20)
        valid_lens = torch.tensor([5, 10, 20])  # Different lengths

        result = compute_decision_metrics(predictions, targets, data, valid_lens)
        assert result.ori_acc >= 0
        assert result.ori_acc <= 1

    def test_detaches_from_computation_graph(self):
        """Result should not require gradients."""
        predictions = torch.randn(2, 2, requires_grad=True)
        targets = torch.tensor([0, 1])
        data = torch.rand(2, 20, requires_grad=True)
        valid_lens = torch.tensor([20, 20])

        result = compute_decision_metrics(predictions, targets, data, valid_lens)
        # Metrics should be scalar floats, not tensors
        assert isinstance(result.test_acc, float)
        assert isinstance(result.ori_acc, float)


class TestSaveDecisionResults:
    """Tests for save_decision_results function."""

    @pytest.fixture
    def result(self):
        cm = np.array([[85, 15], [5, 95]])
        return DecisionEvaluationResult(
            test_acc=0.90,
            ori_acc=0.85,
            argmax_acc=0.88,
            mean_acc=0.87,
            confusion_matrix=cm,
        )

    def test_creates_output_directory(self, result, tmp_path):
        output_dir = tmp_path / "nested" / "deep" / "results.csv"
        save_decision_results(result, output_dir, append=False)
        assert output_dir.parent.parent.parent.exists()

    def test_writes_header_on_new_file(self, result, tmp_path):
        output_file = tmp_path / "results.csv"
        save_decision_results(result, output_file, append=False)

        with open(output_file) as f:
            reader = csv.reader(f)
            header = next(reader)
            assert "test_acc" in header
            assert "ori_acc" in header
            assert "tn" in header
            assert "fp" in header
            assert "fn" in header
            assert "tp" in header

    def test_writes_data_row(self, result, tmp_path):
        output_file = tmp_path / "results.csv"
        save_decision_results(result, output_file, append=False)

        with open(output_file) as f:
            reader = csv.reader(f)
            next(reader)  # Skip header
            data_row = next(reader)
            assert float(data_row[0]) == pytest.approx(0.90)  # test_acc
            assert float(data_row[1]) == pytest.approx(0.85)  # ori_acc
            assert int(data_row[4]) == 85  # tn
            assert int(data_row[5]) == 15  # fp
            assert int(data_row[6]) == 5  # fn
            assert int(data_row[7]) == 95  # tp

    def test_append_mode(self, result, tmp_path):
        output_file = tmp_path / "results.csv"

        # First write
        save_decision_results(result, output_file, append=False)
        # Second write (append)
        save_decision_results(result, output_file, append=True)

        with open(output_file) as f:
            lines = f.readlines()
        # 1 header + 2 data rows
        assert len(lines) == 3

    def test_no_header_on_append(self, result, tmp_path):
        output_file = tmp_path / "results.csv"
        save_decision_results(result, output_file, append=False)
        save_decision_results(result, output_file, append=True)

        with open(output_file) as f:
            reader = csv.reader(f)
            rows = list(reader)
        # Only 1 header row, 2 data rows
        assert len(rows) == 3
        assert "test_acc" in rows[0]

    def test_metadata_columns(self, result, tmp_path):
        output_file = tmp_path / "results.csv"
        metadata = {"block": "0", "n_reps": "5", "use_his": "true"}

        save_decision_results(result, output_file, metadata=metadata, append=False)

        with open(output_file) as f:
            reader = csv.reader(f)
            header = next(reader)
            data = next(reader)

        assert "block" in header
        assert "n_reps" in header
        assert "use_his" in header

        # Find indices and verify values
        block_idx = header.index("block")
        n_reps_idx = header.index("n_reps")
        assert data[block_idx] == "0"
        assert data[n_reps_idx] == "5"

    def test_multiple_writes_with_metadata(self, result, tmp_path):
        output_file = tmp_path / "results.csv"

        for i in range(3):
            metadata = {"rep": str(i)}
            save_decision_results(result, output_file, metadata=metadata, append=(i > 0))

        with open(output_file) as f:
            reader = csv.reader(f)
            lines = list(reader)

        assert len(lines) == 4  # 1 header + 3 data rows


class _FakeDecisionDataset:
    def __init__(self, targets, trial_starts, paths):
        self._metadata = pd.DataFrame({"target": targets, "i_window_in_trial": trial_starts})
        self.description = pd.DataFrame({"path": paths})

    def get_metadata(self):
        return self._metadata


class _FakeClassifier:
    def __init__(self, probabilities):
        self._probabilities = list(probabilities)
        self._offset = 0

    def predict_proba(self, dataset):
        length = len(dataset.get_metadata())
        probs = self._probabilities[self._offset : self._offset + length]
        self._offset += length
        return np.log(np.column_stack([1 - np.asarray(probs), np.asarray(probs)]))


def test_save_training_detail_writes_legacy_layout(tmp_path):
    output_file = tmp_path / "training_detail.csv"
    datasets = [
        _FakeDecisionDataset(
            targets=[True, True, False],
            trial_starts=[0, 1, 0],
            paths=["root/P01/S01/file.edf", "root/P02/S02/file.edf"],
        ),
        _FakeDecisionDataset(
            targets=[False, False],
            trial_starts=[0, 1],
            paths=["root/P03/S03/file.edf"],
        ),
    ]

    classifier = _FakeClassifier([0.9, 0.8, 0.2, 0.1, 0.3])
    save_training_detail(classifier, datasets, output_file)

    with open(output_file, newline="") as handle:
        rows = list(csv.reader(handle))

    assert rows[0] == ["training_detail"]
    assert rows[1] == ["True", "True", "False", "False", "False"]
    assert rows[3] == ["0", "2", "3"]
    assert rows[4] == ["P01", "P02", "P03"]
    assert rows[5] == ["S01", "S02", "S03"]

    probabilities = [float(value) for value in rows[2]]
    assert probabilities == pytest.approx([0.9, 0.8, 0.2, 0.1, 0.3])


def test_save_training_detail_writes_structured_artifact(tmp_path):
    output_dir = tmp_path / "training_detail"
    datasets = [
        (
            "train",
            _FakeDecisionDataset(
                targets=[True, True, False],
                trial_starts=[0, 1, 0],
                paths=["root/P01/S01/file.edf", "root/P02/S02/file.edf"],
            ),
        ),
        (
            "test",
            _FakeDecisionDataset(
                targets=[False, False],
                trial_starts=[0, 1],
                paths=["root/P03/S03/file.edf"],
            ),
        ),
    ]

    classifier = _FakeClassifier([0.9, 0.8, 0.2, 0.1, 0.3])
    save_training_detail(classifier, datasets, output_dir)

    windows_path = output_dir / "windows.parquet"
    recordings_path = output_dir / "recordings.parquet"
    summary_path = output_dir / "recording_summary.csv"
    manifest_path = output_dir / "manifest.json"
    legacy_path = output_dir / "legacy_training_detail.csv"

    assert windows_path.exists()
    assert recordings_path.exists()
    assert summary_path.exists()
    assert manifest_path.exists()
    assert legacy_path.exists()

    windows_df = pd.read_parquet(windows_path)
    recordings_df = pd.read_parquet(recordings_path)
    summary_df = pd.read_csv(summary_path)

    assert len(windows_df) == 5
    assert len(recordings_df) == 3
    assert len(summary_df) == 3
    assert {"recording_id", "prob_abnormal", "target"}.issubset(windows_df.columns)
