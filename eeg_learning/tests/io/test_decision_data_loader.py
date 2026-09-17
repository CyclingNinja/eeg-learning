"""Tests for decision data loading (eeg_learning/io/decision_data_loader.py)."""

from __future__ import annotations

import csv

import numpy as np
import pandas as pd
import pytest
import torch

from eeg_learning.io.decision_data_loader import (
    DecisionDataLoader,
    DecisionDataset,
)


@pytest.fixture
def sample_csv_content():
    """Sample training_detail.csv rows for testing.

    Four recordings, one window each, for two patients (P01, P02). With the
    default ``start_row=1`` the first row is a skipped header (mirroring the real
    artifact), so the label/probability blocks are one value per row.
    """
    return [
        # Header row (skipped by default start_row=1)
        ["training_detail"],
        # Labels (rows 1-4): one value per recording -> [1, 0, 1, 0]
        ["True"],
        ["False"],
        ["True"],
        ["False"],
        # Probabilities (rows 5-8): aligned with the labels above
        ["0.1"],
        ["0.9"],
        ["0.2"],
        ["0.8"],
        # Valid lengths: cumulative offsets into the flat window list (one window
        # per recording); the parser appends a trailing len(labels)=4.
        ["0", "1", "2", "3"],
        # Patients: one per recording
        ["P01", "P01", "P02", "P02"],
        # Sessions: one per recording
        ["S01", "S02", "S01", "S02"],
    ]


@pytest.fixture
def tmp_csv(sample_csv_content, tmp_path):
    """Create a temporary CSV file for testing."""
    csv_file = tmp_path / "training_detail.csv"
    with open(csv_file, "w", newline="") as f:
        writer = csv.writer(f)
        for row in sample_csv_content:
            writer.writerow(row)
    return csv_file


@pytest.fixture
def structured_detail_dir(tmp_path):
    detail_dir = tmp_path / "training_detail"
    detail_dir.mkdir(parents=True, exist_ok=True)

    windows_df = pd.DataFrame(
        {
            "recording_id": ["rec_000000", "rec_000001", "rec_000002", "rec_000003"],
            "recording_order": [0, 1, 2, 3],
            "split": ["train", "train", "test", "test"],
            "patient_id": ["P01", "P01", "P02", "P02"],
            "session_id": ["S01", "S02", "S01", "S02"],
            "window_index_in_recording": [0, 0, 0, 0],
            "global_window_index": [0, 1, 2, 3],
            "target": [True, False, True, False],
            "prob_abnormal": [0.1, 0.9, 0.2, 0.8],
        }
    )
    recordings_df = pd.DataFrame(
        {
            "recording_id": ["rec_000000", "rec_000001", "rec_000002", "rec_000003"],
            "recording_order": [0, 1, 2, 3],
            "split": ["train", "train", "test", "test"],
            "patient_id": ["P01", "P01", "P02", "P02"],
            "session_id": ["S01", "S02", "S01", "S02"],
            "target": [True, False, True, False],
            "window_count": [1, 1, 1, 1],
            "window_start_global": [0, 1, 2, 3],
        }
    )

    windows_df.to_parquet(detail_dir / "windows.parquet", index=False)
    recordings_df.to_parquet(detail_dir / "recordings.parquet", index=False)
    return detail_dir


class TestDecisionDataset:
    """Tests for DecisionDataset PyTorch wrapper."""

    @pytest.fixture
    def dataset(self):
        data = [
            np.array([0.1, 0.2, 0.15, 0.18]),
            np.array([0.9, 0.8, 0.85, 0.82]),
        ]
        labels = [0, 1]
        valid_lens = [4, 4]
        return DecisionDataset(data, labels, valid_lens)

    def test_len(self, dataset):
        assert len(dataset) == 2

    def test_getitem_returns_tuple(self, dataset):
        item = dataset[0]
        assert isinstance(item, tuple)
        assert len(item) == 3

    def test_getitem_tensor_type(self, dataset):
        data, label, valid_len = dataset[0]
        assert isinstance(data, torch.Tensor)
        assert isinstance(label, int)
        assert isinstance(valid_len, int)

    def test_getitem_values(self, dataset):
        data, label, valid_len = dataset[0]
        assert data.shape == (4,)
        assert label == 0
        assert valid_len == 4

    def test_data_dtype_float32(self, dataset):
        data, _, _ = dataset[0]
        assert data.dtype == torch.float32


class TestDecisionDataLoader:
    """Tests for DecisionDataLoader CSV parser."""

    def test_parse_csv_basic(self, tmp_csv):
        loader = DecisionDataLoader(tmp_csv)
        assert len(loader.labels) == 4
        assert loader.labels == [1, 0, 1, 0]
        assert len(loader.data) == 4
        assert len(loader.patients) == 4
        assert len(loader.sessions) == 4

    def test_parse_csv_values(self, tmp_csv):
        loader = DecisionDataLoader(tmp_csv)
        # Check data values are float
        assert all(isinstance(d, float) for d in loader.data)
        # Valid lengths are cumulative offsets into the flat window list:
        # 4 recordings (one window each) plus the appended total length.
        assert loader.valid_lens == [0, 1, 2, 3, 4]

    def test_parse_csv_patient_sessions(self, tmp_csv):
        loader = DecisionDataLoader(tmp_csv)
        assert loader.patients == ["P01", "P01", "P02", "P02"]
        assert loader.sessions == ["S01", "S02", "S01", "S02"]

    def test_parse_structured_artifact(self, structured_detail_dir):
        loader = DecisionDataLoader(structured_detail_dir)
        assert loader.labels == [1, 0, 1, 0]
        assert loader.valid_lens == [0, 1, 2, 3, 4]
        assert loader.patients == ["P01", "P01", "P02", "P02"]
        assert loader.sessions == ["S01", "S02", "S01", "S02"]

    def test_stats_method(self, tmp_csv):
        loader = DecisionDataLoader(tmp_csv)
        stats = loader.stats()
        assert stats["n_recordings"] == 4
        assert stats["n_patients"] == 2
        # 4 distinct patient+session combos: P01-S01, P01-S02, P02-S01, P02-S02
        assert stats["n_sessions"] == 4
        assert stats["n_labels"] == 4
        assert stats["n_positive"] == 2
        assert abs(stats["positive_ratio"] - 0.5) < 1e-5

    def test_create_histogram_basic(self, tmp_csv):
        loader = DecisionDataLoader(tmp_csv)
        raw = [0.1, 0.2, 0.15, 0.18, 0.25, 0.3]
        hist = loader.create_histogram(raw, length=10)

        assert hist.shape == (10,)
        assert np.allclose(hist.sum(), 1.0)  # Normalized

    def test_create_histogram_bins(self, tmp_csv):
        loader = DecisionDataLoader(tmp_csv)
        raw = [0.05, 0.15, 0.25, 0.35, 0.45, 0.55, 0.65, 0.75, 0.85, 0.95]
        hist = loader.create_histogram(raw, length=10)

        assert hist.shape == (10,)
        # Each bin should have ~1 count (10 values / 10 bins), normalized to 0.1
        assert np.allclose(hist, 0.1, atol=0.05)

    def test_load_per_recording(self, tmp_csv):
        loader = DecisionDataLoader(tmp_csv)
        dataset = loader.load(aggregation=None, length=10)

        assert isinstance(dataset, DecisionDataset)
        assert len(dataset) == 4
        data, _, _ = dataset[0]
        assert data.shape == (10,)  # Histogram of length 10

    def test_load_per_recording_with_hybrid(self, tmp_csv):
        loader = DecisionDataLoader(tmp_csv)
        dataset = loader.load(aggregation=None, length=10, use_hybrid=True)

        data, _, _ = dataset[0]
        assert data.shape == (30,)  # 10 histogram + 20 padded raw

    def test_load_by_patients(self, tmp_csv):
        loader = DecisionDataLoader(tmp_csv)
        dataset = loader.load(aggregation="patients", length=10)

        # 2 unique patients
        assert len(dataset) == 2
        data, _, _ = dataset[0]
        assert data.shape == (10,)

    def test_load_by_sessions(self, tmp_csv):
        loader = DecisionDataLoader(tmp_csv)
        dataset = loader.load(aggregation="sessions", length=10)

        # P01-S01, P01-S02, P02-S01, P02-S02 (4 unique combinations)
        assert len(dataset) == 4

    def test_load_n_recordings_limit(self, tmp_csv):
        loader = DecisionDataLoader(tmp_csv)
        dataset = loader.load(aggregation=None, length=10, n_recordings=2)

        assert len(dataset) == 2

    def test_remove_empty(self, tmp_csv):
        loader = DecisionDataLoader(tmp_csv)
        result = loader._remove_empty(["a", "", "b", "", "c"])
        assert result == ["a", "b", "c"]

    def test_aggregate_by_criterion(self, tmp_csv):
        loader = DecisionDataLoader(tmp_csv)
        data_list, labels, valid_lens = loader.aggregate_by_criterion(loader.patients, length=10)

        assert len(data_list) == 2  # 2 unique patients
        assert len(labels) == 2
        assert len(valid_lens) == 2
        assert all(isinstance(d, np.ndarray) for d in data_list)

    def test_aggregate_with_hybrid(self, tmp_csv):
        loader = DecisionDataLoader(tmp_csv)
        data_list, _, _ = loader.aggregate_by_criterion(loader.patients, length=10, use_hybrid=True)

        assert data_list[0].shape == (30,)  # Hybrid: 10 + 20

    def test_custom_csv_config(self, tmp_path):
        """Test with custom start_row and n_rows.

        For ``block=0`` the label and probability blocks are contiguous;
        ``row_gap`` only spaces successive blocks apart, so it is inert here.
        """
        csv_file = tmp_path / "custom.csv"
        rows = [
            # Header row (skipped by start_row=1)
            ["header"],
            # Labels (start_row=1, n_rows=2): one value per row
            ["True"],
            ["False"],
            # Probabilities (contiguous with labels for block=0)
            ["0.1"],
            ["0.9"],
            # Valid lengths: cumulative offsets
            ["0", "1"],
            # Patients
            ["P01", "P02"],
            # Sessions
            ["S01", "S01"],
        ]
        with open(csv_file, "w", newline="") as f:
            writer = csv.writer(f)
            for row in rows:
                writer.writerow(row)

        loader = DecisionDataLoader(csv_file, start_row=1, n_rows=2, row_gap=3, block=0)
        assert len(loader.labels) == 2
        assert len(loader.data) == 2


class TestDecisionDataIntegration:
    """Integration tests combining loader and dataset."""

    def test_end_to_end_loading(self, tmp_csv):
        loader = DecisionDataLoader(tmp_csv)
        dataset = loader.load(aggregation="patients", length=10)

        # Can iterate through dataset
        for i in range(len(dataset)):
            data, label, valid_len = dataset[i]
            assert isinstance(data, torch.Tensor)
            assert isinstance(label, int)
            assert isinstance(valid_len, int)

    def test_dataloader_integration(self, tmp_csv):
        loader = DecisionDataLoader(tmp_csv)
        dataset = loader.load(aggregation=None, length=10)

        batch_loader = torch.utils.data.DataLoader(dataset, batch_size=2, shuffle=False)
        batch_data, batch_labels, batch_lens = next(iter(batch_loader))

        assert batch_data.shape == (2, 10)
        assert len(batch_labels) == 2
        assert len(batch_lens) == 2
