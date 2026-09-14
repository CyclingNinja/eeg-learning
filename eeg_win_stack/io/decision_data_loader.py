"""Data loading and aggregation for second-stage decision models.

Handles CSV parsing from training_detail.csv (first-stage model output),
histogram generation, and patient/session-level aggregation.
"""

from __future__ import annotations

import csv
from pathlib import Path
from typing import Literal

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

from eeg_preprocessing.tools.paths import findall


class DecisionDataset(Dataset):
    """PyTorch Dataset for decision-stage training.

    Holds aggregated data, labels, and valid sequence lengths.
    """

    def __init__(
        self,
        data: list[np.ndarray],
        labels: list[int],
        valid_lens: list[int],
    ):
        """Initialize dataset.

        Parameters
        ----------
        data : list[np.ndarray]
            List of feature arrays (histogram or hybrid features).
        labels : list[int]
            List of binary labels (0 or 1).
        valid_lens : list[int]
            List of valid sequence lengths for each sample.
        """
        self.data = torch.tensor(np.asarray(data), dtype=torch.float32)
        self.labels = labels
        self.valid_lens = valid_lens

    def __len__(self) -> int:
        """Return dataset size."""
        return len(self.data)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, int, int]:
        """Return (data, label, valid_len) for index."""
        return self.data[index], self.labels[index], self.valid_lens[index]


class DecisionDataLoader:
    """Load and aggregate first-stage predictions.

    Supports the structured ``training_detail/`` artifact directory as the
    canonical format, and the legacy ``training_detail.csv`` layout as a
    compatibility fallback.
    """

    def __init__(
        self,
        csv_path: str | Path,
        start_row: int = 1,
        n_rows: int = 4,
        row_gap: int = 4,
        block: int = 0,
    ):
        """Initialize data loader.

        Parameters
        ----------
        csv_path : str or Path
            Path to a structured training_detail directory, a structured parquet
            file, or a legacy training_detail.csv file.
        start_row : int, default=1
            Starting row index for label block.
        n_rows : int, default=4
            Number of rows containing labels.
        row_gap : int, default=4
            Gap between label block and probability block.
        block : int, default=0
            Which block of results to use (for multiple experiment blocks).
        """
        self.csv_path = Path(csv_path)
        self.start_row = start_row
        self.n_rows = n_rows
        self.row_gap = row_gap
        self.block = block

        # Parse structured artifact or legacy CSV.
        (
            self.labels,
            self.data,
            self.valid_lens,
            self.patients,
            self.sessions,
        ) = self._parse_source()

    def _parse_source(self) -> tuple[list[int], list[float], list[int], list[str], list[str]]:
        if self.csv_path.is_dir() or self.csv_path.suffix.lower() == ".parquet":
            return self._parse_structured()
        return self._parse_csv()

    @staticmethod
    def _to_binary_label(value) -> int:
        if isinstance(value, bool):
            return int(value)
        if isinstance(value, (int, np.integer)):
            return int(value != 0)
        text = str(value).strip().lower()
        return int(text in {"true", "1"})

    def _parse_structured(self) -> tuple[list[int], list[float], list[int], list[str], list[str]]:
        if self.csv_path.is_dir():
            windows_path = self.csv_path / "windows.parquet"
            recordings_path = self.csv_path / "recordings.parquet"
        else:
            windows_path = self.csv_path
            recordings_path = self.csv_path.with_name("recordings.parquet")

        if not windows_path.exists():
            raise FileNotFoundError(f"Structured training detail file not found: {windows_path}")

        windows_df = pd.read_parquet(windows_path)

        required_window_columns = {"recording_id", "window_index_in_recording", "target", "prob_abnormal"}
        missing_window_columns = required_window_columns - set(windows_df.columns)
        if missing_window_columns:
            raise ValueError(
                f"Structured training detail missing required window columns: {sorted(missing_window_columns)}"
            )

        if recordings_path.exists():
            recordings_df = pd.read_parquet(recordings_path)
        else:
            recordings_df = (
                windows_df[["recording_id"]]
                .drop_duplicates()
                .assign(recording_order=lambda df: np.arange(len(df), dtype=int))
            )

        if "recording_order" in recordings_df.columns:
            recordings_df = recordings_df.sort_values("recording_order")
        elif "window_start_global" in recordings_df.columns:
            recordings_df = recordings_df.sort_values("window_start_global")

        pd_labels = []
        pd_data = []
        pd_valid_lens = []
        patients = []
        sessions = []
        cursor = 0

        for recording in recordings_df.to_dict("records"):
            recording_id = recording.get("recording_id")
            rec_windows = windows_df[windows_df["recording_id"] == recording_id].sort_values(
                "window_index_in_recording"
            )
            if rec_windows.empty:
                continue

            pd_valid_lens.append(cursor)
            labels_segment = [self._to_binary_label(v) for v in rec_windows["target"].tolist()]
            probs_segment = [float(v) for v in rec_windows["prob_abnormal"].tolist()]

            pd_labels.extend(labels_segment)
            pd_data.extend(probs_segment)
            cursor += len(labels_segment)

            patients.append(str(recording.get("patient_id", "")))
            sessions.append(str(recording.get("session_id", "")))

        pd_valid_lens.append(len(pd_labels))
        return pd_labels, pd_data, pd_valid_lens, patients, sessions

    @staticmethod
    def _is_label_row(row: list[str]) -> bool:
        values = DecisionDataLoader._remove_empty(row)
        if not values:
            return False
        return all(value in {"True", "TRUE", "False", "FALSE"} for value in values)

    @staticmethod
    def _is_int_row(row: list[str]) -> bool:
        values = DecisionDataLoader._remove_empty(row)
        if not values:
            return False
        try:
            [int(value) for value in values]
        except ValueError:
            return False
        return True

    @staticmethod
    def _is_float_row(row: list[str]) -> bool:
        values = DecisionDataLoader._remove_empty(row)
        if not values:
            return False
        try:
            [float(value) for value in values]
        except ValueError:
            return False
        return True

    def _parse_csv(
        self,
    ) -> tuple[list[int], list[float], list[int], list[str], list[str]]:
        """Parse legacy training_detail.csv into component arrays.

        Returns
        -------
        tuple
            (labels, data, valid_lens, patients, sessions)
        """
        with open(self.csv_path, newline="") as csvfile:
            rows = [self._remove_empty(row) for row in csv.reader(csvfile, delimiter=",")]

        index = self.start_row
        current_block = 0
        while index < len(rows):
            while index < len(rows) and not rows[index]:
                index += 1
            if index >= len(rows):
                break

            labels = []
            while index < len(rows) and self._is_label_row(rows[index]):
                labels.extend(rows[index])
                index += 1

            if not labels:
                index += 1
                continue

            probabilities = []
            while index < len(rows) and self._is_float_row(rows[index]) and not self._is_int_row(rows[index]):
                probabilities.extend(rows[index])
                index += 1

            if index + 2 >= len(rows):
                break

            if current_block == self.block:
                pd_labels = labels
                pd_data = probabilities
                pd_valid_lens = rows[index]
                patients = rows[index + 1]
                sessions = rows[index + 2]
                break

            current_block += 1
            index += 3
        else:
            pd_labels = []
            pd_data = []
            pd_valid_lens = []
            patients = []
            sessions = []

        # Add final valid length
        pd_valid_lens.append(len(pd_labels))

        # Convert types
        pd_valid_lens = [int(v) for v in pd_valid_lens]
        pd_data = [float(d) for d in pd_data]
        pd_labels = [self._to_binary_label(label) for label in pd_labels]

        return pd_labels, pd_data, pd_valid_lens, patients, sessions

    @staticmethod
    def _remove_empty(row: list[str]) -> list[str]:
        """Remove empty strings from row."""
        return [x for x in row if x != ""]

    @staticmethod
    def _pad_raw(raw: list[float], width: int = 20) -> list[float]:
        """Right-pad (or truncate) raw probabilities to a fixed ``width``.

        Gives the raw-probability feature a fixed size so batches collate and
        ``DecisionModel``'s ``Linear(20, ...)`` receives a consistent input.
        """
        return (list(raw) + [0] * (width - len(raw)))[:width]

    def _build_feature(
        self,
        raw_segment: list[float],
        length: int,
        use_his: bool,
        use_hybrid: bool,
    ) -> np.ndarray:
        """Build a single decision feature from a recording's raw probabilities.

        - ``use_hybrid``: histogram concatenated with padded raw (``HistogramModel``).
        - ``use_his``: histogram only (``HistogramModel``).
        - otherwise: padded raw probabilities (``DecisionModel``).
        """
        if use_hybrid:
            hist = self.create_histogram(raw_segment, length=length)
            return np.concatenate([hist, self._pad_raw(raw_segment)])
        if use_his:
            return self.create_histogram(raw_segment, length=length)
        return np.asarray(self._pad_raw(raw_segment), dtype=float)

    def create_histogram(
        self,
        raw: list[float],
        length: int = 10,
    ) -> np.ndarray:
        """Generate histogram from raw probabilities.

        Parameters
        ----------
        raw : list[float]
            Raw probability values.
        length : int, default=10
            Number of histogram bins.

        Returns
        -------
        np.ndarray
            Normalized histogram of shape (length,).
        """
        hist = np.zeros(length)
        for val in raw:
            bin_idx = int(val // (1 / length + 0.001))
            bin_idx = min(bin_idx, length - 1)  # Clamp to last bin
            hist[bin_idx] += 1
        return hist / (np.sum(hist) + 1e-8)  # Normalize

    def aggregate_by_criterion(
        self,
        criterion: list[str],
        length: int = 10,
        use_his: bool = True,
        use_hybrid: bool = False,
    ) -> tuple[list[np.ndarray], list[int], list[int]]:
        """Aggregate data by criterion (patient or session).

        Parameters
        ----------
        criterion : list[str]
            List of criterion values (e.g., patient IDs or session IDs).
        length : int, default=10
            Histogram bins.
        use_his : bool, default=True
            If True, build histogram features; otherwise padded raw probabilities.
        use_hybrid : bool, default=False
            If True, concatenate histogram with padded raw data.

        Returns
        -------
        tuple
            (data_list, labels, valid_lens)
        """
        data_list = []
        labels = []
        valid_lens = []

        # First-appearance order, not ``set`` order: set iteration over strings is
        # salted per process, which would otherwise reshuffle the dataset between
        # runs and make ``fix_testset`` splits irreproducible.
        for criterion_val in dict.fromkeys(criterion):
            indexes = findall(criterion, criterion_val)
            data_pa = []
            valid_len = 0

            for idx in indexes:
                data_pa += self.data[self.valid_lens[idx] : self.valid_lens[idx + 1]]
                valid_len += self.valid_lens[idx + 1] - self.valid_lens[idx]

            data_list.append(self._build_feature(data_pa, length, use_his, use_hybrid))
            valid_lens.append(valid_len)
            labels.append(self.labels[self.valid_lens[indexes[0]]])

        return data_list, labels, valid_lens

    def load(
        self,
        aggregation: Literal["patients", "sessions", None] = None,
        length: int = 10,
        use_his: bool = True,
        use_hybrid: bool = False,
        n_recordings: int | None = None,
    ) -> DecisionDataset:
        """Load and aggregate data into a DecisionDataset.

        Parameters
        ----------
        aggregation : {"patients", "sessions", None}, default=None
            How to group data:
            - "patients": Aggregate per patient
            - "sessions": Aggregate per session
            - None: Use per-recording data
        length : int, default=10
            Histogram bins.
        use_his : bool, default=True
            If True, features are histograms (for ``HistogramModel``); if False,
            features are padded raw probabilities (for ``DecisionModel``).
        use_hybrid : bool, default=False
            Concatenate histogram with padded raw data.
        n_recordings : int or None
            If aggregation is None, limit to first n_recordings.
            If None, use all recordings.

        Returns
        -------
        DecisionDataset
            PyTorch Dataset ready for training.
        """
        data_list = []
        labels = []
        valid_lens = []

        # Raw features pair with DecisionModel, which is only selected when not
        # aggregating; any aggregated run uses HistogramModel, so force histograms
        # there to keep the feature width matched to the model.
        use_his = use_his or aggregation is not None

        if aggregation == "patients":
            data_list, labels, valid_lens = self.aggregate_by_criterion(
                self.patients,
                length=length,
                use_his=use_his,
                use_hybrid=use_hybrid,
            )
        elif aggregation == "sessions":
            # Create session identifiers
            sessions_patients = [str(p) + str(s) for p, s in zip(self.patients, self.sessions)]
            data_list, labels, valid_lens = self.aggregate_by_criterion(
                sessions_patients,
                length=length,
                use_his=use_his,
                use_hybrid=use_hybrid,
            )
        else:  # None: per-recording
            n_use = n_recordings if n_recordings else len(self.valid_lens) - 1
            for i in range(n_use):
                valid_len = self.valid_lens[i + 1] - self.valid_lens[i]
                valid_lens.append(valid_len)
                labels.append(self.labels[self.valid_lens[i]])

                raw_segment = self.data[self.valid_lens[i] : self.valid_lens[i + 1]]
                data_list.append(self._build_feature(raw_segment, length, use_his, use_hybrid))

        return DecisionDataset(data_list, labels, valid_lens)

    def stats(self) -> dict:
        """Return dataset statistics.

        Returns
        -------
        dict
            Summary statistics about the parsed data.
        """
        n_recordings = len(self.valid_lens) - 1
        n_patients = len(set(self.patients))
        sessions_patients = [str(p) + str(s) for p, s in zip(self.patients, self.sessions)]
        n_sessions = len(set(sessions_patients))
        n_positive = sum(self.labels)
        pos_ratio = n_positive / len(self.labels) if self.labels else 0

        return {
            "n_recordings": n_recordings,
            "n_patients": n_patients,
            "n_sessions": n_sessions,
            "n_labels": len(self.labels),
            "n_positive": n_positive,
            "positive_ratio": pos_ratio,
        }
