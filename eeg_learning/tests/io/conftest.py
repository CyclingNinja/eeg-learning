"""Shared fixtures for the io test suite."""

from __future__ import annotations

from unittest.mock import MagicMock

import mne
import numpy as np
import pytest


# Channel names as they appear in a raw BrainVision file before EEGLoader renames them.
RAW_CHANNEL_NAMES = [
    "FP1",
    "FP2",
    "F3",
    "F4",
    "C3",
    "C4",
    "P3",
    "P4",
    "O1",
    "O2",
    "F7",
    "F8",
    "T7",
    "T8",
    "P7",
    "P8",
    "FZ",
    "PZ",
    "FC1",
    "FC2",
    "CP1",
    "CP2",
]


@pytest.fixture
def synthetic_raw():
    """22-channel MNE RawArray matching EEGLoader's expected input channel layout."""
    sfreq = 256.0
    n_times = int(sfreq * 10)
    rng = np.random.default_rng()
    data = rng.standard_normal((len(RAW_CHANNEL_NAMES), n_times)) * 1e-6
    info = mne.create_info(RAW_CHANNEL_NAMES, sfreq=sfreq, ch_types="eeg")
    return mne.io.RawArray(data, info, verbose=False)


@pytest.fixture
def mock_concat_dataset():
    """Mock BaseConcatDataset suitable as input to DatasetBuilder._window (sfreq=100)."""
    sub = MagicMock()
    sub.raw.info = {"sfreq": 100.0}
    recordings = MagicMock()
    recordings.datasets = [sub]
    return recordings


@pytest.fixture
def mock_windows_dataset():
    """Mock windowed dataset returned by create_fixed_length_windows (preload=True)."""
    sub = MagicMock()
    sub.windows.preload = True
    wins = MagicMock()
    wins.datasets = [sub]
    return wins
