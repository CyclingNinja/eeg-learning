"""Shared fixtures for the io test suite."""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest


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
    """Mock windowed dataset as window_recordings returns it (preload=True)."""
    sub = MagicMock()
    sub.windows.preload = True
    wins = MagicMock()
    wins.datasets = [sub]
    return wins
