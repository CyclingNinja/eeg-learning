"""Unit tests for eeg_learning.pipeline.validation.validate_window_length."""

from __future__ import annotations

import pytest

from eeg_learning.pipeline.validation import validate_window_length


def _cfg(sampling_freq=100, window_len_s=60):
    return {
        "preprocessing": {"sampling_freq": sampling_freq},
        "windowing": {"window_len_s": window_len_s},
    }


def test_matching_window_length_passes():
    # 100 Hz x 60 s = 6000 samples; a matching window must not raise.
    assert validate_window_length(6000, _cfg(100, 60)) is None


def test_mismatched_window_length_raises():
    # The fresh_error.png scenario: config implies 6000 samples but the windows
    # on disk are a different length -> fail fast here instead of crashing later
    # inside NLLLoss with a cryptic shape error.
    with pytest.raises(ValueError, match=r"3000 samples") as excinfo:
        validate_window_length(3000, _cfg(100, 60))
    msg = str(excinfo.value)
    assert "3000" in msg  # actual loaded length
    assert "6000" in msg  # expected length from config


def test_float_config_values_truncated_to_int():
    # sampling_freq / window_len_s may be floats in params.toml; the expected
    # length uses int() truncation, matching create_fixed_length_windows.
    assert validate_window_length(150, _cfg(2.5, 60.0)) is None  # int(2.5 * 60) == 150
    with pytest.raises(ValueError, match=r"151 samples"):
        validate_window_length(151, _cfg(2.5, 60.0))
