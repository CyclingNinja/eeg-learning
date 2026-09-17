"""Tests for the config loader (eeg_learning/config/loader.py)."""

from __future__ import annotations

import pytest
from tomllib import TOMLDecodeError

from eeg_learning.config.loader import load


class TestDefaultParams:
    """load() with no args must resolve the params.toml bundled next to loader.py.

    Guards against the file being moved out of eeg_learning/config/ or
    Path(__file__).parent resolution breaking (see the DVC params-path note).
    """

    def test_default_path_returns_populated_dict(self):
        cfg = load()
        assert isinstance(cfg, dict)
        assert cfg  # not empty

    def test_default_has_expected_top_level_tables(self):
        cfg = load()
        assert {"run", "data"} <= cfg.keys()

    def test_default_anchor_value(self):
        # A single anchor keeps this from becoming a change-detector for every
        # hyperparameter tweak, while still proving values are actually parsed.
        cfg = load()
        assert cfg["run"]["random_state"] == 87


class TestRoundTrip:
    def test_preserves_native_types(self, tmp_path):
        params = tmp_path / "params.toml"
        params.write_text('[data]\nn_tuab = 2993\nfactor_new = 1e-3\npreload = false\nchannels = ["C3", "C4"]\n')
        cfg = load(params)
        assert cfg["data"]["n_tuab"] == 2993
        assert cfg["data"]["factor_new"] == pytest.approx(1e-3)
        assert cfg["data"]["preload"] is False
        assert cfg["data"]["channels"] == ["C3", "C4"]

    def test_nested_tables_become_nested_dicts(self, tmp_path):
        params = tmp_path / "params.toml"
        params.write_text("[run]\nn_jobs = 8\n\n[data]\nuse_tuab = true\n")
        cfg = load(params)
        assert cfg["run"]["n_jobs"] == 8
        assert cfg["data"]["use_tuab"] is True


class TestErrors:
    def test_missing_file_raises(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            load(tmp_path / "does_not_exist.toml")

    def test_malformed_toml_raises(self, tmp_path):
        params = tmp_path / "bad.toml"
        params.write_text("this is = = not valid toml")
        # tomllib.TOMLDecodeError subclasses ValueError.
        with pytest.raises(TOMLDecodeError):
            load(params)
