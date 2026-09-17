"""Tests for the decision DVC stage (eeg_learning/pipeline/decision.py)."""

from __future__ import annotations

import csv
import json
from pathlib import Path
from unittest.mock import patch

import pytest

from eeg_learning.pipeline import decision as decision_stage


@pytest.fixture
def stage_config(tmp_path):
    """Config pointing every decision output inside a temporary ``target/``."""
    target = tmp_path / "target"
    return {
        "logging": {"level": "WARNING", "file": "", "console": False},
        # This suite covers the decision stage, not run threading; opt out so a
        # missing run token does not abort the stage under the fail-loud default.
        "run": {"mlflow_required": False},
        "output": {"decision_models_path": str(target / "saved_models" / "decision_model")},
        "decision": {
            "backend": "xgboost",
            "xgboost": {"n_estimators": 5, "max_depth": 2, "random_state": 0},
            "detail_path": "",  # filled in per test
            "csv_result_path": str(target / "decision_results.csv"),
            "metrics_path": str(target / "decision_metrics.json"),
            "length": 10,
            "use_his": True,
            "use_hybrid": False,
            "adap_pool": False,
            "hidden_layers": 0,
            "hidden_length": 5,
            "use_session_or_patients": None,
            "batch_size": 2,
            "n_epochs": 2,
            "train_ratio": 0.7,
            "valid_ratio": 0.75,
            "fix_testset": True,
            "device": "cpu",
            "n_repetitions": 1,
        },
    }


@pytest.fixture
def training_detail_csv(tmp_path):
    """A legacy training_detail.csv with four recordings of four windows each."""
    csv_file = tmp_path / "training_detail.csv"
    rows = [
        ["training_detail"],
        ["True"] * 8,
        ["True"] * 8,
        ["False"] * 8,
        ["False"] * 8,
        ["0.80", "0.90", "0.85", "0.82", "0.78", "0.88", "0.91", "0.80"],
        ["0.83", "0.87", "0.86", "0.84", "0.81", "0.90", "0.85", "0.88"],
        ["0.10", "0.20", "0.15", "0.12", "0.11", "0.18", "0.14", "0.16"],
        ["0.09", "0.13", "0.17", "0.10", "0.12", "0.14", "0.19", "0.15"],
        ["0", "4", "8", "12", "16", "20", "24", "28"],
        ["P01", "P01", "P02", "P02", "P03", "P03", "P04", "P04"],
        ["S01", "S02", "S01", "S02", "S01", "S02", "S01", "S02"],
    ]
    with open(csv_file, "w", newline="") as handle:
        writer = csv.writer(handle)
        for row in rows:
            writer.writerow(row)
    return csv_file


def test_stage_writes_all_outputs_under_target(stage_config, training_detail_csv):
    """Results CSV, metrics JSON, and the model artifact all land in target/."""
    stage_config["decision"]["detail_path"] = str(training_detail_csv)

    with patch("eeg_learning.pipeline.decision.load", return_value=stage_config):
        decision_stage.main()

    metrics = json.loads(Path(stage_config["decision"]["metrics_path"]).read_text())
    assert set(metrics) == {"test_acc", "ori_acc", "argmax_acc", "mean_acc"}
    assert all(0 <= value <= 1 for value in metrics.values())

    with open(stage_config["decision"]["csv_result_path"]) as handle:
        rows = list(csv.DictReader(handle))
    assert len(rows) == 1

    models_dir = Path(stage_config["output"]["decision_models_path"])
    assert len(list(models_dir.glob("decision_xgboost_*.manifest.json"))) == 1
