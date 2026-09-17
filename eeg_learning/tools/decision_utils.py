"""Utilities for decision-stage model training and evaluation."""

from __future__ import annotations

import csv
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
)

from eeg_learning.tools.metrics import find_all_zero


@dataclass
class DecisionTrainingResult:
    """Result from decision-stage training.

    Attributes
    ----------
    model : torch.nn.Module
        Trained model (best checkpoint from validation).
    best_valid_loss : float
        Best validation loss achieved during training.
    train_losses : list[float]
        Training loss per epoch.
    valid_losses : list[float]
        Validation loss per epoch.
    """

    model: torch.nn.Module
    best_valid_loss: float
    train_losses: list[float]
    valid_losses: list[float]


@dataclass
class DecisionEvaluationResult:
    """Result from decision-stage evaluation.

    Attributes
    ----------
    test_acc : float
        Accuracy of model predictions using argmax.
    ori_acc : float
        Accuracy of raw probability thresholding (> 0.5).
    argmax_acc : float
        Accuracy using argmax (> 50% of valid length).
    mean_acc : float
        Accuracy using mean probability vs threshold.
    confusion_matrix : np.ndarray
        2x2 confusion matrix from model predictions.
    """

    test_acc: float
    ori_acc: float
    argmax_acc: float
    mean_acc: float
    confusion_matrix: np.ndarray


def compute_decision_metrics(
    predictions: torch.Tensor,
    targets: torch.Tensor,
    data: torch.Tensor,
    valid_lens: torch.Tensor,
    device: str = "cpu",
) -> DecisionEvaluationResult:
    """Compute all four decision evaluation metrics.

    Parameters
    ----------
    predictions : torch.Tensor
        Model logits of shape (batch_size, 2).
    targets : torch.Tensor
        True labels of shape (batch_size,).
    data : torch.Tensor
        Raw probability data of shape (batch_size, seq_len).
    valid_lens : torch.Tensor
        Valid sequence lengths of shape (batch_size,).
    device : str, default="cpu"
        Torch device.

    Returns
    -------
    DecisionEvaluationResult
        Struct containing all metrics.
    """
    predictions = predictions.detach().cpu()
    targets = targets.detach().cpu()
    data = data.detach().cpu()
    valid_lens = valid_lens.detach().cpu()

    # Model accuracy (argmax)
    pred_labels = torch.argmax(predictions, dim=-1).numpy()
    target_labels = targets.numpy()
    test_acc = accuracy_score(target_labels, pred_labels)
    cm = confusion_matrix(target_labels, pred_labels, labels=[0, 1])

    # Raw threshold accuracy (> 0.5)
    ori_correct = 0
    ori_total = 0
    for x, y, v in zip(data, target_labels, valid_lens):
        x_binary = (x > 0.5).float()
        y_expanded = torch.full_like(x, y)
        matches = (x_binary == y_expanded)[: int(v)].sum().item()
        ori_correct += matches
        ori_total += int(v)
    ori_acc = ori_correct / ori_total if ori_total > 0 else 0

    # Argmax accuracy (> 50% of valid len)
    argmax_correct = 0
    argmax_total = 0
    for x, y, v in zip(data, target_labels, valid_lens):
        x_binary = (x > 0.5).float()
        y_expanded = torch.full_like(x, y)
        matches = (x_binary == y_expanded)[: int(v)].sum().item()
        if matches > int(v) / 2:
            argmax_correct += 1
        argmax_total += 1
    argmax_acc = argmax_correct / argmax_total if argmax_total > 0 else 0

    # Mean accuracy
    mean_correct = 0
    mean_total = 0
    for x, y, v in zip(data, target_labels, valid_lens):
        mean_prob = torch.mean(x).item()
        threshold = 0.5 * int(v) / 20
        if (mean_prob > threshold) == y:
            mean_correct += 1
        mean_total += 1
    mean_acc = mean_correct / mean_total if mean_total > 0 else 0

    return DecisionEvaluationResult(
        test_acc=test_acc,
        ori_acc=ori_acc,
        argmax_acc=argmax_acc,
        mean_acc=mean_acc,
        confusion_matrix=cm,
    )


def save_decision_results(
    result: DecisionEvaluationResult,
    output_path: str | Path,
    metadata: dict | None = None,
    append: bool = True,
) -> None:
    """Save decision evaluation results to CSV.

    Parameters
    ----------
    result : DecisionEvaluationResult
        Evaluation result to save.
    output_path : str or Path
        Path to output CSV file.
    metadata : dict or None
        Optional metadata to include as columns (e.g., config params).
    append : bool, default=True
        If True, append to existing file; if False, overwrite.
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    mode = "a" if append and output_path.exists() else "w"
    with open(output_path, mode, newline="") as f:
        writer = csv.writer(f, delimiter=",", lineterminator="\n")

        # Write header if creating new file
        if mode == "w":
            header = [
                "test_acc",
                "ori_acc",
                "argmax_acc",
                "mean_acc",
                "tn",
                "fp",
                "fn",
                "tp",
            ]
            if metadata:
                header.extend(metadata.keys())
            writer.writerow(header)

        # Write data row
        cm = result.confusion_matrix
        row = [
            f"{result.test_acc:.6f}",
            f"{result.ori_acc:.6f}",
            f"{result.argmax_acc:.6f}",
            f"{result.mean_acc:.6f}",
            str(cm[0, 0]),  # TN
            str(cm[0, 1]),  # FP
            str(cm[1, 0]),  # FN
            str(cm[1, 1]),  # TP
        ]
        if metadata:
            row.extend([str(v) for v in metadata.values()])
        writer.writerow(row)


def save_training_detail(
    eeg_classifier,
    datasets: list,
    output_path: str | Path,
    *,
    chunk_size: int = 16384,
) -> None:
    """Persist first-stage predictions consumed by decision-stage training.

    Preferred output is a structured directory containing parquet + manifest
    files for machine readability and an easy-to-scan summary CSV. Passing a
    ``.csv`` path preserves the legacy single-file layout.
    """

    output_path = Path(output_path)
    if output_path.suffix.lower() == ".csv":
        split_items = _normalize_splits(datasets)
        _write_legacy_training_detail_csv(
            eeg_classifier,
            [dataset for _, dataset in split_items],
            output_path,
            chunk_size=chunk_size,
        )
        return

    output_path.mkdir(parents=True, exist_ok=True)

    split_items = _normalize_splits(datasets)
    windows_df, recordings_df = _build_training_detail_tables(eeg_classifier, split_items)

    windows_path = output_path / "windows.parquet"
    recordings_path = output_path / "recordings.parquet"
    summary_path = output_path / "recording_summary.csv"
    manifest_path = output_path / "manifest.json"

    windows_df.to_parquet(windows_path, index=False)
    recordings_df.to_parquet(recordings_path, index=False)

    summary_df = _build_recording_summary(recordings_df, windows_df)
    summary_df.to_csv(summary_path, index=False)

    manifest = {
        "format": "training_detail_v2",
        "format_version": 2,
        "files": {
            "windows": windows_path.name,
            "recordings": recordings_path.name,
            "summary": summary_path.name,
            "legacy_csv": "legacy_training_detail.csv",
        },
        "counts": {
            "n_windows": len(windows_df),
            "n_recordings": len(recordings_df),
            "n_patients": int(recordings_df["patient_id"].nunique()) if not recordings_df.empty else 0,
            "n_sessions": int(recordings_df[["patient_id", "session_id"]].drop_duplicates().shape[0])
            if not recordings_df.empty
            else 0,
        },
    }
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    # Keep a compatibility copy while consumers migrate off the legacy parser.
    _write_legacy_training_detail_csv(
        eeg_classifier,
        [dataset for _, dataset in split_items],
        output_path / "legacy_training_detail.csv",
        chunk_size=chunk_size,
    )


def _normalize_splits(datasets: list) -> list[tuple[str, object]]:
    if not datasets:
        return []
    if isinstance(datasets[0], tuple):
        return [(str(name), ds) for name, ds in datasets]

    split_names = ["train", "valid", "test"]
    return [(split_names[i] if i < len(split_names) else f"split_{i}", ds) for i, ds in enumerate(datasets)]


def _split_path_parts(path_value: str) -> tuple[str, str]:
    parts = str(path_value).replace("/", "\\").split("\\")
    if len(parts) >= 3:
        return parts[-3], parts[-2]
    return "", ""


def _build_training_detail_tables(
    eeg_classifier,
    split_items: list[tuple[str, object]],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    window_rows = []
    recording_rows = []
    global_window_index = 0
    recording_index = 0

    for split_name, dataset in split_items:
        metadata = dataset.get_metadata()
        targets = [bool(target) for target in metadata.target.tolist()]
        probs = np.exp(np.asarray(eeg_classifier.predict_proba(dataset)[:, 1])).tolist()
        starts = find_all_zero(metadata["i_window_in_trial"].tolist())
        if not starts or starts[0] != 0:
            starts = [0, *starts]

        description_paths = [row[0] for row in np.asarray(dataset.description.loc[:, ["path"]]).tolist()]

        for rec_i, rec_start in enumerate(starts):
            rec_end = starts[rec_i + 1] if rec_i + 1 < len(starts) else len(targets)
            if rec_end <= rec_start:
                continue

            path_value = description_paths[rec_i] if rec_i < len(description_paths) else ""
            patient_id, session_id = _split_path_parts(path_value)
            recording_id = f"rec_{recording_index:06d}"

            recording_rows.append(
                {
                    "recording_id": recording_id,
                    "recording_order": recording_index,
                    "split": split_name,
                    "path": str(path_value),
                    "patient_id": patient_id,
                    "session_id": session_id,
                    "target": bool(targets[rec_start]),
                    "window_count": rec_end - rec_start,
                    "window_start_global": global_window_index,
                }
            )

            for local_window_index, source_index in enumerate(range(rec_start, rec_end)):
                window_rows.append(
                    {
                        "recording_id": recording_id,
                        "recording_order": recording_index,
                        "split": split_name,
                        "path": str(path_value),
                        "patient_id": patient_id,
                        "session_id": session_id,
                        "window_index_in_recording": local_window_index,
                        "global_window_index": global_window_index,
                        "target": bool(targets[source_index]),
                        "prob_abnormal": float(probs[source_index]),
                    }
                )
                global_window_index += 1

            recording_index += 1

    windows_df = pd.DataFrame.from_records(
        window_rows,
        columns=[
            "recording_id",
            "recording_order",
            "split",
            "path",
            "patient_id",
            "session_id",
            "window_index_in_recording",
            "global_window_index",
            "target",
            "prob_abnormal",
        ],
    )
    recordings_df = pd.DataFrame.from_records(
        recording_rows,
        columns=[
            "recording_id",
            "recording_order",
            "split",
            "path",
            "patient_id",
            "session_id",
            "target",
            "window_count",
            "window_start_global",
        ],
    )
    return windows_df, recordings_df


def _build_recording_summary(recordings_df: pd.DataFrame, windows_df: pd.DataFrame) -> pd.DataFrame:
    if recordings_df.empty:
        return pd.DataFrame(
            columns=[
                "recording_id",
                "split",
                "patient_id",
                "session_id",
                "target",
                "window_count",
                "prob_mean",
                "prob_std",
                "prob_min",
                "prob_max",
                "positive_window_ratio",
                "pred_label",
            ]
        )

    stats = (
        windows_df.groupby("recording_id")
        .agg(
            prob_mean=("prob_abnormal", "mean"),
            prob_std=("prob_abnormal", "std"),
            prob_min=("prob_abnormal", "min"),
            prob_max=("prob_abnormal", "max"),
            positive_window_ratio=("target", "mean"),
        )
        .reset_index()
    )

    summary_df = recordings_df.merge(stats, on="recording_id", how="left")
    summary_df["prob_std"] = summary_df["prob_std"].fillna(0.0)
    summary_df["pred_label"] = (summary_df["prob_mean"] > 0.5).astype(int)
    return summary_df[
        [
            "recording_id",
            "split",
            "patient_id",
            "session_id",
            "target",
            "window_count",
            "prob_mean",
            "prob_std",
            "prob_min",
            "prob_max",
            "positive_window_ratio",
            "pred_label",
        ]
    ]


def _write_legacy_training_detail_csv(
    eeg_classifier,
    datasets: list,
    output_path: Path,
    *,
    chunk_size: int,
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)

    window_labels = []
    window_probs = []
    recording_offsets = []
    patients = []
    sessions = []
    offset = 0

    for dataset in datasets:
        metadata = dataset.get_metadata()
        targets = [bool(target) for target in metadata.target.tolist()]
        probs = np.exp(np.asarray(eeg_classifier.predict_proba(dataset)[:, 1])).tolist()

        window_labels.extend(targets)
        window_probs.extend(probs)
        recording_offsets.extend(start + offset for start in find_all_zero(metadata["i_window_in_trial"].tolist()))

        for row in np.asarray(dataset.description.loc[:, ["path"]]).tolist():
            parts = str(row[0]).replace("/", "\\").split("\\")
            patients.append(parts[-3])
            sessions.append(parts[-2])

        offset += len(targets)

    with output_path.open("w", newline="") as handle:
        writer = csv.writer(handle, delimiter=",", lineterminator="\n")
        writer.writerow(["training_detail"])

        for start in range(0, len(window_labels), chunk_size):
            writer.writerow(window_labels[start : start + chunk_size])

        for start in range(0, len(window_probs), chunk_size):
            writer.writerow(window_probs[start : start + chunk_size])

        writer.writerow(recording_offsets)
        writer.writerow(patients)
        writer.writerow(sessions)
