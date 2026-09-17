"""Importable, path-explicit orchestration for the EEG win-stack API.

These functions are the real work behind every entrypoint — the LocalBackend, the
CLI, and the FastAPI service all funnel here. Unlike ``pipeline/train.py`` (a
procedural DVC stage with hardcoded paths and no return value), they take a
resolved config dict and explicit I/O paths and *return a result object*, so they
can be driven from code, a request handler, or a remote job alike.
"""

from __future__ import annotations

import time
import copy
from dataclasses import asdict, dataclass
from pathlib import Path

import torch
from braindecode.datautil import load_concat_dataset

from eeg_learning.api.artifacts import ModelArtifact
from eeg_learning.api.decision_artifacts import DecisionArtifact
from eeg_learning.models import ModelFactory
from eeg_learning.models.decision_models import build_decision_model
from eeg_learning.tools.dataset_splitting import DatasetSplitter
from eeg_learning.training.trainer import Trainer, TrainingConfig
from eeg_learning.training.decision import (
    split_decision_data,
    train_torch_decision_model,
    train_xgboost_decision_model,
)
from eeg_learning.io.decision_data_loader import DecisionDataLoader
from eeg_learning.tools.decision_utils import DecisionEvaluationResult
from eeg_learning.tools.decision_utils import compute_decision_metrics


@dataclass
class TrainResult:
    """Outcome of :func:`run_training`.

    Attributes
    ----------
    model_id : str
        Identifier of the saved artifact.
    model_path : pathlib.Path
        Path to the saved ``.pt`` weights.
    manifest_path : pathlib.Path
        Path to the saved ``.json`` manifest.
    artifact : ModelArtifact
        The full artifact, ready to reload for evaluation or inference.
    """

    model_id: str
    model_path: Path
    manifest_path: Path
    artifact: ModelArtifact


def _model_build_kwargs(model_cfg: dict, *, n_classes: int, n_channels: int, window_len_samples: int) -> dict:
    """Assemble the kwargs for :meth:`ModelFactory.create` from the ``[model]`` config.

    Only the subsection matching the model name is applied (e.g. ``[model.deep4]``
    for ``name = "deep4"``). This differs deliberately from the old pipeline, which
    spread *every* ``[model.*]`` subsection into one ``create`` call — with the
    default config that raises ``TypeError`` on keys shared between ``deep4`` and
    ``shallow``, and otherwise lets one model's hyperparameters leak into another.

    Parameters
    ----------
    model_cfg : dict
        The ``[model]`` config section, including per-model subsections.
    n_classes : int
        Number of output classes.
    n_channels : int
        Number of input channels, taken from the windowed data.
    window_len_samples : int
        Input window length in samples, taken from the windowed data.

    Returns
    -------
    dict
        The exact kwargs to pass to ``ModelFactory.create(name, **kwargs)``; also
        stored verbatim in the artifact manifest for deterministic reloading.
    """
    build_kwargs = {
        "n_channels": n_channels,
        "n_classes": n_classes,
        "input_window_samples": window_len_samples,
        "drop_prob": model_cfg["dropout"],
        "final_conv_length": model_cfg["final_conv_length"],
    }
    build_kwargs.update(model_cfg.get(model_cfg["name"], {}))
    return build_kwargs


def _training_config(training_cfg: dict) -> TrainingConfig:
    """Build a :class:`TrainingConfig` from the ``[training]`` config section."""
    return TrainingConfig(
        learning_rate=training_cfg["learning_rate"],
        weight_decay=training_cfg["weight_decay"],
        batch_size=training_cfg["batch_size"],
        n_epochs=training_cfg["n_epochs"],
        early_stopping=training_cfg["earlystopping"],
        es_threshold=training_cfg["es_threshold"],
        es_patience=training_cfg["es_patience"],
        test_on_eval=training_cfg["test_on_eval"],
        checkpoint_dir=training_cfg["checkpoint_dir"],
    )


def run_training(config: dict, *, windows_path, output_dir, model_id: str | None = None) -> TrainResult:
    """Train a model from windowed data and save it as a :class:`ModelArtifact`.

    The library form of ``pipeline/train.py``: load windows, split, build the
    model, fit, and persist a weights + manifest pair.

    Parameters
    ----------
    config : dict
        Resolved configuration (as returned by :func:`eeg_learning.config.load`),
        with at least the ``training``, ``model``, ``split``, and ``run`` sections.
    windows_path : str or pathlib.Path
        Directory of saved windowed data to load via ``load_concat_dataset``.
    output_dir : str or pathlib.Path
        Directory the artifact pair is written into.
    model_id : str, optional
        Identifier for the artifact. Defaults to ``"<name>_<timestamp>"``.

    Returns
    -------
    TrainResult
        The saved model's id, file paths, and artifact handle.
    """
    training_cfg = config["training"]
    model_cfg = config["model"]
    split_cfg = config["split"]
    run_cfg = config["run"]

    if torch.cuda.is_available():
        torch.backends.cudnn.benchmark = True
    torch.set_num_threads(run_cfg["n_jobs"])

    windows_ds = load_concat_dataset(
        path=str(windows_path),
        preload=False,
        target_name="pathological",
        n_jobs=1,
    )

    splitter = DatasetSplitter(
        windows_ds,
        split_cfg["train_size"],
        split_cfg["valid_size"],
        split_cfg["test_size"],
        run_cfg["random_state"],
        shuffle=split_cfg["shuffle"],
        remove_attribute=None,
    )
    train_set, valid_set, _ = splitter.split_data(split_cfg["split_way"])

    n_channels = windows_ds[0][0].shape[0]
    window_len_samples = windows_ds[0][0].shape[1]

    build_kwargs = _model_build_kwargs(
        model_cfg,
        n_classes=training_cfg["n_classes"],
        n_channels=n_channels,
        window_len_samples=window_len_samples,
    )
    model = ModelFactory.create(model_cfg["name"], **build_kwargs)

    training_config = _training_config(training_cfg)
    eeg_classifier = Trainer(training_config).fit(model, train_set, valid_set)

    if model_id is None:
        model_id = model_cfg["name"] + "_" + time.strftime("%Y-%m-%d_%H-%M-%S")

    artifact = ModelArtifact.save(
        eeg_classifier,
        model_id=model_id,
        model_name=model_cfg["name"],
        build_kwargs=build_kwargs,
        output_dir=output_dir,
        training_config=asdict(training_config),
    )

    return TrainResult(
        model_id=artifact.model_id,
        model_path=artifact.model_path,
        manifest_path=artifact.manifest_path,
        artifact=artifact,
    )


def _resolve_decision_device(decision_cfg: dict) -> str:
    """Pick the torch device for decision-stage work.

    Uses an explicit ``[decision] device`` if set, otherwise CUDA when available,
    else CPU.
    """
    return decision_cfg.get("device") or ("cuda" if torch.cuda.is_available() else "cpu")


def _load_decision_dataset(
    config: dict,
    csv_path: str | Path,
    *,
    start_row: int | None,
    n_rows: int | None,
    row_gap: int | None,
    block: int | None,
):
    """Load the decision dataset from a first-stage training detail artifact.

    The canonical format is a structured directory (parquet + manifest). Legacy
    CSV layout settings (``start_row``, ``n_rows``, ``row_gap``, ``block``) are
    applied when reading old single-file artifacts.
    """

    decision_cfg = config.get("decision", {})
    start_row = decision_cfg.get("start_row", 1) if start_row is None else start_row
    n_rows = decision_cfg.get("n_rows", 4) if n_rows is None else n_rows
    row_gap = decision_cfg.get("row_gap", 4) if row_gap is None else row_gap
    block = decision_cfg.get("block", 0) if block is None else block

    loader = DecisionDataLoader(
        csv_path,
        start_row=start_row,
        n_rows=n_rows,
        row_gap=row_gap,
        block=block,
    )
    dataset = loader.load(
        aggregation=decision_cfg.get("use_session_or_patients"),
        length=decision_cfg.get("length", 10),
        use_his=decision_cfg.get("use_his", True),
        use_hybrid=decision_cfg.get("use_hybrid", False),
    )
    if len(dataset) == 0:
        raise ValueError(
            "Decision dataset parsed zero samples from "
            f"{csv_path}. Check decision.detail_path (or decision.csv_path for legacy), "
            "and for legacy CSV inputs verify layout settings "
            f"(start_row={start_row}, n_rows={n_rows}, row_gap={row_gap}, block={block})."
        )
    return dataset


def _evaluate_decision_model(model, loader, device) -> DecisionEvaluationResult:
    """Run ``model`` over ``loader`` and compute the decision metrics.

    Shared by :func:`run_decision_training` (on its held-out test split) and
    :func:`run_decision_evaluation` (on the full dataset). The model is assumed to
    already be on ``device`` and in eval mode.
    """

    all_preds = []
    all_targets = []
    all_data = []
    all_valid_lens = []

    with torch.no_grad():
        for batch in loader:
            X, Y, valid_len = [x.to(device) for x in batch]
            Y_hat = model(X, valid_len)
            all_preds.append(Y_hat)
            all_targets.append(Y)
            all_data.append(X)
            all_valid_lens.append(valid_len)

    preds = torch.cat(all_preds, dim=0)
    targets = torch.cat(all_targets, dim=0)
    data = torch.cat(all_data, dim=0)
    valid_lens = torch.cat(all_valid_lens, dim=0)

    return compute_decision_metrics(preds, targets, data, valid_lens)


def decision_training(
    config: dict,
    *,
    training_detail_csv_path: str | Path,
    output_dir: str | Path,
    start_row: int | None = None,
    n_rows: int | None = None,
    row_gap: int | None = None,
    block: int | None = None,
    n_repetitions: int = 1,
) -> list[dict]:
    """Train second-stage decision models from first-stage predictions.

    Loads aggregated first-stage probabilities from a training detail artifact,
    trains decision-stage models with specified hyperparameters, and returns
    training metrics. The learner is chosen by ``[decision] backend`` — the torch
    models, or gradient boosting via ``xgboost``.

    Every repetition is trained; the best of them (highest test accuracy, ties
    broken by validation loss) is saved into ``output_dir`` as a
    :class:`~eeg_learning.api.decision_artifacts.DecisionArtifact`.

    Parameters
    ----------
    config : dict
        Resolved configuration with at least the ``[decision]`` section.
    training_detail_csv_path : str or pathlib.Path
        Path to the first-stage training detail artifact (structured directory
        preferred; legacy CSV is still supported).
    output_dir : str or pathlib.Path
        Directory to save trained models and results.
    start_row : int, optional
        Legacy CSV-only setting for the starting label row.
    n_rows : int, optional
        Legacy CSV-only setting for label row count.
    row_gap : int, optional
        Legacy CSV-only setting for row gap between blocks.
    block : int, optional
        Legacy CSV-only setting for selecting an experiment block.
    n_repetitions : int, default=1
        Number of train/test iterations with different random splits.

    Returns
    -------
    list[dict]
        List of result dicts, one per repetition, with keys:
        - "repetition": int
        - "train_loss": float
        - "valid_loss": float
        - "test_acc": float
        - "ori_acc": float
        - "argmax_acc": float
        - "mean_acc": float
        - "saved_model_id": str, set on the repetition that was persisted
    """

    decision_cfg = config.get("decision", {})
    device = _resolve_decision_device(decision_cfg)
    backend = (decision_cfg.get("backend") or "mlp").lower()

    dataset = _load_decision_dataset(
        config,
        training_detail_csv_path,
        start_row=start_row,
        n_rows=n_rows,
        row_gap=row_gap,
        block=block,
    )
    model_template = build_decision_model(decision_cfg)

    batch_size = decision_cfg.get("batch_size", 64)

    results = []
    best_rep = None
    for rep in range(n_repetitions):
        model = copy.deepcopy(model_template)
        model.to(device)

        train_loader, valid_loader, test_loader = split_decision_data(
            dataset,
            seed=rep,
            batch_size=batch_size,
            train_ratio=decision_cfg.get("train_ratio", 0.9072),
            valid_ratio=decision_cfg.get("valid_ratio", 0.75),
            fix_testset=decision_cfg.get("fix_testset", True),
        )

        if backend == "xgboost":
            training_result = train_xgboost_decision_model(model, train_loader, valid_loader)
        else:
            training_result = train_torch_decision_model(
                model,
                train_loader,
                valid_loader,
                decision_cfg=decision_cfg,
                device=device,
            )

        # Evaluate the best checkpoint on the held-out test split
        model = training_result.model
        model.eval()
        eval_result = _evaluate_decision_model(model, test_loader, device)

        result = {
            "repetition": rep,
            "train_loss": training_result.train_losses[-1],
            "valid_loss": training_result.valid_losses[-1],
            "test_acc": eval_result.test_acc,
            "ori_acc": eval_result.ori_acc,
            "argmax_acc": eval_result.argmax_acc,
            "mean_acc": eval_result.mean_acc,
        }
        results.append(result)

        # Track the winner: best test accuracy, ties broken by validation loss.
        if best_rep is None or (result["test_acc"], -result["valid_loss"]) > (
            best_rep[0]["test_acc"],
            -best_rep[0]["valid_loss"],
        ):
            best_rep = (result, model)

    if best_rep is not None:
        best_result, best_model = best_rep
        artifact = DecisionArtifact.save(
            best_model,
            model_id=f"decision_{backend}_{time.strftime('%Y-%m-%d_%H-%M-%S')}",
            backend=backend,
            decision_cfg=decision_cfg,
            output_dir=output_dir,
            metrics={key: value for key, value in best_result.items() if key != "repetition"},
            repetition=best_result["repetition"],
        )
        for result in results:
            result["saved_model_id"] = artifact.model_id if result["repetition"] == best_result["repetition"] else ""

    return results


def decision_evaluation(
    config: dict,
    *,
    training_detail_csv_path: str | Path,
    model_path: str | Path,
    start_row: int | None = None,
    n_rows: int | None = None,
    row_gap: int | None = None,
    block: int | None = None,
) -> dict:
    """Evaluate a trained decision model on test data.

    Parameters
    ----------
    config : dict
        Resolved configuration with the ``[decision]`` section.
    training_detail_csv_path : str or pathlib.Path
        Path to the first-stage training detail artifact (structured directory
        preferred; legacy CSV is still supported).
    model_path : str or pathlib.Path
        Path to a saved decision model. When a ``<stem>.manifest.json`` sits
        beside it the model is rebuilt from that manifest (any backend);
        otherwise the file is loaded as a pickled torch module.
    start_row : int, default=1
        Starting row index for CSV label block.
    n_rows : int, default=4
        Number of rows containing labels in CSV.
    row_gap : int, default=4
        Gap between label and probability blocks in CSV.
    block : int, default=0
        Which block of results to use.

    Returns
    -------
    dict
        Evaluation metrics:
        - "test_acc": float
        - "ori_acc": float
        - "argmax_acc": float
        - "mean_acc": float
        - "confusion_matrix": np.ndarray
    """
    device = _resolve_decision_device(config.get("decision", {}))

    dataset = _load_decision_dataset(
        config,
        training_detail_csv_path,
        start_row=start_row,
        n_rows=n_rows,
        row_gap=row_gap,
        block=block,
    )

    # Prefer the manifest beside the weights: it carries the backend and feature
    # recipe, so the model is rebuilt exactly as trained. A bare checkpoint with
    # no manifest is loaded as a pickled module, as before.
    artifact = DecisionArtifact.load_for_weights(model_path)
    model = artifact.build_model() if artifact else torch.load(model_path, weights_only=False)
    model.to(device)
    model.eval()

    # Evaluate on full dataset
    loader_full = torch.utils.data.DataLoader(dataset, batch_size=16, shuffle=False, num_workers=0)
    eval_result = _evaluate_decision_model(model, loader_full, device)

    return {
        "test_acc": eval_result.test_acc,
        "ori_acc": eval_result.ori_acc,
        "argmax_acc": eval_result.argmax_acc,
        "mean_acc": eval_result.mean_acc,
        "confusion_matrix": eval_result.confusion_matrix.tolist(),
    }
