"""DVC pipeline stage for second-stage decision model training and evaluation."""

from __future__ import annotations

import csv
import json
from pathlib import Path

from eeg_learning.api.jobs import decision_training
from eeg_learning.config import load
from eeg_learning.tools import tracking
from eeg_learning.tools.logger import Logger, get_logger

log = get_logger(__name__)


def main(
    training_detail_path: str = "target/training_detail",
    output_dir: str = "target/saved_models/decision_model",
    results_csv: str = "target/decision_results.csv",
    metrics_path: str = "target/decision_metrics.json",
    start_row: int | None = None,
    n_rows: int | None = None,
    row_gap: int | None = None,
    block: int | None = None,
) -> None:
    """Train and evaluate second-stage decision models.

    Parameters
    ----------
    training_detail_path : str
        Path to the first-stage training detail artifact.
    output_dir : str
        Directory for saving the trained decision model artifact.
    results_csv : str
        Path to output results CSV.
    metrics_path : str
        Path to the DVC metrics JSON.
    start_row : int
        Starting row for CSV parsing.
    n_rows : int
        Number of rows with labels in CSV.
    row_gap : int
        Gap between label and data blocks in CSV.
    block : int
        Which block of results to use.
    """
    config = load()
    Logger.from_config(config)
    decision_cfg = config.get("decision", {})
    output_cfg = config.get("output", {})
    training_detail_path = decision_cfg.get("detail_path", decision_cfg.get("csv_path", training_detail_path))
    results_csv = decision_cfg.get("csv_result_path", results_csv)
    metrics_path = decision_cfg.get("metrics_path", metrics_path)
    output_dir = Path(output_cfg.get("decision_models_path", output_dir))
    output_dir.mkdir(parents=True, exist_ok=True)

    log.info(
        "Training %s decision models from %s",
        decision_cfg.get("backend", "mlp"),
        training_detail_path,
    )

    # Train
    training_results = decision_training(
        config,
        training_detail_csv_path=training_detail_path,
        output_dir=output_dir,
        start_row=start_row,
        n_rows=n_rows,
        row_gap=row_gap,
        block=block,
        n_repetitions=config.get("decision", {}).get("n_repetitions", 1),
    )

    # Log results
    results_path = Path(results_csv)
    results_path.parent.mkdir(parents=True, exist_ok=True)
    if training_results:
        fieldnames = list(training_results[0].keys())
        with results_path.open("w", newline="") as results_file:
            writer = csv.DictWriter(results_file, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(training_results)

        for result in training_results:
            log.info("Repetition %s: test_acc=%.4f", result["repetition"], result["test_acc"])

        saved = next((result["saved_model_id"] for result in training_results if result.get("saved_model_id")), None)
        if saved:
            log.info("Saved best decision model as %s in %s", saved, output_dir)

        metric_names = ("test_acc", "ori_acc", "argmax_acc", "mean_acc")
        metrics = {
            name: sum(result[name] for result in training_results) / len(training_results) for name in metric_names
        }
    else:
        metrics = {}

    metrics_file = Path(metrics_path)
    metrics_file.parent.mkdir(parents=True, exist_ok=True)
    metrics_file.write_text(json.dumps(metrics, indent=2))

    # Rejoin the run the train stage minted, so second-stage results sit
    # alongside the first-stage model that produced their input features. Each
    # repetition is its own experiment run, so they go in as nested children;
    # only the averages stay on the parent, where DVC reads them as metrics.
    with tracking.start_run(config, resume=True) as tracker:
        tracker.log_params({"decision_backend": decision_cfg.get("backend", "mlp")})
        tracker.log_metrics({f"decision_{name}": value for name, value in metrics.items()})

        for result in training_results:
            repetition = result["repetition"]
            with tracker.nested_run(run_name=f"decision-repetition-{repetition}") as child:
                child.log_params({"repetition": repetition})
                child.log_metrics(
                    {
                        name: value
                        for name, value in result.items()
                        if name != "repetition" and isinstance(value, (int, float))
                    }
                )
                if result.get("saved_model_id"):
                    # Marks the repetition whose model was actually persisted.
                    child.set_tags({"saved_model_id": result["saved_model_id"]})


if __name__ == "__main__":
    import sys

    main(*sys.argv[1:])
