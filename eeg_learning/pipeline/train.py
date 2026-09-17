"""DVC train stage: split windowed data, build model, train, save params."""

import mne
import time
from pathlib import Path

import torch
from braindecode.datautil import load_concat_dataset

from eeg_learning.config import load
from eeg_learning.tools import tracking
from eeg_learning.tools.logger import Logger, get_logger
from eeg_learning.models import ModelFactory
from eeg_learning.pipeline.validation import validate_window_length
from eeg_learning.tools.dataset_splitting import DatasetSplitter
from eeg_learning.training.trainer import Trainer, TrainingConfig

log = get_logger(__name__)


def main():
    cfg = load()
    Logger.from_config(cfg)
    training_cfg = cfg["training"]
    model_cfg = cfg["model"]
    split_cfg = cfg["split"]
    run_cfg = cfg["run"]

    mne.set_log_level(run_cfg["mne_log_level"])

    if torch.cuda.is_available():
        torch.backends.cudnn.benchmark = True
    torch.set_num_threads(run_cfg["n_jobs"])

    log.info("loading windows from %s", cfg["data"]["save_windows_path"])
    windows_ds = load_concat_dataset(
        path=cfg["data"]["save_windows_path"],
        preload=False,
        target_name="pathological",
        n_jobs=1,
    )
    log.info(
        "loaded %d windows from %d recordings, shape %d channels x %d samples",
        len(windows_ds),
        len(windows_ds.datasets),
        windows_ds[0][0].shape[0],
        windows_ds[0][0].shape[1],
    )

    data_choice = DatasetSplitter(
        windows_ds,
        split_cfg["train_size"],
        split_cfg["valid_size"],
        split_cfg["test_size"],
        run_cfg["random_state"],
        shuffle=split_cfg["shuffle"],
        remove_attribute=None,
    )

    train_set, valid_set, _ = data_choice.split_data(split_cfg["split_way"])

    n_channels = windows_ds[0][0].shape[0]
    window_len_samples = windows_ds[0][0].shape[1]
    validate_window_length(window_len_samples, cfg)

    model = ModelFactory.create(
        model_cfg["name"],
        n_channels=n_channels,
        n_classes=training_cfg["n_classes"],
        input_window_samples=window_len_samples,
        drop_prob=model_cfg["dropout"],
        final_conv_length=model_cfg["final_conv_length"],
        **model_cfg.get(model_cfg["name"], {}),
    )
    # Log the class the registry actually resolved to, not just the configured
    # name: duplicate @register keys silently shadow one another.
    log.info(
        "model %r resolved to %s with %d parameters",
        model_cfg["name"],
        type(model).__name__,
        sum(p.numel() for p in model.parameters()),
    )

    training_config = TrainingConfig(
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

    trainer = Trainer(training_config)

    # This stage mints the MLflow run that evaluate and decision resume, so it
    # logs the whole parameter set — including the preprocessing and windowing
    # values it did not itself apply, since the preprocess stage runs before any
    # run exists and cannot join one.
    with tracking.start_run(cfg, resume=False) as tracker:
        tracker.log_params(
            {
                "model": model_cfg["name"],
                "model_class": type(model).__name__,
                "learning_rate": training_cfg["learning_rate"],
                "weight_decay": training_cfg["weight_decay"],
                "batch_size": training_cfg["batch_size"],
                "n_epochs": training_cfg["n_epochs"],
                "early_stopping": training_cfg["earlystopping"],
                "es_patience": training_cfg["es_patience"],
                "split_way": split_cfg["split_way"],
                "random_state": run_cfg["random_state"],
                "sampling_freq": cfg["preprocessing"]["sampling_freq"],
                "bandpass_filter": cfg["preprocessing"]["bandpass_filter"],
                "standardization": cfg["preprocessing"]["standardization"],
                "window_len_s": cfg["windowing"]["window_len_s"],
                "n_channels": n_channels,
                "window_len_samples": window_len_samples,
            }
        )

        eeg_classifier = trainer.fit(model, train_set, valid_set)

        # Per-epoch curves are MLflow-native, so they go in as stepped metrics
        # rather than becoming a DVC output. The CSV below stays as the durable
        # on-disk fallback.
        present, history = Trainer.history_rows(eeg_classifier)
        for epoch, values in history:
            tracker.log_metrics(values, step=int(epoch))
        log.info(
            "logged %d epochs of history as stepped metrics (columns: %s)",
            len(history),
            ", ".join(present) or "none",
        )

        save_dir = Path(cfg["output"]["saved_models_path"])
        save_dir.mkdir(parents=True, exist_ok=True)
        save_path = save_dir / f"{model_cfg['name']}_{time.strftime('%Y-%m-%d_%H-%M-%S')}_params.pt"
        Trainer.save(eeg_classifier, save_path)
        tracker.log_artifact(save_path, artifact_path="model")

        # Record the run ID and the model just written, so evaluate and decision
        # rejoin this run and load this exact checkpoint rather than globbing for
        # the newest .pt.
        tracking.write_token(cfg, run_id=tracker.run_id, model_path=save_path)

        # Persist the per-epoch loss/accuracy table, mirroring the decision stage's
        # results CSV so both training stages leave a comparable training record.
        output_cfg = cfg["output"]
        training_results_path = output_cfg.get("training_results_path", output_cfg.get("log_path"))
        if not training_results_path:
            raise KeyError("Expected output.training_results_path (or legacy output.log_path) in config")
        Trainer.save_history(eeg_classifier, training_results_path)


if __name__ == "__main__":
    main()
