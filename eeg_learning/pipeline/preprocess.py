"""DVC preprocess stage: load raw EEG, preprocess, window, and save to disk."""

import mne

from eeg_learning.config import load
from eeg_learning.tools.logger import Logger, get_logger
from eeg_learning.io.dataset_builder import DatasetBuilder

log = get_logger(__name__)


def main():
    cfg = load()
    Logger.from_config(cfg)
    data_cfg = cfg["data"]
    preprocessing_cfg = cfg["preprocessing"]
    windowing_cfg = cfg["windowing"]
    run_cfg = cfg["run"]

    mne.set_log_level(run_cfg["mne_log_level"])

    log.info("preprocess stage starting (n_jobs=%s, mne_log_level=%s)", run_cfg["n_jobs"], run_cfg["mne_log_level"])

    windows_ds = DatasetBuilder(
        use_tuab=data_cfg["use_tuab"],
        use_tueg=data_cfg["use_tueg"],
        tuab_path=data_cfg["tuab_path"],
        tueg_path=data_cfg["tueg_path"],
        n_tuab=data_cfg["n_tuab"],
        n_tueg=data_cfg["n_tueg"],
        tmin=windowing_cfg["tmin"],
        tmax=windowing_cfg.get("tmax"),
        channels=windowing_cfg["channels"],
        relabel_dataset=data_cfg["relabel_datasets"],
        relabel_label=data_cfg["relabel_labels"],
        sampling_freq=preprocessing_cfg["sampling_freq"],
        sec_to_cut=windowing_cfg["sec_to_cut"],
        duration_recording_sec=windowing_cfg["duration_recording_sec"],
        max_abs_val=preprocessing_cfg["max_abs_val"],
        multiple=windowing_cfg["multiple"],
        bandpass_filter=preprocessing_cfg["bandpass_filter"],
        low_cut_hz=preprocessing_cfg["low_cut_hz"],
        high_cut_hz=preprocessing_cfg["high_cut_hz"],
        standardization=preprocessing_cfg["standardization"],
        factor_new=preprocessing_cfg["factor_new"],
        init_block_size=preprocessing_cfg["init_block_size"],
        window_len_s=windowing_cfg["window_len_s"],
        window_stride_samples=windowing_cfg.get("window_stride_samples"),
        n_load=data_cfg["n_load"],
        preload=data_cfg["preload"],
        n_jobs=run_cfg["n_jobs"],
        save_windows=data_cfg["save_windows"],
        saved_windows_path=data_cfg["save_windows_path"],
        load_saved_windows=data_cfg["load_saved_windows"],
        load_saved_data=data_cfg["load_saved_recordings"],
        saved_data_path=data_cfg["save_recordings_path"],
        save_preprocessed=data_cfg["save_recordings"],
    ).build()

    log.info("preprocess stage complete: %d windows ready", len(windows_ds))


if __name__ == "__main__":
    main()
