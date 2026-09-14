"""Loading and conversion of raw TUAB/TUEG EDF recordings to BrainVision format."""

from __future__ import annotations

import contextlib
import os
import re
from pathlib import Path

import joblib
import mne
import numpy as np
from braindecode.datasets import BaseConcatDataset, TUH, TUHAbnormal
from braindecode.preprocessing import Preprocessor, exponential_moving_standardize, preprocess
from tqdm import tqdm

from eeg_win_stack.io.labeling import relabel
from eeg_win_stack.tools.filters import (
    remove_tuab_from_dataset,
    select_by_channel,
    select_by_duration,
    exclude_by_undefined_pathology,
)


@contextlib.contextmanager
def _tqdm_joblib(tqdm_bar):
    """Report joblib batch completions (used by braindecode's ``preprocess``) to ``tqdm_bar``."""

    class _TqdmBatchCompletionCallback(joblib.parallel.BatchCompletionCallBack):
        def __call__(self, *args, **kwargs):
            tqdm_bar.update(n=self.batch_size)
            return super().__call__(*args, **kwargs)

    original_callback = joblib.parallel.BatchCompletionCallBack
    joblib.parallel.BatchCompletionCallBack = _TqdmBatchCompletionCallback
    try:
        yield tqdm_bar
    finally:
        joblib.parallel.BatchCompletionCallBack = original_callback
        tqdm_bar.close()


def _parse_additional_description_from_file_path(file_path):
    """Replaces braindecode's version of this method, which does ``tokens[-9]``
    assuming the old, deeper v2.0.0 folder layout (.../subject/session_date/file.edf).
    TUAB v3.0.1's real layout is flatter (.../01_tcp_ar/file.edf), so that index is
    always out of range; find the "vX.Y.Z" token instead of counting from the end.
    """
    file_path = os.path.normpath(file_path)
    tokens = file_path.split(os.sep)
    assert "abnormal" in tokens or "normal" in tokens, "No pathology labels found."
    assert "train" in tokens or "eval" in tokens, "No train or eval set information found."
    version = next((t for t in tokens if re.match(r"v\d+\.\d+\.\d+", t)), None)
    return {
        "version": version,
        "train": "train" in tokens,
        "pathological": "abnormal" in tokens,
    }


TUHAbnormal._parse_additional_description_from_file_path = staticmethod(
    _parse_additional_description_from_file_path
)


def custom_crop(raw, tmin=0.0, tmax=None, include_tmax=True):
    tmax = min((raw.n_times - 1) / raw.info["sfreq"], tmax)
    raw.crop(tmin=tmin, tmax=tmax, include_tmax=include_tmax)


_FIF_MEAS_DATE_SECONDS_RANGE = (-2147483648, 2147483647)


def clamp_invalid_meas_date(raw):
    """Null out de-identified TUAB recording dates MNE's FIF format can't store.

    A handful of TUAB recordings carry a ``meas_date`` older than 1901, outside the
    int32-seconds-since-epoch range FIF requires, which otherwise raises a
    RuntimeError only when the recording is saved.
    """
    meas_date = raw.info.get("meas_date")
    if meas_date is not None:
        seconds = int(meas_date.timestamp())
        if not (_FIF_MEAS_DATE_SECONDS_RANGE[0] <= seconds <= _FIF_MEAS_DATE_SECONDS_RANGE[1]):
            raw.set_meas_date(None)
    return raw


class RawEEGLoader:
    """Loads TUAB/TUEG EDF recordings, preprocesses them, and saves as BrainVision.

    This is an offline preprocessing step. The output .vhdr files are consumed
    by EEGLoader in visual_eeg_loading.py for model training.
    """

    def __init__(
        self,
        tuab_path: str | None = None,
        tueg_path: str | None = None,
        n_tuab: int | None = None,
        n_tueg: int | None = None,
        use_tuab: bool = True,
        use_tueg: bool = False,
        preload: bool = True,
        n_jobs: int = 1,
        tuab_version: str = "v3.0.1",
    ):
        self.tuab_path = tuab_path
        self.tueg_path = tueg_path
        self.n_tuab = n_tuab
        self.n_tueg = n_tueg
        self.use_tuab = use_tuab
        self.use_tueg = use_tueg
        self.preload = preload
        self.n_jobs = n_jobs
        self.tuab_version = tuab_version

    def load(self) -> BaseConcatDataset:
        """Load raw TUAB and/or TUEG recordings into a BaseConcatDataset."""
        datasets = []

        if self.use_tuab:
            tuab_ids = list(range(self.n_tuab)) if self.n_tuab else None
            ds_tuab = TUHAbnormal(
                self.tuab_path,
                recording_ids=tuab_ids,
                target_name="pathological",
                preload=self.preload,
                version=self.tuab_version,
            )
            datasets.extend(ds_tuab.datasets)

        if self.use_tueg and self.tueg_path:
            tueg_ids = list(range(self.n_tueg)) if self.n_tueg else None
            ds_tueg = TUH(
                self.tueg_path,
                recording_ids=tueg_ids,
                target_name="pathological",
                preload=self.preload,
            )
            if self.use_tuab:
                ds_tueg = remove_tuab_from_dataset(ds_tueg, self.tuab_path)
            datasets.extend(ds_tueg.datasets)

        return BaseConcatDataset(datasets)

    def filter(
        self,
        recordings: BaseConcatDataset,
        tmin: float,
        tmax: float,
        channels: list[str],
        relabel_label: list | None = None,
        relabel_dataset: list | None = None,
    ) -> BaseConcatDataset:
        """Apply duration, label, and channel filters to a dataset."""
        recordings = select_by_duration(recordings, tmin, tmax)

        if relabel_label:
            for label_path, dataset_folder in zip(relabel_label, relabel_dataset):
                recordings.set_description(relabel(recordings, label_path, dataset_folder), overwrite=True)

        recordings = exclude_by_undefined_pathology(recordings)
        return select_by_channel(recordings, channels)

    def preprocess_recordings(
        self,
        recordings: BaseConcatDataset,
        sampling_freq: float,
        sec_to_cut: float,
        duration_recording_sec: float,
        max_abs_val: float,
        channels: list[str] | None = None,
        multiple: float | None = None,
        bandpass_filter: bool = False,
        low_cut_hz: float | None = None,
        high_cut_hz: float | None = None,
        standardization: bool = False,
        factor_new: float = 1e-3,
        init_block_size: int = 1000,
        save_dir: str | None = None,
    ) -> BaseConcatDataset:
        """Resample, crop, scale, clip, and optionally filter/standardise recordings."""
        preprocessors = [
            Preprocessor(clamp_invalid_meas_date, apply_on_array=False),
            Preprocessor("pick_types", eeg=True, meg=False, stim=False),
            *([Preprocessor("pick_channels", ch_names=channels, ordered=True)] if channels else []),
            Preprocessor(fn="resample", sfreq=sampling_freq),
            Preprocessor(
                custom_crop,
                tmin=sec_to_cut,
                tmax=duration_recording_sec + sec_to_cut,
                include_tmax=False,
                apply_on_array=False,
            ),
            Preprocessor(lambda x: x * 1e6, apply_on_array=True),  # V → µV
            Preprocessor(lambda x: np.clip(x, -max_abs_val, max_abs_val), apply_on_array=True),
        ]
        if multiple:
            factor = multiple
            preprocessors.append(Preprocessor(lambda x: x * factor, apply_on_array=True))
        if bandpass_filter:
            preprocessors.append(Preprocessor("filter", l_freq=low_cut_hz, h_freq=high_cut_hz))
        if standardization:
            preprocessors.append(
                Preprocessor(
                    exponential_moving_standardize,
                    factor_new=factor_new,
                    init_block_size=init_block_size,
                )
            )

        # overwrite=True: save_dir isn't a DVC-tracked out, so stale subdirectories
        # from a previous n_tuab (recording count) would otherwise raise
        # FileExistsError instead of being replaced by this run's recordings.
        with _tqdm_joblib(tqdm(desc="Preprocessing recordings", total=len(recordings.datasets), unit="rec")):
            preprocess(recordings, preprocessors, save_dir=save_dir, overwrite=True, n_jobs=self.n_jobs)
        return recordings

    def save_as_brainvision(self, recordings: BaseConcatDataset, output_dir: str) -> None:
        """Export each recording in the dataset to BrainVision (.vhdr) format."""
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        for i, dataset in enumerate(recordings.datasets):
            raw = dataset.raw
            fname = output_path / f"recording_{i:04d}.vhdr"
            mne.export.export_raw(str(fname), raw, fmt="brainvision", overwrite=True)
