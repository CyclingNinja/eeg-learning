"""Builds a windowed EEG dataset from raw, preprocessed, or saved sources."""

from __future__ import annotations

import time
from pathlib import Path

from braindecode.datautil import load_concat_dataset
from braindecode.datasets import BaseConcatDataset

from eeg_preprocessing.io.preprocessing import preprocess_recordings
from eeg_preprocessing.io.sources import TUHCorpusLoader
from eeg_preprocessing.io.windowing import window_recordings
from eeg_preprocessing.tools.filters import curate_recordings
from eeg_win_stack.tools.logger import get_logger

log = get_logger(__name__)


class DatasetBuilder:
    """Encapsulates all data loading, preprocessing, and windowing.

    Three load paths in priority order:
    1. Saved windows   — fastest, skips all preprocessing.
    2. Saved recordings — preprocessed EDF already on disk, just window.
    3. Raw sources     — load TUAB/TUEG EDFs, filter, preprocess, window.

    Call build() to get a windowed BaseConcatDataset ready for splitting.
    """

    def __init__(
        self,
        # --- saved windows (path 1) ---
        load_saved_windows: bool = False,
        saved_windows_path: str | None = None,
        save_windows: bool = False,
        # --- saved preprocessed recordings (path 2) ---
        load_saved_data: bool = False,
        saved_data_path: str | None = None,
        save_preprocessed: bool = False,
        # --- raw source (path 3) ---
        use_tuab: bool = True,
        use_tueg: bool = False,
        tuab_path: str | None = None,
        tueg_path: str | None = None,
        n_tuab: int | None = None,
        n_tueg: int | None = None,
        # --- filtering ---
        tmin: float = 0.0,
        tmax: float = 3600.0,
        channels: list[str] | None = None,
        relabel_label: list | None = None,
        relabel_dataset: list | None = None,
        # --- preprocessing ---
        sampling_freq: float = 100.0,
        sec_to_cut: float = 60.0,
        duration_recording_sec: float = 600.0,
        max_abs_val: float = 800.0,
        multiple: float | None = None,
        bandpass_filter: bool = False,
        low_cut_hz: float | None = None,
        high_cut_hz: float | None = None,
        standardization: bool = True,
        factor_new: float = 1e-3,
        init_block_size: int = 1000,
        # --- windowing ---
        window_len_s: float = 60.0,
        window_stride_samples: int | None = None,
        # --- general ---
        n_load: int | None = None,
        preload: bool = True,
        n_jobs: int = 1,
    ):
        self.load_saved_windows = load_saved_windows
        self.saved_windows_path = saved_windows_path
        self.save_windows = save_windows
        self.load_saved_data = load_saved_data
        self.saved_data_path = saved_data_path
        self.save_preprocessed = save_preprocessed
        self.use_tuab = use_tuab
        self.use_tueg = use_tueg
        self.tuab_path = tuab_path
        self.tueg_path = tueg_path
        self.n_tuab = n_tuab
        self.n_tueg = n_tueg
        self.tmin = tmin
        self.tmax = tmax
        self.channels = channels
        self.relabel_label = relabel_label
        self.relabel_dataset = relabel_dataset
        self.sampling_freq = sampling_freq
        self.sec_to_cut = sec_to_cut
        self.duration_recording_sec = duration_recording_sec
        self.max_abs_val = max_abs_val
        self.multiple = multiple
        self.bandpass_filter = bandpass_filter
        self.low_cut_hz = low_cut_hz
        self.high_cut_hz = high_cut_hz
        self.standardization = standardization
        self.factor_new = factor_new
        self.init_block_size = init_block_size
        self.window_len_s = window_len_s
        self.window_stride_samples = window_stride_samples
        self.n_load = n_load
        self.preload = preload
        self.n_jobs = n_jobs

    def build(self) -> BaseConcatDataset:
        """Return a windowed dataset ready for splitting."""
        # Log the branch actually taken, not the flags requested: a mismatch
        # between the two is the failure mode these load paths keep hitting.
        if self.load_saved_windows:
            log.info("build path: saved windows from %s", self.saved_windows_path)
            return self._load_saved_windows()

        if self.load_saved_data:
            log.info("build path: saved recordings from %s", self.saved_data_path)
            recordings = self._load_saved_recordings()
        else:
            log.info("build path: raw sources (use_tuab=%s, use_tueg=%s)", self.use_tuab, self.use_tueg)
            recordings = self._load_and_preprocess_raw()

        return self._window(recordings)

    # ------------------------------------------------------------------
    # Private: load paths
    # ------------------------------------------------------------------

    def _load_saved_windows(self) -> BaseConcatDataset:
        load_ids = list(range(self.n_load)) if self.n_load else None
        start = time.perf_counter()
        windows_ds = load_concat_dataset(
            path=self.saved_windows_path,
            preload=False,
            ids_to_load=load_ids,
            target_name="pathological",
            n_jobs=1,
        )
        log.info(
            "loaded %d windows from %d saved recordings in %.1fs (n_load=%s)",
            len(windows_ds),
            len(windows_ds.datasets),
            time.perf_counter() - start,
            self.n_load,
        )
        return windows_ds

    def _load_saved_recordings(self) -> BaseConcatDataset:
        load_ids = list(range(self.n_load)) if self.n_load else None
        start = time.perf_counter()
        recordings = load_concat_dataset(
            path=self.saved_data_path,
            preload=self.preload,
            ids_to_load=load_ids,
            target_name="pathological",
        )
        log.info(
            "loaded %d saved recordings in %.1fs (n_load=%s, preload=%s)",
            len(recordings.datasets),
            time.perf_counter() - start,
            self.n_load,
            self.preload,
        )
        return recordings

    def _load_and_preprocess_raw(self) -> BaseConcatDataset:
        loader = TUHCorpusLoader(
            tuab_path=self.tuab_path,
            tueg_path=self.tueg_path,
            n_tuab=self.n_tuab,
            n_tueg=self.n_tueg,
            use_tuab=self.use_tuab,
            use_tueg=self.use_tueg,
            preload=self.preload,
        )
        # Each phase is logged separately: this path holds whole recordings in
        # RAM, so knowing which phase was running when it died is the diagnosis.
        start = time.perf_counter()
        recordings = loader.load()
        log.info(
            "loaded %d raw recordings in %.1fs (preload=%s, n_tuab=%s, n_tueg=%s)",
            len(recordings.datasets),
            time.perf_counter() - start,
            self.preload,
            self.n_tuab,
            self.n_tueg,
        )

        start = time.perf_counter()
        recordings = curate_recordings(
            recordings,
            tmin=self.tmin,
            tmax=self.tmax,
            channels=self.channels,
            relabel_label=self.relabel_label,
            relabel_dataset=self.relabel_dataset,
        )
        log.info(
            "filtered to %d recordings in %.1fs",
            len(recordings.datasets),
            time.perf_counter() - start,
        )

        save_dir = self.saved_data_path if self.save_preprocessed else None
        start = time.perf_counter()
        recordings = preprocess_recordings(
            recordings,
            sampling_freq=self.sampling_freq,
            sec_to_cut=self.sec_to_cut,
            duration_recording_sec=self.duration_recording_sec,
            max_abs_val=self.max_abs_val,
            channels=self.channels,
            multiple=self.multiple,
            bandpass_filter=self.bandpass_filter,
            low_cut_hz=self.low_cut_hz,
            high_cut_hz=self.high_cut_hz,
            standardization=self.standardization,
            factor_new=self.factor_new,
            init_block_size=self.init_block_size,
            save_dir=save_dir,
            n_jobs=self.n_jobs,
        )
        log.info(
            "preprocessed %d recordings in %.1fs (sfreq=%s, save_dir=%s)",
            len(recordings.datasets),
            time.perf_counter() - start,
            self.sampling_freq,
            save_dir,
        )
        return recordings

    # ------------------------------------------------------------------
    # Private: windowing
    # ------------------------------------------------------------------

    def _window(self, recordings: BaseConcatDataset) -> BaseConcatDataset:
        """Window ``recordings``, then persist if configured.

        The windowing itself is the shared transform; only the saving is this
        orchestrator's business, which is why the two are split.
        """
        windows_ds = window_recordings(
            recordings,
            window_len_s=self.window_len_s,
            window_stride_samples=self.window_stride_samples,
            preload=self.preload,
        )

        if self.save_windows:
            Path(self.saved_windows_path).mkdir(parents=True, exist_ok=True)
            windows_ds.save(self.saved_windows_path, overwrite=True)
            log.info("saved windows to %s", self.saved_windows_path)

        return windows_ds
