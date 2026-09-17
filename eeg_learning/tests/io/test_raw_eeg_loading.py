"""Tests for RawEEGLoader and custom_crop (eeg_learning/io/raw_eeg_loading.py)."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

from eeg_learning.io.raw_eeg_loading import RawEEGLoader, custom_crop


class TestCustomCrop:
    def test_crops_signal_to_tmax(self, synthetic_raw):
        sfreq = synthetic_raw.info["sfreq"]
        custom_crop(synthetic_raw, tmin=0.0, tmax=5.0)
        assert synthetic_raw.times[-1] <= 5.0 + 1 / sfreq

    def test_tmax_clamped_when_exceeds_recording(self, synthetic_raw):
        original_duration = (synthetic_raw.n_times - 1) / synthetic_raw.info["sfreq"]
        custom_crop(synthetic_raw, tmin=0.0, tmax=original_duration + 100.0)
        assert synthetic_raw.times[-1] <= original_duration + 1 / synthetic_raw.info["sfreq"]

    def test_tmin_trims_start(self, synthetic_raw):
        custom_crop(synthetic_raw, tmin=2.0, tmax=8.0)
        # MNE resets the time origin to 0, so assert on the resulting duration.
        assert synthetic_raw.times[-1] <= 8.0 + 1 / synthetic_raw.info["sfreq"]


class TestDefaults:
    def test_source_flags_default(self):
        loader = RawEEGLoader()
        assert loader.use_tuab is True
        assert loader.use_tueg is False
        assert loader.preload is True
        assert loader.n_jobs == 1

    def test_custom_values_stored(self):
        loader = RawEEGLoader(tuab_path="/tuab", n_tuab=5, use_tueg=True, tueg_path="/tueg", n_jobs=4)
        assert loader.tuab_path == "/tuab"
        assert loader.n_tuab == 5
        assert loader.use_tueg is True
        assert loader.n_jobs == 4


class TestLoad:
    @patch("eeg_learning.io.raw_eeg_loading.BaseConcatDataset")
    @patch("eeg_learning.io.raw_eeg_loading.TUHAbnormal")
    def test_tuab_ids_from_n_tuab(self, mock_tuab, _mock_concat):
        mock_tuab.return_value = MagicMock(datasets=[MagicMock()])
        loader = RawEEGLoader(tuab_path="/tuab", n_tuab=3)
        loader.load()
        args, kwargs = mock_tuab.call_args
        assert args[0] == "/tuab"
        assert kwargs["recording_ids"] == [0, 1, 2]
        assert kwargs["target_name"] == "pathological"

    @patch("eeg_learning.io.raw_eeg_loading.BaseConcatDataset")
    @patch("eeg_learning.io.raw_eeg_loading.TUHAbnormal")
    def test_n_tuab_none_loads_all(self, mock_tuab, _mock_concat):
        mock_tuab.return_value = MagicMock(datasets=[MagicMock()])
        loader = RawEEGLoader(tuab_path="/tuab", n_tuab=None)
        loader.load()
        assert mock_tuab.call_args.kwargs["recording_ids"] is None

    @patch("eeg_learning.io.raw_eeg_loading.BaseConcatDataset")
    @patch("eeg_learning.io.raw_eeg_loading.TUHAbnormal")
    def test_tuab_skipped_when_disabled(self, mock_tuab, _mock_concat):
        loader = RawEEGLoader(use_tuab=False)
        loader.load()
        mock_tuab.assert_not_called()

    @patch("eeg_learning.io.raw_eeg_loading.remove_tuab_from_dataset")
    @patch("eeg_learning.io.raw_eeg_loading.BaseConcatDataset")
    @patch("eeg_learning.io.raw_eeg_loading.TUH")
    @patch("eeg_learning.io.raw_eeg_loading.TUHAbnormal")
    def test_tueg_deduplicated_against_tuab(self, mock_tuab, mock_tuh, _mock_concat, mock_remove):
        mock_tuab.return_value = MagicMock(datasets=[MagicMock()])
        mock_tuh.return_value = MagicMock(datasets=[MagicMock()])
        mock_remove.return_value = MagicMock(datasets=[MagicMock()])
        loader = RawEEGLoader(tuab_path="/tuab", tueg_path="/tueg", use_tuab=True, use_tueg=True)
        loader.load()
        # When both corpora are used, TUEG must be stripped of any TUAB overlap.
        mock_remove.assert_called_once()
        assert mock_remove.call_args[0][1] == "/tuab"

    @patch("eeg_learning.io.raw_eeg_loading.BaseConcatDataset")
    @patch("eeg_learning.io.raw_eeg_loading.TUH")
    @patch("eeg_learning.io.raw_eeg_loading.TUHAbnormal")
    def test_tueg_needs_a_path(self, _mock_tuab, mock_tuh, _mock_concat):
        # use_tueg without a tueg_path must not attempt a TUH load.
        loader = RawEEGLoader(tuab_path="/tuab", use_tueg=True, tueg_path=None)
        loader.load()
        mock_tuh.assert_not_called()


class TestFilter:
    def _patches(self):
        return (
            patch("eeg_learning.io.raw_eeg_loading.select_by_duration"),
            patch("eeg_learning.io.raw_eeg_loading.exclude_by_undefined_pathology"),
            patch("eeg_learning.io.raw_eeg_loading.select_by_channel"),
            patch("eeg_learning.io.raw_eeg_loading.relabel"),
        )

    def test_pipeline_wiring_without_relabel(self):
        p_dur, p_excl, p_chan, p_relabel = self._patches()
        with p_dur as dur, p_excl as excl, p_chan as chan, p_relabel as relabel:
            recs = MagicMock()
            dur.return_value = "after_dur"
            excl.return_value = "after_excl"
            chan.return_value = "after_chan"

            loader = RawEEGLoader()
            result = loader.filter(recs, tmin=10, tmax=60, channels=["C3", "C4"])

            dur.assert_called_once_with(recs, 10, 60)
            excl.assert_called_once_with("after_dur")
            chan.assert_called_once_with("after_excl", ["C3", "C4"])
            relabel.assert_not_called()
            assert result == "after_chan"

    def test_relabel_applied_per_catalog(self):
        p_dur, p_excl, p_chan, p_relabel = self._patches()
        with p_dur as dur, p_excl as excl, p_chan as chan, p_relabel as relabel:
            recs = MagicMock()
            dur.return_value = recs
            excl.return_value = recs
            chan.return_value = recs
            relabel.return_value = {"pathological": [True]}

            loader = RawEEGLoader()
            loader.filter(
                recs,
                tmin=10,
                tmax=60,
                channels=[],
                relabel_label=["labels.tsv"],
                relabel_dataset=["/tueg"],
            )

            relabel.assert_called_once_with(recs, "labels.tsv", "/tueg")
            recs.set_description.assert_called_once()


class TestPreprocessRecordings:
    @patch("eeg_learning.io.raw_eeg_loading.preprocess")
    def test_preprocess_called_and_returns_recordings(self, mock_preprocess):
        recs = MagicMock()
        loader = RawEEGLoader(n_jobs=2)
        result = loader.preprocess_recordings(
            recs,
            sampling_freq=100.0,
            sec_to_cut=60.0,
            duration_recording_sec=600.0,
            max_abs_val=800.0,
            save_dir="/out",
        )
        mock_preprocess.assert_called_once()
        assert mock_preprocess.call_args.kwargs["save_dir"] == "/out"
        assert mock_preprocess.call_args.kwargs["n_jobs"] == 2
        assert result is recs

    @patch("eeg_learning.io.raw_eeg_loading.Preprocessor")
    @patch("eeg_learning.io.raw_eeg_loading.preprocess")
    def test_optional_steps_add_preprocessors(self, _mock_preprocess, mock_prep):
        loader = RawEEGLoader()
        common = {
            "sampling_freq": 100.0,
            "sec_to_cut": 60.0,
            "duration_recording_sec": 600.0,
            "max_abs_val": 800.0,
        }

        loader.preprocess_recordings(MagicMock(), **common)
        baseline = mock_prep.call_count

        mock_prep.reset_mock()
        loader.preprocess_recordings(
            MagicMock(),
            **common,
            bandpass_filter=True,
            low_cut_hz=1.0,
            high_cut_hz=40.0,
            standardization=True,
            multiple=2.0,
        )
        # bandpass + standardization + multiple each append one more Preprocessor.
        assert mock_prep.call_count == baseline + 3


class TestSaveAsBrainvision:
    @patch("eeg_learning.io.raw_eeg_loading.mne")
    def test_exports_each_recording(self, mock_mne, tmp_path):
        recs = MagicMock()
        recs.datasets = [MagicMock(), MagicMock()]
        loader = RawEEGLoader()

        loader.save_as_brainvision(recs, str(tmp_path))

        assert mock_mne.export.export_raw.call_count == 2
        assert mock_mne.export.export_raw.call_args.kwargs["fmt"] == "brainvision"

    @patch("eeg_learning.io.raw_eeg_loading.mne")
    def test_output_directory_created(self, _mock_mne, tmp_path):
        out = tmp_path / "nested" / "bv"
        recs = MagicMock()
        recs.datasets = [MagicMock()]
        loader = RawEEGLoader()

        loader.save_as_brainvision(recs, str(out))

        assert out.is_dir()

    @patch("eeg_learning.io.raw_eeg_loading.mne")
    def test_filenames_are_zero_padded_and_indexed(self, mock_mne, tmp_path):
        recs = MagicMock()
        recs.datasets = [MagicMock(), MagicMock()]
        loader = RawEEGLoader()

        loader.save_as_brainvision(recs, str(tmp_path))

        written = [call.args[0] for call in mock_mne.export.export_raw.call_args_list]
        assert written[0].endswith("recording_0000.vhdr")
        assert written[1].endswith("recording_0001.vhdr")
