"""Tests for DatasetBuilder (eeg_win_stack/io/dataset_builder.py)."""

from __future__ import annotations

from unittest.mock import MagicMock, patch


from eeg_win_stack.io.dataset_builder import DatasetBuilder


class TestDefaults:
    def test_load_flags_default_to_false(self):
        db = DatasetBuilder()
        assert db.load_saved_windows is False
        assert db.load_saved_data is False

    def test_preprocessing_defaults(self):
        db = DatasetBuilder()
        assert db.sampling_freq == 100.0
        assert db.window_len_s == 60.0
        assert db.standardization is True
        assert db.window_stride_samples is None

    def test_custom_values_stored(self):
        db = DatasetBuilder(
            load_saved_windows=True,
            saved_windows_path="/tmp/wins",
            n_load=5,
            window_len_s=30.0,
        )
        assert db.load_saved_windows is True
        assert db.saved_windows_path == "/tmp/wins"
        assert db.n_load == 5
        assert db.window_len_s == 30.0


class TestBuildRouting:
    @patch.object(DatasetBuilder, "_window")
    @patch.object(DatasetBuilder, "_load_saved_windows")
    def test_saved_windows_path_skips_windowing(self, mock_load, mock_window):
        mock_load.return_value = MagicMock()
        db = DatasetBuilder(load_saved_windows=True)
        db.build()
        mock_load.assert_called_once()
        mock_window.assert_not_called()

    @patch.object(DatasetBuilder, "_window")
    @patch.object(DatasetBuilder, "_load_saved_recordings")
    def test_saved_recordings_path_windows_result(self, mock_load, mock_window):
        mock_ds = MagicMock()
        mock_load.return_value = mock_ds
        mock_window.return_value = MagicMock()
        db = DatasetBuilder(load_saved_data=True)
        db.build()
        mock_load.assert_called_once()
        mock_window.assert_called_once_with(mock_ds)

    @patch.object(DatasetBuilder, "_window")
    @patch.object(DatasetBuilder, "_load_and_preprocess_raw")
    def test_raw_path_used_by_default(self, mock_load, mock_window):
        mock_ds = MagicMock()
        mock_load.return_value = mock_ds
        mock_window.return_value = MagicMock()
        db = DatasetBuilder()
        db.build()
        mock_load.assert_called_once()
        mock_window.assert_called_once_with(mock_ds)


class TestLoadSavedWindows:
    @patch("eeg_win_stack.io.dataset_builder.load_concat_dataset")
    def test_passes_correct_args(self, mock_load):
        mock_load.return_value = MagicMock()
        db = DatasetBuilder(saved_windows_path="/tmp/wins")
        db._load_saved_windows()
        mock_load.assert_called_once_with(
            path="/tmp/wins",
            preload=False,
            ids_to_load=None,
            target_name="pathological",
            n_jobs=1,
        )

    @patch("eeg_win_stack.io.dataset_builder.load_concat_dataset")
    def test_n_load_generates_id_list(self, mock_load):
        mock_load.return_value = MagicMock()
        db = DatasetBuilder(saved_windows_path="/tmp/wins", n_load=3)
        db._load_saved_windows()
        _, kwargs = mock_load.call_args
        assert kwargs["ids_to_load"] == [0, 1, 2]

    @patch("eeg_win_stack.io.dataset_builder.load_concat_dataset")
    def test_n_load_none_passes_none(self, mock_load):
        mock_load.return_value = MagicMock()
        db = DatasetBuilder(saved_windows_path="/tmp/wins", n_load=None)
        db._load_saved_windows()
        _, kwargs = mock_load.call_args
        assert kwargs["ids_to_load"] is None


class TestLoadSavedRecordings:
    @patch("eeg_win_stack.io.dataset_builder.load_concat_dataset")
    def test_passes_correct_path_and_preload(self, mock_load):
        mock_load.return_value = MagicMock()
        db = DatasetBuilder(saved_data_path="/tmp/recs", preload=False)
        db._load_saved_recordings()
        _, kwargs = mock_load.call_args
        assert kwargs["path"] == "/tmp/recs"
        assert kwargs["preload"] is False

    @patch("eeg_win_stack.io.dataset_builder.load_concat_dataset")
    def test_n_load_generates_id_list(self, mock_load):
        mock_load.return_value = MagicMock()
        db = DatasetBuilder(saved_data_path="/tmp/recs", n_load=4)
        db._load_saved_recordings()
        _, kwargs = mock_load.call_args
        assert kwargs["ids_to_load"] == [0, 1, 2, 3]


class TestWindow:
    """What _window still owns after the transform moved out: persistence.

    The window-size and stride arithmetic now lives in
    eeg_preprocessing.io.windowing and is tested there — asserting it again here
    would only pin the delegation twice.
    """

    @patch("eeg_win_stack.io.dataset_builder.window_recordings")
    def test_delegates_windowing_with_the_configured_parameters(self, mock_window, mock_concat_dataset):
        db = DatasetBuilder(window_len_s=30.0, window_stride_samples=500, preload=False)
        db._window(mock_concat_dataset)
        mock_window.assert_called_once_with(
            mock_concat_dataset,
            window_len_s=30.0,
            window_stride_samples=500,
            preload=False,
        )

    @patch("eeg_win_stack.io.dataset_builder.window_recordings")
    def test_returns_the_windowed_dataset(self, mock_window, mock_concat_dataset, mock_windows_dataset):
        mock_window.return_value = mock_windows_dataset
        assert DatasetBuilder()._window(mock_concat_dataset) is mock_windows_dataset

    @patch("eeg_win_stack.io.dataset_builder.window_recordings")
    def test_saves_when_flag_set(self, mock_window, mock_concat_dataset, mock_windows_dataset):
        mock_window.return_value = mock_windows_dataset
        db = DatasetBuilder(save_windows=True, saved_windows_path="/tmp/wins")
        db._window(mock_concat_dataset)
        mock_windows_dataset.save.assert_called_once_with("/tmp/wins", overwrite=True)

    @patch("eeg_win_stack.io.dataset_builder.window_recordings")
    def test_does_not_save_by_default(self, mock_window, mock_concat_dataset, mock_windows_dataset):
        mock_window.return_value = mock_windows_dataset
        DatasetBuilder()._window(mock_concat_dataset)
        mock_windows_dataset.save.assert_not_called()
