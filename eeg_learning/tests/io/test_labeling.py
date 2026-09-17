"""Tests for relabel (eeg_learning/io/labeling.py)."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from eeg_learning.io.labeling import relabel


def _make_dataset(description):
    """A dataset whose .description is a plain dict-of-lists.

    relabel() only ever treats description as a mapping of column-name -> list,
    so a dict is a faithful stand-in for the real pandas frame.
    """
    ds = MagicMock()
    ds.description = description
    return ds


@pytest.fixture
def label_file(tmp_path):
    """Write a TSV label catalog and return a factory keyed on its rows.

    Columns match the reader in relabel(): folder, file, p_abnormal, ml_label.
    A header row is always written first (relabel skips it).
    """

    def _write(rows):
        path = tmp_path / "labels.tsv"
        lines = ["folder\tfile\tp_ab\tlabel"]
        lines += ["\t".join(str(c) for c in row) for row in rows]
        path.write_text("\n".join(lines) + "\n")
        return str(path)

    return _write


class TestConfidenceThreshold:
    @patch("eeg_learning.io.labeling.read_all_file_names")
    def test_high_prob_marks_pathological_true(self, mock_read, label_file):
        mock_read.return_value = ["/tueg/aaaaaaaa_s001_t000.edf"]
        ds = _make_dataset({"path": ["/data/aaaaaaaa_s001_t000.edf"], "age": [50]})
        path = label_file([["v1.0.0/xx/rec", "sub/aaaaaaaa_s001_t000.edf", 0.995, "1"]])

        des = relabel(ds, path, "/tueg")

        assert des["pathological"] == [True]

    @patch("eeg_learning.io.labeling.read_all_file_names")
    def test_low_prob_marks_pathological_false(self, mock_read, label_file):
        mock_read.return_value = ["/tueg/aaaaaaaa_s001_t000.edf"]
        ds = _make_dataset({"path": ["/data/aaaaaaaa_s001_t000.edf"], "age": [50]})
        path = label_file([["v1.0.0/xx/rec", "sub/aaaaaaaa_s001_t000.edf", 0.005, "0"]])

        des = relabel(ds, path, "/tueg")

        assert des["pathological"] == [False]

    @patch("eeg_learning.io.labeling.read_all_file_names")
    def test_uncertain_prob_leaves_default(self, mock_read, label_file):
        mock_read.return_value = ["/tueg/aaaaaaaa_s001_t000.edf"]
        ds = _make_dataset({"path": ["/data/aaaaaaaa_s001_t000.edf"], "age": [50]})
        # 0.5 is inside (0.01, 0.99): the row is skipped, default sentinel 2 stays.
        path = label_file([["v1.0.0/xx/rec", "sub/aaaaaaaa_s001_t000.edf", 0.5, "1"]])

        des = relabel(ds, path, "/tueg")

        assert des["pathological"] == [2]

    @patch("eeg_learning.io.labeling.read_all_file_names")
    def test_boundary_prob_is_inclusive(self, mock_read, label_file):
        mock_read.return_value = ["/tueg/aaaaaaaa_s001_t000.edf"]
        ds = _make_dataset({"path": ["/data/aaaaaaaa_s001_t000.edf"], "age": [50]})
        # p_ab == 0.99 exactly is accepted (>= comparison).
        path = label_file([["v1.0.0/xx/rec", "sub/aaaaaaaa_s001_t000.edf", 0.99, "1"]])

        des = relabel(ds, path, "/tueg")

        assert des["pathological"] == [True]


class TestPathologicalColumn:
    @patch("eeg_learning.io.labeling.read_all_file_names")
    def test_column_created_when_absent(self, mock_read, label_file):
        mock_read.return_value = ["/tueg/aaaaaaaa_s001_t000.edf"]
        ds = _make_dataset({"path": ["/data/aaaaaaaa_s001_t000.edf"], "age": [50]})
        path = label_file([["v1.0.0/xx/rec", "sub/aaaaaaaa_s001_t000.edf", 0.995, "1"]])

        des = relabel(ds, path, "/tueg")

        assert "pathological" in des

    @patch("eeg_learning.io.labeling.read_all_file_names")
    def test_unmatched_files_keep_default(self, mock_read, label_file):
        mock_read.return_value = ["/tueg/aaaaaaaa_s001_t000.edf"]
        # Two recordings; only the first appears in the label catalog.
        ds = _make_dataset(
            {
                "path": [
                    "/data/aaaaaaaa_s001_t000.edf",
                    "/data/bbbbbbbb_s001_t000.edf",
                ],
                "age": [50, 40],
            }
        )
        path = label_file([["v1.0.0/xx/rec", "sub/aaaaaaaa_s001_t000.edf", 0.995, "1"]])

        des = relabel(ds, path, "/tueg")

        assert des["pathological"][0] is True
        assert des["pathological"][1] == 2


class TestCatalogParsing:
    @patch("eeg_learning.io.labeling.read_all_file_names")
    def test_blank_rows_are_ignored(self, mock_read, label_file):
        mock_read.return_value = ["/tueg/aaaaaaaa_s001_t000.edf"]
        ds = _make_dataset({"path": ["/data/aaaaaaaa_s001_t000.edf"], "age": [50]})
        # csv.reader yields [] for the blank line; relabel must skip it, not crash.
        path = label_file(
            [
                [],
                ["v1.0.0/xx/rec", "sub/aaaaaaaa_s001_t000.edf", 0.995, "1"],
            ]
        )

        des = relabel(ds, path, "/tueg")

        assert des["pathological"] == [True]

    @patch("eeg_learning.io.labeling.read_all_file_names")
    def test_folder_prefix_stripped_for_lookup(self, mock_read, label_file):
        mock_read.return_value = ["/tueg/aaaaaaaa_s001_t000.edf"]
        ds = _make_dataset({"path": ["/data/aaaaaaaa_s001_t000.edf"], "age": [50]})
        path = label_file([["123456789tail/rec", "sub/aaaaaaaa_s001_t000.edf", 0.995, "1"]])

        relabel(ds, path, "/tueg")

        # row[0][9:] strips the first 9 chars before joining onto dataset_folder.
        called_path = mock_read.call_args[0][0]
        assert called_path.endswith("tail/rec")
        assert "123456789" not in called_path
