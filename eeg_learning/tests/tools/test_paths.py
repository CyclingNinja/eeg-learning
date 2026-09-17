"""Test Suite for paths toolkit"""

from __future__ import annotations

from pathlib import Path


from eeg_learning.tools.paths import (
    findall,
    get_full_filelist,
    read_all_file_names,
    time_key,
)


def test_findall_returns_all_matching_indices():
    # Pure list helper, no filesystem: returns every index where value occurs.
    assert findall(["A", "B", "B"], "B") == [1, 2]


def test_get_full_filelist_recurses_and_filters_extension(make_tree, tmp_path):
    make_tree(
        [
            "TUAB/train/normal/01_tcp_ar/aaaaapia_s001_t000.edf",
            "TUAB/train/abnormal/01_tcp_ar/aaaaapib_s001_t001.edf",
            "TUAB/train/normal/01_tcp_ar/notes.txt",  # wrong extension
            "TUEG/003/aaaaaapi/s001_2003/02_tcp_le/aaaaaapi_s001_t001.edf",
        ],
    )

    files = get_full_filelist(str(tmp_path), ".edf")

    names = {Path(p).name for p in files}
    assert names == {
        "aaaaapia_s001_t000.edf",
        "aaaaapib_s001_t001.edf",
        "aaaaaapi_s001_t001.edf",
    }
    assert "notes.txt" not in names  # extension filter drops it


def test_get_full_filelist_empty_ext_returns_all_files(make_tree, tmp_path):
    make_tree(["a/one.edf", "b/two.txt", "three.csv"])

    files = get_full_filelist(str(tmp_path), "")

    assert {Path(p).name for p in files} == {"one.edf", "two.txt", "three.csv"}


def test_read_all_file_names_orders_by_time(make_tree, tmp_path):
    # time_key sorts on the date folder (parts[-2]) then session/recording ids.
    make_tree(
        [
            "2003_01_02/aaaa_s001_t001.edf",
            "2003_01_01/aaaa_s001_t000.edf",
        ],
    )

    files = read_all_file_names(str(tmp_path), ".edf", key="time")

    assert [Path(p).name for p in files] == [
        "aaaa_s001_t000.edf",  # 2003_01_01 sorts first
        "aaaa_s001_t001.edf",
    ]


def test_time_key_parses_date_session_recording():
    # Pure parser check, no filesystem: date folder + s###/t### in the filename.
    key = time_key("/data/2003_01_02/aaaa_s001_t005.edf")
    assert key == [2003, 1, 2, 1, 5]
