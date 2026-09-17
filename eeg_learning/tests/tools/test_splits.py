from __future__ import annotations

import itertools
import warnings
from pathlib import Path

import pandas as pd
import pytest
from unittest.mock import Mock

from eeg_learning.tools.dataset_splitting import DatasetSplitter


@pytest.fixture
def windows_ds():
    ds = Mock()
    # Four patients, three recordings each. ``parts[-3]`` is the patient id, which
    # is what ``split_by_patient`` groups on, so paths need at least three parts.
    paths = [
        f"patient_{patient:03d}/recording_{recording:03d}/data.fif" for patient in range(1, 5) for recording in range(3)
    ]
    ds.description = pd.DataFrame({"path": paths})
    ds.split.return_value = {
        "train": [0, 1, 2, 3, 4, 5],
        "valid": [6, 7, 8],
        "test": [9, 10, 11],
    }
    return ds


@pytest.fixture
def mock_dataset_splitter(windows_ds):
    return DatasetSplitter(windows_ds, 0.5, 0.25, 0.25, 42, False)


def test_split_indices_by_groups_keeps_groups_together(mock_dataset_splitter):
    groups = ["a", "a", "b", "b", "c", "c", "d", "d"]
    idx_train, idx_valid, idx_test = mock_dataset_splitter._split_indices_by_group_labels(groups=groups)
    # Positive case: all returned indices are valid, unique, and partition the input
    all_indices = idx_train + idx_valid + idx_test
    assert sorted(all_indices) == list(range(len(groups)))
    assert len(all_indices) == len(set(all_indices))

    # Positive case: each group appears in only one split
    def group_members(indices):
        return {groups[i] for i in indices}

    train_groups = group_members(idx_train)
    valid_groups = group_members(idx_valid)
    test_groups = group_members(idx_test)

    assert train_groups.isdisjoint(valid_groups)
    assert train_groups.isdisjoint(test_groups)
    assert valid_groups.isdisjoint(test_groups)

    # Positive case: indices returned for a group are exactly the positions of that group
    for split_indices in (idx_train, idx_valid, idx_test):
        for group_name in group_members(split_indices):
            expected_positions = [i for i, g in enumerate(groups) if g == group_name]
            actual_positions = [i for i in split_indices if groups[i] == group_name]
            assert actual_positions == expected_positions


def test_split_by_proportion_partitions_indices_correctly(mock_dataset_splitter, windows_ds):
    result = mock_dataset_splitter.split_by_proportion()

    windows_ds.split.assert_called_once()
    split_arg = windows_ds.split.call_args.args[0]

    assert set(split_arg.keys()) == {"train", "valid", "test"}
    assert len(split_arg["train"]) == 6
    assert len(split_arg["valid"]) == 3
    assert len(split_arg["test"]) == 3

    all_indices = list(itertools.chain.from_iterable(split_arg.values()))
    assert sorted(all_indices) == list(range(12))
    assert len(all_indices) == len(set(all_indices))

    assert result[0] == windows_ds.split.return_value["train"]
    assert result[1] == windows_ds.split.return_value["valid"]
    assert result[2] == windows_ds.split.return_value["test"]


@pytest.fixture
def folder_windows_ds():
    """A TUAB-shaped dataset: half the recordings live under an ``eval`` folder."""
    ds = Mock()
    paths = [f"tuab/{folder}/patient_{i:03d}/session/data.fif" for folder in ("train", "eval") for i in range(4)]
    ds.description = pd.DataFrame({"path": paths})

    train_valid_set = Mock()
    train_valid_set.description = pd.DataFrame({"path": paths[:4]})

    def split(by):
        # First call splits on the "train" column, second on explicit indices.
        if isinstance(by, str):
            return {"True": train_valid_set, "False": "test_set"}
        return {name: f"{name}_set" for name in by}

    ds.split.side_effect = split
    return ds


def test_split_by_folder_marks_eval_as_test_and_others_as_train(folder_windows_ds):
    splitter = DatasetSplitter(folder_windows_ds, 0.5, 0.25, 0.25, 42, False)

    train_set, valid_set, test_set = splitter.split_by_folder()

    written = folder_windows_ds.set_description.call_args.args[0]
    # Real bools, not 1/0: braindecode's split(by="train") stringifies the group
    # key, and the method looks the results up as "True"/"False".
    assert written["train"].dtype == bool
    assert written["train"].tolist() == [True, True, True, True, False, False, False, False]
    assert folder_windows_ds.split.call_args_list[0].args[0] == "train"
    assert (train_set, valid_set, test_set) == ("train_set", "valid_set", "test_set")


def test_split_by_folder_keeps_rows_already_assigned(folder_windows_ds):
    """Explicit bools set upstream survive; only the sentinel rows are derived."""
    description = folder_windows_ds.description
    # Row 0 sits under "train" but is pinned to the test side; row 4 is the reverse.
    description["train"] = [False, 2, 2, 2, True, 2, 2, 2]
    folder_windows_ds.description = description

    DatasetSplitter(folder_windows_ds, 0.5, 0.25, 0.25, 42, False).split_by_folder()

    written = folder_windows_ds.set_description.call_args.args[0]
    assert written["train"].tolist() == [False, True, True, True, True, False, False, False]


def test_split_by_folder_does_not_trip_pandas_warnings(folder_windows_ds):
    """Guards the chained-assignment / incompatible-dtype write this replaced."""
    splitter = DatasetSplitter(folder_windows_ds, 0.5, 0.25, 0.25, 42, False)

    with warnings.catch_warnings():
        warnings.simplefilter("error")
        splitter.split_by_folder()


def test_split_tuab_tueg_tags_unassigned_rows_without_warnings(monkeypatch):
    """TUEG rows become "others"; TUAB rows keep the bool the loader gave them."""
    ds = Mock()
    tueg_paths = [f"tueg/patient_{i:03d}/session/data.fif" for i in range(4)]
    ds.description = pd.DataFrame(
        {
            "path": ["tuab/train/p/s/data.fif", "tuab/eval/p/s/data.fif", *tueg_paths],
            "train": [True, False, 2, 2, 2, 2],
        }
    )

    tueg_whole = Mock()
    tueg_whole.description = pd.DataFrame({"path": tueg_paths})
    tueg_whole.split.return_value = {"train": Mock(datasets=[]), "test": "tueg_test"}
    ds.split.return_value = {"True": Mock(datasets=[]), "False": "tuab_test", "others": tueg_whole}

    combined = Mock()
    combined.description = pd.DataFrame({"path": tueg_paths})
    combined.split.return_value = {"train": "train_set", "valid": "valid_set"}
    monkeypatch.setattr(
        "eeg_learning.tools.dataset_splitting.BaseConcatDataset",
        Mock(return_value=combined),
    )

    splitter = DatasetSplitter(ds, 0.5, 0.25, 0.25, 42, False)
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        train_set, valid_set, test_set = splitter.split_tuab_tueg(test_on="tueg")

    written = ds.set_description.call_args.args[0]
    assert written["train"].tolist() == [True, False, "others", "others", "others", "others"]
    assert (train_set, valid_set, test_set) == ("train_set", "valid_set", "tueg_test")


def test_split_by_patient(mock_dataset_splitter, windows_ds):
    result = mock_dataset_splitter.split_by_patient()

    windows_ds.split.assert_called_once()
    split_arg = windows_ds.split.call_args.args[0]

    assert set(split_arg.keys()) == {"train", "valid", "test"}
    assert len(split_arg["train"]) == 6
    assert len(split_arg["valid"]) == 3
    assert len(split_arg["test"]) == 3

    all_indices = list(itertools.chain.from_iterable(split_arg.values()))
    assert sorted(all_indices) == list(range(12))
    assert len(all_indices) == len(set(all_indices))

    def patients_in(indices):
        return {Path(windows_ds.description.iloc[i]["path"]).parts[-3] for i in indices}

    train_patients = patients_in(split_arg["train"])
    valid_patients = patients_in(split_arg["valid"])
    test_patients = patients_in(split_arg["test"])

    assert train_patients.isdisjoint(valid_patients)
    assert train_patients.isdisjoint(test_patients)
    assert valid_patients.isdisjoint(test_patients)

    assert len(train_patients) == 2
    assert len(valid_patients) == 1
    assert len(test_patients) == 1

    assert all(
        len([i for i in split_arg["train"] if Path(windows_ds.description.iloc[i]["path"]).parts[-3] == patient]) == 3
        for patient in train_patients
    )
    assert all(
        len([i for i in split_arg["valid"] if Path(windows_ds.description.iloc[i]["path"]).parts[-3] == patient]) == 3
        for patient in valid_patients
    )
    assert all(
        len([i for i in split_arg["test"] if Path(windows_ds.description.iloc[i]["path"]).parts[-3] == patient]) == 3
        for patient in test_patients
    )

    assert result[0] == windows_ds.split.return_value["train"]
    assert result[1] == windows_ds.split.return_value["valid"]
    assert result[2] == windows_ds.split.return_value["test"]


def test_split_by_session(windows_ds):
    pass
