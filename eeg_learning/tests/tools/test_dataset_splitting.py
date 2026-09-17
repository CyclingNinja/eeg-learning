"""Tests for DatasetSplitter (eeg_learning/tools/dataset_splitting.py).

DatasetSplitter operates on braindecode ``BaseConcatDataset`` objects. Rather
than mock every attribute access, these tests use ``FakeConcatDataset`` — a
minimal stand-in implementing only the surface the splitter touches:
``description`` (a real DataFrame), ``split()``, ``set_description()`` and
``datasets``. Assertions are made on the returned sub-datasets' descriptions,
i.e. on behaviour rather than on internal calls.

Several tests are explicit regression guards for inversions that existed in the
original refactor branch:
  * grouped splits must give the *majority* of groups to train (train_size),
  * folder/tuab marking must recompute *sentinel* rows and leave real bools
    untouched.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from eeg_learning.tools import dataset_splitting
from eeg_learning.tools.dataset_splitting import DatasetSplitter


class _Recording:
    """Stand-in for a braindecode per-recording dataset; only ``row`` is used."""

    def __init__(self, row):
        self.row = dict(row)


class FakeConcatDataset:
    """Minimal stand-in for braindecode ``BaseConcatDataset``."""

    def __init__(self, description: pd.DataFrame):
        self.description = description.reset_index(drop=True)

    @property
    def datasets(self):
        return [_Recording(self.description.iloc[i].to_dict()) for i in range(len(self.description))]

    def set_description(self, des, overwrite=False):
        self.description = pd.DataFrame(des).reset_index(drop=True)

    def split(self, by):
        if isinstance(by, str):
            groups: dict[str, list[int]] = {}
            for pos, value in enumerate(self.description[by]):
                groups.setdefault(str(value), []).append(pos)
            return {key: FakeConcatDataset(self.description.iloc[idx]) for key, idx in groups.items()}
        return {name: FakeConcatDataset(self.description.iloc[list(idx)]) for name, idx in by.items()}


def make_ds(paths, train=None):
    data = {"path": list(paths)}
    if train is not None:
        # object dtype keeps bool/sentinel values distinct (no int coercion).
        data["train"] = pd.Series(train, dtype=object)
    return FakeConcatDataset(pd.DataFrame(data))


def patients_of(ds):
    return {Path(p).parts[-3] for p in ds.description["path"]}


# --------------------------------------------------------------------------- #
# split_data dispatch
# --------------------------------------------------------------------------- #
class TestSplitDataDispatch:
    def test_unknown_split_way_raises(self):
        splitter = DatasetSplitter(make_ds(["a/b/c.edf"]), 0.5, 0.25, 0.25, 42)
        with pytest.raises(ValueError, match="Unknown split_way: nope"):
            splitter.split_data("nope")

    @pytest.mark.parametrize(
        ("split_way", "method"),
        [
            ("proportion", "split_by_proportion"),
            ("folder", "split_by_folder"),
            ("patients", "split_by_patient"),
            ("sessions", "split_by_session"),
        ],
    )
    def test_routes_to_strategy(self, split_way, method, monkeypatch):
        splitter = DatasetSplitter(make_ds(["a/b/c.edf"]), 0.5, 0.25, 0.25, 42)
        sentinel = (make_ds(["t/t/t.edf"]), make_ds(["v/v/v.edf"]), make_ds(["e/e/e.edf"]))
        monkeypatch.setattr(splitter, method, lambda: sentinel)
        assert splitter.split_data(split_way) == sentinel

    @pytest.mark.parametrize(
        ("split_way", "expected_test_on"),
        [
            ("train_on_tuab_tueg_test_on_tueg", "tueg"),
            ("train_on_tuab_tueg_test_on_tuab", "tuab"),
        ],
    )
    def test_tuab_tueg_passes_test_on(self, split_way, expected_test_on, monkeypatch):
        splitter = DatasetSplitter(make_ds(["a/b/c.edf"]), 0.5, 0.25, 0.25, 42)
        captured = {}
        sentinel = (make_ds(["t/t/t.edf"]), make_ds(["v/v/v.edf"]), make_ds(["e/e/e.edf"]))

        def fake_tuab(test_on="tueg"):
            captured["test_on"] = test_on
            return sentinel

        monkeypatch.setattr(splitter, "split_tuab_tueg", fake_tuab)
        splitter.split_data(split_way)
        assert captured["test_on"] == expected_test_on


# --------------------------------------------------------------------------- #
# grouped index helper
# --------------------------------------------------------------------------- #
class TestSplitIndicesByGroupLabels:
    def test_groups_kept_together_and_fully_partitioned(self):
        groups = ["a", "a", "b", "b", "c", "c", "d", "d", "e", "e", "f", "f", "g", "g", "h", "h"]
        splitter = DatasetSplitter(make_ds(["x/y/z.edf"]), 0.75, 0.125, 0.125, 42)
        idx_train, idx_valid, idx_test = splitter._split_indices_by_group_labels(groups)

        all_idx = idx_train + idx_valid + idx_test
        assert sorted(all_idx) == list(range(len(groups)))
        assert len(all_idx) == len(set(all_idx))

        def labels(idx):
            return {groups[i] for i in idx}

        assert labels(idx_train).isdisjoint(labels(idx_valid))
        assert labels(idx_train).isdisjoint(labels(idx_test))
        assert labels(idx_valid).isdisjoint(labels(idx_test))

    def test_train_receives_majority_of_groups(self):
        # Regression guard: train_size=0.75 must give train the bulk of groups.
        # The original refactor passed test_size=train_size, inverting this so
        # train got only ~25% of groups.
        groups = ["a", "a", "b", "b", "c", "c", "d", "d", "e", "e", "f", "f", "g", "g", "h", "h"]
        splitter = DatasetSplitter(make_ds(["x/y/z.edf"]), 0.75, 0.125, 0.125, 42)
        idx_train, idx_valid, idx_test = splitter._split_indices_by_group_labels(groups)

        train_labels = {groups[i] for i in idx_train}
        # 8 groups, 75% -> 6 train groups, 1 valid, 1 test.
        assert len(train_labels) == 6
        assert len(idx_train) > len(idx_valid) + len(idx_test)


# --------------------------------------------------------------------------- #
# proportion
# --------------------------------------------------------------------------- #
class TestSplitByProportion:
    def test_partitions_all_rows_with_train_majority(self):
        paths = [f"p{i}/s/rec.edf" for i in range(8)]
        splitter = DatasetSplitter(make_ds(paths), 0.75, 0.125, 0.125, 42)
        train_set, valid_set, test_set = splitter.split_by_proportion()

        all_paths = (
            list(train_set.description["path"])
            + list(valid_set.description["path"])
            + list(test_set.description["path"])
        )
        assert sorted(all_paths) == sorted(paths)
        assert len(train_set.description) == 6
        assert len(valid_set.description) == 1
        assert len(test_set.description) == 1


# --------------------------------------------------------------------------- #
# patient / session
# --------------------------------------------------------------------------- #
class TestSplitByPatient:
    def test_patients_kept_whole_and_disjoint(self):
        paths = [f"patient_{p:03d}/s001/rec_{r}.edf" for p in range(1, 5) for r in range(2)]
        splitter = DatasetSplitter(make_ds(paths), 0.5, 0.25, 0.25, 42)
        train_set, valid_set, test_set = splitter.split_by_patient()

        assert patients_of(train_set).isdisjoint(patients_of(valid_set))
        assert patients_of(train_set).isdisjoint(patients_of(test_set))
        assert patients_of(valid_set).isdisjoint(patients_of(test_set))

        all_patients = patients_of(train_set) | patients_of(valid_set) | patients_of(test_set)
        assert all_patients == {f"patient_{p:03d}" for p in range(1, 5)}
        # every recording for a patient travels with that patient (2 each)
        total = len(train_set.description) + len(valid_set.description) + len(test_set.description)
        assert total == len(paths)


class TestSplitBySession:
    def test_sessions_disjoint(self):
        # Same patient, different sessions -> sessions must not straddle splits.
        paths = [f"patient_001/s{sess:03d}/rec_{r}.edf" for sess in range(1, 5) for r in range(2)]
        splitter = DatasetSplitter(make_ds(paths), 0.5, 0.25, 0.25, 42)
        train_set, valid_set, test_set = splitter.split_by_session()

        def sessions_of(ds):
            return {Path(p).parts[-2] + Path(p).parts[-3] for p in ds.description["path"]}

        assert sessions_of(train_set).isdisjoint(sessions_of(valid_set))
        assert sessions_of(train_set).isdisjoint(sessions_of(test_set))
        assert sessions_of(valid_set).isdisjoint(sessions_of(test_set))


# --------------------------------------------------------------------------- #
# folder
# --------------------------------------------------------------------------- #
class TestSplitByFolder:
    def test_eval_paths_go_to_test_others_to_train(self):
        paths = [
            "data/train/rec_0.edf",
            "data/train/rec_1.edf",
            "data/eval/rec_0.edf",
            "data/eval/rec_1.edf",
        ]
        train = [2, 2, 2, 2]  # all sentinels -> derived from path
        splitter = DatasetSplitter(make_ds(paths, train=train), 0.5, 0.25, 0.25, 42)
        train_set, valid_set, test_set = splitter.split_by_folder()

        assert all("eval" in p for p in test_set.description["path"])
        kept = list(train_set.description["path"]) + list(valid_set.description["path"])
        assert all("eval" not in p for p in kept)

    def test_existing_bool_membership_is_not_recomputed(self):
        # Regression guard: a row pre-marked False (not a sentinel) must stay in
        # test even though its path has no "eval". The original refactor used
        # isinstance(x, bool) and would have recomputed exactly these rows.
        paths = [
            "data/train/rec_0.edf",
            "data/train/rec_1.edf",
            "data/eval/rec_0.edf",
            "data/train/special.edf",  # pre-marked False despite non-eval path
        ]
        train = [2, 2, 2, False]
        splitter = DatasetSplitter(make_ds(paths, train=train), 0.5, 0.25, 0.25, 42)
        _, _, test_set = splitter.split_by_folder()

        test_paths = set(test_set.description["path"])
        assert "data/train/special.edf" in test_paths
        assert "data/eval/rec_0.edf" in test_paths


# --------------------------------------------------------------------------- #
# tuab / tueg
# --------------------------------------------------------------------------- #
class TestSplitTuabTueg:
    @pytest.fixture
    def splitter(self):
        paths = [
            "tuab/aaaaaaaa/s001/r.edf",  # True  -> tuab train
            "tuab/bbbbbbbb/s001/r.edf",  # True  -> tuab train
            "tuab/cccccccc/s001/r.edf",  # False -> tuab test
            "tueg/p001/s001/r.edf",  # sentinel -> tueg
            "tueg/p002/s001/r.edf",
            "tueg/p003/s001/r.edf",
            "tueg/p004/s001/r.edf",
        ]
        train = [True, True, False, 2, 2, 2, 2]
        return DatasetSplitter(make_ds(paths, train=train), 0.5, 0.25, 0.25, 42)

    @pytest.fixture(autouse=True)
    def patch_concat(self, monkeypatch):
        def fake_concat(recordings):
            return FakeConcatDataset(pd.DataFrame([r.row for r in recordings]))

        monkeypatch.setattr(dataset_splitting, "BaseConcatDataset", fake_concat)

    def test_test_on_tueg_returns_held_out_tueg(self, splitter):
        train_set, valid_set, test_set = splitter.split_tuab_tueg(test_on="tueg")
        assert all("tueg" in p for p in test_set.description["path"])
        # train/valid mix tuab and tueg recordings
        train_valid_paths = list(train_set.description["path"]) + list(valid_set.description["path"])
        assert any("tuab" in p for p in train_valid_paths)
        assert any("tueg" in p for p in train_valid_paths)

    def test_test_on_tuab_returns_tuab_eval_split(self, splitter):
        _, _, test_set = splitter.split_tuab_tueg(test_on="tuab")
        # tuab test == the single row originally marked False
        assert list(test_set.description["path"]) == ["tuab/cccccccc/s001/r.edf"]


# --------------------------------------------------------------------------- #
# remove_attribute hook
# --------------------------------------------------------------------------- #
class TestRemoveAttribute:
    def test_drop_duplicates_invoked_and_substitutes_train_set(self, monkeypatch):
        replacement = make_ds(["filtered/x/y.edf"])
        calls = {}

        def fake_drop_duplicates(test_set, train_set, attribute):
            calls["args"] = (test_set, train_set, attribute)
            return replacement

        monkeypatch.setattr(dataset_splitting, "drop_duplicates", fake_drop_duplicates)

        paths = [f"p{i}/s/rec.edf" for i in range(8)]
        splitter = DatasetSplitter(make_ds(paths), 0.75, 0.125, 0.125, 42, remove_attribute="path")
        train_set, _, test_set = splitter.split_data("proportion")

        assert train_set is replacement
        # drop_duplicates(test_set, train_set, attribute)
        assert calls["args"][0] is test_set
        assert calls["args"][2] == "path"

    def test_drop_duplicates_not_called_when_attribute_none(self, monkeypatch):
        def boom(*args, **kwargs):
            raise AssertionError("drop_duplicates should not be called when remove_attribute is None")

        monkeypatch.setattr(dataset_splitting, "drop_duplicates", boom)

        paths = [f"p{i}/s/rec.edf" for i in range(8)]
        splitter = DatasetSplitter(make_ds(paths), 0.75, 0.125, 0.125, 42)
        splitter.split_data("proportion")  # must not raise
