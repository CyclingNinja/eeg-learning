"""Dataset splitting helpers.

`DatasetSplitter` groups the train/valid/test splitting strategies behind one
class, one method per `split_way`. The module-level `split_data` function is a
backwards-compatible wrapper with the same signature the rest of the codebase
already calls.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
from braindecode.datasets import BaseConcatDataset
from sklearn.model_selection import train_test_split

from eeg_learning.tools.filters import drop_duplicates
from eeg_learning.tools.logger import get_logger
from eeg_learning.tools.paths import findall

log = get_logger(__name__)


def _is_split_sentinel(value) -> bool:
    """True when a description "train" entry is not yet an explicit bool.

    The pipeline seeds the "train" column with a non-bool sentinel (e.g. ``2``)
    for rows whose split membership still has to be derived from the path. Rows
    that already hold ``True``/``False`` are left untouched.
    """
    return not isinstance(value, bool)


class DatasetSplitter:
    def __init__(
        self,
        windows_ds,
        train_size,
        valid_size,
        test_size,
        random_state,
        shuffle=True,
        remove_attribute=None,
    ):
        """

        Parameters
        ----------
        windows_ds : BaseConcatDataset
        train_size : float
        valid_size : float
        test_size : float
        random_state : int
        shuffle : bool, optional
        remove_attribute : str, optional
        """
        self.windows_ds = windows_ds
        self.train_size = train_size
        self.valid_size = valid_size
        self.test_size = test_size
        self.shuffle = shuffle
        self.random_state = random_state
        self.remove_attribute = remove_attribute

    def _remove_attribute_check(self, train_set, test_set):
        if self.remove_attribute:
            return drop_duplicates(test_set, train_set, self.remove_attribute)
        return train_set

    def _split_indices_by_group_labels(self, groups):
        unique_groups = list(set(groups))
        idx_train_groups, idx_valid_test_groups = train_test_split(
            np.arange(len(unique_groups)),
            random_state=self.random_state,
            train_size=self.train_size,
            shuffle=self.shuffle,
        )
        idx_valid_groups, idx_test_groups = train_test_split(
            idx_valid_test_groups,
            random_state=self.random_state,
            test_size=self.test_size / (self.test_size + self.valid_size),
            shuffle=self.shuffle,
        )

        idx_train = []
        for i in idx_train_groups:
            idx_train += findall(groups, unique_groups[i])

        idx_valid = []
        for i in idx_valid_groups:
            idx_valid += findall(groups, unique_groups[i])

        idx_test = []
        for i in idx_test_groups:
            idx_test += findall(groups, unique_groups[i])

        return idx_train, idx_valid, idx_test

    def split_by_proportion(self):
        idx_train, idx_valid_test = train_test_split(
            np.arange(len(self.windows_ds.description["path"])),
            random_state=self.random_state,
            train_size=self.train_size,
            shuffle=self.shuffle,
        )
        idx_valid, idx_test = train_test_split(
            idx_valid_test,
            random_state=self.random_state,
            test_size=self.test_size / (self.test_size + self.valid_size),
            shuffle=self.shuffle,
        )
        splits = self.windows_ds.split({"train": idx_train, "valid": idx_valid, "test": idx_test})
        return splits["train"], splits["valid"], splits["test"]

    def split_by_folder(self):
        des = self.windows_ds.description
        existing = des["train"] if "train" in des else [None] * len(des)

        # TUAB keeps its held-out recordings under an "eval" folder. Rows that
        # already carry an explicit bool were decided upstream and are kept.
        # Assigning the whole column at once (rather than cell by cell) keeps the
        # result a real bool column, which is what ``split("train")`` needs to
        # produce the "True"/"False" keys used below.
        derived = ~des["path"].str.contains("eval", regex=False)
        des["train"] = [
            value if not _is_split_sentinel(value) else bool(is_train) for value, is_train in zip(existing, derived)
        ]

        self.windows_ds.set_description(des, overwrite=True)
        splits = self.windows_ds.split("train")
        train_valid_set = splits["True"]
        test_set = splits["False"]

        idx_train, idx_valid = train_test_split(
            np.arange(len(train_valid_set.description["path"])),
            random_state=self.random_state,
            train_size=self.train_size / (self.train_size + self.valid_size),
            shuffle=self.shuffle,
        )
        splits = self.windows_ds.split({"train": idx_train, "valid": idx_valid})
        train_set = splits["train"]
        valid_set = splits["valid"]

        return train_set, valid_set, test_set

    def split_by_patient(self):
        paths = np.array(self.windows_ds.description.loc[:, ["path"]]).tolist()
        patients = []
        for i in range(len(paths)):
            splits = Path(paths[i][0]).parts
            patients.append(splits[-3])

        idx_train, idx_valid, idx_test = self._split_indices_by_group_labels(patients)
        splits = self.windows_ds.split({"train": idx_train, "valid": idx_valid, "test": idx_test})
        return splits["train"], splits["valid"], splits["test"]

    def split_by_session(self):
        paths = np.array(self.windows_ds.description.loc[:, ["path"]]).tolist()
        sessions = []
        for i in range(len(paths)):
            splits = Path(paths[i][0]).parts
            sessions.append(splits[-2] + splits[-3])

        idx_train, idx_valid, idx_test = self._split_indices_by_group_labels(sessions)
        splits = self.windows_ds.split({"train": idx_train, "valid": idx_valid, "test": idx_test})
        return splits["train"], splits["valid"], splits["test"]

    def split_tuab_tueg(self, test_on="tueg"):
        des = self.windows_ds.description
        # Deliberately a mixed column: TUAB rows keep their bool, TUEG rows are
        # tagged "others". Written in one assignment so pandas settles on object
        # dtype instead of warning about an incompatible per-cell write.
        des["train"] = [value if not _is_split_sentinel(value) else "others" for value in des["train"]]

        self.windows_ds.set_description(des, overwrite=True)
        splits = self.windows_ds.split("train")
        tuab_train = splits["True"]
        tuab_test = splits["False"]
        tueg_whole = splits["others"]

        paths = np.array(tueg_whole.description.loc[:, ["path"]]).tolist()
        patients = []
        for i in range(len(paths)):
            splits = Path(paths[i][0]).parts
            patients.append(splits[-3])

        unique_patients = list(set(patients))
        idx_train_patients, idx_test_patients = train_test_split(
            np.arange(len(unique_patients)),
            random_state=self.random_state,
            train_size=self.train_size + self.valid_size,
            shuffle=self.shuffle,
        )

        idx_train = []
        for i in idx_train_patients:
            idx_train += findall(patients, unique_patients[i])

        idx_test = []
        for i in idx_test_patients:
            idx_test += findall(patients, unique_patients[i])

        splits = tueg_whole.split({"train": idx_train, "test": idx_test})
        tueg_train = splits["train"]
        tueg_test = splits["test"]

        train_valid_set = BaseConcatDataset(list(tueg_train.datasets) + list(tuab_train.datasets))
        idx_train, idx_valid = train_test_split(
            np.arange(len(train_valid_set.description["path"])),
            random_state=self.random_state,
            train_size=self.train_size / (self.train_size + self.valid_size),
            shuffle=self.shuffle,
        )
        splits = train_valid_set.split({"train": idx_train, "valid": idx_valid})
        train_set = splits["train"]
        valid_set = splits["valid"]

        test_set = tueg_test if test_on == "tueg" else tuab_test
        return train_set, valid_set, test_set

    def split_data(self, split_way):
        strategies = {
            "proportion": self.split_by_proportion,
            "folder": self.split_by_folder,
            "patients": self.split_by_patient,
            "sessions": self.split_by_session,
            "train_on_tuab_tueg_test_on_tueg": lambda: self.split_tuab_tueg(test_on="tueg"),
            "train_on_tuab_tueg_test_on_tuab": lambda: self.split_tuab_tueg(test_on="tuab"),
        }

        if split_way not in strategies:
            raise ValueError(f"Unknown split_way: {split_way}")

        train_set, valid_set, test_set = strategies[split_way]()
        train_set = self._remove_attribute_check(train_set, test_set)

        log.info(
            "split %r produced train=%d valid=%d test=%d recordings",
            split_way,
            len(train_set.description),
            len(valid_set.description),
            len(test_set.description),
        )
        for name, split in (("train", train_set), ("valid", valid_set), ("test", test_set)):
            log.debug("%s_set:\n%s", name, split.description)
        return train_set, valid_set, test_set
