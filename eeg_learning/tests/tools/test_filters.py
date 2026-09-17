"""Tests for eeg_learning.tools.filters."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock

import numpy as np
import pandas as pd
import pytest

from eeg_learning.io.dataset_builder import DatasetBuilder
from eeg_learning.tools.filters import (
    drop_duplicates,
    remove_tuab_from_dataset,
    select_by_duration,
    select_by_channel,
    exclude_by_undefined_pathology,
    exclude_by_name,
)


def _len_aware_split(ids):
    """Mimic braindecode's ``ds.split(ids)`` -> ``{"0": subset}``.

    The returned subset is tagged with ``selected_ids`` and reports
    ``len(subset) == len(ids)``, so assertions can check both how many and
    which recordings survived a filter.
    """
    ids = list(ids)
    subset = MagicMock(name="subset")
    subset.selected_ids = ids
    subset.__len__.return_value = len(ids)
    return {"0": subset}


def _paths_dataset(paths):
    """Minimal dataset mock exposing only what ``drop_duplicates`` touches:
    a ``description`` with a ``path`` column and a len-aware ``split()``."""
    ds = MagicMock()
    ds.description = pd.DataFrame({"path": list(paths)})
    ds.split = MagicMock(side_effect=_len_aware_split)
    return ds


@pytest.fixture
def dataset_builder():
    """A real DatasetBuilder instance with in-memory args (no I/O).

    ``__init__`` only stores attributes, so constructing one touches no disk.
    ``build()`` is mocked to return a mock dataset, letting tests exercise the
    builder -> filter handoff without loading real EDFs.

    NOTE: the functions in ``tools/filters.py`` operate on a *dataset* (``ds``),
    not on a ``DatasetBuilder``. Use this only if you are testing code that
    drives the builder itself; for the filter functions, use ``mock_dataset``.
    """
    builder = DatasetBuilder(
        load_saved_windows=True,
        saved_windows_path="/tmp/does-not-exist",
        use_tuab=True,
        channels=["C3", "C4"],
        window_len_s=60.0,
    )
    builder.build = MagicMock(return_value=_mock_dataset())
    return builder


def _mock_dataset():
    """Build a mock braindecode-style dataset matching what filters.py expects.

    Provides the three attributes the filter functions touch:
      * ``description`` -- DataFrame with ``path`` and ``pathological`` columns
      * ``datasets``    -- per-recording objects with ``raw.n_times`` /
                           ``raw.info`` (``sfreq``, ``ch_names``)
      * ``split(ids)``  -- returns ``{"0": <subset>}`` keyed as braindecode does
    """
    # Mixed corpus, three recordings: a normal and an abnormal TUAB recording
    # (labelled) plus an unlabelled TUEG one. braindecode derives `pathological`
    # from the path ('abnormal' in tokens -> a real Python bool); TUEG recordings
    # carry no normal/abnormal label, so the column is NaN there (also what pandas
    # yields when concatenating datasets that lack the column).
    # exclude_by_undefined_pathology exists to drop exactly those NaN rows.
    paths = [
        "/data/TUAB/v3.0.1/edf/train/normal/01_tcp_ar/aaaaapia_s001_t000.edf",
        "/data/TUAB/v3.0.1/edf/train/abnormal/01_tcp_ar/aaaaapib_s001_t001.edf",
        "/data/TUEG/v2.0.1/edf/003/aaaaaapi/s001_2003/02_tcp_le/aaaaaapi_s001_t001.edf",
    ]
    description = pd.DataFrame(
        {
            "path": paths,
            "pathological": [False, True, np.nan],
        }
    )

    sub_datasets = []
    for n_times, ch_names in (
        (256 * 600, ["C3", "C4", "Cz"]),  # normal:   600 s, has C4
        (256 * 45, ["C3", "C4", "P3"]),  # abnormal:  45 s, has C4
        (256 * 30, ["C3", "Fz"]),  # TUEG:      30 s, missing C4
    ):
        sub = MagicMock()
        sub.raw.n_times = n_times
        sub.raw.info = {"sfreq": 256.0, "ch_names": ch_names}
        sub_datasets.append(sub)

    ds = MagicMock()
    ds.description = description
    ds.datasets = sub_datasets

    # filters.py uniformly does ds.split(ids)["0"]; the len-aware subset lets
    # assertions use both len(result) and result.selected_ids.
    ds.split = MagicMock(side_effect=_len_aware_split)
    return ds


@pytest.fixture
def mock_dataset():
    """Mock dataset consumed by the tools/filters.py functions."""
    return _mock_dataset()


# drop_duplicates(ds1, ds2, attribute) returns the recordings in ds2 whose
# `attribute` value does NOT appear in ds1. Tested by file name (parts[-1]).
_A = "/data/TUAB/v3.0.1/edf/train/normal/01_tcp_ar/aaaaapia_s001_t000.edf"
_B = "/data/TUAB/v3.0.1/edf/train/abnormal/01_tcp_ar/aaaaapib_s001_t001.edf"
_C = "/data/TUAB/v3.0.1/edf/train/normal/01_tcp_ar/aaaaapic_s001_t000.edf"
_TUEG = "/data/TUEG/v2.0.1/edf/003/aaaaaapi/s001_2003/02_tcp_le/aaaaaapi_s001_t001.edf"


def test_drop_duplicates_disjoint_keeps_all():
    ds1 = _paths_dataset([_A, _B])
    ds2 = _paths_dataset([_C])
    result = drop_duplicates(ds1, ds2, "file_name")
    assert len(result) == 1
    assert result.selected_ids == [0]


def test_drop_duplicates_partial_overlap_keeps_unique():
    ds1 = _paths_dataset([_A, _B])
    ds2 = _paths_dataset([_B, _C])  # _B overlaps ds1, _C does not
    result = drop_duplicates(ds1, ds2, "file_name")
    assert len(result) == 1
    assert result.selected_ids == [1]  # index of _C within ds2


def test_drop_duplicates_identical_removes_all():
    ds = _paths_dataset([_A, _B])
    result = drop_duplicates(ds, ds, "file_name")
    assert len(result) == 0


def test_drop_duplicates_unknown_attribute_raises():
    ds = _paths_dataset([_A])
    with pytest.raises(ValueError, match="Unknown attribute: nonsense"):
        drop_duplicates(ds, ds, "nonsense")


def test_remove_tuab_from_dataset(make_tree, tmp_path):
    # A mixed dataset: two TUAB recordings and one TUEG recording.
    ds = _paths_dataset([_A, _B, _TUEG])

    # remove_tuab_from_dataset reads the TUAB file names off disk via
    # get_full_filelist, so give it a real directory holding the TUAB .edf files
    # (only the basenames matter; contents are irrelevant).
    make_tree([Path(_A).name, Path(_B).name])

    result = remove_tuab_from_dataset(ds, str(tmp_path))

    # Both TUAB recordings are dropped; only the TUEG one (index 2) survives.
    assert len(result) == 1
    assert result.selected_ids == [2]


def test_select_by_duration(mock_dataset):
    dataset = select_by_duration(mock_dataset, 60.0)
    assert len(dataset) == 1
    dataset_long = select_by_duration(mock_dataset, 1000.0)
    assert len(dataset_long) == 0


def test_exclude_by_undefined_pathology(mock_dataset):
    dataset = exclude_by_undefined_pathology(mock_dataset)
    # Keeps the two bool-labelled TUAB recordings, drops the NaN TUEG one.
    assert len(dataset) == 2


def test_exclude_by_name_str(mock_dataset):
    dataset = exclude_by_name(mock_dataset, ["aaaaapia_s001_t000.edf"])
    assert len(dataset) == 2


def test_exclude_by_name_list(mock_dataset):
    dataset = exclude_by_name(mock_dataset, ["aaaaapia_s001_t000.edf", "aaaaapib_s001_t001.edf"])
    assert len(dataset) == 1


def test_select_by_channel(mock_dataset):
    dataset = select_by_channel(mock_dataset, ["C3", "C4"])
    assert len(dataset) == 2


def test_select_by_channel_no_resultset(mock_dataset):
    dataset = select_by_channel(mock_dataset, ["C3", "C5"])
    assert len(dataset) == 0


def test_select_by_channel_no_channels(mock_dataset):
    dataset = select_by_channel(mock_dataset, [])
    assert len(dataset) == 3
