"""Dataset filtering helpers."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from eeg_learning.tools.logger import get_logger
from eeg_learning.tools.paths import get_full_filelist

log = get_logger(__name__)


def remove_tuab_from_dataset(ds, tuab_loc):
    """
    Remove the tuab recordings from the whole dataset
    Retrieves all fileneames from the TUAB directory to filter out

    Parameters
    ----------
    ds : DatasetBuilder
    tuab_loc : str
        path to the tuab files

    Returns
    -------
    DatasetBuilder

    """
    tuab_list = get_full_filelist(tuab_loc, ".edf")
    tuab_list = [Path(path).name for path in tuab_list]

    split_ids = []
    for d_i, d in enumerate(ds.description["path"]):
        file_name = Path(d).name
        if file_name not in tuab_list:
            split_ids.append(d_i)

    splits = ds.split(split_ids)
    return splits["0"]


def drop_duplicates(ds1, ds2, attribute):
    """
    Drops the duplicates from two datasets.
    Parameters
    ----------
    ds1 : DatasetBuilder instance
        typically
    ds2 : DatasetBuilder instance
    attribute : str
        attribute to remove by, valid attrs 'sessions' 'patients' or None

    Returns
    -------
    DatasetBuilder instance

    """
    remove_num = 0
    if attribute == "file_name":
        loc = -1
    elif attribute == "patients":
        loc = -3
    elif attribute == "sessions":
        loc = -2
    else:
        raise ValueError(f"Unknown attribute: {attribute}")

    paths = np.array(ds1.description.loc[:, ["path"]]).tolist()
    attributes = []
    for i in range(len(paths)):
        splits = Path(paths[i][0]).parts
        attributes.append(splits[loc])

    unique_attributes = list(set(attributes))
    split_ids = []
    for d_i, d in enumerate(ds2.description["path"]):
        attributes2 = Path(d).parts[loc]
        if attributes2 not in unique_attributes:
            split_ids.append(d_i)
        else:
            remove_num += 1

    log.info("dropped %d of %d recordings duplicated by %s", remove_num, len(ds2.description), attribute)
    splits = ds2.split(split_ids)
    return splits["0"]


def select_by_duration(dataset, tmin=0, tmax=None):
    """

    Parameters
    ----------
    dataset : DatasetBuilder instance
    tmin : int
        min duration to select recordings above
    tmax : int
        max duration to select recordings below

    Returns
    -------
    DatasetBuilder instance

    """
    if tmax is None:
        tmax = np.inf

    split_ids = []
    for d_i, d in enumerate(dataset.datasets):
        duration = d.raw.n_times / d.raw.info["sfreq"]
        if tmin <= duration <= tmax:
            split_ids.append(d_i)

    splits = dataset.split(split_ids)
    return splits["0"]


def exclude_by_undefined_pathology(dataset):
    """
    Selects only the recordings that have been confirmed as
    pathological True or False
    Parameters
    ----------
    dataset : DatasetBuilder instance

    Returns
    -------

    """
    split_ids = []
    for d_i, d in enumerate(dataset.description["pathological"]):
        log.debug("recording %d pathology label: %r", d_i, d)
        if d is True or d is False:
            split_ids.append(d_i)

    splits = dataset.split(split_ids)
    return splits["0"]


def exclude_by_name(ds, names):
    split_ids = []
    for d_i, d in enumerate(ds.description["path"]):
        if Path(d).name not in names:
            split_ids.append(d_i)
        else:
            log.debug("excluding %s: name overlaps the exclusion list", Path(d).name)

    splits = ds.split(split_ids)
    return splits["0"]


def select_by_channel(ds, channels):
    """
    Select recordings based on a a list of channels
    Parameters
    ----------
    ds : DatasetBuilder
    channels : list[str]
        List of valid channel names, pass [] for all

    Returns
    -------

    """
    split_ids = []
    for d_i, d in enumerate(ds.datasets):
        include = True
        for chan in channels:
            if chan not in d.raw.info["ch_names"]:
                include = False
                break
        if include:
            split_ids.append(d_i)

    splits = ds.split(split_ids)
    return splits["0"]


def check_inf(ds):
    for d_i, d in enumerate(ds.datasets):
        log.info("recording %d info: %s", d_i, d.raw.info)
