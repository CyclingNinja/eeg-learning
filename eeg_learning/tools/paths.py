"""Path and filename helpers."""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any


def findall(items: list[Any], value: Any) -> list[int]:
    """
    Find all indices of a specified value in a list.

    This function searches through the given list and identifies all the indices
    where the specified value is found. It returns a list of these indices. If the
    value does not exist in the list, an empty list is returned.

    Parameters
    ----------
    items : list[Any]
        The list to search for the given value.
    value : Any
        The value to look for in the list.

    Returns
    -------
    list[int]
        A list of indices where the specified value is found in the input list.
        Returns an empty list if the value is not present.
    """
    start = 0
    res = []
    while value in items[start:]:
        res.append(items.index(value, start))
        start = res[-1] + 1
    return res


def get_full_filelist(base_dir: str = ".", target_ext: str = "") -> list[str]:
    fname_list = []

    for path in Path(base_dir).iterdir():
        if path.is_file():
            if target_ext == "" or path.suffix == target_ext:
                fname_list.append(str(path))
        elif path.is_dir():
            temp_list = get_full_filelist(str(path), target_ext)
            fname_list = fname_list + temp_list

    return fname_list


def session_key(file_name: str) -> list[str]:
    """Sort the file name by session."""
    return re.findall(r"(s\d{2})", file_name)


def natural_key(file_name: str) -> list[int | None]:
    """Provide a human-like sorting key of a string."""
    return [int(token) if token.isdigit() else None for token in re.split(r"(\d+)", file_name)]


def time_key(file_name: str) -> list[int]:
    """Provide a time-based sorting key."""
    splits = Path(file_name).parts
    [date] = re.findall(r"(\d{4}_\d{2}_\d{2})", splits[-2])
    date_id = [int(token) for token in date.split("_")]
    [recording_id] = re.findall(r"t(\d{3})", splits[-1])
    [session_id] = re.findall(r"s(\d{3})", splits[-1])
    return [*date_id, int(session_id), int(recording_id)]


def read_all_file_names(path: str, extension: str, key: str = "time") -> list[str]:
    """Read all files with specified extension from a path."""
    file_paths = list(Path(path).glob(f"**/*{extension}"))

    file_paths_str = [str(p) for p in file_paths]
    if key == "time":
        return sorted(file_paths_str, key=time_key)
    if key == "natural":
        return sorted(file_paths_str, key=natural_key)
    return file_paths_str
