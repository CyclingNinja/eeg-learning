"""Shared fail-fast validation for the DVC pipeline stages."""

from __future__ import annotations


def validate_window_length(window_len_samples: int, cfg: dict) -> None:
    """Raise if the loaded windows don't match the configured window length.

    The model's temporal layers are sized from the runtime window length, so a
    mismatch between the ``saved_windows`` on disk and the current ``params.toml``
    (e.g. a stale DVC cache built under a different config) would otherwise surface
    as a cryptic loss-shape error deep inside training. Catch it here, at the point
    the drift is first detectable, with an actionable message.

    Parameters
    ----------
    window_len_samples : int
        Temporal length of a single loaded window (``windows_ds[0][0].shape[1]``).
    cfg : dict
        Parsed params.toml; ``preprocessing.sampling_freq`` and
        ``windowing.window_len_s`` define the expected length.

    Raises
    ------
    ValueError
        If ``window_len_samples`` differs from ``sampling_freq * window_len_s``.
    """
    expected = int(cfg["preprocessing"]["sampling_freq"] * cfg["windowing"]["window_len_s"])
    if window_len_samples != expected:
        raise ValueError(
            f"Loaded windows are {window_len_samples} samples long but the config implies "
            f"{expected} (sampling_freq {cfg['preprocessing']['sampling_freq']} "
            f"x window_len_s {cfg['windowing']['window_len_s']}). The saved_windows on disk were "
            "built under a different config -- re-run the preprocess stage or pull windows that "
            "match this params.toml."
        )
