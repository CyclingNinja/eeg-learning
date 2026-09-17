"""Metric and voting helpers."""

from __future__ import annotations

import numpy as np
import torch


def weight_function(targets, device="cpu"):
    """
    weighting function for datasets
    Parameters
    ----------
    targets : np.array
        A torch tensor of shape (n_samples,)
    device : str
        default 'cpu'

    Returns
    -------
    torch.tensor
    """
    return max(np.count_nonzero(targets == 0), np.count_nonzero(targets == 1)) / torch.tensor(
        [np.count_nonzero(targets == 0), np.count_nonzero(targets == 1)],
        dtype=torch.float,
        device=device,
    )


def convolution_matrix(starts, b, c, use_prob=False, prob=None):
    """
    Recording-level confusion matrix from window-level predictions.

    Each recording spans the windows ``c[start:next_start]``; its predicted
    label is a majority vote (``top1``) or a summed-probability vote
    (``top1_prob``) when ``use_prob`` is set, and its true label is taken from
    the recording's first window.

    Parameters
    ----------
    starts : list[int]
        Window-index boundaries marking the first window of each recording.
    b : array-like
        Per-window true labels; only the value at each recording start is used.
    c : array-like
        Per-window predicted labels (majority-voting path).
    use_prob : bool
        Vote by summed class probability (``top1_prob``) instead of majority.
    prob : array-like, optional
        Per-window (log) probabilities, shape (n_windows, 2); exponentiated
        before voting. Required when ``use_prob`` is set.

    Returns
    -------
    np.ndarray
        2x2 confusion matrix laid out as ``[[FF, TF], [FT, TT]]``
        (rows = truth, cols = prediction).
    """
    if use_prob:
        prob = np.exp(np.array(prob))
    b = np.asarray(b)
    c = np.asarray(c)

    bounds = [*starts, len(c)]
    cm = np.zeros((2, 2), dtype=int)  # cm[true, pred]

    for begin, end in zip(bounds[:-1], bounds[1:]):
        if use_prob:
            pred = top1_prob(prob[begin:end])
        else:
            pred = bool(top1(c[begin:end].tolist()))
        true = bool(b[begin])
        cm[int(true), int(pred)] += 1

    return cm


def matthews_correlation_coefficient(con_matrix):
    """
    Mean square contingency coefficient

    Parameters
    ----------
    con_matrix : np.array
        2 x 2 matrix

    Returns
    -------

    """
    sum1 = con_matrix[0, 0] + con_matrix[0, 1]
    sum2 = con_matrix[0, 1] + con_matrix[1, 1]
    sum3 = con_matrix[1, 0] + con_matrix[1, 1]
    sum4 = con_matrix[1, 0] + con_matrix[0, 0]
    if sum1 == 0 or sum2 == 0 or sum3 == 0 or sum4 == 0:
        return 0
    return (con_matrix[0, 0] * con_matrix[1, 1] - con_matrix[1, 0] * con_matrix[0, 1]) / (
        (sum1 * sum2 * sum3 * sum4) ** 0.5
    )


def find_all_zero(values):
    """
    Finds all non zero elements in an array
    Parameters
    ----------
    values : np.array

    Returns
    -------
    nd.array of all zero elements

    """
    return [i for i in range(len(values)) if values[i] == 0]


def top1(a_list):
    """
    Returns the label with the most occurrences in a list.
    Parameters
    ----------
    a_list : list

    Returns
    -------
    type of list passed
    """
    if a_list:
        return max(a_list, default="empty", key=lambda v: a_list.count(v))
    raise ValueError("Empty predictions passed to convolutional matrix")


def top1_prob(prob):
    """
    Aggregates per-window class probabilities into a single label by
    comparing the summed probability mass of each class.

    Columns are assumed to be [normal, abnormal]. The probabilities are
    summed across all windows and the class with the greater total wins;
    ties resolve to normal.

    Parameters
    ----------
    prob : np.array
        2D array of shape (n_windows, 2) holding per-window probabilities,
        column 0 = normal, column 1 = abnormal.

    Returns
    -------
    int
        1 if the abnormal probability mass exceeds normal, else 0.
    """
    normal = sum(prob[:, 0])
    abnormal = sum(prob[:, 1])
    return 1 if abnormal > normal else 0


def timecost(time_duration):
    """
    Timing function for calculating costing on cloud
    Parameters
    ----------
    time_duration int

    Returns
    -------
    str
        format hh:mm:ss

    """
    m, s = divmod(time_duration, 60)
    h, m = divmod(m, 60)
    return f"{int(h)}h:{int(m)}m:{int(s)}s"
