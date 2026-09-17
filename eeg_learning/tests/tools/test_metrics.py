"""Test for metrics toolset"""

import numpy as np
import pytest
import torch

from eeg_learning.tools.metrics import (
    find_all_zero,
    weight_function,
    matthews_correlation_coefficient,
    convolution_matrix,
    top1,
    top1_prob,
)


def test_weight_function():
    target = np.array([0.0, 0.0, 1.0, 1.0])
    weights = weight_function(target)
    torch.testing.assert_close(weights, torch.tensor([1.0, 1.0]))
    assert isinstance(weights, torch.Tensor)
    assert weights.shape == (2,)


def test_weights_apply():
    target = np.array([0.0, 0.0, 0.0, 1.0])
    weights = weight_function(target)
    torch.testing.assert_close(weights, torch.tensor([1.0, 3.0]))


def test_bool_weights_apply():
    """This is for the pathological case in braindecode"""
    target = np.array([False, False, True, True])
    weights = weight_function(target)
    torch.testing.assert_close(weights, torch.tensor([1.0, 1.0]))


def test_convolution_matrix():
    """
    4 recordings, 2 windows each; returned layout is [[FF, TF], [FT, TT]].
    rec1 pred True / true True  -> TT
    rec2 pred False/ true False -> FF
    rec3 pred False/ true True  -> FT
    rec4 pred True / true False -> TF
    """
    starts = [0, 2, 4, 6]
    b = np.array([True, True, False, False, True, True, False, False])
    c = np.array([True, True, False, False, False, False, True, True])
    result = convolution_matrix(starts, b, c)
    np.testing.assert_array_equal(result, np.array([[1, 1], [1, 1]]))


def test_int_targets_count_by_truthiness():
    """
    Integer 0/1 labels are normalised via truthiness, so 1 counts as positive.
    pred 1 / true 1 -> TT; [[FF, TF], [FT, TT]]
    """
    starts = [0]
    b = np.array([1, 1, 1])
    c = np.array([1, 1, 1])
    result = convolution_matrix(starts, b, c)
    assert result[1, 1] == 1  # TT
    assert result[0, 0] == 0  # FF


def test_single_recording_whole_array():
    """
    starts has one entry -> loop skipped, all of c is one recording.
    pred True, true True -> TT only; [[FF, TF], [FT, TT]]
    """
    starts = [0]
    b = np.array([True, True, True])
    c = np.array([True, False, True])  # majority True
    result = convolution_matrix(starts, b, c)
    np.testing.assert_array_equal(result, np.array([[0, 0], [0, 1]]))


def test_convolution_matrix_use_prob():
    """
    use_prob path: the recording label is the summed-probability vote
    (top1_prob over exp(prob)), applied to every recording. A positive
    prob-vote must land in TT/TF (regression guard: the old `is True` check
    forced every prob-voted recording into the negative branch).
    """
    starts = [0, 2]
    # cols = [normal, abnormal], given as log-probs (exponentiated internally)
    prob = np.log(
        np.array(
            [
                [0.2, 0.8],  # rec1 -> abnormal mass wins -> pred True
                [0.3, 0.7],
                [0.9, 0.1],  # rec2 -> normal mass wins  -> pred False
                [0.6, 0.4],
            ]
        )
    )
    b = np.array([True, True, False, False])  # rec1 true True, rec2 true False
    c = np.zeros(4, dtype=bool)  # unused on the use_prob path (len only)
    result = convolution_matrix(starts, b, c, use_prob=True, prob=prob)
    # rec1 pred True / true True -> TT ; rec2 pred False / true False -> FF
    np.testing.assert_array_equal(result, np.array([[1, 0], [0, 1]]))


def test_matthews_correlation_coef_identical():
    test_con_matrix = np.array([[1, 1], [1, 1]])
    test_mcc = matthews_correlation_coefficient(test_con_matrix)
    assert test_mcc == 0.0


def test_matthews_correlation_coef_different():
    test_con_matrix = np.array([[6, 2], [1, 3]])
    test_mcc = matthews_correlation_coefficient(test_con_matrix)
    assert test_mcc == pytest.approx(0.47809144)


def test_find_all_zeros():
    values = [1, 2, 1, 2, 0]
    res = find_all_zero(values)
    assert res == [4]
    assert isinstance(res, list)


def test_find_all_zeros_empty():
    input_empty = []
    res_empty = find_all_zero(input_empty)
    assert res_empty == []

    input_no_zeros = [1, 2, 3]
    assert find_all_zero(input_no_zeros) == []


def test_top1():
    """
    Test mode is returned from list
    """
    test_list = [1, 3, 4, 5, 4, 3, 4]
    top_test = top1(test_list)
    assert top_test == 4


def test_top1_strings():
    """
    Test whethe function has string compatibility.
    """
    test_list = ["1", "3", "4", "5", "4", "3", "4"]
    top_test = top1(test_list)
    assert top_test == "4"


def test_top1_multimodes():
    """
    In cases where there are multiple modes, the mode which achieves most
    candidates first is returned.

    This is here for documentation purposes.
    """
    test_list = [1, 3, 3, 5, 4, 3, 4, 4]
    top_test = top1(test_list)
    assert top_test == 3


def test_prob():
    """
    Summed abnormal mass > summed normal mass -> 1 (abnormal).
    normal=0.5, abnormal=1.5
    """
    prob = np.array([[0.2, 0.8], [0.3, 0.7]])
    assert top1_prob(prob) == 1


def test_prob_normal_wins():
    """
    Summed normal mass > abnormal -> 0 (normal).
    normal=1.5, abnormal=0.5
    """
    prob = np.array([[0.9, 0.1], [0.6, 0.4]])
    assert top1_prob(prob) == 0


def test_prob_tie_returns_normal():
    """
    Equal mass is not strictly greater (abnormal > normal) -> 0.
    """
    prob = np.array([[0.5, 0.5], [0.5, 0.5]])  # normal=1.0, abnormal=1.0
    assert top1_prob(prob) == 0


def test_prob_sums_mass_not_votes():
    """
    One confident abnormal window outweighs two mildly-normal windows, so
    prob-voting (mass) returns 1 where a majority vote would return 0.
    normal = 1.25, abnormal = 1.75 -> abnormal wins
    """
    prob = np.array([[0.6, 0.4], [0.6, 0.4], [0.05, 0.95]])
    assert top1_prob(prob) == 1
