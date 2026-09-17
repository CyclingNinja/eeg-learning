"""Tests for Evaluator (eeg_learning/evaluation/evaluator.py)."""

from __future__ import annotations

import warnings
from unittest.mock import MagicMock

import numpy as np
import pytest
from sklearn.exceptions import UndefinedMetricWarning

from eeg_learning.evaluation.evaluator import EvaluationResult, Evaluator


@pytest.fixture
def make_eeg_classifier():
    """Factory for a mock fitted classifier returning fixed predictions."""

    def _make(y_pred):
        eeg_classifier = MagicMock(name="eeg_classifier")
        eeg_classifier.predict.return_value = np.array(y_pred)
        return eeg_classifier

    return _make


@pytest.fixture
def make_test_set():
    """Factory for a mock windowed dataset exposing get_metadata().target."""

    def _make(y_true):
        test_set = MagicMock(name="test_set")
        test_set.get_metadata.return_value.target = np.array(y_true)
        return test_set

    return _make


class TestEvaluate:
    def test_metrics_match_known_confusion(self, make_eeg_classifier, make_test_set):
        y_true = [0, 0, 0, 0, 1, 1, 1, 1]
        y_pred = [0, 0, 0, 1, 1, 1, 0, 0]
        result = Evaluator().evaluate(make_eeg_classifier(y_pred), make_test_set(y_true))
        # pos_label=0: predicted-0 = {idx 0,1,2,6,7} = 5, of which true-0 = 3
        assert result.precision == pytest.approx(0.6)
        # actual-0 = 4, correctly predicted 0 = 3
        assert result.recall == pytest.approx(0.75)
        assert result.accuracy == pytest.approx(0.625)
        assert result.mcc == pytest.approx(0.2581988897471611)
        np.testing.assert_array_equal(result.confusion_matrix, np.array([[3, 1], [2, 2]]))

    def test_perfect_predictions(self, make_eeg_classifier, make_test_set):
        y = [0, 1, 0, 1]
        result = Evaluator().evaluate(make_eeg_classifier(y), make_test_set(y))
        assert result.precision == pytest.approx(1.0)
        assert result.recall == pytest.approx(1.0)
        assert result.accuracy == pytest.approx(1.0)
        assert result.mcc == pytest.approx(1.0)

    def test_pos_label_one_changes_precision_and_recall(self, make_eeg_classifier, make_test_set):
        y_true = [0, 0, 0, 0, 1, 1, 1, 1]
        y_pred = [0, 0, 0, 1, 1, 1, 0, 0]
        result = Evaluator(pos_label=1).evaluate(make_eeg_classifier(y_pred), make_test_set(y_true))
        # predicted-1 = {idx 3,4,5} = 3, of which true-1 = 2
        assert result.precision == pytest.approx(2 / 3)
        # actual-1 = 4, correctly predicted 1 = 2
        assert result.recall == pytest.approx(0.5)

    def test_predicts_on_the_test_set(self, make_eeg_classifier, make_test_set):
        test_set = make_test_set([0, 1])
        eeg_classifier = make_eeg_classifier([0, 1])
        Evaluator().evaluate(eeg_classifier, test_set)
        eeg_classifier.predict.assert_called_once_with(test_set)

    def test_returns_evaluation_result(self, make_eeg_classifier, make_test_set):
        result = Evaluator().evaluate(make_eeg_classifier([0, 1]), make_test_set([0, 1]))
        assert isinstance(result, EvaluationResult)

    def test_bool_targets_match_int_equivalent(self, make_eeg_classifier, make_test_set):
        # braindecode 'pathological' targets are bool; evaluate() casts via .astype(int).
        # Bool input must yield identical metrics to the int-equivalent input.
        bool_true = [False, False, True, True]
        int_true = [0, 0, 1, 1]
        y_pred = [0, 1, 1, 0]

        bool_result = Evaluator().evaluate(make_eeg_classifier(y_pred), make_test_set(bool_true))
        int_result = Evaluator().evaluate(make_eeg_classifier(y_pred), make_test_set(int_true))

        assert bool_result.precision == int_result.precision
        assert bool_result.recall == int_result.recall
        assert bool_result.accuracy == int_result.accuracy
        assert bool_result.mcc == int_result.mcc
        np.testing.assert_array_equal(bool_result.confusion_matrix, int_result.confusion_matrix)

    def test_no_undefined_metric_warning_when_class_never_predicted(self, make_eeg_classifier, make_test_set):
        # Class 0 is never predicted -> precision_0 denominator is zero.
        # zero_division=0 must give 0.0 silently (no UndefinedMetricWarning).
        y_true = [0, 0, 1, 1]
        y_pred = [1, 1, 1, 1]

        with warnings.catch_warnings():
            warnings.simplefilter("error", UndefinedMetricWarning)
            result = Evaluator().evaluate(make_eeg_classifier(y_pred), make_test_set(y_true))

        assert result.precision == 0.0
        assert result.recall == 0.0
