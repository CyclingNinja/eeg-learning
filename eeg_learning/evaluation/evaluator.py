"""Evaluation: window-level metrics for a fitted EEGClassifier."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    matthews_corrcoef,
    precision_score,
    recall_score,
)


@dataclass
class EvaluationResult:
    """Window-level classification metrics on a test set.

    Attributes
    ----------
    precision : float
        Precision for the positive class (see :attr:`Evaluator.pos_label`).
    recall : float
        Recall for the positive class.
    accuracy : float
        Overall window-level accuracy.
    mcc : float
        Matthews correlation coefficient (symmetric; independent of which
        class is treated as positive).
    confusion_matrix : numpy.ndarray
        2x2 matrix from sklearn (rows = true, cols = predicted), label
        order ``[0, 1]``.
    """

    precision: float
    recall: float
    accuracy: float
    mcc: float
    confusion_matrix: np.ndarray


class Evaluator:
    """Compute window-level metrics for a fitted classifier on a test set.

    Parameters
    ----------
    pos_label : int, default=0
        Class treated as positive for precision and recall. Defaults to
        ``0`` ("normal"), matching the original train_and_eval convention.
    """

    def __init__(self, pos_label: int = 0):
        self.pos_label = pos_label

    def evaluate(self, eeg_classifier, test_set) -> EvaluationResult:
        """Predict on ``test_set`` and return window-level metrics.

        Parameters
        ----------
        eeg_classifier : EEGClassifier
            A fitted classifier exposing ``predict``.
        test_set : braindecode.datasets.BaseConcatDataset
            Windowed test dataset whose metadata holds the targets.

        Returns
        -------
        EvaluationResult
            Precision, recall, accuracy, MCC, and the confusion matrix.
        """
        y_true = np.asarray(test_set.get_metadata().target).astype(int)
        y_pred = np.asarray(eeg_classifier.predict(test_set))

        return EvaluationResult(
            precision=precision_score(y_true, y_pred, pos_label=self.pos_label, zero_division=0),
            recall=recall_score(y_true, y_pred, pos_label=self.pos_label, zero_division=0),
            accuracy=accuracy_score(y_true, y_pred),
            mcc=matthews_corrcoef(y_true, y_pred),
            confusion_matrix=confusion_matrix(y_true, y_pred),
        )
