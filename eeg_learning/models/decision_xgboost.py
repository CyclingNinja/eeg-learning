"""Gradient-boosted decision-stage model.

The decision stage turns a recording's per-window abnormality probabilities into
a single normal/abnormal verdict. Its features are a short fixed-width tabular
vector (a probability histogram, optionally concatenated with padded raw
probabilities), which is exactly the shape gradient boosting handles well.

:class:`XGBoostDecisionModel` is duck-typed to the torch models in
:mod:`eeg_learning.models.decision_models` rather than subclassing
``nn.Module``: it is callable as ``model(x, valid_len)`` and returns
log-probabilities of shape ``(n_samples, 2)``, so the shared evaluation path
(``_evaluate_decision_model`` and ``compute_decision_metrics``) needs no
branching on backend.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from xgboost import XGBClassifier

from eeg_learning.tools.logger import get_logger

log = get_logger(__name__)

#: Clamp for probabilities before taking logs, so a confident 0.0 does not
#: become ``-inf`` and poison the downstream metrics.
_PROB_EPS = 1e-7

#: Applied under any caller-supplied ``[decision.xgboost]`` values. These use the
#: modern xgboost 2.x spelling (``n_jobs``/``random_state``/``verbosity``), not
#: the deprecated ``nthread``/``seed``/``silent`` arguments.
DEFAULT_PARAMS = {
    "objective": "binary:logistic",
    "eval_metric": "logloss",
    "max_depth": 6,
    "learning_rate": 0.05,
    "n_estimators": 300,
    "subsample": 0.85,
    "colsample_bytree": 0.7,
    "min_child_weight": 1,
    "gamma": 0.0,
    "reg_alpha": 0.0,
    "reg_lambda": 1.0,
    "n_jobs": -1,
    "random_state": 0,
    "verbosity": 0,
}


class XGBoostDecisionModel:
    """An :class:`xgboost.XGBClassifier` wearing the decision-model interface.

    Parameters
    ----------
    **params
        Hyperparameters forwarded to :class:`xgboost.XGBClassifier`, layered over
        :data:`DEFAULT_PARAMS`. Typically the ``[decision.xgboost]`` config table.
    """

    def __init__(self, **params):
        self.params = {**DEFAULT_PARAMS, **params}
        self.classifier = XGBClassifier(**self.params)
        self.fitted = False

    def fit(
        self,
        train_features: np.ndarray,
        train_labels: np.ndarray,
        valid_features: np.ndarray | None = None,
        valid_labels: np.ndarray | None = None,
    ) -> tuple[list[float], list[float]]:
        """Fit the booster, returning per-round train and validation logloss.

        The curves stand in for the torch loop's per-epoch losses so both
        backends report the same ``train_loss``/``valid_loss`` columns.

        A validation set is optional: without one there is nothing to early-stop
        on, so early stopping is disabled for that fit and the returned
        validation curve mirrors the training curve. The same fallback covers a
        degenerate split whose validation set holds a class the training set
        never saw — xgboost rejects such an eval_set outright.
        """

        train_classes = set(np.unique(train_labels).tolist())
        if train_classes != {0, 1}:
            # xgboost requires labels to be exactly 0..n_classes-1, so a one-class
            # split fails deep inside sklearn with an opaque message. Say what is
            # actually wrong: there is nothing for the decision model to separate.
            raise ValueError(
                f"Decision training split contains a single class {sorted(train_classes)}; "
                "gradient boosting needs both. Widen decision.train_ratio, or check "
                "that the first-stage training detail covers both labels."
            )

        has_valid = valid_features is not None and len(valid_features) > 0
        if has_valid and not set(np.unique(valid_labels).tolist()) <= train_classes:
            log.warning(
                "Validation split contains classes absent from the training split; "
                "fitting without early stopping. Check the decision split ratios."
            )
            has_valid = False

        eval_set = [(train_features, train_labels)]
        if has_valid:
            eval_set.append((valid_features, valid_labels))
        else:
            # ``early_stopping_rounds`` watches the last eval_set entry; with only
            # the training set there, stopping early would fit noise in the data.
            self.classifier.set_params(early_stopping_rounds=None)

        self.classifier.fit(train_features, train_labels, eval_set=eval_set, verbose=False)
        self.fitted = True

        results = self.classifier.evals_result()
        metric = self.params.get("eval_metric", "logloss")
        train_losses = [float(value) for value in results["validation_0"][metric]]
        valid_losses = [float(value) for value in results["validation_1"][metric]] if has_valid else list(train_losses)
        return train_losses, valid_losses

    def __call__(self, x: torch.Tensor, valid_len: torch.Tensor | None = None) -> torch.Tensor:
        """Predict log-probabilities for a batch of decision features.

        Parameters
        ----------
        x : torch.Tensor
            Features of shape ``(batch_size, feature_dim)``.
        valid_len : torch.Tensor, optional
            Accepted and ignored, mirroring
            :meth:`~eeg_learning.models.decision_models.HistogramModel.forward`;
            the valid length is already baked into the feature vector.

        Returns
        -------
        torch.Tensor
            Log-probabilities of shape ``(batch_size, 2)``, matching the
            ``LogSoftmax`` output of the torch decision models.
        """

        if not self.fitted:
            raise RuntimeError("XGBoostDecisionModel must be fitted (or loaded) before it can predict.")

        features = np.asarray(x.detach().cpu().numpy(), dtype=np.float32)
        probabilities = np.clip(self.classifier.predict_proba(features), _PROB_EPS, 1.0)
        return torch.from_numpy(np.log(probabilities)).to(dtype=torch.float32)

    def eval(self) -> XGBoostDecisionModel:
        """No-op mode switch, for interface parity with ``nn.Module``."""
        return self

    def train(self, mode: bool = True) -> XGBoostDecisionModel:
        """No-op mode switch, for interface parity with ``nn.Module``."""
        return self

    def to(self, device) -> XGBoostDecisionModel:
        """No-op device move; the booster manages its own placement."""
        return self

    def save(self, path: str | Path) -> Path:
        """Write the booster to ``path`` in xgboost's native JSON format.

        JSON rather than pickle: it is readable, version-portable, and safe to
        load without executing arbitrary code.
        """

        if not self.fitted:
            raise RuntimeError("Refusing to save an unfitted XGBoostDecisionModel.")

        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        self.classifier.save_model(str(path))
        return path

    @classmethod
    def load(cls, path: str | Path, **params) -> XGBoostDecisionModel:
        """Rebuild a fitted model from a booster JSON written by :meth:`save`."""

        model = cls(**params)
        model.classifier.load_model(str(path))
        model.fitted = True
        return model
