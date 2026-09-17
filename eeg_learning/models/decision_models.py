"""Second-stage decision models for EEG probability aggregation.

These models operate on aggregated first-stage model predictions
(histograms or raw probabilities) grouped at patient/session/recording level.
"""

from __future__ import annotations

import torch
from torch import nn
from torch.nn import init

from eeg_learning.models.decision_xgboost import XGBoostDecisionModel


class DecisionModel(nn.Module):
    """Raw probability decision model with optional adaptive pooling.

    Operates on raw first-stage probabilities (unwindowed predictions)
    and uses adaptive pooling or padding to create fixed-size features.

    Parameters
    ----------
    adap_pool : bool, default=True
        If True, uses AdaptiveAvgPool1d to pool to 10 bins.
        If False, pads/truncates to 20 samples and applies linear layer.
    """

    def __init__(self, adap_pool: bool = True):
        super().__init__()
        self.adap_pool = adap_pool

        if adap_pool:
            self.pooling = nn.AdaptiveAvgPool1d(10)
            self.classifier = nn.Linear(10, 2)
        else:
            self.classifier = nn.Linear(20, 2)

        self.log_softmax = nn.LogSoftmax(dim=1)
        init.normal_(self.classifier.weight, 0, 0.01)

    def forward(self, x: torch.Tensor, valid_len: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Parameters
        ----------
        x : torch.Tensor
            Input tensor of shape (batch_size, seq_len) containing raw probabilities.
        valid_len : torch.Tensor
            Tensor of shape (batch_size,) indicating valid length for each sample.

        Returns
        -------
        torch.Tensor
            Log-softmax predictions of shape (batch_size, 2).
        """
        if self.adap_pool:
            x = x[:, : valid_len.max()]
            x = self.pooling(x)
        x = self.classifier(x)
        return self.log_softmax(x)


class HistogramModel(nn.Module):
    """Histogram-based decision model with optional hybrid mode.

    Operates on histogram features derived from first-stage probabilities.
    Can optionally concatenate histogram with raw padded probabilities.

    Parameters
    ----------
    length : int, default=10
        Number of histogram bins.
    use_hybrid : bool, default=False
        If True, concatenates histogram with padded raw probabilities.
    hidden_layers : int, default=0
        Number of hidden layers (0 = linear classifier only).
    hidden_length : int, default=5
        Width of each hidden layer.
    """

    def __init__(
        self,
        length: int = 10,
        use_hybrid: bool = False,
        hidden_layers: int = 0,
        hidden_length: int = 5,
    ):
        super().__init__()
        self.use_hybrid = use_hybrid
        self.hidden_layers = hidden_layers
        self.hidden_length = hidden_length

        # Determine input dimension
        if use_hybrid:
            self.input_dim = length + 20  # histogram + padded raw
        else:
            self.input_dim = length

        # Build hidden layers if needed
        if self.hidden_layers > 0:
            self.hidden = nn.Sequential()
            self.hidden.add_module("hidden0", nn.Linear(self.input_dim, self.hidden_length))
            self.hidden.add_module("activation0", nn.ReLU())

            for i in range(1, self.hidden_layers):
                self.hidden.add_module(
                    f"hidden{i}",
                    nn.Linear(self.hidden_length, self.hidden_length),
                )
                self.hidden.add_module(f"activation{i}", nn.ReLU())

            # Initialize weights
            for layer in self.hidden:
                if hasattr(layer, "weight"):
                    init.normal_(layer.weight, 0, 0.01)

        # Classifier
        classifier_in_dim = self.hidden_length if self.hidden_layers > 0 else self.input_dim
        self.classifier = nn.Linear(classifier_in_dim, 2)
        init.normal_(self.classifier.weight, 0, 0.01)

        self.log_softmax = nn.LogSoftmax(dim=1)

    def forward(self, x: torch.Tensor, valid_len: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Parameters
        ----------
        x : torch.Tensor
            Input tensor of shape (batch_size, feature_dim) containing
            histogram or hybrid features.
        valid_len : torch.Tensor
            Tensor of shape (batch_size,) indicating valid length for each sample.
            (Unused for histogram model; kept for API consistency.)

        Returns
        -------
        torch.Tensor
            Log-softmax predictions of shape (batch_size, 2).
        """
        if self.hidden_layers > 0:
            x = self.hidden(x)
        x = self.classifier(x)
        return self.log_softmax(x)


#: Accepted values of ``[decision] backend``. The torch aliases select the
#: ``nn.Module`` models above; ``xgboost`` selects the gradient-boosted wrapper.
DECISION_BACKENDS = ("mlp", "torch", "xgboost")


def build_decision_model(decision_cfg: dict):
    """Construct a decision-stage model from the ``[decision]`` config.

    The ``backend`` key selects the learner: ``"mlp"`` (default, also spelled
    ``"torch"``) builds one of the :class:`~torch.nn.Module` models below, while
    ``"xgboost"`` builds an
    :class:`~eeg_learning.models.decision_xgboost.XGBoostDecisionModel` from the
    ``[decision.xgboost]`` table. Both consume the same features, so the two are
    directly comparable.

    Histogram features are required for patient/session aggregation, and they are
    also the default per-recording representation when ``use_his`` is enabled.
    """

    backend = (decision_cfg.get("backend") or "mlp").lower()
    if backend not in DECISION_BACKENDS:
        raise ValueError(f"Unknown decision backend '{backend}'. Available: {sorted(DECISION_BACKENDS)}")

    if backend == "xgboost":
        return XGBoostDecisionModel(**decision_cfg.get("xgboost", {}))

    use_his = decision_cfg.get("use_his", True)
    use_session = decision_cfg.get("use_session_or_patients")

    if use_session or use_his:
        return HistogramModel(
            length=decision_cfg.get("length", 10),
            use_hybrid=decision_cfg.get("use_hybrid", False),
            hidden_layers=decision_cfg.get("hidden_layers", 0),
            hidden_length=decision_cfg.get("hidden_length", 5),
        )
    return DecisionModel(adap_pool=decision_cfg.get("adap_pool", False))
