"""Tests for the gradient-boosted decision model (eeg_learning/models/decision_xgboost.py)."""

from __future__ import annotations

import numpy as np
import pytest
import torch

from eeg_learning.models.decision_models import HistogramModel, build_decision_model
from eeg_learning.models.decision_xgboost import XGBoostDecisionModel

# Small, fast boosters: these tests check the interface, not predictive quality.
FAST_PARAMS = {"n_estimators": 5, "max_depth": 2, "random_state": 0}


@pytest.fixture
def separable_data():
    """Ten-bin histogram-shaped features with a trivially learnable split."""
    rng = np.random.default_rng(0)
    positives = rng.uniform(0.6, 1.0, size=(16, 10)).astype(np.float32)
    negatives = rng.uniform(0.0, 0.4, size=(16, 10)).astype(np.float32)
    features = np.concatenate([positives, negatives])
    labels = np.concatenate([np.ones(16, dtype=np.int64), np.zeros(16, dtype=np.int64)])
    return features, labels


@pytest.fixture
def fitted_model(separable_data):
    features, labels = separable_data
    model = XGBoostDecisionModel(**FAST_PARAMS)
    model.fit(features[:24], labels[:24], features[24:], labels[24:])
    return model


class TestFit:
    """Fitting and the loss curves that stand in for per-epoch losses."""

    def test_returns_loss_curves(self, separable_data):
        features, labels = separable_data
        model = XGBoostDecisionModel(**FAST_PARAMS)

        train_losses, valid_losses = model.fit(features[:24], labels[:24], features[24:], labels[24:])

        assert len(train_losses) == len(valid_losses) > 0
        assert all(isinstance(value, float) for value in train_losses)
        assert model.fitted

    def test_fits_without_validation_set(self, separable_data):
        """No validation split: early stopping is disabled rather than crashing."""
        features, labels = separable_data
        model = XGBoostDecisionModel(early_stopping_rounds=3, **FAST_PARAMS)

        train_losses, valid_losses = model.fit(features, labels)

        assert train_losses == valid_losses
        assert model.fitted

    def test_learns_a_separable_split(self, separable_data):
        features, labels = separable_data
        model = XGBoostDecisionModel(**FAST_PARAMS)
        model.fit(features, labels)

        predictions = torch.argmax(model(torch.from_numpy(features)), dim=-1).numpy()

        assert (predictions == labels).mean() > 0.9


class TestPredict:
    """The ``model(x, valid_len)`` contract shared with the torch models."""

    def test_returns_log_probabilities(self, fitted_model, separable_data):
        features, _ = separable_data
        x = torch.from_numpy(features)

        output = fitted_model(x, torch.full((len(features),), 10))

        assert output.shape == (len(features), 2)
        assert output.dtype == torch.float32
        assert torch.all(output <= 0)
        assert torch.allclose(output.exp().sum(dim=-1), torch.ones(len(features)), atol=1e-5)

    def test_valid_len_is_optional(self, fitted_model, separable_data):
        features, _ = separable_data
        x = torch.from_numpy(features)

        assert torch.equal(fitted_model(x), fitted_model(x, torch.full((len(features),), 3)))

    def test_predicting_before_fitting_raises(self, separable_data):
        features, _ = separable_data
        model = XGBoostDecisionModel(**FAST_PARAMS)

        with pytest.raises(RuntimeError, match="must be fitted"):
            model(torch.from_numpy(features))

    def test_mode_and_device_switches_are_no_ops(self, fitted_model):
        assert fitted_model.eval() is fitted_model
        assert fitted_model.train() is fitted_model
        assert fitted_model.to("cpu") is fitted_model


class TestSaveLoad:
    """Round-tripping through xgboost's native JSON format."""

    def test_round_trip_preserves_predictions(self, fitted_model, separable_data, tmp_path):
        features, _ = separable_data
        x = torch.from_numpy(features)
        path = tmp_path / "booster.json"

        fitted_model.save(path)
        reloaded = XGBoostDecisionModel.load(path, **FAST_PARAMS)

        assert path.exists()
        assert torch.allclose(fitted_model(x), reloaded(x), atol=1e-6)

    def test_saving_unfitted_raises(self, tmp_path):
        model = XGBoostDecisionModel(**FAST_PARAMS)

        with pytest.raises(RuntimeError, match="unfitted"):
            model.save(tmp_path / "booster.json")


class TestBuildDecisionModelDispatch:
    """``[decision] backend`` selects the learner."""

    def test_defaults_to_torch_backend(self):
        assert isinstance(build_decision_model({"use_his": True}), HistogramModel)

    @pytest.mark.parametrize("backend", ["mlp", "torch"])
    def test_torch_aliases(self, backend):
        assert isinstance(build_decision_model({"backend": backend, "use_his": True}), HistogramModel)

    def test_xgboost_backend_receives_hyperparameters(self):
        model = build_decision_model({"backend": "xgboost", "xgboost": {"max_depth": 3}})

        assert isinstance(model, XGBoostDecisionModel)
        assert model.params["max_depth"] == 3

    def test_unknown_backend_raises(self):
        with pytest.raises(ValueError, match="Unknown decision backend"):
            build_decision_model({"backend": "lightgbm"})
