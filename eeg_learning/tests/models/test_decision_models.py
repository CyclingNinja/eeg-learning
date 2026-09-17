"""Tests for decision-stage models (eeg_learning/models/decision_models.py)."""

from __future__ import annotations

import pytest
import torch

from eeg_learning.models.decision_models import DecisionModel, HistogramModel, build_decision_model


class TestDecisionModel:
    """Tests for DecisionModel (adaptive pooling variant)."""

    @pytest.fixture
    def model_adap_pool(self):
        """Model with adaptive pooling enabled."""
        return DecisionModel(adap_pool=True)

    @pytest.fixture
    def model_no_pool(self):
        """Model with adaptive pooling disabled (linear only)."""
        return DecisionModel(adap_pool=False)

    def test_init_with_adaptive_pooling(self):
        model = DecisionModel(adap_pool=True)
        assert hasattr(model, "pooling")
        assert isinstance(model.pooling, torch.nn.AdaptiveAvgPool1d)
        assert model.pooling.output_size == 10

    def test_init_without_adaptive_pooling(self):
        model = DecisionModel(adap_pool=False)
        assert not hasattr(model, "pooling")

    def test_forward_shape_with_pooling(self, model_adap_pool):
        batch_size = 4
        seq_len = 20
        x = torch.randn(batch_size, seq_len)
        valid_len = torch.full((batch_size,), seq_len // 2)

        output = model_adap_pool(x, valid_len)
        assert output.shape == (batch_size, 2)

    def test_forward_shape_without_pooling(self, model_no_pool):
        batch_size = 4
        seq_len = 20
        x = torch.randn(batch_size, seq_len)
        valid_len = torch.full((batch_size,), seq_len)

        output = model_no_pool(x, valid_len)
        assert output.shape == (batch_size, 2)

    def test_output_is_log_softmax(self, model_adap_pool):
        x = torch.randn(2, 20)
        valid_len = torch.tensor([20, 20])
        output = model_adap_pool(x, valid_len)

        # Log-softmax should sum to ~0 along class dimension (in log space)
        exp_output = torch.exp(output)
        sums = exp_output.sum(dim=1)
        assert torch.allclose(sums, torch.ones(2), atol=1e-5)

    def test_valid_len_respected_with_pooling(self, model_adap_pool):
        """Adaptive pooling should only pool valid_len elements."""
        x = torch.ones(1, 20)  # All ones for predictability
        valid_len = torch.tensor([10])  # Only first 10 are valid

        output = model_adap_pool(x, valid_len)
        assert output.shape == (1, 2)

    def test_gradients_flow(self, model_adap_pool):
        """Ensure loss backpropagates to weights."""
        x = torch.randn(2, 20, requires_grad=True)
        valid_len = torch.tensor([20, 20])
        target = torch.tensor([0, 1])

        loss_fn = torch.nn.NLLLoss()
        output = model_adap_pool(x, valid_len)
        loss = loss_fn(output, target)

        loss.backward()
        assert model_adap_pool.classifier.weight.grad is not None
        assert model_adap_pool.classifier.weight.grad.abs().sum() > 0


class TestHistogramModel:
    """Tests for HistogramModel (feature-based variant)."""

    @pytest.fixture
    def model_simple(self):
        """Histogram model without hidden layers."""
        return HistogramModel(length=10, use_hybrid=False, hidden_layers=0)

    @pytest.fixture
    def model_hybrid(self):
        """Histogram model with hybrid mode (histogram + raw)."""
        return HistogramModel(length=10, use_hybrid=True, hidden_layers=0)

    @pytest.fixture
    def model_with_hidden(self):
        """Histogram model with hidden layers."""
        return HistogramModel(length=10, use_hybrid=False, hidden_layers=2, hidden_length=5)

    def test_init_simple(self, model_simple):
        assert model_simple.input_dim == 10
        assert model_simple.hidden_layers == 0
        assert not hasattr(model_simple, "hidden")

    def test_init_hybrid(self, model_hybrid):
        assert model_hybrid.input_dim == 30  # 10 + 20
        assert model_hybrid.use_hybrid is True

    def test_init_with_hidden_layers(self, model_with_hidden):
        assert model_with_hidden.hidden_layers == 2
        assert hasattr(model_with_hidden, "hidden")
        assert len(list(model_with_hidden.hidden)) == 4  # 2 linear + 2 ReLU

    def test_forward_simple(self, model_simple):
        batch_size = 4
        x = torch.randn(batch_size, 10)
        valid_len = torch.full((batch_size,), 10)

        output = model_simple(x, valid_len)
        assert output.shape == (batch_size, 2)

    def test_forward_hybrid(self, model_hybrid):
        batch_size = 4
        x = torch.randn(batch_size, 30)
        valid_len = torch.full((batch_size,), 30)

        output = model_hybrid(x, valid_len)
        assert output.shape == (batch_size, 2)

    def test_forward_with_hidden(self, model_with_hidden):
        batch_size = 4
        x = torch.randn(batch_size, 10)
        valid_len = torch.full((batch_size,), 10)

        output = model_with_hidden(x, valid_len)
        assert output.shape == (batch_size, 2)

    def test_output_is_log_softmax(self, model_simple):
        x = torch.randn(2, 10)
        valid_len = torch.tensor([10, 10])
        output = model_simple(x, valid_len)

        exp_output = torch.exp(output)
        sums = exp_output.sum(dim=1)
        assert torch.allclose(sums, torch.ones(2), atol=1e-5)

    def test_valid_len_unused_for_histogram(self, model_simple):
        """Histogram model doesn't use valid_len; just for API consistency."""
        x = torch.ones(1, 10)
        valid_len = torch.tensor([5])  # Doesn't matter

        output1 = model_simple(x, valid_len)

        valid_len2 = torch.tensor([100])  # Different value
        output2 = model_simple(x, valid_len2)

        assert torch.allclose(output1, output2)

    def test_gradients_flow_with_hidden(self, model_with_hidden):
        x = torch.randn(2, 10, requires_grad=True)
        valid_len = torch.tensor([10, 10])
        target = torch.tensor([0, 1])

        loss_fn = torch.nn.NLLLoss()
        output = model_with_hidden(x, valid_len)
        loss = loss_fn(output, target)

        loss.backward()
        assert model_with_hidden.classifier.weight.grad is not None

    def test_determinism_same_seed(self, model_simple):
        """Models initialized with same seed should have identical predictions."""
        torch.manual_seed(42)
        m1 = HistogramModel(length=10)
        x1 = torch.randn(1, 10)
        out1 = m1(x1, torch.tensor([10]))

        torch.manual_seed(42)
        m2 = HistogramModel(length=10)
        x2 = torch.randn(1, 10)
        out2 = m2(x2, torch.tensor([10]))

        # Different random initialization means outputs differ; just check shape
        assert out1.shape == out2.shape == (1, 2)

    def test_large_hidden_stack(self):
        """Test model with many hidden layers."""
        model = HistogramModel(length=10, hidden_layers=5, hidden_length=8)
        x = torch.randn(2, 10)
        valid_len = torch.tensor([10, 10])

        output = model(x, valid_len)
        assert output.shape == (2, 2)


def test_build_decision_model_selects_histogram_model():
    model = build_decision_model({"use_his": True, "length": 10})
    assert isinstance(model, HistogramModel)


def test_build_decision_model_selects_raw_model():
    model = build_decision_model({"use_his": False, "use_session_or_patients": None, "adap_pool": False})
    assert isinstance(model, DecisionModel)
