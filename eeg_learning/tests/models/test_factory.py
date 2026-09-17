"""Tests for the model registry and factory (eeg_learning/models/factory.py)."""

from __future__ import annotations

import pytest

from eeg_learning.models import factory
from eeg_learning.models.factory import ModelFactory, register


class DummyModel:
    """Plain stand-in for a registered model.

    The factory only calls cls(**core_args, **filtered_kwargs), so a lightweight
    class avoids pulling torch/braindecode into these tests. `dropout` is an
    optional kwarg used to exercise signature-based filtering.
    """

    def __init__(self, n_channels, n_classes, input_window_samples, dropout=0.5):
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.input_window_samples = input_window_samples
        self.dropout = dropout


@pytest.fixture
def clean_registry(monkeypatch):
    """Swap the module-global registry for an empty dict for the test's duration.

    register/create/available all resolve `_registry` from module globals at call
    time, so replacing the attribute isolates each test and prevents leaking dummy
    entries into the real registry populated by the model modules.
    """
    reg: dict = {}
    monkeypatch.setattr(factory, "_registry", reg)
    return reg


class TestRegister:
    def test_adds_class_under_name(self, clean_registry):
        register("dummy")(DummyModel)
        assert clean_registry["dummy"] is DummyModel

    def test_returns_class_unchanged(self, clean_registry):
        decorated = register("dummy")(DummyModel)
        # The decorator must be transparent so `@register(...)` above a class
        # definition leaves the class itself bound to its name.
        assert decorated is DummyModel

    def test_last_registration_wins_on_name_clash(self, clean_registry):
        class Other(DummyModel):
            pass

        register("dummy")(DummyModel)
        register("dummy")(Other)
        assert clean_registry["dummy"] is Other


class TestAvailable:
    def test_empty_registry(self, clean_registry):
        assert ModelFactory.available() == []

    def test_returns_sorted_names(self, clean_registry):
        register("zeta")(DummyModel)
        register("alpha")(DummyModel)
        register("mu")(DummyModel)
        assert ModelFactory.available() == ["alpha", "mu", "zeta"]


class TestCreate:
    def test_unknown_name_raises_valueerror(self, clean_registry):
        with pytest.raises(ValueError, match="Unknown model 'nope'"):
            ModelFactory.create("nope", n_channels=21, n_classes=2, input_window_samples=6000)

    def test_error_lists_available_models(self, clean_registry):
        register("real")(DummyModel)
        with pytest.raises(ValueError, match="real"):
            ModelFactory.create("nope", n_channels=21, n_classes=2, input_window_samples=6000)

    def test_instantiates_with_core_args(self, clean_registry):
        register("dummy")(DummyModel)
        model = ModelFactory.create("dummy", n_channels=21, n_classes=2, input_window_samples=6000)
        assert isinstance(model, DummyModel)
        assert model.n_channels == 21
        assert model.n_classes == 2
        assert model.input_window_samples == 6000

    def test_accepted_kwarg_passed_through(self, clean_registry):
        register("dummy")(DummyModel)
        model = ModelFactory.create("dummy", n_channels=21, n_classes=2, input_window_samples=6000, dropout=0.9)
        assert model.dropout == 0.9

    def test_unknown_kwarg_is_filtered_out(self, clean_registry):
        register("dummy")(DummyModel)
        # `not_a_param` is absent from DummyModel.__init__; without signature
        # filtering this would raise TypeError. The factory must silently drop it.
        model = ModelFactory.create(
            "dummy",
            n_channels=21,
            n_classes=2,
            input_window_samples=6000,
            not_a_param=123,
        )
        assert not hasattr(model, "not_a_param")

    def test_mixed_kwargs_keeps_only_accepted(self, clean_registry):
        register("dummy")(DummyModel)
        model = ModelFactory.create(
            "dummy",
            n_channels=21,
            n_classes=2,
            input_window_samples=6000,
            dropout=0.25,
            bogus="drop me",
        )
        assert model.dropout == 0.25
        assert not hasattr(model, "bogus")
