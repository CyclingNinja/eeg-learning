"""Forward-pass shape tests for the TCN model (eeg_learning/models/tcn.py).

These assert the final adaptive-pooling layer collapses the temporal axis to one
prediction per window -- including when the model is constructed for a different
window length than it is fed. That mismatch produced the uncollapsed [8, 2, 5380]
output which crashed NLLLoss in fresh_error.png.

Requires braindecode (pinned 1.6.x); skipped where not installed.
"""

from __future__ import annotations

import warnings

import pytest

torch = pytest.importorskip("torch")

try:
    from eeg_learning.models import ModelFactory
    from eeg_learning.models.tcn import Tcn
except ImportError:
    # tcn.py imports symbols from braindecode.modules. Skip cleanly if an
    # incompatible braindecode is installed rather than erroring at collection.
    pytest.skip(
        "TCN model requires braindecode 1.6.x (this branch's pin)",
        allow_module_level=True,
    )


def test_factory_resolves_tcn_to_local_collapsing_model():
    # Regression guard for the registry name clash: both this module and
    # braindecode_models registered "tcn", and the dict silently kept whichever
    # imported last (braindecode's per-timestep TCN), so the pipeline built a
    # model whose output did not collapse -> the [8, 2, 5380] NLLLoss crash.
    # The factory (the path the pipeline actually uses) must resolve "tcn" to the
    # local collapsing Tcn and produce one prediction per window.
    model = ModelFactory.create(
        "tcn",
        n_channels=19,
        n_classes=2,
        input_window_samples=6000,
        n_blocks=5,
        n_filters=55,
        kernel_size=11,
        drop_prob=0.05,
        add_log_softmax=True,
        last_layer_type="max_pool",
    ).eval()
    assert isinstance(model, Tcn)
    with torch.no_grad():
        out = model(torch.randn(8, 19, 6000))
    assert out.shape == (8, 2)


def _make_tcn(input_window_samples, last_layer_type):
    return Tcn(
        n_channels=19,
        n_classes=2,
        input_window_samples=input_window_samples,
        n_blocks=5,
        n_filters=55,
        kernel_size=11,
        drop_prob=0.05,
        add_log_softmax=True,
        last_layer_type=last_layer_type,
    ).eval()


@pytest.mark.parametrize("last_layer_type", ["max_pool", "ave_pool"])
def test_output_is_one_prediction_per_window(last_layer_type):
    # Model built and fed the same 6000-sample window (100 Hz x 60 s).
    model = _make_tcn(6000, last_layer_type)
    with torch.no_grad():
        out = model(torch.randn(8, 19, 6000))
    assert out.shape == (8, 2)


@pytest.mark.parametrize("last_layer_type", ["max_pool", "ave_pool"])
def test_window_length_mismatch_still_collapses(last_layer_type):
    # Model constructed for a short window but fed real 6000-sample windows.
    # Before adaptive pooling this left an uncollapsed [8, 2, 5380] output that
    # crashed NLLLoss; it must now collapse to one prediction per window.
    model = _make_tcn(600, last_layer_type)
    with torch.no_grad():
        out = model(torch.randn(8, 19, 6000))
    assert out.shape == (8, 2)


def test_tcn_forward_does_not_emit_dropout2d_warning():
    model = _make_tcn(6000, "max_pool")
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        with torch.no_grad():
            model(torch.randn(2, 19, 6000))

    assert not any("dropout2d" in str(w.message).lower() for w in caught)
