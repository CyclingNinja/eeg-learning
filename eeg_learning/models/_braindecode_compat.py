"""Vendored braindecode helpers removed in braindecode 1.x.

``squeeze_final_output`` (was ``braindecode.models.functions``) and the
free-function form of ``to_dense_prediction_model`` (was
``braindecode.models.util``; in 1.x it only survives as an ``EEGModuleMixin``
method) were public API through braindecode 0.x. The hand-rolled models in this
package (:mod:`~eeg_learning.models.hybrid`, :mod:`~eeg_learning.models.tcn`)
build plain ``nn.Sequential`` sub-models and still rely on the free-function
form, so we keep local copies matching the original braindecode behaviour.
"""

import numpy as np


def squeeze_final_output(x):
    """Remove empty trailing dims while always keeping the batch dim.

    Mirrors braindecode's original ``models.functions.squeeze_final_output``.
    """
    assert x.size()[3] == 1
    x = x[:, :, :, 0]
    if x.size()[2] == 1:
        x = x[:, :, 0]
    return x


def to_dense_prediction_model(model, axis=(2, 3)):
    """Convert a strided model into a dense-prediction model, in place.

    Removes strides and inserts equivalent dilations. Free-function form of
    braindecode's original ``models.util.to_dense_prediction_model`` (now the
    ``EEGModuleMixin.to_dense_prediction_model`` method).
    """
    if not hasattr(axis, "__len__"):
        axis = [axis]
    assert all(ax in [2, 3] for ax in axis), "Only 2 and 3 allowed for axis"
    axis = np.array(axis) - 2
    stride_so_far = np.array([1, 1])
    for module in model.modules():
        if hasattr(module, "dilation"):
            assert module.dilation == 1 or (module.dilation == (1, 1)), (
                "Dilation should equal 1 before conversion, maybe the model is already converted?"
            )
            new_dilation = [1, 1]
            for ax in axis:
                new_dilation[ax] = int(stride_so_far[ax])
            module.dilation = tuple(new_dilation)
        if hasattr(module, "stride"):
            if not hasattr(module.stride, "__len__"):
                module.stride = (module.stride, module.stride)
            stride_so_far *= np.array(module.stride)
            new_stride = list(module.stride)
            for ax in axis:
                new_stride[ax] = 1
            module.stride = tuple(new_stride)
