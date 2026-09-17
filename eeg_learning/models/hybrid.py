import torch
from torch import nn
from torch.nn import ConstantPad2d, init

from braindecode.models.deep4 import Deep4Net
from braindecode.models.shallow_fbcsp import ShallowFBCSPNet

from ._braindecode_compat import to_dense_prediction_model
from .base import AbstractModel
from .factory import register


@register("hybridnet")
class HybridNet(AbstractModel):
    """Hybrid ConvNet model from Schirrmeister et al 2017.

    See [Schirrmeister2017]_ for details.

    References
    ----------
    .. [Schirrmeister2017] Schirrmeister, R. T., et al. (2017).
       Deep learning with convolutional neural networks for EEG decoding and
       visualization. Human Brain Mapping, Aug. 2017.
       http://dx.doi.org/10.1002/hbm.23730
    """

    def __init__(self, n_channels, n_classes, input_window_samples):
        super().__init__(n_channels, n_classes, input_window_samples)

        deep_model = Deep4Net(
            n_chans=n_channels,
            n_outputs=n_classes,
            n_times=input_window_samples,
            n_filters_time=20,
            n_filters_spat=30,
            n_filters_2=40,
            n_filters_3=50,
            n_filters_4=60,
            final_conv_length=2,
        )
        shallow_model = ShallowFBCSPNet(
            n_chans=n_channels,
            n_outputs=n_classes,
            n_times=input_window_samples,
            n_filters_time=30,
            n_filters_spat=40,
            filter_time_length=28,
            final_conv_length=29,
        )

        reduced_deep_model = nn.Sequential()
        for name, module in deep_model.named_children():
            if name == "final_layer":
                conv = module.conv_classifier
                reduced_deep_model.add_module(
                    "deep_final_conv",
                    nn.Conv2d(
                        conv.in_channels,
                        60,
                        kernel_size=conv.kernel_size,
                        stride=conv.stride,
                    ),
                )
                break
            reduced_deep_model.add_module(name, module)

        reduced_shallow_model = nn.Sequential()
        for name, module in shallow_model.named_children():
            if name == "final_layer":
                conv = module.conv_classifier
                reduced_shallow_model.add_module(
                    "shallow_final_conv",
                    nn.Conv2d(
                        conv.in_channels,
                        40,
                        kernel_size=conv.kernel_size,
                        stride=conv.stride,
                    ),
                )
                break
            reduced_shallow_model.add_module(name, module)

        to_dense_prediction_model(reduced_deep_model)
        to_dense_prediction_model(reduced_shallow_model)
        self.reduced_deep_model = reduced_deep_model
        self.reduced_shallow_model = reduced_shallow_model
        self.final_conv = nn.Conv2d(
            100,
            n_classes,
            kernel_size=(input_window_samples - 521, 1),
            stride=1,
        )
        init.normal_(self.final_conv.weight, 0, 0.01)

    def forward(self, x):
        deep_out = self.reduced_deep_model(x)
        shallow_out = self.reduced_shallow_model(x)

        n_diff = deep_out.size()[2] - shallow_out.size()[2]
        if n_diff < 0:
            deep_out = ConstantPad2d((0, 0, -n_diff, 0), 0)(deep_out)
        elif n_diff > 0:
            shallow_out = ConstantPad2d((0, 0, n_diff, 0), 0)(shallow_out)

        merged = torch.cat((deep_out, shallow_out), dim=1)
        out = self.final_conv(merged)
        out = nn.LogSoftmax(dim=1)(out)
        return out.squeeze(3).squeeze(2)
