from torch import nn

from braindecode.models import (
    Deep4Net as _Deep4Net,
    EEGNet as _EEGNet,
    HybridNet as _HybridNet,
    ShallowFBCSPNet as _ShallowFBCSPNet,
    SleepStagerBlanco2020 as _SleepStagerBlanco2020,
    SleepStagerChambon2018 as _SleepStagerChambon2018,
    TCN as _TCN,
    TIDNet as _TIDNet,
    USleep as _USleep,
)

from .base import AbstractModel
from .factory import register

# braindecode 1.x standardised every model on ``n_chans`` / ``n_outputs`` /
# ``n_times``. These wrappers keep the project-facing API
# (``n_channels`` / ``n_classes`` / ``input_window_samples``) stable and
# translate to the braindecode names internally. Models also emit raw logits
# now (``add_log_softmax`` was removed), so the trainer pairs them with
# ``CrossEntropyLoss``.


@register("deep4")
class Deep4Net(AbstractModel):
    def __init__(
        self,
        n_channels,
        n_classes,
        input_window_samples,
        final_conv_length="auto",
        drop_prob=0.5,
        deep4_n_filters_time=25,
        deep4_n_filters_spat=25,
        deep4_filter_time_length=10,
        deep4_pool_time_length=3,
        deep4_pool_time_stride=3,
        deep4_n_filters_2=50,
        deep4_filter_length_2=10,
        deep4_n_filters_3=100,
        deep4_filter_length_3=10,
        deep4_n_filters_4=200,
        deep4_filter_length_4=10,
        deep4_first_pool_mode="max",
        deep4_later_pool_mode="max",
        first_nonlin=nn.ELU,
        later_nonlin=nn.ELU,
    ):
        super().__init__(n_channels, n_classes, input_window_samples)
        self._inner = _Deep4Net(
            n_chans=n_channels,
            n_outputs=n_classes,
            n_times=input_window_samples,
            final_conv_length=final_conv_length,
            n_filters_time=deep4_n_filters_time,
            n_filters_spat=deep4_n_filters_spat,
            filter_time_length=deep4_filter_time_length,
            pool_time_length=deep4_pool_time_length,
            pool_time_stride=deep4_pool_time_stride,
            n_filters_2=deep4_n_filters_2,
            filter_length_2=deep4_filter_length_2,
            n_filters_3=deep4_n_filters_3,
            filter_length_3=deep4_filter_length_3,
            n_filters_4=deep4_n_filters_4,
            filter_length_4=deep4_filter_length_4,
            first_pool_mode=deep4_first_pool_mode,
            later_pool_mode=deep4_later_pool_mode,
            drop_prob=drop_prob,
            double_time_convs=False,
            split_first_layer=True,
            batch_norm=True,
            batch_norm_alpha=0.1,
            stride_before_pool=False,
            activation_first_conv_nonlin=first_nonlin,
            activation_later_conv_nonlin=later_nonlin,
        )

    def forward(self, x):
        return self._inner(x)


@register("shallow_smac")
class ShallowSMACNet(AbstractModel):
    def __init__(
        self,
        n_channels,
        n_classes,
        input_window_samples,
        final_conv_length="auto",
        drop_prob=0.5,
        shallow_n_filters_time=40,
        shallow_filter_time_length=25,
        shallow_n_filters_spat=40,
        shallow_pool_time_length=75,
        shallow_pool_time_stride=15,
        shallow_split_first_layer=True,
        shallow_batch_norm=True,
        shallow_batch_norm_alpha=0.1,
    ):
        super().__init__(n_channels, n_classes, input_window_samples)
        self._inner = _ShallowFBCSPNet(
            n_chans=n_channels,
            n_outputs=n_classes,
            n_times=input_window_samples,
            n_filters_time=shallow_n_filters_time,
            filter_time_length=shallow_filter_time_length,
            n_filters_spat=shallow_n_filters_spat,
            pool_time_length=shallow_pool_time_length,
            pool_time_stride=shallow_pool_time_stride,
            final_conv_length=final_conv_length,
            split_first_layer=shallow_split_first_layer,
            batch_norm=shallow_batch_norm,
            batch_norm_alpha=shallow_batch_norm_alpha,
            drop_prob=drop_prob,
        )

    def forward(self, x):
        return self._inner(x)


@register("eegnetv4")
class EEGNet(AbstractModel):
    """braindecode 1.x consolidated EEGNetv4 into a single ``EEGNet`` class."""

    def __init__(
        self,
        n_channels,
        n_classes,
        input_window_samples,
        final_conv_length="auto",
        drop_prob=0.5,
        pool_mode="mean",
        F1=8,
        D=2,
        F2=16,
        kernel_length=64,
        third_kernel_size=(8, 4),
    ):
        super().__init__(n_channels, n_classes, input_window_samples)
        self._inner = _EEGNet(
            n_chans=n_channels,
            n_outputs=n_classes,
            n_times=input_window_samples,
            final_conv_length=final_conv_length,
            pool_mode=pool_mode,
            F1=F1,
            D=D,
            F2=F2,
            kernel_length=kernel_length,
            third_kernel_size=third_kernel_size,
            drop_prob=drop_prob,
        )

    def forward(self, x):
        return self._inner(x)


# Registered under its own name, not "tcn": the local collapsing Tcn in tcn.py
# owns that key. Both registered "tcn" once and the dict silently kept whichever
# imported last, so the pipeline built braindecode's per-timestep TCN and its
# uncollapsed output crashed the loss.
@register("braindecode_tcn")
class BraindecodeTCN(AbstractModel):
    def __init__(
        self,
        n_channels,
        n_classes,
        input_window_samples,
        n_blocks=8,
        n_filters=2,
        kernel_size=12,
        drop_prob=0.5,
    ):
        super().__init__(n_channels, n_classes, input_window_samples)
        # braindecode's TCN is length-agnostic: it takes no n_times.
        self._inner = _TCN(
            n_chans=n_channels,
            n_outputs=n_classes,
            n_blocks=n_blocks,
            n_filters=n_filters,
            kernel_size=kernel_size,
            drop_prob=drop_prob,
        )

    def forward(self, x):
        return self._inner(x)


@register("sleep2020")
class SleepNet2020(AbstractModel):
    def __init__(
        self,
        n_channels,
        n_classes,
        input_window_samples,
        sampling_freq=100,
        n_conv_chans=20,
        input_size_s=60,
        n_groups=3,
        max_pool_size=2,
        drop_prob=0.5,
        apply_batch_norm=False,
        return_feats=False,
    ):
        super().__init__(n_channels, n_classes, input_window_samples)
        self._inner = _SleepStagerBlanco2020(
            n_chans=n_channels,
            sfreq=sampling_freq,
            n_conv_chans=n_conv_chans,
            input_window_seconds=input_size_s,
            n_outputs=n_classes,
            n_groups=n_groups,
            max_pool_size=max_pool_size,
            drop_prob=drop_prob,
            apply_batch_norm=apply_batch_norm,
            return_feats=return_feats,
        )

    def forward(self, x):
        return self._inner(x)


@register("sleep2018")
class SleepNet2018(AbstractModel):
    def __init__(
        self,
        n_channels,
        n_classes,
        input_window_samples,
        sampling_freq=100,
        n_conv_chs=8,
        time_conv_size_s=0.5,
        max_pool_size_s=0.125,
        pad_size_s=0.25,
        input_size_s=60,
        drop_prob=0.5,
        apply_batch_norm=False,
        return_feats=False,
    ):
        super().__init__(n_channels, n_classes, input_window_samples)
        self._inner = _SleepStagerChambon2018(
            n_chans=n_channels,
            sfreq=sampling_freq,
            n_conv_chs=n_conv_chs,
            time_conv_size_s=time_conv_size_s,
            max_pool_size_s=max_pool_size_s,
            pad_size_s=pad_size_s,
            input_window_seconds=input_size_s,
            n_outputs=n_classes,
            drop_prob=drop_prob,
            apply_batch_norm=apply_batch_norm,
            return_feats=return_feats,
        )

    def forward(self, x):
        return self._inner(x)


@register("usleep")
class USleepWrapper(AbstractModel):
    def __init__(
        self,
        n_channels,
        n_classes,
        input_window_samples,
        sampling_freq=100,
        depth=12,
        n_time_filters=5,
        complexity_factor=1.67,
        with_skip_connection=True,
        input_size_s=60,
        time_conv_size_s=0.0703125,
        ensure_odd_conv_size=False,
        apply_softmax=False,
    ):
        super().__init__(n_channels, n_classes, input_window_samples)
        self._inner = _USleep(
            n_chans=n_channels,
            sfreq=sampling_freq,
            depth=depth,
            n_time_filters=n_time_filters,
            complexity_factor=complexity_factor,
            with_skip_connection=with_skip_connection,
            n_outputs=n_classes,
            input_window_seconds=input_size_s,
            time_conv_size_s=time_conv_size_s,
            ensure_odd_conv_size=ensure_odd_conv_size,
            apply_softmax=apply_softmax,
        )

    def forward(self, x):
        return self._inner(x)


@register("tidnet")
class TIDNetWrapper(AbstractModel):
    def __init__(
        self,
        n_channels,
        n_classes,
        input_window_samples,
        s_growth=24,
        t_filters=32,
        drop_prob=0.4,
        pooling=15,
        temp_layers=2,
        spat_layers=2,
        temp_span=0.05,
        bottleneck=3,
        summary=-1,
    ):
        super().__init__(n_channels, n_classes, input_window_samples)
        self._inner = _TIDNet(
            n_chans=n_channels,
            n_outputs=n_classes,
            n_times=input_window_samples,
            s_growth=s_growth,
            t_filters=t_filters,
            drop_prob=drop_prob,
            pooling=pooling,
            temp_layers=temp_layers,
            spat_layers=spat_layers,
            temp_span=temp_span,
            bottleneck=bottleneck,
            summary=summary,
        )

    def forward(self, x):
        return self._inner(x)


@register("hybridnet")
class HybridNetWrapper(AbstractModel):
    def __init__(self, n_channels, n_classes, input_window_samples):
        super().__init__(n_channels, n_classes, input_window_samples)
        self._inner = _HybridNet(
            n_chans=n_channels,
            n_outputs=n_classes,
            n_times=input_window_samples,
        )

    def forward(self, x):
        return self._inner(x)
