from abc import ABC, abstractmethod
import torch.nn as nn


class AbstractModel(ABC, nn.Module):
    def __init__(self, n_channels: int, n_classes: int, input_window_samples: int):
        super().__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.input_window_samples = input_window_samples

    @abstractmethod
    def forward(self, x): ...
