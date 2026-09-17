from .base import AbstractModel
from .factory import ModelFactory, register
from . import tcn, vit, hybrid, braindecode_models  # trigger @register decorators

__all__ = ["AbstractModel", "ModelFactory", "register"]
