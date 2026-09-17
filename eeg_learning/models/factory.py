import inspect

_registry: dict = {}


def register(name: str):
    def decorator(cls):
        _registry[name] = cls
        return cls

    return decorator


class ModelFactory:
    @staticmethod
    def create(
        name: str,
        n_channels: int,
        n_classes: int,
        input_window_samples: int,
        **kwargs,
    ):
        if name not in _registry:
            raise ValueError(f"Unknown model '{name}'. Available: {sorted(_registry)}")
        cls = _registry[name]
        accepted_params = inspect.signature(cls.__init__).parameters
        filtered_kwargs = {k: v for k, v in kwargs.items() if k in accepted_params}
        return cls(
            n_channels=n_channels,
            n_classes=n_classes,
            input_window_samples=input_window_samples,
            **filtered_kwargs,
        )

    @staticmethod
    def available() -> list[str]:
        return sorted(_registry)
