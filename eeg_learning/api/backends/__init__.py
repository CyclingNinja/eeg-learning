"""Execution backends and a small factory for selecting one by name."""

from .base import Backend, Job, JobHandle, JobKind, JobStatus
from .local import LocalBackend

__all__ = [
    "Backend",
    "Job",
    "JobHandle",
    "JobKind",
    "JobStatus",
    "LocalBackend",
    "get_backend",
]

#: Backends not yet implemented; named here so ``get_backend`` gives a clear error.
_PLANNED = {"azureml", "slurm"}


def get_backend(name: str = "local", **kwargs) -> Backend:
    """Return a backend instance by name.

    Parameters
    ----------
    name : str
        ``"local"`` (implemented), or ``"azureml"`` / ``"slurm"`` (planned).
    **kwargs
        Backend-specific construction options (unused by ``LocalBackend``).

    Raises
    ------
    NotImplementedError
        For a planned-but-unbuilt backend.
    ValueError
        For an unknown backend name.
    """
    if name == "local":
        return LocalBackend(**kwargs)
    if name in _PLANNED:
        raise NotImplementedError(f"Backend '{name}' is not implemented yet")
    raise ValueError(f"Unknown backend '{name}'. Available: ['local']")
