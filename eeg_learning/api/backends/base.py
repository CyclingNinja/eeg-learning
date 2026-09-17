"""Execution backends: the abstraction that lets one job run anywhere.

A :class:`Job` is a backend-agnostic description of work (what to run + config +
I/O paths). A :class:`Backend` knows how to execute it: ``submit`` hands the job
off and returns a :class:`JobHandle`; ``status`` and ``result`` poll on that
handle. The same job can therefore run in-process (``LocalBackend``) or be
dispatched to remote compute (Azure ML, Slurm) without callers changing.

The handle/status/result shape is deliberately poll-friendly so the cloud
backends — whose ``submit`` returns immediately with a native run/job id — slot in
without reworking the interface. See [[project-api-async]] for the sync-now plan.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum


class JobKind(str, Enum):
    """The kind of work a :class:`Job` describes."""

    TRAIN = "train"
    EVALUATE = "evaluate"


class JobStatus(str, Enum):
    """Lifecycle state of a submitted job.

    ``LocalBackend`` is synchronous, so callers only ever observe ``COMPLETED``
    (or a raised error); ``PENDING``/``RUNNING``/``FAILED`` exist for the async
    cloud backends that report progress over time.
    """

    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"


@dataclass
class Job:
    """A backend-agnostic description of work to run.

    Attributes
    ----------
    kind : JobKind
        Which operation to perform.
    config : dict
        Resolved configuration (see :func:`eeg_learning.config.load`).
    windows_path : str
        Directory of saved windowed data the job reads.
    output_dir : str
        Directory the job writes artifacts into.
    options : dict
        Kind-specific extras, e.g. ``{"model_id": ...}`` to name a training
        artifact or to select the model an evaluation should load.
    """

    kind: JobKind
    config: dict
    windows_path: str
    output_dir: str
    options: dict = field(default_factory=dict)


@dataclass
class JobHandle:
    """A reference to a submitted job, used to poll status and fetch results.

    Attributes
    ----------
    id : str
        Backend-unique identifier for this submission.
    backend : str
        Name of the backend that owns the job (e.g. ``"local"``).
    native : object
        Backend-native handle (an Azure run object, a Slurm job id, …) or
        ``None`` for in-process backends.
    """

    id: str
    backend: str
    native: object = None


class Backend(ABC):
    """Executes :class:`Job` instances.

    Implementations: ``LocalBackend`` (in-process, now); Azure ML and Slurm
    (remote, later). Subclasses set :attr:`name` and implement the three methods.
    """

    name: str = "backend"

    @abstractmethod
    def submit(self, job: Job) -> JobHandle:
        """Dispatch ``job`` and return a handle to track it."""

    @abstractmethod
    def status(self, handle: JobHandle) -> JobStatus:
        """Return the current :class:`JobStatus` for ``handle``."""

    @abstractmethod
    def result(self, handle: JobHandle) -> dict:
        """Return the job's result as a JSON-serialisable dict.

        Raises
        ------
        KeyError
            If no result is available for ``handle`` (unknown or not finished).
        """
