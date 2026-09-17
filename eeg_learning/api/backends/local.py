"""In-process execution backend.

``LocalBackend`` runs jobs synchronously in the current process: :meth:`submit`
executes the job to completion before returning, so :meth:`status` reports
``COMPLETED`` at once and :meth:`result` is immediately available. Failures
propagate as exceptions (fail loud — the natural behaviour for a local dev run),
rather than being captured as a ``FAILED`` status the way an async backend would.
"""

from __future__ import annotations

import uuid

from eeg_learning.api.backends.base import Backend, Job, JobHandle, JobKind, JobStatus
from eeg_learning.api.jobs import run_training


class LocalBackend(Backend):
    """Execute jobs in-process, synchronously."""

    name = "local"

    def __init__(self):
        self._results: dict[str, dict] = {}

    def submit(self, job: Job) -> JobHandle:
        """Run ``job`` to completion and return a handle to its stored result."""
        result = self._run(job)
        handle = JobHandle(id=uuid.uuid4().hex, backend=self.name)
        self._results[handle.id] = result
        return handle

    def _run(self, job: Job) -> dict:
        """Dispatch on job kind, returning a JSON-serialisable result dict."""
        if job.kind == JobKind.TRAIN:
            result = run_training(
                job.config,
                windows_path=job.windows_path,
                output_dir=job.output_dir,
                model_id=job.options.get("model_id"),
            )
            return {
                "model_id": result.model_id,
                "model_path": str(result.model_path),
                "manifest_path": str(result.manifest_path),
            }
        raise NotImplementedError(f"LocalBackend does not yet support job kind '{job.kind.value}'")

    def status(self, handle: JobHandle) -> JobStatus:
        """Return ``COMPLETED`` once the job has run, else ``PENDING``."""
        return JobStatus.COMPLETED if handle.id in self._results else JobStatus.PENDING

    def result(self, handle: JobHandle) -> dict:
        if handle.id not in self._results:
            raise KeyError(f"No result for job '{handle.id}'")
        return self._results[handle.id]
