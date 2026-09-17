"""Tests for execution backends (eeg_learning/api/backends/)."""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import patch

import pytest

from eeg_learning.api.backends import (
    Backend,
    Job,
    JobHandle,
    JobKind,
    JobStatus,
    LocalBackend,
    get_backend,
)


@pytest.fixture
def train_job():
    return Job(
        kind=JobKind.TRAIN,
        config={"some": "config"},
        windows_path="data/saved_windows",
        output_dir="out",
        options={"model_id": "run42"},
    )


class TestJob:
    def test_options_default_empty(self):
        job = Job(kind=JobKind.TRAIN, config={}, windows_path="w", output_dir="o")
        assert job.options == {}

    def test_job_kind_values(self):
        assert JobKind.TRAIN.value == "train"
        assert JobKind.EVALUATE.value == "evaluate"


class TestGetBackend:
    def test_local_returns_backend(self):
        backend = get_backend("local")
        assert isinstance(backend, LocalBackend)
        assert isinstance(backend, Backend)

    def test_default_is_local(self):
        assert isinstance(get_backend(), LocalBackend)

    @pytest.mark.parametrize("name", ["azureml", "slurm"])
    def test_planned_backends_not_implemented(self, name):
        with pytest.raises(NotImplementedError, match=name):
            get_backend(name)

    def test_unknown_backend_raises_value_error(self):
        with pytest.raises(ValueError, match="Unknown backend"):
            get_backend("bogus")


class TestLocalBackendTrain:
    @patch("eeg_learning.api.backends.local.run_training")
    def test_submit_runs_training_and_returns_handle(self, mock_run, train_job):
        mock_run.return_value = SimpleNamespace(
            model_id="run42", model_path="/m/run42.pt", manifest_path="/m/run42.json"
        )
        backend = LocalBackend()
        handle = backend.submit(train_job)

        assert isinstance(handle, JobHandle)
        assert handle.backend == "local"
        mock_run.assert_called_once_with(
            {"some": "config"},
            windows_path="data/saved_windows",
            output_dir="out",
            model_id="run42",
        )

    @patch("eeg_learning.api.backends.local.run_training")
    def test_status_completed_after_submit(self, mock_run, train_job):
        mock_run.return_value = SimpleNamespace(
            model_id="run42", model_path="/m/run42.pt", manifest_path="/m/run42.json"
        )
        backend = LocalBackend()
        handle = backend.submit(train_job)
        assert backend.status(handle) == JobStatus.COMPLETED

    @patch("eeg_learning.api.backends.local.run_training")
    def test_result_is_json_serialisable_dict(self, mock_run, train_job):
        mock_run.return_value = SimpleNamespace(
            model_id="run42", model_path="/m/run42.pt", manifest_path="/m/run42.json"
        )
        backend = LocalBackend()
        handle = backend.submit(train_job)
        assert backend.result(handle) == {
            "model_id": "run42",
            "model_path": "/m/run42.pt",
            "manifest_path": "/m/run42.json",
        }

    @patch("eeg_learning.api.backends.local.run_training")
    def test_model_id_none_when_not_in_options(self, mock_run):
        mock_run.return_value = SimpleNamespace(model_id="auto", model_path="/m/a.pt", manifest_path="/m/a.json")
        job = Job(kind=JobKind.TRAIN, config={}, windows_path="w", output_dir="o")
        LocalBackend().submit(job)
        assert mock_run.call_args.kwargs["model_id"] is None

    @patch("eeg_learning.api.backends.local.run_training")
    def test_distinct_handles_and_results_across_submits(self, mock_run, train_job):
        mock_run.side_effect = [
            SimpleNamespace(model_id="a", model_path="/a.pt", manifest_path="/a.json"),
            SimpleNamespace(model_id="b", model_path="/b.pt", manifest_path="/b.json"),
        ]
        backend = LocalBackend()
        h1 = backend.submit(train_job)
        h2 = backend.submit(train_job)
        assert h1.id != h2.id
        assert backend.result(h1)["model_id"] == "a"
        assert backend.result(h2)["model_id"] == "b"


class TestLocalBackendErrors:
    def test_status_unknown_handle_is_pending(self):
        backend = LocalBackend()
        assert backend.status(JobHandle(id="nope", backend="local")) == JobStatus.PENDING

    def test_result_unknown_handle_raises_keyerror(self):
        backend = LocalBackend()
        with pytest.raises(KeyError):
            backend.result(JobHandle(id="nope", backend="local"))

    def test_unsupported_kind_raises_not_implemented(self):
        job = Job(kind=JobKind.EVALUATE, config={}, windows_path="w", output_dir="o")
        with pytest.raises(NotImplementedError, match="evaluate"):
            LocalBackend().submit(job)
