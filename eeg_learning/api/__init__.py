"""Production API layer over the EEG win-stack components.

Importable orchestration (jobs), self-describing model artifacts, and — later —
execution backends (local/Azure ML/Slurm), a CLI, and a FastAPI service. Distinct
from the DVC pipeline, which is for experimentation only.
"""

from .artifacts import ModelArtifact
from .jobs import TrainResult, run_training

__all__ = ["ModelArtifact", "TrainResult", "run_training"]
