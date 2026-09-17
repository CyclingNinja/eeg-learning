"""Model artifacts: a checkpoint paired with a manifest describing how to rebuild it.

A saved braindecode/skorch checkpoint (``.pt``) holds *weights only*; reloading it
requires first reconstructing the exact model architecture. The manifest is a small
JSON file written beside the weights that stores that build recipe — the model name
and the ``ModelFactory.create`` kwargs — plus a snapshot of the training config. A
model can then be reloaded standalone, without the training config file or the
original windowed dataset.

The two files form a pair named by ``model_id``: ``<model_id>.pt`` and
``<model_id>.json``. The JSON is human-readable, safe to parse without unpickling,
and tooling-agnostic, so models can be listed and compared without importing torch.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

from eeg_learning.models import ModelFactory

#: Bumped when the manifest schema changes in a backwards-incompatible way.
MANIFEST_FORMAT_VERSION = 1


@dataclass
class ModelArtifact:
    """A trained model on disk: its weights, its manifest, and the parsed manifest.

    Attributes
    ----------
    model_id : str
        Identifier shared by the weights and manifest files (their stem).
    model_path : pathlib.Path
        Path to the ``.pt`` weights file written by skorch ``save_params``.
    manifest_path : pathlib.Path
        Path to the ``.json`` manifest describing how to rebuild the model.
    manifest : dict
        The parsed manifest contents.
    """

    model_id: str
    model_path: Path
    manifest_path: Path
    manifest: dict

    @property
    def model_name(self) -> str:
        """Registered model name recorded in the manifest."""
        return self.manifest["model"]["name"]

    @property
    def build_kwargs(self) -> dict:
        """The kwargs passed to :meth:`ModelFactory.create` at training time."""
        return self.manifest["model"]["build_kwargs"]

    def build_model(self):
        """Reconstruct the (untrained) model skeleton from the manifest.

        The returned module has the architecture the weights expect; load the
        weights into it via :meth:`Trainer.load` to obtain a usable classifier.

        Returns
        -------
        torch.nn.Module
            A freshly constructed model matching the saved checkpoint.
        """
        spec = self.manifest["model"]
        return ModelFactory.create(spec["name"], **spec["build_kwargs"])

    @classmethod
    def save(
        cls,
        eeg_classifier,
        *,
        model_id: str,
        model_name: str,
        build_kwargs: dict,
        output_dir,
        training_config: dict | None = None,
        extra: dict | None = None,
    ) -> ModelArtifact:
        """Persist a fitted classifier as a weights + manifest pair.

        Parameters
        ----------
        eeg_classifier : braindecode.EEGClassifier
            A fitted (or initialised) classifier to save.
        model_id : str
            Identifier used as the stem of both output files.
        model_name : str
            Registered model name, sufficient for :meth:`ModelFactory.create`.
        build_kwargs : dict
            The exact kwargs the model was created with, so the architecture can
            be reconstructed deterministically on load.
        output_dir : str or pathlib.Path
            Directory to write the pair into; created if missing.
        training_config : dict, optional
            Snapshot of the :class:`TrainingConfig` used, stored for provenance
            and to rebuild the classifier for evaluation/inference.
        extra : dict, optional
            Additional top-level manifest entries (e.g. dataset or git metadata).

        Returns
        -------
        ModelArtifact
            The artifact describing the files just written.
        """
        out = Path(output_dir)
        out.mkdir(parents=True, exist_ok=True)
        model_path = out / f"{model_id}.pt"
        manifest_path = out / f"{model_id}.json"

        eeg_classifier.save_params(str(model_path))

        manifest = {
            "format_version": MANIFEST_FORMAT_VERSION,
            "model_id": model_id,
            "created_at": datetime.now().isoformat(timespec="seconds"),
            "weights_file": model_path.name,
            "model": {"name": model_name, "build_kwargs": build_kwargs},
            "training": training_config or {},
        }
        if extra:
            manifest.update(extra)
        manifest_path.write_text(json.dumps(manifest, indent=2))

        return cls(
            model_id=model_id,
            model_path=model_path,
            manifest_path=manifest_path,
            manifest=manifest,
        )

    @classmethod
    def load(cls, model_id: str, *, models_dir) -> ModelArtifact:
        """Load an artifact by id from a directory of saved models.

        Parameters
        ----------
        model_id : str
            Identifier whose ``<model_id>.json`` manifest is read.
        models_dir : str or pathlib.Path
            Directory containing the weights + manifest pair.

        Returns
        -------
        ModelArtifact
            The loaded artifact. Call :meth:`build_model` to reconstruct the
            architecture before loading weights.

        Raises
        ------
        FileNotFoundError
            If the manifest or the weights file it references is missing.
        """
        out = Path(models_dir)
        manifest_path = out / f"{model_id}.json"
        if not manifest_path.exists():
            raise FileNotFoundError(f"No manifest for model_id '{model_id}' in {out}")
        manifest = json.loads(manifest_path.read_text())
        model_path = out / manifest.get("weights_file", f"{model_id}.pt")
        if not model_path.exists():
            raise FileNotFoundError(f"Manifest '{manifest_path}' references missing weights: {model_path}")
        return cls(
            model_id=model_id,
            model_path=model_path,
            manifest_path=manifest_path,
            manifest=manifest,
        )
