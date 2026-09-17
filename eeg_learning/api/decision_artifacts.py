"""Decision-stage artifacts: a trained second-stage model plus its manifest.

The counterpart to :mod:`eeg_learning.api.artifacts`, which does the same job
for first-stage (windowed EEG) models. A decision model is only meaningful
alongside the feature recipe it was trained on — a booster fitted on 10-bin
histograms cannot score hybrid features — so the manifest records the feature
configuration together with the backend, its hyperparameters, the split that
produced it, and the metrics it scored.

The pair is named by ``model_id``: ``<model_id>.pt`` (torch ``state_dict``) or
``<model_id>.json`` (xgboost booster), beside ``<model_id>.manifest.json``.
Decision artifacts live in their own directory
(``target/saved_models/decision_model`` by default, beside the first stage's
``window_model``) so the two never mix and each is a distinct DVC output.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import torch

from eeg_learning.models.decision_models import build_decision_model
from eeg_learning.models.decision_xgboost import XGBoostDecisionModel

#: Bumped when the manifest schema changes in a backwards-incompatible way.
MANIFEST_FORMAT_VERSION = 1

#: ``[decision]`` keys that determine the feature vector and model architecture,
#: and so must travel with the weights for a model to be reloadable standalone.
FEATURE_KEYS = (
    "use_his",
    "use_hybrid",
    "length",
    "use_session_or_patients",
    "adap_pool",
    "hidden_layers",
    "hidden_length",
)

#: ``[decision]`` keys describing how the data was partitioned.
SPLIT_KEYS = ("train_ratio", "valid_ratio", "fix_testset")


def feature_config(decision_cfg: dict) -> dict:
    """Extract the reload-critical subset of a ``[decision]`` config."""
    return {key: decision_cfg[key] for key in FEATURE_KEYS if key in decision_cfg}


@dataclass
class DecisionArtifact:
    """A trained decision model on disk: its weights, manifest, and parsed manifest.

    Attributes
    ----------
    model_id : str
        Identifier shared by the weights and manifest files.
    model_path : pathlib.Path
        Path to the weights (``.pt`` state dict, or ``.json`` booster).
    manifest_path : pathlib.Path
        Path to the ``<model_id>.manifest.json`` describing how to rebuild it.
    manifest : dict
        The parsed manifest contents.
    """

    model_id: str
    model_path: Path
    manifest_path: Path
    manifest: dict

    @property
    def backend(self) -> str:
        """Decision backend recorded in the manifest (``"mlp"`` or ``"xgboost"``)."""
        return self.manifest["backend"]

    @property
    def build_cfg(self) -> dict:
        """The ``[decision]``-shaped dict needed to rebuild this model."""
        return {
            **self.manifest.get("features", {}),
            "backend": self.backend,
            "xgboost": self.manifest.get("hyperparameters", {}),
        }

    def build_model(self):
        """Rebuild the trained model, weights loaded, ready to evaluate.

        Returns
        -------
        object
            A model callable as ``model(x, valid_len)``, in eval mode.
        """

        if self.backend == "xgboost":
            return XGBoostDecisionModel.load(self.model_path, **self.manifest.get("hyperparameters", {}))

        model = build_decision_model(self.build_cfg)
        model.load_state_dict(torch.load(self.model_path, map_location="cpu"))
        model.eval()
        return model

    @classmethod
    def save(
        cls,
        model,
        *,
        model_id: str,
        backend: str,
        decision_cfg: dict,
        output_dir,
        metrics: dict | None = None,
        repetition: int | None = None,
        extra: dict | None = None,
    ) -> DecisionArtifact:
        """Persist a trained decision model as a weights + manifest pair.

        Parameters
        ----------
        model : object
            The trained model — an ``nn.Module`` for torch backends, or an
            :class:`~eeg_learning.models.decision_xgboost.XGBoostDecisionModel`.
        model_id : str
            Identifier used as the stem of the output files.
        backend : str
            Decision backend the model was built with.
        decision_cfg : dict
            The resolved ``[decision]`` config, mined for the feature, split, and
            hyperparameter sections of the manifest.
        output_dir : str or pathlib.Path
            Directory to write into; created if missing.
        metrics : dict, optional
            Metrics scored by this model, stored for provenance.
        repetition : int, optional
            Which training repetition produced it.
        extra : dict, optional
            Additional top-level manifest entries.

        Returns
        -------
        DecisionArtifact
            The artifact describing the files just written.
        """

        out = Path(output_dir)
        out.mkdir(parents=True, exist_ok=True)

        if backend == "xgboost":
            model_path = out / f"{model_id}.json"
            model.save(model_path)
            hyperparameters = dict(getattr(model, "params", {}))
        else:
            model_path = out / f"{model_id}.pt"
            torch.save(model.state_dict(), model_path)
            hyperparameters = {
                key: decision_cfg[key]
                for key in ("learning_rate", "weight_decay", "batch_size", "n_epochs")
                if key in decision_cfg
            }

        manifest = {
            "format_version": MANIFEST_FORMAT_VERSION,
            "model_id": model_id,
            "created_at": datetime.now().isoformat(timespec="seconds"),
            "backend": backend,
            "weights_file": model_path.name,
            "features": feature_config(decision_cfg),
            "hyperparameters": hyperparameters,
            "split": {
                **{key: decision_cfg[key] for key in SPLIT_KEYS if key in decision_cfg},
                "seed": repetition,
            },
            "repetition": repetition,
            "metrics": metrics or {},
        }
        if extra:
            manifest.update(extra)

        manifest_path = out / f"{model_id}.manifest.json"
        manifest_path.write_text(json.dumps(manifest, indent=2))

        return cls(
            model_id=model_id,
            model_path=model_path,
            manifest_path=manifest_path,
            manifest=manifest,
        )

    @classmethod
    def load(cls, model_id: str, *, models_dir) -> DecisionArtifact:
        """Load an artifact by id from a directory of saved decision models.

        Raises
        ------
        FileNotFoundError
            If the manifest, or the weights file it references, is missing.
        """

        out = Path(models_dir)
        manifest_path = out / f"{model_id}.manifest.json"
        if not manifest_path.exists():
            raise FileNotFoundError(f"No decision manifest for model_id '{model_id}' in {out}")

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

    @classmethod
    def load_for_weights(cls, model_path) -> DecisionArtifact | None:
        """Load the artifact whose weights are at ``model_path``, if it has one.

        Returns ``None`` when no manifest sits beside the weights — the caller
        then falls back to loading a bare checkpoint.
        """

        model_path = Path(model_path)
        manifest_path = model_path.with_name(f"{model_path.stem}.manifest.json")
        if not manifest_path.exists():
            return None
        return cls.load(model_path.stem, models_dir=model_path.parent)
