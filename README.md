# EEGScopeAndArbitration

Window-level EEG abnormality classification: loading and windowing raw recordings,
building braindecode-compatible models, training, and evaluation. The functionality
lives in the **`eeg_learning`** package.

This README documents the **`api` layer** — the importable, production-facing
interface for triggering training runs (locally or, later, on cloud compute),
evaluating saved models, and (eventually) inference. It is distinct from the DVC
`eeg_learning/pipeline/` stages, which exist for experimentation and parameter
sweeps only.

## Layout

```
eeg_learning/
  api/                 production interface (this document)
    jobs.py            run_training(...) -> TrainResult         — the real work
    artifacts.py       ModelArtifact: checkpoint + JSON manifest
    backends/          where a job runs
      base.py          Backend ABC, Job, JobHandle, JobKind, JobStatus
      local.py         LocalBackend (in-process, synchronous)
  config/              params.toml + load()
  io/                  raw loading, labeling, windowing, DatasetBuilder
  models/              ModelFactory + registered architectures
  tools/               splitting, filters, metrics, paths
  training/            Trainer, TrainingConfig
  evaluation/          Evaluator, EvaluationResult
  pipeline/            DVC stages (experimentation only — not the API)
```

## Design

Three layers, every entrypoint funnelling to the same core:

1. **`jobs.py`** holds the orchestration. Functions take a resolved config dict and
   explicit I/O paths and *return a result object* (unlike the `pipeline/` stages,
   which hardcode paths and return nothing). This is the library form of
   `pipeline/train.py`.
2. **`backends/`** decide *where* a job runs. A `Backend` exposes
   `submit(job) -> JobHandle`, `status(handle)`, and `result(handle)`. `LocalBackend`
   runs in-process; planned Azure ML and Slurm backends will dispatch the **same
   command** to remote compute. Callers don't change when the backend does.
3. A **CLI** and a **FastAPI service** (both planned) are thin shells over 1 and 2.

### Model artifacts and the manifest

A skorch/braindecode checkpoint (`.pt`) stores **weights only** — reloading it
requires first reconstructing the exact architecture. `ModelArtifact` therefore
saves a checkpoint together with a small JSON **manifest** describing how to rebuild
it. The two files share a `model_id` stem: `<model_id>.pt` + `<model_id>.json`.

```json
{
  "format_version": 1,
  "model_id": "deep4_2026-06-10_14-38-14",
  "created_at": "2026-06-10T14:38:14",
  "weights_file": "deep4_2026-06-10_14-38-14.pt",
  "model": {
    "name": "deep4",
    "build_kwargs": {
      "n_channels": 19, "n_classes": 2, "input_window_samples": 6000,
      "drop_prob": 0.1, "final_conv_length": "auto", "n_filters_time": 25
    }
  },
  "training": { "learning_rate": 0.001, "n_epochs": 30, "batch_size": 1 }
}
```

The manifest makes a model **self-describing and portable**: it can be reloaded
without the original training config or the windowed dataset. It is human-readable,
safe to parse without unpickling, and tooling-agnostic — keeping it aligned with
what MLflow logs, so a future MLflow/Azure Blob backend can adopt it directly.

## Usage

### Train (direct)

```python
from eeg_learning.config import load
from eeg_learning.api import run_training

config = load()  # reads eeg_learning/config/params.toml
result = run_training(
    config,
    windows_path="data/saved_windows",  # produced by the preprocess step
    output_dir="data/saved_models",
)
print(result.model_id, result.model_path, result.manifest_path)
```

### Train (via a backend)

The backend interface is what the CLI and service will use, and what cloud
execution will plug into:

```python
from eeg_learning.config import load
from eeg_learning.api.backends import get_backend, Job, JobKind

config = load()
backend = get_backend("local")  # "azureml" / "slurm" planned

handle = backend.submit(
    Job(
        kind=JobKind.TRAIN,
        config=config,
        windows_path="data/saved_windows",
        output_dir="data/saved_models",
        options={"model_id": "deep4_run1"},  # optional; defaults to "<name>_<timestamp>"
    )
)

print(backend.status(handle))  # JobStatus.COMPLETED (LocalBackend is synchronous)
print(backend.result(handle))  # {"model_id", "model_path", "manifest_path"}
```

### Reload a saved model

```python
from eeg_learning.api import ModelArtifact
from eeg_learning.training.trainer import Trainer, TrainingConfig

artifact = ModelArtifact.load("deep4_run1", models_dir="data/saved_models")
model = artifact.build_model()  # rebuilds the architecture from the manifest
classifier = Trainer(TrainingConfig()).load(model, artifact.model_path)
```

## API reference

### `eeg_learning.api`

- **`run_training(config, *, windows_path, output_dir, model_id=None) -> TrainResult`**
  Load windowed data, split, build the model from `config["model"]`, fit, and save a
  `ModelArtifact`. Returns `TrainResult(model_id, model_path, manifest_path, artifact)`.
- **`ModelArtifact`**
  - `save(eeg_classifier, *, model_id, model_name, build_kwargs, output_dir, training_config=None, extra=None) -> ModelArtifact`
  - `load(model_id, *, models_dir) -> ModelArtifact`
  - `build_model()` — reconstruct the (untrained) model from the manifest.
  - properties: `model_name`, `build_kwargs`, `model_path`, `manifest_path`, `manifest`.

### `eeg_learning.api.backends`

- **`get_backend(name="local", **kwargs) -> Backend`** — `"local"` is implemented;
  `"azureml"`/`"slurm"` raise `NotImplementedError`; unknown names raise `ValueError`.
- **`Backend`** (ABC) — `submit(job) -> JobHandle`, `status(handle) -> JobStatus`,
  `result(handle) -> dict`.
- **`Job(kind, config, windows_path, output_dir, options={})`** — `kind` is a `JobKind`.
- **`JobHandle(id, backend, native=None)`**, **`JobKind`** (`TRAIN`, `EVALUATE`),
  **`JobStatus`** (`PENDING`, `RUNNING`, `COMPLETED`, `FAILED`).
- **`LocalBackend`** — runs jobs in-process, synchronously; failures propagate as
  exceptions.

## Status

| Capability                          | State                                            |
| ----------------------------------- | ------------------------------------------------ |
| `run_training` + `ModelArtifact`    | ✅ implemented                                    |
| `LocalBackend` (TRAIN)              | ✅ implemented                                    |
| Evaluate saved model (`run_evaluation`, backend `EVALUATE`) | ⬜ planned        |
| CLI (`python -m eeg_win_stack ...`) | ⬜ planned (the command remote backends invoke)   |
| FastAPI service                     | ⬜ planned (synchronous first; async with cloud)  |
| Azure ML / Slurm backends           | ⬜ planned (`get_backend` stubs them)             |
| Inference (`Predictor`)             | ⬜ interface stub only — no architecture yet      |

> **Note:** the `api` layer requires the full runtime deps (torch, braindecode,
> mne). The unit test suites mock these; running real jobs needs an environment
> with them installed.
