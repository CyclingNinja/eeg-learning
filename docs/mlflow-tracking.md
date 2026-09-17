# MLflow experiment tracking

This document covers how MLflow is wired into the EEG win-stack pipeline: how to
set it up, what each stage records, where training detail and model artifacts
are stored, and how to point artifact storage at Azure Blob Storage.

All of the integration lives in one module —
[`eeg_win_stack/tools/tracking.py`](../eeg_learning/tools/tracking.py). It is
the only place in the package that imports `mlflow`, so that is the file to read
when this document and the code disagree.

## What MLflow is for here (and what DVC is for)

The pipeline records results in two systems with a deliberate division of
labour:

| Question | Answered by |
| --- | --- |
| "Given these params and inputs, what artifacts came out?" | **DVC** — aggregate scalars (`metrics.json`, `target/decision_metrics.json`) and real cached outputs |
| "How did run A compare to run B, and what happened *during* each run?" | **MLflow** — per-epoch curves, per-repetition rows, params, tags, model artifacts |

The practical rule: anything that is one number per pipeline invocation stays a
DVC metric so `dvc metrics diff` and `dvc exp show` keep working. Anything that
is a *series* — a loss curve, a set of repetitions — goes to MLflow, because a
static DVC output cannot represent it usefully.

See [`dvc-pipeline.md`](dvc-pipeline.md) for the DVC half.

## The run-threading model

DVC launches every stage as its own process, so a single "experiment run" cannot
just be held open across stages. Instead:

```
train ──mints run──▶ writes run.json ──▶ evaluate resumes ──▶ decision resumes
```

- **`train` mints the run** and logs the full parameter set.
- **`evaluate` and `decision` resume it** via `mlflow.start_run(run_id=...)`.
- **`preprocess` does not participate.** It runs before any run exists, so
  `train` logs the preprocessing and windowing params on its behalf.

One experiment run therefore equals `train + evaluate + decision`.

### The run token

The handoff is a small JSON file called `run.json`, written by `train`:

```json
{
  "run_id": "8f2c1d4e5a6b7c8d9e0f1a2b3c4d5e6f",
  "experiment": "eeg_win_stack_local",
  "model_path": "target/saved_models/window_model/deep4_2026-08-18_14-22-05_params.pt",
  "created": "2026-08-18T14:22:05"
}
```

It lives **inside `output.saved_models_path`**, which is a *cached* DVC output of
the train stage. That is the load-bearing design choice: the run ID becomes a
property of the trained model. When DVC restores a cached train stage it
restores the token with it, so re-running `evaluate` alone rejoins the run that
actually produced the model being evaluated — rather than starting an orphan run
or, worse, attaching results to the wrong one.

The token also records **which `.pt` file train wrote**. Downstream stages call
`tracking.resolve_model_path(cfg)` instead of globbing the directory for the
newest checkpoint, so an accumulation of old checkpoints cannot cause the wrong
model to be evaluated. That handoff works whether or not MLflow is reachable.

If the token is missing or points at a deleted file, `resolve_model_path` falls
back to the newest `*.pt` by mtime and logs a warning saying so.

## Setup

### Install

Nothing to do — `mlflow>=2.0,<3.0` is a core dependency in
[`pyproject.toml`](../pyproject.toml). Any environment that can run the pipeline
can track it.

### Configure

Everything is driven from the `[run]` section of
[`eeg_win_stack/config/params.toml`](../eeg_learning/config/params.toml):

```toml
[run]
experiment_name = "eeg_win_stack"
mlflow_tracking_uri = "mlruns"
mlflow_required = true
use_azure_artifacts = false
azure_artifact_root = "wasbs://mlflow-artifacts@medshedeegwinstack.blob.core.windows.net/experiments"
```

| Key | Default | Meaning |
| --- | --- | --- |
| `experiment_name` | — | Base experiment name. Suffixed with `_local` unless `use_azure_artifacts` is true (see below). |
| `mlflow_tracking_uri` | `"mlruns"` | Where run metadata goes. A relative path is a local file store; can also be `http://host:5000` for a tracking server, or a database URI. |
| `mlflow_required` | `true` | Whether a tracking failure aborts the stage. See [Failure behaviour](#failure-behaviour). |
| `use_azure_artifacts` | `false` | Whether artifacts go to Azure Blob Storage instead of the local store. |
| `azure_artifact_root` | — | `wasbs://` URI used as the experiment's artifact location. Only read when `use_azure_artifacts` is true. |

### Run the pipeline, then look at it

```bash
dvc repro                        # train mints a run; evaluate and decision join it
mlflow ui                        # serves the dashboard from ./mlruns
```

`mlflow ui` defaults to `./mlruns`, which matches the default
`mlflow_tracking_uri`. If you changed the URI, pass it explicitly:

```bash
mlflow ui --backend-store-uri <your-uri>
```

> **Run stages from the repo root.** A relative `mlflow_tracking_uri` is
> resolved against the current working directory. DVC always invokes stages from
> the repo root, so `dvc repro` is safe; invoking a stage module by hand from a
> subdirectory would create a second, stray `mlruns/`.

`mlruns/` is gitignored. It is a local experiment log, not a versioned artifact.

### The `_local` experiment-name suffix

`experiment_name(cfg)` appends `_local` when `use_azure_artifacts` is false. So
with the config above you will find your runs under **`eeg_win_stack_local`** in
the UI, not `eeg_win_stack`.

This exists because `mlflow.start_run` raises if `set_experiment()` selected a
different experiment than the run being resumed belongs to. Separating the two
storage modes by name means a local run and an Azure-backed run can never end up
in the same experiment with inconsistent artifact locations.

Resuming stages pass the experiment name **recorded in the token**, not the one
derived from current config — so toggling `use_azure_artifacts` midway through a
pipeline will not break the resume.

## What gets logged

### `train` — params, curves, and the model

Logged once, on the parent run:

**Params**

| | |
| --- | --- |
| Model | `model`, `model_class` |
| Training | `learning_rate`, `weight_decay`, `batch_size`, `n_epochs`, `early_stopping`, `es_patience` |
| Split | `split_way`, `random_state` |
| Preprocessing (applied by the earlier `preprocess` stage) | `sampling_freq`, `bandpass_filter`, `standardization`, `window_len_s` |
| Resolved from the data | `n_channels`, `window_len_samples` |

`model_class` is logged alongside `model` on purpose: the model registry is a
plain dict, so a duplicate `@register` key silently shadows another entry. The
param records the class that was *actually* built, not just the name requested.

**Stepped metrics** — the per-epoch training curves, one MLflow step per epoch:

`train_loss`, `valid_loss`, `train_accuracy`, `valid_accuracy`

The `valid_*` columns are absent when training without a validation split. The
column-presence logic lives in `Trainer.history_rows`, which is shared by the
stepped metrics and the CSV writer, so the two cannot disagree about which
columns skorch actually recorded.

**Artifact** — the trained checkpoint, under the `model/` prefix:

```
model/deep4_2026-08-18_14-22-05_params.pt
```

### `evaluate` — test-set scores

Resumes the run and contributes only the test-set metrics, since params and the
model artifact were already logged by `train`:

- **Metrics**: `accuracy`, `precision`, `recall`, `mcc` — the same four values
  written to `metrics.json` for DVC.
- **Tag**: `evaluated_model`, the resolved checkpoint path.

### `decision` — aggregates on the parent, repetitions as children

- **Param**: `decision_backend` (`mlp` or `xgboost`).
- **Metrics on the parent**: `decision_test_acc`, `decision_ori_acc`,
  `decision_argmax_acc`, `decision_mean_acc` — averaged across repetitions.
  These stay on the parent so DVC still reads the same aggregates from
  `target/decision_metrics.json` as `metrics:`.
- **Nested child run per repetition**, named `decision-repetition-<n>`, each
  carrying that repetition's numeric results as metrics, a `repetition` param,
  and — on the repetition whose model was actually persisted — a
  `saved_model_id` tag.

Repetitions are genuinely separate experiment runs that share a parent, which is
exactly what MLflow nesting is for. Reading them as rows of a table is what the
old `decision_results.csv` was doing badly.

The number of repetitions comes from `[decision] n_repetitions` (default `5`).
Note that `[run] n_repetitions` is *not* consumed by anything.

## Training detail: saved to disk, not to MLflow

`evaluate` writes a **training detail artifact** — the first-stage per-window
predictions that the decision stage consumes as its input features. This is
deliberately **not** an MLflow artifact: it is a real pipeline output that a
later stage depends on, so DVC owns it.

`save_training_detail` writes a structured directory to
`output.training_detail_path` (default `target/training_detail/`):

| File | Contents |
| --- | --- |
| `windows.parquet` | One row per window: predictions, probabilities, split membership |
| `recordings.parquet` | One row per recording, with patient/session identifiers |
| `recording_summary.csv` | Human-scannable per-recording summary |
| `manifest.json` | `format: training_detail_v2`, file map, and counts (windows, recordings, patients, sessions) |
| `legacy_training_detail.csv` | Compatibility copy for consumers not yet migrated off the old single-file parser |

Passing a path ending in `.csv` instead selects the legacy single-file layout
outright.

Two smaller on-disk records exist alongside the MLflow data as durable
fallbacks. Neither is a DVC output — MLflow is the record for both — but "not a
DVC output" never means "gone from disk": the stages still write them every run,
which is what you fall back on when `mlflow_required = false` and tracking was
off.

- `output.training_results_path` (default `target/result.csv`) — the per-epoch
  table, the CSV twin of the stepped metrics above.
- `decision.csv_result_path` (default `target/decision_results.csv`) — the
  per-repetition rows, the CSV twin of the nested child runs.

Both live under `target/`, which is gitignored, so they are local to the machine
that produced them.

## Artifact storage

### Local (default)

With `use_azure_artifacts = false`, artifacts are written under the tracking
store — `mlruns/<experiment_id>/<run_id>/artifacts/`. Nothing else to configure.

### Azure Blob Storage

Set both keys:

```toml
[run]
use_azure_artifacts = true
azure_artifact_root = "wasbs://mlflow-artifacts@<account>.blob.core.windows.net/experiments"
```

`tracking.configure()` then passes `azure_artifact_root` as the experiment's
`artifact_location` at creation time. Run metadata still goes to
`mlflow_tracking_uri`; only the artifact bytes go to Azure.

Credentials: MLflow delegates blob access to `azure-storage-blob`, which reads
the same `AZURE_STORAGE_CONNECTION_STRING` environment variable that the DVC
remote uses. If DVC push/pull works, MLflow artifact upload will too. Full
walkthrough in [`azure-blob-setup.md`](azure-blob-setup.md).

> **`artifact_location` is only honoured when the experiment is first created.**
> Flipping `use_azure_artifacts` on an experiment that already exists will not
> retroactively move its artifact root. Because of the `_local` suffix you
> normally get a fresh experiment name when you flip the flag, which sidesteps
> this — but if you rename experiments by hand, delete the old one
> (`mlflow experiments delete -n <name>`) or pick a new name.

## Failure behaviour

`run.mlflow_required` defaults to **`true`**: MLflow is the system of record for
run history, so a silently missing run is a hole in that record. A tracking
failure raises `MLflowUnavailableError` with an actionable message:

```
Could not resume MLflow run. Set run.mlflow_required = false to continue without tracking.
```

Set it to `false` for offline work. Tracking calls then route to a
`NullTracker`, which implements the same surface as `Tracker` and does nothing
but log at DEBUG. Call sites need no `try`/`except` and no `if tracking_enabled`
branches — including `nested_run`, which yields the null tracker back to itself.

Things that raise when required, and no-op when not:

- The tracking store is unreachable, or auth fails.
- `evaluate`/`decision` run with no token to resume from (e.g. `train` never
  ran, or `output.saved_models_path` was wiped).
- The config has no `output.saved_models_path` at all.

Note that the `model_path` handoff still works with tracking off: `train` writes
the token with `"run_id": null` regardless, so `resolve_model_path` keeps
working.

## Using the tracker from code

`start_run` is a context manager yielding a `Tracker`:

```python
from eeg_learning.config import load
from eeg_learning.tools import tracking

cfg = load()

with tracking.start_run(cfg, resume=False) as tracker:  # mint
    tracker.log_params({"model": "deep4"})
    tracker.log_metrics({"train_loss": 0.42}, step=3)  # step= for curves
    tracker.log_artifact("target/plot.png", artifact_path="figures")
    tracker.set_tags({"note": "smoke run"})

    with tracker.nested_run(run_name="repetition-1") as child:
        child.log_metrics({"test_acc": 0.81})
```

| Method | Notes |
| --- | --- |
| `log_params(dict)` | Params are immutable in MLflow — logging the same key twice with different values errors. |
| `log_metrics(dict, step=None)` | Pass `step` to build a curve; omit it for a single value. |
| `log_artifact(path, artifact_path=None)` | `artifact_path` is the prefix inside the run's artifact dir. |
| `set_tags(dict)` | Mutable, unlike params. |
| `nested_run(run_name=None)` | Context manager yielding a child `Tracker`. |
| `.run_id` | The active run ID; `None` on `NullTracker`. |

Helpers worth knowing:

```python
tracking.read_token(cfg)          # the run.json dict, or None
tracking.resolve_model_path(cfg)  # the .pt train wrote (token first, mtime fallback)
tracking.experiment_name(cfg)     # the resolved name, _local suffix included
tracking.is_required(cfg)         # whether failures abort
```

The API and CLI layers do **not** currently open runs — tracking is wired into
the DVC pipeline stages only.

## Recipes

**Find the run that produced a given model**

```bash
cat target/saved_models/window_model/run.json
```

**Compare runs from the shell**

```bash
mlflow runs list --experiment-name eeg_win_stack_local
```

**Sweep params and get one run per point**

```bash
dvc exp run -S 'eeg_learning/config/params.toml:training.learning_rate=0.0005'
```

Each `dvc exp run` invocation runs `train` afresh, so each mints its own MLflow
run with its own curves. (`params.toml` lives under `eeg_win_stack/config/`, not
the repo root — DVC needs the full path in `-S`.)

**Work offline**

```bash
dvc exp run -S 'eeg_learning/config/params.toml:run.mlflow_required=false'
```

## Troubleshooting

| Symptom | Cause and fix |
| --- | --- |
| `MLflowUnavailableError: No MLflow run to resume from ...` | `evaluate`/`decision` ran without a token. Run `train` first, or `dvc repro` the whole pipeline. If the model came from elsewhere, set `mlflow_required = false`. |
| Runs missing from the UI | You are probably looking at `eeg_win_stack`; local runs land in **`eeg_win_stack_local`**. |
| A second `mlruns/` appeared in a subdirectory | A stage was invoked by hand from somewhere other than the repo root; the relative tracking URI resolved against the wrong cwd. |
| `mlflow.start_run` complains about a mismatched experiment | Something bypassed `tracking.configure()` and called `set_experiment` with a different name. Resuming stages must use the name in the token. |
| Metrics land on the wrong run | A stale `MLFLOW_RUN_ID` in the environment. `start_run` clears it on exit; check for one exported in your shell. |
| Artifacts not appearing in Azure | `AZURE_STORAGE_CONNECTION_STRING` unset, or the experiment predates `use_azure_artifacts = true` — `artifact_location` is fixed at creation. |
| `resolved model by mtime fallback` in the logs | The token was missing or pointed at a deleted checkpoint. Harmless if intended; otherwise the wrong `.pt` may be evaluated. |

## See also

- [`dvc-pipeline.md`](dvc-pipeline.md) — the pipeline stages and DVC metrics
- [`azure-blob-setup.md`](azure-blob-setup.md) — full Azure storage walkthrough
- [`eeg_win_stack/tools/tracking.py`](../eeg_learning/tools/tracking.py) — the
  implementation
