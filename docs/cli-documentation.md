# Command-line interface

A user guide for the `eeg-learning` command — the quickest way to train a model
from windowed EEG data without writing any Python.

The CLI is a thin shell over the `api` layer: it turns your arguments into a job,
hands it to an execution backend, and prints the result. Everything it does is also
available from Python (see [`api-documentation.md`](api-documentation.md)) — the CLI
just makes the common path a one-liner, and is the same command the remote backends
will invoke.

## Installing

Install the package into your environment (editable is convenient during
development):

```bash
pip install -e .
```

This registers the `eeg-learning` console script. You can invoke the CLI either way:

```bash
eeg-learning --help              # console script
python -m eeg_learning --help    # module form (identical)
```

Both require the full runtime dependencies (torch, braindecode, mne) — training runs
real models. See the note at the bottom.

## Quick start

Train a model from a directory of pre-windowed data and save the result:

```bash
eeg-learning train \
  --windows-path data/saved_windows \
  --output-dir   data/saved_models
```

On success it prints the saved artifact as JSON and exits `0`:

```json
{
  "model_id": "deep4_2026-07-21_14-38-14",
  "model_path": "data/saved_models/deep4_2026-07-21_14-38-14.pt",
  "manifest_path": "data/saved_models/deep4_2026-07-21_14-38-14.json"
}
```

Because the output is JSON, you can pipe it straight into other tooling:

```bash
eeg-learning train --windows-path data/saved_windows --output-dir data/saved_models \
  | jq -r .model_id
```

## Commands

### `train`

Train a model from pre-windowed data on disk and save a weights + manifest artifact
pair (see "Model artifacts" in [`api-documentation.md`](api-documentation.md) for what
those two files are).

| Argument         | Required | Default            | Description                                                      |
| ---------------- | -------- | ------------------ | ---------------------------------------------------------------- |
| `--windows-path` | yes      | —                  | Directory of saved windowed data to train on.                    |
| `--output-dir`   | yes      | —                  | Directory the model artifact pair is written into.               |
| `--config`       | no       | packaged `params.toml` | Path to a `params.toml` overriding the defaults.             |
| `--backend`      | no       | `local`            | Where the job runs. `local` is implemented; `azureml`/`slurm` are planned. |
| `--model-id`     | no       | `<name>_<timestamp>` | Identifier for the saved artifact (becomes the file stem).     |

**Input expectations.** `--windows-path` must point at data already windowed and
saved to disk (the output of the preprocess step / DVC `preprocess` stage). The CLI
does **not** load or window raw recordings — it starts from saved windows.

## Configuration

Training reads all of its hyperparameters — model choice, learning rate, epochs,
split ratios, and so on — from a `params.toml` file. With no `--config` flag, the CLI
uses the packaged default at `eeg_learning/config/params.toml`, which has these
sections:

| Section          | Controls                                        |
| ---------------- | ----------------------------------------------- |
| `[run]`          | thread count, random seed                       |
| `[split]`        | train/valid/test ratios, split strategy         |
| `[training]`     | learning rate, epochs, batch size, early stopping |
| `[model]`        | model `name` and shared build kwargs            |
| `[model.deep4]` … | per-architecture hyperparameters                |

To train with different settings, copy that file, edit it, and pass it:

```bash
eeg-learning train \
  --config my_params.toml \
  --windows-path data/saved_windows \
  --output-dir   data/saved_models
```

> For parameter **sweeps**, use the DVC experiment pipeline (`examples/sweep.sh`)
> rather than scripting the CLI in a loop — see [`dvc-pipeline.md`](dvc-pipeline.md).
> The CLI is for single, explicit runs.

## Naming a run

By default the artifact id is `<model-name>_<timestamp>`, e.g.
`deep4_2026-07-21_14-38-14`. Pass `--model-id` to give it a stable, memorable name —
useful when you want to reload it later by that name:

```bash
eeg-learning train \
  --windows-path data/saved_windows \
  --output-dir   data/saved_models \
  --model-id     deep4_baseline
```

This writes `deep4_baseline.pt` and `deep4_baseline.json` into the output directory.

## Choosing where the job runs

`--backend local` (the default) runs training in the current process, synchronously —
the command blocks until training finishes, then prints the result. This is the right
choice for a dev machine or a single GPU box.

`azureml` and `slurm` are placeholders for dispatching the same job to remote compute.
They are **not implemented yet**; selecting one fails cleanly:

```bash
$ eeg-learning train --backend azureml --windows-path w --output-dir o
error: Backend 'azureml' is not implemented yet
```

## Exit codes

| Code | Meaning                                                                 |
| ---- | ---------------------------------------------------------------------- |
| `0`  | Success — the result JSON is on stdout.                                 |
| `1`  | A user-facing error (unknown backend, or a planned-but-unbuilt one). The message is on **stderr**, prefixed `error:`. |
| `2`  | Argument error from argparse (missing/unknown flag). Usage is on stderr. |

Errors from the training run itself (a bad config value, missing data, a model that
won't build) are **not** swallowed — they propagate as a Python traceback, so a
failed local run is loud and debuggable rather than reduced to an exit code.

## Reloading a trained model

The CLI trains and saves; reloading for evaluation or inference is a Python step
today (a `run_evaluation` / evaluate command is planned). The saved artifact is
self-describing:

```python
from eeg_learning.api import ModelArtifact
from eeg_learning.training.trainer import Trainer, TrainingConfig

artifact = ModelArtifact.load("deep4_baseline", models_dir="data/saved_models")
model = artifact.build_model()  # rebuilt from the manifest
classifier = Trainer(TrainingConfig()).load(model, artifact.model_path)
```

## Related documentation

- [`api-documentation.md`](api-documentation.md) — the Python interface the CLI wraps,
  including model artifacts and the manifest format.
- [`dvc-pipeline.md`](dvc-pipeline.md) — the experimentation/sweep pipeline.

> **Note:** the CLI imports the `api` layer at startup, so it needs an environment
> with the full runtime deps (torch, braindecode, mne) installed — even to print
> `--help`. Running `train` additionally needs the windowed data on disk.
