# Running the DVC pipeline

This document describes how to run the EEG training pipeline and where the
results are logged. The pipeline is defined declaratively in
[`dvc.yaml`](../dvc.yaml) and parameterised entirely from
[`eeg_win_stack/config/params.toml`](../eeg_learning/config/params.toml).

## Overview

The pipeline is a four-stage DAG. DVC runs each stage in order, skipping any
stage whose inputs (code, deps, and params) are unchanged since the last run.

```
preprocess ──▶ train ──▶ evaluate ──▶ decision
```

| Stage        | Command                                   | Reads (deps + params)                     | Produces (outs/metrics)                       |
| ------------ | ----------------------------------------- | ----------------------------------------- | --------------------------------------------- |
| `preprocess` | `uv run python -m eeg_win_stack.pipeline.preprocess` | `data`, `preprocessing`, `windowing`, `run` | `target/saved_windows/` (cached)              |
| `train`      | `uv run python -m eeg_win_stack.pipeline.train`      | `target/saved_windows`; `split`, `training`, `model`, `run` | `target/saved_models/window_model/` (cached) |
| `evaluate`   | `uv run python -m eeg_win_stack.pipeline.evaluate`   | `target/saved_windows`, `target/saved_models/window_model`; `split`, `model`, `run`, `output.training_detail_path` | `metrics.json` + `target/training_detail/` + MLflow run |
| `decision`   | `uv run python -m eeg_win_stack.pipeline.decision`   | `target/training_detail`; `decision`, `output.decision_models_path`, `run` | `target/decision_metrics.json` + `target/saved_models/decision_model/` + MLflow nested runs |

Each stage entry point is a thin `main()` that calls `eeg_win_stack.config.load()`
to read `params.toml` and then runs the relevant subpackage
(`DatasetBuilder`, `Trainer`, `Evaluator`).

## One-time setup

DVC must be initialised once per clone before the pipeline can run:

```bash
dvc init        # creates .dvc/ and the DVC config; commit the result
```

DVC is provided by the `dev` dependency group (`pyproject.toml`). If `dvc` is
not on your `PATH`, use the project virtualenv (`.venv/bin/dvc`) or run inside
`tox`.

You also need the TUAB/TUEG datasets on disk at the paths configured under
`[data]` in `params.toml` (`tuab_path`, `tueg_path`).

## Where parameters live

All tunable parameters live in **`eeg_win_stack/config/params.toml`**, grouped
into sections (`run`, `data`, `preprocessing`, `windowing`, `split`,
`training`, `output`, `model`). The pipeline always loads this file via
`eeg_win_stack/config/loader.py`; `dvc.yaml` references the same path so DVC
tracks which parameters each stage depends on.

To change behaviour you can either edit `params.toml` directly (for a plain
`dvc repro`) or override values on the command line with `-S` (for tracked
experiments — see below). The `-S` form is preferred because it records the
exact parameter values alongside the resulting metrics.

## Making a run

### Baseline run

`dvc repro` runs the full pipeline using the current `params.toml`, rebuilding
only the stages whose inputs changed:

```bash
dvc repro                 # run/refresh the whole pipeline
dvc repro evaluate        # run only up to a named stage (and its deps)
dvc repro -f              # force a full rerun, ignoring the cache
```

The first run writes `dvc.lock`, which pins the resolved deps, params, and
output hashes. Commit `dvc.lock` so the run is reproducible.

### Experiment run (recommended)

`dvc exp run` runs the pipeline as a tracked *experiment*, capturing the params
that produced each set of metrics without committing on every attempt. Override
any parameter with `-S 'params.toml:section.key=value'`:

```bash
dvc exp run \
    -S 'eeg_learning/config/params.toml:training.learning_rate=0.0005' \
    -S 'eeg_learning/config/params.toml:training.n_epochs=50'
```

Each experiment is recorded against its parameters and metrics; promote a good
one into your working tree with `dvc exp apply <exp-name>`, or discard the rest
with `dvc exp remove`.

### Sweeps

Queue a grid of experiments and run them together. See
[`examples/sweep.sh`](../examples/sweep.sh) for a worked learning-rate ×
batch-size sweep. The pattern is:

```bash
PARAMS="eeg_win_stack/config/params.toml"

for lr in 0.001 0.0005 0.0001; do
    for batch_size in 1 8 32; do
        dvc exp run --queue \
            -S "${PARAMS}:training.learning_rate=${lr}" \
            -S "${PARAMS}:training.batch_size=${batch_size}"
    done
done

dvc queue run --jobs 1     # increase --jobs if you have multiple GPUs
```

### Choosing the decision-stage learner

The `decision` stage turns the per-window probabilities from `evaluate` into a
single normal/abnormal verdict per recording (or per session/patient, via
`decision.use_session_or_patients`). Two learners are available, selected by
`decision.backend`:

| `backend` | Model | Hyperparameters |
| --------- | ----- | --------------- |
| `mlp` (default, alias `torch`) | `HistogramModel` / `DecisionModel` — a linear or shallow-MLP classifier trained with Adam + `NLLLoss` | `[decision]`: `learning_rate`, `weight_decay`, `batch_size`, `n_epochs`, `hidden_layers`, `hidden_length` |
| `xgboost` | Gradient-boosted trees (`XGBClassifier`) | `[decision.xgboost]`: `max_depth`, `n_estimators`, `subsample`, `early_stopping_rounds`, … |

Both consume exactly the same features — the probability histogram, the hybrid
histogram + padded raw vector, or padded raw probabilities, per `use_his` /
`use_hybrid` / `length` — and are scored with the same metrics, so runs are
directly comparable:

```bash
dvc exp run -S 'eeg_learning/config/params.toml:decision.backend=xgboost'
dvc exp run -S 'eeg_learning/config/params.toml:decision.backend=xgboost' \
            -S 'eeg_learning/config/params.toml:decision.xgboost.max_depth=4'
dvc exp show    # compare test_acc / mean_acc across backends
```

Every repetition (`decision.n_repetitions`) is trained; the best one — highest
`test_acc`, ties broken by validation loss — is saved to
`target/saved_models/decision_model/` as `decision_<backend>_<timestamp>` plus a
`.manifest.json` recording the backend, feature recipe, hyperparameters, split,
and metrics. That manifest is what lets `decision_evaluation` reload the model
later without the config file.

## Logging and viewing results

The pipeline records results in two complementary places.

### 1. `metrics.json` (DVC metrics)

The `evaluate` stage writes `metrics.json` at the repo root with the
test-set scores: `accuracy`, `precision`, `recall`, and `mcc`. It is declared
as `cache: false` in `dvc.yaml`, so it is **tracked by git** (not the DVC
cache) and travels with your commits.

```bash
dvc metrics show                 # metrics for the current workspace
dvc exp show                     # table of all experiments: params + metrics
dvc exp show --csv | column -t -s,   # same, as a CSV table
dvc metrics diff                 # compare metrics against the last commit
```

### 2. MLflow

One MLflow run spans `train + evaluate + decision`. The `train` stage mints the
run and logs the full parameter set, the per-epoch curves as stepped metrics,
and the checkpoint as an artifact; `evaluate` and `decision` resume that same
run to add their metrics. Tracking URI defaults to `mlruns/`, relative to the
repo root.

Browse and compare runs visually with:

```bash
mlflow ui                        # serves the dashboard from ./mlruns
```

Local runs land in the experiment **`eeg_win_stack_local`** — the `_local`
suffix is added whenever `run.use_azure_artifacts` is false.

`mlruns/` is gitignored — it is a local experiment log, not a versioned
artifact.

See [`mlflow-tracking.md`](mlflow-tracking.md) for setup, the full list of what
each stage logs, artifact storage, and troubleshooting.

## What gets cached vs. committed

| Path                      | Tracking            | In git? |
| ------------------------- | ------------------- | ------- |
| `target/saved_windows/`   | DVC cache (`cache: true`) | no |
| `target/saved_models/window_model/` | DVC cache (`cache: true`) | no |
| `target/saved_models/decision_model/` | DVC cache (`cache: true`) | no |
| `target/training_detail/` | DVC out (`cache: false`)  | no — `target/` is gitignored |
| `target/decision_results.csv` | not a DVC output; local fallback only | no — `target/` is gitignored |
| `metrics.json`            | DVC metric (`cache: false`) | **yes** |
| `target/decision_metrics.json` | DVC metric (`cache: false`) | no — `target/` is gitignored |
| `dvc.lock`                | git                 | **yes** |
| `mlruns/`                 | local only          | no (gitignored) |

Large data and model artifacts stay in the DVC cache and out of git; the lock
file is committed so a run can be reproduced.

> **Known gap.** `.gitignore` ignores `target/` wholesale (line 104), so
> `target/decision_metrics.json` cannot be committed even though it is declared
> as a DVC `metrics:` output. `dvc metrics diff` and `dvc exp show` therefore
> have no history to compare the decision-stage scores against — only the
> root-level `metrics.json` from `evaluate` works as intended. Fixing it means
> either un-ignoring that one file (`!target/decision_metrics.json`) or moving
> it out of `target/`.

## Notes / gotchas

- **Model selection in `evaluate`.** The train stage saves a timestamped
  checkpoint (`<model>_<timestamp>_params.pt`) into `output.saved_models_path`
  (`target/saved_models/window_model/`) and records its path in the `run.json`
  token beside it. `evaluate` reads that token, so accumulated checkpoints no
  longer cause the wrong model to be evaluated. It falls back to the newest
  `*.pt` by mtime only when the token is missing or stale, and logs which path
  it took — see [`mlflow-tracking.md`](mlflow-tracking.md#the-run-token).
- **One stage, one directory.** DVC refuses overlapping stage outputs, which is
  why the two stages write to sibling directories (`saved_models/window_model`
  and `saved_models/decision_model`) rather than sharing `saved_models/`.
- **Editing `params.toml` vs. `-S`.** A bare `dvc repro` uses whatever is
  currently in `params.toml`. `dvc exp run -S` applies the override for that
  experiment only and records it — prefer it for anything you want to compare
  later.
- **DAG short-circuiting.** DVC skips stages whose inputs are unchanged. Use
  `dvc repro -f` (or `-f <stage>`) to force a rerun, e.g. after changing code
  that DVC doesn't track as a dependency.
