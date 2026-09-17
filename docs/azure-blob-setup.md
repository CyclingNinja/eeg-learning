# Azure Blob Storage — DVC Remote and MLflow Artifact Store

This guide sets up Azure Blob Storage as:

1. A **DVC remote** — so `saved_windows_data` and `saved_models` are pushed/pulled from the cloud instead of the Seagate drive.
2. An **MLflow artifact store** — so model checkpoints and artifacts logged during `evaluate` land in Azure rather than the local `mlruns/` directory.

---
                                                                            
## Quick Reference
                                                                            
Once everything has been configured and install basic operations are as follows:

```bash
# One-time setup on a new machine after cloning
export AZURE_STORAGE_CONNECTION_STRING="..."
dvc pull                        # fetch cached data from Azure
                                                                            
# After a pipeline run
dvc repro
dvc push                        # push new outputs to Azure
git add dvc.lock metrics.json && git commit -m "Pipeline run: <description>"
                                                                            
# View MLflow runs (metadata local, artifacts streamed from Azure)
mlflow ui---
```

## Part 1 — Azure Portal: Create the Storage Account

The Azure Portal URL is `portal.azure.com`. The UI is notoriously deep; follow each step exactly.

### 1.1 — Create a Resource Group (skip if you have one)

A resource group is just a logical container for billing and access control.

1. In the top search bar type **Resource groups** and click the result.
2. Click **+ Create** (top-left blue button).
3. Fill in:
   - **Subscription**: your subscription.
   - **Resource group**: `eeg-ml-rg` (or any name you like).
   - **Region**: pick the region closest to you (e.g. `UK South`, `West Europe`, `East US`). This region will apply to everything below.
4. Click **Review + create** → **Create**.

### 1.2 — Create a Storage Account

1. In the top search bar type **Storage accounts** and click the result.
2. Click **+ Create**.
3. **Basics** tab:
   - **Subscription**: same as above.
   - **Resource group**: `eeg-ml-rg`.
   - **Storage account name**: must be 3–24 lowercase letters/digits, globally unique — e.g. `eegpipelinedata` (add a suffix if taken).
   - **Region**: same region as the resource group.
   - **Performance**: `Standard`.
   - **Redundancy**: `Locally-redundant storage (LRS)` — cheapest, fine for ML data that is reproducible.
4. Click **Next: Advanced** — leave defaults.
5. Click **Next: Networking** — leave defaults (public access is fine; you can add a VNet later).
6. Click **Next: Data protection** — uncheck soft-delete options if you want to keep costs low and have DVC as your version history.
7. Click **Review + create** → **Create**.
8. When the deployment completes click **Go to resource**.

### 1.3 — Create Blob Containers

You need two containers: one for DVC, one for MLflow.

1. In the storage account left-hand menu, click **Containers** (under *Data storage*).
2. Click **+ Container**.
   - **Name**: `dvc-cache`
   - **Public access level**: `Private (no anonymous access)`
   - Click **Create**.
3. Click **+ Container** again.
   - **Name**: `mlflow-artifacts`
   - **Public access level**: `Private`
   - Click **Create**.

### 1.4 — Get the Connection String

1. In the storage account left-hand menu click **Access keys** (under *Security + networking*).
2. Click **Show keys** if they are hidden.
3. Copy the **Connection string** under **key1**. It looks like:

   ```
   DefaultEndpointsProtocol=https;AccountName=eegpipelinedata;AccountKey=<long-base64-key>;EndpointSuffix=core.windows.net
   ```

   Keep this somewhere safe (a password manager). **Do not commit it to git.**

---

## Part 2 — Install Dependencies

### 2.1 — Add `dvc[azure]` to the project

`dvc[azure]` pulls in `azure-storage-blob` which is also required by the MLflow Azure backend. Edit `pyproject.toml`:

```toml
[dependency-groups]
dev = [
    "pytest>=8.4.2",
    "ruff>=0.15.14",
    "dvc[azure]>=3.0,<4.0",   # <-- was "dvc>=3.0,<4.0"
    "tox>=4.22",
    "tox-uv>=1.13",
]
```

Also add `azure-storage-blob` to the main `dependencies` list so MLflow can reach it at runtime:

```toml
[project]
dependencies = [
  ...
  "azure-storage-blob>=12.19,<13.0",
]
```

Reinstall:

```bash
uv sync --all-groups
# or, if using pip directly:
pip install "dvc[azure]" "azure-storage-blob>=12.19,<13.0"
```

---

## Part 3 — Configure the DVC Remote

### 3.1 — Add the remote

```bash
# Register a remote called "azure" pointing at the dvc-cache container.
# The path segment after the container name is the prefix inside it.
dvc remote add -d azure azure://dvc-cache/eeg-pipeline

# Store the connection string — this writes to .dvc/config.local (git-ignored)
dvc remote modify --local azure connection_string \
  "DefaultEndpointsProtocol=https;AccountName=eegpipelinedata;AccountKey=<YOUR_KEY>;EndpointSuffix=core.windows.net"
```

`-d` makes `azure` the default remote so `dvc push` / `dvc pull` use it without extra flags.

`.dvc/config.local` is **automatically git-ignored** by DVC — the credential never touches git. What lands in `.dvc/config` (and should be committed) is just:

```ini
[core]
    remote = azure
['remote "azure"']
    url = azure://dvc-cache/eeg-pipeline
```

### 3.2 — Alternatively: use an environment variable

If you prefer not to run `dvc remote modify`, export the connection string in your shell profile (`~/.zshrc`) instead:

```bash
export AZURE_STORAGE_CONNECTION_STRING="DefaultEndpointsProtocol=https;AccountName=eegpipelinedata;AccountKey=<YOUR_KEY>;EndpointSuffix=core.windows.net"
```

DVC (and `azure-storage-blob`) both pick this up automatically.

### 3.3 — Push/pull data

```bash
# Push all cached outputs to Azure after a pipeline run
dvc push

# Pull cached outputs on another machine or after a fresh clone
dvc pull

# Push/pull a specific stage's output only
dvc push -r azure saved_windows_data
```

---

## Part 4 — Update `dvc.yaml` Paths (optional but recommended)

The current `dvc.yaml` hardcodes paths to the Seagate drive. To make the pipeline portable (runnable on any machine that has `dvc pull`), switch the outputs to local `data/` paths and let DVC manage the sync to Azure:

```yaml
stages:
  preprocess:
    cmd: python -m eeg_learning.pipeline.preprocess
    params:
      - eeg_learning/config/params.toml:
          - data
          - preprocessing
          - windowing
          - run
    outs:
      - data/saved_windows_data:
          cache: true

  train:
    cmd: python -m eeg_learning.pipeline.train
    deps:
      - data/saved_windows_data
    params:
      - eeg_learning/config/params.toml:
          - split
          - training
          - model
          - run
    outs:
      - data/saved_models:
          cache: true

  evaluate:
    cmd: python -m eeg_learning.pipeline.evaluate
    deps:
      - data/saved_windows_data
      - data/saved_models
    params:
      - eeg_learning/config/params.toml:
          - split
          - model
          - run
    metrics:
      - metrics.json:
          cache: false
```

Update `params.toml` to match:

```toml
[data]
save_windows_path      = "data/saved_windows"
save_recordings_path   = "data/saved_recordings"

[output]
saved_models_path = "data/saved_models/"
```

The local `data/` directory should be git-ignored (DVC handles its contents). If it is not already:

```bash
echo '/data/' >> .gitignore
```

---

## Part 5 — Configure MLflow to Use Azure Blob Storage

### 5.1 — Set the artifact root when starting a run

> **This is now config-driven — no code change needed.** Set
> `use_azure_artifacts = true` and `azure_artifact_root` under `[run]` in
> `eeg_learning/config/params.toml`, and `eeg_win_stack/tools/tracking.py`
> applies the `artifact_location` for you when it creates the experiment. See
> [`mlflow-tracking.md`](mlflow-tracking.md#artifact-storage). The snippet below
> is retained to show what that code does under the hood.

MLflow resolves the artifact store from the tracking URI or from an explicit
`artifact_location`. The simplest approach is to pass it when you create the
experiment:

```python
import os
import mlflow

AZURE_ARTIFACT_ROOT = "wasbs://mlflow-artifacts@eegpipelinedata.blob.core.windows.net/experiments"

mlflow.set_tracking_uri("mlruns")  # keep local tracking DB
mlflow.set_experiment(
    experiment_name="eeg_pipeline",
    artifact_location=AZURE_ARTIFACT_ROOT,  # artifacts go to Azure
)
```

`wasbs://` is the URI scheme MLflow uses for Azure Blob Storage.
`eegpipelinedata` is your storage account name; `mlflow-artifacts` is the
container name from Part 1.

> **Note**: `artifact_location` is only respected when the *experiment is first
> created*. If the experiment already exists in `mlruns/`, delete it first with
> `mlflow experiments delete -n eeg_pipeline` (or pick a new name).

### 5.2 — Credentials

MLflow delegates blob access to `azure-storage-blob`, which reads the same
`AZURE_STORAGE_CONNECTION_STRING` environment variable used by DVC. As long as
that variable is set (see Part 3.2), no additional configuration is needed.

### 5.3 — Verify

After a pipeline run:

1. Open `portal.azure.com` → your storage account → **Containers** →
   `mlflow-artifacts`.
2. You should see an `experiments/` prefix containing run subdirectories with
   artifacts inside.

You can still browse runs locally with `mlflow ui` — the UI reads metadata from
`mlruns/` but fetches artifact previews from Azure transparently.

---

## Part 6 — Committing the Configuration

Files to commit:

```bash
git add .dvc/config pyproject.toml uv.lock dvc.yaml eeg_learning/config/params.toml
git commit -m "Configure Azure Blob Storage remote for DVC and MLflow artifacts"
```

Files **not** to commit (should already be git-ignored):

| File | Why |
|---|---|
| `.dvc/config.local` | Contains the connection string |
| `data/` | Large binary data managed by DVC |
| `mlruns/` | Local experiment tracking DB |


















