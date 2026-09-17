#!/usr/bin/env bash
# =============================================================================
# sweep.sh — Example hyperparameter sweep using DVC experiments
#
# DVC replaces the old itertools.product loop. Each experiment runs the full
# pipeline (preprocess → train → evaluate) with one set of parameters, and
# results are tracked automatically against the params that produced them.
#
# USAGE
#   bash scripts/sweep.sh
#
# PREREQUISITES
#   dvc init          (once per repo)
#   dvc repro         (run baseline pipeline at least once)
#
# VIEWING RESULTS
#   dvc exp show      (table of all experiments with params + metrics)
#   dvc exp show --csv | column -t -s,   (CSV view)
#   mlflow ui         (visual comparison in the browser)
#
# PARAMS FILE
#   All parameters live in eeg_learning/config/params.toml.
#   Override syntax: -S 'path/to/params.toml:section.key=value'
# =============================================================================

set -euo pipefail

PARAMS="eeg_learning/config/params.toml"

# =============================================================================
# Example sweep: learning rate × batch size grid
#
# Queues 9 experiments (3 learning rates × 3 batch sizes).
# DVC runs them sequentially by default; use --jobs N for parallelism.
# =============================================================================

for lr in 0.001 0.0005 0.0001; do
    for batch_size in 1 8 32; do
        dvc exp run --queue \
            -S "${PARAMS}:training.learning_rate=${lr}" \
            -S "${PARAMS}:training.batch_size=${batch_size}"
    done
done

# Run all queued experiments (increase --jobs if you have multiple GPUs)
dvc queue run --jobs 1

# =============================================================================
# Example sweep: compare models at fixed training settings
# =============================================================================

# for model in deep4 tcn; do
#     dvc exp run --queue \
#         -S "${PARAMS}:model.name=${model}"
# done
# dvc queue run --jobs 1

# =============================================================================
# Example: single one-off run without queuing
# =============================================================================

# dvc exp run \
#     -S "${PARAMS}:training.learning_rate=0.0005" \
#     -S "${PARAMS}:training.n_epochs=50"
