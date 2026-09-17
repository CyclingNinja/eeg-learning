"""Configuration for second-stage decision model training."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import torch


@dataclass
class DecisionConfig:
    """Hyperparameters for decision-stage model training.

    Attributes
    ----------
    detail_path : str
        Path to the training detail artifact (structured directory preferred).
    csv_path : str or None
        Legacy fallback key for older configs still referring to CSV-style paths.
    csv_result_path : str
        Path to save decision_result.csv.
    learning_rate : float
        Optimizer learning rate.
    weight_decay : float
        Optimizer weight decay (L2 regularization strength).
    batch_size : int
        Number of samples per batch.
    n_epochs : int
        Number of training epochs.
    n_repetitions : int
        Number of repeated training runs (for averaging results).
    histogram_length : int
        Number of histogram bins for aggregated probabilities.
    use_histogram : bool
        Whether to use histogram representation of probabilities.
    use_hybrid : bool
        Whether to concatenate histogram with raw probabilities.
    use_adaptive_pool : bool
        Whether to use adaptive pooling (for raw data model).
    use_session_or_patients : str or None
        Aggregation strategy: None (per-recording), "patients", or "sessions".
    hidden_layers : int
        Number of hidden layers in the MLP.
    hidden_length : int
        Size of each hidden layer.
    fix_testset : bool
        Whether to use fixed test set splits across repetitions.
    train_ratio : float
        Fraction of data for training (rest split into validation and test).
    valid_ratio : float
        Fraction of the non-test subset used for training.
    start_row : int
        Starting row index for the labels block in a legacy training_detail.csv.
    n_rows : int
        Number of label rows in a legacy training_detail.csv.
    row_gap : int
        Number of rows between the probability block and the next block.
    block : int
        Zero-based block index within a legacy training_detail.csv.
    device : str or None
        Torch device string (e.g. ``"cpu"`` or ``"cuda"``). When ``None``,
        the device is autodetected.
    """

    detail_path: str | Path = "./target/training_detail"
    csv_path: str | Path | None = None
    csv_result_path: str | Path = "./decision_result.csv"
    learning_rate: float = 0.01
    weight_decay: float = 0.01
    batch_size: int = 64
    n_epochs: int = 60
    n_repetitions: int = 5
    histogram_length: int = 10
    use_histogram: bool = True
    use_hybrid: bool = False
    use_adaptive_pool: bool = False
    use_session_or_patients: str | None = None
    hidden_layers: int = 0
    hidden_length: int = 5
    fix_testset: bool = True
    train_ratio: float = 0.9072
    valid_ratio: float = 0.75
    start_row: int = 1
    n_rows: int = 4
    row_gap: int = 4
    block: int = 0
    device: str | None = None

    def resolve_device(self) -> str:
        """Return the device to train on, autodetecting CUDA when unset.

        Returns
        -------
        str
            :attr:`device` if explicitly set, otherwise ``"cuda"`` when a
            CUDA device is available and ``"cpu"`` otherwise.
        """
        if self.device is not None:
            return self.device
        return "cuda" if torch.cuda.is_available() else "cpu"

    def __post_init__(self):
        """Validate and normalize configuration."""
        if self.csv_path is not None and self.detail_path == "./target/training_detail":
            # Backward compatibility for old config objects.
            self.detail_path = self.csv_path

        if self.use_session_or_patients is not None:
            if self.use_session_or_patients not in ("patients", "sessions"):
                raise ValueError(
                    f"use_session_or_patients must be None, 'patients', or 'sessions'; "
                    f"got {self.use_session_or_patients!r}"
                )

        if self.histogram_length <= 0:
            raise ValueError(f"histogram_length must be > 0; got {self.histogram_length}")

        if self.batch_size <= 0:
            raise ValueError(f"batch_size must be > 0; got {self.batch_size}")

        if self.n_epochs <= 0:
            raise ValueError(f"n_epochs must be > 0; got {self.n_epochs}")

        if self.n_repetitions <= 0:
            raise ValueError(f"n_repetitions must be > 0; got {self.n_repetitions}")

        if not (0 < self.train_ratio < 1):
            raise ValueError(f"train_ratio must be in (0, 1); got {self.train_ratio}")

        if not (0 < self.valid_ratio < 1):
            raise ValueError(f"valid_ratio must be in (0, 1); got {self.valid_ratio}")

        if self.start_row < 0:
            raise ValueError(f"start_row must be >= 0; got {self.start_row}")

        if self.n_rows <= 0:
            raise ValueError(f"n_rows must be > 0; got {self.n_rows}")

        if self.row_gap < 0:
            raise ValueError(f"row_gap must be >= 0; got {self.row_gap}")

        if self.block < 0:
            raise ValueError(f"block must be >= 0; got {self.block}")
