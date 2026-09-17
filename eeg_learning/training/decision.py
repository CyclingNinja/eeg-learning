"""Training helpers for second-stage decision models."""

from __future__ import annotations

import copy

import numpy as np
import torch
from sklearn.model_selection import train_test_split

from eeg_learning.tools.decision_utils import DecisionTrainingResult


def split_decision_data(
    dataset,
    *,
    seed: int,
    batch_size: int,
    train_ratio: float = 0.9072,
    valid_ratio: float = 0.75,
    fix_testset: bool = True,
    test_batch_size: int = 16,
):
    """Split a decision dataset into train/valid/test DataLoaders.

    ``seed`` drives both split steps so each repetition is deterministic.
    When ``fix_testset`` is true, the train/test split keeps the test partition
    stable across repetitions by disabling shuffle on that first split.
    """

    idx_train, idx_test = train_test_split(
        torch.arange(len(dataset)),
        random_state=seed,
        train_size=train_ratio,
        shuffle=not fix_testset,
    )
    idx_train, idx_valid = train_test_split(
        idx_train,
        random_state=seed,
        train_size=valid_ratio,
        shuffle=True,
    )

    train_set = torch.utils.data.Subset(dataset, idx_train)
    valid_set = torch.utils.data.Subset(dataset, idx_valid)
    test_set = torch.utils.data.Subset(dataset, idx_test)

    train_loader = torch.utils.data.DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=0)
    valid_loader = torch.utils.data.DataLoader(valid_set, batch_size=batch_size, shuffle=True, num_workers=0)
    test_loader = torch.utils.data.DataLoader(test_set, batch_size=test_batch_size, shuffle=False, num_workers=0)
    return train_loader, valid_loader, test_loader


def train_torch_decision_model(
    model,
    train_loader,
    valid_loader,
    *,
    decision_cfg: dict,
    device,
) -> DecisionTrainingResult:
    """Fit a torch decision model, keeping the best-validation checkpoint.

    Adam with cosine-annealed learning rate and ``NLLLoss``, matching the
    ``LogSoftmax`` output of the decision models.
    """

    n_epochs = decision_cfg.get("n_epochs", 60)
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=decision_cfg.get("learning_rate", 0.01),
        weight_decay=decision_cfg.get("weight_decay", 0.01),
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, n_epochs, eta_min=0, last_epoch=-1)
    criterion = torch.nn.NLLLoss()
    model.train()

    min_loss_val = float("inf")
    best_model = model
    train_losses = []
    valid_losses = []
    iters = len(train_loader)

    for epoch in range(n_epochs):
        total_loss = 0
        for batch in train_loader:
            optimizer.zero_grad()
            X, Y, valid_len = [x.to(device) for x in batch]
            Y_hat = model(X, valid_len)
            loss = criterion(Y_hat, Y)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            scheduler.step(epoch + len(train_loader) / iters)

        train_losses.append(total_loss / len(train_loader))

        model.eval()
        total_loss = 0
        with torch.no_grad():
            for batch in valid_loader:
                X, Y, valid_len = [x.to(device) for x in batch]
                Y_hat = model(X, valid_len)
                loss = criterion(Y_hat, Y)
                total_loss += loss.item()

        avg_valid_loss = total_loss / len(valid_loader)
        valid_losses.append(avg_valid_loss)

        if avg_valid_loss < min_loss_val:
            min_loss_val = avg_valid_loss
            best_model = copy.deepcopy(model)

        model.train()

    return DecisionTrainingResult(
        model=best_model,
        best_valid_loss=min_loss_val,
        train_losses=train_losses,
        valid_losses=valid_losses,
    )


def _drain_loader(loader) -> tuple[np.ndarray, np.ndarray]:
    """Collect a decision DataLoader into ``(features, labels)`` arrays.

    The tree backend fits in one shot rather than per batch, so the same loaders
    that drive the torch loop are simply materialised here — which keeps both
    backends on the identical train/valid/test split.
    """

    features = []
    labels = []
    for X, Y, _ in loader:
        features.append(X.detach().cpu().numpy())
        labels.append(Y.detach().cpu().numpy())

    if not features:
        return np.empty((0, 0), dtype=np.float32), np.empty((0,), dtype=np.int64)
    return np.concatenate(features).astype(np.float32), np.concatenate(labels)


def train_xgboost_decision_model(model, train_loader, valid_loader) -> DecisionTrainingResult:
    """Fit an :class:`XGBoostDecisionModel` on the same split the torch path uses.

    Boosting rounds replace epochs, so the returned loss curves are per-round
    logloss and ``best_valid_loss`` is the lowest validation logloss reached.
    """

    train_features, train_labels = _drain_loader(train_loader)
    valid_features, valid_labels = _drain_loader(valid_loader)

    train_losses, valid_losses = model.fit(train_features, train_labels, valid_features, valid_labels)

    return DecisionTrainingResult(
        model=model,
        best_valid_loss=min(valid_losses) if valid_losses else float("inf"),
        train_losses=train_losses,
        valid_losses=valid_losses,
    )
