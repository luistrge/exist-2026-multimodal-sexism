"""Leakage-aware evaluation helpers for the reproducible baseline."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)


def task21_metrics(
    labels: Sequence[str],
    yes_probabilities: Sequence[float],
    threshold: float = 0.5,
) -> dict[str, Any]:
    """Evaluate fixed-threshold predictions on an untouched partition."""

    target = np.asarray([1 if label == "YES" else 0 for label in labels], dtype=np.int8)
    scores = np.asarray(yes_probabilities, dtype=float)
    if target.shape != scores.shape:
        raise ValueError(f"Label/score shape mismatch: {target.shape} != {scores.shape}")
    prediction = (scores >= threshold).astype(np.int8)
    matrix = confusion_matrix(target, prediction, labels=[0, 1])
    return {
        "evaluation_scope": "fixed-threshold held-out baseline validation",
        "threshold": float(threshold),
        "n_examples": int(target.size),
        "macro_f1": float(f1_score(target, prediction, average="macro", zero_division=0)),
        "f1_yes": float(f1_score(target, prediction, pos_label=1, zero_division=0)),
        "precision_yes": float(precision_score(target, prediction, pos_label=1, zero_division=0)),
        "recall_yes": float(recall_score(target, prediction, pos_label=1, zero_division=0)),
        "accuracy": float(accuracy_score(target, prediction)),
        "confusion_matrix": {
            "labels": ["NO", "YES"],
            "values": matrix.astype(int).tolist(),
        },
    }
