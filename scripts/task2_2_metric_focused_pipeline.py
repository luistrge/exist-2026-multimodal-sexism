"""Shared Task 2.2 helpers reconstructed from the audited experiment interface.

The master notebook imports this small compatibility module for text features,
probability alignment, threshold selection, cached embeddings, and interpretable
sensor aggregates. Heavy model caches are intentionally external to the repository.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import f1_score
from sklearn.pipeline import FeatureUnion

from exist2026_meme_utils import text_inputs


SEED = 42
NEG_LABEL = "JUDGEMENTAL"
POS_LABEL = "DIRECT"
LABELS_BIN = [NEG_LABEL, POS_LABEL]


def _word_vectorizer(max_features: int) -> TfidfVectorizer:
    return TfidfVectorizer(
        analyzer="word",
        ngram_range=(1, 2),
        min_df=2,
        max_df=0.995,
        sublinear_tf=True,
        strip_accents="unicode",
        max_features=max_features,
    )


def _char_vectorizer(max_features: int) -> TfidfVectorizer:
    return TfidfVectorizer(
        analyzer="char_wb",
        ngram_range=(3, 5),
        min_df=2,
        sublinear_tf=True,
        max_features=max_features,
    )


def _union_vectorizer(word_features: int, char_features: int) -> FeatureUnion:
    return FeatureUnion(
        [
            ("word", _word_vectorizer(word_features)),
            ("char", _char_vectorizer(char_features)),
        ]
    )


def normalize_binary(probabilities: np.ndarray) -> np.ndarray:
    """Return finite, row-normalized probabilities in label order.

    A one-dimensional input is interpreted as ``P(DIRECT)``.
    """
    probs = np.asarray(probabilities, dtype=float)
    if probs.ndim == 1:
        probs = np.column_stack([1.0 - probs, probs])
    if probs.ndim != 2 or probs.shape[1] != 2:
        raise ValueError(f"Expected an (n, 2) probability matrix, got {probs.shape}")
    probs = np.nan_to_num(probs, nan=0.5, posinf=1.0, neginf=0.0)
    probs = np.clip(probs, 1e-9, None)
    return probs / probs.sum(axis=1, keepdims=True)


def align_binary_proba(classes: Sequence[str], probabilities: np.ndarray) -> np.ndarray:
    """Align a classifier's probability columns to JUDGEMENTAL, DIRECT."""
    classes = [str(value) for value in classes]
    probs = np.asarray(probabilities, dtype=float)
    if probs.ndim != 2:
        raise ValueError("Classifier probabilities must be a two-dimensional matrix")
    aligned = np.zeros((len(probs), 2), dtype=float)
    for target_index, label in enumerate(LABELS_BIN):
        if label in classes:
            aligned[:, target_index] = probs[:, classes.index(label)]
    return normalize_binary(aligned)


def soft_gold_matrix(frame: pd.DataFrame) -> np.ndarray:
    """Convert Task 2.2 soft-label dictionaries to the binary label order."""
    rows = []
    for value in frame["soft_gold"]:
        value = value if isinstance(value, dict) else {}
        rows.append([float(value.get(label, 0.0)) for label in LABELS_BIN])
    return normalize_binary(np.asarray(rows, dtype=float))


def fast_metrics(y_true: Iterable[str], probabilities: np.ndarray, threshold: float) -> dict[str, float]:
    probs = normalize_binary(probabilities)
    pred = [POS_LABEL if score >= threshold else NEG_LABEL for score in probs[:, 1]]
    return {
        "macro_f1": float(
            f1_score(list(y_true), pred, labels=LABELS_BIN, average="macro", zero_division=0)
        )
    }


def optimize_threshold(
    y_true: Iterable[str],
    _soft_gold: np.ndarray,
    probabilities: np.ndarray,
) -> tuple[float, dict[str, float]]:
    """Select the macro-F1 threshold on the calibration partition."""
    y_true = list(y_true)
    probs = normalize_binary(probabilities)
    best_threshold = 0.5
    best_score = -1.0
    for threshold in np.linspace(0.05, 0.95, 361):
        score = fast_metrics(y_true, probs, float(threshold))["macro_f1"]
        if score > best_score + 1e-12:
            best_threshold = float(threshold)
            best_score = score
    return best_threshold, {"macro_f1": best_score}


def _align_matrix(path: Path | str, requested_ids: Sequence[str]) -> np.ndarray | None:
    """Load an NPZ embedding cache and align its rows to ``requested_ids``."""
    path = Path(path)
    if not path.exists():
        return None
    with np.load(path, allow_pickle=True) as cache:
        id_key = next((key for key in ("ids", "id", "item_ids", "sample_ids") if key in cache), None)
        matrix_key = next(
            (key for key in ("embeddings", "features", "matrix", "X", "vectors") if key in cache),
            None,
        )
        if id_key is None or matrix_key is None:
            return None
        ids = [str(value) for value in cache[id_key].tolist()]
        matrix = np.asarray(cache[matrix_key], dtype=float)
    lookup = {item_id: index for index, item_id in enumerate(ids)}
    if any(str(item_id) not in lookup for item_id in requested_ids):
        return None
    return np.stack([matrix[lookup[str(item_id)]] for item_id in requested_ids])


def _row_mean(frame: pd.DataFrame, columns: list[str]) -> pd.Series:
    if not columns:
        return pd.Series(np.nan, index=frame.index, dtype=float)
    numeric = frame[columns].apply(pd.to_numeric, errors="coerce")
    return numeric.mean(axis=1)


def add_paper_guided_sensor_axes(frame: pd.DataFrame) -> pd.DataFrame:
    """Add interpretable ET, HR, and EEG aggregates without inventing channels."""
    out = frame.copy()
    sensor_cols = [
        column
        for column in out.columns
        if column.startswith("sens_") and pd.api.types.is_numeric_dtype(out[column])
    ]
    groups = {
        "paper_et_fixation": ("et_", "fixation"),
        "paper_et_saccade": ("et_", "saccade"),
        "paper_et_blink": ("et_", "blink"),
        "paper_et_pupil": ("et_", "pupil"),
        "paper_et_reaction": ("et_", "reaction"),
        "paper_hr_all": ("sens_hr_",),
        "paper_eeg_delta": ("sens_eeg_", "delta"),
        "paper_eeg_theta": ("sens_eeg_", "theta"),
        "paper_eeg_alpha": ("sens_eeg_", "alpha"),
        "paper_eeg_beta": ("sens_eeg_", "beta"),
        "paper_eeg_gamma": ("sens_eeg_", "gamma"),
        "paper_eeg_all": ("sens_eeg_",),
    }
    for output_name, tokens in groups.items():
        selected = [
            column
            for column in sensor_cols
            if tokens[0] in column.lower()
            and (len(tokens) == 1 or any(token in column.lower() for token in tokens[1:]))
        ]
        if selected:
            out[output_name] = _row_mean(out, selected)
    return out


def _vlm_enriched_text_inputs(frame: pd.DataFrame) -> pd.Series:
    """Append cached VLM descriptions when the cache covers a sample ID."""
    base = frame.get("model_text", text_inputs(frame)).fillna("").astype(str)
    project_root = Path(
        os.environ.get("EXIST2026_PROJECT_ROOT", Path.cwd())
    ).expanduser().resolve()
    cache_path = project_root / "outputs" / "cache" / "task2_2_vlm_reasoning.csv"
    if not cache_path.exists():
        return base
    cache = pd.read_csv(cache_path, dtype={"id": str})
    text_column = next(
        (
            column
            for column in ("reasoning", "vlm_reasoning", "description", "vlm_text", "response")
            if column in cache.columns
        ),
        None,
    )
    if text_column is None or "id" not in cache:
        return base
    reasoning = cache.set_index("id")[text_column].fillna("").astype(str).to_dict()
    ids = frame["id"].astype(str)
    extra = ids.map(reasoning).fillna("").str.strip()
    return pd.Series(
        [f"{original} [VLM] {vlm}" if vlm else original for original, vlm in zip(base, extra)],
        index=frame.index,
    )
