"""End-to-end train/evaluate workflow for the public baseline."""

from __future__ import annotations

import json
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import numpy as np
from sklearn.model_selection import train_test_split

from exist2026.config import BaselineConfig
from exist2026.data import MemeExample
from exist2026.evaluation import task21_metrics
from exist2026.model import MultimodalBaseline


def split_examples(
    examples: Sequence[MemeExample], config: BaselineConfig
) -> tuple[list[MemeExample], list[MemeExample]]:
    """Create the deterministic language/label-stratified holdout."""

    if any(example.label not in {"NO", "YES"} for example in examples):
        raise ValueError("Every split example must have a hard NO/YES label")
    indices = np.arange(len(examples))
    strata = [f"{example.language}_{example.label}" for example in examples]
    train_indices, validation_indices = train_test_split(
        indices,
        test_size=config.validation_size,
        random_state=config.seed,
        stratify=strata,
    )
    return (
        [examples[int(index)] for index in train_indices],
        [examples[int(index)] for index in validation_indices],
    )


def _write_json(path: Path, payload: Any) -> None:
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def train_and_evaluate(
    examples: Sequence[MemeExample],
    output_dir: str | Path,
    config: BaselineConfig | None = None,
) -> dict[str, Any]:
    """Fit once, evaluate once, and persist the exact split and model."""

    selected_config = config or BaselineConfig()
    destination = Path(output_dir).expanduser().resolve()
    destination.mkdir(parents=True, exist_ok=True)
    train_examples, validation_examples = split_examples(examples, selected_config)

    model = MultimodalBaseline(selected_config).fit(train_examples)
    probabilities = model.predict_yes_probability(validation_examples)
    labels = [str(example.label) for example in validation_examples]
    metrics = task21_metrics(labels, probabilities, selected_config.decision_threshold)
    metrics.update(
        {
            "model": "tfidf_word_char_plus_visual_descriptor_logistic_regression",
            "seed": selected_config.seed,
            "train_examples": len(train_examples),
            "validation_examples": len(validation_examples),
        }
    )

    model.save(destination / "model.joblib")
    _write_json(destination / "metrics.json", metrics)
    _write_json(destination / "model_metadata.json", model.metadata())
    _write_json(
        destination / "split.json",
        {
            "seed": selected_config.seed,
            "train_ids": [example.item_id for example in train_examples],
            "validation_ids": [example.item_id for example in validation_examples],
        },
    )
    return metrics


def evaluate_saved_model(examples: Sequence[MemeExample], model_dir: str | Path) -> dict[str, Any]:
    """Re-evaluate a saved model on the validation IDs recorded at training time."""

    source = Path(model_dir).expanduser().resolve()
    model = MultimodalBaseline.load(source / "model.joblib")
    split = json.loads((source / "split.json").read_text(encoding="utf-8"))
    by_id = {example.item_id: example for example in examples}
    validation_ids = [str(item_id) for item_id in split["validation_ids"]]
    missing = [item_id for item_id in validation_ids if item_id not in by_id]
    if missing:
        raise KeyError(f"Saved validation IDs are missing from the dataset: {missing[:5]}")
    validation_examples = [by_id[item_id] for item_id in validation_ids]
    probabilities = model.predict_yes_probability(validation_examples)
    labels = [str(example.label) for example in validation_examples]
    metrics = task21_metrics(labels, probabilities, model.config.decision_threshold)
    metrics.update(
        {
            "model": "tfidf_word_char_plus_visual_descriptor_logistic_regression",
            "seed": model.config.seed,
            "validation_examples": len(validation_examples),
        }
    )
    return metrics
