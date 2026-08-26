from __future__ import annotations

import json
from pathlib import Path

from exist2026.cli import main


def test_cli_train_and_evaluate_round_trip(
    synthetic_release: Path, tmp_path: Path, capsys: object
) -> None:
    model_dir = tmp_path / "model"
    train_exit = main(
        [
            "train",
            "--data-root",
            str(synthetic_release),
            "--output-dir",
            str(model_dir),
            "--max-word-features",
            "200",
            "--max-char-features",
            "300",
        ]
    )

    assert train_exit == 0
    assert (model_dir / "model.joblib").is_file()
    assert (model_dir / "metrics.json").is_file()
    assert (model_dir / "split.json").is_file()
    training_metrics = json.loads((model_dir / "metrics.json").read_text())
    assert training_metrics["validation_examples"] == 8
    assert training_metrics["macro_f1"] == 1.0

    evaluate_exit = main(
        [
            "evaluate",
            "--data-root",
            str(synthetic_release),
            "--model-dir",
            str(model_dir),
        ]
    )
    assert evaluate_exit == 0
