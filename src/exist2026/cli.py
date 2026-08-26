"""Command-line entry point for the reproducible baseline."""

from __future__ import annotations

import argparse
import json
from collections.abc import Sequence
from pathlib import Path

from exist2026.config import BaselineConfig
from exist2026.data import load_task21_training
from exist2026.training import evaluate_saved_model, train_and_evaluate


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="exist2026-baseline",
        description="Train or evaluate the reproducible CPU-only Task 2.1 baseline.",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    train = subparsers.add_parser("train", help="Train once and evaluate on a fixed holdout.")
    train.add_argument("--data-root", type=Path, required=True)
    train.add_argument("--gold-root", type=Path)
    train.add_argument("--output-dir", type=Path, default=Path("outputs/baseline-task21"))
    train.add_argument("--seed", type=int, default=42)
    train.add_argument("--validation-size", type=float, default=0.20)
    train.add_argument("--threshold", type=float, default=0.50)
    train.add_argument("--max-word-features", type=int, default=20_000)
    train.add_argument("--max-char-features", type=int, default=30_000)
    train.add_argument("--text-only", action="store_true", help="Disable visual descriptors.")

    evaluate = subparsers.add_parser("evaluate", help="Re-evaluate a persisted holdout.")
    evaluate.add_argument("--data-root", type=Path, required=True)
    evaluate.add_argument("--gold-root", type=Path)
    evaluate.add_argument("--model-dir", type=Path, default=Path("outputs/baseline-task21"))
    return parser


def _print_metrics(metrics: dict[str, object]) -> None:
    print(json.dumps(metrics, ensure_ascii=False, indent=2))


def main(argv: Sequence[str] | None = None) -> int:
    """Execute the selected baseline command."""

    args = _parser().parse_args(argv)
    examples = load_task21_training(args.data_root, args.gold_root)
    if args.command == "train":
        config = BaselineConfig(
            seed=args.seed,
            validation_size=args.validation_size,
            decision_threshold=args.threshold,
            max_word_features=args.max_word_features,
            max_char_features=args.max_char_features,
            use_images=not args.text_only,
        )
        metrics = train_and_evaluate(examples, args.output_dir, config)
    else:
        metrics = evaluate_saved_model(examples, args.model_dir)
    _print_metrics(metrics)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
