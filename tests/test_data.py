from __future__ import annotations

from pathlib import Path

from exist2026.data import load_task21_training, resolve_memes_root


def test_load_task21_training_discovers_legacy_gold(synthetic_release: Path) -> None:
    examples = load_task21_training(synthetic_release)

    assert len(examples) == 40
    assert {example.label for example in examples} == {"NO", "YES"}
    assert {example.language for example in examples} == {"en", "es"}
    assert all(example.image_path and example.image_path.is_file() for example in examples)


def test_resolve_memes_root_accepts_release_or_dataset_directory(
    synthetic_release: Path,
) -> None:
    expected = synthetic_release / "EXIST 2026 Memes Dataset"

    assert resolve_memes_root(synthetic_release) == expected
    assert resolve_memes_root(expected) == expected
