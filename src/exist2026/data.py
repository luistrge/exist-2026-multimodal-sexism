"""Load the official EXIST 2026 meme release without redistributing it."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True, slots=True)
class MemeExample:
    """One meme and the fields required by the reproducible baseline."""

    item_id: str
    language: str
    text: str
    image_path: Path | None
    label: str | None = None


def resolve_memes_root(path: str | Path) -> Path:
    """Resolve either the meme dataset directory or its release parent."""

    candidate = Path(path).expanduser().resolve()
    roots = [candidate, candidate / "EXIST 2026 Memes Dataset"]
    for root in roots:
        if (root / "training" / "EXIST2026_training.json").is_file():
            return root
    raise FileNotFoundError("Could not find training/EXIST2026_training.json below " f"{candidate}")


def _load_records(path: Path) -> dict[str, dict[str, Any]]:
    raw = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(raw, dict):
        records = raw
    elif isinstance(raw, list):
        records = {str(row["id_EXIST"]): row for row in raw}
    else:
        raise ValueError(f"Unsupported JSON structure in {path}")
    return {str(key): dict(value) for key, value in records.items()}


def _resolve_gold_path(memes_root: Path, gold_root: str | Path | None) -> Path:
    if gold_root is not None:
        root = Path(gold_root).expanduser().resolve()
    else:
        root = memes_root.parent / "evaluation" / "golds"
    matches = sorted(root.glob("*training_task2_1_gold_hard.json"))
    if len(matches) != 1:
        raise FileNotFoundError(
            "Expected exactly one *training_task2_1_gold_hard.json file in "
            f"{root}; found {len(matches)}"
        )
    return matches[0]


def _image_path(partition_dir: Path, row: dict[str, Any]) -> Path | None:
    relative = row.get("path_memes") or row.get("meme")
    if not relative:
        return None
    candidate = partition_dir / str(relative)
    if candidate.is_file():
        return candidate
    fallback = partition_dir / "memes" / Path(str(relative)).name
    return fallback if fallback.is_file() else candidate


def load_task21_training(
    data_root: str | Path,
    gold_root: str | Path | None = None,
) -> list[MemeExample]:
    """Load the hard-labelled Task 2.1 modelling subset.

    The organizer-provided gold filename still contains ``EXIST2025``. The
    loader discovers it by task suffix so downstream code does not depend on
    that legacy prefix.
    """

    root = resolve_memes_root(data_root)
    partition_dir = root / "training"
    records = _load_records(partition_dir / "EXIST2026_training.json")
    gold_path = _resolve_gold_path(root, gold_root)
    gold_rows = json.loads(gold_path.read_text(encoding="utf-8"))
    if not isinstance(gold_rows, list):
        raise ValueError(f"Expected a list of hard-gold records in {gold_path}")

    examples: list[MemeExample] = []
    for gold in gold_rows:
        item_id = str(gold["id"])
        label = str(gold["value"])
        if label not in {"NO", "YES"}:
            continue
        if item_id not in records:
            raise KeyError(f"Gold ID {item_id} is missing from {partition_dir}")
        row = records[item_id]
        examples.append(
            MemeExample(
                item_id=item_id,
                language=str(row.get("lang", "unknown")),
                text=str(row.get("text", "") or ""),
                image_path=_image_path(partition_dir, row),
                label=label,
            )
        )
    if not examples:
        raise ValueError("The Task 2.1 training subset is empty")
    return examples


def load_test_memes(data_root: str | Path) -> list[MemeExample]:
    """Load the unlabelled official meme test partition."""

    root = resolve_memes_root(data_root)
    partition_dir = root / "test"
    records = _load_records(partition_dir / "EXIST2026_test_clean.json")
    return [
        MemeExample(
            item_id=str(item_id),
            language=str(row.get("lang", "unknown")),
            text=str(row.get("text", "") or ""),
            image_path=_image_path(partition_dir, row),
        )
        for item_id, row in records.items()
    ]
