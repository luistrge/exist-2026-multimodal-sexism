"""Synthetic EXIST-style fixture used by tests and CI."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from PIL import Image


@pytest.fixture()
def synthetic_release(tmp_path: Path) -> Path:
    release = tmp_path / "EXIST 2026 Dataset V0.2"
    memes_root = release / "EXIST 2026 Memes Dataset"
    training = memes_root / "training"
    images = training / "memes"
    gold_root = release / "evaluation" / "golds"
    images.mkdir(parents=True)
    gold_root.mkdir(parents=True)

    records: dict[str, dict[str, object]] = {}
    gold: list[dict[str, str]] = []
    for index in range(40):
        item_id = str(110000 + index)
        label = "YES" if index % 2 else "NO"
        language = "es" if (index // 2) % 2 else "en"
        marker = "positive marker" if label == "YES" else "negative marker"
        filename = f"{item_id}.png"
        records[item_id] = {
            "id_EXIST": item_id,
            "lang": language,
            "text": f"{marker} example {index}",
            "meme": filename,
            "path_memes": f"memes/{filename}",
        }
        color = (210, 40, 40) if label == "YES" else (35, 40, 210)
        Image.new("RGB", (32 + index % 3, 24), color=color).save(images / filename)
        gold.append({"test_case": "EXIST2025", "id": item_id, "value": label})

    (training / "EXIST2026_training.json").write_text(json.dumps(records), encoding="utf-8")
    (gold_root / "EXIST2025_training_task2_1_gold_hard.json").write_text(
        json.dumps(gold), encoding="utf-8"
    )
    return release
