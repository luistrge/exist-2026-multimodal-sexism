"""Deterministic OCR and image features for the public baseline."""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

import numpy as np
from PIL import Image, UnidentifiedImageError

from exist2026.data import MemeExample

HISTOGRAM_BINS = 8
VISUAL_FEATURE_DIM = 44


def _normalized_histogram(values: np.ndarray, bins: int = HISTOGRAM_BINS) -> np.ndarray:
    counts, _ = np.histogram(values, bins=bins, range=(0.0, 1.0))
    total = counts.sum()
    return counts.astype(np.float32) / float(total if total else 1)


def visual_descriptor(path: Path | None) -> np.ndarray:
    """Create a small visual descriptor using only Pillow and NumPy.

    It includes RGB and grayscale histograms, channel moments, edge density,
    brightness statistics, aspect ratio, and a missing-image indicator. It is
    intentionally simpler than CLIP/DINOv2 so the baseline works offline.
    """

    missing = np.zeros(VISUAL_FEATURE_DIM, dtype=np.float32)
    missing[-1] = 1.0
    if path is None or not path.is_file():
        return missing
    try:
        with Image.open(path) as image:
            original_width, original_height = image.size
            rgb = image.convert("RGB").resize((128, 128))
            array = np.asarray(rgb, dtype=np.float32) / 255.0
    except (OSError, UnidentifiedImageError):
        return missing

    grayscale = np.dot(array[..., :3], np.array([0.299, 0.587, 0.114], dtype=np.float32))
    rgb_histograms = [_normalized_histogram(array[..., channel]) for channel in range(3)]
    gray_histogram = _normalized_histogram(grayscale)
    channel_means = array.mean(axis=(0, 1)).astype(np.float32)
    channel_stds = array.std(axis=(0, 1)).astype(np.float32)
    edge_x = float(np.abs(np.diff(grayscale, axis=1)).mean())
    edge_y = float(np.abs(np.diff(grayscale, axis=0)).mean())
    brightness = np.array([grayscale.mean(), grayscale.std()], dtype=np.float32)
    aspect_ratio = np.array(
        [float(original_width) / float(max(1, original_height))], dtype=np.float32
    )
    present = np.array([0.0], dtype=np.float32)
    descriptor = np.concatenate(
        [
            *rgb_histograms,
            gray_histogram,
            channel_means,
            channel_stds,
            np.array([edge_x, edge_y], dtype=np.float32),
            brightness,
            aspect_ratio,
            present,
        ]
    )
    if descriptor.shape != (VISUAL_FEATURE_DIM,):
        raise RuntimeError(f"Unexpected visual feature shape: {descriptor.shape}")
    return descriptor


def visual_matrix(examples: Sequence[MemeExample], use_images: bool = True) -> np.ndarray:
    """Return one deterministic visual descriptor per example."""

    if not use_images:
        return np.zeros((len(examples), VISUAL_FEATURE_DIM), dtype=np.float32)
    if not examples:
        return np.empty((0, VISUAL_FEATURE_DIM), dtype=np.float32)
    return np.vstack([visual_descriptor(example.image_path) for example in examples])


def model_text(example: MemeExample) -> str:
    """Prefix OCR with language so the linear model can learn bilingual cues."""

    normalized = " ".join(example.text.split())
    return f"[LANG={example.language}] {normalized}"
