from __future__ import annotations

from pathlib import Path

import numpy as np

from exist2026.features import VISUAL_FEATURE_DIM, visual_descriptor


def test_missing_image_has_explicit_indicator(tmp_path: Path) -> None:
    descriptor = visual_descriptor(tmp_path / "missing.png")

    assert descriptor.shape == (VISUAL_FEATURE_DIM,)
    assert descriptor[-1] == 1.0
    assert np.count_nonzero(descriptor[:-1]) == 0
