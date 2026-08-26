from __future__ import annotations

from exist2026.evaluation import task21_metrics


def test_task21_metrics_uses_fixed_threshold() -> None:
    metrics = task21_metrics(["NO", "NO", "YES", "YES"], [0.1, 0.7, 0.8, 0.9])

    assert metrics["threshold"] == 0.5
    assert metrics["n_examples"] == 4
    assert metrics["confusion_matrix"]["values"] == [[1, 1], [0, 2]]
    assert metrics["evaluation_scope"] == "fixed-threshold held-out baseline validation"
