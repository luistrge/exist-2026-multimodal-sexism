"""Configuration for the lightweight Task 2.1 baseline."""

from __future__ import annotations

from dataclasses import asdict, dataclass


@dataclass(frozen=True, slots=True)
class BaselineConfig:
    """Hyperparameters with deterministic, CPU-friendly defaults."""

    seed: int = 42
    validation_size: float = 0.20
    decision_threshold: float = 0.50
    max_word_features: int = 20_000
    max_char_features: int = 30_000
    visual_weight: float = 0.35
    logistic_c: float = 1.0
    use_images: bool = True

    def __post_init__(self) -> None:
        if not 0.0 < self.validation_size < 1.0:
            raise ValueError("validation_size must be between 0 and 1")
        if not 0.0 < self.decision_threshold < 1.0:
            raise ValueError("decision_threshold must be between 0 and 1")
        if self.max_word_features < 1 or self.max_char_features < 1:
            raise ValueError("TF-IDF feature limits must be positive")
        if self.visual_weight < 0.0:
            raise ValueError("visual_weight cannot be negative")
        if self.logistic_c <= 0.0:
            raise ValueError("logistic_c must be positive")

    def to_dict(self) -> dict[str, int | float | bool]:
        """Return a JSON-serializable representation."""

        return asdict(self)
