"""CPU-only multimodal baseline for Task 2.1."""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path
from typing import Any

import joblib
import numpy as np
from scipy import sparse
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

from exist2026.config import BaselineConfig
from exist2026.data import MemeExample
from exist2026.features import model_text, visual_matrix


class MultimodalBaseline:
    """Word/character TF-IDF plus a deterministic visual descriptor."""

    def __init__(self, config: BaselineConfig | None = None) -> None:
        self.config = config or BaselineConfig()
        self.word_vectorizer = TfidfVectorizer(
            lowercase=True,
            ngram_range=(1, 2),
            min_df=2,
            sublinear_tf=True,
            max_features=self.config.max_word_features,
            strip_accents="unicode",
        )
        self.char_vectorizer = TfidfVectorizer(
            analyzer="char_wb",
            lowercase=True,
            ngram_range=(3, 5),
            min_df=2,
            sublinear_tf=True,
            max_features=self.config.max_char_features,
        )
        self.visual_scaler = StandardScaler()
        self.classifier = LogisticRegression(
            C=self.config.logistic_c,
            class_weight="balanced",
            max_iter=2_000,
            random_state=self.config.seed,
            solver="liblinear",
        )
        self.is_fitted = False

    def _fit_features(self, examples: Sequence[MemeExample]) -> sparse.csr_matrix:
        texts = [model_text(example) for example in examples]
        word = self.word_vectorizer.fit_transform(texts)
        char = self.char_vectorizer.fit_transform(texts)
        visual = visual_matrix(examples, use_images=self.config.use_images)
        scaled_visual = self.visual_scaler.fit_transform(visual) * self.config.visual_weight
        return sparse.hstack([word, char, sparse.csr_matrix(scaled_visual)], format="csr")

    def _transform_features(self, examples: Sequence[MemeExample]) -> sparse.csr_matrix:
        if not self.is_fitted:
            raise RuntimeError("The baseline must be fitted before prediction")
        texts = [model_text(example) for example in examples]
        word = self.word_vectorizer.transform(texts)
        char = self.char_vectorizer.transform(texts)
        visual = visual_matrix(examples, use_images=self.config.use_images)
        scaled_visual = self.visual_scaler.transform(visual) * self.config.visual_weight
        return sparse.hstack([word, char, sparse.csr_matrix(scaled_visual)], format="csr")

    def fit(self, examples: Sequence[MemeExample]) -> MultimodalBaseline:
        """Fit all transformations on the training partition only."""

        labels = [example.label for example in examples]
        if not examples or any(label not in {"NO", "YES"} for label in labels):
            raise ValueError("Training examples require hard NO/YES labels")
        target = np.asarray([1 if label == "YES" else 0 for label in labels], dtype=np.int8)
        features = self._fit_features(examples)
        self.classifier.fit(features, target)
        self.is_fitted = True
        return self

    def predict_yes_probability(self, examples: Sequence[MemeExample]) -> np.ndarray:
        """Return P(YES) for each example."""

        features = self._transform_features(examples)
        positive_index = int(np.where(self.classifier.classes_ == 1)[0][0])
        return np.asarray(self.classifier.predict_proba(features)[:, positive_index], dtype=float)

    def predict(self, examples: Sequence[MemeExample]) -> list[str]:
        """Apply the fixed baseline decision threshold."""

        scores = self.predict_yes_probability(examples)
        return ["YES" if score >= self.config.decision_threshold else "NO" for score in scores]

    def metadata(self) -> dict[str, Any]:
        """Describe the fitted feature space without exposing training data."""

        if not self.is_fitted:
            raise RuntimeError("The baseline is not fitted")
        return {
            "config": self.config.to_dict(),
            "word_features": len(self.word_vectorizer.vocabulary_),
            "character_features": len(self.char_vectorizer.vocabulary_),
            "visual_features": int(self.visual_scaler.n_features_in_),
            "classes": [int(value) for value in self.classifier.classes_],
        }

    def save(self, path: str | Path) -> Path:
        """Persist the fitted model bundle with joblib."""

        if not self.is_fitted:
            raise RuntimeError("Cannot save an unfitted baseline")
        destination = Path(path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self, destination)
        return destination

    @classmethod
    def load(cls, path: str | Path) -> MultimodalBaseline:
        """Load a model created by :meth:`save`."""

        model = joblib.load(Path(path))
        if not isinstance(model, cls):
            raise TypeError(f"Expected {cls.__name__}, found {type(model).__name__}")
        return model
