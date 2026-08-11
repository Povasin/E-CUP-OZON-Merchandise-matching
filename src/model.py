"""Portable category-aware linear matcher.

Only NumPy is used at inference.  The saved coefficients therefore do not depend
on the scikit-learn/joblib version installed in the competition image.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np


THRESHOLDS = np.asarray((0.0, 0.10, 0.25, 0.50, 0.75, 0.90, 0.99), dtype=np.float32)
FALLBACK_CATEGORY = "__all__"


def expand_features(features: np.ndarray) -> np.ndarray:
    """Add simple nonlinearities while keeping inference a single dot product."""
    features = np.asarray(features, dtype=np.float32)
    blocks = [features, features * features, np.sqrt(np.clip(features, 0.0, None))]
    blocks.extend((features > threshold).astype(np.float32) for threshold in THRESHOLDS)
    return np.column_stack(blocks).astype(np.float32, copy=False)


class PairModel:
    def __init__(self, path: str | Path) -> None:
        artifact = np.load(path, allow_pickle=False)
        from src.features import MODEL_FEATURE_NAMES

        saved_features = artifact["feature_names"].astype(str).tolist()
        if saved_features != MODEL_FEATURE_NAMES:
            raise ValueError("Model artifact and runtime feature definitions differ")
        self.categories = artifact["categories"].astype(str)
        self.coef = artifact["coef"].astype(np.float32)
        self.intercept = artifact["intercept"].astype(np.float32)
        expected_width = len(saved_features) * (3 + len(THRESHOLDS))
        if self.coef.shape[1] != expected_width:
            raise ValueError("Model artifact and runtime feature expansion differ")
        self.category_to_row = {category: row for row, category in enumerate(self.categories)}
        self.fallback_row = self.category_to_row[FALLBACK_CATEGORY]

    def predict(self, features: np.ndarray, categories: np.ndarray) -> np.ndarray:
        design = expand_features(features)
        categories = np.asarray(categories).astype(str)
        scores = np.empty(len(design), dtype=np.float32)
        for category in np.unique(categories):
            mask = categories == category
            row = self.category_to_row.get(category, self.fallback_row)
            scores[mask] = design[mask] @ self.coef[row] + self.intercept[row]
        return scores
