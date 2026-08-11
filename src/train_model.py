"""Train and validate the portable supervised pair matcher.

Usage:
    .venv/bin/python -m src.train_model
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

from src.features import MODEL_FEATURE_NAMES, extract_model_features
from src.metrics import macro_pr_auc
from src.model import FALLBACK_CATEGORY, expand_features


def _fit_one(x: np.ndarray, y: np.ndarray, c: float) -> tuple[np.ndarray, float]:
    model = LogisticRegression(C=c, solver="liblinear", max_iter=500, random_state=2026)
    model.fit(x, y)
    return model.coef_[0].astype(np.float32), float(model.intercept_[0])


def fit_models(
    features: np.ndarray,
    target: np.ndarray,
    categories: np.ndarray,
    indices: np.ndarray,
    c: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    design = expand_features(features)
    category_names = np.asarray(sorted(np.unique(categories).astype(str).tolist()) + [FALLBACK_CATEGORY])
    coefficients: list[np.ndarray] = []
    intercepts: list[float] = []
    for category in category_names:
        selected = indices if category == FALLBACK_CATEGORY else indices[categories[indices] == category]
        coef, intercept = _fit_one(design[selected], target[selected], c)
        coefficients.append(coef)
        intercepts.append(intercept)
    return category_names, np.vstack(coefficients), np.asarray(intercepts, dtype=np.float32)


def predict_models(
    features: np.ndarray,
    categories: np.ndarray,
    category_names: np.ndarray,
    coefficients: np.ndarray,
    intercepts: np.ndarray,
) -> np.ndarray:
    design = expand_features(features)
    row_by_category = {category: row for row, category in enumerate(category_names)}
    fallback = row_by_category[FALLBACK_CATEGORY]
    scores = np.empty(len(features), dtype=np.float32)
    for category in np.unique(categories):
        mask = categories == category
        row = row_by_category.get(category, fallback)
        scores[mask] = design[mask] @ coefficients[row] + intercepts[row]
    return scores


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--items", default="assets/items_human.parquet")
    parser.add_argument("--matches", default="assets/matches.parquet")
    parser.add_argument("--output", default="models/pair_logreg.npz")
    parser.add_argument("--valid-size", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--c", type=float, default=1.0)
    args = parser.parse_args()

    print("Loading labelled pairs and product cards...")
    items = pd.read_parquet(args.items, columns=["id", "name", "attributes", "category"])
    matches = pd.read_parquet(args.matches, columns=["id1", "id2", "target"])
    category_by_id = dict(zip(items["id"], items["category"].astype(str)))
    categories = matches["id1"].map(category_by_id).fillna(matches["id2"].map(category_by_id)).astype(str).to_numpy()
    target = matches["target"].to_numpy(dtype=np.int8)

    print("Extracting relational and lexical features...")
    features = extract_model_features(matches, items)
    print(f"Features: {features.shape} ({', '.join(MODEL_FEATURE_NAMES)})")

    indices = np.arange(len(matches))
    stratify = np.char.add(np.char.add(categories.astype(str), "|"), target.astype(str))
    train_idx, valid_idx = train_test_split(
        indices, test_size=args.valid_size, random_state=args.seed, stratify=stratify
    )
    print(f"Training category models on {len(train_idx)} pairs...")
    names, coef, intercept = fit_models(features, target, categories, train_idx, args.c)
    valid_scores = predict_models(features[valid_idx], categories[valid_idx], names, coef, intercept)
    macro, per_category = macro_pr_auc(target[valid_idx], valid_scores, categories[valid_idx])
    baseline, _ = macro_pr_auc(target[valid_idx], features[valid_idx, -2], categories[valid_idx])
    print(f"Holdout text TF-IDF: {baseline:.6f}")
    print(f"Holdout supervised: {macro:.6f}")
    for category, score in sorted(per_category.items(), key=lambda item: item[1]):
        print(f"  {category:<28} {score:.4f}")

    print("Refitting on all labelled pairs...")
    names, coef, intercept = fit_models(features, target, categories, indices, args.c)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        output,
        categories=names,
        coef=coef,
        intercept=intercept,
        feature_names=np.asarray(MODEL_FEATURE_NAMES),
    )
    print(f"Saved {output} ({output.stat().st_size / 1024:.1f} KiB)")


if __name__ == "__main__":
    main()
