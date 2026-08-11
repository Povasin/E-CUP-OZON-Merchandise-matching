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


def make_validation_split(
    mode: str,
    matches: pd.DataFrame,
    items: pd.DataFrame,
    target: np.ndarray,
    categories: np.ndarray,
    valid_size: float,
    seed: int,
) -> tuple[np.ndarray, np.ndarray]:
    indices = np.arange(len(matches))
    if mode == "random":
        stratify = np.char.add(np.char.add(categories.astype(str), "|"), target.astype(str))
        return train_test_split(
            indices, test_size=valid_size, random_state=seed, stratify=stratify
        )

    if mode == "id-tail":
        valid_parts: list[np.ndarray] = []
        frame = pd.DataFrame({"idx": indices, "category": categories, "target": target,
                              "id1": matches["id1"].to_numpy()})
        for _, group in frame.groupby(["category", "target"], sort=False):
            ordered = group.sort_values("id1")["idx"].to_numpy()
            n_valid = max(1, int(round(len(ordered) * valid_size)))
            valid_parts.append(ordered[-n_valid:])
        valid_idx = np.sort(np.concatenate(valid_parts))
        train_idx = np.setdiff1d(indices, valid_idx, assume_unique=True)
        return train_idx, valid_idx

    if mode == "name-group":
        normalised_names = (
            items["name"].fillna("").astype(str).str.lower()
            .str.replace(r"[^0-9a-zа-яё]+", " ", regex=True).str.strip()
        )
        name_by_id = dict(zip(items["id"], normalised_names))
        signatures = matches["id1"].map(name_by_id).fillna("").astype(str)
        signatures = pd.Series(categories).str.cat(signatures, sep="|")
        hashes = pd.util.hash_pandas_object(signatures, index=False).to_numpy(dtype=np.uint64)
        folds = max(2, int(round(1.0 / valid_size)))
        valid_mask = hashes % folds == seed % folds
        return indices[~valid_mask], indices[valid_mask]

    raise ValueError(f"Unknown validation split: {mode}")


def category_ranks(scores: np.ndarray, categories: np.ndarray) -> np.ndarray:
    frame = pd.DataFrame({"score": scores, "category": categories})
    return frame.groupby("category", sort=False)["score"].rank(pct=True).to_numpy(dtype=np.float32)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--items", default="assets/items_human.parquet")
    parser.add_argument("--matches", default="assets/matches.parquet")
    parser.add_argument("--output", default="models/pair_logreg.npz")
    parser.add_argument("--valid-size", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--c", type=float, default=1.0)
    parser.add_argument("--feature-cache", default=None, help="Optional .npy cache for extracted features")
    parser.add_argument("--validation-split", choices=["random", "id-tail", "name-group"], default="random")
    parser.add_argument("--skip-refit", action="store_true", help="Validate without replacing the production model")
    args = parser.parse_args()

    print("Loading labelled pairs and product cards...")
    items = pd.read_parquet(args.items, columns=["id", "name", "attributes", "category"])
    matches = pd.read_parquet(args.matches, columns=["id1", "id2", "target"])
    category_by_id = dict(zip(items["id"], items["category"].astype(str)))
    categories = matches["id1"].map(category_by_id).fillna(matches["id2"].map(category_by_id)).astype(str).to_numpy()
    target = matches["target"].to_numpy(dtype=np.int8)

    cache_path = Path(args.feature_cache) if args.feature_cache else None
    if cache_path and cache_path.exists():
        print(f"Loading cached features: {cache_path}")
        features = np.load(cache_path, mmap_mode="r")
        if features.shape != (len(matches), len(MODEL_FEATURE_NAMES)):
            raise ValueError(f"Unexpected cached feature shape: {features.shape}")
    else:
        print("Extracting relational and lexical features...")
        features = extract_model_features(matches, items)
        if cache_path:
            cache_path.parent.mkdir(parents=True, exist_ok=True)
            np.save(cache_path, features)
            print(f"Saved feature cache: {cache_path}")
    print(f"Features: {features.shape} ({', '.join(MODEL_FEATURE_NAMES)})")

    indices = np.arange(len(matches))
    train_idx, valid_idx = make_validation_split(
        args.validation_split, matches, items, target, categories, args.valid_size, args.seed
    )
    print(f"Split={args.validation_split}: train={len(train_idx)}, valid={len(valid_idx)}, "
          f"valid positive rate={target[valid_idx].mean():.4f}")
    print(f"Training category models on {len(train_idx)} pairs...")
    names, coef, intercept = fit_models(features, target, categories, train_idx, args.c)
    valid_scores = predict_models(features[valid_idx], categories[valid_idx], names, coef, intercept)
    macro, per_category = macro_pr_auc(target[valid_idx], valid_scores, categories[valid_idx])
    fallback_row = int(np.flatnonzero(names == FALLBACK_CATEGORY)[0])
    valid_design = expand_features(features[valid_idx])
    global_scores = valid_design @ coef[fallback_row] + intercept[fallback_row]
    global_macro, _ = macro_pr_auc(target[valid_idx], global_scores, categories[valid_idx])
    baseline, _ = macro_pr_auc(target[valid_idx], features[valid_idx, -2], categories[valid_idx])
    print(f"Holdout text TF-IDF: {baseline:.6f}")
    print(f"Holdout global supervised: {global_macro:.6f}")
    print(f"Holdout supervised: {macro:.6f}")
    model_rank = category_ranks(valid_scores, categories[valid_idx])
    global_rank = category_ranks(global_scores, categories[valid_idx])
    tfidf_rank = category_ranks(np.asarray(features[valid_idx, -2]), categories[valid_idx])
    for weight in (0.25, 0.50, 0.75):
        blend = weight * model_rank + (1.0 - weight) * global_rank
        blend_macro, _ = macro_pr_auc(target[valid_idx], blend, categories[valid_idx])
        print(f"Holdout category/global blend category={weight:.2f}: {blend_macro:.6f}")
    for weight in (0.25, 0.50, 0.75):
        blend = weight * model_rank + (1.0 - weight) * tfidf_rank
        blend_macro, _ = macro_pr_auc(target[valid_idx], blend, categories[valid_idx])
        print(f"Holdout rank blend supervised={weight:.2f}: {blend_macro:.6f}")
    for category, score in sorted(per_category.items(), key=lambda item: item[1]):
        print(f"  {category:<28} {score:.4f}")

    if args.skip_refit:
        print("Validation only; production artifact was not changed.")
        return

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
