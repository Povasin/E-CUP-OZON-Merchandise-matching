"""Train and serialize the portable category-specific boosted matcher."""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier

from src.features import MODEL_FEATURE_NAMES, extract_model_features
from src.model import FALLBACK_CATEGORY


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--items", default="assets/items_human.parquet")
    parser.add_argument("--matches", default="assets/matches.parquet")
    parser.add_argument("--feature-cache", default="output/pair_features_v4.npy")
    parser.add_argument("--output", default="models/pair_boost.npz")
    parser.add_argument("--max-iter", type=int, default=250)
    parser.add_argument("--max-leaves", type=int, default=31)
    parser.add_argument("--min-leaf", type=int, default=40)
    parser.add_argument("--l2", type=float, default=2.0)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--use-llm", action="store_true")
    parser.add_argument("--llm-features", default="output/llm_all_features_v4.npy")
    parser.add_argument("--llm-pairs", default="output/llm_all_pairs_v4.parquet")
    parser.add_argument("--human-weight", type=float, default=10.0)
    parser.add_argument(
        "--hybrid-categories",
        default=(
            "Бытовая техника,Бытовая химия,Галантерея и аксессуары,Дом и сад,"
            "Красота и гигиена,Продукты питания,Строительство и ремонт,"
            "Электроника,Ювелирные изделия"
        ),
    )
    args = parser.parse_args()

    items = pd.read_parquet(args.items, columns=["id", "name", "attributes", "category"])
    matches = pd.read_parquet(args.matches, columns=["id1", "id2", "target"])
    cache = Path(args.feature_cache)
    if cache.exists():
        print(f"Loading cached features: {cache}")
        features = np.load(cache, mmap_mode="r")
    else:
        print("Extracting model features...")
        features = extract_model_features(matches, items)
        cache.parent.mkdir(parents=True, exist_ok=True)
        np.save(cache, features)
    if features.shape != (len(matches), len(MODEL_FEATURE_NAMES)):
        raise ValueError(f"Unexpected feature shape: {features.shape}")

    target = matches["target"].to_numpy(dtype=np.int8)
    category_by_id = dict(zip(items["id"], items["category"].astype(str)))
    categories = matches["id1"].map(category_by_id).astype(str).to_numpy()
    model_names = np.asarray(sorted(np.unique(categories).tolist()) + [FALLBACK_CATEGORY])

    hybrid_categories = {value.strip() for value in args.hybrid_categories.split(",") if value.strip()}
    llm_features = None
    llm_target = None
    llm_categories = None
    if args.use_llm:
        llm_features = np.load(args.llm_features, mmap_mode="r")
        llm_pairs = pd.read_parquet(args.llm_pairs, columns=["target", "category"])
        if len(llm_features) != len(llm_pairs):
            raise ValueError("LLM feature and pair caches have different lengths")
        llm_target = (llm_pairs["target"].to_numpy(dtype=np.float32) >= 0.5).astype(np.int8)
        llm_categories = llm_pairs["category"].astype(str).to_numpy()
        print(f"Selective LLM hybrid categories: {sorted(hybrid_categories)}")

    max_nodes = 2 * args.max_leaves - 1
    shape = (len(model_names), args.max_iter, max_nodes)
    feature_idx = np.zeros(shape, dtype=np.int16)
    threshold = np.zeros(shape, dtype=np.float32)
    missing_left = np.zeros(shape, dtype=np.uint8)
    left = np.zeros(shape, dtype=np.int16)
    right = np.zeros(shape, dtype=np.int16)
    is_leaf = np.ones(shape, dtype=np.uint8)
    value = np.zeros(shape, dtype=np.float32)
    max_depth = np.zeros(shape[:2], dtype=np.int8)
    baseline = np.zeros(len(model_names), dtype=np.float32)

    for model_row, category in enumerate(model_names):
        rows = np.arange(len(matches)) if category == FALLBACK_CATEGORY else np.flatnonzero(categories == category)
        fit_features = features[rows]
        fit_target = target[rows]
        sample_weight = None
        source = "human"
        if args.use_llm and category in hybrid_categories:
            assert llm_features is not None and llm_target is not None and llm_categories is not None
            llm_rows = np.flatnonzero(llm_categories == category)
            fit_features = np.vstack((llm_features[llm_rows], fit_features))
            fit_target = np.concatenate((llm_target[llm_rows], fit_target))
            sample_weight = np.concatenate((
                np.ones(len(llm_rows), dtype=np.float32),
                np.full(len(rows), args.human_weight, dtype=np.float32),
            ))
            source = f"hybrid(llm={len(llm_rows)}, human={len(rows)}x{args.human_weight:g})"
        print(f"Training {category}: {len(fit_target)} pairs [{source}]", flush=True)
        model = HistGradientBoostingClassifier(
            learning_rate=0.05,
            max_iter=args.max_iter,
            max_leaf_nodes=args.max_leaves,
            min_samples_leaf=args.min_leaf,
            l2_regularization=args.l2,
            early_stopping=False,
            random_state=args.seed,
        )
        model.fit(fit_features, fit_target, sample_weight=sample_weight)
        baseline[model_row] = float(model._baseline_prediction.ravel()[0])
        if len(model._predictors) != args.max_iter:
            raise ValueError(f"Unexpected number of trees for {category}: {len(model._predictors)}")
        for tree, predictors in enumerate(model._predictors):
            nodes = predictors[0].nodes
            n_nodes = len(nodes)
            feature_idx[model_row, tree, :n_nodes] = nodes["feature_idx"]
            threshold[model_row, tree, :n_nodes] = nodes["num_threshold"]
            missing_left[model_row, tree, :n_nodes] = nodes["missing_go_to_left"]
            left[model_row, tree, :n_nodes] = nodes["left"]
            right[model_row, tree, :n_nodes] = nodes["right"]
            is_leaf[model_row, tree, :n_nodes] = nodes["is_leaf"]
            value[model_row, tree, :n_nodes] = nodes["value"]
            max_depth[model_row, tree] = int(nodes["depth"].max())

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        output,
        categories=model_names,
        feature_names=np.asarray(MODEL_FEATURE_NAMES),
        baseline=baseline,
        feature_idx=feature_idx,
        threshold=threshold,
        missing_left=missing_left,
        left=left,
        right=right,
        is_leaf=is_leaf,
        value=value,
        max_depth=max_depth,
    )
    print(f"Saved {output} ({output.stat().st_size / 1024 / 1024:.2f} MiB)")


if __name__ == "__main__":
    main()
