"""Compare nonlinear estimators on the cached pair features.

This is a development utility; it never changes the production model artifact.
"""
from __future__ import annotations

import argparse
import time

import numpy as np
import pandas as pd
from sklearn.ensemble import ExtraTreesClassifier, HistGradientBoostingClassifier

from src.metrics import macro_pr_auc
from src.train_model import make_validation_split


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--features", default="output/pair_features_v3.npy")
    parser.add_argument("--items", default="assets/items_human.parquet")
    parser.add_argument("--matches", default="assets/matches.parquet")
    parser.add_argument("--split", choices=["random", "id-tail", "name-group"], default="random")
    parser.add_argument("--estimator", choices=["hist", "extra"], default="hist")
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--max-iter", type=int, default=250)
    parser.add_argument("--max-leaves", type=int, default=15)
    parser.add_argument("--min-leaf", type=int, default=40)
    parser.add_argument("--l2", type=float, default=2.0)
    args = parser.parse_args()

    items = pd.read_parquet(args.items, columns=["id", "name", "category"])
    matches = pd.read_parquet(args.matches, columns=["id1", "id2", "target"])
    features = np.load(args.features, mmap_mode="r")
    target = matches["target"].to_numpy(dtype=np.int8)
    category_by_id = dict(zip(items["id"], items["category"].astype(str)))
    categories = matches["id1"].map(category_by_id).astype(str).to_numpy()
    train_idx, valid_idx = make_validation_split(
        args.split, matches, items, target, categories, 0.2, args.seed
    )

    scores = np.empty(len(valid_idx), dtype=np.float32)
    valid_categories = categories[valid_idx]
    started = time.perf_counter()
    for category in sorted(np.unique(categories)):
        train_rows = train_idx[categories[train_idx] == category]
        valid_positions = np.flatnonzero(valid_categories == category)
        rows = valid_idx[valid_positions]
        if args.estimator == "hist":
            model = HistGradientBoostingClassifier(
                learning_rate=0.05,
                max_iter=args.max_iter,
                max_leaf_nodes=args.max_leaves,
                min_samples_leaf=args.min_leaf,
                l2_regularization=args.l2,
                early_stopping=False,
                random_state=args.seed,
            )
        else:
            model = ExtraTreesClassifier(
                n_estimators=300,
                max_features=0.8,
                min_samples_leaf=20,
                n_jobs=-1,
                random_state=args.seed,
            )
        model.fit(features[train_rows], target[train_rows])
        scores[valid_positions] = model.predict_proba(features[rows])[:, 1]
        category_ap, _ = macro_pr_auc(target[rows], scores[valid_positions], valid_categories[valid_positions])
        print(f"{category:<28} {category_ap:.4f}", flush=True)

    macro, per_category = macro_pr_auc(target[valid_idx], scores, valid_categories)
    print(f"split={args.split} estimator={args.estimator} macro={macro:.6f} "
          f"iter={args.max_iter} leaves={args.max_leaves} min_leaf={args.min_leaf} l2={args.l2:g} "
          f"elapsed={time.perf_counter() - started:.1f}s")
    print(f"range={min(per_category.values()):.4f}..{max(per_category.values()):.4f}")


if __name__ == "__main__":
    main()
