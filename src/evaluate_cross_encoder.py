"""Evaluate saved cross-encoder scores against the product-disjoint LLM fold.

The cross-encoder is expensive to run, so its predictions are cached separately.
This script recreates the exact structured v9 fold and measures category-wise rank
blends without scoring texts again.
"""
from __future__ import annotations

import argparse

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier

from src.hybrid import (
    class_balanced_source_weights,
    confident_llm_mask,
    hard_llm_labels,
    product_disjoint_pair_masks,
)
from src.metrics import macro_pr_auc
from src.train_model import category_ranks, make_validation_split


def estimator(seed: int) -> HistGradientBoostingClassifier:
    return HistGradientBoostingClassifier(
        learning_rate=0.05,
        max_iter=250,
        max_leaf_nodes=31,
        min_samples_leaf=40,
        l2_regularization=2.0,
        early_stopping=False,
        random_state=seed,
    )


def report(
    name: str, target: np.ndarray, scores: np.ndarray, categories: np.ndarray
) -> tuple[float, dict[str, float]]:
    macro, per_category = macro_pr_auc(target, scores, categories)
    print(f"{name:<32} {macro:.6f}", flush=True)
    return macro, per_category


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--human-items", default="assets/items_human.parquet")
    parser.add_argument("--human-matches", default="assets/matches.parquet")
    parser.add_argument("--human-features", default="output/pair_features_v6.npy")
    parser.add_argument("--llm-pairs", default="output/llm_all_pairs_v6.parquet")
    parser.add_argument("--llm-features", default="output/llm_all_features_v6.npy")
    parser.add_argument("--ce-scores", default="output/minilm_fold0_baseline256.npy")
    parser.add_argument("--fold", type=int, default=0)
    parser.add_argument("--n-folds", type=int, default=3)
    parser.add_argument("--llm-strength", type=float, default=0.25)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--save-scores", default="output/boost_v9_fold0_scores.npy")
    args = parser.parse_args()

    human_items = pd.read_parquet(args.human_items, columns=["id", "name", "category"])
    human_pairs = pd.read_parquet(args.human_matches, columns=["id1", "id2", "target"])
    human_features = np.load(args.human_features, mmap_mode="r")
    human_target = human_pairs["target"].to_numpy(dtype=np.int8)
    category_by_id = dict(zip(human_items["id"], human_items["category"].astype(str)))
    human_categories = human_pairs["id1"].map(category_by_id).astype(str).to_numpy()
    human_train, _ = make_validation_split(
        "id-tail", human_pairs, human_items, human_target, human_categories, 0.2, args.seed
    )

    llm_pairs = pd.read_parquet(
        args.llm_pairs, columns=["id1", "id2", "target", "category"]
    )
    llm_features = np.load(args.llm_features, mmap_mode="r")
    llm_soft = llm_pairs["target"].to_numpy(dtype=np.float32)
    llm_target = hard_llm_labels(llm_soft)
    llm_categories = llm_pairs["category"].astype(str).to_numpy()
    confident = confident_llm_mask(llm_soft)
    train_mask, valid_mask = product_disjoint_pair_masks(
        llm_pairs["id1"].to_numpy(),
        llm_pairs["id2"].to_numpy(),
        args.fold,
        args.n_folds,
    )
    llm_train = np.flatnonzero(confident & train_mask)
    llm_valid = np.flatnonzero(confident & valid_mask)
    valid_target = llm_target[llm_valid]
    valid_categories = llm_categories[llm_valid]

    ce_scores = np.load(args.ce_scores)
    if ce_scores.shape != (len(llm_valid),):
        raise ValueError(
            f"Expected {len(llm_valid)} CE scores in fold order, got {ce_scores.shape}"
        )

    structured_scores = np.empty(len(llm_valid), dtype=np.float32)
    for category in sorted(np.unique(valid_categories)):
        human_rows = human_train[human_categories[human_train] == category]
        llm_rows = llm_train[llm_categories[llm_train] == category]
        valid_positions = np.flatnonzero(valid_categories == category)
        human_weights, llm_weights = class_balanced_source_weights(
            human_target[human_rows], llm_target[llm_rows], args.llm_strength
        )
        model = estimator(args.seed).fit(
            np.vstack((human_features[human_rows], llm_features[llm_rows])),
            np.concatenate((human_target[human_rows], llm_target[llm_rows])),
            sample_weight=np.concatenate((human_weights, llm_weights)),
        )
        structured_scores[valid_positions] = model.predict_proba(
            llm_features[llm_valid[valid_positions]]
        )[:, 1]

    np.save(args.save_scores, structured_scores)
    structured_macro, structured_per_category = report(
        "structured v9", valid_target, structured_scores, valid_categories
    )
    ce_macro, ce_per_category = report(
        "cross-encoder", valid_target, ce_scores, valid_categories
    )

    structured_rank = category_ranks(structured_scores, valid_categories)
    ce_rank = category_ranks(ce_scores, valid_categories)
    best = (structured_macro, 0.0)
    for ce_weight in np.linspace(0.05, 0.75, 15):
        blend = (1.0 - ce_weight) * structured_rank + ce_weight * ce_rank
        macro, _ = report(
            f"rank blend CE={ce_weight:.2f}", valid_target, blend, valid_categories
        )
        best = max(best, (macro, float(ce_weight)))

    chosen = {
        category: "CE" if ce_per_category[category] > structured_per_category[category] else "v9"
        for category in sorted(structured_per_category)
    }
    category_choice_scores = structured_rank.copy()
    for category, source in chosen.items():
        if source == "CE":
            category_choice_scores[valid_categories == category] = ce_rank[
                valid_categories == category
            ]
    oracle_macro, _ = report(
        "per-category source (upper)",
        valid_target,
        category_choice_scores,
        valid_categories,
    )
    print("\nPer-category AP and selected source:")
    for category in sorted(structured_per_category):
        print(
            f"  {category:<28} v9={structured_per_category[category]:.4f} "
            f"CE={ce_per_category[category]:.4f} {chosen[category]}"
        )
    print(
        f"\nBest global blend={best[0]:.6f} at CE={best[1]:.2f}; "
        f"in-fold source-selection ceiling={oracle_macro:.6f}; "
        f"CE standalone={ce_macro:.6f}"
    )


if __name__ == "__main__":
    main()
