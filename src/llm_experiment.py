"""Evaluate LLM-labelled data from selected full-item parquet row groups.

The large derived caches are written under output/ and are ignored by git.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
from sklearn.ensemble import HistGradientBoostingClassifier

from src.features import MODEL_FEATURE_NAMES, extract_model_features
from src.hybrid import confident_llm_mask
from src.metrics import macro_pr_auc
from src.train_model import category_ranks, make_validation_split


def build_llm_cache(
    item_path: str,
    match_path: str,
    row_groups: list[int],
    feature_path: Path,
    pair_path: Path,
    max_pairs_per_category_group: int,
) -> tuple[np.ndarray, pd.DataFrame]:
    item_file = pq.ParquetFile(item_path)
    all_matches = pd.read_parquet(match_path)
    parts: list[pd.DataFrame] = []
    item_parts: list[pd.DataFrame] = []
    print(f"Sampling item row groups: {row_groups}")
    for group in row_groups:
        item_index = item_file.read_row_group(group, columns=["id", "category"]).to_pandas()
        item_ids = set(item_index["id"].to_numpy())
        keep = all_matches["id1"].isin(item_ids) & all_matches["id2"].isin(item_ids)
        group_pairs = all_matches[keep].copy()
        category_by_id = dict(zip(item_index["id"], item_index["category"].astype(str)))
        group_pairs["category"] = group_pairs["id1"].map(category_by_id).astype(str)
        group_pairs = pd.concat(
            [frame.sample(n=min(len(frame), max_pairs_per_category_group), random_state=2026)
             for _, frame in group_pairs.groupby("category", sort=False)],
            ignore_index=True,
        )
        used_ids = pd.unique(pd.concat([group_pairs["id1"], group_pairs["id2"]], ignore_index=True))
        full_group = item_file.read_row_group(group).to_pandas()
        selected_items = full_group[full_group["id"].isin(used_ids)].copy()
        item_parts.append(selected_items)
        parts.append(group_pairs)
        cats = ", ".join(sorted(group_pairs["category"].unique()))
        print(f"  row_group={group}: pairs={len(group_pairs)}, items={len(selected_items)}, cats={cats}",
              flush=True)

    pairs = pd.concat(parts, ignore_index=True)
    items = pd.concat(item_parts, ignore_index=True).drop_duplicates("id")
    print(f"Selected {len(pairs)} LLM pairs across {pairs['category'].nunique()} categories")

    features = extract_model_features(pairs, items)
    feature_path.parent.mkdir(parents=True, exist_ok=True)
    np.save(feature_path, features)
    pairs[["id1", "id2", "target", "category"]].to_parquet(pair_path, index=False)
    print(f"Saved caches: {feature_path}, {pair_path}")
    return features, pairs


def fit_predict(
    train_features: np.ndarray,
    train_target: np.ndarray,
    train_categories: np.ndarray,
    valid_features: np.ndarray,
    valid_categories: np.ndarray,
    sample_weight: np.ndarray | None = None,
) -> np.ndarray:
    scores = np.empty(len(valid_features), dtype=np.float32)
    for category in sorted(np.unique(valid_categories)):
        train_rows = np.flatnonzero(train_categories == category)
        valid_rows = np.flatnonzero(valid_categories == category)
        model = HistGradientBoostingClassifier(
            learning_rate=0.05,
            max_iter=250,
            max_leaf_nodes=31,
            min_samples_leaf=40,
            l2_regularization=2.0,
            early_stopping=False,
            random_state=2026,
        )
        weights = sample_weight[train_rows] if sample_weight is not None else None
        model.fit(train_features[train_rows], train_target[train_rows], sample_weight=weights)
        scores[valid_rows] = model.predict_proba(valid_features[valid_rows])[:, 1]
    return scores


def report(name: str, target: np.ndarray, scores: np.ndarray, categories: np.ndarray) -> float:
    macro, _ = macro_pr_auc(target, scores, categories)
    print(f"{name:<28} {macro:.6f}", flush=True)
    return macro


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--items", default="assets/items.parquet")
    parser.add_argument("--llm-matches", default="assets/matches_llm.parquet")
    parser.add_argument("--human-items", default="assets/items_human.parquet")
    parser.add_argument("--human-matches", default="assets/matches.parquet")
    parser.add_argument("--human-features", default="output/pair_features_v6.npy")
    parser.add_argument("--row-groups", default="all")
    parser.add_argument("--max-pairs-per-category-group", type=int, default=20_000)
    parser.add_argument("--llm-features", default="output/llm_all_features_v6.npy")
    parser.add_argument("--llm-pairs", default="output/llm_all_pairs_v6.parquet")
    parser.add_argument("--quick", action="store_true", help="Only human/LLM models and rank blends")
    parser.add_argument("--hybrid-only", action="store_true", help="Only human and weight-10 hybrid")
    parser.add_argument("--validation-split", choices=["random", "id-tail", "name-group"], default="id-tail")
    args = parser.parse_args()

    feature_path = Path(args.llm_features)
    pair_path = Path(args.llm_pairs)
    if feature_path.exists() and pair_path.exists():
        print("Loading cached LLM sample")
        llm_features = np.load(feature_path, mmap_mode="r")
        llm_pairs = pd.read_parquet(pair_path)
    else:
        row_groups = (
            list(range(pq.ParquetFile(args.items).num_row_groups))
            if args.row_groups == "all"
            else [int(value) for value in args.row_groups.split(",")]
        )
        llm_features, llm_pairs = build_llm_cache(
            args.items, args.llm_matches, row_groups, feature_path, pair_path,
            args.max_pairs_per_category_group,
        )
    if llm_features.shape != (len(llm_pairs), len(MODEL_FEATURE_NAMES)):
        raise ValueError(f"Unexpected LLM feature shape: {llm_features.shape}")

    human_items = pd.read_parquet(args.human_items, columns=["id", "name", "category"])
    human_pairs = pd.read_parquet(args.human_matches, columns=["id1", "id2", "target"])
    human_features = np.load(args.human_features, mmap_mode="r")
    human_target = human_pairs["target"].to_numpy(dtype=np.int8)
    category_by_id = dict(zip(human_items["id"], human_items["category"].astype(str)))
    human_categories = human_pairs["id1"].map(category_by_id).astype(str).to_numpy()
    train_idx, valid_idx = make_validation_split(
        args.validation_split, human_pairs, human_items, human_target, human_categories, 0.2, 2026
    )
    valid_features = human_features[valid_idx]
    valid_target = human_target[valid_idx]
    valid_categories = human_categories[valid_idx]

    llm_soft = llm_pairs["target"].to_numpy(dtype=np.float32)
    llm_target = (llm_soft >= 0.5).astype(np.int8)
    llm_categories = llm_pairs["category"].astype(str).to_numpy()

    print(f"\nManual {args.validation_split} validation:")
    human_scores = fit_predict(
        human_features[train_idx], human_target[train_idx], human_categories[train_idx],
        valid_features, valid_categories,
    )
    report("human only", valid_target, human_scores, valid_categories)

    if args.hybrid_only:
        hybrid_features = np.vstack((llm_features, human_features[train_idx]))
        hybrid_target = np.concatenate((llm_target, human_target[train_idx]))
        hybrid_categories = np.concatenate((llm_categories, human_categories[train_idx]))
        weights = np.concatenate((np.ones(len(llm_target), dtype=np.float32),
                                  np.full(len(train_idx), 10.0, dtype=np.float32)))
        hybrid_scores = fit_predict(
            hybrid_features, hybrid_target, hybrid_categories,
            valid_features, valid_categories, weights,
        )
        report("hybrid human_weight=10", valid_target, hybrid_scores, valid_categories)
        _, human_per_category = macro_pr_auc(valid_target, human_scores, valid_categories)
        _, hybrid_per_category = macro_pr_auc(valid_target, hybrid_scores, valid_categories)
        print("Per-category hybrid delta:")
        for category in sorted(human_per_category):
            delta = hybrid_per_category[category] - human_per_category[category]
            print(f"  {category:<28} {delta:+.5f}")
        return

    llm_scores = fit_predict(
        llm_features, llm_target, llm_categories, valid_features, valid_categories
    )
    report("LLM all", valid_target, llm_scores, valid_categories)

    human_rank = category_ranks(human_scores, valid_categories)
    llm_rank = category_ranks(llm_scores, valid_categories)
    for llm_weight in (0.05, 0.10, 0.20, 0.30):
        blend = (1.0 - llm_weight) * human_rank + llm_weight * llm_rank
        report(f"rank blend LLM={llm_weight:.2f}", valid_target, blend, valid_categories)

    if args.quick:
        return

    confident = confident_llm_mask(llm_soft)
    confident_scores = fit_predict(
        llm_features[confident], llm_target[confident], llm_categories[confident],
        valid_features, valid_categories,
    )
    report("LLM confident", valid_target, confident_scores, valid_categories)

    hybrid_features = np.vstack((llm_features, human_features[train_idx]))
    hybrid_target = np.concatenate((llm_target, human_target[train_idx]))
    hybrid_categories = np.concatenate((llm_categories, human_categories[train_idx]))
    for human_weight in (2.0, 5.0, 10.0):
        weights = np.concatenate((np.ones(len(llm_target), dtype=np.float32),
                                  np.full(len(train_idx), human_weight, dtype=np.float32)))
        hybrid_scores = fit_predict(
            hybrid_features, hybrid_target, hybrid_categories,
            valid_features, valid_categories, weights,
        )
        report(f"hybrid human_weight={human_weight:g}", valid_target, hybrid_scores, valid_categories)


if __name__ == "__main__":
    main()
