"""Evaluate class-balanced LLM hybrid training on product-disjoint OOD folds."""
from __future__ import annotations

import argparse
import time

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier

from src.hybrid import (
    class_balanced_source_weights,
    confident_llm_mask,
    hard_llm_labels,
    product_disjoint_pair_masks,
    reliability_balanced_source_weights,
)
from src.metrics import macro_pr_auc
from src.train_model import make_validation_split


def _estimator(args: argparse.Namespace) -> HistGradientBoostingClassifier:
    return HistGradientBoostingClassifier(
        learning_rate=0.05,
        max_iter=args.max_iter,
        max_leaf_nodes=args.max_leaves,
        min_samples_leaf=args.min_leaf,
        l2_regularization=args.l2,
        early_stopping=False,
        random_state=args.seed,
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--human-items", default="assets/items_human.parquet")
    parser.add_argument("--human-matches", default="assets/matches.parquet")
    parser.add_argument("--human-features", default="output/pair_features_v6.npy")
    parser.add_argument("--llm-pairs", default="output/llm_all_pairs_v6.parquet")
    parser.add_argument("--llm-features", default="output/llm_all_features_v6.npy")
    parser.add_argument("--extra-llm-pairs", default="")
    parser.add_argument("--extra-llm-features", default="")
    parser.add_argument("--extra-min-target", type=float, default=0.0)
    parser.add_argument("--extra-categories", default="all")
    parser.add_argument("--save-prefix", default="")
    parser.add_argument("--report-per-category", action="store_true")
    parser.add_argument("--folds", default="0,1,2")
    parser.add_argument("--n-folds", type=int, default=3)
    parser.add_argument("--llm-strength", type=float, default=0.25)
    parser.add_argument(
        "--train-confidence", choices=["confident", "all"], default="confident"
    )
    parser.add_argument("--soft-reliability", action="store_true")
    parser.add_argument("--required-relative-gain", type=float, default=0.10)
    parser.add_argument("--max-human-drop", type=float, default=0.01)
    parser.add_argument("--max-iter", type=int, default=250)
    parser.add_argument("--max-leaves", type=int, default=31)
    parser.add_argument("--min-leaf", type=int, default=40)
    parser.add_argument("--l2", type=float, default=2.0)
    parser.add_argument("--seed", type=int, default=2026)
    args = parser.parse_args()

    started = time.perf_counter()
    human_items = pd.read_parquet(args.human_items, columns=["id", "name", "category"])
    human_pairs = pd.read_parquet(args.human_matches, columns=["id1", "id2", "target"])
    human_features = np.load(args.human_features, mmap_mode="r")
    human_target = human_pairs["target"].to_numpy(dtype=np.int8)
    category_by_id = dict(zip(human_items["id"], human_items["category"].astype(str)))
    human_categories = human_pairs["id1"].map(category_by_id).astype(str).to_numpy()
    human_train, human_valid = make_validation_split(
        "id-tail", human_pairs, human_items, human_target, human_categories, 0.2, args.seed
    )

    llm_pairs = pd.read_parquet(args.llm_pairs, columns=["id1", "id2", "target", "category"])
    llm_features = np.load(args.llm_features, mmap_mode="r")
    llm_soft = llm_pairs["target"].to_numpy(dtype=np.float32)
    confident = confident_llm_mask(llm_soft)
    llm_target = hard_llm_labels(llm_soft)
    llm_categories = llm_pairs["category"].astype(str).to_numpy()

    extra_pairs = None
    extra_features = None
    extra_target = None
    extra_categories = None
    extra_confident = None
    if bool(args.extra_llm_pairs) != bool(args.extra_llm_features):
        raise ValueError("Both --extra-llm-pairs and --extra-llm-features are required")
    if args.extra_llm_pairs:
        extra_pairs = pd.read_parquet(
            args.extra_llm_pairs, columns=["id1", "id2", "target", "category"]
        )
        extra_features = np.load(args.extra_llm_features, mmap_mode="r")
        if len(extra_pairs) != len(extra_features):
            raise ValueError("Extra LLM pair and feature caches have different lengths")
        selected_extra_categories = (
            None
            if args.extra_categories.strip().lower() == "all"
            else {
                value.strip()
                for value in args.extra_categories.split(",")
                if value.strip()
            }
        )
        extra_keep = extra_pairs["target"].to_numpy(dtype=np.float32) >= args.extra_min_target
        if selected_extra_categories is not None:
            extra_keep &= extra_pairs["category"].astype(str).isin(
                selected_extra_categories
            ).to_numpy()
        extra_pairs = extra_pairs[extra_keep].reset_index(drop=True)
        extra_features = extra_features[extra_keep]
        extra_soft = extra_pairs["target"].to_numpy(dtype=np.float32)
        extra_confident = confident_llm_mask(extra_soft)
        extra_target = hard_llm_labels(extra_soft)
        extra_categories = extra_pairs["category"].astype(str).to_numpy()
        print(f"Extra LLM train source: {len(extra_pairs)} rows", flush=True)

    human_ids = set(human_items["id"].to_numpy())
    folds = [int(value) for value in args.folds.split(",")]
    baseline_models: dict[str, HistGradientBoostingClassifier] = {}
    for category in sorted(np.unique(human_categories)):
        rows = human_train[human_categories[human_train] == category]
        baseline_models[category] = _estimator(args).fit(
            human_features[rows], human_target[rows]
        )

    summaries: list[tuple[int, float, float, float, float]] = []
    for fold in folds:
        llm_train_mask, llm_valid_mask = product_disjoint_pair_masks(
            llm_pairs["id1"].to_numpy(), llm_pairs["id2"].to_numpy(), fold, args.n_folds
        )
        train_selection = (
            confident
            if args.train_confidence == "confident"
            else np.isfinite(llm_soft)
        )
        llm_train = np.flatnonzero(train_selection & llm_train_mask)
        llm_valid = np.flatnonzero(confident & llm_valid_mask)
        train_features = llm_features[llm_train]
        train_target = llm_target[llm_train]
        train_soft = llm_soft[llm_train]
        train_categories = llm_categories[llm_train]
        extra_train_count = 0
        if extra_pairs is not None:
            assert extra_features is not None
            assert extra_target is not None
            assert extra_categories is not None
            assert extra_confident is not None
            extra_train_mask, _ = product_disjoint_pair_masks(
                extra_pairs["id1"].to_numpy(),
                extra_pairs["id2"].to_numpy(),
                fold,
                args.n_folds,
            )
            extra_train = np.flatnonzero(extra_confident & extra_train_mask)
            extra_train_count = len(extra_train)
            train_features = np.vstack((train_features, extra_features[extra_train]))
            train_target = np.concatenate((train_target, extra_target[extra_train]))
            train_soft = np.concatenate((
                train_soft,
                extra_pairs["target"].to_numpy(dtype=np.float32)[extra_train],
            ))
            train_categories = np.concatenate(
                (train_categories, extra_categories[extra_train])
            )
        valid_ids = set(llm_pairs.iloc[llm_valid]["id1"]) | set(llm_pairs.iloc[llm_valid]["id2"])
        overlap = len(valid_ids & human_ids)
        if overlap:
            raise ValueError(f"Fold {fold} leaks {overlap} products into human training")

        human_scores = np.empty(len(human_valid), dtype=np.float32)
        baseline_scores = np.empty(len(llm_valid), dtype=np.float32)
        hybrid_scores = np.empty(len(llm_valid), dtype=np.float32)
        human_valid_categories = human_categories[human_valid]
        llm_valid_categories = llm_categories[llm_valid]

        print(
            f"Fold {fold}: LLM train={len(llm_train)}+{extra_train_count}, "
            f"valid={len(llm_valid)}, overlap=0"
        )
        for category in sorted(baseline_models):
            human_rows = human_train[human_categories[human_train] == category]
            llm_rows = np.flatnonzero(train_categories == category)
            human_positions = np.flatnonzero(human_valid_categories == category)
            llm_positions = np.flatnonzero(llm_valid_categories == category)
            baseline_scores[llm_positions] = baseline_models[category].predict_proba(
                llm_features[llm_valid[llm_positions]]
            )[:, 1]

            if args.soft_reliability:
                human_weights, llm_weights = reliability_balanced_source_weights(
                    human_target[human_rows],
                    train_target[llm_rows],
                    train_soft[llm_rows],
                    args.llm_strength,
                )
            else:
                human_weights, llm_weights = class_balanced_source_weights(
                    human_target[human_rows], train_target[llm_rows], args.llm_strength
                )
            model = _estimator(args).fit(
                np.vstack((human_features[human_rows], train_features[llm_rows])),
                np.concatenate((human_target[human_rows], train_target[llm_rows])),
                sample_weight=np.concatenate((human_weights, llm_weights)),
            )
            human_scores[human_positions] = model.predict_proba(
                human_features[human_valid[human_positions]]
            )[:, 1]
            hybrid_scores[llm_positions] = model.predict_proba(
                llm_features[llm_valid[llm_positions]]
            )[:, 1]

        human_macro, _ = macro_pr_auc(
            human_target[human_valid], human_scores, human_valid_categories
        )
        baseline_macro, _ = macro_pr_auc(
            llm_target[llm_valid], baseline_scores, llm_valid_categories
        )
        hybrid_macro, hybrid_per_category = macro_pr_auc(
            llm_target[llm_valid], hybrid_scores, llm_valid_categories
        )
        if args.save_prefix:
            np.save(f"{args.save_prefix}_fold{fold}.npy", hybrid_scores)
            np.save(f"{args.save_prefix}_human_fold{fold}.npy", human_scores)
        if args.report_per_category:
            for category in sorted(hybrid_per_category):
                print(f"    {category:<28} {hybrid_per_category[category]:.6f}")
        relative_gain = hybrid_macro / baseline_macro - 1.0
        summaries.append((fold, baseline_macro, hybrid_macro, relative_gain, human_macro))
        print(
            f"  baseline_ood={baseline_macro:.6f} hybrid_ood={hybrid_macro:.6f} "
            f"relative_gain={relative_gain:+.2%} human_id_tail={human_macro:.6f}",
            flush=True,
        )

    gains = np.asarray([row[3] for row in summaries])
    human_scores = np.asarray([row[4] for row in summaries])
    human_baseline_scores = np.empty(len(human_valid), dtype=np.float32)
    human_valid_categories = human_categories[human_valid]
    for category, model in baseline_models.items():
        positions = np.flatnonzero(human_valid_categories == category)
        human_baseline_scores[positions] = model.predict_proba(
            human_features[human_valid[positions]]
        )[:, 1]
    human_baseline, _ = macro_pr_auc(
        human_target[human_valid], human_baseline_scores, human_valid_categories
    )
    if args.save_prefix:
        np.save(f"{args.save_prefix}_human_baseline.npy", human_baseline_scores)
    worst_human_drop = human_baseline - float(human_scores.min())
    passed = float(gains.min()) >= args.required_relative_gain and worst_human_drop <= args.max_human_drop
    print(
        f"Summary: min_gain={gains.min():+.2%}, mean_gain={gains.mean():+.2%}, "
        f"human_baseline={human_baseline:.6f}, worst_human_drop={worst_human_drop:.6f}, "
        f"elapsed={time.perf_counter() - started:.1f}s -> {'PASS' if passed else 'FAIL'}"
    )
    if not passed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
