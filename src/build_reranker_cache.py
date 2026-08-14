"""Build a class/category-balanced, product-disjoint reranker training cache."""
from __future__ import annotations

import argparse

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

from src.hybrid import confident_llm_mask, hard_llm_labels, product_disjoint_pair_masks


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--items", default="assets/items.parquet")
    parser.add_argument("--pairs", default="output/llm_all_pairs_v6.parquet")
    parser.add_argument("--pairs-output", default="output/gte_train_pairs.parquet")
    parser.add_argument("--items-output", default="output/gte_train_items.parquet")
    parser.add_argument("--holdout-fold", type=int, default=0)
    parser.add_argument("--n-folds", type=int, default=3)
    parser.add_argument("--per-class-category", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=2026)
    args = parser.parse_args()

    pairs = pd.read_parquet(
        args.pairs, columns=["id1", "id2", "target", "category"]
    )
    soft = pairs["target"].to_numpy(dtype=np.float32)
    train_mask, _ = product_disjoint_pair_masks(
        pairs["id1"].to_numpy(),
        pairs["id2"].to_numpy(),
        args.holdout_fold,
        args.n_folds,
    )
    eligible = pairs[confident_llm_mask(soft) & train_mask].copy()
    eligible["label"] = hard_llm_labels(eligible["target"].to_numpy(dtype=np.float32))
    sampled = pd.concat(
        [
            frame.sample(
                n=min(len(frame), args.per_class_category), random_state=args.seed
            )
            for _, frame in eligible.groupby(["category", "label"], sort=True)
        ],
        ignore_index=True,
    )
    sampled = sampled.sample(frac=1.0, random_state=args.seed).reset_index(drop=True)
    sampled.to_parquet(args.pairs_output, index=False)

    needed_ids = set(sampled["id1"].to_numpy()) | set(sampled["id2"].to_numpy())
    item_file = pq.ParquetFile(args.items)
    item_parts: list[pd.DataFrame] = []
    for group in range(item_file.num_row_groups):
        item_ids = item_file.read_row_group(group, columns=["id"]).to_pandas()["id"]
        present = set(item_ids.to_numpy()) & needed_ids
        if present:
            full = item_file.read_row_group(group).to_pandas()
            item_parts.append(full[full["id"].isin(present)])
        print(
            f"row_group={group} found={len(present)} remaining={len(needed_ids) - sum(len(x) for x in item_parts)}",
            flush=True,
        )
    items = pd.concat(item_parts, ignore_index=True).drop_duplicates("id")
    missing = needed_ids - set(items["id"].to_numpy())
    if missing:
        raise ValueError(f"Missing {len(missing)} selected products")
    items.to_parquet(args.items_output, index=False)
    counts = sampled.groupby(["category", "label"]).size().unstack(fill_value=0)
    print(f"Saved {len(sampled)} pairs and {len(items)} items")
    print(counts.to_string())


if __name__ == "__main__":
    main()
