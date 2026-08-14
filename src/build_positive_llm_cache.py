"""Add rare confident-positive LLM pairs missed by the original random cap."""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

from src.features import MODEL_FEATURE_NAMES, extract_model_features
from src.hybrid import LLM_CONFIDENT_POSITIVE_THRESHOLD


def pair_keys(frame: pd.DataFrame) -> list[tuple[int, int]]:
    left = frame["id1"].to_numpy()
    right = frame["id2"].to_numpy()
    return list(zip(np.minimum(left, right), np.maximum(left, right)))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--items", default="assets/items.parquet")
    parser.add_argument("--matches", default="assets/matches_llm.parquet")
    parser.add_argument("--existing-pairs", default="output/llm_all_pairs_v6.parquet")
    parser.add_argument("--pairs-output", default="output/llm_positive_pairs_v10.parquet")
    parser.add_argument("--features-output", default="output/llm_positive_features_v10.npy")
    parser.add_argument("--max-per-category-group", type=int, default=20_000)
    parser.add_argument("--seed", type=int, default=2026)
    args = parser.parse_args()

    existing = pd.read_parquet(args.existing_pairs, columns=["id1", "id2"])
    seen = set(pair_keys(existing))
    all_matches = pd.read_parquet(
        args.matches, columns=["id1", "id2", "target"]
    )
    all_matches = all_matches[
        all_matches["target"] >= LLM_CONFIDENT_POSITIVE_THRESHOLD - 1e-6
    ].copy()
    print(f"Raw confident positives: {len(all_matches)}; existing keys: {len(seen)}")

    item_file = pq.ParquetFile(args.items)
    pair_parts: list[pd.DataFrame] = []
    feature_parts: list[np.ndarray] = []
    for group in range(item_file.num_row_groups):
        item_index = item_file.read_row_group(
            group, columns=["id", "category"]
        ).to_pandas()
        item_ids = set(item_index["id"].to_numpy())
        keep = all_matches["id1"].isin(item_ids) & all_matches["id2"].isin(item_ids)
        candidates = all_matches[keep].copy()
        if candidates.empty:
            print(f"row_group={group}: no positives", flush=True)
            continue
        category_by_id = dict(
            zip(item_index["id"], item_index["category"].astype(str))
        )
        candidates["category"] = candidates["id1"].map(category_by_id).astype(str)
        keys = pair_keys(candidates)
        candidates = candidates[
            np.fromiter((key not in seen for key in keys), dtype=bool, count=len(keys))
        ]
        if candidates.empty:
            print(f"row_group={group}: only duplicates", flush=True)
            continue
        sampled = pd.concat(
            [
                frame.sample(
                    n=min(len(frame), args.max_per_category_group),
                    random_state=args.seed + group,
                )
                for _, frame in candidates.groupby("category", sort=True)
            ],
            ignore_index=True,
        )
        seen.update(pair_keys(sampled))
        used_ids = set(sampled["id1"].to_numpy()) | set(sampled["id2"].to_numpy())
        full_items = item_file.read_row_group(group).to_pandas()
        selected_items = full_items[full_items["id"].isin(used_ids)]
        features = extract_model_features(sampled, selected_items)
        if features.shape != (len(sampled), len(MODEL_FEATURE_NAMES)):
            raise ValueError(f"Unexpected row-group feature shape {features.shape}")
        pair_parts.append(sampled[["id1", "id2", "target", "category"]])
        feature_parts.append(features)
        summary = sampled.groupby("category").size().to_dict()
        print(
            f"row_group={group}: selected={len(sampled)}, items={len(selected_items)}, "
            f"categories={summary}",
            flush=True,
        )

    pairs = pd.concat(pair_parts, ignore_index=True)
    features = np.vstack(feature_parts).astype(np.float32, copy=False)
    Path(args.pairs_output).parent.mkdir(parents=True, exist_ok=True)
    pairs.to_parquet(args.pairs_output, index=False)
    np.save(args.features_output, features)
    print(f"Saved {len(pairs)} extra positives and feature shape {features.shape}")
    print(pairs.groupby("category").size().sort_index().to_string())


if __name__ == "__main__":
    main()
