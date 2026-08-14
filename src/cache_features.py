"""Build a reusable model-feature matrix for an explicit items/pairs cache."""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from src.features import (
    MODEL_FEATURE_NAMES,
    extract_model_features,
    extract_v10_pair_features,
    extract_v11_pair_features,
    extract_v12_pair_features,
    extract_v13_pair_features,
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--items", required=True)
    parser.add_argument("--pairs", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument(
        "--reuse-tfidf",
        default="",
        help="Old feature matrix whose final text/name TF-IDF columns are unchanged",
    )
    parser.add_argument(
        "--incremental-block", choices=["v10", "v11", "v12", "v13"], default="v10"
    )
    args = parser.parse_args()

    items = pd.read_parquet(args.items, columns=["id", "name", "attributes", "category"])
    pairs = pd.read_parquet(args.pairs, columns=["id1", "id2"])
    if args.reuse_tfidf:
        old_features = np.load(args.reuse_tfidf, mmap_mode="r")
        if old_features.shape[0] != len(pairs) or old_features.shape[1] < 2:
            raise ValueError(f"Cannot reuse TF-IDF from shape {old_features.shape}")
        extractors = {
            "v10": extract_v10_pair_features,
            "v11": extract_v11_pair_features,
            "v12": extract_v12_pair_features,
            "v13": extract_v13_pair_features,
        }
        incremental = extractors[args.incremental_block](pairs, items)
        features = np.column_stack(
            (old_features[:, :-2], incremental, old_features[:, -2:])
        ).astype(np.float32)
    else:
        features = extract_model_features(pairs, items)
    if features.shape != (len(pairs), len(MODEL_FEATURE_NAMES)):
        raise ValueError(f"Unexpected feature shape {features.shape}")
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    np.save(output, features)
    print(f"Saved {features.shape} to {output}")


if __name__ == "__main__":
    main()
