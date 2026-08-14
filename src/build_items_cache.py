"""Materialize only the product cards referenced by a pair parquet."""
from __future__ import annotations

import argparse

import pandas as pd
import pyarrow.parquet as pq


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--items", default="assets/items.parquet")
    parser.add_argument("--pairs", default="output/llm_all_pairs_v6.parquet")
    parser.add_argument("--output", default="output/llm_all_items_v10.parquet")
    args = parser.parse_args()

    pairs = pd.read_parquet(args.pairs, columns=["id1", "id2"])
    needed = set(pairs["id1"].to_numpy()) | set(pairs["id2"].to_numpy())
    item_file = pq.ParquetFile(args.items)
    parts: list[pd.DataFrame] = []
    found_ids: set[int] = set()
    for group in range(item_file.num_row_groups):
        ids = item_file.read_row_group(group, columns=["id"]).to_pandas()["id"]
        present = set(ids.to_numpy()) & needed
        if present:
            frame = item_file.read_row_group(group).to_pandas()
            selected = frame[frame["id"].isin(present)]
            parts.append(selected)
            found_ids.update(selected["id"].to_numpy())
        print(
            f"row_group={group} found={len(present)} total={len(found_ids)}/{len(needed)}",
            flush=True,
        )
    missing = needed - found_ids
    if missing:
        raise ValueError(f"Missing {len(missing)} referenced product cards")
    items = pd.concat(parts, ignore_index=True).drop_duplicates("id")
    items.to_parquet(args.output, index=False)
    print(f"Saved {len(items)} rows to {args.output}")


if __name__ == "__main__":
    main()
