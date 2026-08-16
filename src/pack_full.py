"""Пакет со ВСЕМИ уверенными LLM-парами фолда 0 для обучения на Kaggle.

Мы обучались на 1.2 миллиона пар из 9.09 доступных — не потому, что больше не нужно, а
потому что канал отдавал 39 KB/s через VPN и заливка 4 GB заняла бы сутки. Без VPN канал
даёт 7.65 MB/s, и ограничение исчезает.

Тексты собираются потоково, по одной row-группе: словарь на 8 миллионов товаров занял бы
около 7 GB и не поместился бы в память этой машины.

Запуск:
    .venv/bin/python -m src.pack_full
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from src.cross_encoder import build_product_texts
from src.hybrid import confident_llm_mask, hard_llm_labels, product_disjoint_pair_masks


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--items", default="assets/items.parquet")
    ap.add_argument("--llm-matches", default="assets/matches_llm.parquet")
    ap.add_argument("--out", default="output/kaggle/full")
    ap.add_argument("--mode", default="compact")
    ap.add_argument("--holdout-fold", type=int, default=0)
    ap.add_argument("--n-folds", type=int, default=3)
    ap.add_argument("--valid-pairs", type=int, default=200_000)
    ap.add_argument("--seed", type=int, default=2026)
    args = ap.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    matches = pd.read_parquet(args.llm_matches, columns=["id1", "id2", "target"])
    matches = matches[confident_llm_mask(matches["target"].to_numpy(np.float32))].reset_index(drop=True)
    matches["label"] = hard_llm_labels(matches["target"].to_numpy(np.float32))
    train_mask, valid_mask = product_disjoint_pair_masks(
        matches["id1"].to_numpy(), matches["id2"].to_numpy(), args.holdout_fold, args.n_folds
    )
    train = matches[train_mask].reset_index(drop=True)
    rng = np.random.default_rng(args.seed)
    valid_rows = np.flatnonzero(valid_mask)
    if args.valid_pairs < len(valid_rows):
        valid_rows = np.sort(rng.choice(valid_rows, args.valid_pairs, replace=False))
    valid = matches.iloc[valid_rows].reset_index(drop=True)
    print(f"train={len(train):,} (положительных {train['label'].mean()*100:.1f}%), "
          f"valid={len(valid):,}", flush=True)

    train[["id1", "id2", "label"]].to_parquet(out / "llm_train.parquet", compression="zstd", index=False)
    valid[["id1", "id2", "label", "target"]].to_parquet(out / "llm_valid.parquet", compression="zstd", index=False)

    needed = set(pd.unique(np.concatenate([
        train["id1"].to_numpy(), train["id2"].to_numpy(),
        valid["id1"].to_numpy(), valid["id2"].to_numpy(),
    ])).tolist())
    print(f"нужно текстов: {len(needed):,}", flush=True)
    del matches, train, valid

    item_file = pq.ParquetFile(args.items)
    writer, written = None, 0
    schema = pa.schema([("id", pa.int64()), ("text", pa.string())])
    for group in range(item_file.num_row_groups):
        frame = item_file.read_row_group(group).to_pandas()
        frame = frame[frame["id"].isin(needed)]
        if len(frame):
            texts = build_product_texts(frame, args.mode)
            table = pa.Table.from_pydict(
                {"id": list(texts.keys()), "text": list(texts.values())}, schema=schema
            )
            if writer is None:
                writer = pq.ParquetWriter(out / "item_texts.parquet", schema, compression="zstd")
            writer.write_table(table)
            written += len(texts)
            del texts, table
        del frame
        print(f"  row-группа {group}: записано {written:,} из {len(needed):,}", flush=True)
    if writer is not None:
        writer.close()

    (out / "dataset-metadata.json").write_text(
        '{"title": "ECUP matching full", "id": "gnkitty/ecup-matching-full", '
        '"licenses": [{"name": "unknown"}]}'
    )
    total = sum(p.stat().st_size for p in out.iterdir() if p.is_file())
    print(f"\nПакет готов: {total/1e6:.0f} MB, текстов {written:,}")


if __name__ == "__main__":
    main()
