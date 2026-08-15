"""Компактный пакет для обучения на Kaggle.

Сырые `assets/` весят 4.4 GB, из которых 4.1 GB — полный корпус товаров. Обучению
нужны не карточки целиком, а только тексты тех товаров, что попали в выборку пар.
После отбора и сжатия объём падает примерно в восемь раз, а на медленном канале это
разница между четырьмя часами и полутора сутками.

Вся тяжёлая подготовка (отбор пар, product-disjoint разбиение, сборка текстов) делается
здесь, локально. На Kaggle уезжает готовое к обучению.

Запуск:
    .venv/bin/python -m src.pack_kaggle --train-pairs 3000000
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from src.cross_encoder import build_product_texts
from src.train_ce_large import load_texts_for_ids, select_llm_pairs


def write(frame: pd.DataFrame, path: Path) -> None:
    frame.to_parquet(path, compression="zstd", index=False)
    print(f"  {path.name:<24} {len(frame):>10,} строк  {path.stat().st_size / 1e6:>8.1f} MB",
          flush=True)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--items", default="assets/items.parquet")
    ap.add_argument("--llm-matches", default="assets/matches_llm.parquet")
    ap.add_argument("--human-items", default="assets/items_human.parquet")
    ap.add_argument("--human-matches", default="assets/matches.parquet")
    ap.add_argument("--out", default="output/kaggle/data")
    ap.add_argument("--mode", choices=["baseline", "compact", "name"], default="compact")
    ap.add_argument("--train-pairs", type=int, default=3_000_000)
    ap.add_argument("--valid-pairs", type=int, default=200_000)
    ap.add_argument("--holdout-fold", type=int, default=0)
    ap.add_argument("--n-folds", type=int, default=3)
    ap.add_argument("--seed", type=int, default=2026)
    args = ap.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    # Симлинки на сырые assets остались от прежней попытки залить всё целиком.
    for stale in out.iterdir():
        if stale.is_symlink():
            stale.unlink()

    train, valid = select_llm_pairs(
        args.llm_matches, args.holdout_fold, args.n_folds,
        args.train_pairs, args.valid_pairs, args.seed,
    )

    print("\nСборка текстов товаров из полного корпуса...", flush=True)
    needed = set(pd.unique(np.concatenate([
        train["id1"].to_numpy(), train["id2"].to_numpy(),
        valid["id1"].to_numpy(), valid["id2"].to_numpy(),
    ])).tolist())
    texts = load_texts_for_ids(args.items, needed, args.mode)

    print("\nЗапись пакета:", flush=True)
    write(train[["id1", "id2", "label"]], out / "llm_train.parquet")
    write(valid[["id1", "id2", "label", "target"]], out / "llm_valid.parquet")
    write(pd.DataFrame({"id": list(texts.keys()), "text": list(texts.values())}),
          out / "item_texts.parquet")
    del texts

    human_items = pd.read_parquet(args.human_items, columns=["id", "name", "attributes", "category"])
    human_texts = build_product_texts(human_items, args.mode)
    write(pd.DataFrame({
        "id": human_items["id"].to_numpy(),
        "text": [human_texts.get(int(i), "") for i in human_items["id"]],
        "category": human_items["category"].astype(str).to_numpy(),
    }), out / "human_texts.parquet")
    write(pd.read_parquet(args.human_matches, columns=["id1", "id2", "target"]),
          out / "human_matches.parquet")

    (out / "pack-info.json").write_text(json.dumps({
        "mode": args.mode, "train_pairs": len(train), "valid_pairs": len(valid),
        "holdout_fold": args.holdout_fold, "n_folds": args.n_folds, "seed": args.seed,
    }, indent=2))

    total = sum(p.stat().st_size for p in out.iterdir() if p.is_file())
    print(f"\nИтого пакет: {total / 1e6:.0f} MB (вместо 4400 MB сырых assets)")


if __name__ == "__main__":
    main()
