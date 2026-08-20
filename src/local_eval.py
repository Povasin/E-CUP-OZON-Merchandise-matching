"""Локальная оценка решения на размеченных данных.

Идея: matches.parquet (ручная разметка target∈{0,1}) + items_human.parquet
дают честный размеченный набор со всеми 20 категориями — это зеркало тестовой
структуры, на котором можно посчитать ту же метрику (macro PR-AUC) локально.

Специфика устройства: у нас нет H100 и 20 ядер, поэтому:
  * метод v1 (tfidf) полностью CPU-шный и здесь работает так же, как в докере;
  * можно урезать выборку (--sample) и категории (--cats) для быстрых итераций;
  * товары ограничиваются id из выбранных пар — так же, как в реальном прогоне
    (в тест подаются только товары, участвующие в парах);
  * замеряется время и экстраполируется на объёмы Public (~115k) и Private (~275k)
    пар. Реальная машина мощнее, поэтому локальные тайминги — консервативная оценка.

Запуск:
    .venv/bin/python -m src.local_eval --sample 60000
    .venv/bin/python -m src.local_eval            # полный прогон (365k пар)
"""
from __future__ import annotations

import argparse
import time

import numpy as np
import pandas as pd

from src.data import attach_texts, build_item_text
from src.metrics import macro_pr_auc
from src.pipeline import predict_scores

PUBLIC_PAIRS = 115_000
PRIVATE_PAIRS = 275_000
PUBLIC_LIMIT_S = 6 * 60
PRIVATE_LIMIT_S = 13 * 60


def stratified_sample(matches: pd.DataFrame, cat_by_id: dict, n: int, seed: int = 1234) -> pd.DataFrame:
    """Подвыборка ~n пар с сохранением баланса по категориям."""
    cats = matches["id1"].map(cat_by_id)
    frac = min(1.0, n / len(matches))
    return (
        matches.assign(_cat=cats)
        .groupby("_cat", group_keys=False)
        .apply(lambda g: g.sample(frac=frac, random_state=seed), include_groups=False)
        .drop(columns=[c for c in ["_cat"] if c in matches.columns], errors="ignore")
    )


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--items", default="assets/items_human.parquet")
    ap.add_argument("--matches", default="assets/matches.parquet")
    ap.add_argument("--method", default="tfidf")
    ap.add_argument("--sample", type=int, default=0, help="0 = все пары; иначе ~N пар (стратиф.)")
    ap.add_argument("--cats", type=str, default=None, help="через запятую: ограничить категории")
    ap.add_argument("--seed", type=int, default=1234)
    args = ap.parse_args()

    print("Загрузка данных…")
    items_all = pd.read_parquet(args.items, columns=["id", "name", "attributes", "category"])
    cat_by_id = dict(zip(items_all["id"], items_all["category"]))
    matches = pd.read_parquet(args.matches)
    assert "target" in matches.columns, "Для локальной оценки нужны target-метки"

    if args.cats:
        wanted = {c.strip() for c in args.cats.split(",")}
        keep = matches["id1"].map(cat_by_id).isin(wanted)
        matches = matches[keep].reset_index(drop=True)

    if args.sample and args.sample < len(matches):
        matches = stratified_sample(matches, cat_by_id, args.sample, args.seed).reset_index(drop=True)

    # Ограничить товары теми, что реально участвуют в парах (как в реальном прогоне)
    used_ids = pd.unique(np.concatenate([matches["id1"].to_numpy(), matches["id2"].to_numpy()]))
    items = items_all[items_all["id"].isin(used_ids)].copy()
    items["text"] = build_item_text(items)
    print(f"Пар: {len(matches)} | товаров в прогоне: {len(items)} | категорий: {items['category'].nunique()}")

    pairs = attach_texts(matches, items)

    t0 = time.perf_counter()
    pairs, scores = predict_scores(matches, items, method=args.method, pairs=pairs)
    score_time = time.perf_counter() - t0

    macro, per_cat = macro_pr_auc(pairs["target"].to_numpy(), scores, pairs["category"].to_numpy())

    print("\n=== PR-AUC по категориям ===")
    for cat, v in sorted(per_cat.items(), key=lambda kv: kv[1]):
        print(f"  {cat:<28} {v:.4f}")
    print(f"\n  MACRO PR-AUC = {macro:.4f}  (метод={args.method}, категорий={len(per_cat)})")

    # Тайминг и экстраполяция
    n = len(pairs)
    print(f"\n=== Тайминг (этот Mac) ===")
    print(f"  скоринг {n} пар: {score_time:.1f}с ({1000*score_time/n:.2f} мс/пара)")
    for name, target_n, limit in [
        ("Public", PUBLIC_PAIRS, PUBLIC_LIMIT_S),
        ("Private", PRIVATE_PAIRS, PRIVATE_LIMIT_S),
    ]:
        est = score_time * target_n / n
        flag = "OK" if est < limit else "!! ПРЕВЫШЕНИЕ"
        print(f"  ~{name}: {target_n} пар -> ~{est:.0f}с при лимите {limit}с  [{flag}]")
    print("  (реальная машина: 20 CPU + H100, мощнее — оценка консервативная)")


if __name__ == "__main__":
    main()
