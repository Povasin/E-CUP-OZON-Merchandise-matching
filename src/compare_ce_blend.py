"""Честное сравнение структурной модели и кросс-энкодера плюс их смесь.

Обе модели должны быть измерены на ОДНОМ наборе, иначе цифры несопоставимы. Здесь
структурная модель обучается на product-disjoint train-части ручной разметки и
оценивается на том же holdout, на котором посчитан кросс-энкодер.

Смешиваются ранги внутри категории: шкалы несопоставимы (вероятность против логита), а
метрика считается внутри категории. Вес подбирается отдельно на категорию и проверяется
двухфолдовой схемой — подбор на одной половине holdout, замер на другой, иначе выбор
20 весов по 11 вариантам подгоняется под шум.

Запуск:
    .venv/bin/python -m src.compare_ce_blend
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import average_precision_score

from src.hybrid import product_disjoint_pair_masks
from src.metrics import macro_pr_auc

WEIGHT_GRID = np.round(np.arange(0.0, 1.01, 0.1), 2)


def category_ranks(scores: np.ndarray, categories: np.ndarray) -> np.ndarray:
    frame = pd.DataFrame({"score": scores, "category": categories})
    return frame.groupby("category", sort=False)["score"].rank(pct=True).to_numpy(dtype=np.float32)


def tune_weights(boost_rank, ce_rank, target, categories, rows) -> dict[str, float]:
    weights: dict[str, float] = {}
    for category in np.unique(categories[rows]):
        mask = rows[categories[rows] == category]
        y = target[mask]
        if len(np.unique(y)) < 2:
            weights[str(category)] = 0.0
            continue
        scored = [
            (average_precision_score(y, w * ce_rank[mask] + (1.0 - w) * boost_rank[mask]), w)
            for w in WEIGHT_GRID
        ]
        weights[str(category)] = float(max(scored)[1])
    return weights


def apply_weights(boost_rank, ce_rank, categories, weights, default) -> np.ndarray:
    blend = np.empty(len(boost_rank), dtype=np.float32)
    for category in np.unique(categories):
        mask = categories == category
        w = weights.get(str(category), default)
        blend[mask] = w * ce_rank[mask] + (1.0 - w) * boost_rank[mask]
    return blend


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--features", default="output/human_features_v13.npy")
    ap.add_argument("--items", default="assets/items_human.parquet")
    ap.add_argument("--matches", default="assets/matches.parquet")
    ap.add_argument("--ce-dir", default="output/kernel_log/ce_large")
    ap.add_argument("--holdout-fold", type=int, default=0)
    ap.add_argument("--n-folds", type=int, default=3)
    ap.add_argument("--max-iter", type=int, default=250)
    ap.add_argument("--seed", type=int, default=2026)
    ap.add_argument("--out", default="output/blend_weights_v13.npz")
    ap.add_argument("--boost-cache", default="output/boost_holdout_v13.npy")
    args = ap.parse_args()

    features = np.load(args.features, mmap_mode="r")
    items = pd.read_parquet(args.items, columns=["id", "category"])
    matches = pd.read_parquet(args.matches, columns=["id1", "id2", "target"])
    target = matches["target"].to_numpy(dtype=np.int8)
    categories = matches["id1"].map(
        dict(zip(items["id"], items["category"].astype(str)))
    ).astype(str).to_numpy()

    train_mask, valid_mask = product_disjoint_pair_masks(
        matches["id1"].to_numpy(), matches["id2"].to_numpy(),
        args.holdout_fold, args.n_folds,
    )
    train_idx = np.flatnonzero(train_mask)
    valid_idx = np.flatnonzero(valid_mask)

    ce_dir = Path(args.ce_dir)
    ce_all = np.load(ce_dir / "human_scores.npy")
    ce_valid_idx = np.load(ce_dir / "human_valid_idx.npy")
    if not np.array_equal(valid_idx, ce_valid_idx):
        raise SystemExit("Holdout кросс-энкодера не совпадает с локальным — сравнивать нельзя")
    print(f"train={len(train_idx):,}, holdout={len(valid_idx):,}, "
          f"положительных в holdout {target[valid_idx].mean():.4f}\n", flush=True)

    valid_categories = categories[valid_idx]
    # Структурная модель между прогонами кросс-энкодера не меняется, поэтому её скоры на
    # holdout считаются один раз. Дальше сравнение любой новой модели стоит секунды.
    cache = Path(args.boost_cache)
    if cache.exists():
        boost = np.load(cache)
        if len(boost) != len(valid_idx):
            raise SystemExit(f"Кэш скоров не соответствует holdout: {len(boost)}")
        print(f"Скоры структурной модели из кэша: {cache}", flush=True)
        return report(boost, ce_all[valid_idx], target[valid_idx], valid_categories, args)

    print("Обучение структурной модели по категориям...", flush=True)
    boost = np.zeros(len(valid_idx), dtype=np.float32)
    train_categories = categories[train_idx]
    for category in sorted(np.unique(valid_categories)):
        rows = train_idx[train_categories == category]
        model = HistGradientBoostingClassifier(
            learning_rate=0.05, max_iter=args.max_iter, max_leaf_nodes=31,
            min_samples_leaf=40, l2_regularization=2.0, early_stopping=False,
            random_state=args.seed,
        )
        model.fit(np.asarray(features[rows]), target[rows])
        selected = valid_categories == category
        boost[selected] = model.predict_proba(
            np.asarray(features[valid_idx[selected]])
        )[:, 1]
        print(f"  {category:<28} обучено на {len(rows):>7,}", flush=True)

    np.save(args.boost_cache, boost)
    print(f"Скоры структурной модели сохранены: {args.boost_cache}", flush=True)
    return report(boost, ce_all[valid_idx], target[valid_idx], valid_categories, args)


def report(boost, ce, y, valid_categories, args) -> None:
    boost_macro, boost_per = macro_pr_auc(y, boost, valid_categories)
    ce_macro, ce_per = macro_pr_auc(y, ce, valid_categories)
    print(f"\nСтруктурная модель  {boost_macro:.6f}")
    print(f"Кросс-энкодер       {ce_macro:.6f}")

    boost_rank = category_ranks(boost, valid_categories)
    ce_rank = category_ranks(ce, valid_categories)
    print(f"Корреляция рангов   {np.corrcoef(boost_rank, ce_rank)[0, 1]:.4f} "
          f"(чем ниже, тем комплементарнее)")

    print("\nЕдиный вес кросс-энкодера:")
    best = (boost_macro, 0.0)
    for w in WEIGHT_GRID:
        macro, _ = macro_pr_auc(y, w * ce_rank + (1.0 - w) * boost_rank, valid_categories)
        mark = ""
        if macro > best[0]:
            best = (macro, float(w))
            mark = "  <-"
        print(f"  {w:.1f}  {macro:.6f}{mark}")

    rng = np.random.default_rng(args.seed)
    fold = rng.permutation(len(y)) % 2
    per_cat, fixed = [], []
    for held in (0, 1):
        tune_rows = np.flatnonzero(fold != held)
        test_rows = np.flatnonzero(fold == held)
        weights = tune_weights(boost_rank, ce_rank, y, valid_categories, tune_rows)
        blend = apply_weights(boost_rank, ce_rank, valid_categories, weights, best[1])
        per_cat.append(macro_pr_auc(y[test_rows], blend[test_rows], valid_categories[test_rows])[0])
        flat = best[1] * ce_rank + (1.0 - best[1]) * boost_rank
        fixed.append(macro_pr_auc(y[test_rows], flat[test_rows], valid_categories[test_rows])[0])
    honest_cat, honest_fixed = float(np.mean(per_cat)), float(np.mean(fixed))
    print(f"\nЧестная проверка (подбор и замер на разных половинах holdout):")
    print(f"  единый вес {best[1]:.1f}   {honest_fixed:.6f}")
    print(f"  вес на категорию      {honest_cat:.6f} ({honest_cat - honest_fixed:+.6f})")
    gain = honest_cat - boost_macro
    print(f"\nПрирост смеси к структурной модели: {gain:+.6f} ({gain / boost_macro * 100:+.2f}%)")

    final = tune_weights(boost_rank, ce_rank, y, valid_categories, np.arange(len(y)))
    np.savez(args.out,
             categories=np.asarray(sorted(final)),
             weights=np.asarray([final[c] for c in sorted(final)], dtype=np.float32),
             global_weight=np.float32(best[1]))
    print(f"\nВеса сохранены: {args.out}")
    print("\nПо категориям (структурная / кросс-энкодер / вес CE):")
    for category in sorted(boost_per, key=lambda c: boost_per[c]):
        print(f"  {category:<28} {boost_per[category]:.4f}  {ce_per[category]:.4f}  "
              f"{final[category]:.1f}")

if __name__ == "__main__":
    main()
