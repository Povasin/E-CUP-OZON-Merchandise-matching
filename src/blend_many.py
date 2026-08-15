"""Смесь произвольного числа моделей на общем product-disjoint holdout.

Две модели смешивать умеет `compare_ce_blend`, но участников может быть больше: разные
основы кросс-энкодера ошибаются по-разному, и англоязычная, при всей своей слабости,
отличается от русских сильнее, чем те друг от друга.

Ранги внутри категории, веса на симплексе с шагом 0.1. Проверка честная: веса подбираются
на одной половине holdout, метрика считается на другой.

Запуск:
    .venv/bin/python -m src.blend_many \
        --model структурная=output/boost_holdout_v13.npy \
        --model англ=output/kernel_log/ce_large/human_scores.npy \
        --model рус=output/ru/ce_large/human_scores.npy
"""
from __future__ import annotations

import argparse
from itertools import product as iter_product

import numpy as np
import pandas as pd

from src.hybrid import product_disjoint_pair_masks
from src.metrics import macro_pr_auc

STEP = 10  # сетка весов: доли от 0.0 до 1.0 с шагом 0.1


def category_ranks(scores: np.ndarray, categories: np.ndarray) -> np.ndarray:
    frame = pd.DataFrame({"score": scores, "category": categories})
    return frame.groupby("category", sort=False)["score"].rank(pct=True).to_numpy(dtype=np.float32)


def simplex_weights(n: int):
    """Все наборы неотрицательных весов с шагом 0.1, дающие в сумме единицу."""
    for combo in iter_product(range(STEP + 1), repeat=n - 1):
        rest = STEP - sum(combo)
        if rest >= 0:
            yield np.asarray([*combo, rest], dtype=np.float32) / STEP


def best_weights(ranks: np.ndarray, y: np.ndarray, categories: np.ndarray, rows: np.ndarray):
    best = (-1.0, None)
    for weights in simplex_weights(ranks.shape[0]):
        blended = weights @ ranks[:, rows]
        macro, _ = macro_pr_auc(y[rows], blended, categories[rows])
        if macro > best[0]:
            best = (macro, weights)
    return best


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", action="append", required=True,
                    help="имя=путь до .npy со скорами (в порядке всех 365k пар или holdout)")
    ap.add_argument("--items", default="assets/items_human.parquet")
    ap.add_argument("--matches", default="assets/matches.parquet")
    ap.add_argument("--holdout-fold", type=int, default=0)
    ap.add_argument("--n-folds", type=int, default=3)
    ap.add_argument("--seed", type=int, default=2026)
    args = ap.parse_args()

    items = pd.read_parquet(args.items, columns=["id", "category"])
    matches = pd.read_parquet(args.matches, columns=["id1", "id2", "target"])
    target = matches["target"].to_numpy(dtype=np.int8)
    categories = matches["id1"].map(
        dict(zip(items["id"], items["category"].astype(str)))
    ).astype(str).to_numpy()
    _, valid_mask = product_disjoint_pair_masks(
        matches["id1"].to_numpy(), matches["id2"].to_numpy(), args.holdout_fold, args.n_folds
    )
    valid_idx = np.flatnonzero(valid_mask)
    y = target[valid_idx]
    valid_categories = categories[valid_idx]

    names, ranks = [], []
    for spec in args.model:
        name, path = spec.split("=", 1)
        scores = np.load(path)
        # Скоры бывают двух видов: по всем парам или уже только по holdout.
        if len(scores) == len(matches):
            scores = scores[valid_idx]
        elif len(scores) != len(valid_idx):
            raise SystemExit(f"{name}: длина {len(scores)} не соответствует ни парам, ни holdout")
        macro, _ = macro_pr_auc(y, scores, valid_categories)
        print(f"  {name:<14} {macro:.6f}")
        names.append(name)
        ranks.append(category_ranks(scores, valid_categories))
    ranks = np.vstack(ranks)

    print("\nКорреляция рангов между моделями:")
    for i in range(len(names)):
        for j in range(i + 1, len(names)):
            print(f"  {names[i]:<14} / {names[j]:<14} {np.corrcoef(ranks[i], ranks[j])[0, 1]:.4f}")

    rng = np.random.default_rng(args.seed)
    fold = rng.permutation(len(y)) % 2
    honest = []
    for held in (0, 1):
        tune = np.flatnonzero(fold != held)
        test = np.flatnonzero(fold == held)
        _, weights = best_weights(ranks, y, valid_categories, tune)
        macro, _ = macro_pr_auc(y[test], weights @ ranks[:, test], valid_categories[test])
        honest.append(macro)
    print(f"\nЧестная проверка (подбор и замер на разных половинах): {np.mean(honest):.6f}")

    macro, weights = best_weights(ranks, y, valid_categories, np.arange(len(y)))
    print(f"Лучшие веса на всём holdout ({macro:.6f}):")
    for name, weight in zip(names, weights):
        print(f"  {name:<14} {weight:.1f}")


if __name__ == "__main__":
    main()
