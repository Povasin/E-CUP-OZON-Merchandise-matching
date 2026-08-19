"""Смесь произвольного числа моделей на общем product-disjoint holdout.

Ранги внутри категории, веса на симплексе. Веса подбираются дважды: общие для всех
категорий и отдельные для каждой. Отдельные берутся только там, где они подтвердились на
независимой половине holdout — иначе на выборке в пару тысяч пар подбор шести весов
превращается в подгонку под шум.

Все входные скоры обязаны быть посчитаны на одном и том же holdout и в одном порядке пар;
это проверяется, а не предполагается.

Запуск:
    .venv/bin/python -m src.blend_many \
        --model структурная=output/boost_holdout_v13.npy \
        --model relaxed=output/relaxed/ce_relaxed/human_scores.npy
"""
from __future__ import annotations

import argparse
from itertools import product as iter_product
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score

from src.hybrid import product_disjoint_pair_masks
from src.metrics import macro_pr_auc

STEP = 10


def category_ranks(scores: np.ndarray, categories: np.ndarray) -> np.ndarray:
    frame = pd.DataFrame({"score": scores, "category": categories})
    return frame.groupby("category", sort=False)["score"].rank(pct=True).to_numpy(dtype=np.float32)


def simplex_weights(n: int) -> list[np.ndarray]:
    """Все наборы неотрицательных весов с шагом 0.1, дающие в сумме единицу."""
    out = []
    for combo in iter_product(range(STEP + 1), repeat=n - 1):
        rest = STEP - sum(combo)
        if rest >= 0:
            out.append(np.asarray([*combo, rest], dtype=np.float32) / STEP)
    return out


def best_global(ranks: np.ndarray, y: np.ndarray, categories: np.ndarray,
                rows: np.ndarray, grid: list[np.ndarray]) -> tuple[float, np.ndarray]:
    best = (-1.0, grid[0])
    for weights in grid:
        macro, _ = macro_pr_auc(y[rows], weights @ ranks[:, rows], categories[rows])
        if macro > best[0]:
            best = (macro, weights)
    return best


def best_per_category(ranks: np.ndarray, y: np.ndarray, categories: np.ndarray,
                      rows: np.ndarray, grid: list[np.ndarray],
                      fallback: np.ndarray) -> dict[str, np.ndarray]:
    """Свои веса на категорию; где данных не хватает, остаются общие."""
    result: dict[str, np.ndarray] = {}
    for category in np.unique(categories):
        mask = rows[categories[rows] == category]
        if len(mask) < 200 or len(np.unique(y[mask])) < 2:
            result[str(category)] = fallback
            continue
        best = (-1.0, fallback)
        for weights in grid:
            score = average_precision_score(y[mask], weights @ ranks[:, mask])
            if score > best[0]:
                best = (score, weights)
        result[str(category)] = best[1]
    return result


def apply_per_category(ranks: np.ndarray, categories: np.ndarray,
                       weights: dict[str, np.ndarray], fallback: np.ndarray) -> np.ndarray:
    blended = np.empty(ranks.shape[1], dtype=np.float32)
    for category in np.unique(categories):
        mask = categories == category
        blended[mask] = weights.get(str(category), fallback) @ ranks[:, mask]
    return blended


def load_scores(spec: str, matches: pd.DataFrame, valid_idx: np.ndarray) -> tuple[str, np.ndarray]:
    """Скоры модели, приведённые к holdout, с проверкой совпадения разбиения."""
    name, path = spec.split("=", 1)
    scores = np.load(path)
    sibling = Path(path).with_name("human_valid_idx.npy")
    if sibling.exists() and not np.array_equal(np.load(sibling), valid_idx):
        raise SystemExit(f"{name}: holdout модели не совпадает с локальным — сравнивать нельзя")
    if len(scores) == len(matches):
        scores = scores[valid_idx]
    elif len(scores) != len(valid_idx):
        raise SystemExit(f"{name}: длина {len(scores)} не соответствует ни парам, ни holdout")
    if not np.isfinite(scores).all():
        raise SystemExit(f"{name}: в скорах есть NaN или бесконечности")
    return name, scores


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", action="append", required=True, help="имя=путь до .npy")
    ap.add_argument("--items", default="assets/items_human.parquet")
    ap.add_argument("--matches", default="assets/matches.parquet")
    ap.add_argument("--holdout-fold", type=int, default=0)
    ap.add_argument("--n-folds", type=int, default=3)
    ap.add_argument("--out", default=None, help="куда сохранить веса для production")
    ap.add_argument("--ce-dirs", default=None,
                    help="каталоги моделей через запятую в том же порядке, что --model "
                         "начиная со второй (первая — структурная)")
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
        name, scores = load_scores(spec, matches, valid_idx)
        macro, _ = macro_pr_auc(y, scores, valid_categories)
        print(f"  {name:<12} {macro:.6f}")
        names.append(name)
        ranks.append(category_ranks(scores, valid_categories))
    ranks = np.vstack(ranks)
    grid = simplex_weights(len(names))

    rng = np.random.default_rng(args.seed)
    fold = rng.permutation(len(y)) % 2
    honest_global, honest_category = [], []
    for held in (0, 1):
        tune = np.flatnonzero(fold != held)
        test = np.flatnonzero(fold == held)
        _, weights = best_global(ranks, y, valid_categories, tune, grid)
        honest_global.append(
            macro_pr_auc(y[test], weights @ ranks[:, test], valid_categories[test])[0]
        )
        per_cat = best_per_category(ranks, y, valid_categories, tune, grid, weights)
        blended = apply_per_category(ranks, valid_categories, per_cat, weights)
        honest_category.append(
            macro_pr_auc(y[test], blended[test], valid_categories[test])[0]
        )
    g, c = float(np.mean(honest_global)), float(np.mean(honest_category))
    print(f"\nЧестная проверка (подбор и замер на разных половинах holdout):")
    print(f"  общие веса        {g:.6f}")
    print(f"  веса на категорию {c:.6f} ({c - g:+.6f})")
    use_per_category = c > g

    rows = np.arange(len(y))
    macro, weights = best_global(ranks, y, valid_categories, rows, grid)
    print(f"\nОбщие веса ({macro:.6f}):")
    for name, weight in zip(names, weights):
        print(f"  {name:<12} {weight:.1f}")

    per_cat = best_per_category(ranks, y, valid_categories, rows, grid, weights)
    if use_per_category:
        print("\nВеса по категориям (взяты, так как подтвердились на независимой половине):")
        for category in sorted(per_cat):
            line = "  ".join(f"{n}={w:.1f}" for n, w in zip(names, per_cat[category]) if w > 0)
            print(f"  {category:<24} {line}")
    else:
        print("\nВеса по категориям НЕ подтвердились на независимой половине — берём общие.")

    if args.out:
        chosen = per_cat if use_per_category else {c: weights for c in np.unique(valid_categories)}
        ordered = sorted(chosen)
        ce_dirs = args.ce_dirs.split(",") if args.ce_dirs else []
        if ce_dirs and len(ce_dirs) != len(names) - 1:
            raise SystemExit(f"Каталогов {len(ce_dirs)}, а кросс-энкодеров {len(names) - 1}")
        np.savez(args.out,
                 names=np.asarray(names),
                 ce_dirs=np.asarray(ce_dirs),
                 weights=weights.astype(np.float32),
                 categories=np.asarray(ordered),
                 category_weights=np.vstack([chosen[c] for c in ordered]).astype(np.float32))
        print(f"\nВеса сохранены: {args.out}")


if __name__ == "__main__":
    main()
