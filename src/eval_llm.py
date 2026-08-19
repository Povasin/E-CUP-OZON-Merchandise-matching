"""Оценка моделей на отложенном LLM-фолде при доле положительных пар, как в тесте.

Ручной holdout нас обманывал. Три точки лидерборда против локальных замеров:

    сборка                     ручной holdout   LLM-фолд   лидерборд
    relaxed + combo                  0.780745   0.671228      0.4651
    lr3e5 + relaxed                  0.784957   0.642733      0.4601

Порядок по ручной разметке обратен порядку на лидерборде, порядок по LLM-фолду совпадает.
Причина в том, как собраны негативы: в тесте они получены ретривалом — тем же процессом,
что породил девять миллионов LLM-пар. В ручной разметке негативы легче, и модель,
настроенная на них, на тесте проигрывает.

Вторая поправка — доля положительных. Average precision отсчитывается от неё: в нашей
разметке положительных 25.7%, в LLM-фолде 21.0%, в закрытом тесте около 11.1%. Поэтому
положительные прореживаются внутри каждой категории до целевой доли, а замер усредняется
по нескольким жеребьёвкам.

Запуск:
    .venv/bin/python -m src.eval_llm --scores relaxed=output/relaxed/ce_relaxed \
        --scores combo=output/combo/ce_combo
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score

CATEGORIES = "output/ood/categories.npy"


def macro_at_rate(scores: np.ndarray, labels: np.ndarray, categories: np.ndarray,
                  rate: float, seeds: int) -> tuple[float, float]:
    """Среднее и разброс macro PR-AUC при заданной доле положительных пар."""
    values = []
    for seed in range(seeds):
        rng = np.random.default_rng(seed)
        per_category = []
        for category in np.unique(categories):
            rows = np.flatnonzero(categories == category)
            positive = rows[labels[rows] == 1]
            negative = rows[labels[rows] == 0]
            if len(positive) < 5 or len(negative) < 5:
                continue
            keep = min(len(positive), max(5, int(round(rate / (1 - rate) * len(negative)))))
            chosen = np.concatenate([rng.choice(positive, keep, replace=False), negative])
            per_category.append(average_precision_score(labels[chosen], scores[chosen]))
        values.append(float(np.mean(per_category)))
    return float(np.mean(values)), float(np.std(values))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--scores", action="append", required=True,
                    help="имя=каталог с llm_ood_scores.npy и llm_ood_pairs.parquet")
    ap.add_argument("--rate", type=float, default=0.111, help="доля положительных в тесте")
    ap.add_argument("--seeds", type=int, default=15)
    args = ap.parse_args()

    categories = np.load(CATEGORIES, allow_pickle=True).astype(str)

    reference, labels, loaded = None, None, []
    for spec in args.scores:
        name, directory = spec.split("=", 1)
        pairs = pd.read_parquet(Path(directory) / "llm_ood_pairs.parquet")
        identifiers = pairs["id1"].to_numpy()
        if reference is None:
            reference, labels = identifiers, pairs["label"].to_numpy(np.int8)
            if len(labels) != len(categories):
                raise SystemExit(
                    f"Категорий {len(categories)}, а пар {len(labels)} — пересоберите {CATEGORIES}"
                )
        elif not np.array_equal(identifiers, reference):
            raise SystemExit(f"{name}: другой набор LLM-пар, сравнивать нельзя")
        scores = np.load(Path(directory) / "llm_ood_scores.npy")
        if not np.isfinite(scores).all():
            raise SystemExit(f"{name}: в скорах есть NaN или бесконечности")
        loaded.append((name, scores))

    print(f"LLM-фолд: {len(labels):,} пар, доля положительных {labels.mean() * 100:.1f}%")
    print(f"замер при доле {args.rate * 100:.1f}%, усреднение по {args.seeds} жеребьёвкам\n")
    print(f"{'модель':<22}{'как есть':>12}{'при тестовой доле':>20}{'разброс':>10}")
    results = []
    for name, scores in loaded:
        as_is, _ = macro_at_rate(scores, labels, categories, labels.mean(), 3)
        value, spread = macro_at_rate(scores, labels, categories, args.rate, args.seeds)
        results.append((value, name, as_is, spread))
    for value, name, as_is, spread in sorted(results, reverse=True):
        print(f"{name:<22}{as_is:>12.6f}{value:>20.6f}{spread:>10.6f}")


if __name__ == "__main__":
    main()
