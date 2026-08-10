"""Инференс-пайплайн: данные -> скоринг -> submit.csv.

Гарантирует предсказание для КАЖДОЙ входной пары в исходном порядке.
"""
from __future__ import annotations

import time

import pandas as pd

from src.data import attach_texts, load_items, load_matches
from src.scoring import score_pairs


def predict_pipeline(
    items_path: str,
    matches_path: str,
    output_path: str,
    method: str = "tfidf",
    **scorer_kwargs,
) -> pd.DataFrame:
    t0 = time.perf_counter()
    print(f"[1/4] Загрузка товаров: {items_path}")
    items = load_items(items_path)

    print(f"[2/4] Загрузка пар: {matches_path}")
    matches = load_matches(matches_path)
    pairs = attach_texts(matches, items)
    n_missing = int((pairs["text1"] == "").sum() + (pairs["text2"] == "").sum())
    print(f"      пар: {len(pairs)} | сторон без текста: {n_missing}")

    print(f"[3/4] Скоринг методом '{method}'")
    scores = score_pairs(pairs, method=method, **scorer_kwargs)

    print(f"[4/4] Сохранение результата: {output_path}")
    result = pd.DataFrame(
        {"id1": pairs["id1"].to_numpy(), "id2": pairs["id2"].to_numpy(), "predict": scores}
    )
    assert len(result) == len(matches), "Число предсказаний должно совпадать с числом пар"
    result.to_csv(output_path, index=False)

    print(f"Готово за {time.perf_counter() - t0:.1f}с")
    return result
