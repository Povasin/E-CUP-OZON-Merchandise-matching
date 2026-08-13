"""Инференс-пайплайн: данные -> скоринг -> submit.csv.

Гарантирует предсказание для КАЖДОЙ входной пары в исходном порядке.
"""
from __future__ import annotations

import os
import time

import pandas as pd

from src.data import attach_texts, load_items, load_matches
from src.scoring import score_pairs


DEFAULT_MODEL_PATH = os.environ.get("PAIR_MODEL_PATH", "models/pair_logreg.npz")
DEFAULT_BOOST_MODEL_PATH = os.environ.get("PAIR_BOOST_MODEL_PATH", "models/pair_boost_hybrid.npz")


def predict_scores(
    matches: pd.DataFrame,
    items: pd.DataFrame,
    method: str,
    pairs: pd.DataFrame | None = None,
    **scorer_kwargs,
) -> tuple[pd.DataFrame, object]:
    """Score input pairs while preserving their order."""
    if pairs is None:
        pairs = attach_texts(matches, items)
    if method in {"supervised", "boosted"}:
        from src.features import extract_model_features
        from src.model import BoostedPairModel, PairModel

        features = extract_model_features(matches, items)
        if method == "boosted":
            model_path = scorer_kwargs.pop("model_path", DEFAULT_BOOST_MODEL_PATH)
            scores = BoostedPairModel(model_path).predict(features, pairs["category"].to_numpy())
        else:
            model_path = scorer_kwargs.pop("model_path", DEFAULT_MODEL_PATH)
            scores = PairModel(model_path).predict(features, pairs["category"].to_numpy())
        missing = (pairs["text1"] == "").to_numpy() | (pairs["text2"] == "").to_numpy()
        scores[missing] = 0.0
    else:
        scores = score_pairs(pairs, method=method, **scorer_kwargs)
    return pairs, scores


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
    pairs, scores = predict_scores(matches, items, method=method, pairs=pairs, **scorer_kwargs)

    print(f"[4/4] Сохранение результата: {output_path}")
    result = pd.DataFrame(
        {"id1": pairs["id1"].to_numpy(), "id2": pairs["id2"].to_numpy(), "predict": scores}
    )
    assert len(result) == len(matches), "Число предсказаний должно совпадать с числом пар"
    result.to_csv(output_path, index=False)

    print(f"Готово за {time.perf_counter() - t0:.1f}с")
    return result
