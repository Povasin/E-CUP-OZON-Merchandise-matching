"""Инференс-пайплайн: данные -> скоринг -> submit.csv.

Гарантирует предсказание для КАЖДОЙ входной пары в исходном порядке.
"""
from __future__ import annotations

import os
import time
from pathlib import Path

import numpy as np
import pandas as pd

from src.data import attach_texts, load_items, load_matches
from src.scoring import score_pairs


DEFAULT_MODEL_PATH = os.environ.get("PAIR_MODEL_PATH", "models/pair_logreg.npz")
DEFAULT_BOOST_MODEL_PATH = os.environ.get("PAIR_BOOST_MODEL_PATH", "models/pair_boost_hybrid.npz")
DEFAULT_BOOST_AUX_MODEL_PATH = os.environ.get(
    "PAIR_BOOST_AUX_MODEL_PATH", "models/pair_boost_hybrid_aux.npz"
)
DEFAULT_BOOST_AUX_WEIGHT = float(os.environ.get("PAIR_BOOST_AUX_WEIGHT", "0.20"))
DEFAULT_CE_DIR = os.environ.get("PAIR_CE_DIR", "models/cross_encoder")
DEFAULT_BLEND_WEIGHTS = os.environ.get("PAIR_BLEND_WEIGHTS", "models/blend_weights.npz")
DEFAULT_CE_BATCH = int(os.environ.get("PAIR_CE_BATCH", "512"))


def _category_ranks(scores, categories):
    frame = pd.DataFrame({"score": scores, "category": categories})
    return frame.groupby("category", sort=False)["score"].rank(pct=True).to_numpy(dtype=np.float32)


def _cross_encoder_scores(matches: pd.DataFrame, items: pd.DataFrame) -> np.ndarray:
    """Логиты дообученного кросс-энкодера в порядке входных пар."""
    import json

    import torch
    from transformers import AutoModelForSequenceClassification, AutoTokenizer

    from src.cross_encoder import build_product_texts

    config = json.loads((Path(DEFAULT_CE_DIR) / "inference_config.json").read_text())
    max_length, mode = int(config["max_length"]), config["mode"]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = AutoTokenizer.from_pretrained(DEFAULT_CE_DIR, local_files_only=True)
    dtype = torch.float16 if device.type == "cuda" else torch.float32
    model = AutoModelForSequenceClassification.from_pretrained(
        DEFAULT_CE_DIR, local_files_only=True, dtype=dtype
    ).to(device).eval()

    texts = build_product_texts(items, mode)
    left = matches["id1"].map(texts).fillna("").astype(str).to_numpy()
    right = matches["id2"].map(texts).fillna("").astype(str).to_numpy()
    del texts

    scores = np.empty(len(matches), dtype=np.float32)
    # Бакетинг по длине: батч дополняется до самой длинной строки в нём, поэтому
    # соседство похожих длин заметно сокращает вычисления на паддинге.
    order = np.argsort(np.fromiter(
        (len(a) + len(b) for a, b in zip(left, right)), dtype=np.int32, count=len(left)
    ))
    with torch.inference_mode():
        for start in range(0, len(order), DEFAULT_CE_BATCH):
            rows = order[start:start + DEFAULT_CE_BATCH]
            encoded = tokenizer(
                left[rows].tolist(), right[rows].tolist(),
                padding=True, truncation=True, max_length=max_length,
                pad_to_multiple_of=8 if device.type == "cuda" else None,
                return_tensors="pt",
            ).to(device)
            scores[rows] = model(**encoded).logits.squeeze(-1).float().cpu().numpy()
    return scores


def _blend_scores(matches: pd.DataFrame, items: pd.DataFrame, pairs: pd.DataFrame) -> np.ndarray:
    """Смесь структурного бустинга и кросс-энкодера рангами внутри категории.

    Модели ошибаются по-разному (корреляция рангов 0.58), поэтому смесь сильнее любой из
    них. Вес свой на категорию: на обуви и одежде кросс-энкодер слаб и получает 0.1, на
    продуктах питания он сильнее структурной модели и получает 0.8.
    """
    from src.features import extract_model_features
    from src.model import BoostedPairModel

    categories = pairs["category"].to_numpy().astype(str)
    features = extract_model_features(matches, items)
    primary = BoostedPairModel(DEFAULT_BOOST_MODEL_PATH).predict_probability(features, categories)
    auxiliary = BoostedPairModel(DEFAULT_BOOST_AUX_MODEL_PATH).predict_probability(
        features, categories
    )
    boost = (1.0 - DEFAULT_BOOST_AUX_WEIGHT) * primary + DEFAULT_BOOST_AUX_WEIGHT * auxiliary
    del features, primary, auxiliary

    ce = _cross_encoder_scores(matches, items)

    artifact = np.load(DEFAULT_BLEND_WEIGHTS, allow_pickle=False)
    weight_by_category = dict(zip(artifact["categories"].astype(str), artifact["weights"]))
    fallback = float(artifact["global_weight"])

    boost_rank = _category_ranks(boost, categories)
    ce_rank = _category_ranks(ce, categories)
    scores = np.empty(len(matches), dtype=np.float32)
    for category in np.unique(categories):
        mask = categories == category
        weight = float(weight_by_category.get(category, fallback))
        scores[mask] = weight * ce_rank[mask] + (1.0 - weight) * boost_rank[mask]
    return scores


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
    if method == "blend":
        scores = _blend_scores(matches, items, pairs)
        # Ранги лежат в (0,1], поэтому 0.0 — корректное «хуже всех» для пар без текста.
        missing = (pairs["text1"] == "").to_numpy() | (pairs["text2"] == "").to_numpy()
        scores[missing] = 0.0
    elif method in {"supervised", "boosted"}:
        from src.features import extract_model_features
        from src.model import BoostedPairModel, PairModel

        features = extract_model_features(matches, items)
        if method == "boosted":
            model_path = scorer_kwargs.pop("model_path", DEFAULT_BOOST_MODEL_PATH)
            aux_model_path = scorer_kwargs.pop("aux_model_path", DEFAULT_BOOST_AUX_MODEL_PATH)
            aux_weight = float(scorer_kwargs.pop("aux_weight", DEFAULT_BOOST_AUX_WEIGHT))
            if not 0.0 <= aux_weight <= 1.0:
                raise ValueError("aux_weight must be between 0 and 1")
            categories = pairs["category"].to_numpy()
            model = BoostedPairModel(model_path)
            if aux_model_path and aux_weight:
                primary = model.predict_probability(features, categories)
                auxiliary = BoostedPairModel(aux_model_path).predict_probability(
                    features, categories
                )
                scores = (1.0 - aux_weight) * primary + aux_weight * auxiliary
            else:
                scores = model.predict(features, categories)
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
