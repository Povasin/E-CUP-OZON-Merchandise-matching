"""v2: скоринг пар косинусом эмбеддингов предобученной модели.

Данные русскоязычные, поэтому берём multilingual-модель. По умолчанию
`intfloat/multilingual-e5-small` (384-dim, ~118M) — хороший баланс качество/скорость.
Модель префиксная (E5): для симметричной близости обе стороны получают "query: ".

Офлайн-запуск: EMBED_MODEL можно указать как локальный путь к весам, вшитым в сабмит/
образ (интернета в докере нет). torch/sentence-transformers импортируются лениво — v1
(tfidf) от них не зависит.
"""
from __future__ import annotations

import os

import numpy as np
import pandas as pd

DEFAULT_MODEL = os.environ.get("EMBED_MODEL", "intfloat/multilingual-e5-small")
# Префикс модели E5. Для не-E5 моделей (paraphrase-*) выставить EMBED_PREFIX="".
DEFAULT_PREFIX = os.environ.get("EMBED_PREFIX", "query: ")


def _pick_device() -> str:
    dev = os.environ.get("EMBED_DEVICE")
    if dev:
        return dev
    import torch

    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def embed_scores(
    pairs: pd.DataFrame,
    model_name: str = DEFAULT_MODEL,
    prefix: str = DEFAULT_PREFIX,
    batch_size: int = 256,
    max_seq_length: int | None = 192,
) -> np.ndarray:
    """Построчный косинус эмбеддингов для (text1, text2).

    Уникальные тексты эмбеддятся один раз (дедупликация), затем разбираются по парам.
    Пары с отсутствующим текстом получают скор 0 (требование ТЗ — предсказание на все).
    """
    from sentence_transformers import SentenceTransformer

    text1 = pairs["text1"].to_numpy()
    text2 = pairs["text2"].to_numpy()

    # Уникальные непустые тексты -> одна прогонка энкодера
    uniq = pd.unique(np.concatenate([text1, text2]))
    uniq = uniq[uniq != ""]
    text_to_row = {t: i for i, t in enumerate(uniq)}

    device = _pick_device()
    model = SentenceTransformer(model_name, device=device)
    if max_seq_length:
        model.max_seq_length = max_seq_length

    inputs = [prefix + t for t in uniq] if prefix else list(uniq)
    emb = model.encode(
        inputs,
        batch_size=batch_size,
        normalize_embeddings=True,  # косинус == скалярное произведение
        convert_to_numpy=True,
        show_progress_bar=True,
    ).astype(np.float32)

    # Индексы строк эмбеддингов; -1 = пустой текст
    idx1 = np.fromiter((text_to_row.get(t, -1) for t in text1), dtype=np.int64, count=len(text1))
    idx2 = np.fromiter((text_to_row.get(t, -1) for t in text2), dtype=np.int64, count=len(text2))

    valid = (idx1 >= 0) & (idx2 >= 0)
    scores = np.zeros(len(pairs), dtype=np.float32)
    scores[valid] = np.sum(emb[idx1[valid]] * emb[idx2[valid]], axis=1)
    return scores
