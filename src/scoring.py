"""Прямые (без обучения) методы скоринга пар.

v1: TF-IDF косинусная близость. Векторизатор обучается unsupervised прямо на
текстах товаров текущего прогона (fit на тесте допустим — меток не использует),
поэтому весов заранее не требуется и решение работает офлайн.
"""
from __future__ import annotations

from typing import Iterable

import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.feature_extraction.text import TfidfVectorizer


def _rowwise_cosine(a: sparse.csr_matrix, b: sparse.csr_matrix) -> np.ndarray:
    """Построчный косинус между двумя наборами L2-нормированных строк.

    TfidfVectorizer по умолчанию L2-нормирует строки, поэтому косинус == скалярное
    произведение соответствующих строк.
    """
    prod = a.multiply(b).sum(axis=1)
    return np.asarray(prod).ravel()


def _tfidf_cosine(
    text1: np.ndarray,
    text2: np.ndarray,
    fit_texts: Iterable[str],
    analyzer: str,
    ngram_range: tuple[int, int],
    min_df: int,
    max_features: int | None,
) -> np.ndarray:
    """Косинус одного TF-IDF пространства (слово или символ)."""
    vectorizer = TfidfVectorizer(
        analyzer=analyzer,
        ngram_range=ngram_range,
        min_df=min_df,
        max_features=max_features,
        sublinear_tf=True,
        dtype=np.float32,
    )
    vectorizer.fit(fit_texts)
    return _rowwise_cosine(vectorizer.transform(text1), vectorizer.transform(text2))


def tfidf_scores(
    pairs: pd.DataFrame,
    fit_texts: Iterable[str] | None = None,
    ngram_range: tuple[int, int] = (1, 2),
    min_df: int = 2,
    max_features: int | None = 400_000,
) -> np.ndarray:
    """Косинусная близость словарного TF-IDF (слова, 1-2-граммы).

    fit_texts — корпус для обучения словаря; по умолчанию это уникальные тексты
    обеих сторон пар (эквивалент товаров текущего прогона).
    """
    text1 = pairs["text1"].to_numpy()
    text2 = pairs["text2"].to_numpy()
    if fit_texts is None:
        fit_texts = pd.unique(np.concatenate([text1, text2]))
    return _tfidf_cosine(text1, text2, fit_texts, "word", ngram_range, min_df, max_features)


def tfidf_wc_scores(
    pairs: pd.DataFrame,
    fit_texts: Iterable[str] | None = None,
    word_weight: float = 0.5,
    word_ngram: tuple[int, int] = (1, 2),
    char_ngram: tuple[int, int] = (2, 5),
    min_df: int = 2,
    max_features: int | None = 400_000,
) -> np.ndarray:
    """Смесь словного и символьного TF-IDF косинусов.

    Символьные n-граммы (char_wb) ловят артикулы/модели/бренды и опечатки — важный
    сигнал для матчинга товаров. Итог — взвешенное среднее двух косинусов (без обучения).
    """
    text1 = pairs["text1"].to_numpy()
    text2 = pairs["text2"].to_numpy()
    if fit_texts is None:
        fit_texts = pd.unique(np.concatenate([text1, text2]))
    word_cos = _tfidf_cosine(text1, text2, fit_texts, "word", word_ngram, min_df, max_features)
    char_cos = _tfidf_cosine(text1, text2, fit_texts, "char_wb", char_ngram, min_df, max_features)
    return word_weight * word_cos + (1.0 - word_weight) * char_cos


def _embed_scores(pairs: pd.DataFrame, **kwargs) -> np.ndarray:
    # Ленивая загрузка: torch/sentence-transformers нужны только методу 'embed'.
    from src.embed import embed_scores

    return embed_scores(pairs, **kwargs)


SCORERS = {
    "tfidf": tfidf_scores,
    "tfidf_wc": tfidf_wc_scores,
    "embed": _embed_scores,
}


def score_pairs(pairs: pd.DataFrame, method: str = "tfidf", **kwargs) -> np.ndarray:
    if method not in SCORERS:
        raise ValueError(f"Unknown method '{method}'. Available: {sorted(SCORERS)}")
    return SCORERS[method](pairs, **kwargs)
