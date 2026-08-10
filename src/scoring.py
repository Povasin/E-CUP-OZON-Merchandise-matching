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


def tfidf_scores(
    pairs: pd.DataFrame,
    fit_texts: Iterable[str] | None = None,
    ngram_range: tuple[int, int] = (1, 2),
    min_df: int = 2,
    max_features: int | None = 400_000,
) -> np.ndarray:
    """Косинусная близость TF-IDF для каждой пары (text1, text2).

    fit_texts — корпус для обучения словаря; по умолчанию это уникальные тексты
    обеих сторон пар (эквивалент товаров текущего прогона).
    """
    text1 = pairs["text1"].to_numpy()
    text2 = pairs["text2"].to_numpy()

    if fit_texts is None:
        fit_texts = pd.unique(np.concatenate([text1, text2]))

    vectorizer = TfidfVectorizer(
        ngram_range=ngram_range,
        min_df=min_df,
        max_features=max_features,
        sublinear_tf=True,
        dtype=np.float32,
    )
    vectorizer.fit(fit_texts)

    m1 = vectorizer.transform(text1)
    m2 = vectorizer.transform(text2)
    return _rowwise_cosine(m1, m2)


SCORERS = {
    "tfidf": tfidf_scores,
}


def score_pairs(pairs: pd.DataFrame, method: str = "tfidf", **kwargs) -> np.ndarray:
    if method not in SCORERS:
        raise ValueError(f"Unknown method '{method}'. Available: {sorted(SCORERS)}")
    return SCORERS[method](pairs, **kwargs)
