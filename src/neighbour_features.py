"""Признаки пары относительно окрестности каждого из её товаров.

То, что объединяет победителей Shopee и Foursquare: выигрыш даёт не сила модели на паре, а
работа с окрестностью. Второе место Shopee строило LightGBM на средних и разбросах top-K
косинусов каждого товара и **нормировало эти агрегаты**, чтобы пережить расхождение
распределений. Первое место Foursquare считало статистики фаззи-метрик по ключу товара и
**отношения к ним**. Первое место Shopee итеративно смешивало эмбеддинг товара с соседями,
и это дало больше, чем любая смена модели.

Смысл в том, что абсолютное сходство несравнимо между товарами. В категории, где сотни
почти одинаковых карточек, сходство 0.8 — это «один из многих», а у товара с уникальным
названием то же 0.8 — «явное совпадение». Признак становится сравнимым, только если
отсчитывать его от окрестности: насколько эта пара выделяется среди прочих соседей.

Отсюда же переносимость на невиданные товары: относительная величина не зависит от того,
какие именно товары пришли, а в закрытом тесте они другие по условию соревнования.

Окрестности строятся по пришедшим данным, а не берутся из обучения, — как и IDF.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.feature_extraction.text import TfidfVectorizer

BLOCK = 512
NEIGHBOURS = 20


def category_neighbourhood(names: list[str], top_k: int = NEIGHBOURS
                           ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Для каждого товара: сходства с ближайшими соседями и их индексы.

    Возвращает (сходства top-k, индексы top-k, среднее, разброс). Считается блоками:
    полная матрица сходства на десятки тысяч товаров в память не поместится.
    """
    matrix = TfidfVectorizer(min_df=1, sublinear_tf=True).fit_transform(names)
    count = matrix.shape[0]
    k = min(top_k, max(count - 1, 1))
    top_similarity = np.zeros((count, k), dtype=np.float32)
    top_index = np.zeros((count, k), dtype=np.int32)
    mean = np.zeros(count, dtype=np.float32)
    spread = np.zeros(count, dtype=np.float32)

    for start in range(0, count, BLOCK):
        block = (matrix[start:start + BLOCK] @ matrix.T).toarray()
        for row in range(block.shape[0]):
            block[row, start + row] = -1.0
        order = np.argpartition(-block, k - 1, axis=1)[:, :k]
        rows = np.arange(block.shape[0])[:, None]
        values = block[rows, order]
        ordering = np.argsort(-values, axis=1)
        top_similarity[start:start + block.shape[0]] = values[rows, ordering]
        top_index[start:start + block.shape[0]] = order[rows, ordering]
        mean[start:start + block.shape[0]] = values.mean(axis=1)
        spread[start:start + block.shape[0]] = values.std(axis=1)
    return top_similarity, top_index, mean, spread


def build(items: pd.DataFrame, categories: list[str] | None = None
          ) -> dict[int, tuple[float, float, float, dict[int, int], float]]:
    """Окрестность каждого товара: среднее, разброс, лучший сосед, места соседей, k.

    `categories` ограничивает расчёт: словарь мест занимает около двухсот байт на товар, и
    на полутора миллионах карточек полный профиль не помещается в память. Поэтому признаки
    считаются по одной категории за раз, а профиль после неё освобождается.
    """
    profile: dict[int, tuple[float, float, float, dict[int, int], float]] = {}
    if categories is not None:
        items = items[items["category"].isin(categories)]
    for _, group in items.groupby("category", sort=False):
        identifiers = group["id"].to_numpy()
        if len(identifiers) < 5:
            for one in identifiers:
                profile[int(one)] = (0.0, 0.0, 0.0, {}, 0.0)
            continue
        similarity, index, mean, spread = category_neighbourhood(
            group["name"].astype(str).tolist()
        )
        for row, one in enumerate(identifiers):
            places = {int(identifiers[column]): position
                      for position, column in enumerate(index[row])}
            profile[int(one)] = (float(mean[row]), float(spread[row]),
                                 float(similarity[row, 0]), places, float(index.shape[1]))
    return profile


def compare(left_id: int, right_id: int, similarity: float,
            profile: dict) -> dict[str, float]:
    """Насколько пара выделяется на фоне окрестностей обеих своих сторон."""
    left = profile.get(int(left_id))
    right = profile.get(int(right_id))
    if left is None or right is None:
        return dict.fromkeys(NEIGHBOUR_FEATURE_NAMES, 0.0)

    scores = []
    places = []
    # Каждая сторона смотрит на свою окрестность и ищет в ней противоположную сторону.
    for own, other_id in ((left, right_id), (right, left_id)):
        mean, spread, _, place_of, k = own
        # Отклонение от среднего по окрестности в единицах её разброса: сколько бы ни было
        # похожих товаров вокруг, величина остаётся сравнимой между категориями.
        scores.append((similarity - mean) / spread if spread > 1e-6 else 0.0)
        # Место второй стороны среди соседей первой. Не попала в список — считаем «дальше
        # последнего», иначе отсутствие выглядело бы как близость.
        places.append(float(place_of.get(int(other_id), int(k))))

    ratio_left = similarity / left[2] if left[2] > 1e-6 else 0.0
    ratio_right = similarity / right[2] if right[2] > 1e-6 else 0.0
    return {
        "nb_z_min": float(min(scores)),
        "nb_z_max": float(max(scores)),
        "nb_place_min": float(min(places)),
        "nb_place_max": float(max(places)),
        "nb_is_best": float(min(places) == 0.0),
        "nb_both_best": float(max(places) == 0.0),
        "nb_ratio_min": float(min(ratio_left, ratio_right)),
        "nb_ratio_max": float(max(ratio_left, ratio_right)),
        "nb_mean_min": float(min(left[0], right[0])),
        "nb_spread_min": float(min(left[1], right[1])),
        "nb_similarity": float(similarity),
    }


NEIGHBOUR_FEATURE_NAMES: tuple[str, ...] = (
    "nb_z_min", "nb_z_max", "nb_place_min", "nb_place_max", "nb_is_best",
    "nb_both_best", "nb_ratio_min", "nb_ratio_max", "nb_mean_min",
    "nb_spread_min", "nb_similarity",
)
