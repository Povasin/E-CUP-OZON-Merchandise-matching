"""Точный поиск соседей на видеокарте: разреженное произведение без сжатия.

Процессорная версия квадратична по числу товаров: замерено 11.6 секунды на 20 тысячах
карточек и 50.9 на 60 тысячах, то есть на боевых 780 тысячах — часы при лимите в 13 минут.

Первая попытка ускорения сжимала TF-IDF случайной проекцией до нескольких сотен измерений.
Замер её похоронил: даже при 1024 измерениях ближайший сосед определялся верно лишь в 81%
случаев, а списки соседей совпадали на две трети. Признаки строятся на порядке соседей,
поэтому такая потеря превращает их в шум.

Здесь сжатия нет. Используется то, что `torch.sparse.mm` умеет «разреженное на плотное»:
вместо «блок строк × вся матрица» считается «вся матрица × блок столбцов», а результат
транспонируется. Арифметика та же, ответ точный, но произведение уходит на видеокарту.

Матрица TF-IDF разрежена — около десяти ненулевых на карточку, — поэтому работа растёт
линейно по числу карточек, а не квадратично: узкое место процессорной версии было в
`.toarray()`, который разворачивал плотный блок на всю ширину категории.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.feature_extraction.text import TfidfVectorizer

NEIGHBOURS = 20
# Ширина блока подбирается по памяти видеокарты: результат имеет размер
# «число карточек категории × ширина блока» и при 40 тысячах и 1024 занимает 160 МБ.
BLOCK = 1024


def to_torch_sparse(matrix: sparse.csr_matrix, device, dtype):
    import torch

    coo = matrix.tocoo()
    indices = torch.from_numpy(np.vstack([coo.row, coo.col])).long()
    values = torch.from_numpy(coo.data.astype(np.float32))
    return torch.sparse_coo_tensor(indices, values, coo.shape).to(device=device,
                                                                  dtype=dtype).coalesce()


def top_neighbours(matrix: sparse.csr_matrix, top_k: int, device
                   ) -> tuple[np.ndarray, np.ndarray]:
    """Точные сходства и индексы ближайших соседей, кроме самого себя."""
    import torch

    count = matrix.shape[0]
    k = min(top_k, count - 1)
    similarity = np.zeros((count, k), dtype=np.float32)
    index = np.zeros((count, k), dtype=np.int32)
    sparse_all = to_torch_sparse(matrix, device, torch.float32)

    with torch.inference_mode():
        for start in range(0, count, BLOCK):
            stop = min(start + BLOCK, count)
            block = torch.from_numpy(
                matrix[start:stop].toarray().astype(np.float32)
            ).to(device)
            # (N × V) на (V × B) даёт (N × B); транспонируем и получаем (B × N).
            scores = torch.sparse.mm(sparse_all, block.T).T
            rows = torch.arange(stop - start, device=device)
            scores[rows, rows + start] = -1.0
            values, positions = torch.topk(scores, k, dim=1)
            similarity[start:stop] = values.cpu().numpy()
            index[start:stop] = positions.cpu().numpy()
            del block, scores
    return similarity, index


def build(items: pd.DataFrame, categories: list[str] | None = None,
          top_k: int = NEIGHBOURS) -> dict[int, tuple]:
    """То же, что `neighbour_features.build`, но соседи ищутся на видеокарте."""
    import torch

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if categories is not None:
        items = items[items["category"].isin(categories)]
    profile: dict[int, tuple] = {}
    for _, group in items.groupby("category", sort=False):
        identifiers = group["id"].to_numpy()
        if len(identifiers) < 5:
            for one in identifiers:
                profile[int(one)] = (0.0, 0.0, 0.0, {}, 0.0)
            continue
        matrix = TfidfVectorizer(min_df=1, sublinear_tf=True).fit_transform(
            group["name"].astype(str).tolist()
        )
        similarity, index = top_neighbours(matrix, top_k, device)
        mean = similarity.mean(axis=1)
        spread = similarity.std(axis=1)
        for row, one in enumerate(identifiers):
            places = {int(identifiers[column]): position
                      for position, column in enumerate(index[row])}
            profile[int(one)] = (float(mean[row]), float(spread[row]),
                                 float(similarity[row, 0]), places, float(index.shape[1]))
    return profile
