"""Сборка всех признаков пары в одну матрицу — общая для обучения и для инференса.

Раньше признаки собирались по месту в каждом эксперименте, и состав легко расходился между
обучением и решением. Здесь один порядок столбцов на всё: обучающий прогон и боевой
инференс зовут одну функцию, поэтому разойтись они не могут.

Товары обрабатываются по категориям: окрестности хранят по два десятка соседей на карточку,
и на полутора миллионах карточек полный профиль не помещается в память — прогон на
LLM-разметке на этом уже падал. После категории её профиль освобождается.
"""
from __future__ import annotations

import gc

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer

from src.attr_features import FEATURE_NAMES, compare, parse
from src.brand_features import (BRAND_FEATURE_NAMES, canonical, colours, compare_brands,
                                compare_colours)
from src.dim_features import DIM_FEATURE_NAMES, compare_dimensions, compare_translit, dimensions
from src.canon_features import CANON_FEATURES, compare_canonical, critical_fields
from src.canon_features import canonical as canonical_fields
from src.measure_features import MEASURE_FEATURES, compare_measures, measures
from src.name_features import NAME_FEATURE_NAMES, build_idf, compare_names, parse_name
from src.neighbour_features import NEIGHBOUR_FEATURE_NAMES
from src.neighbour_features import build as build_neighbourhood
from src.neighbour_features import compare as compare_neighbourhood
from src.string_features import STRING_FEATURE_NAMES, compare_strings

# Порядок и состав обязаны совпадать с тем, на чём обучалась модель. Признаки по полям и
# код категории написаны позже обучающего прогона и в модель не вошли: если добавить их
# сюда, инференс подаст 147 столбцов вместо 128 и модель молча выдаст мусор.
# Проверка на совпадение стоит в `src.pipeline`, а не только в комментарии.
ALL_FEATURE_NAMES: tuple[str, ...] = (
    FEATURE_NAMES + NAME_FEATURE_NAMES + STRING_FEATURE_NAMES
    + NEIGHBOUR_FEATURE_NAMES + BRAND_FEATURE_NAMES + DIM_FEATURE_NAMES
)


def category_codes(categories: np.ndarray, known: list[str]) -> np.ndarray:
    """Категория числом. Отдаётся признаком, а не только группировкой: без неё модель не
    знает, что размер входит в тождество товара в обуви и не входит в бытовой химии."""
    order = {name: index for index, name in enumerate(known)}
    return np.asarray([order.get(str(c), -1) for c in categories], dtype=np.float32)


# Признаки окрестности зависят не от пары, а от пула товаров вокруг: среднее сходство с
# соседями, место второй стороны в списке. Обе наши линейки построены на том же пуле, что и
# обучение, поэтому сдвиг пула на закрытом тесте они увидеть не могли — а он там есть, в
# тесте другие товары по условию. Отправка с ними дала на лидерборде минус при обещанных
# линейкой плюс три сотых, поэтому их можно отключить.
NEIGHBOUR_SET = frozenset(NEIGHBOUR_FEATURE_NAMES)


def feature_names(with_neighbours: bool = True, with_measures: bool = False) -> tuple[str, ...]:
    names = (ALL_FEATURE_NAMES if with_neighbours
             else tuple(n for n in ALL_FEATURE_NAMES if n not in NEIGHBOUR_SET))
    return names + MEASURE_FEATURES + CANON_FEATURES if with_measures else names


def build_matrix_parallel(items: pd.DataFrame, pairs: pd.DataFrame,
                          known_categories: list[str] | None = None,
                          workers: int | None = None,
                          with_neighbours: bool = True,
                          with_measures: bool = False) -> np.ndarray:
    """То же, что `build_matrix`, но категории считаются на разных ядрах.

    Дороже всего окрестности: внутри категории каждый товар сравнивается с каждым, и на
    сорока тысячах карточек это 23 секунды. Категорий двадцать, значит в один поток около
    460 секунд при общем лимите 780.

    Категории независимы — ни IDF, ни окрестности через границу не считаются, — поэтому
    разбиение по ним точное, в отличие от разбиения по парам. На двадцати ядрах те же
    двадцать категорий укладываются примерно в стоимость самой крупной.
    """
    import os
    from concurrent.futures import ProcessPoolExecutor

    workers = workers or min(os.cpu_count() or 1, 20)
    categories = sorted(set(items["category"].astype(str).tolist()))
    known = known_categories or categories
    if workers <= 1 or len(categories) <= 1:
        return build_matrix(items, pairs, known, with_neighbours, with_measures)

    category_of = dict(zip(items["id"].tolist(), items["category"].astype(str).tolist()))
    pair_categories = pairs["id1"].map(category_of).fillna("?").astype(str).to_numpy()

    tasks = []
    for category in categories:
        rows = np.flatnonzero(pair_categories == category)
        if len(rows):
            tasks.append((category, rows))
    # Крупные категории запускаются первыми: иначе они достанутся последнему свободному
    # ядру и растянут весь расчёт.
    tasks.sort(key=lambda task: -len(task[1]))

    result = np.zeros((len(pairs), len(feature_names(with_neighbours, with_measures))),
                      dtype=np.float32)
    with ProcessPoolExecutor(max_workers=workers) as pool:
        futures = {
            pool.submit(build_matrix,
                        items[items["category"].astype(str) == category].reset_index(drop=True),
                        pairs.iloc[rows].reset_index(drop=True), known, with_neighbours,
                        with_measures): rows
            for category, rows in tasks
        }
        for future in futures:
            result[futures[future]] = future.result()
    return result


def build_matrix(items: pd.DataFrame, pairs: pd.DataFrame,
                 known_categories: list[str] | None = None,
                 with_neighbours: bool = True,
                 with_measures: bool = False) -> np.ndarray:
    """Матрица признаков для пар в порядке их следования во входной таблице."""
    identifiers = items["id"].to_numpy()
    names = items["name"].astype(str).tolist()
    attributes = items["attributes"].tolist()
    categories = items["category"].astype(str).to_numpy()

    cards = {int(i): parse(n, a, name=n) for i, n, a in zip(identifiers, names, attributes)}
    parsed_names = {int(i): parse_name(n) for i, n in zip(identifiers, names)}
    idf, average = build_idf(list(parsed_names.values()))
    palette = {int(i): colours(n + " " + str(a))
               for i, n, a in zip(identifiers, names, attributes)}
    brands = {int(i): frozenset(x for x in (canonical(v) for v in c.slots.get("brand", ())) if x)
              for i, c in cards.items()}
    sizes = {int(i): dimensions(n + " " + str(a))
             for i, n, a in zip(identifiers, names, attributes)}
    measured = ({int(i): measures(a) for i, a in zip(identifiers, attributes)}
                if with_measures else {})
    canon = ({int(i): canonical_fields(a) for i, a in zip(identifiers, attributes)}
             if with_measures else {})
    position = {int(x): row for row, x in enumerate(identifiers)}
    category_of = dict(zip(identifiers.tolist(), categories.tolist()))

    pair_categories = pairs["id1"].map(category_of).fillna("?").astype(str).to_numpy()
    columns = feature_names(with_neighbours, with_measures)
    result = np.zeros((len(pairs), len(columns)), dtype=np.float32)

    left_ids = pairs["id1"].to_numpy()
    right_ids = pairs["id2"].to_numpy()

    for category in sorted(set(categories.tolist())):
        rows = np.flatnonzero(pair_categories == category)
        if not len(rows):
            continue
        critical = critical_fields(category)
        profile = (build_neighbourhood(items[["id", "name", "category"]], categories=[category])
                   if with_neighbours else {})
        group = items[items["category"] == category]
        # Матрица нужна только косинусу пары, а он идёт лишь в признаки окрестности.
        matrix, index = None, {}
        if with_neighbours:
            matrix = TfidfVectorizer(min_df=1, sublinear_tf=True).fit_transform(
                group["name"].astype(str).tolist()
            )
            index = {int(x): row for row, x in enumerate(group["id"].to_numpy())}
        for row in rows:
            left, right = int(left_ids[row]), int(right_ids[row])
            if left not in cards or right not in cards:
                continue
            values = compare(cards[left], cards[right])
            values.update(compare_names(parsed_names[left], parsed_names[right], idf, average))
            values.update(compare_strings(names[position[left]], names[position[right]]))
            if with_neighbours:
                similarity = (float((matrix[index[left]] @ matrix[index[right]].T).toarray()[0, 0])
                              if left in index and right in index else 0.0)
                values.update(compare_neighbourhood(left, right, similarity, profile))
            values.update(compare_brands(brands[left], brands[right], {}))
            values.update(compare_colours(palette[left], palette[right]))
            values.update(compare_dimensions(sizes[left], sizes[right]))
            if with_measures:
                values.update(compare_measures(measured[left], measured[right]))
                values.update(compare_canonical(canon[left], canon[right], critical))
            values.update(compare_translit(brands[left], brands[right]))
            result[row] = [values[name] for name in columns]
        del profile, matrix
        gc.collect()
    return result
