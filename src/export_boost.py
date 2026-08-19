"""Выгрузка обученного `HistGradientBoosting` в массивы numpy и предсказание по ним.

Решение запускается в чужом докер-образе, состав которого нам неизвестен: заглянуть внутрь
можно только скачав 6.7 ГБ. Полагаться на наличие и версию `sklearn` там нельзя — при
несовпадении версии распакованная модель либо не загрузится, либо загрузится молча
неправильно. Поэтому обученные деревья выгружаются в обычные массивы, а предсказание
считается несколькими строками на numpy.

Так снимается ограничение, которое до сих пор держало нас на самописном бустинге: обучать
можно чем угодно, а в архив уезжают только числа.

Деревья обходятся сразу для всех строк векторно: по уровню за шаг, вместо цикла по строкам.
На 390 тысячах пар и четырёхстах деревьях это разница между секундами и минутами.
"""
from __future__ import annotations

import numpy as np


def export(model) -> dict[str, np.ndarray]:
    """Плоское представление всех деревьев ансамбля.

    Узлы всех деревьев складываются в один массив, а `starts` помнит, где начинается
    каждое. Индексы потомков делаются глобальными, чтобы обход не зависел от границ.
    """
    features, thresholds, lefts, rights, values, leaves, missing_left = [], [], [], [], [], [], []
    starts = []
    offset = 0
    for stage in model._predictors:
        if len(stage) != 1:
            raise ValueError("Поддерживается только двухклассовая задача")
        nodes = stage[0].nodes
        if nodes["is_categorical"].any():
            raise ValueError(
                "Категориальные разбиения не выгружаются: обучайте с категорией как "
                "обычным числом, иначе потребуется переносить битовые маски"
            )
        starts.append(offset)
        features.append(nodes["feature_idx"].astype(np.int32))
        thresholds.append(nodes["num_threshold"].astype(np.float64))
        lefts.append(nodes["left"].astype(np.int32) + offset)
        rights.append(nodes["right"].astype(np.int32) + offset)
        values.append(nodes["value"].astype(np.float64))
        leaves.append(nodes["is_leaf"].astype(np.uint8))
        missing_left.append(nodes["missing_go_to_left"].astype(np.uint8))
        offset += len(nodes)

    return {
        "feature": np.concatenate(features),
        "threshold": np.concatenate(thresholds),
        "left": np.concatenate(lefts),
        "right": np.concatenate(rights),
        "value": np.concatenate(values),
        "is_leaf": np.concatenate(leaves),
        "missing_left": np.concatenate(missing_left),
        "starts": np.asarray(starts, dtype=np.int32),
        "baseline": np.asarray(model._baseline_prediction, dtype=np.float64).ravel(),
    }


def predict_raw(trees: dict[str, np.ndarray], features: np.ndarray) -> np.ndarray:
    """Сырой скор ансамбля (до сигмоиды) для всех строк сразу."""
    feature = trees["feature"]
    threshold = trees["threshold"]
    left, right = trees["left"], trees["right"]
    is_leaf, missing_left = trees["is_leaf"], trees["missing_left"]
    total = np.full(len(features), float(trees["baseline"][0]), dtype=np.float64)

    for start in trees["starts"]:
        node = np.full(len(features), start, dtype=np.int32)
        active = ~is_leaf[node].astype(bool)
        # Спускаемся по уровню за шаг: на каждом шаге часть строк доходит до листа и
        # выбывает. Глубина ограничена, поэтому шагов немного.
        while active.any():
            rows = np.flatnonzero(active)
            current = node[rows]
            column = feature[current]
            taken = features[rows, column]
            go_left = taken <= threshold[current]
            # Пропуск в признаке уводит в ту сторону, которую выбрало обучение.
            blank = np.isnan(taken)
            if blank.any():
                go_left = np.where(blank, missing_left[current].astype(bool), go_left)
            node[rows] = np.where(go_left, left[current], right[current])
            active[rows] = ~is_leaf[node[rows]].astype(bool)
        total += trees["value"][node]
    return total


def predict_proba(trees: dict[str, np.ndarray], features: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-np.clip(predict_raw(trees, features), -30.0, 30.0)))


def save(path: str, trees: dict[str, np.ndarray]) -> None:
    np.savez_compressed(path, **trees)


def load(path: str) -> dict[str, np.ndarray]:
    with np.load(path) as data:
        return {key: data[key] for key in data.files}
