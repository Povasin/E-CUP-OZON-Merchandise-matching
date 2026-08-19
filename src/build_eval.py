"""Валидационный набор, воспроизводящий устройство закрытого теста.

Наш holdout из ручной разметки систематически завышал качество: три точки лидерборда
показали, что порядок моделей по нему обратен настоящему. Причина в негативах. В тесте они
получены ретривалом — берут товар и подтягивают похожие, поэтому пары трудные. В ручной
разметке негативы набраны иначе и потому лёгкие.

Здесь набор собирается так же, как тест: положительные пары берутся из ручной разметки
(метки настоящие, размечено людьми), а негативы добываются ретривалом по товарам того же
holdout-фолда — ближайшие соседи внутри категории по лексической близости названий.
Доля положительных доводится до тестовой.

Оговорка, которую надо помнить: наши негативы никем не проверены, среди них неизбежно
попадутся настоящие совпадения. В тесте негативы после ретривала размечены людьми. Поэтому
абсолютное значение метрики здесь занижено; сравнивать модели между собой можно, объявлять
предсказанием лидерборда — нельзя.

Запуск:
    .venv/bin/python -m src.build_eval --out output/eval_retrieval
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer

from src.attr_features import parse_attributes
from src.hybrid import product_disjoint_pair_masks


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--items", default="assets/items_human.parquet")
    ap.add_argument("--matches", default="assets/matches.parquet")
    ap.add_argument("--out", default="output/eval_retrieval")
    ap.add_argument("--holdout-fold", type=int, default=0)
    ap.add_argument("--n-folds", type=int, default=3)
    ap.add_argument("--rate", type=float, default=0.111, help="доля положительных, как в тесте")
    ap.add_argument("--neighbours", type=int, default=8)
    ap.add_argument("--seed", type=int, default=2026)
    args = ap.parse_args()

    items = pd.read_parquet(args.items, columns=["id", "name", "attributes", "category"])
    matches = pd.read_parquet(args.matches, columns=["id1", "id2", "target"])
    left, right = matches["id1"].to_numpy(), matches["id2"].to_numpy()
    _, valid_mask = product_disjoint_pair_masks(left, right, args.holdout_fold, args.n_folds)
    valid = np.flatnonzero(valid_mask)
    target = matches["target"].to_numpy(np.int8)

    positive = valid[target[valid] == 1]
    print(f"положительных пар из ручной разметки: {len(positive):,}")

    # Любая уже размеченная пара исключается из кандидатов: как положительная (иначе
    # получим ложный негатив), так и отрицательная (иначе продублируем лёгкий негатив).
    known = set(map(tuple, np.sort(np.c_[left, right], axis=1).tolist()))

    holdout_ids = np.unique(np.concatenate([left[valid], right[valid]]))
    pool = items[items["id"].isin(holdout_ids)].reset_index(drop=True)
    needed = int(round(len(positive) / args.rate * (1 - args.rate)))
    print(f"нужно негативов ретривалом: {needed:,}")

    rng = np.random.default_rng(args.seed)
    # Негативы берутся из трёх источников намеренно. Набор только по лексической близости
    # состязателен к признакам, которые считаются по названиям: та же модель давала на нём
    # 0.334, а на случайных негативах 0.528. Настоящий ретривал организаторов — не наш
    # TF-IDF, поэтому смесь источников ближе к правде, чем любой один.
    lexical: list[tuple[int, int]] = []
    attribute: list[tuple[int, int]] = []
    random_pairs: list[tuple[int, int]] = []

    for category, group in pool.groupby("category", sort=False):
        names = group["name"].astype(str).tolist()
        identifiers = group["id"].to_numpy()
        if len(identifiers) < 20:
            continue
        matrix = TfidfVectorizer(min_df=1, sublinear_tf=True).fit_transform(names)
        # Соседей ищем блоками: полная матрица сходства на 5000 товаров — 200 МБ, а таких
        # категорий двадцать.
        for start in range(0, len(identifiers), 512):
            block = matrix[start:start + 512]
            similarity = (block @ matrix.T).toarray()
            for row in range(similarity.shape[0]):
                similarity[row, start + row] = -1.0
            top = np.argpartition(-similarity, args.neighbours, axis=1)[:, :args.neighbours]
            for row, neighbours in enumerate(top):
                source = int(identifiers[start + row])
                for column in neighbours:
                    other = int(identifiers[column])
                    key = (source, other) if source < other else (other, source)
                    if key not in known:
                        known.add(key)
                        lexical.append(key)

        # Совпадение атрибутов: товары одного бренда различаются вариантом, а не сутью —
        # такие негативы трудны для атрибутных признаков, но легки для лексических.
        by_slot: dict[str, list[int]] = {}
        for one, attributes in zip(identifiers, group["attributes"]):
            for value in parse_attributes(attributes).get("brand", ()):  # type: ignore[arg-type]
                by_slot.setdefault(value, []).append(int(one))
        for members in by_slot.values():
            if not 2 <= len(members) <= 200:
                continue
            for _ in range(min(len(members), 40)):
                a, b = int(rng.choice(members)), int(rng.choice(members))
                if a == b:
                    continue
                key = (a, b) if a < b else (b, a)
                if key not in known:
                    known.add(key)
                    attribute.append(key)

        for _ in range(len(identifiers)):
            a, b = int(rng.choice(identifiers)), int(rng.choice(identifiers))
            if a == b:
                continue
            key = (a, b) if a < b else (b, a)
            if key not in known:
                known.add(key)
                random_pairs.append(key)

    parts = []
    for name, source, share in (("лексические", lexical, 0.4),
                                ("по бренду", attribute, 0.3),
                                ("случайные", random_pairs, 0.3)):
        want = min(int(needed * share), len(source))
        picked = rng.permutation(len(source))[:want]
        parts.append(np.asarray([source[i] for i in picked], dtype=np.int64).reshape(-1, 2))
        print(f"  негативы {name}: доступно {len(source):,}, взято {want:,}")
    negatives = np.vstack([p for p in parts if len(p)])

    frame = pd.DataFrame({
        "id1": np.concatenate([left[positive], negatives[:, 0]]),
        "id2": np.concatenate([right[positive], negatives[:, 1]]),
        "target": np.concatenate([np.ones(len(positive), np.int8),
                                  np.zeros(len(negatives), np.int8)]),
    })
    category_of = dict(zip(items["id"], items["category"].astype(str)))
    frame["category"] = frame["id1"].map(category_of).astype(str)

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    frame.to_parquet(out / "eval_pairs.parquet", compression="zstd", index=False)
    print(f"\nсобрано пар: {len(frame):,}, доля положительных {frame['target'].mean() * 100:.1f}%")
    print(frame.groupby("category")["target"].agg(["size", "mean"]).round(3).to_string())
    print(f"\nсохранено: {out / 'eval_pairs.parquet'}")


if __name__ == "__main__":
    main()
