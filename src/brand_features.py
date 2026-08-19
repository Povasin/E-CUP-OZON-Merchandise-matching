"""Нормализация брендов и цветов: словари алиасов, добытые из данных.

Победитель SIGMOD 2021 держал ядро решения на словарях алиасов вида `{"2320":"3435"}` —
приведении разных написаний одного значения к общему виду. Организаторы в разборе называют
это вместе с извлечением бренда и модели главным, что отделило призёров.

У нас бренд сравнивается как есть, поэтому «victor-reinz» и «victor reinz» расходятся, а
«nike» и «найк» тем более. Написания склеиваются двумя способами: механически (пунктуация,
регистр, гомоглифы) и по данным — если два написания регулярно стоят по разные стороны
пары с меткой «один товар», это одно и то же.

Цвета: у призёра E-CUP 2024 отдельный разборщик, раскладывающий составные названия
(«огненно-красный») на базовые оттенки. Без этого «огненно-красный» и «красный» —
несовпадение, хотя товар один.
"""
from __future__ import annotations

import json
import re
from collections import defaultdict
from pathlib import Path

from src.attr_features import PUNCTUATION, normalize

BASE_COLOURS: tuple[str, ...] = (
    "красн", "оранж", "желт", "жёлт", "зелен", "зелён", "голуб", "син", "фиолет",
    "розов", "коричн", "беж", "серебр", "золот", "черн", "чёрн", "бел", "сер",
    "прозрачн", "разноцвет", "бирюз", "салат", "бордов", "мятн", "хаки", "сирен",
)
ALIASES_PATH = Path(__file__).resolve().parent.parent / "models" / "brand_aliases.json"


def canonical(value: str) -> str:
    """Написание без пунктуации, регистра и латиницы, похожей на кириллицу."""
    return PUNCTUATION.sub("", normalize(value))


def colours(text: str) -> frozenset[str]:
    """Базовые оттенки, встреченные в строке.

    Ищем по корню, а не по точному слову: «огненно-красный», «красная», «красн.» дают
    один и тот же корень, а составное название честно раскладывается на несколько.
    """
    lowered = normalize(text)
    return frozenset(base for base in BASE_COLOURS if base in lowered)


def mine_aliases(pairs, brand_of, threshold: int = 3) -> dict[str, str]:
    """Написания, регулярно стоящие по разные стороны совпадающих пар, сводятся к одному.

    `pairs` — последовательность (id1, id2, target), `brand_of` — бренд по id. Считаем
    только положительные пары: там разные написания заведомо обозначают один бренд.
    """
    together: dict[tuple[str, str], int] = defaultdict(int)
    for left_id, right_id, target in pairs:
        if not target:
            continue
        left, right = brand_of.get(left_id), brand_of.get(right_id)
        if not left or not right or left == right:
            continue
        together[(left, right) if left < right else (right, left)] += 1

    # Объединение непересекающихся множеств: написания связываются в группы, за
    # представителя берётся самое короткое — обычно оно и есть каноническое.
    parent: dict[str, str] = {}

    def find(item: str) -> str:
        parent.setdefault(item, item)
        while parent[item] != item:
            parent[item] = parent[parent[item]]
            item = parent[item]
        return item

    for (left, right), count in together.items():
        if count < threshold:
            continue
        a, b = find(left), find(right)
        if a != b:
            parent[a] = b

    groups: dict[str, list[str]] = defaultdict(list)
    for item in list(parent):
        groups[find(item)].append(item)
    aliases: dict[str, str] = {}
    for members in groups.values():
        best = min(members, key=lambda value: (len(value), value))
        for member in members:
            if member != best:
                aliases[member] = best
    return aliases


def load_aliases() -> dict[str, str]:
    try:
        return json.loads(ALIASES_PATH.read_text())
    except OSError:
        return {}


def compare_brands(left: frozenset[str], right: frozenset[str],
                   aliases: dict[str, str]) -> dict[str, float]:
    """Совпадение брендов после приведения написаний."""
    a = frozenset(aliases.get(value, value) for value in left)
    b = frozenset(aliases.get(value, value) for value in right)
    shared = a & b
    return {
        "brand_alias_agree": float(bool(shared)),
        "brand_alias_conflict": float(bool(a) and bool(b) and not shared),
        "brand_alias_missing": float(bool(a) != bool(b)),
        "brand_alias_rescued": float(bool(shared) and not (left & right)),
    }


def compare_colours(left: frozenset[str], right: frozenset[str]) -> dict[str, float]:
    shared = left & right
    return {
        "colour_agree": float(bool(shared)),
        "colour_conflict": float(bool(left) and bool(right) and not shared),
        "colour_missing": float(bool(left) != bool(right)),
        "colour_jaccard": len(shared) / len(left | right) if (left | right) else 0.0,
        "colour_count_diff": float(abs(len(left) - len(right))),
    }


BRAND_FEATURE_NAMES: tuple[str, ...] = (
    "brand_alias_agree", "brand_alias_conflict", "brand_alias_missing",
    "brand_alias_rescued", "colour_agree", "colour_conflict", "colour_missing",
    "colour_jaccard", "colour_count_diff",
)
