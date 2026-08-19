"""Канонические поля карточки: ключи атрибутов, приведённые к смыслу.

Наш разбор сваливает «артикул», «партномер», «oem» и «код производителя» в один слот и
считает все расхождения одинаковыми. Здесь ключ приводится к смысловому полю (оптическая
сила, цилиндр, ось, радиус кривизны, размер производителя, российский размер), у каждой
категории свой набор решающих полей, а у полей — веса: расхождение в оптической силе для
аптеки значит больше, чем расхождение в цвете.

Таблицы соответствий взяты из ветки `povas` командного репозитория — это предметное
знание, которого у нас не было. Замерено на отложенном фолде поверх полной сборки:
0.647432 против 0.649824, то есть +0.0024 при разбросе 0.0022. Это чуть больше одного
стандартного отклонения — берём потому, что расчёт стоит четыре секунды на 156 тысяч пар,
а знак положительный, но на многое рассчитывать не стоит: это третья попытка выжать
что-то из атрибутов, и все три дали от 0.0015 до 0.0024.
"""
from __future__ import annotations

import json
import re

import numpy as np


IDENTIFIER_KEYS = (
    "sku", "код товара", "артикул", "арт", "партномер", "part number",
    "номер детали", "oem", "оем", "артикул производителя", "партномер производителя",
)
MODEL_KEYS = ("модель", "model", "серия", "линейка", "коллекция")
BRAND_KEYS = ("бренд", "brand", "производитель", "producer", "марка")
TYPE_KEYS = ("тип", "вид", "тип изделия", "вид изделия", "тип продукта", "вид товара", "назначение")
COLOR_KEYS = ("цвет", "color", "оттенок")

# More specific keys must appear before generic parent concepts.
SEMANTIC_KEY_RULES: tuple[tuple[str, tuple[str, ...]], ...] = (
    ("package_quantity", ("количество предметов в упаковке", "единиц в одном товаре", "количество в упаковке", "количество упаковок", "штук в упаковке", "количество ручек", "количество линз")),
    ("pet_size", ("размер животного", "размер птицы", "размер собаки", "размер кошки")),
    ("manufacturer_size", ("размер производителя", "eur размер", "размер обуви производителя")),
    ("shoe_size_ru", ("российский размер", "ru размер")),
    ("package_dimension", ("размер упаковки", "габарит упаковки", "длина упаковки", "ширина упаковки", "высота упаковки", "размеры упаковки")),
    ("product_dimension", ("габарит", "размеры", "размер товара", "длина", "ширина", "высота", "диаметр", "размер моста", "ширина линзы", "высота линзы", "длина заушника", "общая ширина")),
    ("weight", ("вес", "масса")),
    ("volume", ("объем", "вместимость")),
    ("power", ("мощность", "power")),
    ("voltage", ("напряжение", "voltage")),
    ("optical_power", ("оптическая сила", "диоптр", "sphere", "сфер")),
    ("cylinder", ("цилиндр", "cyl")),
    ("axis", ("ось", "axis", "ах")),
    ("radius", ("радиус кривизны", "радиус")),
    ("quantity", ("количество", "шт", "pcs", "units")),
    ("format", ("формат",)),
    ("density", ("плотность",)),
    ("type", TYPE_KEYS),
    ("brand", BRAND_KEYS),
    ("color", COLOR_KEYS),
    ("identifier", IDENTIFIER_KEYS),
    ("model", MODEL_KEYS),
    ("composition", ("состав", "ингредиенты", "материал", "материал изделия", "материал корпуса", "материал линз", "материал оправы")),
)

CATEGORY_CRITICAL_FIELDS = {
    "аптека": {"optical_power", "cylinder", "axis", "radius", "product_dimension", "package_quantity", "identifier", "model"},
    "автотовары": {"identifier", "model", "brand", "type", "product_dimension"},
    "обувь": {"model", "manufacturer_size", "shoe_size_ru", "color", "type", "brand"},
    "товары для животных": {"brand", "model", "pet_size", "package_quantity", "weight", "volume", "composition", "type"},
    "канцелярские товары": {"format", "product_dimension", "package_quantity", "density", "type", "brand"},
    "красота и гигиена": {"brand", "model", "volume", "weight", "composition", "type", "color"},
    "бытовая техника": {"brand", "model", "product_dimension", "volume", "weight", "power", "type", "identifier"},
}

FIELD_WEIGHT = {
    "identifier": 5.0, "model": 4.0, "type": 3.0, "package_quantity": 3.0,
    "weight": 3.0, "volume": 3.0, "shoe_size_ru": 3.0, "manufacturer_size": 3.0,
    "optical_power": 5.0, "cylinder": 5.0, "axis": 5.0, "radius": 4.0,
    "pet_size": 2.5, "product_dimension": 2.5, "brand": 2.0, "composition": 2.0,
    "color": 1.0, "format": 1.5, "density": 1.5, "power": 2.0, "voltage": 2.0,
    "quantity": 2.5,
}


SEMANTIC_RULES = tuple((name, tuple(k.lower() for k in keys)) for name, keys in SEMANTIC_KEY_RULES)
NUMBER = re.compile(r"-?\d+(?:[.,]\d+)?")
TOKEN = re.compile(r"[a-zа-яё0-9]+")
DEFAULT_CRITICAL = frozenset({"identifier", "model", "type", "package_quantity",
                              "weight", "volume", "product_dimension"})
TOLERANCE = 0.02


def semantic_of(key: str) -> str | None:
    """Смысловое поле для ключа атрибута. Более узкие правила стоят раньше общих."""
    lowered = key.lower()
    for name, keys in SEMANTIC_RULES:
        if any(k in lowered for k in keys):
            return name
    return None


def canonical(raw_attributes: object) -> tuple[dict, dict]:
    """Токены и числа карточки, разложенные по смысловым полям."""
    try:
        parsed = json.loads(str(raw_attributes)) if raw_attributes else {}
    except (TypeError, ValueError):
        return {}, {}
    if not isinstance(parsed, dict):
        return {}, {}
    fields: dict[str, set] = {}
    numbers: dict[str, set] = {}
    for key, value in parsed.items():
        field = semantic_of(str(key))
        if field is None:
            continue
        text = str(value).lower()
        fields.setdefault(field, set()).update(TOKEN.findall(text))
        found = NUMBER.findall(text.replace(",", "."))
        if found:
            numbers.setdefault(field, set()).add(float(found[0]))
    return ({k: frozenset(v) for k, v in fields.items()},
            {k: frozenset(v) for k, v in numbers.items()})


def critical_fields(category: str) -> frozenset[str]:
    lowered = str(category).lower()
    for name, fields in CATEGORY_CRITICAL_FIELDS.items():
        if name in lowered or lowered in name:
            return frozenset(fields)
    return DEFAULT_CRITICAL


def compare_canonical(left: tuple[dict, dict], right: tuple[dict, dict],
                      critical: frozenset[str]) -> dict[str, float]:
    left_fields, left_numbers = left
    right_fields, right_numbers = right
    shared = left_fields.keys() & right_fields.keys()
    exact = weighted_exact = weighted_total = conflicts = 0.0
    critical_conflicts = critical_shared = 0.0
    for field in shared:
        intersection = len(left_fields[field] & right_fields[field])
        union = len(left_fields[field] | right_fields[field])
        similarity = intersection / union if union else 0.0
        weight = FIELD_WEIGHT.get(field, 1.0)
        weighted_total += weight
        if similarity == 1.0:
            exact += 1
            weighted_exact += weight
        if similarity == 0.0:
            conflicts += 1
            if field in critical:
                critical_conflicts += 1
        elif field in critical:
            critical_shared += 1

    shared_numbers = left_numbers.keys() & right_numbers.keys()
    equal = clashing = 0
    relative = []
    for field in shared_numbers:
        a, b = min(left_numbers[field]), min(right_numbers[field])
        difference = abs(a - b) / max(abs(a), abs(b), 1e-9)
        relative.append(difference)
        if difference <= TOLERANCE:
            equal += 1
        else:
            clashing += 1
    return {
        "canon_shared": float(len(shared)),
        "canon_exact_ratio": exact / len(shared) if shared else 0.0,
        "canon_weighted_exact": weighted_exact / weighted_total if weighted_total else 0.0,
        "canon_conflicts": conflicts,
        "canon_conflict_ratio": conflicts / len(shared) if shared else 0.0,
        "canon_critical_conflicts": critical_conflicts,
        "canon_critical_any": float(critical_conflicts > 0),
        "canon_critical_shared": critical_shared,
        "canon_critical_ratio": critical_shared / len(critical) if critical else 0.0,
        "canon_numeric_shared": float(len(shared_numbers)),
        "canon_numeric_equal": float(equal),
        "canon_numeric_clash": float(clashing),
        "canon_numeric_mean_diff": float(np.mean(relative)) if relative else -1.0,
        "canon_numeric_min_diff": float(np.min(relative)) if relative else -1.0,
    }


CANON_FEATURES: tuple[str, ...] = (
    "canon_shared", "canon_exact_ratio", "canon_weighted_exact", "canon_conflicts",
    "canon_conflict_ratio", "canon_critical_conflicts", "canon_critical_any",
    "canon_critical_shared", "canon_critical_ratio", "canon_numeric_shared",
    "canon_numeric_equal", "canon_numeric_clash", "canon_numeric_mean_diff",
    "canon_numeric_min_diff",
)
