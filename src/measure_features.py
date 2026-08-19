"""Числовые величины карточки, приведённые к базовым единицам.

Единица измерения в данных стоит то в значении (`"длина упаковки": "17 см"`), то в
ключе (`"высота, см": "17"`). Прежние извлекатели понимали только первый способ:
`src.dim_features.dimensions` ищет слитную запись «42х51х83см», а `NUMBER` в
`src.attr_features` требует число, за которым следует единица. Второй способ записи —
190 432 вхождения ключей из 1 634 552, то есть 11.7% — не видел никто.

Кросс-энкодер к тем же величинам слеп по другой причине: `SKIP_ATTRIBUTE_TERMS` в
`src.cross_encoder` намеренно выбрасывает всё, что относится к упаковке. Между тем
размеры упаковки одной стороны регулярно совпадают с размерами товара другой — это
разные способы описать один предмет. Замерено на отложенном LLM-фолде: такое сравнение
возможно для 29.7% пар, и при совпадении доля положительных 0.387 против 0.225.

Прибавка проверена дважды независимо: грубая версия отдельным участником поверх готовой
смеси дала +0.0045 при честном подборе веса (разброс замера 0.0026), разбор по смыслу
ключа внутри бустинга по семейству — +0.0050.
"""
from __future__ import annotations

import json
import re

NUMBER = re.compile(r"(\d+(?:[.,]\d+)?)")
# Единицы и множители к базовой: длина в миллиметрах, вес в граммах, объём в миллилитрах,
# мощность в ваттах. Порядок важен: «мм» должно проверяться раньше «м».
UNITS: tuple[tuple[str, float, str], ...] = (
    ("мм", 1.0, "L"), ("см", 10.0, "L"), ("метр", 1000.0, "L"), ("м", 1000.0, "L"),
    ("кг", 1000.0, "W"), ("гр", 1.0, "W"), ("г", 1.0, "W"),
    ("мл", 1.0, "V"), ("литр", 1000.0, "V"), ("л", 1000.0, "V"),
    ("квт", 1000.0, "P"), ("вт", 1.0, "P"),
)
# Смысл ключа. Габариты разведены с весом и объёмом: без этого «ширина упаковки» = 65
# совпадала с «длина, см» = 65, и признак «совпало» указывал в сторону «разные товары».
KINDS: tuple[tuple[str, str], ...] = (
    ("length", r"длин"), ("width", r"ширин"), ("height", r"высот"), ("depth", r"глубин"),
    ("diameter", r"диаметр"), ("thickness", r"толщин"), ("size", r"размер|габарит"),
    ("weight", r"\bвес|масс"), ("volume", r"об[ъь]?[её]м"), ("power", r"мощност"),
)
KIND_RE = tuple((name, re.compile(pattern)) for name, pattern in KINDS)
DIMENSION_KINDS = frozenset({"length", "width", "height", "depth", "diameter",
                             "thickness", "size"})
PACKAGING = re.compile(r"упаковк|упаковочн|с упаковкой|коробк|транспорт")
WITHOUT_PACKAGING = re.compile(r"без упаковки")
# Больше четырёх чисел в одном значении — это перечисление совместимости, а не величина
# товара: такие списки бывают в сотни номеров и только зашумляют сравнение.
MAX_NUMBERS_PER_VALUE = 4
MAX_PLAUSIBLE = 100_000.0
TOLERANCE = 0.02


def _unit(text: str) -> tuple[float, str] | None:
    lowered = text.lower()
    for name, multiplier, kind in UNITS:
        if re.search(rf"(^|[^а-яa-zё]){name}([^а-яa-zё]|$)", lowered):
            return multiplier, kind
    return None


def _kind(key: str) -> str | None:
    lowered = key.lower()
    for name, pattern in KIND_RE:
        if pattern.search(lowered):
            return name
    return None


class Measures:
    """Величины карточки: по единицам, по смыслу ключа и отдельно тарные."""

    __slots__ = ("product", "packaging", "by_kind")

    def __init__(self, product: frozenset, packaging: frozenset, by_kind: dict):
        self.product = product
        self.packaging = packaging
        self.by_kind = by_kind

    @property
    def everything(self) -> frozenset:
        return self.product | self.packaging


def measures(raw_attributes: object) -> Measures:
    try:
        parsed = json.loads(str(raw_attributes)) if raw_attributes else {}
    except (TypeError, ValueError):
        parsed = {}
    if not isinstance(parsed, dict):
        parsed = {}

    product: set[tuple[str, float]] = set()
    packaging: set[tuple[str, float]] = set()
    by_kind: dict[str, set[float]] = {}
    for key, value in parsed.items():
        key, text = str(key), str(value)
        unit = _unit(text) or _unit(key)
        if unit is None:
            continue
        multiplier, unit_kind = unit
        packed = bool(PACKAGING.search(key.lower())) and not WITHOUT_PACKAGING.search(key.lower())
        target = packaging if packed else product
        kind = _kind(key)
        for found in NUMBER.findall(text)[:MAX_NUMBERS_PER_VALUE]:
            amount = float(found.replace(",", ".")) * multiplier
            if not 0.0 < amount < MAX_PLAUSIBLE:
                continue
            target.add((unit_kind, round(amount, 1)))
            if kind is not None:
                by_kind.setdefault(("pack_" if packed else "") + kind, set()).add(round(amount, 1))
    return Measures(frozenset(product), frozenset(packaging),
                    {name: frozenset(values) for name, values in by_kind.items()})


def _jaccard(left: frozenset, right: frozenset) -> float:
    union = left | right
    return len(left & right) / len(union) if union else 0.0


def _close(one: float, other: float) -> bool:
    """Допуск на округление: «0.5 л» и «500 мл» сходятся, «500 мл» и «750 мл» — нет."""
    return abs(one - other) <= TOLERANCE * max(abs(one), abs(other), 1e-9)


def compare_measures(left: Measures, right: Measures) -> dict[str, float]:
    left_all, right_all = left.everything, right.everything
    # Упаковка против товара: у одной стороны предмет описан габаритами коробки, у
    # другой — габаритами самого предмета, и это единственный способ их сопоставить.
    cross = (left.packaging & right.product) | (right.packaging & left.product)

    shared_kinds = left.by_kind.keys() & right.by_kind.keys()
    agree = conflict = dimension_agree = dimension_conflict = 0
    for kind in shared_kinds:
        hit = any(_close(a, b) for a in left.by_kind[kind] for b in right.by_kind[kind])
        agree += hit
        conflict += not hit
        if kind in DIMENSION_KINDS:
            dimension_agree += hit
            dimension_conflict += not hit
    return {
        "measure_jaccard": _jaccard(left_all, right_all),
        "measure_shared": float(len(left_all & right_all)),
        "measure_conflict": float(bool(left_all) and bool(right_all)
                                  and not (left_all & right_all)),
        "measure_missing": float(bool(left_all) != bool(right_all)),
        "measure_product_jaccard": _jaccard(left.product, right.product),
        "measure_cross_pack": float(bool(cross)),
        "measure_cross_count": float(len(cross)),
        "measure_kind_shared": float(len(shared_kinds)),
        "measure_kind_agree": float(agree),
        "measure_kind_conflict": float(conflict),
        "measure_kind_all_agree": float(bool(shared_kinds) and conflict == 0),
        "measure_kind_all_clash": float(bool(shared_kinds) and agree == 0),
        "measure_kind_frac": agree / len(shared_kinds) if shared_kinds else 0.0,
        "measure_dim_agree": float(dimension_agree),
        "measure_dim_conflict": float(dimension_conflict),
        "measure_one_sided": float(len(left.by_kind.keys() ^ right.by_kind.keys())),
    }


MEASURE_FEATURES: tuple[str, ...] = (
    "measure_jaccard", "measure_shared", "measure_conflict", "measure_missing",
    "measure_product_jaccard", "measure_cross_pack", "measure_cross_count",
    "measure_kind_shared", "measure_kind_agree", "measure_kind_conflict",
    "measure_kind_all_agree", "measure_kind_all_clash", "measure_kind_frac",
    "measure_dim_agree", "measure_dim_conflict", "measure_one_sided",
)


def measure_signal(left: Measures, right: Measures) -> float:
    """Одно число: насколько величины двух карточек говорят «один товар».

    Отдельным участником смеси эта монотонная свёртка работает лучше, чем бустинг на
    всех шестнадцати признаках (+0.0043 против +0.0020 на отложенном фолде). Причина в
    том, что метрика ранжирует внутри категории, а бустинг учится предсказывать метку и
    тратит ёмкость на то, что метрике безразлично.
    """
    return (_jaccard(left.everything, right.everything)
            + 0.35 * float(bool((left.packaging & right.product)
                                | (right.packaging & left.product)))
            + 0.25 * _jaccard(left.product, right.product))
