"""Те же метрики сходства, но по каждому полю карточки отдельно.

Так набраны 120 признаков у победителя Kaggle Foursquare Location Matching: около десяти
метрик, применённых к нескольким полям, каждая в нескольких видах. У нас строковые метрики
считались только по названию — то есть одна девятая от возможного.

Совпадение по разным полям означает разное. Одинаковое название при разных брендах — это
почти наверняка разные товары одной категории. Одинаковый артикул при разных названиях —
наоборот, один товар у двух продавцов. Сведённые в одну строку, эти случаи неразличимы, и
именно поэтому поля надо сравнивать порознь.

Поля берутся из слотов `attr_features`, то есть уже сведённые: «артикул», «партномер
(артикул производителя)» и «oem-номер» — один слот, а не три разных ключа.
"""
from __future__ import annotations

from src.attr_features import Card
from src.name_features import jaccard

# Поля, по которым сравнение осмысленно. `type` включён потому, что «кроссовки» против
# «кеды» — это различие товара, а не оформления карточки.
FIELDS: tuple[str, ...] = ("brand", "article", "model", "type", "material")


def field_values(card: Card, slot: str) -> frozenset[str]:
    return card.slots.get(slot, frozenset())


def compare_field(left: frozenset[str], right: frozenset[str], prefix: str) -> dict[str, float]:
    """Три дешёвых признака на поле: совпало, пересеклось, отсутствует.

    Первая версия считала по каждому полю полный набор строковых метрик через `difflib` —
    вышло 56 пар в секунду, то есть два часа на закрытый тест при лимите в тринадцать минут.
    Замер показал, что метрики похожести давали от −3 до −9 пунктов, а весь сигнал сидел в
    самом факте наличия поля: `material_missing` даёт +12.5 при покрытии 53%. Поэтому
    осталось только то, что считается на множествах.

    Отсутствие поля — не то же самое, что расхождение: ноль в метрике похожести означал бы
    «совсем непохоже», хотя мы просто ничего не знаем. Поэтому признак отдельный.
    """
    if not left or not right:
        return {f"f_{prefix}_exact": 0.0, f"f_{prefix}_overlap": 0.0, f"f_{prefix}_missing": 1.0}
    return {
        f"f_{prefix}_exact": float(left == right),
        f"f_{prefix}_overlap": jaccard(left, right),
        f"f_{prefix}_missing": 0.0,
    }


def compare_fields(left: Card, right: Card) -> dict[str, float]:
    result: dict[str, float] = {}
    for slot in FIELDS:
        result.update(compare_field(field_values(left, slot), field_values(right, slot), slot))
    # Все значения атрибутов вместе: ловит совпадения в полях, не попавших ни в один слот.
    everything_left = frozenset(v for values in left.slots.values() for v in values)
    everything_right = frozenset(v for values in right.slots.values() for v in values)
    result.update(compare_field(everything_left, everything_right, "attrs"))
    return result


# Префикс обязателен: без него `article_exact` сталкивается с одноимённым признаком из
# `attr_features`, и при слиянии словарей один молча затирает другой.
FIELD_FEATURE_NAMES: tuple[str, ...] = tuple(
    f"f_{slot}_{metric}"
    for slot in (*FIELDS, "attrs")
    for metric in ("exact", "overlap", "missing")
)
