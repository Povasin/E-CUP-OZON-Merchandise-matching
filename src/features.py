"""Field-aware, leakage-free pair features for supervised product matching (Hybrid)."""
from __future__ import annotations

import json
import math
import re
from collections import Counter
from dataclasses import dataclass
from functools import lru_cache
from typing import Iterable

import os
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"

import numpy as np
import pandas as pd


TOKEN_RE = re.compile(r"\d+[.,]\d+|\d+|[a-zа-яё]+")
ALNUM_RE = re.compile(r"[a-zа-яё]*\d[a-zа-яё\d]*|\d+[a-zа-яё]+[a-zа-яё\d]*", re.I)
SIGN_NUM_RE = re.compile(r"[+-]?\d+(?:[.,]\d+)?")
MEASURE_RE = re.compile(
    r"(?P<value>[+-]?\d+(?:[.,]\d+)?)\s*(?P<unit>kg|кг|g|г|mg|мг|ml|мл|l|л|mm|мм|cm|см|m|м|km|км|gb|гб|mb|мб|tb|тб|w|вт|kw|квт|v|в|шт|pcs|pc|units|ед|\bшт\b|%)\b",
    re.I,
)
DIMENSION_RE = re.compile(
    r"(?P<a>[+-]?\d+(?:[.,]\d+)?)\s*[xх×]\s*(?P<b>[+-]?\d+(?:[.,]\d+)?)(?:\s*[xх×]\s*(?P<c>[+-]?\d+(?:[.,]\d+)?))?\s*(?P<unit>mm|мм|cm|см|m|м)\b",
    re.I,
)

UNIT_FACTORS = {
    "mg": ("mass_g", 0.001), "мг": ("mass_g", 0.001),
    "g": ("mass_g", 1.0), "г": ("mass_g", 1.0),
    "kg": ("mass_g", 1000.0), "кг": ("mass_g", 1000.0),
    "ml": ("volume_ml", 1.0), "мл": ("volume_ml", 1.0),
    "l": ("volume_ml", 1000.0), "л": ("volume_ml", 1000.0),
    "mm": ("length_mm", 1.0), "мм": ("length_mm", 1.0),
    "cm": ("length_mm", 10.0), "см": ("length_mm", 10.0),
    "m": ("length_mm", 1000.0), "м": ("length_mm", 1000.0),
    "km": ("length_mm", 1_000_000.0), "км": ("length_mm", 1_000_000.0),
    "gb": ("storage_mb", 1024.0), "гб": ("storage_mb", 1024.0),
    "mb": ("storage_mb", 1.0), "мб": ("storage_mb", 1.0),
    "tb": ("storage_mb", 1024.0 * 1024.0), "тб": ("storage_mb", 1024.0 * 1024.0),
    "w": ("power_w", 1.0), "вт": ("power_w", 1.0),
    "kw": ("power_w", 1000.0), "квт": ("power_w", 1000.0),
    "v": ("voltage_v", 1.0), "в": ("voltage_v", 1.0),
    "шт": ("quantity", 1.0), "pcs": ("quantity", 1.0), "pc": ("quantity", 1.0),
    "units": ("quantity", 1.0), "ед": ("quantity", 1.0),
    "%": ("percent", 1.0),
}

IDENTIFIER_KEYS = (
    "sku", "код товара", "артикул", "арт", "партномер", "part number",
    "номер детали", "oem", "оем", "артикул производителя", "партномер производителя",
)
MODEL_KEYS = ("модель", "model", "серия", "линейка", "коллекция")
BRAND_KEYS = ("бренд", "brand", "производитель", "producer", "марка")
TYPE_KEYS = ("тип", "вид", "тип изделия", "вид изделия", "тип продукта", "вид товара", "назначение")
COLOR_KEYS = ("цвет", "color", "оттенок")

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

def _is_missing(value: object) -> bool:
    if value is None:
        return True
    if isinstance(value, float):
        return math.isnan(value)
    return False

def _normalise(value: object) -> str:
    if _is_missing(value):
        return ""
    s = str(value).lower().replace("ё", "е")
    return " ".join(TOKEN_RE.findall(s))

def _compact_alnum(value: object) -> str:
    if _is_missing(value):
        return ""
    s = str(value).lower().replace("ё", "е")
    return re.sub(r"[^0-9a-zа-я]+", "", s)

def _tokens(value: str) -> frozenset[str]:
    return frozenset(value.split())

def _ordered_tokens(value: str) -> tuple[str, ...]:
    return tuple(value.split())

def _char_ngrams(value: str, n: int = 3) -> frozenset[str]:
    compact = value.replace(" ", "")
    if not compact:
        return frozenset()
    if len(compact) <= n:
        return frozenset((compact,))
    return frozenset(compact[i:i+n] for i in range(len(compact) - n + 1))

def _safe_json(raw: object) -> object:
    if _is_missing(raw):
        return {}
    raw_str = str(raw)
    if not raw_str or raw_str == "{}" or raw_str.lower() == "null":
        return {}
    try:
        return json.loads(raw_str)
    except (TypeError, ValueError, json.JSONDecodeError):
        return {}

def _flatten_text(value: object) -> str:
    if isinstance(value, (list, tuple, set)):
        return " ".join(_flatten_text(x) for x in value)
    if isinstance(value, dict):
        return " ".join(f"{k} {_flatten_text(v)}" for k, v in value.items())
    return str(value)

def _parse_attributes(raw: object) -> dict[str, str]:
    parsed = _safe_json(raw)
    if not isinstance(parsed, dict):
        return {}
    result: dict[str, str] = {}
    for key, value in parsed.items():
        norm_key = _normalise(key)
        norm_value = _normalise(_flatten_text(value))
        if norm_key and norm_value:
            result[norm_key] = norm_value
    return result

_NORMALIZED_SEMANTIC_RULES = None

@lru_cache(maxsize=8192)
def _canonical_field(key: str) -> str | None:
    global _NORMALIZED_SEMANTIC_RULES
    if _NORMALIZED_SEMANTIC_RULES is None:
        _NORMALIZED_SEMANTIC_RULES = tuple(
            (field, tuple(_normalise(needle) for needle in needles))
            for field, needles in SEMANTIC_KEY_RULES
        )
    k = _normalise(key)
    if not k:
        return None
    for field, needles in _NORMALIZED_SEMANTIC_RULES:
        for needle in needles:
            if k == needle or needle in k:
                return field
    return None

def _selected_values(attrs: dict[str, str], fields: Iterable[str]) -> frozenset[str]:
    wanted = set(fields)
    values: set[str] = set()
    for key, value in attrs.items():
        if _canonical_field(key) in wanted:
            values.update(value.split())
    return frozenset(values)

def _extract_identifier_strings(value: object) -> frozenset[str]:
    if _is_missing(value):
        return frozenset()
    s = str(value).lower().replace("ё", "е")
    candidates = re.findall(r"(?=[a-zа-я0-9]*[a-zа-я])(?=[a-zа-я0-9]*\d)[a-zа-я0-9]+", re.sub(r"[^a-zа-я0-9]+", " ", s))
    return frozenset(x for x in candidates if len(x) >= 2)

def _numbers(value: object) -> tuple[float, ...]:
    if _is_missing(value):
        return ()
    out = []
    for m in SIGN_NUM_RE.findall(str(value)):
        try:
            out.append(float(m.replace(",", ".")))
        except ValueError:
            continue
    return tuple(out)

def _canonical_measurements(key: str | None, value: object) -> list[tuple[str, float]]:
    if _is_missing(value):
        return []
    raw = str(value).lower().replace("ё", "е")
    semantic = key or ""
    result: list[tuple[str, float]] = []

    for m in DIMENSION_RE.finditer(raw):
        unit = m.group("unit").lower()
        family, factor = UNIT_FACTORS[unit]
        for name, number in (("d1", m.group("a")), ("d2", m.group("b")), ("d3", m.group("c"))):
            if number is not None:
                result.append(("product_dimension" if semantic in {"product_dimension", "package_dimension"} else family, float(number.replace(",", ".")) * factor))

    for m in MEASURE_RE.finditer(raw):
        unit = m.group("unit").lower()
        if unit not in UNIT_FACTORS:
            continue
        family, factor = UNIT_FACTORS[unit]
        num = float(m.group("value").replace(",", "."))
        if semantic == "weight": family = "mass_g"
        elif semantic == "volume": family = "volume_ml"
        elif semantic == "power": family = "power_w"
        elif semantic == "voltage": family = "voltage_v"
        elif semantic in {"package_quantity", "quantity"}: family = "quantity"
        elif semantic in {"optical_power", "cylinder", "axis", "radius", "shoe_size_ru", "manufacturer_size", "pet_size"}:
            family = semantic
            factor = 1.0
        elif semantic in {"product_dimension", "package_dimension"} and family == "length_mm":
            family = semantic
        result.append((family, num * factor))

    if not result and semantic in {"optical_power", "cylinder", "axis", "radius", "shoe_size_ru", "manufacturer_size", "quantity", "package_quantity", "density", "weight", "volume", "power", "voltage"}:
        for num in _numbers(raw):
            result.append((semantic, num))
    return result

def _canonical_text(value: object) -> str:
    return " ".join(_normalise(value).split())

def _set_stats(left: frozenset[str], right: frozenset[str]) -> tuple[float, float, float, float]:
    len_l = len(left); len_r = len(right)
    if not len_l and not len_r:
        return 0.0, 0.0, 0.0, 0.0
    common = len(left & right)
    union = len_l + len_r - common
    jaccard = common / union if union else 0.0
    dice = 2.0 * common / (len_l + len_r) if (len_l or len_r) else 0.0
    containment = common / min(len_l, len_r) if min(len_l, len_r) else 0.0
    count_ratio = min(len_l, len_r) / max(len_l, len_r) if max(len_l, len_r) else 0.0
    return jaccard, dice, containment, count_ratio

def _weighted_set_overlap(left: frozenset[str], right: frozenset[str]) -> tuple[float, float]:
    if not left and not right:
        return 0.0, 0.0
    def w(token: str) -> float:
        length_bonus = min(len(token), 12) / 12.0
        id_bonus = 1.8 if any(ch.isdigit() for ch in token) else 0.0
        alpha_num_bonus = 1.4 if bool(ALNUM_RE.fullmatch(token)) else 0.0
        generic_penalty = 0.65 if len(token) <= 3 and token.isalpha() else 0.0
        return max(0.25, 0.8 + length_bonus + id_bonus + alpha_num_bonus - generic_penalty)
    l_sum = sum(w(t) for t in left); r_sum = sum(w(t) for t in right)
    inter = sum(w(t) for t in left & right)
    union = l_sum + r_sum - inter
    containment = inter / min(l_sum, r_sum) if min(l_sum, r_sum) else 0.0
    return (inter / union if union else 0.0), containment

def _lcs_ratio(a: tuple[str, ...], b: tuple[str, ...]) -> float:
    if not a or not b:
        return 0.0
    a = a[:80]; b = b[:80]
    prev = [0] * (len(b) + 1)
    for x in a:
        cur = [0]
        for j, y in enumerate(b, 1):
            cur.append(prev[j-1] + 1 if x == y else max(prev[j], cur[-1]))
        prev = cur
    return prev[-1] / max(1, min(len(a), len(b)))

def _common_prefix_tokens(a: tuple[str, ...], b: tuple[str, ...]) -> float:
    n = min(len(a), len(b))
    i = 0
    while i < n and a[i] == b[i]:
        i += 1
    return i / max(1, n)

def _field_similarity_dict(left: dict[str, tuple[str, ...]], right: dict[str, tuple[str, ...]]) -> tuple[float, float, int, int]:
    keys = set(left) | set(right)
    if not keys:
        return 0.0, 0.0, 0, 0
    weighted_sum = 0.0; weight_total = 0.0; conflicts = 0; shared = 0
    for key in keys:
        lv = left.get(key); rv = right.get(key)
        w = FIELD_WEIGHT.get(key, 1.0)
        if lv and rv:
            shared += 1
            sim = _set_stats(frozenset(lv), frozenset(rv))[0]
            weighted_sum += w * sim
            weight_total += w
            if sim == 0.0:
                conflicts += 1
    return (weighted_sum / weight_total if weight_total else 0.0), (weight_total / sum(FIELD_WEIGHT.get(k, 1.0) for k in keys) if keys else 0.0), conflicts, shared

@dataclass(frozen=True, slots=True)
class ProductView:
    category: str
    name: str
    compact_name: str
    name_tokens: frozenset[str]
    ordered_tokens: tuple[str, ...]
    name_grams: frozenset[str]
    name_grams4: frozenset[str]
    attrs: dict[str, str]
    attr_keys: frozenset[str]
    value_tokens: frozenset[str]
    all_tokens: frozenset[str]
    digit_tokens: frozenset[str]
    identifier_tokens: frozenset[str]
    article_tokens: frozenset[str]
    model_tokens: frozenset[str]
    sku_tokens: frozenset[str]
    oem_tokens: frozenset[str]
    brand_tokens: frozenset[str]
    color_tokens: frozenset[str]
    type_tokens: frozenset[str]
    size_tokens: frozenset[str]
    composition_tokens: frozenset[str]
    canonical_fields: dict[str, tuple[str, ...]]
    numeric_fields: dict[str, tuple[float, ...]]
    measurements: tuple[tuple[str, float], ...]
    field_presence: frozenset[str]

EMPTY_VIEW = ProductView(
    category="", name="", compact_name="", name_tokens=frozenset(), ordered_tokens=(),
    name_grams=frozenset(), name_grams4=frozenset(), attrs={}, attr_keys=frozenset(),
    value_tokens=frozenset(), all_tokens=frozenset(), digit_tokens=frozenset(),
    identifier_tokens=frozenset(), article_tokens=frozenset(), model_tokens=frozenset(),
    sku_tokens=frozenset(), oem_tokens=frozenset(), brand_tokens=frozenset(),
    color_tokens=frozenset(), type_tokens=frozenset(), size_tokens=frozenset(),
    composition_tokens=frozenset(), canonical_fields={}, numeric_fields={},
    measurements=(), field_presence=frozenset()
)

def _make_view(name: object, raw_attrs: object, category: object = "") -> ProductView:
    norm_name = _normalise(name)
    attrs = _parse_attributes(raw_attrs)
    category_norm = _canonical_text(category)
    name_tokens = _tokens(norm_name)
    ordered = _ordered_tokens(norm_name)
    value_tokens = frozenset(token for value in attrs.values() for token in value.split())
    all_tokens = name_tokens | value_tokens
    digit_tokens = frozenset(token for token in all_tokens if any(ch.isdigit() for ch in token))

    field_values: dict[str, list[str]] = {}
    numeric_fields: dict[str, list[float]] = {}
    measurements: list[tuple[str, float]] = []

    for key, value in attrs.items():
        field = _canonical_field(key)
        if field:
            field_values.setdefault(field, []).extend(value.split())
            for family, num in _canonical_measurements(field, value):
                measurements.append((family, num))
                numeric_fields.setdefault(family, []).append(num)

    for family, num in _canonical_measurements(None, name):
        measurements.append((family, num))
        numeric_fields.setdefault(family, []).append(num)

    for m in DIMENSION_RE.finditer(str(name or "").lower()):
        unit = m.group("unit").lower(); _, factor = UNIT_FACTORS[unit]
        vals = [float(m.group("a").replace(",", ".")), float(m.group("b").replace(",", "."))]
        if m.group("c"):
            vals.append(float(m.group("c").replace(",", ".")))
        numeric_fields.setdefault("product_dimension", []).extend([v * factor for v in vals])

    name_identifiers = _extract_identifier_strings(name)
    modelish = frozenset(t for t in name_identifiers if len(t) >= 3)

    article = _selected_values(attrs, {"identifier"})
    model = _selected_values(attrs, {"model"})
    brand = _selected_values(attrs, {"brand"})
    color = _selected_values(attrs, {"color"})
    type_tokens = _selected_values(attrs, {"type"})
    composition = _selected_values(attrs, {"composition"})

    field_values_t = {k: tuple(sorted(set(v))) for k, v in field_values.items() if v}
    numeric_fields_t = {k: tuple(v) for k, v in numeric_fields.items() if v}
    identifier_tokens = frozenset(article | name_identifiers)

    return ProductView(
        category=category_norm,
        name=norm_name,
        compact_name=_compact_alnum(name),
        name_tokens=name_tokens,
        ordered_tokens=ordered,
        name_grams=_char_ngrams(norm_name, 3),
        name_grams4=_char_ngrams(norm_name, 4),
        attrs=attrs,
        attr_keys=frozenset(attrs),
        value_tokens=value_tokens,
        all_tokens=all_tokens,
        digit_tokens=digit_tokens,
        identifier_tokens=identifier_tokens,
        article_tokens=article,
        model_tokens=model | frozenset(modelish),
        sku_tokens=frozenset(t for t in article if any(x in str(raw_attrs).lower() for x in ("sku", "код товара"))),
        oem_tokens=frozenset(t for t in article if "oem" in str(raw_attrs).lower() or "оем" in str(raw_attrs).lower()),
        brand_tokens=brand,
        color_tokens=color,
        type_tokens=type_tokens,
        size_tokens=_selected_values(attrs, {"pet_size", "manufacturer_size", "shoe_size_ru", "package_dimension", "product_dimension", "weight", "volume"}),
        composition_tokens=composition,
        canonical_fields=field_values_t,
        numeric_fields=numeric_fields_t,
        measurements=tuple(measurements),
        field_presence=frozenset(field_values_t),
    )

def build_product_views(items: pd.DataFrame) -> dict[int, ProductView]:
    columns = ["id", "name", "attributes"]
    has_category = "category" in items.columns
    if has_category:
        columns.append("category")

    views: dict[int, ProductView] = {}
    for row in items[columns].itertuples(index=False, name=None):
        item_id, name, attrs = row[:3]
        category = row[3] if has_category else ""
        views[int(item_id)] = _make_view(name, attrs, category)
    return views

def _numeric_pair(left: ProductView, right: ProductView, field: str) -> tuple[float, float, float, float, float]:
    a = left.numeric_fields.get(field, ())
    b = right.numeric_fields.get(field, ())
    both = float(bool(a) and bool(b))
    if not a or not b:
        return both, 0.0, 0.0, 0.0, 0.0
    best = min(abs(x - y) for x in a for y in b)
    scale = max(1.0, max(abs(x) for x in a), max(abs(y) for y in b))
    rel = best / scale
    ratio = min(abs(x) / abs(y) if y else 0.0 for x in a for y in b)
    equal = float(best <= max(1e-9, 0.005 * scale))
    return both, equal, best, rel, ratio

def _numeric_aggregate(left: ProductView, right: ProductView) -> tuple[float, float, float, float, float, float, float]:
    fields = sorted(set(left.numeric_fields) | set(right.numeric_fields))
    present = 0; equal = 0; conflicts = 0; rel_sum = 0.0; ratio_sum = 0.0; abs_sum = 0.0
    for f in fields:
        both, eq, diff, rel, ratio = _numeric_pair(left, right, f)
        if both:
            present += 1
            equal += int(eq)
            conflicts += int(diff > 0 and eq == 0)
            rel_sum += rel; ratio_sum += ratio; abs_sum += diff
    return float(present), float(equal), float(conflicts), rel_sum / max(1, present), ratio_sum / max(1, present), abs_sum / max(1, present), float(len(fields))

def _identifier_stats(left: frozenset[str], right: frozenset[str]) -> tuple[float, float, float, float, float]:
    exact = float(bool(left and right and (left & right)))
    contain = _set_stats(left, right)[0]
    disjoint = float(bool(left and right) and not (left & right))
    presence_equal = float(bool(left) == bool(right))
    char = _set_stats(frozenset("".join(sorted(left))), frozenset("".join(sorted(right))))[0] if left and right else 0.0
    return exact, contain, disjoint, presence_equal, char

def _critical_fields(category: str) -> set[str]:
    key = category.lower()
    for cat, fields in CATEGORY_CRITICAL_FIELDS.items():
        if cat in key or key in cat:
            return set(fields)
    return {"identifier", "model", "type", "package_quantity", "weight", "volume", "product_dimension"}

def _pair_features(left: ProductView, right: ProductView) -> list[float]:
    name = _set_stats(left.name_tokens, right.name_tokens)
    grams3 = _set_stats(left.name_grams, right.name_grams)
    grams4 = _set_stats(left.name_grams4, right.name_grams4)
    values = _set_stats(left.value_tokens, right.value_tokens)
    all_tokens = _set_stats(left.all_tokens, right.all_tokens)
    keys = _set_stats(left.attr_keys, right.attr_keys)
    weighted_name_j, weighted_name_c = _weighted_set_overlap(left.name_tokens, right.name_tokens)

    category = left.category or right.category
    critical = _critical_fields(category)
    field_sim, field_coverage, field_conflicts, shared_fields = _field_similarity_dict(left.canonical_fields, right.canonical_fields)

    digits = _set_stats(left.digit_tokens, right.digit_tokens)
    articles = _identifier_stats(left.article_tokens, right.article_tokens)
    brands = _set_stats(left.brand_tokens, right.brand_tokens)
    colors = _set_stats(left.color_tokens, right.color_tokens)
    sizes = _set_stats(left.size_tokens, right.size_tokens)
    types = _set_stats(left.type_tokens, right.type_tokens)
    compositions = _set_stats(left.composition_tokens, right.composition_tokens)

    both_digits = bool(left.digit_tokens and right.digit_tokens)
    both_brands = bool(left.brand_tokens and right.brand_tokens)

    exact_shared = 0; weighted_exact = 0.0; weighted_total = 0.0; critical_conflicts = 0
    for f in set(left.canonical_fields) | set(right.canonical_fields):
        lv = left.canonical_fields.get(f); rv = right.canonical_fields.get(f)
        w = FIELD_WEIGHT.get(f, 1.0)
        if lv and rv:
            sim = _set_stats(frozenset(lv), frozenset(rv))[0]
            weighted_total += w
            weighted_exact += w * float(sim == 1.0)
            exact_shared += int(sim == 1.0)
            if f in critical and sim == 0.0:
                critical_conflicts += 1

    num_present, num_equal, num_conflict, num_rel, num_ratio, num_abs, num_field_count = _numeric_aggregate(left, right)

    name_brand = _set_stats(left.name_tokens, right.brand_tokens)
    rev_name_brand = _set_stats(right.name_tokens, left.brand_tokens)
    name_article = _set_stats(left.name_tokens, right.article_tokens)
    rev_name_article = _set_stats(right.name_tokens, left.article_tokens)
    name_model = _set_stats(left.name_tokens, right.model_tokens)
    rev_name_model = _set_stats(right.name_tokens, left.model_tokens)

    shared_key_n = len(left.attr_keys & right.attr_keys)
    union_key_n = len(left.attr_keys | right.attr_keys)
    missing_left = len(right.attr_keys - left.attr_keys)
    missing_right = len(left.attr_keys - right.attr_keys)
    critical_present_left = len(left.field_presence & critical)
    critical_present_right = len(right.field_presence & critical)
    critical_shared = len((left.field_presence & right.field_presence) & critical)

    seq_lcs = _lcs_ratio(left.ordered_tokens, right.ordered_tokens)
    prefix = _common_prefix_tokens(left.ordered_tokens, right.ordered_tokens)
    contains_tokens = float(bool(left.name_tokens and left.name_tokens <= right.name_tokens) or bool(right.name_tokens and right.name_tokens <= left.name_tokens))
    name_exact = float(bool(left.name) and left.name == right.name)
    compact_exact = float(bool(left.compact_name) and left.compact_name == right.compact_name)
    category_exact = float(bool(left.category) and bool(right.category) and left.category == right.category)

    return [
        name_exact, compact_exact, float(((left.name in right.name) or (right.name in left.name)) if left.name and right.name else False),
        *name, grams3[0], grams3[2], grams4[0], grams4[2],
        min(len(left.name), len(right.name)) / max(len(left.name), len(right.name)) if left.name and right.name else 0.0,
        contains_tokens, seq_lcs, prefix, weighted_name_j, weighted_name_c,
        values[0], values[2], values[3], all_tokens[0], all_tokens[2], all_tokens[3],
        keys[0], keys[2], keys[3],
        field_sim, field_coverage, float(exact_shared) / max(1, shared_fields),
        weighted_exact / max(1e-9, weighted_total), float(field_conflicts),
        float(shared_key_n), float(shared_key_n / max(1, union_key_n)), float(missing_left), float(missing_right),
        digits[0], digits[2], digits[3], float(len(left.digit_tokens ^ right.digit_tokens) if both_digits else 0.0),
        float(both_digits and not (left.digit_tokens & right.digit_tokens)), float(bool(left.digit_tokens) == bool(right.digit_tokens)),
        num_present, num_equal, num_conflict, num_rel, num_ratio, num_abs, num_field_count,
        *articles,
        *_identifier_stats(left.model_tokens, right.model_tokens),
        *_identifier_stats(left.sku_tokens, right.sku_tokens),
        *_identifier_stats(left.oem_tokens, right.oem_tokens),
        float((left.article_tokens | left.model_tokens) & (right.article_tokens | right.model_tokens) != frozenset()),
        float(critical_conflicts), float(critical_conflicts > 0),
        float(critical_shared), float(critical_shared / max(1, len(critical))),
        brands[0], brands[2], float(both_brands and left.brand_tokens == right.brand_tokens),
        float(both_brands and not (left.brand_tokens & right.brand_tokens)), float(bool(left.brand_tokens) == bool(right.brand_tokens)),
        colors[0], colors[2], float(bool(left.color_tokens and right.color_tokens and not (left.color_tokens & right.color_tokens))),
        sizes[0], sizes[2], types[0], types[2], compositions[0], compositions[2],
        *_numeric_pair(left, right, "mass_g"),
        *_numeric_pair(left, right, "volume_ml"),
        *_numeric_pair(left, right, "quantity"),
        *_numeric_pair(left, right, "product_dimension"),
        *_numeric_pair(left, right, "package_quantity"),
        *_numeric_pair(left, right, "power_w"),
        *_numeric_pair(left, right, "optical_power"),
        *_numeric_pair(left, right, "cylinder"),
        *_numeric_pair(left, right, "axis"),
        *_numeric_pair(left, right, "radius"),
        *_numeric_pair(left, right, "shoe_size_ru"),
        *_numeric_pair(left, right, "manufacturer_size"),
        max(name_brand[0], rev_name_brand[0]), max(name_brand[2], rev_name_brand[2]),
        max(name_article[0], rev_name_article[0]), max(name_article[2], rev_name_article[2]),
        max(name_model[0], rev_name_model[0]), max(name_model[2], rev_name_model[2]),
        float(category_exact),
        float(critical_present_left / max(1, len(critical))), float(critical_present_right / max(1, len(critical))),
    ]

def _document_frequency(items: pd.DataFrame) -> tuple[dict[str, int], dict[str, int], int]:
    name_df: dict[str, int] = {}
    attr_df: dict[str, int] = {}
    total = 0
    for row in items[["name", "attributes"]].itertuples(index=False, name=None):
        name, raw_attrs = row
        total += 1
        name_tokens = _tokens(_normalise(name))
        for token in name_tokens:
            name_df[token] = name_df.get(token, 0) + 1
        
        attrs = _parse_attributes(raw_attrs)
        value_tokens = set()
        for value in attrs.values():
            value_tokens.update(value.split())
        for token in value_tokens:
            attr_df[token] = attr_df.get(token, 0) + 1
    return name_df, attr_df, total

def _weighted_overlap(left: frozenset[str], right: frozenset[str], idf: dict[str, float]) -> tuple[float, float]:
    if not left or not right:
        return 0.0, 0.0
    weight_inter = sum(idf.get(t, 1.0) for t in left & right)
    weight_union = sum(idf.get(t, 1.0) for t in left | right)
    weight_min = min(sum(idf.get(t, 1.0) for t in left), sum(idf.get(t, 1.0) for t in right))
    jaccard = weight_inter / weight_union if weight_union else 0.0
    containment = weight_inter / weight_min if weight_min else 0.0
    return jaccard, containment

def _extra_features(left: ProductView, right: ProductView, name_idf: dict[str, float], name_max_idf: float, attr_idf: dict[str, float]) -> list[float]:
    idf_jaccard, idf_containment = _weighted_overlap(left.name_tokens, right.name_tokens, name_idf)
    shared = left.name_tokens & right.name_tokens
    only = (left.name_tokens - right.name_tokens) | (right.name_tokens - left.name_tokens)
    max_shared = max((name_idf.get(t, 1.0) for t in shared), default=0.0) / name_max_idf if name_max_idf else 0.0
    max_unshared = max((name_idf.get(t, 1.0) for t in only), default=0.0) / name_max_idf if name_max_idf else 0.0
    
    rarest_left = max(left.name_tokens, key=lambda t: (name_idf.get(t, 1.0), t)) if left.name_tokens else None
    rarest_right = max(right.name_tokens, key=lambda t: (name_idf.get(t, 1.0), t)) if right.name_tokens else None
    rarest_shared = float(
        rarest_left is not None and rarest_left in right.name_tokens
        and rarest_right is not None and rarest_right in left.name_tokens
    )
    attr_jaccard, attr_containment = _weighted_overlap(left.value_tokens, right.value_tokens, attr_idf)
    return [idf_jaccard, idf_containment, max_shared, max_unshared, rarest_shared, attr_jaccard, attr_containment]

FEATURE_NAMES = [
    "name_exact", "name_compact_exact", "name_str_contains",
    "name_jaccard", "name_dice", "name_containment", "name_count_ratio",
    "name_char3_jaccard", "name_char3_containment", "name_char4_jaccard", "name_char4_containment",
    "name_length_ratio", "name_token_full_contains", "name_lcs_ratio", "name_common_prefix_ratio",
    "name_weighted_jaccard", "name_weighted_containment",
    "value_jaccard", "value_containment", "value_count_ratio",
    "all_jaccard", "all_containment", "all_count_ratio",
    "key_jaccard", "key_containment", "key_count_ratio",
    "field_weighted_similarity", "field_weighted_coverage", "shared_field_exact_ratio",
    "weighted_shared_exact", "field_conflict_count",
    "shared_key_count", "shared_key_ratio", "missing_key_left", "missing_key_right",
    "digit_jaccard", "digit_containment", "digit_count_ratio", "digit_sym_diff", "digit_disjoint", "digit_presence_equal",
    "numeric_both_field_count", "numeric_equal_field_count", "numeric_conflict_field_count", "numeric_mean_rel_diff", "numeric_mean_ratio", "numeric_mean_abs_diff", "numeric_field_union_count",
    "article_exact", "article_containment", "article_disjoint", "article_presence_equal", "article_char_similarity",
    "model_exact", "model_containment", "model_disjoint", "model_presence_equal", "model_char_similarity",
    "sku_exact", "sku_containment", "sku_disjoint", "sku_presence_equal", "sku_char_similarity",
    "oem_exact", "oem_containment", "oem_disjoint", "oem_presence_equal", "oem_char_similarity",
    "identifier_cross_exact", "critical_conflict_count", "critical_conflict_any", "critical_shared_count", "critical_shared_ratio",
    "brand_jaccard", "brand_containment", "brand_exact", "brand_disjoint", "brand_presence_equal",
    "color_jaccard", "color_containment", "color_conflict",
    "size_jaccard", "size_containment", "type_jaccard", "type_containment",
    "composition_jaccard", "composition_containment",
]

_NUMERIC_FEATURE_TEMPLATE = (
    "mass_g", "volume_ml", "quantity", "product_dimension", "package_quantity", "power_w",
    "optical_power", "cylinder", "axis", "radius", "shoe_size_ru", "manufacturer_size"
)
for _field in _NUMERIC_FEATURE_TEMPLATE:
    FEATURE_NAMES.extend([
        f"{_field}_both", f"{_field}_equal", f"{_field}_abs_diff", f"{_field}_rel_diff", f"{_field}_ratio",
    ])

FEATURE_NAMES.extend([
    "name_x_brand_jaccard", "name_x_brand_containment",
    "name_x_article_jaccard", "name_x_article_containment",
    "name_x_model_jaccard", "name_x_model_containment",
    "category_exact", "critical_coverage_left", "critical_coverage_right",
])

EXTRA_FEATURE_NAMES = [
    "idf_name_jaccard", "idf_name_containment", "idf_max_shared", "idf_max_unshared", "idf_rarest_shared",
    "attr_idf_jaccard", "attr_idf_containment"
]

FEATURE_NAMES.extend(EXTRA_FEATURE_NAMES)
MODEL_FEATURE_NAMES = FEATURE_NAMES + ["text_tfidf", "name_tfidf"]

def _features_for_chunk(payload: tuple) -> np.ndarray:
    matches, items, name_idf, name_max_idf, attr_idf = payload
    views = build_product_views(items)
    result = np.empty((len(matches), len(FEATURE_NAMES)), dtype=np.float32)
    for row, (id1, id2) in enumerate(matches[["id1", "id2"]].itertuples(index=False, name=None)):
        left = views.get(int(id1), EMPTY_VIEW)
        right = views.get(int(id2), EMPTY_VIEW)
        result[row] = _pair_features(left, right) + _extra_features(left, right, name_idf, name_max_idf, attr_idf)
    return result

def extract_pair_features_parallel(matches: pd.DataFrame, items: pd.DataFrame, workers: int | None = None) -> np.ndarray:
    import os
    from concurrent.futures import ProcessPoolExecutor

    workers = workers or min(os.cpu_count() or 1, 20)
    
    item_chunks = [items.iloc[idx].reset_index(drop=True) for idx in np.array_split(np.arange(len(items)), workers)]
    name_df: dict[str, int] = {}
    attr_df: dict[str, int] = {}
    total = 0
    if workers > 1 and len(matches) >= 20_000:
        with ProcessPoolExecutor(max_workers=workers) as pool:
            for chunk_name, chunk_attr, chunk_total in pool.map(_document_frequency, item_chunks):
                total += chunk_total
                for token, count in chunk_name.items():
                    name_df[token] = name_df.get(token, 0) + count
                for token, count in chunk_attr.items():
                    attr_df[token] = attr_df.get(token, 0) + count
    else:
        for chunk in item_chunks:
            chunk_name, chunk_attr, chunk_total = _document_frequency(chunk)
            total += chunk_total
            for token, count in chunk_name.items():
                name_df[token] = name_df.get(token, 0) + count
            for token, count in chunk_attr.items():
                attr_df[token] = attr_df.get(token, 0) + count

    name_idf = {t: math.log((total + 1) / (c + 1)) + 1.0 for t, c in name_df.items()}
    attr_idf = {t: math.log((total + 1) / (c + 1)) + 1.0 for t, c in attr_df.items()}
    name_max_idf = math.log(total + 1) + 1.0 if total else 0.0
    del name_df, attr_df

    row_by_id = {int(item_id): row for row, item_id in enumerate(items["id"].to_numpy())}
    left_ids = matches["id1"].to_numpy()
    right_ids = matches["id2"].to_numpy()
    
    if workers > 1 and len(matches) >= 20_000:
        payloads = []
        for index in np.array_split(np.arange(len(matches)), workers * 2):
            rows = {row_by_id[int(i)] for i in left_ids[index] if int(i) in row_by_id}
            rows.update(row_by_id[int(i)] for i in right_ids[index] if int(i) in row_by_id)
            payloads.append((matches.iloc[index].reset_index(drop=True),
                             items.iloc[sorted(rows)].reset_index(drop=True),
                             name_idf, name_max_idf, attr_idf))
        with ProcessPoolExecutor(max_workers=workers) as pool:
            parts = list(pool.map(_features_for_chunk, payloads))
        return np.vstack(parts)
    else:
        return _features_for_chunk((matches, items, name_idf, name_max_idf, attr_idf))

def extract_model_features(matches: pd.DataFrame, items: pd.DataFrame) -> np.ndarray:
    from src.data import attach_texts, build_item_text
    from src.scoring import tfidf_wc_scores

    structured = extract_pair_features_parallel(matches, items)
    text_items = items[["id", "category"]].copy()
    text_items["text"] = build_item_text(items)
    text_pairs = attach_texts(matches, text_items)
    try:
        text_cosine = tfidf_wc_scores(text_pairs)
    except ValueError:
        text_cosine = np.zeros(len(matches), dtype=np.float32)

    id_to_name = dict(zip(items["id"], items["name"].fillna("").astype(str)))
    name_pairs = matches[["id1", "id2"]].copy()
    name_pairs["text1"] = name_pairs["id1"].map(id_to_name).fillna("")
    name_pairs["text2"] = name_pairs["id2"].map(id_to_name).fillna("")
    try:
        name_cosine = tfidf_wc_scores(name_pairs, max_features=300_000)
    except ValueError:
        name_cosine = np.zeros(len(matches), dtype=np.float32)
    
    return np.column_stack((structured, text_cosine, name_cosine)).astype(np.float32)
