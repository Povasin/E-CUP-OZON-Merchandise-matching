"""Fast, model-free pair features for supervised product matching.

The features describe relations between two cards rather than memorising concrete
products.  This is important for the competition test, where every product is new.
"""
from __future__ import annotations

import json
import math
import re
from dataclasses import dataclass

import numpy as np
import pandas as pd


TOKEN_RE = re.compile(r"[0-9a-zа-яё]+", re.IGNORECASE)
ARTICLE_KEYS = ("артикул", "модель", "партномер", "oem", "код товара", "sku")
BRAND_KEYS = ("бренд", "производитель", "марка")
SIZE_KEYS = ("размер",)
COLOR_KEYS = ("цвет",)

# Discriminative but low-IDF tokens: two apparel/footwear cards that differ only in
# size/colour/gender are DIFFERENT products, yet lexically near-identical.  Explicit
# conflict signals rescue exactly the categories where set overlap cannot separate them.
COLOR_LEXICON = frozenset(
    """черный белый красный синий зеленый желтый серый коричневый розовый оранжевый
    фиолетовый голубой бежевый золотой серебряный бордовый бирюзовый сиреневый бронзовый
    малиновый салатовый хаки пудровый молочный кремовый графитовый черная белая черное
    белое black white red blue green yellow gray grey brown pink orange purple beige gold
    silver""".split()
)
GENDER_LEXICON = frozenset(
    """мужской женский детский унисекс мужская женская мужчин женщин male female unisex
    мальчик девочка мужские женские детские""".split()
)

# Quantity with units generalises across the whole catalogue: two otherwise identical
# cards with a different pack size / weight / volume are different products.  Units are
# normalised to a base per dimension so "1 л" and "500 мл" become comparable numbers.
QUANTITY_UNITS = {
    "мл": ("vol", 1.0), "ml": ("vol", 1.0), "л": ("vol", 1000.0), "l": ("vol", 1000.0),
    "литр": ("vol", 1000.0),
    "мг": ("mass", 0.001), "мкг": ("mass", 1e-6), "г": ("mass", 1.0), "гр": ("mass", 1.0),
    "g": ("mass", 1.0), "кг": ("mass", 1000.0), "kg": ("mass", 1000.0),
    "мм": ("len", 1.0), "см": ("len", 10.0), "м": ("len", 1000.0), "mm": ("len", 1.0),
    "cm": ("len", 10.0),
    "шт": ("count", 1.0), "pcs": ("count", 1.0), "таб": ("count", 1.0), "капс": ("count", 1.0),
}
QUANTITY_RE = re.compile(r"(\d+(?:[.,]\d+)?)\s?([а-яёa-z]+)")


def _extract_quantities(name: object, raw_attrs: object) -> frozenset[str]:
    """Set of "dimension:base_value" strings parsed from name + attribute values."""
    parts: list[str] = []
    if name is not None and not (isinstance(name, float) and pd.isna(name)):
        parts.append(str(name))
    if raw_attrs is not None and not (isinstance(raw_attrs, float) and pd.isna(raw_attrs)):
        try:
            parsed = json.loads(str(raw_attrs))
        except (TypeError, ValueError, json.JSONDecodeError):
            parsed = None
        if isinstance(parsed, dict):
            parts.extend(str(value) for value in parsed.values())
        elif parsed is not None:
            parts.append(str(parsed))
    text = " ".join(parts).lower().replace("ё", "е")
    if not text:
        return frozenset()
    values: set[str] = set()
    for number, unit in QUANTITY_RE.findall(text):
        info = QUANTITY_UNITS.get(unit)
        if info is None:
            continue
        dimension, multiplier = info
        base_value = round(float(number.replace(",", ".")) * multiplier, 4)
        values.add(f"{dimension}:{base_value:g}")
    return frozenset(values)


def _normalise(value: object) -> str:
    if value is None or pd.isna(value):
        return ""
    return " ".join(TOKEN_RE.findall(str(value).lower().replace("ё", "е")))


def _tokens(value: str) -> frozenset[str]:
    return frozenset(value.split())


def _char_ngrams(value: str, n: int = 3) -> frozenset[str]:
    compact = value.replace(" ", "")
    if not compact:
        return frozenset()
    if len(compact) <= n:
        return frozenset((compact,))
    return frozenset(compact[i : i + n] for i in range(len(compact) - n + 1))


def _parse_attributes(raw: object) -> dict[str, str]:
    if raw is None or pd.isna(raw):
        return {}
    try:
        parsed = json.loads(str(raw))
    except (TypeError, ValueError, json.JSONDecodeError):
        return {}
    if not isinstance(parsed, dict):
        return {}
    result: dict[str, str] = {}
    for key, value in parsed.items():
        norm_key = _normalise(key)
        norm_value = _normalise(value)
        if norm_key and norm_value:
            result[norm_key] = norm_value
    return result


def _selected_values(attrs: dict[str, str], needles: tuple[str, ...]) -> frozenset[str]:
    values: set[str] = set()
    for key, value in attrs.items():
        if any(needle in key for needle in needles):
            values.update(value.split())
    return frozenset(values)


def _compact_identifier_tokens(attrs: dict[str, str]) -> frozenset[str]:
    """Article/model identifiers invariant to spaces, hyphens and underscores.

    Marketplace schemas frequently store the same identifier as ``sh-p7251`` and
    ``shp7251`` or as ``gdb2070 zfr`` and ``gdb2070zfr``. Normal token overlap misses
    these exact matches, while concatenating only adjacent identifier fragments keeps
    the signal precise.
    """
    result: set[str] = set()
    for key, value in attrs.items():
        if not any(needle in key for needle in ARTICLE_KEYS):
            continue
        parts = value.split()
        for part in parts:
            if len(part) >= 3 and any(char.isdigit() for char in part):
                result.add(part)
        for width in (2, 3):
            for start in range(len(parts) - width + 1):
                compact = "".join(parts[start : start + width])
                if 4 <= len(compact) <= 64 and any(char.isdigit() for char in compact):
                    result.add(compact)
    return frozenset(result)


def _compact_name_identifiers(name: str) -> frozenset[str]:
    """Model-like name fragments, invariant to separators."""
    parts = name.split()
    result: set[str] = {
        part for part in parts if len(part) >= 3 and any(char.isdigit() for char in part)
    }
    for width in (2, 3):
        for start in range(len(parts) - width + 1):
            compact = "".join(parts[start : start + width])
            if 4 <= len(compact) <= 48 and any(char.isdigit() for char in compact):
                result.add(compact)
    return frozenset(result)


def _semantic_value_groups(attrs: dict[str, str]) -> tuple[frozenset[str], ...]:
    """Schema-tolerant token sets for important product-identity fields."""
    grouped: list[set[str]] = [set() for _ in range(4)]
    for key, value in attrs.items():
        keys = (
            key == "тип" or key in {"вид товара", "название группы"},
            ("модель" in key.split() and "на модели" not in key)
            or any(term in key for term in ("серия", "линейка")),
            any(term in key for term in ("материал", "состав")),
            any(term in key for term in ("коллекц", "серия", "линейка")),
        )
        for index, selected in enumerate(keys):
            if selected:
                grouped[index].update(value.split())
    return tuple(frozenset(values) for values in grouped)


def _is_family_key(key: str, family: str) -> bool:
    words = set(key.split())
    if family == "size":
        return "размер" in words and not any(
            noise in key for noise in ("на модели", "модели на фото", "упаков", "параметр")
        )
    if family == "color":
        return "цвет" in words
    if family == "gender":
        return "пол" in words
    if family == "type":
        return key == "тип" or key in {"вид товара", "название группы"}
    if family == "model":
        return "модель" in words and not any(
            noise in key for noise in ("на модели", "модели на фото", "параметр")
        )
    raise ValueError(f"Unknown attribute family: {family}")


def _family_attributes(attrs: dict[str, str]) -> tuple[dict[str, str], ...]:
    """Precompute small semantic-key maps once per product, not once per pair."""
    return tuple(
        {key: value for key, value in attrs.items() if _is_family_key(key, family)}
        for family in ("size", "color", "gender", "type", "model")
    )


def _family_key_stats(
    left: dict[str, str], right: dict[str, str]
) -> tuple[float, float, float, float]:
    """Exact-key value agreement without dilution by unrelated shared attributes."""
    left_keys = set(left)
    right_keys = set(right)
    shared = left_keys & right_keys
    if not shared:
        return 0.0, 0.0, 0.0, float(bool(left_keys) == bool(right_keys))
    exact = 0
    similarity = 0.0
    conflicts = 0
    for key in shared:
        left_value = left[key]
        right_value = right[key]
        exact += int(left_value == right_value)
        value_jaccard = _set_stats(_tokens(left_value), _tokens(right_value))[0]
        similarity += value_jaccard
        conflicts += int(value_jaccard == 0.0)
    count = len(shared)
    return exact / count, similarity / count, conflicts / count, 1.0


def _v10_feature_values(
    left_identifiers: frozenset[str],
    right_identifiers: frozenset[str],
    left_families: tuple[dict[str, str], ...],
    right_families: tuple[dict[str, str], ...],
) -> list[float]:
    compact_ids = _set_stats(left_identifiers, right_identifiers)
    both_compact_ids = bool(left_identifiers and right_identifiers)
    family_stats = tuple(
        value
        for left_family, right_family in zip(left_families, right_families)
        for value in _family_key_stats(left_family, right_family)
    )
    return [
        compact_ids[0], compact_ids[2],
        float(both_compact_ids and bool(left_identifiers & right_identifiers)),
        float(both_compact_ids and not (left_identifiers & right_identifiers)),
        float(bool(left_identifiers) == bool(right_identifiers)),
        *family_stats,
    ]


def _v11_feature_values(
    left_name_ids: frozenset[str],
    right_name_ids: frozenset[str],
    left_value_phrases: frozenset[str],
    right_value_phrases: frozenset[str],
    left_key_values: frozenset[str],
    right_key_values: frozenset[str],
    left_semantic: tuple[frozenset[str], ...],
    right_semantic: tuple[frozenset[str], ...],
) -> list[float]:
    name_ids = _set_stats(left_name_ids, right_name_ids)
    both_name_ids = bool(left_name_ids and right_name_ids)
    value_phrases = _set_stats(left_value_phrases, right_value_phrases)
    key_values = _set_stats(left_key_values, right_key_values)
    semantic = tuple(
        value
        for left_values, right_values in zip(left_semantic, right_semantic)
        for value in _class_conflict(left_values, right_values)
    )
    return [
        name_ids[0], name_ids[2],
        float(both_name_ids and bool(left_name_ids & right_name_ids)),
        float(both_name_ids and not (left_name_ids & right_name_ids)),
        float(bool(left_name_ids) == bool(right_name_ids)),
        *value_phrases,
        *key_values,
        *semantic,
    ]


def _count_features(left: frozenset, right: frozenset) -> tuple[float, float, float]:
    left_count = len(left)
    right_count = len(right)
    return (
        math.log1p(min(left_count, right_count)),
        math.log1p(max(left_count, right_count)),
        math.log1p(abs(left_count - right_count)),
    )


def _v12_feature_values(
    left_name_tokens: frozenset[str],
    right_name_tokens: frozenset[str],
    left_attr_keys: frozenset[str],
    right_attr_keys: frozenset[str],
    left_value_tokens: frozenset[str],
    right_value_tokens: frozenset[str],
    left_value_phrases: frozenset[str],
    right_value_phrases: frozenset[str],
    left_key_values: frozenset[str],
    right_key_values: frozenset[str],
    left_name_ids: frozenset[str],
    right_name_ids: frozenset[str],
    left_compact_ids: frozenset[str],
    right_compact_ids: frozenset[str],
) -> list[float]:
    return [
        *_count_features(left_name_tokens, right_name_tokens),
        *_count_features(left_attr_keys, right_attr_keys),
        *_count_features(left_value_tokens, right_value_tokens),
        *_count_features(left_value_phrases, right_value_phrases),
        math.log1p(len(left_value_phrases & right_value_phrases)),
        math.log1p(len(left_key_values & right_key_values)),
        math.log1p(len(left_name_ids & right_name_ids)),
        math.log1p(len(left_compact_ids & right_compact_ids)),
        float(bool(left_attr_keys) != bool(right_attr_keys)),
        float(bool(left_name_ids) != bool(right_name_ids)),
    ]


def _v13_feature_values(
    left_name: str,
    right_name: str,
    left_attrs: dict[str, str],
    right_attrs: dict[str, str],
    left_digits: frozenset[str],
    right_digits: frozenset[str],
    left_articles: frozenset[str],
    right_articles: frozenset[str],
    left_compact_ids: frozenset[str],
    right_compact_ids: frozenset[str],
    left_quantities: frozenset[str],
    right_quantities: frozenset[str],
) -> list[float]:
    shared_keys = set(left_attrs) & set(right_attrs)
    exact = 0
    conflicts = 0
    for key in shared_keys:
        left_value = left_attrs[key]
        right_value = right_attrs[key]
        exact += int(left_value == right_value)
        conflicts += int(not (_tokens(left_value) & _tokens(right_value)))
    left_attr_chars = sum(len(key) + len(value) for key, value in left_attrs.items())
    right_attr_chars = sum(len(key) + len(value) for key, value in right_attrs.items())
    return [
        *_count_features(left_digits, right_digits),
        *_count_features(left_articles, right_articles),
        *_count_features(left_compact_ids, right_compact_ids),
        *_count_features(left_quantities, right_quantities),
        math.log1p(min(len(left_name), len(right_name))),
        math.log1p(max(len(left_name), len(right_name))),
        math.log1p(abs(len(left_name) - len(right_name))),
        math.log1p(min(left_attr_chars, right_attr_chars)),
        math.log1p(max(left_attr_chars, right_attr_chars)),
        math.log1p(abs(left_attr_chars - right_attr_chars)),
        math.log1p(len(shared_keys)),
        math.log1p(exact),
        math.log1p(conflicts),
    ]


@dataclass(frozen=True, slots=True)
class ProductView:
    name: str
    name_tokens: frozenset[str]
    name_grams: frozenset[str]
    name_grams4: frozenset[str]
    name_grams5: frozenset[str]
    attrs: dict[str, str]
    family_attrs: tuple[dict[str, str], ...]
    attr_keys: frozenset[str]
    value_tokens: frozenset[str]
    all_tokens: frozenset[str]
    digit_tokens: frozenset[str]
    num_values: frozenset[float]
    article_tokens: frozenset[str]
    compact_identifiers: frozenset[str]
    name_identifiers: frozenset[str]
    value_phrases: frozenset[str]
    key_value_pairs: frozenset[str]
    semantic_values: tuple[frozenset[str], ...]
    brand_tokens: frozenset[str]
    size_tokens: frozenset[str]
    color_tokens: frozenset[str]
    gender_tokens: frozenset[str]
    quantities: frozenset[str]


EMPTY_VIEW = ProductView(
    name="",
    name_tokens=frozenset(),
    name_grams=frozenset(),
    name_grams4=frozenset(),
    name_grams5=frozenset(),
    attrs={},
    family_attrs=({}, {}, {}, {}, {}),
    attr_keys=frozenset(),
    value_tokens=frozenset(),
    all_tokens=frozenset(),
    digit_tokens=frozenset(),
    num_values=frozenset(),
    article_tokens=frozenset(),
    compact_identifiers=frozenset(),
    name_identifiers=frozenset(),
    value_phrases=frozenset(),
    key_value_pairs=frozenset(),
    semantic_values=(frozenset(),) * 4,
    brand_tokens=frozenset(),
    size_tokens=frozenset(),
    color_tokens=frozenset(),
    gender_tokens=frozenset(),
    quantities=frozenset(),
)


def _gender_values(attrs: dict[str, str], name_tokens: frozenset[str]) -> frozenset[str]:
    values: set[str] = set()
    for key, value in attrs.items():
        if "пол" in key.split():  # whole word, so "полнота" (width) is not gender
            values.update(value.split())
    values.update(token for token in name_tokens if token in GENDER_LEXICON)
    return frozenset(values)


def _make_view(name: object, raw_attrs: object) -> ProductView:
    norm_name = _normalise(name)
    name_tokens = _tokens(norm_name)
    attrs = _parse_attributes(raw_attrs)
    value_tokens = frozenset(token for value in attrs.values() for token in value.split())
    all_tokens = name_tokens | value_tokens
    digit_tokens = frozenset(token for token in all_tokens if any(ch.isdigit() for ch in token))
    num_values = frozenset(float(token) for token in all_tokens if token.isdigit())
    color_tokens = _selected_values(attrs, COLOR_KEYS) | frozenset(
        token for token in name_tokens if token in COLOR_LEXICON
    )
    return ProductView(
        name=norm_name,
        name_tokens=name_tokens,
        name_grams=_char_ngrams(norm_name),
        name_grams4=_char_ngrams(norm_name, 4),
        name_grams5=_char_ngrams(norm_name, 5),
        attrs=attrs,
        family_attrs=_family_attributes(attrs),
        attr_keys=frozenset(attrs),
        value_tokens=value_tokens,
        all_tokens=all_tokens,
        digit_tokens=digit_tokens,
        num_values=num_values,
        article_tokens=_selected_values(attrs, ARTICLE_KEYS),
        compact_identifiers=_compact_identifier_tokens(attrs),
        name_identifiers=_compact_name_identifiers(norm_name),
        value_phrases=frozenset(attrs.values()),
        key_value_pairs=frozenset(f"{key}={value}" for key, value in attrs.items()),
        semantic_values=_semantic_value_groups(attrs),
        brand_tokens=_selected_values(attrs, BRAND_KEYS),
        size_tokens=_selected_values(attrs, SIZE_KEYS),
        color_tokens=color_tokens,
        gender_tokens=_gender_values(attrs, name_tokens),
        quantities=_extract_quantities(name, raw_attrs),
    )


def build_product_views(items: pd.DataFrame) -> dict[int, ProductView]:
    """Build compact parsed views once per product id."""
    return {
        int(item_id): _make_view(name, attrs)
        for item_id, name, attrs in items[["id", "name", "attributes"]].itertuples(index=False, name=None)
    }


def _build_idf(values, attr: bool) -> tuple[dict[str, float], float]:
    """Document-frequency IDF over the run corpus (name or attribute tokens)."""
    document_frequency: dict[str, int] = {}
    n_documents = 0
    for view in values:
        n_documents += 1
        tokens = view.value_tokens if attr else view.name_tokens
        for token in tokens:
            document_frequency[token] = document_frequency.get(token, 0) + 1
    idf = {t: math.log((n_documents + 1) / (c + 1)) + 1.0 for t, c in document_frequency.items()}
    max_idf = math.log(n_documents + 1) + 1.0
    return idf, max_idf


def _weighted_overlap(
    left: frozenset[str], right: frozenset[str], idf: dict[str, float]
) -> tuple[float, float]:
    if not left or not right:
        return 0.0, 0.0
    weight_inter = sum(idf.get(t, 1.0) for t in left & right)
    weight_union = sum(idf.get(t, 1.0) for t in left | right)
    weight_min = min(
        sum(idf.get(t, 1.0) for t in left), sum(idf.get(t, 1.0) for t in right)
    )
    jaccard = weight_inter / weight_union if weight_union else 0.0
    containment = weight_inter / weight_min if weight_min else 0.0
    return jaccard, containment


def _set_stats(left: frozenset[str], right: frozenset[str]) -> tuple[float, float, float, float]:
    if not left and not right:
        return 0.0, 0.0, 0.0, 0.0
    common = len(left & right)
    union = len(left | right)
    jaccard = common / union if union else 0.0
    dice = 2.0 * common / (len(left) + len(right)) if left or right else 0.0
    containment = common / min(len(left), len(right)) if left and right else 0.0
    count_ratio = min(len(left), len(right)) / max(len(left), len(right)) if left and right else 0.0
    return jaccard, dice, containment, count_ratio


def _extra_features(
    left: ProductView,
    right: ProductView,
    name_idf: dict[str, float],
    name_max_idf: float,
    attr_idf: dict[str, float],
) -> list[float]:
    """IDF-weighted, numeric and finer-grained relational signals."""
    left_names, right_names = left.name_tokens, right.name_tokens

    idf_jaccard, idf_containment = _weighted_overlap(left_names, right_names, name_idf)

    shared = left_names & right_names
    only = (left_names - right_names) | (right_names - left_names)
    max_shared = max((name_idf.get(t, 1.0) for t in shared), default=0.0) / name_max_idf
    max_unshared = max((name_idf.get(t, 1.0) for t in only), default=0.0) / name_max_idf
    rarest_left = max(left_names, key=lambda t: name_idf.get(t, 1.0)) if left_names else None
    rarest_right = max(right_names, key=lambda t: name_idf.get(t, 1.0)) if right_names else None
    rarest_shared = float(
        rarest_left is not None and rarest_left in right_names
        and rarest_right is not None and rarest_right in left_names
    )

    intersection = len(shared)
    extra_left = len(left_names - right_names)
    extra_right = len(right_names - left_names)
    denom = intersection + extra_left + extra_right
    extra_min = min(extra_left, extra_right) / denom if denom else 0.0
    extra_max = max(extra_left, extra_right) / denom if denom else 0.0

    left_nums, right_nums = left.num_values, right.num_values
    if left_nums and right_nums:
        common = len(left_nums & right_nums)
        num_jaccard = common / len(left_nums | right_nums)
        num_containment = common / min(len(left_nums), len(right_nums))
        num_disjoint = float(common == 0)
    else:
        num_jaccard = num_containment = num_disjoint = 0.0
    num_presence_equal = float(bool(left_nums) == bool(right_nums))

    grams4 = _set_stats(left.name_grams4, right.name_grams4)
    grams5 = _set_stats(left.name_grams5, right.name_grams5)

    attr_jaccard, attr_containment = _weighted_overlap(left.value_tokens, right.value_tokens, attr_idf)

    left_tokens_list = left.name.split()
    right_tokens_list = right.name.split()
    first_match = float(
        bool(left_tokens_list) and bool(right_tokens_list)
        and left_tokens_list[0] == right_tokens_list[0]
    )
    last_match = float(
        bool(left_tokens_list) and bool(right_tokens_list)
        and left_tokens_list[-1] == right_tokens_list[-1]
    )

    size = _class_conflict(left.size_tokens, right.size_tokens)
    color = _class_conflict(left.color_tokens, right.color_tokens)
    gender = _class_conflict(left.gender_tokens, right.gender_tokens)
    any_conflict = float(bool(size[2] or color[2] or gender[2]))

    quantity = _quantity_features(left.quantities, right.quantities)
    v10 = _v10_feature_values(
        left.compact_identifiers,
        right.compact_identifiers,
        left.family_attrs,
        right.family_attrs,
    )
    v11 = _v11_feature_values(
        left.name_identifiers,
        right.name_identifiers,
        left.value_phrases,
        right.value_phrases,
        left.key_value_pairs,
        right.key_value_pairs,
        left.semantic_values,
        right.semantic_values,
    )
    v12 = _v12_feature_values(
        left.name_tokens,
        right.name_tokens,
        left.attr_keys,
        right.attr_keys,
        left.value_tokens,
        right.value_tokens,
        left.value_phrases,
        right.value_phrases,
        left.key_value_pairs,
        right.key_value_pairs,
        left.name_identifiers,
        right.name_identifiers,
        left.compact_identifiers,
        right.compact_identifiers,
    )
    v13 = _v13_feature_values(
        left.name,
        right.name,
        left.attrs,
        right.attrs,
        left.digit_tokens,
        right.digit_tokens,
        left.article_tokens,
        right.article_tokens,
        left.compact_identifiers,
        right.compact_identifiers,
        left.quantities,
        right.quantities,
    )

    return [
        idf_jaccard, idf_containment,
        extra_min, extra_max,
        num_jaccard, num_containment, num_disjoint, num_presence_equal,
        grams4[0], grams4[2],
        first_match, last_match,
        max_shared, max_unshared, rarest_shared,
        attr_jaccard, attr_containment,
        grams5[0], grams5[2],
        *size, *color, *gender, any_conflict,
        *quantity,
        *v10,
        *v11,
        *v12,
        *v13,
    ]


def _quantity_features(left: frozenset[str], right: frozenset[str]) -> tuple[float, ...]:
    """(jaccard, containment, disjoint, presence_equal, same_dimension_conflict)."""
    presence_equal = float(bool(left) == bool(right))
    if not left or not right:
        return 0.0, 0.0, 0.0, presence_equal, 0.0
    common = len(left & right)
    jaccard = common / len(left | right)
    containment = common / min(len(left), len(right))
    disjoint = float(common == 0)
    left_dims = {q.split(":", 1)[0] for q in left}
    right_dims = {q.split(":", 1)[0] for q in right}
    dimension_conflict = 0.0
    for dimension in left_dims & right_dims:
        prefix = dimension + ":"
        left_values = {q for q in left if q.startswith(prefix)}
        right_values = {q for q in right if q.startswith(prefix)}
        if left_values and right_values and not (left_values & right_values):
            dimension_conflict = 1.0
            break
    return jaccard, containment, disjoint, presence_equal, dimension_conflict


def _class_conflict(left: frozenset[str], right: frozenset[str]) -> tuple[float, float, float, float]:
    """(jaccard, containment, disjoint, presence_equal) for one attribute class."""
    presence_equal = float(bool(left) == bool(right))
    if not left or not right:
        return 0.0, 0.0, 0.0, presence_equal
    common = len(left & right)
    jaccard = common / len(left | right)
    containment = common / min(len(left), len(right))
    disjoint = float(common == 0)
    return jaccard, containment, disjoint, presence_equal


V10_FEATURE_NAMES = [
    "compact_id_jaccard", "compact_id_containment", "compact_id_match", "compact_id_disjoint",
    "compact_id_presence_equal",
    "size_key_exact", "size_key_jaccard", "size_key_conflict", "size_key_presence_equal",
    "color_key_exact", "color_key_jaccard", "color_key_conflict", "color_key_presence_equal",
    "gender_key_exact", "gender_key_jaccard", "gender_key_conflict", "gender_key_presence_equal",
    "type_key_exact", "type_key_jaccard", "type_key_conflict", "type_key_presence_equal",
    "model_key_exact", "model_key_jaccard", "model_key_conflict", "model_key_presence_equal",
]

V11_FEATURE_NAMES = [
    "name_id_jaccard", "name_id_containment", "name_id_match", "name_id_disjoint",
    "name_id_presence_equal",
    "value_phrase_jaccard", "value_phrase_dice", "value_phrase_containment",
    "value_phrase_count_ratio",
    "key_value_jaccard", "key_value_dice", "key_value_containment", "key_value_count_ratio",
    "type_value_jaccard", "type_value_containment", "type_value_disjoint", "type_value_presence_equal",
    "model_value_jaccard", "model_value_containment", "model_value_disjoint", "model_value_presence_equal",
    "material_value_jaccard", "material_value_containment", "material_value_disjoint",
    "material_value_presence_equal",
    "collection_value_jaccard", "collection_value_containment", "collection_value_disjoint",
    "collection_value_presence_equal",
]

V12_FEATURE_NAMES = [
    "name_count_min_log", "name_count_max_log", "name_count_diff_log",
    "attr_key_count_min_log", "attr_key_count_max_log", "attr_key_count_diff_log",
    "value_token_count_min_log", "value_token_count_max_log", "value_token_count_diff_log",
    "value_phrase_count_min_log", "value_phrase_count_max_log", "value_phrase_count_diff_log",
    "shared_value_phrase_count_log", "shared_key_value_count_log",
    "shared_name_id_count_log", "shared_compact_id_count_log",
    "attr_presence_asymmetric", "name_id_presence_asymmetric",
]

V13_FEATURE_NAMES = [
    "digit_count_min_log", "digit_count_max_log", "digit_count_diff_log",
    "article_count_min_log", "article_count_max_log", "article_count_diff_log",
    "compact_id_count_min_log", "compact_id_count_max_log", "compact_id_count_diff_log",
    "quantity_count_min_log", "quantity_count_max_log", "quantity_count_diff_log",
    "name_char_min_log", "name_char_max_log", "name_char_diff_log",
    "attr_char_min_log", "attr_char_max_log", "attr_char_diff_log",
    "shared_key_count_log", "shared_exact_value_count_log", "shared_conflict_count_log",
]


EXTRA_FEATURE_NAMES = [
    "idf_name_jaccard", "idf_name_containment",
    "name_extra_min", "name_extra_max",
    "num_jaccard", "num_containment", "num_disjoint", "num_presence_equal",
    "name_char4_jaccard", "name_char4_containment",
    "name_first_match", "name_last_match",
    "idf_max_shared", "idf_max_unshared", "idf_rarest_shared",
    "attr_idf_jaccard", "attr_idf_containment",
    "name_char5_jaccard", "name_char5_containment",
    "size_jaccard", "size_containment", "size_disjoint", "size_presence_equal",
    "color_jaccard", "color_containment", "color_disjoint", "color_presence_equal",
    "gender_jaccard", "gender_containment", "gender_disjoint", "gender_presence_equal",
    "any_class_conflict",
    "qty_jaccard", "qty_containment", "qty_disjoint", "qty_presence_equal", "qty_dim_conflict",
] + V10_FEATURE_NAMES + V11_FEATURE_NAMES + V12_FEATURE_NAMES + V13_FEATURE_NAMES


def _pair_features(left: ProductView, right: ProductView) -> list[float]:
    name = _set_stats(left.name_tokens, right.name_tokens)
    grams = _set_stats(left.name_grams, right.name_grams)
    values = _set_stats(left.value_tokens, right.value_tokens)
    all_tokens = _set_stats(left.all_tokens, right.all_tokens)
    keys = _set_stats(left.attr_keys, right.attr_keys)
    digits = _set_stats(left.digit_tokens, right.digit_tokens)
    articles = _set_stats(left.article_tokens, right.article_tokens)
    brands = _set_stats(left.brand_tokens, right.brand_tokens)

    shared_keys = left.attr_keys & right.attr_keys
    exact_values = 0
    value_similarity = 0.0
    conflicts = 0
    for key in shared_keys:
        l_value = left.attrs[key]
        r_value = right.attrs[key]
        exact_values += int(l_value == r_value)
        similarity = _set_stats(_tokens(l_value), _tokens(r_value))[0]
        value_similarity += similarity
        conflicts += int(similarity == 0.0)
    shared_n = len(shared_keys)

    both_digits = bool(left.digit_tokens and right.digit_tokens)
    both_articles = bool(left.article_tokens and right.article_tokens)
    both_brands = bool(left.brand_tokens and right.brand_tokens)
    name_len_ratio = (
        min(len(left.name), len(right.name)) / max(len(left.name), len(right.name))
        if left.name and right.name else 0.0
    )

    return [
        float(bool(left.name) and left.name == right.name),
        *name,
        grams[0], grams[2],
        name_len_ratio,
        values[0], values[2], values[3],
        all_tokens[0], all_tokens[2], all_tokens[3],
        keys[0], keys[2], keys[3],
        exact_values / shared_n if shared_n else 0.0,
        value_similarity / shared_n if shared_n else 0.0,
        conflicts / shared_n if shared_n else 0.0,
        min(shared_n / 10.0, 1.0),
        digits[0], digits[2], digits[3],
        float(both_digits and not (left.digit_tokens & right.digit_tokens)),
        float(bool(left.digit_tokens) == bool(right.digit_tokens)),
        articles[0], articles[2], articles[3],
        float(both_articles and not (left.article_tokens & right.article_tokens)),
        float(bool(left.article_tokens) == bool(right.article_tokens)),
        brands[0], brands[2],
        float(both_brands and left.brand_tokens == right.brand_tokens),
        float(both_brands and not (left.brand_tokens & right.brand_tokens)),
        float(bool(left.brand_tokens) == bool(right.brand_tokens)),
    ]


FEATURE_NAMES = [
    "name_exact", "name_jaccard", "name_dice", "name_containment", "name_count_ratio",
    "name_char3_jaccard", "name_char3_containment", "name_length_ratio",
    "value_jaccard", "value_containment", "value_count_ratio",
    "all_jaccard", "all_containment", "all_count_ratio",
    "key_jaccard", "key_containment", "key_count_ratio",
    "shared_value_exact", "shared_value_jaccard", "shared_value_conflict", "shared_key_count",
    "digit_jaccard", "digit_containment", "digit_count_ratio", "digit_disjoint", "digit_presence_equal",
    "article_jaccard", "article_containment", "article_count_ratio", "article_disjoint",
    "article_presence_equal", "brand_jaccard", "brand_containment", "brand_exact",
    "brand_disjoint", "brand_presence_equal",
] + EXTRA_FEATURE_NAMES

MODEL_FEATURE_NAMES = FEATURE_NAMES + ["text_tfidf", "name_tfidf"]


def extract_pair_features(matches: pd.DataFrame, items: pd.DataFrame) -> np.ndarray:
    """Return a dense float32 feature matrix in input-pair order."""
    views = build_product_views(items)
    name_idf, name_max_idf = _build_idf(views.values(), attr=False)
    attr_idf, _ = _build_idf(views.values(), attr=True)
    result = np.empty((len(matches), len(FEATURE_NAMES)), dtype=np.float32)
    for row, (id1, id2) in enumerate(matches[["id1", "id2"]].itertuples(index=False, name=None)):
        left = views.get(int(id1), EMPTY_VIEW)
        right = views.get(int(id2), EMPTY_VIEW)
        result[row] = _pair_features(left, right) + _extra_features(
            left, right, name_idf, name_max_idf, attr_idf
        )
    return result


def extract_v10_pair_features(matches: pd.DataFrame, items: pd.DataFrame) -> np.ndarray:
    """Compute only the incremental v10 block for fast feature experiments."""
    lightweight: dict[
        int, tuple[frozenset[str], tuple[dict[str, str], ...]]
    ] = {}
    for item_id, raw_attrs in items[["id", "attributes"]].itertuples(
        index=False, name=None
    ):
        attrs = _parse_attributes(raw_attrs)
        lightweight[int(item_id)] = (
            _compact_identifier_tokens(attrs),
            _family_attributes(attrs),
        )
    empty = (frozenset(), ({}, {}, {}, {}, {}))
    result = np.empty((len(matches), len(V10_FEATURE_NAMES)), dtype=np.float32)
    for row, (id1, id2) in enumerate(
        matches[["id1", "id2"]].itertuples(index=False, name=None)
    ):
        left_ids, left_families = lightweight.get(int(id1), empty)
        right_ids, right_families = lightweight.get(int(id2), empty)
        result[row] = _v10_feature_values(
            left_ids, right_ids, left_families, right_families
        )
    return result


def extract_v11_pair_features(matches: pd.DataFrame, items: pd.DataFrame) -> np.ndarray:
    """Compute only the incremental v11 block for fast feature experiments."""
    lightweight: dict[int, tuple] = {}
    for item_id, name, raw_attrs in items[["id", "name", "attributes"]].itertuples(
        index=False, name=None
    ):
        norm_name = _normalise(name)
        attrs = _parse_attributes(raw_attrs)
        lightweight[int(item_id)] = (
            _compact_name_identifiers(norm_name),
            frozenset(attrs.values()),
            frozenset(f"{key}={value}" for key, value in attrs.items()),
            _semantic_value_groups(attrs),
        )
    empty = (
        frozenset(), frozenset(), frozenset(), (frozenset(),) * 4
    )
    result = np.empty((len(matches), len(V11_FEATURE_NAMES)), dtype=np.float32)
    for row, (id1, id2) in enumerate(
        matches[["id1", "id2"]].itertuples(index=False, name=None)
    ):
        left = lightweight.get(int(id1), empty)
        right = lightweight.get(int(id2), empty)
        result[row] = _v11_feature_values(
            left[0], right[0], left[1], right[1], left[2], right[2], left[3], right[3]
        )
    return result


def extract_v12_pair_features(matches: pd.DataFrame, items: pd.DataFrame) -> np.ndarray:
    """Compute only the incremental v12 density/count block."""
    lightweight: dict[int, tuple] = {}
    for item_id, name, raw_attrs in items[["id", "name", "attributes"]].itertuples(
        index=False, name=None
    ):
        norm_name = _normalise(name)
        name_tokens = _tokens(norm_name)
        attrs = _parse_attributes(raw_attrs)
        value_tokens = frozenset(
            token for value in attrs.values() for token in value.split()
        )
        lightweight[int(item_id)] = (
            name_tokens,
            frozenset(attrs),
            value_tokens,
            frozenset(attrs.values()),
            frozenset(f"{key}={value}" for key, value in attrs.items()),
            _compact_name_identifiers(norm_name),
            _compact_identifier_tokens(attrs),
        )
    empty = (frozenset(),) * 7
    result = np.empty((len(matches), len(V12_FEATURE_NAMES)), dtype=np.float32)
    for row, (id1, id2) in enumerate(
        matches[["id1", "id2"]].itertuples(index=False, name=None)
    ):
        left = lightweight.get(int(id1), empty)
        right = lightweight.get(int(id2), empty)
        result[row] = _v12_feature_values(
            left[0], right[0], left[1], right[1], left[2], right[2],
            left[3], right[3], left[4], right[4], left[5], right[5],
            left[6], right[6],
        )
    return result


def extract_v13_pair_features(matches: pd.DataFrame, items: pd.DataFrame) -> np.ndarray:
    """Compute only the incremental v13 absolute-density block."""
    lightweight: dict[int, tuple] = {}
    for item_id, name, raw_attrs in items[["id", "name", "attributes"]].itertuples(
        index=False, name=None
    ):
        norm_name = _normalise(name)
        attrs = _parse_attributes(raw_attrs)
        all_tokens = _tokens(norm_name) | frozenset(
            token for value in attrs.values() for token in value.split()
        )
        lightweight[int(item_id)] = (
            norm_name,
            attrs,
            frozenset(token for token in all_tokens if any(char.isdigit() for char in token)),
            _selected_values(attrs, ARTICLE_KEYS),
            _compact_identifier_tokens(attrs),
            _extract_quantities(name, raw_attrs),
        )
    empty = ("", {}, frozenset(), frozenset(), frozenset(), frozenset())
    result = np.empty((len(matches), len(V13_FEATURE_NAMES)), dtype=np.float32)
    for row, (id1, id2) in enumerate(
        matches[["id1", "id2"]].itertuples(index=False, name=None)
    ):
        left = lightweight.get(int(id1), empty)
        right = lightweight.get(int(id2), empty)
        result[row] = _v13_feature_values(
            left[0], right[0], left[1], right[1], left[2], right[2],
            left[3], right[3], left[4], right[4], left[5], right[5],
        )
    return result


def extract_model_features(matches: pd.DataFrame, items: pd.DataFrame) -> np.ndarray:
    """Structured features plus two unsupervised lexical cosine signals."""
    from src.data import attach_texts, build_item_text
    from src.scoring import tfidf_scores

    structured = extract_pair_features(matches, items)

    text_items = items[["id", "category"]].copy()
    text_items["text"] = build_item_text(items)
    text_pairs = attach_texts(matches, text_items)
    try:
        text_cosine = tfidf_scores(text_pairs)
    except ValueError:
        # Tiny/empty diagnostic inputs can lose every term at min_df=2.
        text_cosine = np.zeros(len(matches), dtype=np.float32)

    id_to_name = dict(zip(items["id"], items["name"].fillna("").astype(str)))
    name_pairs = matches[["id1", "id2"]].copy()
    name_pairs["text1"] = name_pairs["id1"].map(id_to_name).fillna("")
    name_pairs["text2"] = name_pairs["id2"].map(id_to_name).fillna("")
    try:
        name_cosine = tfidf_scores(name_pairs, ngram_range=(1, 2), max_features=300_000)
    except ValueError:
        name_cosine = np.zeros(len(matches), dtype=np.float32)

    return np.column_stack((structured, text_cosine, name_cosine)).astype(np.float32)
