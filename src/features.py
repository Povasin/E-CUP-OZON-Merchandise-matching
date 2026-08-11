"""Fast, model-free pair features for supervised product matching.

The features describe relations between two cards rather than memorising concrete
products.  This is important for the competition test, where every product is new.
"""
from __future__ import annotations

import json
import re
from dataclasses import dataclass

import numpy as np
import pandas as pd


TOKEN_RE = re.compile(r"[0-9a-zа-яё]+", re.IGNORECASE)
ARTICLE_KEYS = ("артикул", "модель", "партномер", "oem", "код товара", "sku")
BRAND_KEYS = ("бренд", "производитель", "марка")


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


@dataclass(frozen=True, slots=True)
class ProductView:
    name: str
    name_tokens: frozenset[str]
    name_grams: frozenset[str]
    attrs: dict[str, str]
    attr_keys: frozenset[str]
    value_tokens: frozenset[str]
    all_tokens: frozenset[str]
    digit_tokens: frozenset[str]
    article_tokens: frozenset[str]
    brand_tokens: frozenset[str]


EMPTY_VIEW = ProductView("", frozenset(), frozenset(), {}, frozenset(), frozenset(),
                         frozenset(), frozenset(), frozenset(), frozenset())


def _make_view(name: object, raw_attrs: object) -> ProductView:
    norm_name = _normalise(name)
    name_tokens = _tokens(norm_name)
    attrs = _parse_attributes(raw_attrs)
    value_tokens = frozenset(token for value in attrs.values() for token in value.split())
    all_tokens = name_tokens | value_tokens
    digit_tokens = frozenset(token for token in all_tokens if any(ch.isdigit() for ch in token))
    return ProductView(
        name=norm_name,
        name_tokens=name_tokens,
        name_grams=_char_ngrams(norm_name),
        attrs=attrs,
        attr_keys=frozenset(attrs),
        value_tokens=value_tokens,
        all_tokens=all_tokens,
        digit_tokens=digit_tokens,
        article_tokens=_selected_values(attrs, ARTICLE_KEYS),
        brand_tokens=_selected_values(attrs, BRAND_KEYS),
    )


def build_product_views(items: pd.DataFrame) -> dict[int, ProductView]:
    """Build compact parsed views once per product id."""
    return {
        int(item_id): _make_view(name, attrs)
        for item_id, name, attrs in items[["id", "name", "attributes"]].itertuples(index=False, name=None)
    }


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
]

MODEL_FEATURE_NAMES = FEATURE_NAMES + ["text_tfidf", "name_tfidf"]


def extract_pair_features(matches: pd.DataFrame, items: pd.DataFrame) -> np.ndarray:
    """Return a dense float32 feature matrix in input-pair order."""
    views = build_product_views(items)
    result = np.empty((len(matches), len(FEATURE_NAMES)), dtype=np.float32)
    for row, (id1, id2) in enumerate(matches[["id1", "id2"]].itertuples(index=False, name=None)):
        result[row] = _pair_features(views.get(int(id1), EMPTY_VIEW), views.get(int(id2), EMPTY_VIEW))
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
