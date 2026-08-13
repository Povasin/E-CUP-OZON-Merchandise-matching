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
    name_grams4: frozenset[str]
    name_grams5: frozenset[str]
    attrs: dict[str, str]
    attr_keys: frozenset[str]
    value_tokens: frozenset[str]
    all_tokens: frozenset[str]
    digit_tokens: frozenset[str]
    num_values: frozenset[float]
    article_tokens: frozenset[str]
    brand_tokens: frozenset[str]


EMPTY_VIEW = ProductView("", frozenset(), frozenset(), frozenset(), frozenset(), {},
                         frozenset(), frozenset(), frozenset(), frozenset(), frozenset(),
                         frozenset(), frozenset())


def _make_view(name: object, raw_attrs: object) -> ProductView:
    norm_name = _normalise(name)
    name_tokens = _tokens(norm_name)
    attrs = _parse_attributes(raw_attrs)
    value_tokens = frozenset(token for value in attrs.values() for token in value.split())
    all_tokens = name_tokens | value_tokens
    digit_tokens = frozenset(token for token in all_tokens if any(ch.isdigit() for ch in token))
    num_values = frozenset(float(token) for token in all_tokens if token.isdigit())
    return ProductView(
        name=norm_name,
        name_tokens=name_tokens,
        name_grams=_char_ngrams(norm_name),
        name_grams4=_char_ngrams(norm_name, 4),
        name_grams5=_char_ngrams(norm_name, 5),
        attrs=attrs,
        attr_keys=frozenset(attrs),
        value_tokens=value_tokens,
        all_tokens=all_tokens,
        digit_tokens=digit_tokens,
        num_values=num_values,
        article_tokens=_selected_values(attrs, ARTICLE_KEYS),
        brand_tokens=_selected_values(attrs, BRAND_KEYS),
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

    return [
        idf_jaccard, idf_containment,
        extra_min, extra_max,
        num_jaccard, num_containment, num_disjoint, num_presence_equal,
        grams4[0], grams4[2],
        first_match, last_match,
        max_shared, max_unshared, rarest_shared,
        attr_jaccard, attr_containment,
        grams5[0], grams5[2],
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
]


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
