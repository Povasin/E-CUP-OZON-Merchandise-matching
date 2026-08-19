"""Признаки по названиям товаров: символьные n-граммы, латиница отдельно, префиксы, BM25.

Что делают призёры прошлых E-CUP и чего у нас не было. Разбор решения, занявшего 4-е место
в 2024 году, называет решающими три вещи: пересечение n-грамм **только по латинице и
цифрам** (артикулы и коды моделей перестают тонуть среди русских слов), длину общего
префикса и BM25 вместо простого косинуса.

Символьные n-граммы берутся из схемы Ристоски: для коротких строк они устойчивее
пословных, потому что переживают опечатки, разную пунктуацию и слитное написание —
«bt-05» против «bt 05» против «bt05».

Все признаки симметричны: порядок сторон в паре произволен, поэтому вместо «левое минус
правое» берутся минимум, максимум и отношение.
"""
from __future__ import annotations

import math
import re
from typing import NamedTuple

from src.attr_features import normalize

LATIN_DIGIT = re.compile(r"[a-z0-9][a-z0-9\-_/.]*")
TOKEN = re.compile(r"[a-zа-яё0-9]+")


class Name(NamedTuple):
    """Разобранное название. Считается по разу на товар."""
    text: str
    tokens: tuple[str, ...]
    token_set: frozenset[str]
    latin: frozenset[str]
    grams3: frozenset[str]
    grams4: frozenset[str]


def char_grams(text: str, size: int) -> frozenset[str]:
    squeezed = re.sub(r"\s+", " ", text)
    return frozenset(squeezed[i:i + size] for i in range(max(0, len(squeezed) - size + 1)))


def parse_name(name: str) -> Name:
    text = normalize(name)
    tokens = tuple(TOKEN.findall(text))
    # Свёртка гомоглифов в `normalize` переводит латиницу в кириллицу, поэтому латинские
    # куски ищем в исходной строке до свёртки — иначе артикулы перестанут выделяться.
    latin = frozenset(t for t in LATIN_DIGIT.findall(str(name).lower()) if len(t) >= 2)
    return Name(text, tokens, frozenset(tokens), latin,
                char_grams(text, 3), char_grams(text, 4))


def jaccard(left: frozenset, right: frozenset) -> float:
    if not left and not right:
        return 0.0
    union = len(left | right)
    return len(left & right) / union if union else 0.0


def containment(left: frozenset, right: frozenset) -> float:
    """Доля меньшего множества, покрытая большим: устойчива к разнице в подробности."""
    smaller = min(len(left), len(right))
    return len(left & right) / smaller if smaller else 0.0


def common_prefix(left: str, right: str) -> int:
    limit = min(len(left), len(right))
    position = 0
    while position < limit and left[position] == right[position]:
        position += 1
    return position


def bm25(left: Name, right: Name, idf: dict[str, float],
         average_length: float, k1: float = 1.5, b: float = 0.75) -> float:
    """BM25 одного названия относительно другого как документа.

    В отличие от косинуса, насыщается по частоте и штрафует длинные документы, поэтому
    редкое совпавшее слово весит больше, чем несколько частых.
    """
    if not right.tokens:
        return 0.0
    counts: dict[str, int] = {}
    for token in right.tokens:
        counts[token] = counts.get(token, 0) + 1
    length = len(right.tokens)
    total = 0.0
    for token in left.token_set:
        frequency = counts.get(token)
        if not frequency:
            continue
        norm = frequency * (k1 + 1) / (
            frequency + k1 * (1 - b + b * length / average_length)
        )
        total += idf.get(token, 0.0) * norm
    return total


def build_idf(names: list[Name]) -> tuple[dict[str, float], float]:
    """IDF и средняя длина по текущему набору товаров.

    Считается по пришедшим данным, а не берётся из обучения: в тесте другие товары, и
    частоты слов там свои.
    """
    document_frequency: dict[str, int] = {}
    total_length = 0
    for name in names:
        total_length += len(name.tokens)
        for token in name.token_set:
            document_frequency[token] = document_frequency.get(token, 0) + 1
    count = max(len(names), 1)
    idf = {
        token: math.log(1 + (count - frequency + 0.5) / (frequency + 0.5))
        for token, frequency in document_frequency.items()
    }
    return idf, max(total_length / count, 1.0)


def compare_names(left: Name, right: Name, idf: dict[str, float],
                  average_length: float) -> dict[str, float]:
    prefix = common_prefix(left.text, right.text)
    lengths = (len(left.text), len(right.text))
    score_left = bm25(left, right, idf, average_length)
    score_right = bm25(right, left, idf, average_length)

    smaller, larger = ((left, right) if len(left.token_set) <= len(right.token_set)
                       else (right, left))
    # Строгое вложение: «брюки acoola» целиком содержится в «брюки спортивные acoola».
    # Замерено на обучающей части: доля совпадений 42.2% против 21.6% — сильнейший из
    # найденных перебором кандидатов.
    subset = bool(smaller.token_set) and smaller.token_set <= larger.token_set
    extra = len(larger.token_set - smaller.token_set)
    first_match = (bool(left.tokens) and bool(right.tokens)
                   and left.tokens[0] == right.tokens[0])
    kit = {"набор", "комплект", "компл"}
    left_kit, right_kit = bool(kit & left.token_set), bool(kit & right.token_set)

    short = len(left.token_set) < 4 and len(right.token_set) < 4
    return {
        # Оба названия коротки — зацепиться не за что: 6.7% совпадений против 26.4%.
        "both_short": float(short),
        "min_tokens": float(min(len(left.token_set), len(right.token_set))),
        # Полное совпадение латинской части сильнее её доли: 36.4% против 23.4%.
        "latin_identical": float(bool(left.latin) and left.latin == right.latin),
        "subset_strict": float(subset),
        "subset_extra": float(extra) if subset else 0.0,
        "first_token_match": float(first_match),
        "last_token_match": float(bool(left.tokens) and bool(right.tokens)
                                  and left.tokens[-1] == right.tokens[-1]),
        "kit_both": float(left_kit and right_kit),
        "kit_one": float(left_kit != right_kit),
        "name_gram3_jaccard": jaccard(left.grams3, right.grams3),
        "name_gram3_containment": containment(left.grams3, right.grams3),
        "name_gram4_jaccard": jaccard(left.grams4, right.grams4),
        "name_gram4_containment": containment(left.grams4, right.grams4),
        # Латиница и цифры отдельно: там живут артикулы и коды моделей.
        "latin_jaccard": jaccard(left.latin, right.latin),
        "latin_containment": containment(left.latin, right.latin),
        "latin_shared": float(len(left.latin & right.latin)),
        "latin_only_one": float(bool(left.latin) != bool(right.latin)),
        "latin_disjoint": float(bool(left.latin) and bool(right.latin)
                                and not (left.latin & right.latin)),
        "prefix_len": float(prefix),
        "prefix_ratio": prefix / max(min(lengths), 1),
        "token_jaccard": jaccard(left.token_set, right.token_set),
        "token_containment": containment(left.token_set, right.token_set),
        "bm25_min": min(score_left, score_right),
        "bm25_max": max(score_left, score_right),
        "len_ratio": min(lengths) / max(max(lengths), 1),
        "token_count_ratio": min(len(left.tokens), len(right.tokens))
                             / max(len(left.tokens), len(right.tokens), 1),
    }


EMPTY_NAME = parse_name("")
NAME_FEATURE_NAMES: tuple[str, ...] = tuple(compare_names(EMPTY_NAME, EMPTY_NAME, {}, 1.0).keys())
