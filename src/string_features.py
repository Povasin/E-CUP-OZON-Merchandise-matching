"""Строковые метрики сходства названий, нормированные на обе длины.

Разбор решения, занявшего 4-е место в Kaggle Foursquare Location Matching — задача с той же
структурой, что наша: дедупликация записей, в тесте неизвестные сущности, жёсткий лимит
инференса. Решающими там названы наибольшая общая подстрока и подпоследовательность,
**нормированные на длины обеих строк сразу**, число и суммарная длина общих кусков от
четырёх символов, и Jaro-Winkler.

Нормировка на обе длины принципиальна. Сырая длина общего куска растёт вместе с длиной
названий, поэтому «Ботинки лыжные Spine Baby 103» и «Ботинки» дадут большое совпадение при
совершенно разных товарах. Отношение к короткой строке показывает, целиком ли она вошла в
длинную; отношение к длинной — какую долю та объяснила.

Считается через `difflib` из стандартной библиотеки: 15 700 пар в секунду, то есть около
25 секунд на весь закрытый тест, и никаких зависимостей в образе.
"""
from __future__ import annotations

from difflib import SequenceMatcher

from src.attr_features import normalize

# Длинные названия обрезаются: полезное различие товаров сидит в начале, а стоимость
# сравнения растёт как произведение длин.
LIMIT = 96


def jaro_winkler(left: str, right: str, prefix_scale: float = 0.1) -> float:
    if not left or not right:
        return 0.0
    if left == right:
        return 1.0
    reach = max(len(left), len(right)) // 2 - 1
    if reach < 0:
        reach = 0
    left_flags = [False] * len(left)
    right_flags = [False] * len(right)

    matches = 0
    for i, char in enumerate(left):
        for j in range(max(0, i - reach), min(len(right), i + reach + 1)):
            if not right_flags[j] and right[j] == char:
                left_flags[i] = right_flags[j] = True
                matches += 1
                break
    if not matches:
        return 0.0

    # Перестановки: совпавшие символы, стоящие в разном порядке, считаются за половину.
    transpositions = 0
    position = 0
    for i, flag in enumerate(left_flags):
        if not flag:
            continue
        while not right_flags[position]:
            position += 1
        if left[i] != right[position]:
            transpositions += 1
        position += 1
    transpositions //= 2

    jaro = (matches / len(left) + matches / len(right)
            + (matches - transpositions) / matches) / 3.0
    common = 0
    for a, b in zip(left[:4], right[:4]):
        if a != b:
            break
        common += 1
    return jaro + common * prefix_scale * (1 - jaro)


def compare_strings(left: str, right: str) -> dict[str, float]:
    a, b = normalize(left)[:LIMIT], normalize(right)[:LIMIT]
    if not a or not b:
        return dict.fromkeys(STRING_FEATURE_NAMES, 0.0)

    # `SequenceMatcher` несимметричен по аргументам: у одной и той же пары `matched_total`
    # расходится на 30.8% случаев в зависимости от того, какая сторона записана первой во
    # входном файле. Порядок сторон — произвол данных, а не свойство пары, поэтому он
    # приводится к каноническому.
    if b < a:
        a, b = b, a
    blocks = SequenceMatcher(None, a, b, autojunk=False).get_matching_blocks()
    sizes = [block.size for block in blocks if block.size]
    longest = max(sizes, default=0)
    matched = sum(sizes)
    substantial = [size for size in sizes if size >= 4]

    shorter, longer = min(len(a), len(b)), max(len(a), len(b))
    return {
        "lcs_longest": float(longest),
        "lcs_over_short": longest / shorter,
        "lcs_over_long": longest / longer,
        "matched_total": float(matched),
        "matched_over_short": matched / shorter,
        "matched_over_long": matched / longer,
        "blocks_long_count": float(len(substantial)),
        "blocks_long_total": float(sum(substantial)),
        "blocks_long_over_short": sum(substantial) / shorter,
        "jaro_winkler": jaro_winkler(a, b),
    }


STRING_FEATURE_NAMES: tuple[str, ...] = (
    "lcs_longest", "lcs_over_short", "lcs_over_long",
    "matched_total", "matched_over_short", "matched_over_long",
    "blocks_long_count", "blocks_long_total", "blocks_long_over_short",
    "jaro_winkler",
)
