"""Размеры и транслитерация: два пробела, найденных разбором ошибок.

**Размеры.** В карточках одно и то же пишут по-разному: «40x32 см», «40*32», «40х32»
(кириллическое х), «(4032 см)» — последнее получается, когда разделитель потерялся при
выгрузке. Токенами это разные строки, числами — тоже, потому что «4032» слипшееся. Пример
из разбора: `салфетка ultra chamois (4032 см)` и `салфетка ultra chamois (42*3...)` —
разметка считает их одним товаром, а у нас не совпадает ничего.

**Транслитерация.** «nike» и «найк», «bosch» и «бош» — один бренд. Свёртка гомоглифов в
`attr_features.normalize` решает только случай визуально одинаковых букв, а здесь написание
разное по существу.
"""
from __future__ import annotations

import re

from src.attr_features import normalize

# Разделители размеров: латинская x, кириллическая х, звёздочка, знак умножения.
DIMENSION = re.compile(r"(\d{1,4}(?:[.,]\d+)?)\s*[x х×*]\s*(\d{1,4}(?:[.,]\d+)?)"
                       r"(?:\s*[x х×*]\s*(\d{1,4}(?:[.,]\d+)?))?")
# Кириллица в латиницу по звучанию: длинные сочетания идут первыми, иначе «щ» разберётся
# как «ш»+«ч», а «ю» как «у».
TRANSLIT: tuple[tuple[str, str], ...] = (
    ("щ", "sch"), ("ш", "sh"), ("ч", "ch"), ("ц", "ts"), ("ж", "zh"), ("ю", "yu"),
    ("я", "ya"), ("ё", "e"), ("э", "e"), ("ы", "y"), ("й", "y"), ("а", "a"), ("б", "b"),
    ("в", "v"), ("г", "g"), ("д", "d"), ("е", "e"), ("з", "z"), ("и", "i"), ("к", "k"),
    ("л", "l"), ("м", "m"), ("н", "n"), ("о", "o"), ("п", "p"), ("р", "r"), ("с", "s"),
    ("т", "t"), ("у", "u"), ("ф", "f"), ("х", "h"), ("ъ", ""), ("ь", ""),
)


def transliterate(text: str) -> str:
    """Кириллица к латинице по звучанию; латиница остаётся как есть.

    Свёртка гомоглифов здесь недопустима: она переводит латинскую `c` в кириллическую, и
    «bosch» превращается в «bossh». Достаточно нижнего регистра.
    """
    result = str(text).lower()
    for source, target in TRANSLIT:
        result = result.replace(source, target)
    return re.sub(r"[^a-z0-9]", "", result)


def dimensions(text: str) -> frozenset[tuple[float, ...]]:
    """Наборы размеров, приведённые к возрастающему порядку.

    Порядок сторон в записи произволен («40x32» и «32x40» — один и тот же лист), поэтому
    числа сортируются: иначе одинаковые размеры не совпадут.
    """
    found = set()
    for match in DIMENSION.finditer(normalize(text)):
        values = tuple(float(v.replace(",", ".")) for v in match.groups() if v)
        if len(values) >= 2:
            found.add(tuple(sorted(values)))
    return frozenset(found)


def compare_dimensions(left: frozenset, right: frozenset) -> dict[str, float]:
    shared = left & right
    return {
        "dim_agree": float(bool(shared)),
        "dim_conflict": float(bool(left) and bool(right) and not shared),
        "dim_missing": float(bool(left) != bool(right)),
        "dim_jaccard": len(shared) / len(left | right) if (left | right) else 0.0,
    }


def compare_translit(left: frozenset[str], right: frozenset[str]) -> dict[str, float]:
    """Близость брендов после приведения к латинице.

    Точное равенство здесь не годится: «найк» даёт «nayk», а не «nike». Звучание совпадает,
    написание — нет, поэтому сравниваем по близости строк.
    """
    from src.string_features import jaro_winkler

    a = [x for x in (transliterate(v) for v in left) if x]
    b = [x for x in (transliterate(v) for v in right) if x]
    if not a or not b:
        return {"translit_agree": 0.0, "translit_rescued": 0.0, "translit_best": 0.0}
    best = max(jaro_winkler(x, y) for x in a for y in b)
    return {
        "translit_agree": float(best >= 0.85),
        "translit_rescued": float(best >= 0.85 and not (left & right)),
        "translit_best": float(best),
    }


DIM_FEATURE_NAMES: tuple[str, ...] = (
    "dim_agree", "dim_conflict", "dim_missing", "dim_jaccard",
    "translit_agree", "translit_rescued", "translit_best",
)
