"""Признаки, понимающие содержимое карточки: величины, коды, слоты атрибутов.

Наши прежние 168 признаков лексические: пересечения токенов, косинусы, длины. Они не
различают «500 мл» и «0.5 л» и не знают, что «23797п-1» — артикул, а «Pro» — часть
названия линейки. Между тем в разборе ошибок именно это и всплывало: `spine baby 103`
против `spine sns baby (103)` — одна модель, разные размеры, метка 0.

Почему это важнее очередного дообучения: на бенчмарке WDC Products при переходе к
невиданным товарам кросс-энкодер теряет 11 пунктов F1, а признаковый метод — ноль. В нашем
тесте товары другие по условию соревнования.

Разбор карточки (`parse`) отделён от сравнения пары (`compare`) намеренно: товар в парах
встречается примерно один раз, но разбор — это регулярки по строке в несколько сотен
символов, а сравнение — операции над готовыми множествами. Раздельно разбор считается один
раз на товар и распараллеливается, а лимит времени у решения жёсткий и быстродействие
оценивается жюри наравне с метрикой.
"""
from __future__ import annotations

import json
import re
import unicodedata
from functools import lru_cache
from pathlib import Path
from typing import NamedTuple

# Приведение к базовой единице: (суффикс, множитель, канон). Цифровые объёмы и частоты
# идут по 1024 и 1000 соответственно: «2 ГБ» — это 2048 МБ, но «2 ГГц» — 2000 МГц.
UNITS: tuple[tuple[str, float, str], ...] = (
    ("мкг", 1e-3, "г"), ("мг", 1.0, "г"), ("кг", 1e6, "г"), ("гр", 1e3, "г"),
    ("г", 1e3, "г"), ("т", 1e9, "г"),
    ("мл", 1.0, "мл"), ("л", 1e3, "мл"), ("см3", 1.0, "мл"),
    ("мм", 1.0, "мм"), ("см", 10.0, "мм"), ("дм", 100.0, "мм"), ("м", 1e3, "мм"),
    ("тб", 1024 * 1024, "мб"), ("гб", 1024.0, "мб"), ("мб", 1.0, "мб"), ("кб", 1 / 1024, "мб"),
    ("ггц", 1e3, "мгц"), ("мгц", 1.0, "мгц"), ("кгц", 1e-3, "мгц"),
    ("мач", 1.0, "мач"), ("ач", 1e3, "мач"),
    ("вт", 1.0, "вт"), ("квт", 1e3, "вт"),
    ("шт", 1.0, "шт"), ("уп", 1.0, "уп"),
)
UNIT_MAP = {suffix: (factor, canonical) for suffix, factor, canonical in UNITS}

NUMBER = re.compile(r"(\d+(?:[.,]\d+)?)\s*([а-яёa-z]{1,3}\d?)")
# Буквенно-цифровой код: «23797п», «bt-05», «z3max».
MIXED_CODE = re.compile(r"[a-zа-яё0-9][a-zа-яё0-9\-_/]{2,}[a-zа-яё0-9]")
# Чисто цифровой номер модели: «103» в «spine baby 103». Три знака и больше, иначе
# наберём размеры и года. Единицы измерения отсекаются отдельно.
DIGIT_CODE = re.compile(r"(?<![\d.,])\d{3,}(?![\d.,])")
WORD = re.compile(r"[a-zа-яё]+")
NAME_TOKEN = re.compile(r"[a-zа-яё]{2,}|\d+[a-zа-яё]*")

# Слова-различители внутри одной линейки: их несовпадение означает разные товары.
MARKERS = frozenset({
    "pro", "max", "plus", "mini", "ultra", "lite", "premium", "про", "макс", "мини",
    "плюс", "xl", "xxl", "xs", "new", "gen", "smart", "eco", "duo", "compact",
})
# Латиница, неотличимая на вид от кириллицы: в названиях маркетплейса их мешают постоянно.
HOMOGLYPHS = str.maketrans("aecopxyABCEHKMOPTXYёЁ", "аесорхуАВСЕНКМОРТХУее")


# Ключей атрибутов почти три тысячи, но смысловых слотов мало, и один смысл размазан по
# нескольким ключам: идентификатор товара лежит в «артикул», «артикул производителя»,
# «партномер (артикул производителя)» и «oem-номер» — вместе это заметная доля карточек.
# Без сведения в слоты бренд одной карточки сравнивается с материалом другой.
# Порядок важен: ключ попадает в первый подошедший слот, поэтому «страна-изготовитель»
# должна проверяться до «изготовитель».
ATTR_SLOTS: tuple[tuple[str, tuple[str, ...]], ...] = (
    ("country", ("страна",)),
    ("article", ("артикул", "партномер", "парт-номер", "oem", "код производителя")),
    ("brand", ("бренд", "торговая марка", "производитель", "изготовитель")),
    ("model", ("модель", "серия")),
    ("color", ("цвет",)),
    ("size", ("размер", "длина", "ширина", "высота", "глубина", "диаметр")),
    ("weight", ("вес", "масса", "объем", "объём")),
    ("material", ("материал", "состав")),
    ("quantity", ("количество", "единиц", "штук", "комплект")),
    ("gender", ("пол",)),
    ("type", ("тип", "вид", "назначение")),
)
PUNCTUATION = re.compile(r"[^a-zа-яё0-9]+")


# Слова-различители добыты из обучающей части, а не выдуманы: среди пар с почти
# одинаковыми названиями считается, насколько одностороннее присутствие слова повышает
# шанс «разные товары». Так находятся «женское», «2шт», «кварцевые», голые числа размеров —
# то есть варианты внутри одной линейки. Заданный руками список из Pro/Max/Plus не содержал
# ни одного из них и давал почти нулевой сигнал.
ANTI_WORDS_PATH = Path(__file__).resolve().parent.parent / "models" / "anti_words.json"


@lru_cache(maxsize=1)
def anti_words() -> dict[str, float]:
    try:
        return json.loads(ANTI_WORDS_PATH.read_text())
    except OSError:
        return {}


# Хвост кода часто обозначает вариант, а не другой товар: «evi.292f-bk» и «evi.292-bg» —
# одна модель в разных цветах, и разметка считает их совпадением. Измерено на обучающей
# части: среди пар с несовпадающими кодами доля совпадений 9.4%, а если совпадает основа
# кода — 16.4%. Без этого признак `code_conflict` толкает пятую часть своих срабатываний
# в неверную сторону.
CODE_BASE = re.compile(r"^([a-zа-яё]*\d+)")
# Окончания русских слов: «крест» и «крестик», «брюки» и «брюк» должны сходиться. Список
# короткий и грубый намеренно — полноценная лемматизация потребовала бы словаря, а он в
# образ не влезет и замедлит инференс.
# Только падежные окончания. Словообразовательные суффиксы («ка», «ик») сюда не входят:
# смешав их с окончаниями, получаем непоследовательность — «индейка» срезалась бы до
# «индей», а «индейкой» до «индейк», и одно слово переставало сходиться с самим собой.
# Родство вроде «крест»/«крестик» ловится символьными n-граммами и общим префиксом.
ENDINGS = ("ами", "ями", "ого", "ому", "ыми", "ими", "ей", "ой", "ый", "ий", "ая", "яя",
           "ое", "ее", "ые", "ие", "ов", "ев", "ам", "ям", "ах", "ях", "ую", "юю", "ом",
           "ем", "а", "я", "ы", "и", "о", "е", "у", "ю")


def stem(word: str) -> str:
    """Отсечение падежного окончания. Короткие слова не трогаем — там резать нечего."""
    for ending in ENDINGS:
        if len(word) - len(ending) >= 4 and word.endswith(ending):
            return word[: -len(ending)]
    return word


def code_base(code: str) -> str:
    """Основа кода без хвоста-варианта: «evi.292f-bk» -> «evi292», «ps-42» -> «ps4»."""
    squeezed = PUNCTUATION.sub("", code)
    found = CODE_BASE.match(squeezed)
    return found.group(1) if found else squeezed[:6]


class Card(NamedTuple):
    """Разобранная карточка: всё, что нужно для сравнения с любой другой."""
    quantities: dict[str, frozenset[float]]
    codes: frozenset[str]
    numbers: frozenset[int]
    markers: frozenset[str]
    slots: dict[str, frozenset[str]]
    name_tokens: frozenset[str]
    stems: frozenset[str]
    code_bases: frozenset[str]
    articles: frozenset[str]
    flat_name: str


def normalize(text: str) -> str:
    """Нижний регистр, единая форма Юникода, кириллица вместо похожей латиницы."""
    return unicodedata.normalize("NFKC", str(text)).lower().translate(HOMOGLYPHS)


def slot_of(key: str) -> str | None:
    lowered = normalize(key)
    for slot, terms in ATTR_SLOTS:
        if any(term in lowered for term in terms):
            return slot
    return None


def parse_attributes(attributes) -> dict[str, frozenset[str]]:
    """Значения атрибутов, разложенные по смысловым слотам и приведённые к сравнимому виду."""
    if attributes is None:
        return {}
    if isinstance(attributes, (str, bytes)):
        try:
            attributes = json.loads(attributes)
        except (ValueError, TypeError):
            return {}
    if not isinstance(attributes, dict):
        try:
            attributes = dict(attributes)
        except Exception:
            return {}
    collected: dict[str, set[str]] = {}
    for key, value in attributes.items():
        slot = slot_of(str(key))
        if slot is None:
            continue
        for part in (value if isinstance(value, (list, tuple, set)) else [value]):
            # Пунктуация в артикулах расставлена как попало: «23797п-1» и «23797П 1» —
            # один номер, поэтому сравниваем по буквам и цифрам.
            cleaned = PUNCTUATION.sub("", normalize(part))
            if cleaned:
                collected.setdefault(slot, set()).add(cleaned)
    return {slot: frozenset(values) for slot, values in collected.items()}


def parse(text: str, attributes=None, name: str = "") -> Card:
    """Разбор одной карточки. Вызывается по разу на товар, не на пару."""
    lowered = normalize(text)

    values: dict[str, set[float]] = {}
    consumed: set[str] = set()
    for value, unit in NUMBER.findall(lowered):
        entry = UNIT_MAP.get(unit)
        if entry is None:
            continue
        factor, canonical = entry
        values.setdefault(canonical, set()).add(
            round(float(value.replace(",", ".")) * factor, 4)
        )
        consumed.add(value)

    codes = {
        token for token in MIXED_CODE.findall(lowered)
        if len(token) >= 4 and any(c.isdigit() for c in token) and any(c.isalpha() for c in token)
    }
    # Числа, уже объяснённые единицей измерения, кодами модели не считаются: «500» в
    # «500 мл» — это объём, а не номер.
    numbers = {int(t) for t in DIGIT_CODE.findall(lowered) if t not in consumed}
    name_tokens = frozenset(NAME_TOKEN.findall(normalize(name or text)))
    slots = parse_attributes(attributes)

    return Card(
        quantities={unit: frozenset(v) for unit, v in values.items()},
        codes=frozenset(codes),
        numbers=frozenset(numbers),
        markers=frozenset(w for w in WORD.findall(lowered) if w in MARKERS),
        slots=slots,
        name_tokens=name_tokens,
        stems=frozenset(stem(w) for w in name_tokens),
        code_bases=frozenset(code_base(c) for c in codes),
        articles=frozenset(v for v in slots.get("article", ()) if len(v) >= 5),
        # Название без пунктуации: артикул противоположной стороны ищется в нём подстрокой,
        # потому что в названии он пишется слитно или через дефис как попало.
        flat_name=PUNCTUATION.sub("", normalize(name or text)),
    )


def compare(left: Card, right: Card) -> dict[str, float]:
    """Признаки пары. Симметричны: порядок сторон в паре произволен."""
    shared_units = left.quantities.keys() & right.quantities.keys()
    agree = sum(1 for u in shared_units if left.quantities[u] & right.quantities[u])
    conflict = len(shared_units) - agree

    codes_shared = left.codes & right.codes
    codes_all = left.codes | right.codes
    numbers_shared = left.numbers & right.numbers
    numbers_all = left.numbers | right.numbers
    markers_diff = left.markers ^ right.markers

    return {
        # Величины сравнимы только там, где обе стороны указали одну и ту же единицу.
        "qty_units_shared": float(len(shared_units)),
        "qty_agree": float(agree),
        "qty_conflict": float(conflict),
        "qty_conflict_only": float(conflict > 0 and agree == 0),
        "qty_one_sided": float(len(left.quantities.keys() ^ right.quantities.keys())),
        # Совпавший код — сильный положительный сигнал, расхождение при наличии кодов у
        # обеих сторон — сильный отрицательный.
        "code_shared": float(len(codes_shared)),
        "code_conflict": float(bool(left.codes) and bool(right.codes) and not codes_shared),
        "code_jaccard": len(codes_shared) / len(codes_all) if codes_all else 0.0,
        "code_one_sided": float(bool(left.codes) != bool(right.codes)),
        "num_shared": float(len(numbers_shared)),
        "num_conflict": float(bool(left.numbers) and bool(right.numbers) and not numbers_shared),
        "num_jaccard": len(numbers_shared) / len(numbers_all) if numbers_all else 0.0,
        # Различители линейки: важна симметрическая разность, а не пересечение.
        "marker_diff": float(len(markers_diff)),
        "marker_shared": float(len(left.markers & right.markers)),
        **slot_features(left.slots, right.slots),
        **anti_word_features(left.name_tokens, right.name_tokens),
        # Ровно одно число у каждой стороны: если они разошлись, это почти наверняка
        # разные варианты. Замерено: доля совпадений 6.4% против 27.1%.
        "single_number_agree": float(len(left.numbers) == 1 == len(right.numbers)
                                     and left.numbers == right.numbers),
        "single_number_clash": float(len(left.numbers) == 1 == len(right.numbers)
                                     and left.numbers != right.numbers),
        # Полное совпадение всех величин с единицами: 42.4% против 22.4%.
        "qty_identical": float(bool(left.quantities) and left.quantities == right.quantities),
        # Артикул одной стороны, найденный в названии другой. Покрытие втрое больше, чем у
        # совпадения слотов (1.34% против 0.40%) при точности 84% против 97% — потому что
        # артикул в атрибутах заполняют не все, а в название его пишут часто.
        **article_features(left, right),
        # Основа кода: хвост часто обозначает цвет или исполнение, а не другой товар.
        "code_base_shared": float(len(left.code_bases & right.code_bases)),
        "code_base_rescue": float(bool(left.codes) and bool(right.codes)
                                  and not (left.codes & right.codes)
                                  and bool(left.code_bases & right.code_bases)),
        # Совпадение после отсечения окончаний: «крест» и «крестик», «брюки» и «брюк».
        "stem_jaccard": (len(left.stems & right.stems) / len(left.stems | right.stems)
                         if (left.stems | right.stems) else 0.0),
        "stem_rescue": float(len(left.stems & right.stems)
                             - len(left.name_tokens & right.name_tokens)),
    }


def anti_word_features(left: frozenset[str], right: frozenset[str]) -> dict[str, float]:
    """Насколько расхождение названий состоит из слов, означающих другой вариант товара."""
    table = anti_words()
    weights = [table[w] for w in (left ^ right) if w in table]
    return {
        "anti_sum": float(sum(weights)),
        "anti_max": float(max(weights)) if weights else 0.0,
        "anti_count": float(len(weights)),
    }


def article_features(left: Card, right: Card) -> dict[str, float]:
    exact = bool(left.articles & right.articles)
    cross = (any(v in right.flat_name for v in left.articles)
             or any(v in left.flat_name for v in right.articles))
    both = bool(left.articles) and bool(right.articles)
    long_numbers = {n for n in left.numbers if n >= 10000} & {n for n in right.numbers if n >= 10000}
    # Сочетания «бренд и код» разделяют лучше, чем каждое по отдельности: совпали оба —
    # 42.1% совпадений, бренд тот же, а коды разные — 10.0% при базовых 25.5%.
    brand_same = bool(left.slots.get("brand", frozenset()) & right.slots.get("brand", frozenset()))
    codes_same = bool(left.codes & right.codes)
    codes_both = bool(left.codes) and bool(right.codes)
    model_cross = (any(v in right.flat_name for v in left.slots.get("model", ()) if len(v) >= 4)
                   or any(v in left.flat_name for v in right.slots.get("model", ()) if len(v) >= 4))
    return {
        "brand_code_agree": float(brand_same and codes_same),
        "brand_code_clash": float(brand_same and codes_both and not codes_same),
        "model_cross": float(model_cross),
        "article_exact": float(exact),
        "article_cross": float(cross),
        "article_any": float(exact or cross),
        # Артикулы есть у обеих сторон, но ни один не нашёлся — сильный довод против.
        "article_both_miss": float(both and not exact and not cross),
        "long_number_shared": float(len(long_numbers)),
    }


def slot_features(left: dict[str, frozenset[str]],
                  right: dict[str, frozenset[str]]) -> dict[str, float]:
    """По каждому слоту: совпали, разошлись, есть только у одной стороны.

    Три состояния держатся раздельно намеренно. «Бренды разные» — сильный отрицательный
    сигнал, «бренд указан только у одного» — почти никакой, и сваливать их в один признак
    значит терять и то, и другое.
    """
    features: dict[str, float] = {}
    for slot, _ in ATTR_SLOTS:
        a, b = left.get(slot), right.get(slot)
        if a and b:
            features[f"slot_{slot}_agree"] = float(bool(a & b))
            features[f"slot_{slot}_conflict"] = float(not (a & b))
            features[f"slot_{slot}_missing"] = 0.0
        else:
            features[f"slot_{slot}_agree"] = 0.0
            features[f"slot_{slot}_conflict"] = 0.0
            features[f"slot_{slot}_missing"] = float(bool(a) != bool(b))
    return features


EMPTY = parse("")
FEATURE_NAMES: tuple[str, ...] = tuple(compare(EMPTY, EMPTY).keys())
