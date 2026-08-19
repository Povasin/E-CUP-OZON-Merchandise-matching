"""Инференс-пайплайн: данные -> скоринг -> submit.csv.

Гарантирует предсказание для КАЖДОЙ входной пары в исходном порядке.
"""
from __future__ import annotations

import json
import os
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd

from src.data import attach_texts, load_items, load_matches
from src.scoring import score_pairs


DEFAULT_MODEL_PATH = os.environ.get("PAIR_MODEL_PATH", "models/pair_logreg.npz")
DEFAULT_BOOST_MODEL_PATH = os.environ.get("PAIR_BOOST_MODEL_PATH", "models/pair_boost_hybrid.npz")
DEFAULT_BOOST_AUX_MODEL_PATH = os.environ.get(
    "PAIR_BOOST_AUX_MODEL_PATH", "models/pair_boost_hybrid_aux.npz"
)
DEFAULT_BOOST_AUX_WEIGHT = float(os.environ.get("PAIR_BOOST_AUX_WEIGHT", "0.20"))
DEFAULT_BLEND_WEIGHTS = os.environ.get("PAIR_BLEND_WEIGHTS", "models/blend_weights.npz")
DEFAULT_CE_BATCH = int(os.environ.get("PAIR_CE_BATCH", "1024"))


def _category_ranks(scores, categories):
    frame = pd.DataFrame({"score": scores, "category": categories})
    return frame.groupby("category", sort=False)["score"].rank(pct=True).to_numpy(dtype=np.float32)


def _item_texts(matches: pd.DataFrame, items: pd.DataFrame, mode: str) -> dict[int, str]:
    """Текст каждой карточки, участвующей в парах.

    Товаров в файле на порядок больше, чем участвует в парах, поэтому текст собирается
    только для нужных: разбор атрибутов стоит около 43 мкс на карточку. Словарь общий для
    кросс-энкодеров и двухбашенной модели — собирать его дважды незачем.
    """
    from src.cross_encoder import build_product_texts

    used = pd.unique(np.concatenate([matches["id1"].to_numpy(), matches["id2"].to_numpy()]))
    return build_product_texts(items[items["id"].isin(used)], mode)


def _pair_texts(matches: pd.DataFrame, items: pd.DataFrame,
                mode: str) -> tuple[np.ndarray, np.ndarray]:
    """Тексты сторон каждой пары в режиме `mode`."""
    texts = _item_texts(matches, items, mode)
    left = matches["id1"].map(texts).fillna("").astype(str).to_numpy()
    right = matches["id2"].map(texts).fillna("").astype(str).to_numpy()
    return left, right


def _biencoder_scores(matches: pd.DataFrame, texts: dict[int, str],
                      model_dir: str) -> np.ndarray:
    """Косинус между вложениями сторон пары.

    Двухбашенная модель кодирует товар отдельно от партнёра, поэтому карточка считается
    один раз, а не столько раз, во скольких парах она встречается. Плюс длина у неё вдвое
    меньше кросс-энкодерной, а внимание растёт квадратично по длине — на объёме теста это
    примерно половина стоимости одного кросс-энкодера.

    Сама по себе она слабее всех наших моделей (0.502 на отложенном фолде против
    0.68–0.72), но согласна с ними лишь на 0.36–0.44 против 0.82–0.97 между ними, и
    столбцом даёт +0.0054.
    """
    import torch
    import torch.nn.functional as F
    from transformers import AutoModel, AutoTokenizer

    config = json.loads((Path(model_dir) / "inference_config.json").read_text())
    max_length = int(config["max_length"])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = torch.float16 if device.type == "cuda" else torch.float32
    tokenizer = AutoTokenizer.from_pretrained(model_dir, local_files_only=True)
    model = AutoModel.from_pretrained(model_dir, local_files_only=True,
                                      dtype=dtype).to(device).eval()
    print(f"      двухбашенная {Path(model_dir).name}: {len(texts):,} карточек на {device.type}",
          flush=True)

    identifiers = list(texts)
    strings = [texts[i] for i in identifiers]
    order = np.argsort(np.fromiter((len(s) for s in strings), dtype=np.int32, count=len(strings)))
    vectors = np.empty((len(identifiers), model.config.hidden_size), dtype=np.float32)
    with torch.inference_mode():
        for start in range(0, len(order), DEFAULT_CE_BATCH):
            rows = order[start:start + DEFAULT_CE_BATCH]
            encoded = tokenizer([strings[i] for i in rows], padding=True, truncation=True,
                                max_length=max_length,
                                pad_to_multiple_of=8 if device.type == "cuda" else None,
                                return_tensors="pt").to(device)
            hidden = model(**encoded).last_hidden_state
            mask = encoded["attention_mask"].unsqueeze(-1).to(hidden.dtype)
            pooled = (hidden * mask).sum(1) / mask.sum(1).clamp(min=1e-6)
            vectors[rows] = F.normalize(pooled.float(), dim=-1).cpu().numpy()
    del model

    # Товар без карточки в items текстом не обзавёлся, вложения у него нет. Такая пара
    # получает нулевой косинус — как у ортогональных векторов, то есть «ничего не знаю».
    position = {int(v): row for row, v in enumerate(identifiers)}
    left = matches["id1"].map(position).fillna(-1).to_numpy(dtype=np.int64)
    right = matches["id2"].map(position).fillna(-1).to_numpy(dtype=np.int64)
    known = (left >= 0) & (right >= 0)
    scores = np.zeros(len(matches), dtype=np.float32)
    scores[known] = (vectors[left[known]] * vectors[right[known]]).sum(-1)
    return scores


def _tokenizer_groups(model_dirs: list[str]) -> list[list[str]]:
    """Модели, у которых токенизатор совпадает побитово, объединяются в группу.

    Четыре из шести энкодеров построены на одной основе и делят `tokenizer.json`.
    Токенизация 365 тысяч пар — работа процессора, и раньше она повторялась для каждой
    модели отдельно, всё это время видеокарта простаивала. Внутри группы текст режется
    один раз на все модели.
    """
    from hashlib import sha256

    groups: dict[str, list[str]] = {}
    for directory in model_dirs:
        digest = sha256((Path(directory) / "tokenizer.json").read_bytes()).hexdigest()
        groups.setdefault(digest, []).append(directory)
    return list(groups.values())


def _cross_encoder_scores(matches: pd.DataFrame, left: np.ndarray, right: np.ndarray,
                          model_dirs: list[str]) -> dict[str, np.ndarray]:
    """Логиты каждой модели группы в порядке входных пар.

    Модели группы держатся на карте одновременно: базовая модель в fp16 занимает 0.36 ГБ,
    вчетвером это полтора гигабайта из восьмидесяти, зато батч токенизируется однажды.
    """
    import json

    import torch
    from transformers import AutoModelForSequenceClassification, AutoTokenizer

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = torch.float16 if device.type == "cuda" else torch.float32
    max_length = min(int(json.loads((Path(d) / "inference_config.json").read_text())["max_length"])
                     for d in model_dirs)
    tokenizer = AutoTokenizer.from_pretrained(model_dirs[0], local_files_only=True)
    models = {d: AutoModelForSequenceClassification.from_pretrained(
        d, local_files_only=True, dtype=dtype).to(device).eval() for d in model_dirs}
    print(f"      группа из {len(models)}: {', '.join(Path(d).name for d in model_dirs)}"
          f" на {device.type}", flush=True)
    # Замер показал, что энкодеры съедают 98.5% времени решения — компиляция единственное
    # место, где ускорение чего-то стоит. Но включается она только по переменной окружения
    # и пока выключена: выигрыш не замерен, а три отправки подряд уже сгорели по времени.
    #
    # `torch.compile` ленив: он возвращает обёртку сразу, а собирает граф при первом
    # вызове. Поэтому одной обёртки вокруг самого вызова мало — ошибка вылезла бы внутри
    # цикла, где её никто не ждёт. Здесь сразу делается прогревочный проход, и если он не
    # удался, возвращаются исходные модели.
    # `dynamic=True` обязателен: батч дополняется до самой длинной строки в нём, форма
    # плавает, и без этого граф пересобирался бы на каждой новой длине.
    if device.type == "cuda" and os.environ.get("PAIR_CE_COMPILE", "0") == "1":
        original = dict(models)
        try:
            stage = time.perf_counter()
            models = {d: torch.compile(m, dynamic=True) for d, m in models.items()}
            warmup = tokenizer(left[:8].tolist(), right[:8].tolist(), padding=True,
                               truncation=True, max_length=max_length,
                               pad_to_multiple_of=8, return_tensors="pt").to(device)
            with torch.inference_mode():
                for model in models.values():
                    model(**warmup)
            print(f"      компиляция включена ({time.perf_counter() - stage:.1f}с)", flush=True)
        except Exception as error:  # noqa: BLE001
            models = original
            print(f"      компиляция не удалась ({type(error).__name__}), считаем как есть",
                  flush=True)

    scores = {d: np.empty(len(matches), dtype=np.float32) for d in model_dirs}
    # Бакетинг по длине: батч дополняется до самой длинной строки в нём, поэтому
    # соседство похожих длин заметно сокращает вычисления на паддинге.
    order = np.argsort(np.fromiter(
        (len(a) + len(b) for a, b in zip(left, right)), dtype=np.int32, count=len(left)
    ))
    with torch.inference_mode():
        for start in range(0, len(order), DEFAULT_CE_BATCH):
            rows = order[start:start + DEFAULT_CE_BATCH]
            encoded = tokenizer(
                left[rows].tolist(), right[rows].tolist(),
                padding=True, truncation=True, max_length=max_length,
                pad_to_multiple_of=8 if device.type == "cuda" else None,
                return_tensors="pt",
            ).to(device)
            for directory, model in models.items():
                scores[directory][rows] = model(**encoded).logits.squeeze(-1).float().cpu().numpy()
    del models
    return scores


def _feature_scores(matches: pd.DataFrame, items: pd.DataFrame, artifact) -> np.ndarray:
    """Оценка признаковой модели для каждой пары.

    Модель обучается на Kaggle обычным `HistGradientBoosting`, а сюда приезжает выгруженной
    в массивы (`src.export_boost`): состав чужого докер-образа нам неизвестен, и полагаться
    на наличие и версию `sklearn` нельзя. Выгрузка проверена — совпадает с исходной моделью
    побитово.
    """
    from src.export_boost import predict_proba
    from src.pair_features import build_matrix_parallel, feature_names

    trees = {name[6:]: artifact[name] for name in artifact.files if name.startswith("trees_")}
    # Окрестности могут быть отключены: они зависят от пула товаров, а он в закрытом тесте
    # другой. Флаг лежит в артефакте рядом с моделью, чтобы состав признаков не разъехался.
    with_neighbours = ("with_neighbours" not in artifact.files
                       or bool(artifact["with_neighbours"][0]))
    names = feature_names(with_neighbours)
    # Состав признаков на инференсе обязан совпадать с обучающим. Расхождение не вызывает
    # ошибки само по себе: модель просто прочтёт не те столбцы и вернёт правдоподобный
    # мусор. Один раз так и вышло — в сборку добавили признаки уже после обучения.
    expected = int(trees["feature"].max()) + 1
    if len(names) != expected:
        raise ValueError(f"Модель обучена на {expected} признаках, сборка даёт {len(names)}")
    known = artifact["boost_categories"].astype(str).tolist()
    features = build_matrix_parallel(items, matches, known_categories=known,
                                     with_neighbours=with_neighbours)
    return predict_proba(trees, features.astype(np.float64))


DEFAULT_FUSION_MODEL = os.environ.get("PAIR_FUSION_MODEL", "models/fusion_boost.npz")
DEFAULT_FUSION_INFO = os.environ.get("PAIR_FUSION_INFO", "models/fusion_info.json")


def _fusion_scores(matches: pd.DataFrame, items: pd.DataFrame, pairs: pd.DataFrame) -> np.ndarray:
    """Одна модель поверх признаков и оценок кросс-энкодера.

    Прежнее устройство складывало ранги трёх независимых мнений с постоянным весом на
    категорию. Здесь оценка энкодера подаётся модели как обычный столбец, и та учится не
    усреднять, а сдвигать её по атрибутам пары. Замерено на отложенном фолде при доле
    положительных 11.1%: сложение рангов 0.5758, слитая модель 0.6157. Разложение
    показывает, что +0.005 из этого даёт обучение на честных негативах, а +0.025 — само
    устройство: признаковая модель, обученная на тех же данных, но оставленная отдельным
    мнением, даёт лишь 0.5812.

    Порядок столбцов задан обучением и лежит рядом с моделью: разойтись они не могут,
    расхождение проверяется до расчёта.
    """
    from src.export_boost import load, predict_proba
    from src.pair_features import build_matrix_parallel, feature_names

    info = json.loads(Path(DEFAULT_FUSION_INFO).read_text())
    columns, known = info["columns"], info["categories"]
    with_structural = bool(info.get("uses_structural"))
    # Состав энкодеров задаётся обученной моделью, а не кодом: на четырёх она даёт 0.6568
    # против 0.6360 на двух, и добавить пятый должно быть достаточно переобучения.
    encoders = list(info.get("encoders") or ["ce_relaxed", "ce_combo"])
    # Список берётся тем же вызовом, что и при обучении: раньше здесь величины
    # дописывались вручную, и когда к признакам добавились канонические поля, пайплайн
    # остался ждать 138 столбцов против 152 обученных.
    names = list(feature_names(False, with_measures=True))
    tail = encoders + (["structural"] if with_structural else []) + ["category_code"]
    expected = names + tail
    if columns != expected:
        raise ValueError(f"Состав столбцов разошёлся с обучением: {len(columns)} против "
                         f"{len(expected)}")

    categories = pairs["category"].to_numpy().astype(str)

    def cpu_work():
        """Всё, что считается на ядрах: признаки, величины и структурная модель.

        Считается в отдельном потоке, пока на видеокарте идут энкодеры. Раньше стадии
        шли по очереди, и двадцать ядер простаивали всё время работы GPU; замер показал,
        что процессорная часть стоит около 1070 секунд на объёме теста, а лимит Private —
        780. Обе части почти не мешают друг другу: одна упирается в видеокарту, другая в
        ядра, а потоку GIL не мешает — тяжёлое внутри уходит в отдельные процессы.
        """
        stage = time.perf_counter()
        built = build_matrix_parallel(items, matches, known_categories=known,
                                      with_neighbours=False, with_measures=True)
        print(f"      признаки и величины: {time.perf_counter() - stage:.1f}с", flush=True)
        if not with_structural:
            return built, None
        # Структурная модель на 168 прежних признаках: сама по себе она слабее энкодеров
        # (0.4930 против 0.5549 на отложенном фолде), но столбцом добавляет +0.0164.
        from src.features import extract_model_features
        from src.model import BoostedPairModel

        stage = time.perf_counter()
        legacy = extract_model_features(matches, items)
        primary = BoostedPairModel(DEFAULT_BOOST_MODEL_PATH).predict_probability(legacy, categories)
        auxiliary = BoostedPairModel(DEFAULT_BOOST_AUX_MODEL_PATH).predict_probability(
            legacy, categories
        )
        del legacy
        print(f"      структурная модель: {time.perf_counter() - stage:.1f}с", flush=True)
        return built, ((1.0 - DEFAULT_BOOST_AUX_WEIGHT) * primary
                       + DEFAULT_BOOST_AUX_WEIGHT * auxiliary)

    # Тексты собираются до запуска фоновой стадии: они тоже раскладываются по ядрам, и
    # два пула процессов одновременно дали бы сорок рабочих на двадцать ядер.
    stage = time.perf_counter()
    texts = _item_texts(matches, items, "compact")
    left = matches["id1"].map(texts).fillna("").astype(str).to_numpy()
    right = matches["id2"].map(texts).fillna("").astype(str).to_numpy()
    print(f"      тексты пар: {time.perf_counter() - stage:.1f}с", flush=True)

    # Двухбашенные модели считаются иначе: карточка кодируется один раз, а скор пары —
    # косинус вложений. Тип модели записан рядом с её весами, а не задан здесь.
    def kind_of(name: str) -> str:
        config = json.loads((Path(f"models/{name}") / "inference_config.json").read_text())
        return str(config.get("kind", "cross"))

    cross = [n for n in encoders if kind_of(n) != "biencoder"]
    towers = [n for n in encoders if kind_of(n) == "biencoder"]

    with ThreadPoolExecutor(max_workers=1) as pool:
        pending = pool.submit(cpu_work)
        computed: dict[str, np.ndarray] = {}
        for group in _tokenizer_groups([f"models/{name}" for name in cross]):
            stage = time.perf_counter()
            computed.update(_cross_encoder_scores(matches, left, right, group))
            print(f"      группа за {time.perf_counter() - stage:.1f}с", flush=True)
        for name in towers:
            stage = time.perf_counter()
            computed[f"models/{name}"] = _biencoder_scores(matches, texts, f"models/{name}")
            print(f"      {name} за {time.perf_counter() - stage:.1f}с", flush=True)
        encoder_columns = [computed[f"models/{name}"] for name in encoders]
        del left, right, texts, computed
        features, structural = pending.result()

    parts = [features, *encoder_columns]
    if structural is not None:
        parts.append(structural)

    order = {name: index for index, name in enumerate(known)}
    parts.append(np.asarray([order.get(str(c), -1) for c in categories], dtype=np.float32))
    matrix = np.column_stack(parts)
    return predict_proba(load(DEFAULT_FUSION_MODEL), matrix.astype(np.float64))


def _blend_scores(matches: pd.DataFrame, items: pd.DataFrame, pairs: pd.DataFrame) -> np.ndarray:
    """Смесь структурного бустинга и кросс-энкодера рангами внутри категории.

    Модели ошибаются по-разному (корреляция рангов 0.58), поэтому смесь сильнее любой из
    них. Вес свой на категорию: на обуви и одежде кросс-энкодер слаб и получает 0.1, на
    продуктах питания он сильнее структурной модели и получает 0.8.
    """
    from src.features import extract_model_features
    from src.model import BoostedPairModel

    # Тайминг стадий печатается всегда: лимит проверяющей системы 780с, и когда решение в
    # него не укладывается, единственный доступный нам разбор — этот вывод.
    stage = time.perf_counter()
    categories = pairs["category"].to_numpy().astype(str)
    features = extract_model_features(matches, items)
    print(f"      структурные признаки: {time.perf_counter() - stage:.1f}с", flush=True)
    primary = BoostedPairModel(DEFAULT_BOOST_MODEL_PATH).predict_probability(features, categories)
    auxiliary = BoostedPairModel(DEFAULT_BOOST_AUX_MODEL_PATH).predict_probability(
        features, categories
    )
    boost = (1.0 - DEFAULT_BOOST_AUX_WEIGHT) * primary + DEFAULT_BOOST_AUX_WEIGHT * auxiliary
    del features, primary, auxiliary

    # Веса участников подобраны на симплексе (src.blend_many) и лежат в артефакте в том
    # же порядке: сначала структурная модель, затем кросс-энкодеры.
    # Список моделей хранится в самом артефакте рядом с весами: если держать его
    # отдельно (в переменной окружения или в коде), они рано или поздно разойдутся.
    artifact = np.load(DEFAULT_BLEND_WEIGHTS, allow_pickle=False)
    ce_dirs = artifact["ce_dirs"].astype(str).tolist()
    fallback = artifact["weights"].astype(np.float32)
    if len(fallback) != 1 + len(ce_dirs):
        raise ValueError(
            f"Весов {len(fallback)}, а моделей {1 + len(ce_dirs)}: артефакт несогласован"
        )
    # Свои веса на категорию, если они подтвердились при подборе; иначе общие.
    by_category = dict(zip(artifact["categories"].astype(str),
                           artifact["category_weights"].astype(np.float32)))

    # Ранги каждой модели считаются один раз, затем смешиваются с весами своей категории.
    # Тексты сторон общие для всех энкодеров: режим у них один, а сборка стоит дорого.
    modes = {json.loads((Path(d) / "inference_config.json").read_text())["mode"] for d in ce_dirs}
    if len(modes) > 1:
        raise ValueError(f"Энкодеры ожидают разные режимы текста: {sorted(modes)}")
    stage = time.perf_counter()
    left, right = _pair_texts(matches, items, modes.pop())
    print(f"      тексты пар: {time.perf_counter() - stage:.1f}с", flush=True)

    all_ranks = [_category_ranks(boost, categories)]
    for model_dir in ce_dirs:
        stage = time.perf_counter()
        ce = _cross_encoder_scores(matches, left, right, [model_dir])[model_dir]
        print(f"      {Path(model_dir).name}: {time.perf_counter() - stage:.1f}с", flush=True)
        all_ranks.append(_category_ranks(ce, categories))
        del ce
    all_ranks = np.vstack(all_ranks)

    scores = np.empty(len(matches), dtype=np.float32)
    for category in np.unique(categories):
        mask = categories == category
        weights = by_category.get(category, fallback)
        scores[mask] = weights @ all_ranks[:, mask]

    if "boost_categories" not in artifact.files:
        return scores

    # Признаковая модель подмешивается своим весом на категорию: в детских товарах она
    # сильнее энкодера (0.857 против 0.759 на валидации) и получает 0.75, в ювелирных
    # изделиях бесполезна и получает ноль. Веса подобраны на двух независимых наборах с
    # разным устройством негативов и совпали между ними.
    stage = time.perf_counter()
    feature_scores = _feature_scores(matches, items, artifact)
    print(f"      признаковая модель: {time.perf_counter() - stage:.1f}с", flush=True)
    feature_ranks = _category_ranks(feature_scores, categories)
    share = dict(zip(artifact["feature_categories"].astype(str),
                     artifact["feature_weights"].astype(np.float32)))
    for category in np.unique(categories):
        weight = float(share.get(category, 0.0))
        if weight <= 0.0:
            continue
        mask = categories == category
        scores[mask] = (1.0 - weight) * scores[mask] + weight * feature_ranks[mask]
    return scores


def predict_scores(
    matches: pd.DataFrame,
    items: pd.DataFrame,
    method: str,
    pairs: pd.DataFrame | None = None,
    **scorer_kwargs,
) -> tuple[pd.DataFrame, object]:
    """Score input pairs while preserving their order."""
    if pairs is None:
        pairs = attach_texts(matches, items)
    if method == "blend":
        # Слитая модель, если она лежит в архиве; иначе прежнее сложение рангов.
        if Path(DEFAULT_FUSION_MODEL).exists() and Path(DEFAULT_FUSION_INFO).exists():
            scores = _fusion_scores(matches, items, pairs)
        else:
            scores = _blend_scores(matches, items, pairs)
        # Скоры лежат в (0,1], поэтому 0.0 — корректное «хуже всех» для пар без текста.
        missing = (pairs["text1"] == "").to_numpy() | (pairs["text2"] == "").to_numpy()
        scores[missing] = 0.0
    elif method in {"supervised", "boosted"}:
        from src.features import extract_model_features
        from src.model import BoostedPairModel, PairModel

        features = extract_model_features(matches, items)
        if method == "boosted":
            model_path = scorer_kwargs.pop("model_path", DEFAULT_BOOST_MODEL_PATH)
            aux_model_path = scorer_kwargs.pop("aux_model_path", DEFAULT_BOOST_AUX_MODEL_PATH)
            aux_weight = float(scorer_kwargs.pop("aux_weight", DEFAULT_BOOST_AUX_WEIGHT))
            if not 0.0 <= aux_weight <= 1.0:
                raise ValueError("aux_weight must be between 0 and 1")
            categories = pairs["category"].to_numpy()
            model = BoostedPairModel(model_path)
            if aux_model_path and aux_weight:
                primary = model.predict_probability(features, categories)
                auxiliary = BoostedPairModel(aux_model_path).predict_probability(
                    features, categories
                )
                scores = (1.0 - aux_weight) * primary + aux_weight * auxiliary
            else:
                scores = model.predict(features, categories)
        else:
            model_path = scorer_kwargs.pop("model_path", DEFAULT_MODEL_PATH)
            scores = PairModel(model_path).predict(features, pairs["category"].to_numpy())
        missing = (pairs["text1"] == "").to_numpy() | (pairs["text2"] == "").to_numpy()
        scores[missing] = 0.0
    else:
        scores = score_pairs(pairs, method=method, **scorer_kwargs)
    return pairs, scores


def predict_pipeline(
    items_path: str,
    matches_path: str,
    output_path: str,
    method: str = "tfidf",
    **scorer_kwargs,
) -> pd.DataFrame:
    t0 = time.perf_counter()
    print(f"[1/4] Загрузка товаров: {items_path}")
    items = load_items(items_path)

    print(f"[2/4] Загрузка пар: {matches_path}")
    matches = load_matches(matches_path)
    pairs = attach_texts(matches, items)
    n_missing = int((pairs["text1"] == "").sum() + (pairs["text2"] == "").sum())
    print(f"      пар: {len(pairs)} | сторон без текста: {n_missing}")

    print(f"[3/4] Скоринг методом '{method}'")
    pairs, scores = predict_scores(matches, items, method=method, pairs=pairs, **scorer_kwargs)

    print(f"[4/4] Сохранение результата: {output_path}")
    result = pd.DataFrame(
        {"id1": pairs["id1"].to_numpy(), "id2": pairs["id2"].to_numpy(), "predict": scores}
    )
    assert len(result) == len(matches), "Число предсказаний должно совпадать с числом пар"
    result.to_csv(output_path, index=False)

    print(f"Готово за {time.perf_counter() - t0:.1f}с")
    return result
