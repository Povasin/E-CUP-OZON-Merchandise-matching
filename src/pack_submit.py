"""Сборка архива решения с проверками перед упаковкой.

Одна попытка уже была потеряна на том, что архив зависел от переменной окружения, которой
у проверяющей системы нет. Поэтому здесь всё, что можно проверить до отправки, проверяется:
состав весов, соответствие каталогов моделей артефакту смеси, наличие модулей и отсутствие
внешних зависимостей от окружения.

Запуск:
    .venv/bin/python -m src.pack_submit --weights output/blend_best.npz \
        --model models/ce_best --model models/ce_relaxed --out output/submit_new.zip
"""
from __future__ import annotations

import argparse
import json
import zipfile
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parent.parent
# Модули, нужные инференсу. Обучающие сюда не входят: в архиве им делать нечего.
CODE_FILES = (
    "__init__.py", "pipeline.py", "features.py", "model.py", "data.py",
    "scoring.py", "cross_encoder.py", "metrics.py", "hybrid.py",
    # Признаковая модель и её признаки: без них смесь считает только энкодеры.
    "pair_features.py", "attr_features.py", "name_features.py", "string_features.py",
    "neighbour_features.py", "brand_features.py", "dim_features.py", "field_features.py",
    "export_boost.py", "measure_features.py", "canon_features.py",
)
# Словари, добытые из обучающих данных. Без них признаки считаются, но молча теряют
# сигнал: `anti_words` отвечает за слова-различители внутри линейки товара, а
# `brand_aliases` сводит разные написания одного бренда. Молчаливость тут и опасна:
# `load_aliases` при отсутствии файла возвращает пустой словарь, признаки считаются
# дальше, и расхождение с обучением вылезает только в итоговой метрике.
RESOURCE_FILES = ("anti_words.json", "brand_aliases.json")
# Слитая модель: скор считается одной моделью поверх признаков и оценок энкодера вместо
# сложения рангов. Если этих файлов в архиве нет, пайплайн возвращается к прежнему пути.
FUSION_FILES = ("fusion_boost.npz", "fusion_info.json")
BOOST_FILES = ("pair_boost.npz", "pair_boost_hybrid.npz", "pair_boost_hybrid_aux.npz",
               "pair_logreg.npz")
MODEL_FILES = ("model.safetensors", "config.json", "tokenizer.json",
               "tokenizer_config.json", "inference_config.json")


def needed_resources() -> set[str]:
    """Файлы из `models/`, которые упоминают модули инференса.

    Один раз архив уехал без `brand_aliases.json`: признаки брендов молча считались без
    приведения написаний, а обучение шло с ним. Ошибка ничего не роняет и в логе не видна,
    поэтому здесь состав словарей выводится из самого кода, а не поддерживается вручную.
    """
    import re

    found: set[str] = set()
    pattern = re.compile(r'"models"\s*/\s*"([^"]+\.json)"')
    for name in CODE_FILES:
        found.update(pattern.findall((REPO / "src" / name).read_text()))
    return found - set(FUSION_FILES)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", required=True, help="артефакт из src.blend_many")
    ap.add_argument("--model", action="append", required=True,
                    help="каталог кросс-энкодера; порядок должен совпадать с артефактом")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    # Состав энкодеров задаёт слитая модель, если она есть: прежний артефакт весов знает
    # только про двух, а обучение идёт уже на четырёх. Если слитой модели нет, решение
    # считает по-старому, и тогда сверяемся с ним.
    fusion = REPO / "models" / "fusion_info.json"
    if fusion.exists():
        info = json.loads(fusion.read_text())
        wanted = [f"models/{name}" for name in info.get("encoders", [])]
        if wanted != args.model:
            raise SystemExit(
                "Энкодеры в слитой модели не совпадают с переданными.\n"
                f"  в модели: {wanted}\n  передано: {args.model}"
            )
        print(f"  слитая модель: {len(info['columns'])} столбцов, "
              f"метрика на фолде {info['honest_macro']:.6f}")
    artifact = np.load(args.weights, allow_pickle=False)
    weights = artifact["weights"]
    if not fusion.exists():
        ce_dirs = artifact["ce_dirs"].astype(str).tolist()
        if len(weights) != 1 + len(args.model) or ce_dirs != args.model:
            raise SystemExit(
                "Каталоги в артефакте не совпадают с переданными — веса относятся к другим "
                f"моделям.\n  в артефакте: {ce_dirs}\n  передано:    {args.model}"
            )

    for directory in args.model:
        for name in MODEL_FILES:
            if not (REPO / directory / name).exists():
                raise SystemExit(f"Нет файла {directory}/{name}")
        config = json.loads((REPO / directory / "inference_config.json").read_text())
        kind = "двухбашенная" if config.get("kind") == "biencoder" else "кросс"
        print(f"  {directory}: {kind}, режим {config['mode']}, длина {config['max_length']}")

    missing = sorted(needed_resources() - set(RESOURCE_FILES))
    if missing:
        raise SystemExit(
            "Модули инференса читают файлы, которых упаковщик не кладёт: "
            + ", ".join(missing) + ".\nДобавьте их в RESOURCE_FILES."
        )

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(out, "w", zipfile.ZIP_STORED) as archive:
        archive.write(REPO / "run.py", "run.py")
        archive.write(REPO / "metadata.json", "metadata.json")
        for name in CODE_FILES:
            archive.write(REPO / "src" / name, f"src/{name}")
        for name in BOOST_FILES:
            archive.write(REPO / "models" / name, f"models/{name}")
        # Веса смеси кладутся под именем, которое пайплайн ищет по умолчанию: полагаться на
        # переменную окружения нельзя, проверяющая система её не задаёт.
        archive.write(args.weights, "models/blend_weights.npz")
        for name in RESOURCE_FILES + FUSION_FILES:
            source = REPO / "models" / name
            if source.exists():
                archive.write(source, f"models/{name}")
        for directory in args.model:
            for name in MODEL_FILES:
                archive.write(REPO / directory / name, f"{directory}/{name}")

    size = out.stat().st_size / 1e9
    print(f"\n{out}: {size:.2f} ГБ, участников {len(weights)} "
          f"(структурная + {len(args.model)} кросс-энкодеров)")
    print(f"веса: {' '.join(f'{w:.1f}' for w in weights)}")


if __name__ == "__main__":
    main()
