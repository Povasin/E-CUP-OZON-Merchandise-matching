"""Полный цикл обучения на Kaggle одной командой.

Локальная машина — M1 с 8 GB, поэтому дообучение трансформера велось на 38 568 парах при
доступных 9 088 499. Этот модуль переносит обучение на бесплатную GPU Kaggle: загружает
данные и код как приватные датасеты, запускает ноутбук, ждёт и забирает артефакты.

Требуется только API-токен Kaggle (Settings -> API -> Create New Token), положенный в
`~/.kaggle/kaggle.json`. Имя пользователя берётся оттуда же.

Запуск:
    .venv/bin/python -m src.kaggle_run --step all
    .venv/bin/python -m src.kaggle_run --step fetch     # забрать результат позже
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
STAGE = REPO / "output" / "kaggle"
DATA_SLUG = "ecup-matching-data"
CODE_SLUG = "ecup-matching-src"
KERNEL_SLUG = "ecup-cross-encoder-large"
PACK_FILES = ("llm_train.parquet", "llm_valid.parquet", "item_texts.parquet",
              "human_texts.parquet", "human_matches.parquet", "pack-info.json")
# Модули, которые нужны train_ce_large; остальное в ноутбук не тянем.
CODE_FILES = (
    "__init__.py", "train_ce_large.py", "cross_encoder.py", "hybrid.py",
    "metrics.py", "features.py", "train_model.py", "model.py", "data.py", "scoring.py",
)


def kaggle_username() -> str:
    path = Path.home() / ".kaggle" / "kaggle.json"
    if not path.exists():
        raise SystemExit(
            "Нет ~/.kaggle/kaggle.json.\n"
            "Kaggle -> Settings -> API -> Create New Token, затем:\n"
            "  mkdir -p ~/.kaggle && mv ~/Downloads/kaggle.json ~/.kaggle/ && chmod 600 ~/.kaggle/kaggle.json"
        )
    if oct(path.stat().st_mode)[-3:] != "600":
        path.chmod(0o600)
    return json.loads(path.read_text())["username"]


def run(command: list[str]) -> str:
    print(f"$ {' '.join(command)}", flush=True)
    result = subprocess.run(command, cwd=REPO, capture_output=True, text=True)
    output = (result.stdout + result.stderr).strip()
    print(output, flush=True)
    if result.returncode != 0:
        raise SystemExit(f"Команда завершилась с кодом {result.returncode}")
    return output


def kaggle_cmd() -> list[str]:
    return [str(REPO / ".venv" / "bin" / "kaggle")]


def push_dataset(directory: Path, slug: str, title: str, user: str, files: list[Path]) -> None:
    directory.mkdir(parents=True, exist_ok=True)
    for source in files:
        link = directory / source.name
        if link.exists() or link.is_symlink():
            link.unlink()
        # Симлинк вместо копии: items.parquet весит 4.1 GB, дублировать его незачем.
        link.symlink_to(source)
    (directory / "dataset-metadata.json").write_text(json.dumps({
        "title": title, "id": f"{user}/{slug}",
        "licenses": [{"name": "unknown"}],
    }, ensure_ascii=False, indent=2))

    listed = run(kaggle_cmd() + ["datasets", "list", "--mine", "-s", slug])
    if f"{user}/{slug}" in listed:
        run(kaggle_cmd() + ["datasets", "version", "-p", str(directory),
                            "-m", "update", "--dir-mode", "zip"])
    else:
        run(kaggle_cmd() + ["datasets", "create", "-p", str(directory), "--dir-mode", "zip"])


def notebook_source(user: str, train_pairs: int, epochs: int, batch_size: int,
                    max_length: int) -> dict:
    code = f"""import sys, os, glob, shutil, subprocess
import torch

# Без видеокарты обучение бессмысленно: на процессоре 2M пар не досчитаются никогда.
# Падаем сразу, а не через часы занятой очереди.
if not torch.cuda.is_available():
    raise SystemExit("GPU не выделена — проверьте ускоритель в настройках ноутбука")
print("GPU:", torch.cuda.get_device_name(0))

# Kaggle распаковывает загруженное сам, поэтому точные пути заранее неизвестны.
found = glob.glob("/kaggle/input/**/llm_train.parquet", recursive=True)
if not found:
    raise SystemExit("Пакет данных не найден в /kaggle/input")
pack = os.path.dirname(found[0])
print("пакет:", pack, sorted(os.listdir(pack)))

# Модули лежат в датасете плоско, а запускать нужно как пакет `src`; дочерний процесс
# не наследует sys.path — поэтому собираем настоящий каталог пакета и задаём PYTHONPATH.
modules = glob.glob("/kaggle/input/**/train_ce_large.py", recursive=True)
if not modules:
    raise SystemExit("Код не найден в /kaggle/input")
os.makedirs("/kaggle/working/src", exist_ok=True)
for path in glob.glob(os.path.dirname(modules[0]) + "/*.py"):
    shutil.copy(path, "/kaggle/working/src/")
open("/kaggle/working/src/__init__.py", "a").close()
print("модули:", sorted(os.listdir("/kaggle/working/src")))

subprocess.run([sys.executable, "-u", "-m", "src.train_ce_large",
                "--prepacked", pack, "--epochs", "{epochs}",
                "--batch-size", "{batch_size}", "--max-length", "{max_length}",
                "--output", "/kaggle/working/ce_large"],
               check=True, cwd="/kaggle/working",
               env=dict(os.environ, PYTHONPATH="/kaggle/working"))
"""
    return {
        "cells": [{"cell_type": "code", "source": code.splitlines(keepends=True),
                   "metadata": {}, "outputs": [], "execution_count": None}],
        "metadata": {"kernelspec": {"language": "python", "name": "python3",
                                    "display_name": "Python 3"}},
        "nbformat": 4, "nbformat_minor": 4,
    }


def push_kernel(user: str, args) -> None:
    directory = STAGE / "kernel"
    directory.mkdir(parents=True, exist_ok=True)
    (directory / f"{KERNEL_SLUG}.ipynb").write_text(json.dumps(
        notebook_source(user, args.train_pairs, args.epochs, args.batch_size, args.max_length)
    ))
    (directory / "kernel-metadata.json").write_text(json.dumps({
        "id": f"{user}/{KERNEL_SLUG}",
        "title": "ECUP cross-encoder large",
        "code_file": f"{KERNEL_SLUG}.ipynb",
        "language": "python",
        "kernel_type": "notebook",
        "is_private": True,
        # enable_gpu в новом API игнорируется — ускоритель задаётся полем accelerator.
        # Прошлый прогон отработал на процессоре именно из-за этого.
        "enable_gpu": True,
        "accelerator": "nvidiaTeslaT4",
        "enable_internet": True,
        "dataset_sources": [f"{user}/{DATA_SLUG}", f"{user}/{CODE_SLUG}"],
        "competition_sources": [], "kernel_sources": [],
    }, indent=2))
    run(kaggle_cmd() + ["kernels", "push", "-p", str(directory)])


def wait_kernel(user: str, poll: int) -> str:
    while True:
        status = run(kaggle_cmd() + ["kernels", "status", f"{user}/{KERNEL_SLUG}"])
        lowered = status.lower()
        if "complete" in lowered:
            return "complete"
        if "error" in lowered or "cancel" in lowered:
            return "error"
        time.sleep(poll)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--step", choices=["all", "data", "code", "kernel", "wait", "fetch"],
                    default="all")
    ap.add_argument("--train-pairs", type=int, default=3_000_000)
    ap.add_argument("--epochs", type=int, default=1)
    ap.add_argument("--batch-size", type=int, default=256)
    ap.add_argument("--max-length", type=int, default=192)
    ap.add_argument("--poll", type=int, default=120)
    ap.add_argument("--fetch-dir", default="output/ce_large")
    args = ap.parse_args()

    user = kaggle_username()
    print(f"Пользователь Kaggle: {user}\n")

    if args.step in {"all", "data"}:
        data_dir = STAGE / "data"
        missing = [name for name in PACK_FILES if not (data_dir / name).exists()]
        if missing:
            raise SystemExit(
                f"В пакете нет файлов: {', '.join(missing)}\n"
                "Сначала выполните: .venv/bin/python -m src.pack_kaggle"
            )
        push_dataset(data_dir, DATA_SLUG, "ECUP matching data", user, [])
    if args.step in {"all", "code"}:
        push_dataset(STAGE / "code", CODE_SLUG, "ECUP matching src", user,
                     [REPO / "src" / name for name in CODE_FILES])
    if args.step in {"all", "kernel"}:
        push_kernel(user, args)
    if args.step in {"all", "wait"}:
        # Датасеты обрабатываются Kaggle не мгновенно, ядро стартует не сразу.
        print("Ожидание завершения ядра (можно прервать и вернуться с --step fetch)...")
        if wait_kernel(user, args.poll) == "error":
            raise SystemExit("Ядро завершилось с ошибкой — смотрите логи в интерфейсе Kaggle")
    if args.step in {"all", "fetch"}:
        target = REPO / args.fetch_dir
        target.mkdir(parents=True, exist_ok=True)
        run(kaggle_cmd() + ["kernels", "output", f"{user}/{KERNEL_SLUG}", "-p", str(target)])
        print(f"\nАртефакты скачаны в {target}")


if __name__ == "__main__":
    main()
