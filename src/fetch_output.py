"""Выборочная загрузка файлов из вывода ядра Kaggle.

`kaggle kernels output` тянет вывод целиком, а в одном ядре лежат три обученные модели по
678 МБ. Здесь скачивается только запрошенный подкаталог.

Запуск:
    KAGGLE_API_TOKEN=... .venv/bin/python -m src.fetch_output \
        --user yllxio --kernel sweep-31 --prefix lr2e5ep3_all/ --out models/ce_best
"""
from __future__ import annotations

import argparse
import json
import os
import urllib.parse
import urllib.request
from pathlib import Path


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--user", required=True)
    ap.add_argument("--kernel", required=True)
    ap.add_argument("--prefix", default="", help="каталог внутри вывода, например lr2e5ep3_all/")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    token = os.environ.get("KAGGLE_API_TOKEN")
    if not token:
        raise SystemExit("Нужен KAGGLE_API_TOKEN аккаунта, на котором считалось ядро")

    url = "https://www.kaggle.com/api/v1/kernels/output?" + urllib.parse.urlencode(
        {"userName": args.user, "kernelSlug": args.kernel}
    )
    request = urllib.request.Request(url, headers={"Authorization": "Bearer " + token})
    listing = json.load(urllib.request.urlopen(request, timeout=120))

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    wanted = [f for f in listing["files"] if f["fileName"].startswith(args.prefix)]
    if not wanted:
        available = sorted({f["fileName"].split("/")[0] for f in listing["files"]})
        raise SystemExit(f"Ничего не найдено по «{args.prefix}». Есть: {', '.join(available)}")

    for entry in wanted:
        # Чекпоинт — это промежуточное состояние обучения, для инференса он не нужен.
        name = entry["fileName"]
        if "/checkpoint/" in name:
            continue
        target = out / name[len(args.prefix):]
        target.parent.mkdir(parents=True, exist_ok=True)
        if target.exists():
            print(f"уже есть: {target.name}")
            continue
        with urllib.request.urlopen(entry["urlNullable"], timeout=3600) as response:
            data = response.read()
        target.write_bytes(data)
        print(f"{target.name}: {len(data) / 1e6:.1f} МБ")

    print(f"\nГотово: {out}")


if __name__ == "__main__":
    main()
