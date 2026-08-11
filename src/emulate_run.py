"""Эмуляция официального прогона решения по условиям ТЗ.

В отличие от `local_eval` (считает метрику на метках), здесь воспроизводится сама
процедура проверки из ТЗ — Docker-стадия + Result-стадия:

  Docker-стадия  — решение запускается ТОЙ ЖЕ командой, что и грейдер
                   (`python -u run.py --items_path … --matches_path … --output_path …`),
                   на данных без меток, и должно уложиться в лимит времени стадии.
  Result-стадия  — выходной .csv строго проверяется на формат и полноту:
                   колонки id1,id2,predict; предсказание для КАЖДОЙ пары; без пропусков.

Тестовые входы готовятся как в реальном тесте (см. ТЗ):
  * `matches` — БЕЗ колонки target (в тесте меток нет);
  * `items`   — только товары, участвующие в парах.

Стадии и лимиты (ТЗ):
  Check   — 1 000 пар,  лимит 1 мин
  Public  — ~115 000 пар, лимит 6 мин
  Private — ~275 000 пар, лимит 13 мин

Запуск:
  .venv/bin/python -m src.emulate_run --stage check
  .venv/bin/python -m src.emulate_run --stage all
  MATCH_METHOD=embed .venv/bin/python -m src.emulate_run --stage check

Важно: у нас нет H100/20 ядер/200GB, поэтому тайминг — консервативная оценка (реальная
машина мощнее). Для GPU-методов локальное время по скорости не показательно, проверяется
корректность и формат.
"""
from __future__ import annotations

import argparse
import os
import subprocess
import sys
import tempfile
import time
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parent.parent

STAGES = {
    # имя: (число пар, лимит времени в секундах)
    "check": (1_000, 1 * 60),
    "public": (115_000, 6 * 60),
    "private": (275_000, 13 * 60),
}


def prepare_inputs(
    matches_path: str, items_path: str, n_pairs: int, workdir: Path, seed: int = 1234
) -> tuple[Path, Path, pd.DataFrame]:
    """Собрать тест-подобные входы: matches без target, items только из пар."""
    matches = pd.read_parquet(matches_path)
    if n_pairs < len(matches):
        matches = matches.sample(n=n_pairs, random_state=seed).reset_index(drop=True)

    # В тесте меток нет — оставляем только id1,id2
    test_matches = matches[["id1", "id2"]].copy()

    # В тест подаются только товары, участвующие в парах
    used_ids = pd.unique(np.concatenate([test_matches["id1"].to_numpy(), test_matches["id2"].to_numpy()]))
    items = pd.read_parquet(items_path, columns=["id", "name", "attributes", "category"])
    items = items[items["id"].isin(used_ids)].reset_index(drop=True)

    m_path = workdir / "test_matches.parquet"
    i_path = workdir / "test_items.parquet"
    test_matches.to_parquet(m_path)
    items.to_parquet(i_path)
    return i_path, m_path, test_matches


def validate_output(out_path: Path, test_matches: pd.DataFrame) -> list[str]:
    """Проверки Result-стадии. Вернуть список ошибок (пустой = успех)."""
    errors: list[str] = []
    if not out_path.exists():
        return [f"выходной файл не создан: {out_path}"]

    df = pd.read_csv(out_path)
    if list(df.columns) != ["id1", "id2", "predict"]:
        errors.append(f"колонки {list(df.columns)} != ['id1','id2','predict']")
    if len(df) != len(test_matches):
        errors.append(f"строк {len(df)} != входных пар {len(test_matches)}")
    if "predict" in df.columns:
        bad = int(df["predict"].isna().sum() + np.isinf(df["predict"]).sum())
        if bad:
            errors.append(f"predict содержит {bad} NaN/inf")
        if not np.issubdtype(df["predict"].dtype, np.number):
            errors.append(f"predict не числовой (dtype={df['predict'].dtype})")

    # Каждая входная пара должна присутствовать ровно один раз
    if {"id1", "id2"}.issubset(df.columns):
        got = set(map(tuple, df[["id1", "id2"]].to_numpy()))
        want = set(map(tuple, test_matches[["id1", "id2"]].to_numpy()))
        missing = want - got
        if missing:
            errors.append(f"нет предсказаний для {len(missing)} пар")
        if len(df) != len(df.drop_duplicates(subset=["id1", "id2"])):
            errors.append("есть дублирующиеся пары в выходе")
    return errors


def run_stage(stage: str, args) -> bool:
    n_pairs, limit_s = STAGES[stage]
    print(f"\n{'='*60}\nСТАДИЯ: {stage.upper()}  |  пар: {n_pairs}  |  лимит: {limit_s}с\n{'='*60}")

    with tempfile.TemporaryDirectory(prefix=f"emulate_{stage}_") as tmp:
        workdir = Path(tmp)
        i_path, m_path, test_matches = prepare_inputs(
            args.matches, args.items, n_pairs, workdir, args.seed
        )
        out_path = workdir / "submit.csv"
        print(f"  входы: {len(test_matches)} пар (без target), items={i_path.name}")

        # Точно та же команда, что у грейдера
        cmd = [
            sys.executable, "-u", "run.py",
            "--items_path", str(i_path),
            "--matches_path", str(m_path),
            "--output_path", str(out_path),
        ]
        env = {**os.environ, "PYTHONPATH": str(REPO)}
        print(f"  команда: {' '.join(cmd)}")

        t0 = time.perf_counter()
        proc = subprocess.run(cmd, cwd=REPO, env=env, capture_output=True, text=True)
        elapsed = time.perf_counter() - t0

        if proc.returncode != 0:
            print(f"  [DOCKER] FAIL — решение упало (код {proc.returncode})")
            print("  --- stderr (хвост) ---")
            print("\n".join(proc.stderr.strip().splitlines()[-15:]))
            return False

        within = elapsed <= limit_s
        print(f"  [DOCKER] время: {elapsed:.1f}с / {limit_s}с — {'OK' if within else 'ПРЕВЫШЕНИЕ'}")

        errors = validate_output(out_path, test_matches)
        if errors:
            print("  [RESULT] FAIL:")
            for e in errors:
                print(f"    - {e}")
            return False
        print("  [RESULT] OK — формат и полнота выхода корректны")

        ok = within and not errors
        if not within:
            print("  (примечание: тайминг на этом Mac консервативен; реальная машина мощнее)")
        print(f"  ИТОГ стадии {stage}: {'ПРОЙДЕНА' if ok else 'НЕ ПРОЙДЕНА (по времени)'}")
        return ok


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", choices=[*STAGES, "all"], default="check")
    ap.add_argument("--items", default="assets/items_human.parquet")
    ap.add_argument("--matches", default="assets/matches.parquet")
    ap.add_argument("--seed", type=int, default=1234)
    args = ap.parse_args()

    method = os.environ.get("MATCH_METHOD", "supervised")
    print(f"Метод скоринга: {method}  (MATCH_METHOD)")

    stages = list(STAGES) if args.stage == "all" else [args.stage]
    all_ok = True
    for st in stages:
        ok = run_stage(st, args)
        all_ok &= ok
        if not ok and args.stage == "all":
            # Как в ТЗ: падение на стадии завершает проверку
            print("\nПроверка остановлена: стадия не пройдена.")
            break

    print(f"\n{'='*60}\nОБЩИЙ ИТОГ: {'ВСЕ СТАДИИ ПРОЙДЕНЫ' if all_ok else 'ЕСТЬ ПРОБЛЕМЫ'}\n{'='*60}")
    sys.exit(0 if all_ok else 1)


if __name__ == "__main__":
    main()
