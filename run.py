"""Точка входа решения. Интерфейс запуска — как в решении-примере:

    python -u run.py --items_path items.parquet --matches_path matches.parquet \
                     --output_path submit.csv

Метод скоринга можно переопределить переменной окружения MATCH_METHOD.
"""
import argparse
import os

from src.pipeline import predict_pipeline

METHOD = os.environ.get("MATCH_METHOD", "boosted")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--items_path", type=str, help="test items data path")
    parser.add_argument("--matches_path", type=str, help="test matches data path")
    parser.add_argument("--output_path", type=str, help="output file")
    args = parser.parse_args()

    predict_pipeline(
        items_path=args.items_path,
        matches_path=args.matches_path,
        output_path=args.output_path,
        method=METHOD,
    )


if __name__ == "__main__":
    main()
