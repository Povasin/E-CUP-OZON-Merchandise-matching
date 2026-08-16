"""Разметка спорных LLM-пар обученным кросс-энкодером.

Разметка организаторов делалась языковой моделью, и на 933 580 парах она не смогла
определиться: её оценки лежат в середине (медиана 0.333), поэтому эти пары отбрасываются
как ненадёжные — почти пятая часть всех данных.

Наш кросс-энкодер даёт 0.741 на product-disjoint holdout ручной разметки, то есть на
трудных случаях он заведомо точнее исходного разметчика. Он размечает спорные пары
заново, и уверенная часть его решений становится обучающими данными.

Оставляются только уверенные предсказания: псевдометка на границе — это шум, который
модель уже не отличит от сигнала, и она лишь закрепит собственные ошибки.

Запуск:
    python -m src.pseudo_label --model models/cross_encoder --out output/pseudo.parquet
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, help="каталог обученного кросс-энкодера")
    ap.add_argument("--pairs", required=True, help="parquet со спорными парами")
    ap.add_argument("--texts", action="append", required=True,
                    help="parquet с текстами (id,text); можно указать несколько")
    ap.add_argument("--out", default="output/pseudo_pairs.parquet")
    ap.add_argument("--batch-size", type=int, default=256)
    ap.add_argument("--low", type=float, default=0.15, help="ниже этого — уверенный ноль")
    ap.add_argument("--high", type=float, default=0.85, help="выше этого — уверенная единица")
    args = ap.parse_args()

    import torch
    from transformers import AutoModelForSequenceClassification, AutoTokenizer

    from src.train_ce_large import predict_scores

    config = json.loads((Path(args.model) / "inference_config.json").read_text())
    max_length = int(config["max_length"])

    pairs = pd.read_parquet(args.pairs)
    texts: dict[int, str] = {}
    for path in args.texts:
        frame = pd.read_parquet(path, columns=["id", "text"])
        texts.update(zip(frame["id"].to_numpy().tolist(), frame["text"].tolist()))
        del frame
    print(f"пар: {len(pairs):,}, текстов: {len(texts):,}", flush=True)

    left = [texts.get(int(i), "") for i in pairs["id1"]]
    right = [texts.get(int(i), "") for i in pairs["id2"]]
    missing = sum(1 for a, b in zip(left, right) if not a or not b)
    if missing:
        print(f"ВНИМАНИЕ: у {missing:,} пар нет текста, они будут отброшены", flush=True)
    del texts

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = AutoTokenizer.from_pretrained(args.model, local_files_only=True)
    model = AutoModelForSequenceClassification.from_pretrained(
        args.model, local_files_only=True
    ).to(device).eval()
    amp_dtype = torch.bfloat16 if (device.type == "cuda"
                                   and torch.cuda.get_device_capability()[0] >= 8) else torch.float16

    logits = predict_scores(model, tokenizer, left, right, device,
                            args.batch_size, max_length, amp_dtype)
    probability = 1.0 / (1.0 + np.exp(-np.clip(logits, -30, 30)))

    confident = (probability <= args.low) | (probability >= args.high)
    has_text = np.asarray([bool(a) and bool(b) for a, b in zip(left, right)])
    keep = confident & has_text
    label = (probability >= 0.5).astype(np.int8)

    print(f"\nуверенных предсказаний: {keep.sum():,} из {len(pairs):,} ({keep.mean()*100:.1f}%)")
    print(f"  из них положительных: {label[keep].mean()*100:.1f}%")

    # Проверка на вменяемость: наши метки должны коррелировать с исходными оценками LLM,
    # но не совпадать с ними — иначе мы либо не добавили информации, либо разошлись с
    # разметчиком настолько, что доверять результату нельзя.
    if "target" in pairs.columns:
        soft = pairs["target"].to_numpy(np.float32)
        agree = ((soft >= 0.5) == (label == 1))[keep].mean()
        print(f"  согласие с исходной LLM-оценкой: {agree*100:.1f}%")
        print(f"  корреляция с исходной оценкой: {np.corrcoef(soft[keep], probability[keep])[0,1]:.3f}")

    result = pairs.loc[keep, ["id1", "id2"]].copy()
    result["label"] = label[keep]
    result["confidence"] = np.abs(probability[keep] - 0.5) * 2.0
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    result.to_parquet(args.out, compression="zstd", index=False)
    print(f"\nСохранено: {args.out} ({len(result):,} пар)")


if __name__ == "__main__":
    main()
