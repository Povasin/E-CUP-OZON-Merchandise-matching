"""Двухбашенная модель с негативами внутри батча.

Все наши кросс-энкодеры обучены одинаково: подать пару, ответить «да или нет». В разборе
чужих решений (WDC Products) вершину держит другая постановка — контрастная: товар
кодируется отдельно от партнёра, и на каждом шаге отталкивается от всех остальных
товаров в батче сразу. Отрицательных примеров получается не один на пару, а столько,
сколько в батче, и модель учится различать похожее, а не запоминать метку.

Для нас у этого есть отдельная ценность помимо качества: оценка такой модели устроена
иначе, чем у кросс-энкодера, и её несогласие с нашими шестью будет больше, чем их
несогласие между собой (сейчас 0.82–0.97). Слитой модели новый столбец полезен ровно
настолько, насколько он приносит то, чего в остальных нет.

Запуск:
    python -m src.train_biencoder --prepacked pack --output output/bi_encoder
"""
from __future__ import annotations

import argparse
import json
import re
import time
from pathlib import Path

import numpy as np
import pandas as pd


CATEGORY = re.compile(r"Категория:\s*([^|]+)")


def batches(count: int, size: int, rng: np.random.Generator, groups=None):
    """Батчи, внутри которых товары из одной категории.

    Со случайным набором задача вырождается: кроссовок отталкивается от утюга, это видно
    сразу, и loss падает с 4.16 до 0.05 за двести шагов из шести тысяч — учиться дальше
    нечему. Внутри категории негативы трудные, а именно их модель и должна различать.
    """
    if groups is None:
        order = rng.permutation(count)
        for start in range(0, count, size):
            rows = order[start:start + size]
            if len(rows) > 1:
                yield rows
        return
    chunks = []
    for rows in groups.values():
        rows = rng.permutation(rows)
        for start in range(0, len(rows), size):
            chunk = rows[start:start + size]
            if len(chunk) > 1:
                chunks.append(chunk)
    for index in rng.permutation(len(chunks)):
        yield chunks[index]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--prepacked", required=True)
    ap.add_argument("--base-model", required=True, help="каталог с основой и токенизатором")
    ap.add_argument("--output", required=True)
    ap.add_argument("--epochs", type=int, default=1)
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--max-length", type=int, default=128,
                    help="на сторону, а не на пару: товар кодируется отдельно")
    ap.add_argument("--learning-rate", type=float, default=2e-5)
    ap.add_argument("--temperature", type=float, default=0.05)
    ap.add_argument("--max-pairs", type=int, default=400_000)
    ap.add_argument("--smoke", type=int, default=30,
                    help="сколько шагов прогнать для проверки до основного обучения")
    ap.add_argument("--seed", type=int, default=2026)
    args = ap.parse_args()

    import torch
    import torch.nn.functional as F
    from transformers import AutoModel, AutoTokenizer

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Устройство: {device}", flush=True)
    torch.manual_seed(args.seed)
    rng = np.random.default_rng(args.seed)

    pack = Path(args.prepacked)
    texts = pd.read_parquet(pack / "item_texts.parquet")
    lookup = dict(zip(texts["id"].tolist(), texts["text"].astype(str).tolist()))
    del texts
    pairs = pd.read_parquet(pack / "llm_train.parquet")
    # Контрастное обучение учится на том, что должно совпасть: отрицательные примеры
    # берутся из батча, а не из разметки.
    pairs = pairs[pairs["label"] > 0].reset_index(drop=True)
    if len(pairs) > args.max_pairs:
        pairs = pairs.sample(args.max_pairs, random_state=args.seed).reset_index(drop=True)
    left = [lookup.get(int(i), "") for i in pairs["id1"]]
    right = [lookup.get(int(i), "") for i in pairs["id2"]]
    keep = [i for i, (a, b) in enumerate(zip(left, right)) if a and b]
    left = [left[i] for i in keep]
    right = [right[i] for i in keep]
    print(f"Совпадающих пар для обучения: {len(left):,}", flush=True)

    # Категория берётся прямо из текста карточки — она там первым полем после названия.
    groups: dict[str, list[int]] = {}
    for index, text in enumerate(left):
        found = CATEGORY.search(text)
        groups.setdefault(found.group(1).strip() if found else "?", []).append(index)
    groups = {name: np.asarray(rows) for name, rows in groups.items() if len(rows) > 1}
    sizes = sorted((len(v) for v in groups.values()), reverse=True)
    print(f"Категорий в обучении: {len(groups)}, крупнейшие: {sizes[:5]}", flush=True)

    tokenizer = AutoTokenizer.from_pretrained(args.base_model, local_files_only=True)
    model = AutoModel.from_pretrained(args.base_model, local_files_only=True).to(device)
    model.train()
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=0.01)
    scaler = torch.amp.GradScaler("cuda", enabled=device.type == "cuda")

    def embed(strings: list[str]) -> "torch.Tensor":
        encoded = tokenizer(strings, padding=True, truncation=True, max_length=args.max_length,
                            return_tensors="pt").to(device)
        hidden = model(**encoded).last_hidden_state
        mask = encoded["attention_mask"].unsqueeze(-1).float()
        pooled = (hidden * mask).sum(1) / mask.sum(1).clamp(min=1e-9)
        return F.normalize(pooled, dim=-1)

    def step(rows) -> float:
        a = embed([left[i] for i in rows])
        b = embed([right[i] for i in rows])
        # Каждая строка сравнивается со всеми колонками: на диагонали свой партнёр,
        # остальные в батче — отрицательные примеры.
        logits = a @ b.T / args.temperature
        target = torch.arange(len(rows), device=device)
        loss = 0.5 * (F.cross_entropy(logits, target) + F.cross_entropy(logits.T, target))
        optimizer.zero_grad(set_to_none=True)
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        scaler.step(optimizer)
        scaler.update()
        return float(loss.detach())

    # Короткий прогон до основного: если постановка сломана, это видно за минуту, а не
    # через три часа. Loss контрастной задачи стартует около log(размер батча).
    started = time.perf_counter()
    first = []
    for index, rows in enumerate(batches(len(left), args.batch_size, rng, groups)):
        with torch.autocast(device.type, dtype=torch.float16, enabled=device.type == "cuda"):
            first.append(step(rows))
        if index + 1 >= args.smoke:
            break
    print(f"Проверка: {len(first)} шагов за {time.perf_counter()-started:.0f}с, "
          f"loss {first[0]:.3f} -> {np.mean(first[-5:]):.3f} "
          f"(ожидаемый старт около {np.log(args.batch_size):.2f})", flush=True)
    if not np.isfinite(first).all():
        raise SystemExit("loss разошёлся на проверке — обучение не начинаю")

    total = 0
    for epoch in range(args.epochs):
        started = time.perf_counter()
        for index, rows in enumerate(batches(len(left), args.batch_size, rng, groups)):
            with torch.autocast(device.type, dtype=torch.float16, enabled=device.type == "cuda"):
                value = step(rows)
            total += 1
            if total % 200 == 0:
                rate = total * args.batch_size / (time.perf_counter() - started)
                print(f"  эпоха {epoch+1} шаг {total} loss={value:.4f} ({rate:.0f} пар/с)",
                      flush=True)

    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    model.eval()
    model.save_pretrained(output)
    tokenizer.save_pretrained(output)
    json.dump({"max_length": args.max_length, "mode": "compact", "kind": "biencoder"},
              open(output / "inference_config.json", "w"))

    # Оценка на отложенном фолде: косинус между сторонами пары. Это и есть будущий
    # столбец слитой модели, поэтому считается тем же способом, что пойдёт в бой.
    valid = pd.read_parquet(pack / "llm_valid.parquet")
    vl = [lookup.get(int(i), "") for i in valid["id1"]]
    vr = [lookup.get(int(i), "") for i in valid["id2"]]
    scores = np.empty(len(valid), dtype=np.float32)
    with torch.inference_mode():
        for start in range(0, len(vl), 512):
            stop = min(start + 512, len(vl))
            with torch.autocast(device.type, dtype=torch.float16, enabled=device.type == "cuda"):
                a = embed(vl[start:stop])
                b = embed(vr[start:stop])
            scores[start:stop] = (a * b).sum(-1).float().cpu().numpy()
    np.save(output / "llm_ood_scores.npy", scores)
    valid[["id1", "id2", "label", "target"]].to_parquet(output / "llm_ood_pairs.parquet",
                                                        index=False)
    print(f"Готово: модель и скоры на {len(valid):,} парах фолда сохранены в {output}",
          flush=True)


if __name__ == "__main__":
    main()
