"""Дообучение кросс-энкодера на LLM-разметке в полном масштабе (для GPU).

Прошлая попытка дообучения дала прирост в смеси +0.0004 — ниже шума. Но училась она на
38 568 парах в одну эпоху, потому что локальная машина имеет 8 GB и не тянет больше.
Уверенных LLM-пар при этом **9 088 499**. То есть проверен был не метод, а его версия,
урезанная более чем в двести раз.

Этот модуль рассчитан на GPU-машину с 30+ GB RAM (Kaggle). Отличия от `finetune_minilm`:

* пары берутся напрямую из `matches_llm.parquet` без промежуточных кэшей, ограниченных
  под 8 GB, — сборщики кэшей теряли ещё 19.7% пар из-за отбора внутри одной row-группы;
* тексты товаров строятся только для товаров, реально попавших в выборку;
* чекпоинты пишутся регулярно, иначе обрыв сессии Kaggle уничтожает весь прогон;
* валидация строго product-disjoint: товар не может попасть и в train, и в valid.

Оценивается модель дважды, и это принципиально:

  llm-ood   — на LLM-фолде, тот же прокси, по которому калибруется production;
  human     — на 365 654 ручных парах, чьи товары не участвовали в обучении.

Успех — не метрика самой модели (структурная её заведомо бьёт), а прирост СМЕСИ,
повторившийся на всех фолдах. Скоры сохраняются, чтобы смесь считалась без повторного
прогона модели.

Запуск на Kaggle:
    python -m src.train_ce_large --train-pairs 3000000 --epochs 1 --batch-size 256
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

from src.cross_encoder import build_product_texts
from src.hybrid import confident_llm_mask, hard_llm_labels, product_disjoint_pair_masks
from src.metrics import macro_pr_auc


def select_llm_pairs(
    matches_path: str,
    holdout_fold: int,
    n_folds: int,
    train_limit: int,
    valid_limit: int,
    seed: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Уверенные LLM-пары, разделённые product-disjoint, с ограничением объёма."""
    matches = pd.read_parquet(matches_path, columns=["id1", "id2", "target"])
    soft = matches["target"].to_numpy(dtype=np.float32)
    confident = confident_llm_mask(soft)
    matches = matches[confident].reset_index(drop=True)
    matches["label"] = hard_llm_labels(matches["target"].to_numpy(dtype=np.float32))
    print(f"Уверенных LLM-пар: {len(matches):,}", flush=True)

    train_mask, valid_mask = product_disjoint_pair_masks(
        matches["id1"].to_numpy(), matches["id2"].to_numpy(), holdout_fold, n_folds
    )
    rng = np.random.default_rng(seed)

    def take(mask: np.ndarray, limit: int) -> pd.DataFrame:
        rows = np.flatnonzero(mask)
        if limit and limit < len(rows):
            rows = rng.choice(rows, limit, replace=False)
        return matches.iloc[np.sort(rows)].reset_index(drop=True)

    train = take(train_mask, train_limit)
    valid = take(valid_mask, valid_limit)
    print(f"train={len(train):,} (положительных {train['label'].mean() * 100:.1f}%), "
          f"valid={len(valid):,} (положительных {valid['label'].mean() * 100:.1f}%)", flush=True)
    return train, valid


def load_texts_for_ids(items_path: str, needed: set[int], mode: str) -> dict[int, str]:
    """Тексты только для нужных товаров: полный корпус — 13.4M карточек."""
    item_file = pq.ParquetFile(items_path)
    texts: dict[int, str] = {}
    for group in range(item_file.num_row_groups):
        frame = item_file.read_row_group(group).to_pandas()
        frame = frame[frame["id"].isin(needed)]
        if len(frame):
            texts.update(build_product_texts(frame, mode))
        print(f"  row-группа {group}: найдено {len(texts):,} из {len(needed):,}", flush=True)
    return texts


def _fold_pair_files(pack: Path, fold: int) -> tuple[Path, Path]:
    """Файлы пар для нужного фолда: ищутся рядом с пакетом и во всех входных каталогах."""
    if fold == 0:
        return pack / "llm_train.parquet", pack / "llm_valid.parquet"
    roots = [pack, pack.parent, *pack.parent.parent.glob("*")] if pack.exists() else [pack]
    for root in roots:
        train = root / f"llm_train_f{fold}.parquet"
        if train.exists():
            return train, root / f"llm_valid_f{fold}.parquet"
    raise SystemExit(f"Не найден llm_train_f{fold}.parquet рядом с {pack}")


def encode_batch(tokenizer, left: list[str], right: list[str], max_length: int, on_cuda: bool):
    return tokenizer(
        left,
        right,
        padding=True,
        truncation=True,
        max_length=max_length,
        # Фиксированная кратность длины: иначе каждая длина батча заводит свой пул в
        # кэширующем аллокаторе, и память течёт. Заодно включает тензорные ядра.
        pad_to_multiple_of=8 if on_cuda else None,
        return_tensors="pt",
    )


def fit_batch_size(
    model, tokenizer, left, right, device, amp_dtype, requested: int, max_length: int
) -> int:
    """Наибольший батч, который реально помещается в память видеокарты.

    Подбирается пробным шагом вперёд-назад с самыми длинными примерами: если оценивать
    на случайных, батч подойдёт в среднем и упадёт на первом же длинном. Падение по
    памяти в середине эпохи стоит всего прогона, а проба — секунды.
    """
    import torch

    if device.type != "cuda":
        return requested
    longest = np.argsort([len(a) + len(b) for a, b in zip(left, right)])[::-1]
    size = requested
    while size >= 8:
        rows = longest[:size]
        # Шаг делается ПОЛНОСТЬЮ, вместе с optimizer.step(): AdamW заводит два буфера
        # размером с модель, и делает это только на первом реальном шаге. Проба без
        # оптимизатора показывает, что батч влезает, а обучение затем падает.
        probe = torch.optim.AdamW(model.parameters(), lr=1e-8)
        try:
            encoded = encode_batch(
                tokenizer, [left[i] for i in rows], [right[i] for i in rows], max_length, True
            ).to(device)
            with torch.amp.autocast("cuda", dtype=amp_dtype):
                loss = model(**encoded).logits.squeeze(-1).float().mean()
            loss.backward()
            probe.step()
            probe.zero_grad(set_to_none=True)
            model.zero_grad(set_to_none=True)
            del encoded, loss, probe
            torch.cuda.empty_cache()
            if size != requested:
                print(f"Батч уменьшен {requested} -> {size} по объёму видеопамяти", flush=True)
            return size
        except torch.OutOfMemoryError:
            model.zero_grad(set_to_none=True)
            del probe
            torch.cuda.empty_cache()
            size //= 2
    raise SystemExit("Не помещается даже батч 8 — уменьшите --max-length")


def train_epochs(
    model, tokenizer, left, right, labels, device, amp_dtype, scaler,
    epochs: int, batch_size: int, max_length: int, learning_rate: float,
    seed: int, checkpoint_every: int, checkpoint_dir: Path, stage: str,
) -> None:
    """Один этап обучения. Вызывается дважды: LLM-предобучение, затем ручные пары."""
    import torch

    on_cuda = device.type == "cuda"
    steps = epochs * ((len(labels) + batch_size - 1) // batch_size)
    if not steps:
        return
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=0.01)
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer, max_lr=learning_rate, total_steps=steps, pct_start=0.06,
        anneal_strategy="linear",
    )
    loss_fn = torch.nn.BCEWithLogitsLoss()
    labels_t = torch.from_numpy(labels)
    rng = np.random.default_rng(seed)

    step = 0
    t0 = time.perf_counter()
    for epoch in range(epochs):
        model.train()
        order = rng.permutation(len(labels))
        running = 0.0
        for start in range(0, len(order), batch_size):
            rows = order[start:start + batch_size]
            encoded = encode_batch(
                tokenizer, [left[i] for i in rows], [right[i] for i in rows],
                max_length, on_cuda,
            ).to(device)
            with torch.amp.autocast("cuda", dtype=amp_dtype, enabled=on_cuda):
                logits = model(**encoded).logits.squeeze(-1)
                loss = loss_fn(logits.float(), labels_t[rows].to(device))
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()
            optimizer.zero_grad(set_to_none=True)
            running += float(loss.detach())
            step += 1
            if step % 200 == 0:
                rate = step * batch_size / (time.perf_counter() - t0)
                print(f"  [{stage}] эпоха {epoch + 1} шаг {step}/{steps} "
                      f"loss={running / 200:.4f} ({rate:.0f} пар/с)", flush=True)
                running = 0.0
            # Сессия Kaggle обрывается по таймауту; без чекпоинтов теряется весь прогон.
            if checkpoint_every and step % checkpoint_every == 0:
                model.save_pretrained(checkpoint_dir)
                tokenizer.save_pretrained(checkpoint_dir)
    print(f"  [{stage}] заняло {(time.perf_counter() - t0) / 60:.1f} мин", flush=True)


def predict_scores(model, tokenizer, left, right, device, batch_size, max_length,
                   amp_dtype=None) -> np.ndarray:
    import torch

    if amp_dtype is None:
        amp_dtype = torch.float16
    model.eval()
    scores = np.empty(len(left), dtype=np.float32)
    order = np.argsort([len(a) + len(b) for a, b in zip(left, right)])
    on_cuda = device.type == "cuda"
    # Без автокаста инференс идёт в fp32 и не задействует тензорные ядра: прошлый прогон
    # выдал 104 пары/с там, где обучение с fp16 держало 115 — то есть скоринг был втрое
    # дороже, чем нужно, и занимал больше времени, чем само обучение на ручных парах.
    with torch.inference_mode(), torch.amp.autocast("cuda", dtype=amp_dtype, enabled=on_cuda):
        for start in range(0, len(order), batch_size):
            rows = order[start:start + batch_size]
            encoded = encode_batch(
                tokenizer, [left[i] for i in rows], [right[i] for i in rows], max_length, on_cuda
            ).to(device)
            scores[rows] = model(**encoded).logits.squeeze(-1).float().cpu().numpy()
    return scores


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--items", default="assets/items.parquet")
    ap.add_argument("--llm-matches", default="assets/matches_llm.parquet")
    ap.add_argument("--human-items", default="assets/items_human.parquet")
    ap.add_argument("--human-matches", default="assets/matches.parquet")
    # Русская основа вместо англоязычной из baseline организаторов: та кодирует
    # карточку в 219 токенов против 74 и рассыпает русские слова побуквенно.
    ap.add_argument("--base-model", default="DeepPavlov/rubert-base-cased")
    ap.add_argument("--output", default="output/ce_large")
    ap.add_argument("--prepacked", default=None,
                    help="каталог пакета от src.pack_kaggle вместо сырых assets")
    ap.add_argument("--mode", choices=["baseline", "compact", "name"], default="compact")
    ap.add_argument("--max-length", type=int, default=256)
    ap.add_argument("--batch-size", type=int, default=256)
    ap.add_argument("--eval-batch-size", type=int, default=512)
    ap.add_argument("--epochs", type=int, default=1, help="эпохи предобучения на LLM")
    ap.add_argument("--learning-rate", type=float, default=3e-5)
    ap.add_argument("--human-epochs", type=int, default=2, help="эпохи дообучения на ручных парах")
    # Меньший шаг на втором этапе: данных на два порядка меньше, легко разрушить
    # представления, выученные на LLM-предобучении.
    ap.add_argument("--human-learning-rate", type=float, default=1e-5)
    ap.add_argument("--train-pairs", type=int, default=3_000_000)
    ap.add_argument("--valid-pairs", type=int, default=200_000)
    # Loss выходит на полку уже к трети эпохи: упор идёт в качество LLM-разметки, а не
    # в число примеров. Поэтому на более тяжёлой основе берём меньше пар, но не хуже.
    ap.add_argument("--max-train-pairs", type=int, default=1_200_000)
    ap.add_argument("--holdout-fold", type=int, default=0)
    ap.add_argument("--n-folds", type=int, default=3)
    ap.add_argument("--checkpoint-every", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=2026)
    args = ap.parse_args()

    import torch
    from transformers import AutoModelForSequenceClassification, AutoTokenizer

    torch.manual_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    on_cuda = device.type == "cuda"
    # is_bf16_supported() на T4 отвечает True за счёт эмуляции, но нативный bf16 появился
    # только с Ampere (sm_80). На Turing он медленнее fp16, поэтому смотрим на архитектуру.
    use_bf16 = on_cuda and torch.cuda.get_device_capability()[0] >= 8
    amp_dtype = torch.bfloat16 if use_bf16 else torch.float16
    print(f"Устройство: {device}"
          + (f" ({torch.cuda.get_device_name(0)}, amp={amp_dtype})" if on_cuda else ""), flush=True)

    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)

    pack = Path(args.prepacked) if args.prepacked else None
    if pack:
        # Отбор пар и сборка текстов уже сделаны локально (src.pack_kaggle): на медленном
        # канале грузить 4.1 GB полного корпуса ради 6M нужных товаров бессмысленно.
        print(f"Готовый пакет: {pack}", flush=True)
        # Списки пар зависят от фолда, а тексты товаров — нет. Поэтому для фолдов 1 и 2
        # достаточно маленьких файлов рядом, тяжёлые тексты переиспользуются.
        train_file, valid_file = _fold_pair_files(pack, args.holdout_fold)
        print(f"Пары: {train_file.name} / {valid_file.name}", flush=True)
        train_pairs = pd.read_parquet(train_file)
        if args.max_train_pairs and args.max_train_pairs < len(train_pairs):
            train_pairs = train_pairs.sample(
                n=args.max_train_pairs, random_state=args.seed
            ).reset_index(drop=True)
        valid_pairs = pd.read_parquet(valid_file)
        text_frame = pd.read_parquet(pack / "item_texts.parquet")
        texts = dict(zip(text_frame["id"].to_numpy().tolist(), text_frame["text"].tolist()))
        del text_frame
        print(f"train={len(train_pairs):,}, valid={len(valid_pairs):,}, "
              f"текстов={len(texts):,}", flush=True)
    else:
        train_pairs, valid_pairs = select_llm_pairs(
            args.llm_matches, args.holdout_fold, args.n_folds,
            args.train_pairs, args.valid_pairs, args.seed,
        )
        print("Сборка текстов товаров...", flush=True)
        needed = set(pd.unique(np.concatenate([
            train_pairs["id1"].to_numpy(), train_pairs["id2"].to_numpy(),
            valid_pairs["id1"].to_numpy(), valid_pairs["id2"].to_numpy(),
        ])).tolist())
        texts = load_texts_for_ids(args.items, needed, args.mode)

    def sides(frame: pd.DataFrame) -> tuple[list[str], list[str]]:
        return ([texts.get(int(i), "") for i in frame["id1"]],
                [texts.get(int(i), "") for i in frame["id2"]])

    train_left, train_right = sides(train_pairs)
    valid_left, valid_right = sides(valid_pairs)
    train_labels = train_pairs["label"].to_numpy(dtype=np.float32)
    del texts

    tokenizer = AutoTokenizer.from_pretrained(args.base_model)
    model = AutoModelForSequenceClassification.from_pretrained(
        args.base_model, num_labels=1
    ).to(device)

    # Масштабирование градиента нужно только fp16: у bf16 диапазон экспоненты как у fp32.
    scaler = torch.amp.GradScaler("cuda", enabled=on_cuda and not use_bf16)

    batch_size = fit_batch_size(
        model, tokenizer, train_left, train_right, device, amp_dtype,
        args.batch_size, args.max_length,
    )

    print("\nЭтап 1 — предобучение на LLM-разметке", flush=True)
    train_epochs(
        model, tokenizer, train_left, train_right, train_labels, device, amp_dtype, scaler,
        args.epochs, batch_size, args.max_length, args.learning_rate,
        args.seed, args.checkpoint_every, output / "checkpoint", "llm",
    )
    model.save_pretrained(output / "stage1_llm")
    del train_left, train_right, train_labels

    # Ручная разметка на два порядка меньше LLM, но точная. LLM служит предобучением,
    # ручные пары — финальной настройкой; иначе лучший сигнал остаётся неиспользованным.
    if pack:
        human_frame = pd.read_parquet(pack / "human_texts.parquet")
        human_matches = pd.read_parquet(pack / "human_matches.parquet")
        human_texts = dict(zip(human_frame["id"].to_numpy().tolist(), human_frame["text"].tolist()))
        category_by_id = dict(zip(human_frame["id"].to_numpy().tolist(),
                                  human_frame["category"].tolist()))
        del human_frame
    else:
        human_items = pd.read_parquet(
            args.human_items, columns=["id", "name", "attributes", "category"]
        )
        human_matches = pd.read_parquet(args.human_matches, columns=["id1", "id2", "target"])
        human_texts = build_product_texts(human_items, args.mode)
        category_by_id = dict(zip(human_items["id"], human_items["category"].astype(str)))
    human_left = [human_texts.get(int(i), "") for i in human_matches["id1"]]
    human_right = [human_texts.get(int(i), "") for i in human_matches["id2"]]
    human_target = human_matches["target"].to_numpy(dtype=np.float32)
    human_categories = human_matches["id1"].map(category_by_id).astype(str).to_numpy()
    del human_texts

    # Тот же product-disjoint принцип: иначе оценка на ручных парах завышается.
    human_train_mask, human_valid_mask = product_disjoint_pair_masks(
        human_matches["id1"].to_numpy(), human_matches["id2"].to_numpy(),
        args.holdout_fold, args.n_folds,
    )
    human_train_idx = np.flatnonzero(human_train_mask)
    human_valid_idx = np.flatnonzero(human_valid_mask)
    print(f"\nЭтап 2 — дообучение на ручной разметке: "
          f"train={len(human_train_idx):,}, holdout={len(human_valid_idx):,}", flush=True)

    print("Метрика ДО дообучения (только LLM-предобучение):", flush=True)
    # Инференс держит меньше, чем обучение, но запас берём от найденного батча.
    eval_batch = min(args.eval_batch_size, batch_size * 2)
    before = predict_scores(
        model, tokenizer, [human_left[i] for i in human_valid_idx],
        [human_right[i] for i in human_valid_idx], device, eval_batch, args.max_length, amp_dtype,
    )
    macro_before, _ = macro_pr_auc(
        human_target[human_valid_idx].astype(np.int8), before, human_categories[human_valid_idx]
    )
    print(f"  macro PR-AUC = {macro_before:.6f}", flush=True)

    train_epochs(
        model, tokenizer,
        [human_left[i] for i in human_train_idx], [human_right[i] for i in human_train_idx],
        human_target[human_train_idx], device, amp_dtype, scaler,
        args.human_epochs, batch_size, args.max_length, args.human_learning_rate,
        args.seed, args.checkpoint_every, output / "checkpoint", "human",
    )
    model.save_pretrained(output)
    tokenizer.save_pretrained(output)
    (output / "inference_config.json").write_text(
        json.dumps({"max_length": args.max_length, "mode": args.mode})
    )

    print("\nОценка на LLM OOD-фолде...", flush=True)
    llm_scores = predict_scores(
        model, tokenizer, valid_left, valid_right, device, eval_batch, args.max_length, amp_dtype
    )
    np.save(output / "llm_ood_scores.npy", llm_scores)
    valid_pairs[["id1", "id2", "label", "target"]].to_parquet(output / "llm_ood_pairs.parquet")

    print("\nСкоринг всех ручных пар (для расчёта смеси)...", flush=True)
    human_scores = predict_scores(
        model, tokenizer, human_left, human_right, device, eval_batch, args.max_length, amp_dtype
    )
    np.save(output / "human_scores.npy", human_scores)
    np.save(output / "human_valid_idx.npy", human_valid_idx)

    # Честная цифра — только на holdout: остальные ручные пары модель видела на этапе 2.
    macro_after, per_cat = macro_pr_auc(
        human_target[human_valid_idx].astype(np.int8), human_scores[human_valid_idx],
        human_categories[human_valid_idx],
    )
    print(f"\nКросс-энкодер на product-disjoint holdout ручной разметки:")
    print(f"  до дообучения (только LLM) {macro_before:.6f}")
    print(f"  после дообучения           {macro_after:.6f}  ({macro_after - macro_before:+.6f})")
    for category in sorted(per_cat, key=lambda c: per_cat[c]):
        print(f"  {category:<28} {per_cat[category]:.4f}")
    print(f"\nАртефакты сохранены в {output}")
    print("Скоры вне holdout использовать для оценки нельзя — модель на них обучалась.")


if __name__ == "__main__":
    main()
