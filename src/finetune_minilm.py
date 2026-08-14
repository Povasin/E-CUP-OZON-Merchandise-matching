"""Fine-tune the compact organizer cross-encoder for strict product equality."""
from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from src.cross_encoder import attach_product_texts
from src.hybrid import confident_llm_mask, hard_llm_labels, product_disjoint_pair_masks
from src.metrics import macro_pr_auc
from src.train_model import category_ranks


class PairDataset(Dataset):
    def __init__(self, left: np.ndarray, right: np.ndarray, labels: np.ndarray) -> None:
        self.left = left
        self.right = right
        self.labels = labels

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, index: int) -> tuple[str, str, float]:
        return str(self.left[index]), str(self.right[index]), float(self.labels[index])


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-model", default="models/cross-encoder-ms-marco-MiniLM-L12-v2")
    parser.add_argument("--output", default="models/minilm-product-matcher")
    parser.add_argument("--train-pairs", default="output/gte_train_pairs.parquet")
    parser.add_argument("--train-items", default="output/gte_train_items.parquet")
    parser.add_argument("--all-pairs", default="output/llm_all_pairs_v6.parquet")
    parser.add_argument("--valid-items", default="output/llm_fold0_items.parquet")
    parser.add_argument("--valid-indices", default="output/gte_fold0_sample_indices.npy")
    parser.add_argument("--structured-scores", default="output/boost_v9_fold0_scores.npy")
    parser.add_argument("--mode", choices=["name", "compact", "baseline"], default="name")
    parser.add_argument("--max-length", type=int, default=128)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--learning-rate", type=float, default=2e-5)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--seed", type=int, default=2026)
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    if torch.cuda.is_available():
        device = torch.device("cuda")
        dtype = torch.bfloat16
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
        dtype = torch.float32
    else:
        device = torch.device("cpu")
        dtype = torch.float32
    print(f"device={device}, dtype={dtype}", flush=True)

    train_pairs = pd.read_parquet(args.train_pairs)
    train_items = pd.read_parquet(args.train_items)
    train_left, train_right = attach_product_texts(train_pairs, train_items, args.mode)
    train_labels = train_pairs["label"].to_numpy(dtype=np.float32)

    all_pairs = pd.read_parquet(
        args.all_pairs, columns=["id1", "id2", "target", "category"]
    )
    soft = all_pairs["target"].to_numpy(dtype=np.float32)
    _, valid_mask = product_disjoint_pair_masks(
        all_pairs["id1"].to_numpy(), all_pairs["id2"].to_numpy(), 0, 3
    )
    fold_rows = np.flatnonzero(confident_llm_mask(soft) & valid_mask)
    positions = np.load(args.valid_indices)
    valid_pairs = all_pairs.iloc[fold_rows].reset_index(drop=True).iloc[positions].reset_index(drop=True)
    valid_items = pd.read_parquet(args.valid_items)
    valid_left, valid_right = attach_product_texts(valid_pairs, valid_items, args.mode)
    valid_labels = hard_llm_labels(valid_pairs["target"].to_numpy(dtype=np.float32))
    valid_categories = valid_pairs["category"].astype(str).to_numpy()

    tokenizer = AutoTokenizer.from_pretrained(args.base_model, local_files_only=True)
    model = AutoModelForSequenceClassification.from_pretrained(
        args.base_model, local_files_only=True, dtype=dtype
    ).to(device)

    def collate(batch: list[tuple[str, str, float]]) -> tuple[dict[str, torch.Tensor], torch.Tensor]:
        left, right, labels = zip(*batch)
        encoded = tokenizer(
            list(left),
            list(right),
            padding=True,
            truncation=True,
            max_length=args.max_length,
            pad_to_multiple_of=8 if device.type == "cuda" else None,
            return_tensors="pt",
        )
        return encoded, torch.tensor(labels, dtype=torch.float32)

    generator = torch.Generator().manual_seed(args.seed)
    loader = DataLoader(
        PairDataset(train_left, train_right, train_labels),
        batch_size=args.batch_size,
        shuffle=True,
        generator=generator,
        collate_fn=collate,
        num_workers=0,
    )
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay
    )
    total_steps = args.epochs * len(loader)
    scheduler = torch.optim.lr_scheduler.LinearLR(
        optimizer, start_factor=1.0, end_factor=0.1, total_iters=max(total_steps, 1)
    )
    loss_function = torch.nn.BCEWithLogitsLoss()

    started = time.perf_counter()
    from tqdm import tqdm
    model.train()
    for epoch in range(args.epochs):
        loss_sum = 0.0
        progress = tqdm(loader, desc=f"Fine-tune epoch {epoch + 1}/{args.epochs}")
        for step, (encoded, labels) in enumerate(progress, start=1):
            encoded = {key: value.to(device) for key, value in encoded.items()}
            labels = labels.to(device)
            optimizer.zero_grad(set_to_none=True)
            logits = model(**encoded, return_dict=True).logits.view(-1).float()
            loss = loss_function(logits, labels)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()
            loss_sum += float(loss.detach().cpu())
            if step % 20 == 0:
                progress.set_postfix(loss=f"{loss_sum / step:.4f}")

    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(output, safe_serialization=True)
    tokenizer.save_pretrained(output)

    model.eval()
    scores = np.empty(len(valid_pairs), dtype=np.float32)
    order = np.argsort(
        np.fromiter(
            (len(left) + len(right) for left, right in zip(valid_left, valid_right)),
            dtype=np.int32,
        )
    )
    with torch.inference_mode():
        for start in tqdm(range(0, len(order), args.batch_size), desc="Validation"):
            rows = order[start : start + args.batch_size]
            encoded = tokenizer(
                valid_left[rows].tolist(),
                valid_right[rows].tolist(),
                padding=True,
                truncation=True,
                max_length=args.max_length,
                return_tensors="pt",
            )
            encoded = {key: value.to(device) for key, value in encoded.items()}
            scores[rows] = model(**encoded, return_dict=True).logits.view(-1).float().cpu().numpy()
    np.save(output / "validation_scores.npy", scores)

    ce_macro, ce_per_category = macro_pr_auc(valid_labels, scores, valid_categories)
    structured = np.load(args.structured_scores)[positions]
    structured_macro, _ = macro_pr_auc(valid_labels, structured, valid_categories)
    structured_rank = category_ranks(structured, valid_categories)
    ce_rank = category_ranks(scores, valid_categories)
    print(f"structured sample={structured_macro:.6f}; fine-tuned CE={ce_macro:.6f}")
    best = (structured_macro, 0.0)
    for weight in (0.05, 0.10, 0.20, 0.30, 0.40, 0.50):
        blend = (1.0 - weight) * structured_rank + weight * ce_rank
        macro, _ = macro_pr_auc(valid_labels, blend, valid_categories)
        print(f"rank blend CE={weight:.2f}: {macro:.6f}")
        best = max(best, (macro, weight))
    print("Per-category fine-tuned CE:")
    for category in sorted(ce_per_category):
        print(f"  {category:<28} {ce_per_category[category]:.6f}")
    print(
        f"Best={best[0]:.6f} at CE={best[1]:.2f}; "
        f"elapsed={time.perf_counter() - started:.1f}s; saved={output}"
    )


if __name__ == "__main__":
    main()
