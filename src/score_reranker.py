"""Score a product-disjoint LLM fold with the multilingual reranker."""
from __future__ import annotations

import argparse
import time

import numpy as np
import pandas as pd

from src.cross_encoder import MultilingualReranker, attach_product_texts
from src.hybrid import confident_llm_mask, hard_llm_labels, product_disjoint_pair_masks
from src.metrics import macro_pr_auc
from src.train_model import category_ranks


def report(name: str, target: np.ndarray, scores: np.ndarray, categories: np.ndarray) -> float:
    macro, _ = macro_pr_auc(target, scores, categories)
    print(f"{name:<28} {macro:.6f}", flush=True)
    return macro


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="models/gte-multilingual-reranker-base")
    parser.add_argument("--items", default="output/llm_fold0_items.parquet")
    parser.add_argument("--pairs", default="output/llm_all_pairs_v6.parquet")
    parser.add_argument("--structured-scores", default="output/boost_v9_fold0_scores.npy")
    parser.add_argument("--output", default="output/gte_fold0_name128.npy")
    parser.add_argument("--indices-output", default="")
    parser.add_argument("--fold", type=int, default=0)
    parser.add_argument("--n-folds", type=int, default=3)
    parser.add_argument("--mode", choices=["name", "compact", "baseline"], default="name")
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--max-length", type=int, default=128)
    parser.add_argument("--sample-per-category", type=int, default=0)
    parser.add_argument("--seed", type=int, default=2026)
    args = parser.parse_args()

    all_pairs = pd.read_parquet(
        args.pairs, columns=["id1", "id2", "target", "category"]
    )
    soft = all_pairs["target"].to_numpy(dtype=np.float32)
    _, fold_mask = product_disjoint_pair_masks(
        all_pairs["id1"].to_numpy(),
        all_pairs["id2"].to_numpy(),
        args.fold,
        args.n_folds,
    )
    fold_rows = np.flatnonzero(confident_llm_mask(soft) & fold_mask)
    pairs = all_pairs.iloc[fold_rows].reset_index(drop=True)

    positions = np.arange(len(pairs))
    if args.sample_per_category:
        rng = np.random.default_rng(args.seed)
        sampled: list[np.ndarray] = []
        categories = pairs["category"].astype(str).to_numpy()
        for category in sorted(np.unique(categories)):
            rows = np.flatnonzero(categories == category)
            sampled.append(
                rng.choice(rows, min(args.sample_per_category, len(rows)), replace=False)
            )
        positions = np.sort(np.concatenate(sampled))
        pairs = pairs.iloc[positions].reset_index(drop=True)

    items = pd.read_parquet(args.items)
    left, right = attach_product_texts(pairs, items, args.mode)
    print(
        f"Scoring {len(pairs)} pairs, mode={args.mode}, max_length={args.max_length}, "
        f"batch={args.batch_size}",
        flush=True,
    )
    started = time.perf_counter()
    scorer = MultilingualReranker(args.model)
    scores = scorer.predict(
        left,
        right,
        batch_size=args.batch_size,
        max_length=args.max_length,
    )
    np.save(args.output, scores)
    if args.indices_output:
        np.save(args.indices_output, positions)

    target = hard_llm_labels(pairs["target"].to_numpy(dtype=np.float32))
    categories = pairs["category"].astype(str).to_numpy()
    structured = np.load(args.structured_scores)[positions]
    structured_macro = report("structured v9", target, structured, categories)
    reranker_macro = report("multilingual reranker", target, scores, categories)
    structured_rank = category_ranks(structured, categories)
    reranker_rank = category_ranks(scores, categories)
    best = (structured_macro, 0.0)
    for weight in (0.05, 0.10, 0.20, 0.30, 0.40, 0.50):
        blend = (1.0 - weight) * structured_rank + weight * reranker_rank
        macro = report(f"rank blend reranker={weight:.2f}", target, blend, categories)
        best = max(best, (macro, weight))
    print(
        f"Best={best[0]:.6f} at reranker={best[1]:.2f}; "
        f"standalone={reranker_macro:.6f}; elapsed={time.perf_counter() - started:.1f}s"
    )


if __name__ == "__main__":
    main()
