"""Train category-specific affine heads on frozen multilingual pair embeddings."""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

from src.cross_encoder import MultilingualReranker, attach_product_texts
from src.hybrid import confident_llm_mask, hard_llm_labels, product_disjoint_pair_masks
from src.metrics import macro_pr_auc
from src.train_model import category_ranks


def encode_or_load(
    path: str,
    pairs: pd.DataFrame,
    items: pd.DataFrame,
    scorer: MultilingualReranker,
    mode: str,
    batch_size: int,
    max_length: int,
) -> np.ndarray:
    output = Path(path)
    if output.exists():
        values = np.load(output, mmap_mode="r")
        if values.shape != (len(pairs), scorer.model.config.hidden_size):
            raise ValueError(f"Unexpected cached embedding shape {values.shape}")
        print(f"Loaded {output}: {values.shape}", flush=True)
        return values
    left, right = attach_product_texts(pairs, items, mode)
    values = scorer.encode(left, right, batch_size=batch_size, max_length=max_length)
    np.save(output, values.astype(np.float16))
    print(f"Saved {output}: {values.shape}", flush=True)
    return values


def report(name: str, target: np.ndarray, scores: np.ndarray, categories: np.ndarray) -> float:
    macro, _ = macro_pr_auc(target, scores, categories)
    print(f"{name:<28} {macro:.6f}", flush=True)
    return macro


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="models/gte-multilingual-reranker-base")
    parser.add_argument("--train-pairs", default="output/gte_train_pairs.parquet")
    parser.add_argument("--train-items", default="output/gte_train_items.parquet")
    parser.add_argument("--train-embeddings", default="output/gte_train_name128.npy")
    parser.add_argument("--all-pairs", default="output/llm_all_pairs_v6.parquet")
    parser.add_argument("--valid-items", default="output/llm_fold0_items.parquet")
    parser.add_argument("--valid-indices", default="output/gte_fold0_sample_indices.npy")
    parser.add_argument("--valid-embeddings", default="output/gte_valid_sample_name128.npy")
    parser.add_argument("--structured-scores", default="output/boost_v9_fold0_scores.npy")
    parser.add_argument("--artifact", default="models/gte_pair_heads.npz")
    parser.add_argument("--mode", choices=["name", "compact", "baseline"], default="name")
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--max-length", type=int, default=128)
    parser.add_argument("--fold", type=int, default=0)
    parser.add_argument("--n-folds", type=int, default=3)
    args = parser.parse_args()

    train_pairs = pd.read_parquet(args.train_pairs)
    train_items = pd.read_parquet(args.train_items)
    all_pairs = pd.read_parquet(
        args.all_pairs, columns=["id1", "id2", "target", "category"]
    )
    soft = all_pairs["target"].to_numpy(dtype=np.float32)
    _, valid_mask = product_disjoint_pair_masks(
        all_pairs["id1"].to_numpy(),
        all_pairs["id2"].to_numpy(),
        args.fold,
        args.n_folds,
    )
    valid_rows = np.flatnonzero(confident_llm_mask(soft) & valid_mask)
    fold_pairs = all_pairs.iloc[valid_rows].reset_index(drop=True)
    positions = np.load(args.valid_indices)
    valid_pairs = fold_pairs.iloc[positions].reset_index(drop=True)
    valid_items = pd.read_parquet(args.valid_items)

    scorer = MultilingualReranker(args.model)
    train_embeddings = encode_or_load(
        args.train_embeddings,
        train_pairs,
        train_items,
        scorer,
        args.mode,
        args.batch_size,
        args.max_length,
    )
    valid_embeddings = encode_or_load(
        args.valid_embeddings,
        valid_pairs,
        valid_items,
        scorer,
        args.mode,
        args.batch_size,
        args.max_length,
    )

    train_target = train_pairs["label"].to_numpy(dtype=np.int8)
    train_categories = train_pairs["category"].astype(str).to_numpy()
    valid_target = hard_llm_labels(valid_pairs["target"].to_numpy(dtype=np.float32))
    valid_categories = valid_pairs["category"].astype(str).to_numpy()
    structured = np.load(args.structured_scores)[positions]
    structured_rank = category_ranks(structured, valid_categories)
    structured_macro = report("structured v9", valid_target, structured, valid_categories)

    best = (structured_macro, 0.0, 0.0)
    artifacts: dict[str, np.ndarray] = {}
    for regularization in (0.003, 0.01, 0.03, 0.1, 0.3, 1.0):
        scores = np.empty(len(valid_pairs), dtype=np.float32)
        coefficients: list[np.ndarray] = []
        intercepts: list[float] = []
        categories = sorted(np.unique(train_categories))
        for category in categories:
            train_rows = np.flatnonzero(train_categories == category)
            valid_rows_in_category = np.flatnonzero(valid_categories == category)
            scaler = StandardScaler().fit(train_embeddings[train_rows])
            model = LogisticRegression(
                C=regularization,
                max_iter=500,
                solver="lbfgs",
                random_state=2026,
            ).fit(scaler.transform(train_embeddings[train_rows]), train_target[train_rows])
            coefficient = model.coef_[0] / scaler.scale_
            intercept = float(model.intercept_[0] - scaler.mean_ @ coefficient)
            scores[valid_rows_in_category] = (
                valid_embeddings[valid_rows_in_category] @ coefficient + intercept
            )
            coefficients.append(coefficient.astype(np.float32))
            intercepts.append(intercept)
        reranker_macro = report(
            f"trained head C={regularization:g}",
            valid_target,
            scores,
            valid_categories,
        )
        trained_rank = category_ranks(scores, valid_categories)
        for weight in (0.1, 0.2, 0.3, 0.4, 0.5):
            blend = (1.0 - weight) * structured_rank + weight * trained_rank
            macro = report(
                f"  blend w={weight:.1f}", valid_target, blend, valid_categories
            )
            if macro > best[0]:
                best = (macro, regularization, weight)
                artifacts = {
                    "categories": np.asarray(categories),
                    "coefficients": np.stack(coefficients),
                    "intercepts": np.asarray(intercepts, dtype=np.float32),
                    "regularization": np.asarray(regularization, dtype=np.float32),
                    "blend_weight": np.asarray(weight, dtype=np.float32),
                    "text_mode": np.asarray(args.mode),
                    "max_length": np.asarray(args.max_length, dtype=np.int32),
                }
        print(f"  standalone={reranker_macro:.6f}", flush=True)

    print(
        f"Best validation={best[0]:.6f}, C={best[1]:g}, blend={best[2]:.1f}"
    )
    if artifacts:
        np.savez_compressed(args.artifact, **artifacts)
        print(f"Saved {args.artifact}")


if __name__ == "__main__":
    main()
