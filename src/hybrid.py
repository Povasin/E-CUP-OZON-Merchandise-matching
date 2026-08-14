"""Shared utilities for robust human + LLM hybrid training.

The LLM source is much larger than the manual source and has a different class
prevalence in every category.  A fixed per-row weight therefore changes both the
source contribution and the class balance in an uncontrolled way.  The helpers in
this module make the intended contribution explicit and provide a product-disjoint
split for out-of-domain validation.
"""
from __future__ import annotations

import numpy as np


LLM_HARD_POSITIVE_THRESHOLD = 0.5
LLM_CONFIDENT_POSITIVE_THRESHOLD = 8.0 / 9.0


def hard_llm_labels(target: np.ndarray) -> np.ndarray:
    """Convert the provided soft LLM score to a binary label."""
    values = np.asarray(target, dtype=np.float32)
    return (values >= LLM_HARD_POSITIVE_THRESHOLD).astype(np.int8)


def confident_llm_mask(target: np.ndarray, tolerance: float = 1e-6) -> np.ndarray:
    """Select the reliable tails: exact negatives and scores at least 8/9."""
    values = np.asarray(target, dtype=np.float32)
    return np.isfinite(values) & (
        (values <= tolerance)
        | (values >= LLM_CONFIDENT_POSITIVE_THRESHOLD - tolerance)
    )


def class_balanced_source_weights(
    human_target: np.ndarray,
    llm_target: np.ndarray,
    llm_strength: float = 0.25,
) -> tuple[np.ndarray, np.ndarray]:
    """Return weights with a controlled LLM contribution for each class.

    Manual rows keep unit weight.  Within each class, all LLM rows together have
    ``llm_strength`` times the total weight of manual rows of that class.  This keeps
    the manual class prevalence intact and makes the strength comparable across
    categories with very different source sizes.
    """
    if llm_strength < 0:
        raise ValueError("llm_strength must be non-negative")
    human = np.asarray(human_target, dtype=np.int8)
    llm = np.asarray(llm_target, dtype=np.int8)
    human_weights = np.ones(len(human), dtype=np.float32)
    llm_weights = np.zeros(len(llm), dtype=np.float32)
    for label in (0, 1):
        human_count = int(np.count_nonzero(human == label))
        llm_mask = llm == label
        llm_count = int(np.count_nonzero(llm_mask))
        if not human_count or not llm_count:
            continue
        llm_weights[llm_mask] = llm_strength * human_count / llm_count
    return human_weights, llm_weights


def reliability_balanced_source_weights(
    human_target: np.ndarray,
    llm_target: np.ndarray,
    llm_soft_target: np.ndarray,
    llm_strength: float = 0.25,
) -> tuple[np.ndarray, np.ndarray]:
    """Class-balanced weights redistributed toward confident soft-label rows."""
    human_weights, llm_weights = class_balanced_source_weights(
        human_target, llm_target, llm_strength
    )
    labels = np.asarray(llm_target, dtype=np.int8)
    soft = np.asarray(llm_soft_target, dtype=np.float32)
    reliability = np.clip(np.abs(soft - 0.5) * 2.0, 0.05, 1.0)
    for label in (0, 1):
        mask = labels == label
        intended = float(llm_weights[mask].sum())
        llm_weights[mask] *= reliability[mask]
        actual = float(llm_weights[mask].sum())
        if actual:
            llm_weights[mask] *= intended / actual
    return human_weights, llm_weights


def product_hash_bucket(ids: np.ndarray, n_folds: int = 3) -> np.ndarray:
    """Stable SplitMix64-style bucket assignment for integer product ids."""
    if n_folds < 2:
        raise ValueError("n_folds must be at least 2")
    values = np.asarray(ids, dtype=np.uint64).copy()
    values ^= values >> np.uint64(30)
    values *= np.uint64(0xBF58476D1CE4E5B9)
    values ^= values >> np.uint64(27)
    values *= np.uint64(0x94D049BB133111EB)
    values ^= values >> np.uint64(31)
    return (values % np.uint64(n_folds)).astype(np.int8)


def product_disjoint_pair_masks(
    id1: np.ndarray,
    id2: np.ndarray,
    holdout_fold: int,
    n_folds: int = 3,
) -> tuple[np.ndarray, np.ndarray]:
    """Return train/valid masks with no product shared between the two sides.

    Validation keeps pairs whose two products belong to ``holdout_fold``.  Training
    keeps pairs whose two products do not belong to that fold.  Cross-fold pairs are
    discarded, which is the price of a strict product-disjoint evaluation.
    """
    if not 0 <= holdout_fold < n_folds:
        raise ValueError("holdout_fold must be in [0, n_folds)")
    left = product_hash_bucket(id1, n_folds)
    right = product_hash_bucket(id2, n_folds)
    valid = (left == holdout_fold) & (right == holdout_fold)
    train = (left != holdout_fold) & (right != holdout_fold)
    return train, valid
