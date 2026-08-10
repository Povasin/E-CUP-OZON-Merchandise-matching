"""Метрика соревнования: Macro Averaged PR-AUC.

PR-AUC (average_precision_score) считается по каждой категории, затем усредняется.
Использование auc() по PR-кривой запрещено (завышает результат) — см. docs/RULES.md.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score


def macro_pr_auc(
    y_true: np.ndarray,
    y_score: np.ndarray,
    categories: np.ndarray,
) -> tuple[float, dict[str, float]]:
    """Вернуть (macro PR-AUC, {категория: PR-AUC}).

    Категории, где присутствует только один класс, пропускаются (PR-AUC не определён).
    """
    df = pd.DataFrame({"y": y_true, "s": y_score, "cat": categories})
    per_cat: dict[str, float] = {}
    for cat, grp in df.groupby("cat"):
        y = grp["y"].to_numpy()
        if len(np.unique(y)) < 2:
            continue
        per_cat[str(cat)] = float(average_precision_score(y, grp["s"].to_numpy()))
    macro = float(np.mean(list(per_cat.values()))) if per_cat else float("nan")
    return macro, per_cat
