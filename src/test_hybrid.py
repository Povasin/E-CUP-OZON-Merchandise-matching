"""Small regression tests for the hybrid-training invariants."""
from __future__ import annotations

import unittest

import numpy as np

from src.hybrid import (
    class_balanced_source_weights,
    confident_llm_mask,
    hard_llm_labels,
    product_disjoint_pair_masks,
)


class HybridUtilitiesTest(unittest.TestCase):
    def test_confident_tails_and_hard_labels(self) -> None:
        values = np.asarray([0.0, 1 / 9, 4 / 9, 5 / 9, 7 / 9, 8 / 9, 1.0])
        np.testing.assert_array_equal(
            confident_llm_mask(values), [True, False, False, False, False, True, True]
        )
        np.testing.assert_array_equal(hard_llm_labels(values), [0, 0, 0, 1, 1, 1, 1])

    def test_balanced_source_weight_totals(self) -> None:
        human = np.asarray([0, 0, 0, 1, 1], dtype=np.int8)
        llm = np.asarray([0, 0, 1, 1, 1, 1], dtype=np.int8)
        human_weights, llm_weights = class_balanced_source_weights(human, llm, 0.25)
        self.assertAlmostEqual(float(human_weights[human == 0].sum()), 3.0)
        self.assertAlmostEqual(float(human_weights[human == 1].sum()), 2.0)
        self.assertAlmostEqual(float(llm_weights[llm == 0].sum()), 0.75)
        self.assertAlmostEqual(float(llm_weights[llm == 1].sum()), 0.50)

    def test_pair_split_is_product_disjoint(self) -> None:
        id1 = np.arange(1, 500, 2, dtype=np.int64)
        id2 = np.arange(2, 501, 2, dtype=np.int64)
        train, valid = product_disjoint_pair_masks(id1, id2, holdout_fold=1, n_folds=3)
        train_ids = set(id1[train]) | set(id2[train])
        valid_ids = set(id1[valid]) | set(id2[valid])
        self.assertFalse(train_ids & valid_ids)
        self.assertTrue(train.any())
        self.assertTrue(valid.any())


if __name__ == "__main__":
    unittest.main()
