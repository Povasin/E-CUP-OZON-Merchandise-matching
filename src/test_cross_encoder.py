"""Tests for deterministic cross-encoder product text construction."""
from __future__ import annotations

import json
import unittest

from src.cross_encoder import baseline_product_text, compact_product_text


class CrossEncoderTextTest(unittest.TestCase):
    def test_baseline_format(self) -> None:
        text = baseline_product_text("Товар", "Категория", json.dumps({"Бренд": "X"}))
        self.assertEqual(text, "Name: Товар Category: Категория Attributes: Бренд: X")

    def test_compact_text_keeps_identity_and_drops_logistics(self) -> None:
        attrs = json.dumps({
            "Бренд": "X", "Артикул": "A-1", "Размер": "42",
            "Страна-изготовитель": "Россия", "Длина упаковки": "10 см",
        })
        text = compact_product_text("Кроссовки", "Обувь", attrs)
        self.assertIn("Артикул: A-1", text)
        self.assertIn("Бренд: X", text)
        self.assertIn("Размер: 42", text)
        self.assertNotIn("Страна", text)
        self.assertNotIn("упаковки", text)


if __name__ == "__main__":
    unittest.main()
