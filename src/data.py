"""Загрузка данных и построение текстового представления товаров.

Схема (см. docs/ASSETS.md):
  items*.parquet   : id, name, attributes (json-строка), category
  matches*.parquet : id1, id2, target (target есть только в обучающих данных)
"""
from __future__ import annotations

import json
from typing import Optional

import pandas as pd


def _attributes_to_text(raw: object) -> str:
    """Развернуть json-строку атрибутов в плоский текст "ключ значение"."""
    if not raw:
        return ""
    try:
        attrs = json.loads(raw)
    except (ValueError, TypeError):
        return str(raw)
    if isinstance(attrs, dict):
        return " ".join(f"{k} {v}" for k, v in attrs.items())
    return str(attrs)


def build_item_text(items: pd.DataFrame, use_category: bool = False) -> pd.Series:
    """Собрать единый текст на товар: name + развёрнутые attributes (+category).

    category по умолчанию не добавляется: внутри пары обе стороны из одной
    категории, поэтому она не помогает различать совпадение/несовпадение.
    """
    name = items["name"].fillna("").astype(str)
    attrs = items["attributes"].map(_attributes_to_text)
    text = name.str.cat(attrs, sep=" ")
    if use_category:
        text = text.str.cat(items["category"].fillna("").astype(str), sep=" ")
    return text.str.strip()


def load_items(items_path: str, use_category: bool = False) -> pd.DataFrame:
    """Прочитать товары, сохранив сырые поля для обучаемого скорера."""
    items = pd.read_parquet(items_path, columns=["id", "name", "attributes", "category"])
    items["text"] = build_item_text(items, use_category=use_category)
    return items[["id", "name", "attributes", "text", "category"]]


def load_matches(matches_path: str) -> pd.DataFrame:
    """Прочитать пары. Колонка target опциональна (её нет в тесте)."""
    return pd.read_parquet(matches_path)


def attach_texts(matches: pd.DataFrame, items: pd.DataFrame) -> pd.DataFrame:
    """Добавить к парам тексты и категорию.

    ВАЖНО: строки НИКОГДА не отбрасываются — ТЗ требует предсказание для каждой
    пары. Отсутствующий текст заменяется пустой строкой (даст скор 0).
    """
    id_to_text = dict(zip(items["id"], items["text"]))
    id_to_cat = dict(zip(items["id"], items["category"]))
    out = matches.copy()
    out["text1"] = out["id1"].map(id_to_text).fillna("")
    out["text2"] = out["id2"].map(id_to_text).fillna("")
    # категория пары — по первому товару (обе стороны из одной категории)
    out["category"] = out["id1"].map(id_to_cat)
    out["category"] = out["category"].fillna(out["id2"].map(id_to_cat))
    return out
