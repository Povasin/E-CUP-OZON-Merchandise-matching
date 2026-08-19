"""Fast pairwise transformer scoring for product cards.

The official lightweight baseline extracts the CLS vector and then applies a
StandardScaler + LogisticRegression pipeline.  At inference those two operations are
exactly one affine layer, so this implementation keeps the vector on the accelerator
and transfers only one score per pair back to CPU.
"""
from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np
import pandas as pd


DEFAULT_CE_MODEL = "models/cross-encoder-ms-marco-MiniLM-L12-v2"
DEFAULT_CE_HEAD = "models/minilm_l12_head.npz"
DEFAULT_RERANKER_MODEL = "models/gte-multilingual-reranker-base"

# Ordered from identity-defining fields to useful descriptive fields.  Packaging and
# logistics fields are deliberately excluded: they often differ between duplicate cards.
ATTRIBUTE_PRIORITIES = (
    ("артикул", "модель", "партномер", "oem", "sku", "код товара"),
    ("бренд", "производитель", "марка"),
    ("размер", "габарит", "ширина", "высота", "глубина", "длина", "диаметр"),
    ("вес", "объем", "объём", "количество", "комплект", "фасовка", "дозиров"),
    ("цвет", "пол", "коллекция", "серия"),
    ("проба", "вставка", "металл", "покрытие"),
    ("процессор", "память", "накопитель", "емкость", "ёмкость", "диагональ", "разрешение"),
    ("тип", "вид", "назначение", "материал", "состав", "мощность", "вкус", "аромат"),
)
SKIP_ATTRIBUTE_TERMS = (
    "упаковк", "транспорт", "страна", "гарант", "валюта", "сертифик",
    "срок годности", "продав", "аннотац", "описан",
)


def _attributes_dict(raw: object) -> dict[str, str]:
    if raw is None or (isinstance(raw, float) and pd.isna(raw)):
        return {}
    try:
        value = json.loads(str(raw))
    except (TypeError, ValueError, json.JSONDecodeError):
        return {}
    if not isinstance(value, dict):
        return {}
    return {str(key): str(item) for key, item in value.items() if item is not None}


def baseline_product_text(name: object, category: object, raw_attributes: object) -> str:
    """Reproduce the text format used to train the organizer's MiniLM head."""
    attrs = _attributes_dict(raw_attributes)
    attr_text = " ".join(f"{key}: {value}" for key, value in attrs.items())
    return f"Name: {name} Category: {category} Attributes: {attr_text}"


def compact_product_text(name: object, category: object, raw_attributes: object) -> str:
    """Name plus a deterministic, identity-focused subset of attributes."""
    attrs = _attributes_dict(raw_attributes)
    selected: list[tuple[int, str, str]] = []
    for key, value in attrs.items():
        normalized = key.lower().replace("ё", "е")
        if any(term in normalized for term in SKIP_ATTRIBUTE_TERMS):
            continue
        # Берём ВСЁ, кроме заведомо бесполезного, а приоритет лишь задаёт порядок.
        # Прежний отбор по белому списку молча выбрасывал 24% атрибутов — среди них
        # «оем номер» кириллицей (шаблон был написан латиницей) и все списки
        # совместимости: 14 496 значений и 5.7 млн символов, в основном в электронике
        # и автотоварах, где совместимость товар практически и определяет.
        priority = next(
            (row for row, terms in enumerate(ATTRIBUTE_PRIORITIES)
             if any(term in normalized for term in terms)),
            len(ATTRIBUTE_PRIORITIES),
        )
        selected.append((priority, key, value))
    selected.sort(key=lambda item: (item[0], item[1]))
    fields = [f"Название: {name}", f"Категория: {category}"]
    fields.extend(f"{key}: {value}" for _, key, value in selected)
    return " | ".join(fields)


def _text_chunk(rows: list[tuple], mode: str) -> dict[int, str]:
    builder = {"baseline": baseline_product_text, "compact": compact_product_text}.get(mode)
    if builder is None:
        return {int(i): str(n) for i, n, _, _ in rows}
    return {int(i): builder(n, c, a) for i, n, a, c in rows}


def build_product_texts(items: pd.DataFrame, mode: str = "baseline",
                        workers: int | None = None) -> dict[int, str]:
    """Тексты карточек. Разбор атрибутов раскладывается по ядрам: последовательным циклом
    он стоил 69 секунд на объёме теста, а на проверке двадцать ядер."""
    if mode not in {"baseline", "compact", "name"}:
        raise ValueError("CE text mode must be baseline, compact or name")
    rows = list(items[["id", "name", "attributes", "category"]].itertuples(index=False, name=None))
    workers = workers or min(os.cpu_count() or 1, 20)
    if workers > 1 and len(rows) > 20_000:
        from concurrent.futures import ProcessPoolExecutor

        size = (len(rows) + workers - 1) // workers
        chunks = [rows[start:start + size] for start in range(0, len(rows), size)]
        merged: dict[int, str] = {}
        with ProcessPoolExecutor(max_workers=workers) as pool:
            for part in pool.map(_text_chunk, chunks, [mode] * len(chunks)):
                merged.update(part)
        return merged
    result: dict[int, str] = {}
    for item_id, name, attrs, category in rows:
        if mode == "baseline":
            text = baseline_product_text(name, category, attrs)
        elif mode == "compact":
            text = compact_product_text(name, category, attrs)
        else:
            text = str(name)
        result[int(item_id)] = text
    return result


def attach_product_texts(
    matches: pd.DataFrame, items: pd.DataFrame, mode: str = "baseline"
) -> tuple[np.ndarray, np.ndarray]:
    by_id = build_product_texts(items, mode)
    left = matches["id1"].map(by_id).fillna("").astype(str).to_numpy()
    right = matches["id2"].map(by_id).fillna("").astype(str).to_numpy()
    return left, right


class FusedCrossEncoder:
    """AutoModel CLS encoder with a fused scaler/logistic affine head."""

    def __init__(
        self,
        model_path: str | Path = DEFAULT_CE_MODEL,
        head_path: str | Path = DEFAULT_CE_HEAD,
        device: str | None = None,
    ) -> None:
        import torch
        from transformers import AutoModel, AutoTokenizer

        if device is None:
            if torch.cuda.is_available():
                device = "cuda"
            elif torch.backends.mps.is_available():
                device = "mps"
            else:
                device = "cpu"
        self.device = torch.device(device)
        self.tokenizer = AutoTokenizer.from_pretrained(model_path, local_files_only=True)
        dtype = torch.bfloat16 if self.device.type == "cuda" else torch.float32
        self.model = AutoModel.from_pretrained(
            model_path, local_files_only=True, dtype=dtype, attn_implementation="sdpa"
        ).to(self.device).eval()
        artifact = np.load(head_path, allow_pickle=False)
        coefficient = artifact["coefficient"].astype(np.float32)
        if coefficient.shape != (self.model.config.hidden_size,):
            raise ValueError("Cross-encoder head and transformer hidden size differ")
        self.coefficient = torch.from_numpy(coefficient).to(self.device)
        self.intercept = torch.tensor(float(artifact["intercept"]), device=self.device)

    def predict(
        self,
        left: np.ndarray,
        right: np.ndarray,
        batch_size: int = 256,
        max_length: int = 256,
        show_progress: bool = True,
    ) -> np.ndarray:
        import torch

        if len(left) != len(right):
            raise ValueError("Cross-encoder sides have different lengths")
        scores = np.empty(len(left), dtype=np.float32)
        order = np.argsort(
            np.fromiter((len(a) + len(b) for a, b in zip(left, right)), dtype=np.int32)
        )
        iterator = range(0, len(order), batch_size)
        if show_progress:
            from tqdm import tqdm
            iterator = tqdm(iterator, desc="Cross-encoder")
        with torch.inference_mode():
            for start in iterator:
                rows = order[start : start + batch_size]
                encoded = self.tokenizer(
                    left[rows].tolist(),
                    right[rows].tolist(),
                    padding=True,
                    truncation=True,
                    max_length=max_length,
                    pad_to_multiple_of=8 if self.device.type == "cuda" else None,
                    return_tensors="pt",
                )
                encoded = {key: value.to(self.device, non_blocking=True) for key, value in encoded.items()}
                cls = self.model(**encoded, return_dict=True).last_hidden_state[:, 0].float()
                batch_scores = cls @ self.coefficient + self.intercept
                scores[rows] = batch_scores.cpu().numpy()
        return scores


class MultilingualReranker:
    """Direct sequence-classification reranker for multilingual product pairs."""

    def __init__(
        self,
        model_path: str | Path = DEFAULT_RERANKER_MODEL,
        device: str | None = None,
        local_files_only: bool = False,
    ) -> None:
        import torch
        from transformers import AutoModelForSequenceClassification, AutoTokenizer

        if device is None:
            if torch.cuda.is_available():
                device = "cuda"
            elif torch.backends.mps.is_available():
                device = "mps"
            else:
                device = "cpu"
        self.device = torch.device(device)
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_path, local_files_only=local_files_only
        )
        # The model's custom packed-attention implementation produces invalid
        # indices with FP16 on MPS. CUDA uses BF16; local Apple validation stays
        # in FP32 for correctness.
        dtype = torch.bfloat16 if self.device.type == "cuda" else torch.float32
        self.model = AutoModelForSequenceClassification.from_pretrained(
            model_path,
            local_files_only=local_files_only,
            trust_remote_code=True,
            dtype=dtype,
        ).to(self.device).eval()

    def predict(
        self,
        left: np.ndarray,
        right: np.ndarray,
        batch_size: int = 128,
        max_length: int = 256,
        show_progress: bool = True,
    ) -> np.ndarray:
        import torch

        if len(left) != len(right):
            raise ValueError("Reranker sides have different lengths")
        scores = np.empty(len(left), dtype=np.float32)
        order = np.argsort(
            np.fromiter((len(a) + len(b) for a, b in zip(left, right)), dtype=np.int32)
        )
        iterator = range(0, len(order), batch_size)
        if show_progress:
            from tqdm import tqdm
            iterator = tqdm(iterator, desc="Multilingual reranker")
        with torch.inference_mode():
            for start in iterator:
                rows = order[start : start + batch_size]
                encoded = self.tokenizer(
                    left[rows].tolist(),
                    right[rows].tolist(),
                    padding=True,
                    truncation=True,
                    max_length=max_length,
                    pad_to_multiple_of=8 if self.device.type == "cuda" else None,
                    return_tensors="pt",
                )
                encoded = {
                    key: value.to(self.device, non_blocking=True)
                    for key, value in encoded.items()
                }
                # Explicit positions avoid an MPS issue with the expanded
                # non-persistent position-id buffer in the remote GTE code.
                if self.device.type == "mps":
                    encoded["position_ids"] = torch.arange(
                        encoded["input_ids"].shape[1], device=self.device
                    ).unsqueeze(0).expand(encoded["input_ids"].shape[0], -1)
                batch_scores = self.model(**encoded, return_dict=True).logits.view(-1).float()
                scores[rows] = batch_scores.cpu().numpy()
        return scores

    def encode(
        self,
        left: np.ndarray,
        right: np.ndarray,
        batch_size: int = 128,
        max_length: int = 128,
        show_progress: bool = True,
    ) -> np.ndarray:
        """Return the pretrained pair-pooler representation before its task head."""
        import torch

        if len(left) != len(right):
            raise ValueError("Reranker sides have different lengths")
        embeddings = np.empty(
            (len(left), self.model.config.hidden_size), dtype=np.float32
        )
        order = np.argsort(
            np.fromiter((len(a) + len(b) for a, b in zip(left, right)), dtype=np.int32)
        )
        iterator = range(0, len(order), batch_size)
        if show_progress:
            from tqdm import tqdm
            iterator = tqdm(iterator, desc="Reranker embeddings")
        with torch.inference_mode():
            for start in iterator:
                rows = order[start : start + batch_size]
                encoded = self.tokenizer(
                    left[rows].tolist(),
                    right[rows].tolist(),
                    padding=True,
                    truncation=True,
                    max_length=max_length,
                    pad_to_multiple_of=8 if self.device.type == "cuda" else None,
                    return_tensors="pt",
                )
                encoded = {
                    key: value.to(self.device, non_blocking=True)
                    for key, value in encoded.items()
                }
                if self.device.type == "mps":
                    encoded["position_ids"] = torch.arange(
                        encoded["input_ids"].shape[1], device=self.device
                    ).unsqueeze(0).expand(encoded["input_ids"].shape[0], -1)
                pooled = self.model.new(**encoded, return_dict=True).pooler_output.float()
                embeddings[rows] = pooled.cpu().numpy()
        return embeddings
