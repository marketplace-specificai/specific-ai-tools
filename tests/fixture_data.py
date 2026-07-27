# Copyright(C) 2026 Specific AI Inc. All rights reserved.
# SPDX-License-Identifier: MIT
"""Builders for self-contained embedding_heads test model directories."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from safetensors.numpy import save_file
from specific_ai_tools.embedding_heads.classification.strategies.bert import BertHeadStrategy

HIDDEN_SIZE = 32
NUM_LABELS = 3
VOCAB = [
    "[PAD]",
    "[UNK]",
    "[CLS]",
    "[SEP]",
    "[MASK]",
    "hello",
    "world",
    "another",
    "example",
    "escalate",
    "to",
    "billing",
]


def make_weights(seed: int = 0) -> dict[str, np.ndarray]:
    rng = np.random.default_rng(seed)
    return {
        "pooler_w": rng.standard_normal((HIDDEN_SIZE, HIDDEN_SIZE)).astype(np.float32),
        "pooler_b": rng.standard_normal(HIDDEN_SIZE).astype(np.float32),
        "classifier_w": rng.standard_normal((NUM_LABELS, HIDDEN_SIZE)).astype(np.float32),
        "classifier_b": rng.standard_normal(NUM_LABELS).astype(np.float32),
    }


def write_config(model_dir: Path) -> None:
    id2label = {str(i): f"label_{i}" for i in range(NUM_LABELS)}
    label2id = {v: int(k) for k, v in id2label.items()}
    config = {
        "architectures": ["BertForSequenceClassification"],
        "model_type": "bert",
        "hidden_size": HIDDEN_SIZE,
        "num_labels": NUM_LABELS,
        "id2label": id2label,
        "label2id": label2id,
        "problem_type": "single_label_classification",
        "max_position_embeddings": 128,
        "vocab_size": len(VOCAB),
    }
    (model_dir / "config.json").write_text(json.dumps(config), encoding="utf-8")


def write_tokenizer(model_dir: Path) -> None:
    (model_dir / "vocab.txt").write_text("\n".join(VOCAB) + "\n", encoding="utf-8")
    (model_dir / "tokenizer_config.json").write_text(
        json.dumps(
            {
                "do_lower_case": True,
                "model_max_length": 128,
                "tokenizer_class": "BertTokenizer",
            }
        ),
        encoding="utf-8",
    )
    (model_dir / "special_tokens_map.json").write_text(
        json.dumps(
            {
                "cls_token": "[CLS]",
                "mask_token": "[MASK]",
                "pad_token": "[PAD]",
                "sep_token": "[SEP]",
                "unk_token": "[UNK]",
            }
        ),
        encoding="utf-8",
    )


def write_npy_weights(model_dir: Path, weights: dict[str, np.ndarray]) -> None:
    for name, filename in BertHeadStrategy.npy_files.items():
        np.save(model_dir / filename, weights[name])


def write_safetensors_weights(model_dir: Path, weights: dict[str, np.ndarray]) -> None:
    tensors = {BertHeadStrategy.safetensors_keys[name]: array for name, array in weights.items()}
    save_file(tensors, str(model_dir / "model.safetensors"))


def build_npy_model(model_dir: Path, weights: dict[str, np.ndarray] | None = None) -> Path:
    model_dir.mkdir(parents=True, exist_ok=True)
    weights = weights or make_weights()
    write_config(model_dir)
    write_tokenizer(model_dir)
    write_npy_weights(model_dir, weights)
    return model_dir


def build_safetensors_model(model_dir: Path, weights: dict[str, np.ndarray] | None = None) -> Path:
    model_dir.mkdir(parents=True, exist_ok=True)
    weights = weights or make_weights()
    write_config(model_dir)
    write_tokenizer(model_dir)
    write_safetensors_weights(model_dir, weights)
    return model_dir
