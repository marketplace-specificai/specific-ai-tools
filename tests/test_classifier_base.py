# Copyright(C) 2026 Specific AI Inc. All rights reserved.
# SPDX-License-Identifier: MIT
"""EmbeddingClassifier orchestration with a stub embedding backend."""

from __future__ import annotations

from pathlib import Path

import numpy as np
from fixture_data import HIDDEN_SIZE
from specific_ai_tools.embedding_heads.classification.head import ClassificationHead
from specific_ai_tools.embedding_heads.classifier import EmbeddingClassifier
from specific_ai_tools.embedding_heads.tokenization import load_tokenizer


class StubEmbeddingClassifier(EmbeddingClassifier):
    def __init__(self, *args, embedding: np.ndarray | None = None, **kwargs):
        self._embedding = embedding
        self.last_input_ids: list[list[int]] | None = None
        self.last_texts: list[str] | None = None
        super().__init__(*args, **kwargs)

    def get_embeddings(
        self,
        input_ids_batch: list[list[int]],
        texts: list[str],
    ) -> np.ndarray:
        self.last_input_ids = input_ids_batch
        self.last_texts = texts
        if self._embedding is not None:
            batch = np.asarray(self._embedding, dtype=np.float64)
            if batch.ndim == 1:
                return np.stack([batch] * len(input_ids_batch), axis=0)
            return batch
        out = np.zeros((len(input_ids_batch), HIDDEN_SIZE), dtype=np.float64)
        for i, ids in enumerate(input_ids_batch):
            out[i, : min(len(ids), HIDDEN_SIZE)] = 0.01
        return out


def test_predict_batch(npy_model_dir: Path):
    head = ClassificationHead.from_model_dir(npy_model_dir)
    tokenizer = load_tokenizer(npy_model_dir)
    clf = StubEmbeddingClassifier(npy_model_dir, head=head, tokenizer=tokenizer)

    preds = clf.predict(["hello world", "another example"])
    assert len(preds) == 2
    assert clf.last_input_ids is not None
    assert clf.last_texts == ["hello world", "another example"]
    assert len(clf.last_input_ids) == 2
    assert all(isinstance(ids, list) and ids for ids in clf.last_input_ids)
    assert all(len(ids) >= 3 for ids in clf.last_input_ids)


def test_predict_one(npy_model_dir: Path):
    head = ClassificationHead.from_model_dir(npy_model_dir)
    tokenizer = load_tokenizer(npy_model_dir)
    rng = np.random.default_rng(42)
    emb = rng.standard_normal(HIDDEN_SIZE)
    clf = StubEmbeddingClassifier(npy_model_dir, head=head, tokenizer=tokenizer, embedding=emb)
    pred = clf.predict_one("escalate to billing")
    assert pred.predicted_labels
    assert pred.predicted_confidences


def test_empty_batch(npy_model_dir: Path):
    head = ClassificationHead.from_model_dir(npy_model_dir)
    tokenizer = load_tokenizer(npy_model_dir)
    clf = StubEmbeddingClassifier(npy_model_dir, head=head, tokenizer=tokenizer)
    assert clf.predict([]) == []
