# Copyright(C) 2026 Specific AI Inc. All rights reserved.
# SPDX-License-Identifier: MIT
"""BERT sequence-classification head: pooler (dense + tanh) → classifier."""

from __future__ import annotations

import numpy as np

from specific_ai_tools.embedding_heads.classification.strategies.base import HeadStrategy


class BertHeadStrategy(HeadStrategy):
    """Reproduce Hugging Face ``BertForSequenceClassification`` head math.

    ``pooled = tanh(W_pool @ cls + b_pool)``
    ``logits = W_cls @ pooled + b_cls``
    """

    npy_files = {
        "pooler_w": "pooler_w.npy",
        "pooler_b": "pooler_b.npy",
        "classifier_w": "classifier_w.npy",
        "classifier_b": "classifier_b.npy",
    }
    safetensors_keys = {
        "pooler_w": "bert.pooler.dense.weight",
        "pooler_b": "bert.pooler.dense.bias",
        "classifier_w": "classifier.weight",
        "classifier_b": "classifier.bias",
    }
    encoder_attr = "bert"

    def forward(self, embedding: np.ndarray) -> np.ndarray:
        cls_token = np.asarray(embedding, dtype=np.float64).reshape(-1)
        pooled = np.dot(self.weights["pooler_w"], cls_token) + self.weights["pooler_b"]
        pooled = np.tanh(pooled)
        logits = np.dot(self.weights["classifier_w"], pooled) + self.weights["classifier_b"]
        return logits
