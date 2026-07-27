# Copyright(C) 2026 Specific AI Inc. All rights reserved.
# SPDX-License-Identifier: MIT
"""BERT head strategy: npy vs safetensors weight loading."""

from __future__ import annotations

from pathlib import Path

import numpy as np
from fixture_data import HIDDEN_SIZE, NUM_LABELS
from specific_ai_tools.embedding_heads.classification.artifacts import load_head_weights
from specific_ai_tools.embedding_heads.classification.head import ClassificationHead
from specific_ai_tools.embedding_heads.classification.strategies.bert import BertHeadStrategy


def test_npy_and_safetensors_weights_match(dual_format_dirs: tuple[Path, Path]):
    npy_dir, st_dir = dual_format_dirs
    npy_weights = load_head_weights(npy_dir, BertHeadStrategy)
    st_weights = load_head_weights(st_dir, BertHeadStrategy)

    for key in BertHeadStrategy.required_weight_names():
        assert npy_weights[key].shape == st_weights[key].shape
        np.testing.assert_allclose(npy_weights[key], st_weights[key], rtol=1e-5, atol=1e-5)


def test_bert_forward_npy_matches_safetensors(dual_format_dirs: tuple[Path, Path]):
    npy_dir, st_dir = dual_format_dirs
    npy_strategy = BertHeadStrategy(load_head_weights(npy_dir, BertHeadStrategy))
    st_strategy = BertHeadStrategy(load_head_weights(st_dir, BertHeadStrategy))

    rng = np.random.default_rng(0)
    cls_token = rng.standard_normal(HIDDEN_SIZE).astype(np.float64)

    npy_logits = npy_strategy.forward(cls_token)
    st_logits = st_strategy.forward(cls_token)
    assert npy_logits.shape == (NUM_LABELS,)
    np.testing.assert_allclose(npy_logits, st_logits, rtol=1e-5, atol=1e-5)


def test_classification_head_from_safetensors(safetensors_model_dir: Path):
    head = ClassificationHead.from_model_dir(safetensors_model_dir)
    assert head.config.num_labels == NUM_LABELS
    assert head.config.model_type == "bert"

    rng = np.random.default_rng(1)
    emb = rng.standard_normal((2, HIDDEN_SIZE))
    preds = head.predict(emb)
    assert len(preds) == 2
    assert all(p.predicted_labels for p in preds)


def test_classification_head_prefers_npy(npy_model_dir: Path):
    head = ClassificationHead.from_model_dir(npy_model_dir)
    preds = head.predict(np.zeros(HIDDEN_SIZE))
    assert len(preds) == 1
    assert preds[0].predicted_labels
