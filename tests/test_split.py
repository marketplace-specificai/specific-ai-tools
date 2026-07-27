# Copyright(C) 2026 Specific AI Inc. All rights reserved.
# SPDX-License-Identifier: MIT
"""Tests for embedding_heads.split helpers (no full HF model download)."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
from specific_ai_tools.embedding_heads.classification.strategies.bert import BertHeadStrategy
from specific_ai_tools.embedding_heads.split import (
    extract_head_weights_from_state_dict,
    gguf_outtype_from_config,
    save_head_npy,
)


def test_gguf_outtype_from_config():
    assert gguf_outtype_from_config({}) == "f32"
    assert gguf_outtype_from_config({"torch_dtype": "float16"}) == "f16"
    assert gguf_outtype_from_config({"dtype": "bfloat16"}) == "bf16"
    assert gguf_outtype_from_config({"torch_dtype": "torch.float32"}) == "f32"
    with pytest.raises(ValueError, match="Unsupported dtype"):
        gguf_outtype_from_config({"torch_dtype": "int8"})


def test_extract_and_save_head_npy(tmp_path: Path):
    state = {
        "bert.pooler.dense.weight": np.ones((4, 4), dtype=np.float32),
        "bert.pooler.dense.bias": np.zeros(4, dtype=np.float32),
        "classifier.weight": np.ones((2, 4), dtype=np.float32),
        "classifier.bias": np.zeros(2, dtype=np.float32),
        "bert.embeddings.word_embeddings.weight": np.ones((10, 4), dtype=np.float32),
    }
    weights = extract_head_weights_from_state_dict(state, BertHeadStrategy)
    assert set(weights) == set(BertHeadStrategy.npy_files)
    written = save_head_npy(tmp_path, weights, BertHeadStrategy)
    assert len(written) == 4
    for path in written:
        assert path.is_file()
        loaded = np.load(path)
        assert loaded.ndim >= 1


def test_extract_missing_keys():
    with pytest.raises(KeyError, match="Missing head tensors"):
        extract_head_weights_from_state_dict({}, BertHeadStrategy)
