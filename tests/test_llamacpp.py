# Copyright(C) 2026 Specific AI Inc. All rights reserved.
# SPDX-License-Identifier: MIT
"""LlamaCppEmbeddingClassifier tests with a fake llama.cpp backend."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
from fixture_data import HIDDEN_SIZE
from specific_ai_tools.embedding_heads.integrations import llamacpp as llamacpp_mod
from specific_ai_tools.embedding_heads.integrations.llamacpp import (
    LlamaCppEmbeddingClassifier,
    resolve_gguf_path,
)


class FakeLlama:
    def __init__(self, *args, **kwargs):
        self.embed_calls: list[dict] = []
        self.kwargs = kwargs

    def embed(self, texts, normalize=True):
        self.embed_calls.append({"texts": texts, "normalize": normalize})
        if isinstance(texts, str):
            texts = [texts]
        return [(np.arange(HIDDEN_SIZE, dtype=np.float64) * 0.001).tolist() for _ in texts]


def test_resolve_gguf_path(tmp_path: Path):
    gguf = tmp_path / "bert-base-only.gguf"
    gguf.write_bytes(b"fake")
    assert resolve_gguf_path(tmp_path, "bert-base-only.gguf") == gguf.resolve()
    assert resolve_gguf_path(tmp_path, "subdir/bert-base-only.gguf") == gguf.resolve()
    with pytest.raises(FileNotFoundError, match="not found"):
        resolve_gguf_path(tmp_path, "missing.gguf")


def test_llamacpp_predict_embeds_text(npy_model_dir: Path, monkeypatch):
    gguf = npy_model_dir / "bert-base-only.gguf"
    gguf.write_bytes(b"fake")

    def fake_import():
        return FakeLlama, 2

    monkeypatch.setattr(llamacpp_mod, "_import_llama_cpp", fake_import)

    llm = FakeLlama()
    clf = LlamaCppEmbeddingClassifier(
        model=npy_model_dir,
        gguf_filename="bert-base-only.gguf",
        llm=llm,
    )
    assert clf.gguf_path == gguf.resolve()
    preds = clf.predict(["hello", "world"])
    assert len(preds) == 2
    assert len(llm.embed_calls) == 1
    assert llm.embed_calls[0]["normalize"] is False
    assert llm.embed_calls[0]["texts"] == ["hello", "world"]


def test_llamacpp_requires_extra(monkeypatch, npy_model_dir: Path):
    (npy_model_dir / "bert-base-only.gguf").write_bytes(b"x")

    def boom():
        raise ImportError("no llama")

    monkeypatch.setattr(llamacpp_mod, "_import_llama_cpp", boom)
    with pytest.raises(ImportError):
        LlamaCppEmbeddingClassifier(model=npy_model_dir)


def test_llamacpp_missing_gguf(npy_model_dir: Path, monkeypatch):
    monkeypatch.setattr(llamacpp_mod, "_import_llama_cpp", lambda: (FakeLlama, 2))
    with pytest.raises(FileNotFoundError, match="bert-base-only.gguf"):
        LlamaCppEmbeddingClassifier(model=npy_model_dir)
