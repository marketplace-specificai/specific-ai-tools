# Copyright(C) 2026 Specific AI Inc. All rights reserved.
# SPDX-License-Identifier: MIT
"""Embedding backend integrations (Lemonade HTTP, llama-cpp-python)."""

from __future__ import annotations

__all__ = [
    "LemonadeEmbeddingClassifier",
    "LlamaCppEmbeddingClassifier",
]


def __getattr__(name: str):
    if name == "LemonadeEmbeddingClassifier":
        from specific_ai_tools.embedding_heads.integrations.lemonade import (
            LemonadeEmbeddingClassifier,
        )

        return LemonadeEmbeddingClassifier
    if name == "LlamaCppEmbeddingClassifier":
        from specific_ai_tools.embedding_heads.integrations.llamacpp import (
            LlamaCppEmbeddingClassifier,
        )

        return LlamaCppEmbeddingClassifier
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
