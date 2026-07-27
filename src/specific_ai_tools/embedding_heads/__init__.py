# Copyright(C) 2026 Specific AI Inc. All rights reserved.
# SPDX-License-Identifier: MIT
"""Run classification heads on top of GGUF CLS embeddings (Lemonade / llama.cpp)."""

from specific_ai_tools.embedding_heads.classification.head import ClassificationHead
from specific_ai_tools.embedding_heads.classification.types import Prediction
from specific_ai_tools.embedding_heads.classifier import EmbeddingClassifier

__all__ = [
    "ClassificationHead",
    "EmbeddingClassifier",
    "LemonadeEmbeddingClassifier",
    "LlamaCppEmbeddingClassifier",
    "Prediction",
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
