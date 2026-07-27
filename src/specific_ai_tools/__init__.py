# Copyright(C) 2026 Specific AI Inc. All rights reserved.
# SPDX-License-Identifier: MIT
"""specific-ai-tools — open toolkit for SpecificAI integrations and edge workflows."""

from specific_ai_tools.embedding_heads import (
    ClassificationHead,
    EmbeddingClassifier,
    Prediction,
)

__version__ = "0.1.0"

__all__ = [
    "ClassificationHead",
    "EmbeddingClassifier",
    "LemonadeEmbeddingClassifier",
    "LlamaCppEmbeddingClassifier",
    "Prediction",
    "__version__",
]


def __getattr__(name: str):
    if name in ("LemonadeEmbeddingClassifier", "LlamaCppEmbeddingClassifier"):
        from specific_ai_tools import embedding_heads

        return getattr(embedding_heads, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
