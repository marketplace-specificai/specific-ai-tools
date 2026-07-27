# Copyright(C) 2026 Specific AI Inc. All rights reserved.
# SPDX-License-Identifier: MIT
"""Minimal LemonadeEmbeddingClassifier usage (Lemonade Server HTTP API).

The classifier auto-pulls/loads the GGUF encoder with:
  --embd-normalize "-1" --pooling cls

Usage (Lemonade Server running on localhost:13305):

    .venv/bin/python -m specific_ai_tools.embedding_heads.examples.lemonade_classifier
"""

from __future__ import annotations

import os

from specific_ai_tools.embedding_heads import LemonadeEmbeddingClassifier

DEFAULT_LEMONADE_MODEL_NAME = os.getenv("SPECIFIC_AI_LEMONADE_MODEL_NAME", "user.email-agent-triage")
DEFAULT_CHECKPOINT = os.getenv(
    "SPECIFIC_AI_CHECKPOINT",
    "specific-AI/email-agent-triage:bert-base-only.gguf",
)
DEFAULT_BASE_URL = os.getenv("LEMONADE_BASE_URL", "http://localhost:13305")


def build_classifier(
    *,
    lemonade_model_name: str | None = None,
    checkpoint: str | None = None,
    lemonade_base_url: str | None = None,
) -> LemonadeEmbeddingClassifier:
    """Create a :class:`LemonadeEmbeddingClassifier` talking to Lemonade over HTTP."""
    return LemonadeEmbeddingClassifier(
        lemonade_model_name=lemonade_model_name or DEFAULT_LEMONADE_MODEL_NAME,
        checkpoint=checkpoint or DEFAULT_CHECKPOINT,
        lemonade_base_url=lemonade_base_url or DEFAULT_BASE_URL,
    )


if __name__ == "__main__":
    classifier = build_classifier()
    result = classifier.predict_one("Please escalate this ticket to billing")
    print(
        f"predicted_labels={result.predicted_labels} "
        f"predicted_confidences={result.predicted_confidences} "
        f"all_confidences={result.all_confidences}"
    )
