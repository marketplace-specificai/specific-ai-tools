# Copyright(C) 2026 Specific AI Inc. All rights reserved.
# SPDX-License-Identifier: MIT
"""Minimal LlamaCppEmbeddingClassifier usage (GGUF inside the model dir).

Requires: pip install "specific-ai-tools[llamacpp]"

The encoder GGUF must live next to the head artifacts on the HF card / local path
(e.g. ``bert-base-only.gguf``).

Usage:

    .venv/bin/python -m specific_ai_tools.embedding_heads.examples.llamacpp_classifier \\
        --model specific-AI/email-agent-triage \\
        --gguf-filename bert-base-only.gguf
"""

from __future__ import annotations

import argparse
import os

from specific_ai_tools.embedding_heads import LlamaCppEmbeddingClassifier
from specific_ai_tools.embedding_heads.integrations.llamacpp import DEFAULT_GGUF_FILENAME

DEFAULT_MODEL = os.getenv("SPECIFIC_AI_MODEL_CARD", "specific-AI/email-agent-triage")
DEFAULT_GGUF_FILENAME_ENV = os.getenv("SPECIFIC_AI_GGUF_FILENAME", DEFAULT_GGUF_FILENAME)


def build_classifier(
    *,
    model: str,
    gguf_filename: str = DEFAULT_GGUF_FILENAME,
) -> LlamaCppEmbeddingClassifier:
    """Create a :class:`LlamaCppEmbeddingClassifier` for a model dir / HF card."""
    return LlamaCppEmbeddingClassifier(model=model, gguf_filename=gguf_filename)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default=DEFAULT_MODEL, help="HF card or local model dir")
    parser.add_argument(
        "--gguf-filename",
        default=DEFAULT_GGUF_FILENAME_ENV,
        help="GGUF filename inside the model directory",
    )
    parser.add_argument(
        "--text",
        default="Please escalate this ticket to billing",
        help="Text to classify",
    )
    args = parser.parse_args()

    classifier = build_classifier(model=args.model, gguf_filename=args.gguf_filename)
    result = classifier.predict_one(args.text)
    print(
        f"predicted_labels={result.predicted_labels} "
        f"predicted_confidences={result.predicted_confidences} "
        f"all_confidences={result.all_confidences}"
    )
