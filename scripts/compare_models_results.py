# Copyright(C) 2026 Specific AI Inc. All rights reserved.
# SPDX-License-Identifier: MIT
"""Compare Hugging Face transformers vs LemonadeEmbeddingClassifier outputs.

The Hugging Face card must already be split (head ``.npy`` + ``bert-base-only.gguf``).

Example:

    .venv/bin/python scripts/compare_models_results.py \\
        specific-AI/email-agent-phishing-detection \\
        path/to/inputs.json \\
        --lemonade-model-name user.specific-ai-phishing-detection
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
from specific_ai_tools.embedding_heads import LemonadeEmbeddingClassifier
from specific_ai_tools.embedding_heads.classification.config import load_classification_config
from specific_ai_tools.embedding_heads.classification.postprocessing import (
    format_predictions,
    probabilities_from_logits,
)
from specific_ai_tools.embedding_heads.integrations.lemonade import DEFAULT_LEMONADE_BASE_URL
from specific_ai_tools.embedding_heads.integrations.llamacpp import DEFAULT_GGUF_FILENAME
from specific_ai_tools.embedding_heads.model_source import resolve_model_dir


def _require_torch():
    try:
        import torch
    except ImportError as exc:
        raise ImportError(
            'torch is required for transformers comparison. Install with: pip install "specific-ai-tools[split]"'
        ) from exc
    return torch


def default_lemonade_model_name(model_card: str) -> str:
    """Derive a Lemonade model id from the HF card basename."""
    name = model_card.strip().rstrip("/").split("/")[-1]
    return f"user.{name}"


def load_examples(path: Path) -> list[dict[str, Any]]:
    """Load a JSON list of examples (each needs an ``example`` text field)."""
    with path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, list):
        raise ValueError(f"Expected a JSON list in {path}, got {type(payload).__name__}")
    for i, item in enumerate(payload):
        if not isinstance(item, dict) or "example" not in item:
            raise ValueError(f"Item {i} in {path} must be an object with an 'example' field")
    return payload


def confidences_to_ordered_probs(
    all_confidences: dict[str, float],
    id2label: dict[int, str],
) -> np.ndarray:
    """Build a probability vector in label-id order."""
    return np.asarray(
        [float(all_confidences[id2label[i]]) for i in range(len(id2label))],
        dtype=np.float64,
    )


def transformers_probs_and_labels(
    *,
    text: str,
    tokenizer: Any,
    model: Any,
    config: Any,
    device: str,
) -> tuple[np.ndarray, list[str]]:
    """Run HF sequence classification; return ordered probs and predicted labels."""
    torch = _require_torch()
    inputs = tokenizer(text, return_tensors="pt", truncation=True).to(device)
    with torch.no_grad():
        outputs = model(**inputs)
    logits = outputs.logits.squeeze(0).detach().cpu().float().numpy()
    probs = probabilities_from_logits(
        logits,
        is_multilabel=config.is_multilabel,
        temperature=config.calibration_temperature,
    )
    prediction = format_predictions(probs, config)[0]
    return np.asarray(probs, dtype=np.float64), list(prediction.predicted_labels)


def lemonade_probs_and_labels(
    *,
    text: str,
    classifier: LemonadeEmbeddingClassifier,
    id2label: dict[int, str],
) -> tuple[np.ndarray, list[str]]:
    """Run LemonadeEmbeddingClassifier; return ordered probs and predicted labels."""
    prediction = classifier.predict_one(text)
    probs = confidences_to_ordered_probs(prediction.all_confidences, id2label)
    return probs, list(prediction.predicted_labels)


def labels_match(a: list[str], b: list[str]) -> bool:
    return sorted(a) == sorted(b)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "model_card",
        help="Hugging Face model card (already split: head .npy + bert-base-only.gguf)",
    )
    parser.add_argument(
        "inputs_json",
        type=Path,
        help="JSON list of objects with an 'example' text field",
    )
    parser.add_argument(
        "--lemonade-model-name",
        default=None,
        help="Lemonade model id (default: user.<model-card-basename>)",
    )
    parser.add_argument(
        "--lemonade-base-url",
        default=DEFAULT_LEMONADE_BASE_URL,
        help=f"Lemonade Server URL (default: {DEFAULT_LEMONADE_BASE_URL})",
    )
    parser.add_argument(
        "--gguf-filename",
        default=DEFAULT_GGUF_FILENAME,
        help=f"GGUF filename on the card (default: {DEFAULT_GGUF_FILENAME})",
    )
    parser.add_argument(
        "--tolerance",
        type=float,
        default=0.05,
        help="Absolute tolerance for probability comparison (default: 0.05)",
    )
    parser.add_argument(
        "--device",
        default="cpu",
        help="Torch device for the transformers model (default: cpu)",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Optional max number of examples to evaluate",
    )
    args = parser.parse_args(argv)

    _require_torch()
    from transformers import AutoModelForSequenceClassification, AutoTokenizer

    examples = load_examples(args.inputs_json.expanduser().resolve())
    if args.limit is not None:
        examples = examples[: max(args.limit, 0)]

    model_card = args.model_card
    lemonade_model_name = args.lemonade_model_name or default_lemonade_model_name(model_card)
    checkpoint = f"{model_card}:{Path(args.gguf_filename).name}"

    model_dir = resolve_model_dir(model_card)
    class_config = load_classification_config(model_dir)
    id2label = class_config.id2label

    print(f"Model card: {model_card}")
    print(f"Resolved model dir: {model_dir}")
    print(f"Lemonade model: {lemonade_model_name}")
    print(f"Checkpoint: {checkpoint}")
    print(f"Examples: {len(examples)}")

    tokenizer = AutoTokenizer.from_pretrained(model_card)
    hf_model = AutoModelForSequenceClassification.from_pretrained(model_card).to(args.device)
    hf_model.eval()

    lemonade = LemonadeEmbeddingClassifier(
        lemonade_model_name=lemonade_model_name,
        checkpoint=checkpoint,
        lemonade_base_url=args.lemonade_base_url,
        model=model_dir,
    )

    matched_labels = 0
    matched_probs = 0
    total = 0

    for ex in examples:
        text = str(ex["example"])
        lemonade_probs, lemonade_labels = lemonade_probs_and_labels(
            text=text,
            classifier=lemonade,
            id2label=id2label,
        )
        hf_probs, hf_labels = transformers_probs_and_labels(
            text=text,
            tokenizer=tokenizer,
            model=hf_model,
            config=class_config,
            device=args.device,
        )

        total += 1
        label_ok = labels_match(lemonade_labels, hf_labels)
        probs_ok = bool(np.allclose(lemonade_probs, hf_probs, atol=args.tolerance))

        if label_ok:
            matched_labels += 1
        else:
            print(f"LABEL MISMATCH\n  text={text[:200]!r}...\n  lemonade={lemonade_labels} transformers={hf_labels}")

        if probs_ok:
            matched_probs += 1
        else:
            print(f"PROB MISMATCH\n  text={text[:200]!r}...\n  lemonade={lemonade_probs}\n  transformers={hf_probs}")

    print(f"Matching labels (lemonade vs transformers): {matched_labels}/{total}")
    print(f"Probability within atol={args.tolerance}: {matched_probs}/{total}")

    return 0 if matched_labels == total and matched_probs == total else 1


if __name__ == "__main__":
    raise SystemExit(main())
