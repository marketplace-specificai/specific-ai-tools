# Copyright(C) 2026 Specific AI Inc. All rights reserved.
# SPDX-License-Identifier: MIT
"""Logit → probability → label selection (platform-parity)."""

from __future__ import annotations

import numpy as np

from specific_ai_tools.embedding_heads.classification.config import ClassificationConfig
from specific_ai_tools.embedding_heads.classification.types import Prediction


def softmax_with_temperature(logits: np.ndarray, temperature: float) -> np.ndarray:
    """Numerically stable softmax after dividing logits by temperature."""
    scaled = logits / max(float(temperature), 1e-6)
    row_max = np.max(scaled, axis=-1, keepdims=True)
    row_max = np.where(np.isfinite(row_max), row_max, 0.0)
    exp_scaled = np.exp(scaled - row_max)
    exp_scaled = np.nan_to_num(exp_scaled, nan=0.0, posinf=0.0, neginf=0.0)
    denom = np.sum(exp_scaled, axis=-1, keepdims=True)
    denom = np.where(denom > 0, denom, 1.0)
    return exp_scaled / denom


def sigmoid_with_temperature(logits: np.ndarray, temperature: float) -> np.ndarray:
    """Numerically stable sigmoid after dividing logits by temperature."""
    x = logits / max(float(temperature), 1e-6)
    pos_mask = x >= 0
    result = np.empty_like(x, dtype=np.float64)
    result[pos_mask] = 1.0 / (1.0 + np.exp(-x[pos_mask]))
    result[~pos_mask] = np.exp(x[~pos_mask]) / (1.0 + np.exp(x[~pos_mask]))
    return result


def probabilities_from_logits(
    logits: np.ndarray,
    *,
    is_multilabel: bool,
    temperature: float,
) -> np.ndarray:
    """Convert logits to probabilities (softmax or sigmoid)."""
    logits = np.asarray(logits, dtype=np.float64)
    if is_multilabel:
        return sigmoid_with_temperature(logits, temperature)
    return softmax_with_temperature(logits, temperature)


def format_predictions(
    probs: np.ndarray,
    config: ClassificationConfig,
) -> list[Prediction]:
    """Apply per-label thresholds / rejection using platform rules.

    Parameters
    ----------
    probs:
        Shape ``(batch, num_labels)`` or ``(num_labels,)``.
    """
    probs = np.asarray(probs, dtype=np.float64)
    if probs.ndim == 1:
        probs = probs[np.newaxis, :]
    if probs.ndim != 2:
        raise ValueError(f"Expected probs with shape (batch, labels), got {probs.shape}")

    thresholds = np.asarray(config.thresholds_array(), dtype=np.float64)
    label_list = config.label_list()
    predictions: list[Prediction] = []

    for i in range(probs.shape[0]):
        row = probs[i]
        all_confidences = {label_list[j]: float(row[j]) for j in range(len(label_list))}

        if config.is_multilabel:
            positive_indices = np.where(row >= thresholds)[0]
            labels = [label_list[int(j)] for j in positive_indices]
            predicted_confidences = {label: all_confidences[label] for label in labels}
            predictions.append(
                Prediction(
                    predicted_labels=labels,
                    predicted_confidences=predicted_confidences,
                    all_confidences=all_confidences,
                )
            )
            continue

        best_idx = int(np.argmax(row))
        best_label = label_list[best_idx]
        best_prob = float(row[best_idx])

        if not config.confidence_rejection_enabled or best_prob >= float(thresholds[best_idx]):
            labels = [best_label]
            predicted_confidences = {best_label: best_prob}
        elif config.rejection_label_name and config.rejection_label_name in config.label2id:
            labels = [config.rejection_label_name]
            predicted_confidences = {config.rejection_label_name: 0.0}
        else:
            labels = []
            predicted_confidences = {}

        predictions.append(
            Prediction(
                predicted_labels=labels,
                predicted_confidences=predicted_confidences,
                all_confidences=all_confidences,
            )
        )

    return predictions


def postprocess_logits(
    logits: np.ndarray,
    config: ClassificationConfig,
) -> list[Prediction]:
    """Temperature-scale logits then format predictions."""
    logits = np.asarray(logits, dtype=np.float64)
    if logits.ndim == 1:
        logits = logits[np.newaxis, :]
    probs = probabilities_from_logits(
        logits,
        is_multilabel=config.is_multilabel,
        temperature=config.calibration_temperature,
    )
    return format_predictions(probs, config)
