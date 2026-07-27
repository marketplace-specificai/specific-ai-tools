# Copyright(C) 2026 Specific AI Inc. All rights reserved.
# SPDX-License-Identifier: MIT
"""Tests for temperature-aware post-processing (platform parity)."""

from __future__ import annotations

import numpy as np
from specific_ai_tools.embedding_heads.classification.config import ClassificationConfig
from specific_ai_tools.embedding_heads.classification.postprocessing import (
    format_predictions,
    postprocess_logits,
    sigmoid_with_temperature,
    softmax_with_temperature,
)


def _config(**kwargs) -> ClassificationConfig:
    defaults = dict(
        id2label={0: "a", 1: "b", 2: "c"},
        label2id={"a": 0, "b": 1, "c": 2},
        model_type="bert",
        architectures=["BertForSequenceClassification"],
        problem_type="single_label_classification",
        calibration_temperature=1.0,
        selected_thresholds=None,
        rejection_label_name=None,
        confidence_rejection_enabled=True,
    )
    defaults.update(kwargs)
    return ClassificationConfig(**defaults)


def test_softmax_with_temperature_sums_to_one():
    logits = np.array([1.0, 2.0, 0.5])
    probs = softmax_with_temperature(logits, 1.0)
    assert probs.shape == (3,)
    assert np.isclose(probs.sum(), 1.0)
    assert int(np.argmax(probs)) == 1


def test_softmax_temperature_flattens_distribution():
    logits = np.array([5.0, 1.0, 0.0])
    sharp = softmax_with_temperature(logits, 0.5)
    flat = softmax_with_temperature(logits, 5.0)
    assert sharp[0] > flat[0]


def test_sigmoid_with_temperature():
    logits = np.array([0.0, 10.0, -10.0])
    probs = sigmoid_with_temperature(logits, 1.0)
    assert np.isclose(probs[0], 0.5)
    assert probs[1] > 0.99
    assert probs[2] < 0.01


def test_single_label_argmax():
    config = _config()
    preds = postprocess_logits(np.array([0.1, 2.0, 0.3]), config)
    assert len(preds) == 1
    assert preds[0].predicted_labels == ["b"]
    assert preds[0].predicted_confidences["b"] > 0.5
    assert set(preds[0].all_confidences) == {"a", "b", "c"}


def test_single_label_batch():
    config = _config()
    logits = np.array([[2.0, 0.0, 0.0], [0.0, 0.0, 3.0]])
    preds = postprocess_logits(logits, config)
    assert [p.predicted_labels[0] for p in preds] == ["a", "c"]


def test_single_label_rejection_to_label():
    config = _config(
        selected_thresholds={"a": 0.9, "b": 0.9, "c": 0.9},
        rejection_label_name="a",
        confidence_rejection_enabled=True,
    )
    # low confidence argmax (temperature=1, nearly uniform-ish with small logits)
    logits = np.array([0.01, 0.02, 0.0])
    preds = postprocess_logits(logits, config)
    assert preds[0].predicted_labels == ["a"]
    assert preds[0].predicted_confidences == {"a": 0.0}
    assert set(preds[0].all_confidences) == {"a", "b", "c"}


def test_single_label_rejection_empty_without_rejection_label():
    config = _config(
        selected_thresholds={"a": 0.99, "b": 0.99, "c": 0.99},
        rejection_label_name=None,
        confidence_rejection_enabled=True,
    )
    logits = np.array([0.1, 0.2, 0.15])
    preds = postprocess_logits(logits, config)
    assert preds[0].predicted_labels == []


def test_single_label_rejection_disabled():
    config = _config(
        selected_thresholds={"a": 0.99, "b": 0.99, "c": 0.99},
        confidence_rejection_enabled=False,
    )
    logits = np.array([0.1, 0.2, 0.15])
    preds = postprocess_logits(logits, config)
    assert preds[0].predicted_labels == ["b"]


def test_multilabel_thresholding():
    config = _config(
        problem_type="multi_label_classification",
        selected_thresholds={"a": 0.5, "b": 0.7, "c": 0.5},
    )
    # Choose logits such that after sigmoid: a high, b medium, c low
    logits = np.array([5.0, 0.5, -5.0])
    preds = postprocess_logits(logits, config)
    assert "a" in preds[0].predicted_labels
    assert "c" not in preds[0].predicted_labels
    assert "c" not in preds[0].predicted_confidences
    assert set(preds[0].predicted_confidences) == set(preds[0].predicted_labels)
    assert set(preds[0].all_confidences) == {"a", "b", "c"}


def test_format_predictions_uses_default_multilabel_threshold():
    config = _config(problem_type="multi_label_classification")
    probs = np.array([[0.6, 0.4, 0.8]])
    preds = format_predictions(probs, config)
    assert set(preds[0].predicted_labels) == {"a", "c"}
