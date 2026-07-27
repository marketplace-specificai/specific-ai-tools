# Copyright(C) 2026 Specific AI Inc. All rights reserved.
# SPDX-License-Identifier: MIT
"""Classification head: strategy forward pass + platform post-processing."""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import numpy as np

from specific_ai_tools.embedding_heads.classification.artifacts import load_head_weights
from specific_ai_tools.embedding_heads.classification.config import (
    ClassificationConfig,
    load_classification_config,
)
from specific_ai_tools.embedding_heads.classification.postprocessing import postprocess_logits
from specific_ai_tools.embedding_heads.classification.strategies.base import HeadStrategy
from specific_ai_tools.embedding_heads.classification.strategies.registry import (
    create_strategy,
    get_strategy_class,
)
from specific_ai_tools.embedding_heads.classification.types import Prediction


class ClassificationHead:
    """Applies a head strategy to CLS embeddings and formats predictions."""

    def __init__(self, strategy: HeadStrategy, config: ClassificationConfig):
        self.strategy = strategy
        self.config = config

    @classmethod
    def from_model_dir(
        cls,
        model_dir: Path | str,
        *,
        calibration_temperature: Optional[float] = None,
        selected_thresholds: Optional[dict[str, float]] = None,
        rejection_label_name: Optional[str] = None,
        confidence_rejection_enabled: Optional[bool] = None,
    ) -> "ClassificationHead":
        """Build a head by loading config + weights from ``model_dir``."""
        model_dir = Path(model_dir)
        config = load_classification_config(
            model_dir,
            calibration_temperature=calibration_temperature,
            selected_thresholds=selected_thresholds,
            rejection_label_name=rejection_label_name,
            confidence_rejection_enabled=confidence_rejection_enabled,
        )
        strategy_cls = get_strategy_class(
            model_type=config.model_type,
            architectures=config.architectures,
        )
        weights = load_head_weights(model_dir, strategy_cls)
        strategy = create_strategy(
            weights,
            model_type=config.model_type,
            architectures=config.architectures,
        )
        return cls(strategy=strategy, config=config)

    def logits(self, embeddings: np.ndarray) -> np.ndarray:
        """Return class logits for one or more CLS embeddings."""
        return self.strategy.forward_batch(embeddings)

    def predict(self, embeddings: np.ndarray) -> list[Prediction]:
        """Run embeddings through the head and post-process to predictions."""
        return postprocess_logits(self.logits(embeddings), self.config)
