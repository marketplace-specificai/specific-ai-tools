# Copyright(C) 2026 Specific AI Inc. All rights reserved.
# SPDX-License-Identifier: MIT
"""Architecture-specific classification head strategies."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Mapping

import numpy as np


class HeadStrategy(ABC):
    """Run a CLS (or equivalent) embedding through classification head layers."""

    #: Logical weight names → preferred ``.npy`` filenames (without path).
    npy_files: Mapping[str, str] = {}
    #: Logical weight names → ``safetensors`` tensor keys.
    safetensors_keys: Mapping[str, str] = {}
    #: Attribute on ``*ForSequenceClassification`` holding the encoder trunk
    #: (e.g. ``"bert"``). Used when exporting an encoder-only directory for GGUF.
    encoder_attr: str | None = None

    def __init__(self, weights: Mapping[str, np.ndarray]):
        self.weights = {name: np.asarray(array, dtype=np.float64) for name, array in weights.items()}
        missing = [name for name in self.required_weight_names() if name not in self.weights]
        if missing:
            raise ValueError(f"{type(self).__name__} missing weights: {missing}")

    @classmethod
    def required_weight_names(cls) -> list[str]:
        return list(cls.npy_files.keys())

    @abstractmethod
    def forward(self, embedding: np.ndarray) -> np.ndarray:
        """Forward pass: embedding → class logits. Shape ``(num_labels,)``."""

    def forward_batch(self, embeddings: np.ndarray) -> np.ndarray:
        """Forward pass over a batch of embeddings. Shape ``(batch, num_labels)``."""
        embeddings = np.asarray(embeddings, dtype=np.float64)
        if embeddings.ndim == 1:
            return self.forward(embeddings)[np.newaxis, :]
        return np.stack([self.forward(row) for row in embeddings], axis=0)
