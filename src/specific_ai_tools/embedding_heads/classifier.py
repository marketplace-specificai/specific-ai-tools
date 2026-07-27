# Copyright(C) 2026 Specific AI Inc. All rights reserved.
# SPDX-License-Identifier: MIT
"""Base classifier: tokenize → abstract embeddings → classification head."""

from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Optional, Sequence, Union

import numpy as np
from transformers import PreTrainedTokenizerBase

from specific_ai_tools.embedding_heads.classification.head import ClassificationHead
from specific_ai_tools.embedding_heads.classification.types import Prediction
from specific_ai_tools.embedding_heads.model_source import resolve_model_dir
from specific_ai_tools.embedding_heads.tokenization import encode_texts, load_tokenizer

TextInput = Union[str, Sequence[str]]


class EmbeddingClassifier(ABC):
    """Orchestrates tokenization, CLS embedding retrieval, and head forward.

    Subclasses implement :meth:`get_embeddings` to plug in Lemonade Server,
    llama-cpp-python, or any other CLS embedding backend.
    """

    def __init__(
        self,
        model: str | Path,
        *,
        max_length: Optional[int] = None,
        calibration_temperature: Optional[float] = None,
        selected_thresholds: Optional[dict[str, float]] = None,
        rejection_label_name: Optional[str] = None,
        confidence_rejection_enabled: Optional[bool] = None,
        tokenizer: Optional[PreTrainedTokenizerBase] = None,
        head: Optional[ClassificationHead] = None,
    ):
        self.model_dir = resolve_model_dir(model)
        self.max_length = max_length
        self.tokenizer = tokenizer or load_tokenizer(self.model_dir)
        self.head = head or ClassificationHead.from_model_dir(
            self.model_dir,
            calibration_temperature=calibration_temperature,
            selected_thresholds=selected_thresholds,
            rejection_label_name=rejection_label_name,
            confidence_rejection_enabled=confidence_rejection_enabled,
        )

    @abstractmethod
    def get_embeddings(
        self,
        input_ids_batch: list[list[int]],
        texts: list[str],
    ) -> np.ndarray:
        """Return CLS embedding vectors for a batch.

        Parameters
        ----------
        input_ids_batch:
            Input ids lists (same order as ``texts``).
        texts:
            Original input strings (same order as ``input_ids_batch``).

        Subclasses choose which input to use (e.g. Lemonade uses token ids;
        llama-cpp uses raw text).

        Returns
        -------
        np.ndarray
            Shape ``(batch, hidden_size)`` of raw (unnormalized) CLS vectors.
        """

    def encode(self, texts: TextInput) -> list[list[int]]:
        """Tokenize texts into ``input_ids``."""
        return encode_texts(self.tokenizer, texts, max_length=self.max_length)

    def predict(self, texts: TextInput) -> list[Prediction]:
        """Classify one or more texts."""
        if isinstance(texts, str):
            text_list: list[str] = [texts]
        else:
            text_list = list(texts)
        if not text_list:
            return []

        input_ids_batch = self.encode(text_list)
        embeddings = np.asarray(
            self.get_embeddings(input_ids_batch, text_list),
            dtype=np.float64,
        )
        if embeddings.ndim == 1:
            embeddings = embeddings[np.newaxis, :]
        if embeddings.shape[0] != len(text_list):
            raise ValueError(f"get_embeddings returned {embeddings.shape[0]} rows for {len(text_list)} texts")
        return self.head.predict(embeddings)

    def predict_one(self, text: str) -> Prediction:
        """Classify a single text and return the sole :class:`Prediction`."""
        return self.predict([text])[0]
