# Copyright(C) 2026 Specific AI Inc. All rights reserved.
# SPDX-License-Identifier: MIT
"""llama-cpp-python integration for CLS embeddings."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Optional

import numpy as np

from specific_ai_tools.embedding_heads.classifier import EmbeddingClassifier
from specific_ai_tools.embedding_heads.model_source import resolve_model_dir

DEFAULT_GGUF_FILENAME = "bert-base-only.gguf"


def _import_llama_cpp():
    try:
        from llama_cpp import LLAMA_POOLING_TYPE_CLS, Llama
    except ImportError as exc:
        raise ImportError(
            "llama-cpp-python is required for LlamaCppEmbeddingClassifier. "
            'Install with: pip install "specific-ai-tools[llamacpp]"'
        ) from exc
    return Llama, LLAMA_POOLING_TYPE_CLS


def resolve_gguf_path(model_dir: Path, gguf_filename: str) -> Path:
    """Resolve ``gguf_filename`` under ``model_dir`` (basename only)."""
    name = Path(gguf_filename).name
    if not name or name in {".", ".."}:
        raise ValueError(f"Invalid gguf_filename: {gguf_filename!r}")
    gguf_path = model_dir / name
    if not gguf_path.is_file():
        raise FileNotFoundError(
            f"GGUF file {name!r} not found under model directory {model_dir}. "
            "Place the encoder GGUF next to the head/tokenizer artifacts "
            "(Hugging Face repo or local path)."
        )
    return gguf_path.resolve()


class LlamaCppEmbeddingClassifier(EmbeddingClassifier):
    """:class:`EmbeddingClassifier` backed by ``llama-cpp-python``.

    Loads the encoder GGUF from the same model directory as the classification
    head (Hugging Face card or local path), with CLS pooling and raw
    (unnormalized) embeddings.

    Note:
        llama.cpp embeds **text** with its own tokenizer. :meth:`get_embeddings`
        uses the original ``texts`` argument and ignores ``input_ids_batch``.

    Parameters
    ----------
    model:
        Hugging Face model card or local directory containing head/tokenizer
        artifacts **and** the GGUF encoder file.
    gguf_filename:
        Name of the GGUF file inside ``model`` (default ``bert-base-only.gguf``).
    n_ctx:
        llama.cpp context size.
    llm:
        Optional pre-constructed ``llama_cpp.Llama`` instance (must have
        ``embedding=True``). When provided, the GGUF file is not loaded.
    """

    def __init__(
        self,
        model: str | Path,
        gguf_filename: str = DEFAULT_GGUF_FILENAME,
        *,
        n_ctx: int = 512,
        llm: Any | None = None,
        max_length: Optional[int] = None,
        calibration_temperature: Optional[float] = None,
        selected_thresholds: Optional[dict[str, float]] = None,
        rejection_label_name: Optional[str] = None,
        confidence_rejection_enabled: Optional[bool] = None,
        **kwargs: Any,
    ):
        Llama, pooling_cls = _import_llama_cpp()
        model_dir = resolve_model_dir(model)
        self.gguf_filename = Path(gguf_filename).name
        self.gguf_path = resolve_gguf_path(model_dir, self.gguf_filename)

        if llm is None:
            self._llm = Llama(
                model_path=str(self.gguf_path),
                embedding=True,
                pooling_type=pooling_cls,
                n_ctx=n_ctx,
                verbose=False,
            )
        else:
            self._llm = llm

        super().__init__(
            model_dir,
            max_length=max_length,
            calibration_temperature=calibration_temperature,
            selected_thresholds=selected_thresholds,
            rejection_label_name=rejection_label_name,
            confidence_rejection_enabled=confidence_rejection_enabled,
            **kwargs,
        )

    def get_embeddings(
        self,
        input_ids_batch: list[list[int]],
        texts: list[str],
    ) -> np.ndarray:
        """Return CLS embeddings for ``texts`` (``input_ids_batch`` unused)."""
        if len(texts) == 0:
            return np.empty((0, 0), dtype=np.float64)
        raw = self._llm.embed(texts, normalize=False)
        arr = np.asarray(raw, dtype=np.float64)
        if arr.ndim == 3:
            # Token-level embeddings: use the CLS (first) token.
            return arr[:, 0, :]
        if arr.ndim != 2:
            raise ValueError(f"Unexpected llama-cpp embedding shape: {arr.shape}")
        return arr
