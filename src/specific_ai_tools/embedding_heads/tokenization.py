# Copyright(C) 2026 Specific AI Inc. All rights reserved.
# SPDX-License-Identifier: MIT
"""Tokenizer helpers for producing batched ``input_ids``."""

from __future__ import annotations

from pathlib import Path
from typing import Optional, Sequence, Union

from transformers import AutoTokenizer, PreTrainedTokenizerBase

TextInput = Union[str, Sequence[str]]


def load_tokenizer(model_dir: Path | str) -> PreTrainedTokenizerBase:
    """Load a Hugging Face tokenizer from ``model_dir``."""
    return AutoTokenizer.from_pretrained(str(model_dir))


def encode_texts(
    tokenizer: PreTrainedTokenizerBase,
    texts: TextInput,
    *,
    max_length: Optional[int] = None,
    truncation: bool = True,
    add_special_tokens: bool = True,
) -> list[list[int]]:
    """Encode one or more texts to ``input_ids`` (list of token-id lists).

    Does not pad — each sequence keeps its natural length (suitable for
    Lemonade ``/v1/embeddings`` token-id ``input``).
    """
    if isinstance(texts, str):
        text_list: list[str] = [texts]
    else:
        text_list = list(texts)

    encode_kwargs: dict = {
        "add_special_tokens": add_special_tokens,
        "truncation": truncation,
    }
    if max_length is not None:
        encode_kwargs["max_length"] = max_length
    else:
        # Respect tokenizer/model context when available.
        model_max = getattr(tokenizer, "model_max_length", None)
        if isinstance(model_max, int) and 0 < model_max < 1_000_000:
            encode_kwargs["max_length"] = model_max

    encoded = tokenizer(
        text_list,
        padding=False,
        return_attention_mask=False,
        return_token_type_ids=False,
        **encode_kwargs,
    )
    return [list(ids) for ids in encoded["input_ids"]]
