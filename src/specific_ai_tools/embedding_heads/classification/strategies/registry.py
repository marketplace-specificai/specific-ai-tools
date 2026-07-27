# Copyright(C) 2026 Specific AI Inc. All rights reserved.
# SPDX-License-Identifier: MIT
"""Registry mapping model_type / architecture name → head strategy class."""

from __future__ import annotations

from typing import Type

from specific_ai_tools.embedding_heads.classification.strategies.base import HeadStrategy
from specific_ai_tools.embedding_heads.classification.strategies.bert import BertHeadStrategy

_REGISTRY: dict[str, Type[HeadStrategy]] = {
    "bert": BertHeadStrategy,
    "BertForSequenceClassification": BertHeadStrategy,
    "BertModel": BertHeadStrategy,
}


def register_strategy(key: str, strategy_cls: Type[HeadStrategy]) -> None:
    """Register or override a head strategy for ``key`` (model_type or architecture)."""
    _REGISTRY[key] = strategy_cls


def get_strategy_class(
    *,
    model_type: str | None = None,
    architectures: list[str] | None = None,
) -> Type[HeadStrategy]:
    """Resolve a :class:`HeadStrategy` subclass for the given HF config fields."""
    if architectures:
        for name in architectures:
            if name in _REGISTRY:
                return _REGISTRY[name]
    if model_type and model_type in _REGISTRY:
        return _REGISTRY[model_type]
    raise KeyError(
        "No head strategy registered for "
        f"model_type={model_type!r}, architectures={architectures!r}. "
        f"Known keys: {sorted(_REGISTRY)}"
    )


def create_strategy(
    weights: dict,
    *,
    model_type: str | None = None,
    architectures: list[str] | None = None,
) -> HeadStrategy:
    """Instantiate the strategy matching the model config."""
    strategy_cls = get_strategy_class(model_type=model_type, architectures=architectures)
    return strategy_cls(weights)
