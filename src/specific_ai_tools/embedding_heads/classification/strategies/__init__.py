# Copyright(C) 2026 Specific AI Inc. All rights reserved.
# SPDX-License-Identifier: MIT
"""Head strategy package exports."""

from specific_ai_tools.embedding_heads.classification.strategies.base import HeadStrategy
from specific_ai_tools.embedding_heads.classification.strategies.bert import BertHeadStrategy
from specific_ai_tools.embedding_heads.classification.strategies.registry import (
    create_strategy,
    get_strategy_class,
    register_strategy,
)

__all__ = [
    "BertHeadStrategy",
    "HeadStrategy",
    "create_strategy",
    "get_strategy_class",
    "register_strategy",
]
