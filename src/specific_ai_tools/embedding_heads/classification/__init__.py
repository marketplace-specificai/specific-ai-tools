# Copyright(C) 2026 Specific AI Inc. All rights reserved.
# SPDX-License-Identifier: MIT
"""Classification subpackage exports."""

from specific_ai_tools.embedding_heads.classification.config import ClassificationConfig
from specific_ai_tools.embedding_heads.classification.head import ClassificationHead
from specific_ai_tools.embedding_heads.classification.types import Prediction

__all__ = [
    "ClassificationConfig",
    "ClassificationHead",
    "Prediction",
]
