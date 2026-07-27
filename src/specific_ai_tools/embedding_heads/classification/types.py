# Copyright(C) 2026 Specific AI Inc. All rights reserved.
# SPDX-License-Identifier: MIT
"""Public classification types."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Prediction:
    """Result of one classification example."""

    predicted_labels: list[str]
    predicted_confidences: dict[str, float]
    all_confidences: dict[str, float]
