# Copyright(C) 2026 Specific AI Inc. All rights reserved.
# SPDX-License-Identifier: MIT
"""Classification runtime configuration loaded from Hugging Face artifacts."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

SINGLE_LABEL_PROBLEM_TYPE = "single_label_classification"
MULTI_LABEL_PROBLEM_TYPE = "multi_label_classification"
MULTILABEL_DEFAULT_THRESHOLD = 0.5
SINGLE_LABEL_DEFAULT_THRESHOLD = 0.0


@dataclass
class ClassificationConfig:
    """Labels, problem type, and optional calibration / threshold settings."""

    id2label: dict[int, str]
    label2id: dict[str, int]
    model_type: str
    architectures: list[str] = field(default_factory=list)
    problem_type: str = SINGLE_LABEL_PROBLEM_TYPE
    calibration_temperature: float = 1.0
    selected_thresholds: Optional[dict[str, float]] = None
    rejection_label_name: Optional[str] = None
    confidence_rejection_enabled: bool = True
    raw_config: dict[str, Any] = field(default_factory=dict, repr=False)

    @property
    def is_multilabel(self) -> bool:
        return self.problem_type == MULTI_LABEL_PROBLEM_TYPE

    @property
    def num_labels(self) -> int:
        return len(self.id2label)

    def default_threshold(self) -> float:
        if self.is_multilabel:
            return MULTILABEL_DEFAULT_THRESHOLD
        return SINGLE_LABEL_DEFAULT_THRESHOLD

    def thresholds_array(self) -> list[float]:
        """Per-class thresholds in ``id2label`` index order."""
        default = self.default_threshold()
        selected = self.selected_thresholds or {}
        return [float(selected.get(self.id2label[i], default)) for i in range(self.num_labels)]

    def label_list(self) -> list[str]:
        return [self.id2label[i] for i in range(self.num_labels)]


def _normalize_id2label(raw: dict[Any, Any]) -> dict[int, str]:
    return {int(k): str(v) for k, v in raw.items()}


def _normalize_label2id(raw: dict[Any, Any]) -> dict[str, int]:
    return {str(k): int(v) for k, v in raw.items()}


def _read_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def load_classification_config(
    model_dir: Path | str,
    *,
    calibration_temperature: Optional[float] = None,
    selected_thresholds: Optional[dict[str, float]] = None,
    rejection_label_name: Optional[str] = None,
    confidence_rejection_enabled: Optional[bool] = None,
) -> ClassificationConfig:
    """Load ``ClassificationConfig`` from ``config.json`` (+ optional metadata).

    Optional calibration fields are taken from constructor overrides first,
    then from ``metadata.json`` / ``model_params.json`` when present.
    """
    model_dir = Path(model_dir)
    config_path = model_dir / "config.json"
    if not config_path.is_file():
        raise FileNotFoundError(f"config.json not found under {model_dir}")

    raw = _read_json(config_path)
    extras: dict[str, Any] = {}
    for name in ("metadata.json", "model_params.json"):
        extra_path = model_dir / name
        if extra_path.is_file():
            extras.update(_read_json(extra_path))

    id2label_raw = raw.get("id2label")
    label2id_raw = raw.get("label2id")
    if not id2label_raw or not label2id_raw:
        raise ValueError(f"config.json under {model_dir} is missing id2label and/or label2id")

    problem_type = raw.get("problem_type") or SINGLE_LABEL_PROBLEM_TYPE
    model_type = str(raw.get("model_type") or "")
    architectures = list(raw.get("architectures") or [])

    temp = (
        calibration_temperature if calibration_temperature is not None else extras.get("calibration_temperature", 1.0)
    )
    thresholds = selected_thresholds if selected_thresholds is not None else extras.get("selected_thresholds")
    rejection = rejection_label_name if rejection_label_name is not None else extras.get("rejection_label_name")
    rejection_enabled = (
        confidence_rejection_enabled
        if confidence_rejection_enabled is not None
        else extras.get("confidence_rejection_enabled", True)
    )

    return ClassificationConfig(
        id2label=_normalize_id2label(id2label_raw),
        label2id=_normalize_label2id(label2id_raw),
        model_type=model_type,
        architectures=architectures,
        problem_type=problem_type,
        calibration_temperature=float(temp if temp is not None else 1.0),
        selected_thresholds=thresholds,
        rejection_label_name=rejection,
        confidence_rejection_enabled=bool(rejection_enabled),
        raw_config=raw,
    )
