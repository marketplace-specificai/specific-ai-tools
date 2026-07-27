# Copyright(C) 2026 Specific AI Inc. All rights reserved.
# SPDX-License-Identifier: MIT
"""Load classification head weights from ``.npy`` files or ``safetensors``."""

from __future__ import annotations

from pathlib import Path
from typing import Type

import numpy as np
from safetensors import safe_open

from specific_ai_tools.embedding_heads.classification.strategies.base import HeadStrategy


def _has_npy_bundle(model_dir: Path, strategy_cls: Type[HeadStrategy]) -> bool:
    return all((model_dir / filename).is_file() for filename in strategy_cls.npy_files.values())


def _load_from_npy(model_dir: Path, strategy_cls: Type[HeadStrategy]) -> dict[str, np.ndarray]:
    weights: dict[str, np.ndarray] = {}
    for name, filename in strategy_cls.npy_files.items():
        weights[name] = np.load(model_dir / filename)
    return weights


def _find_safetensors_file(model_dir: Path) -> Path:
    preferred = model_dir / "model.safetensors"
    if preferred.is_file():
        return preferred
    candidates = sorted(model_dir.glob("*.safetensors"))
    if not candidates:
        raise FileNotFoundError(f"No .safetensors file under {model_dir}")
    return candidates[0]


def _load_from_safetensors(model_dir: Path, strategy_cls: Type[HeadStrategy]) -> dict[str, np.ndarray]:
    st_path = _find_safetensors_file(model_dir)
    weights: dict[str, np.ndarray] = {}
    with safe_open(str(st_path), framework="np") as handle:
        available = set(handle.keys())
        for name, key in strategy_cls.safetensors_keys.items():
            if key not in available:
                relevant = sorted(
                    k for k in available if any(part in k for part in ("pooler", "classifier", "pre_classifier"))
                )
                raise KeyError(
                    f"Tensor {key!r} for weight {name!r} not found in {st_path}. Available keys include: {relevant}"
                )
            weights[name] = np.asarray(handle.get_tensor(key), dtype=np.float64)
    return weights


def load_head_weights(
    model_dir: Path | str,
    strategy_cls: Type[HeadStrategy],
) -> dict[str, np.ndarray]:
    """Load head weights, preferring ``.npy`` artifacts then safetensors."""
    model_dir = Path(model_dir)
    if _has_npy_bundle(model_dir, strategy_cls):
        return _load_from_npy(model_dir, strategy_cls)
    return _load_from_safetensors(model_dir, strategy_cls)
