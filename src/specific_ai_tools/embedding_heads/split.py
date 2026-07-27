# Copyright(C) 2026 Specific AI Inc. All rights reserved.
# SPDX-License-Identifier: MIT
"""Split a sequence-classification model into head ``.npy`` + encoder-only dir."""

from __future__ import annotations

import json
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Type

import numpy as np

from specific_ai_tools.embedding_heads.classification.strategies.base import HeadStrategy
from specific_ai_tools.embedding_heads.classification.strategies.registry import get_strategy_class

DEFAULT_ENCODER_DIRNAME = "bert-base-only"
DEFAULT_GGUF_FILENAME = "bert-base-only.gguf"

_DTYPE_TO_OUTTYPE = {
    "float32": "f32",
    "f32": "f32",
    "float16": "f16",
    "f16": "f16",
    "half": "f16",
    "bfloat16": "bf16",
    "bf16": "bf16",
}


def _require_torch():
    try:
        import torch
    except ImportError as exc:
        raise ImportError(
            'torch is required to split models. Install with: pip install "specific-ai-tools[split]"'
        ) from exc
    return torch


def gguf_outtype_from_config(raw_config: dict[str, Any]) -> str:
    """Map Hugging Face ``torch_dtype`` / ``dtype`` to convert-hf ``--outtype``."""
    dtype = raw_config.get("torch_dtype") or raw_config.get("dtype") or "float32"
    key = str(dtype).lower().removeprefix("torch.")
    if key not in _DTYPE_TO_OUTTYPE:
        raise ValueError(
            f"Unsupported dtype {dtype!r} for GGUF outtype. "
            f"Pass --outtype explicitly (known: {sorted(set(_DTYPE_TO_OUTTYPE.values()))})."
        )
    return _DTYPE_TO_OUTTYPE[key]


def read_model_config(model_dir: Path) -> dict[str, Any]:
    """Load ``config.json`` from ``model_dir``."""
    config_path = Path(model_dir) / "config.json"
    if not config_path.is_file():
        raise FileNotFoundError(f"config.json not found under {model_dir}")
    with config_path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def resolve_strategy_class(raw_config: dict[str, Any]) -> Type[HeadStrategy]:
    """Resolve the head strategy for a model ``config.json`` payload."""
    return get_strategy_class(
        model_type=str(raw_config.get("model_type") or "") or None,
        architectures=list(raw_config.get("architectures") or []),
    )


def extract_head_weights_from_state_dict(
    state_dict: dict[str, Any],
    strategy_cls: Type[HeadStrategy],
) -> dict[str, np.ndarray]:
    """Pull head tensors named in ``strategy_cls.safetensors_keys`` into NumPy."""
    weights: dict[str, np.ndarray] = {}
    missing: list[str] = []
    for name, key in strategy_cls.safetensors_keys.items():
        tensor = state_dict.get(key)
        if tensor is None:
            missing.append(f"{name} ({key})")
            continue
        if hasattr(tensor, "detach"):
            array = tensor.detach().cpu().numpy()
        else:
            array = np.asarray(tensor)
        weights[name] = np.asarray(array)
    if missing:
        raise KeyError(f"Missing head tensors in state_dict: {missing}")
    return weights


def save_head_npy(
    model_dir: Path | str,
    weights: dict[str, np.ndarray],
    strategy_cls: Type[HeadStrategy],
) -> list[Path]:
    """Write ``.npy`` head files into ``model_dir``; return written paths."""
    model_dir = Path(model_dir)
    model_dir.mkdir(parents=True, exist_ok=True)
    written: list[Path] = []
    for name, filename in strategy_cls.npy_files.items():
        if name not in weights:
            raise KeyError(f"Missing weight {name!r} for {strategy_cls.__name__}")
        path = model_dir / filename
        np.save(path, np.asarray(weights[name]))
        written.append(path)
    return written


def save_encoder_only(
    model: Any,
    tokenizer: Any,
    encoder_dir: Path | str,
    strategy_cls: Type[HeadStrategy],
) -> Path:
    """Save the encoder trunk (+ tokenizer) under ``encoder_dir`` for GGUF convert."""
    if not strategy_cls.encoder_attr:
        raise ValueError(f"{strategy_cls.__name__} does not define encoder_attr")
    encoder = getattr(model, strategy_cls.encoder_attr, None)
    if encoder is None:
        raise AttributeError(
            f"Model has no encoder attribute {strategy_cls.encoder_attr!r} (strategy={strategy_cls.__name__})"
        )
    encoder_dir = Path(encoder_dir)
    encoder_dir.mkdir(parents=True, exist_ok=True)
    encoder.save_pretrained(encoder_dir)
    tokenizer.save_pretrained(encoder_dir)
    return encoder_dir


@dataclass(frozen=True)
class SplitModelResult:
    """Artifacts produced by :func:`split_classification_model`."""

    model_dir: Path
    strategy_cls: Type[HeadStrategy]
    npy_paths: list[Path]
    encoder_dir: Path
    outtype: str


def split_classification_model(
    model_dir: Path | str,
    *,
    encoder_dirname: str = DEFAULT_ENCODER_DIRNAME,
    outtype: str | None = None,
) -> SplitModelResult:
    """Load a classification model, write head ``.npy`` files, save encoder-only dir.

    Parameters
    ----------
    model_dir:
        Directory containing a Hugging Face sequence-classification checkpoint.
    encoder_dirname:
        Subdirectory name under ``model_dir`` for the encoder-only export
        (default ``bert-base-only``). Callers typically delete this after GGUF
        conversion.
    outtype:
        GGUF ``--outtype`` override. When ``None``, derived from ``config.json``.
    """
    _require_torch()
    from transformers import AutoModelForSequenceClassification, AutoTokenizer

    model_dir = Path(model_dir).resolve()
    if not model_dir.is_dir():
        raise FileNotFoundError(f"Model directory not found: {model_dir}")

    raw_config = read_model_config(model_dir)
    strategy_cls = resolve_strategy_class(raw_config)
    resolved_outtype = outtype or gguf_outtype_from_config(raw_config)

    model = AutoModelForSequenceClassification.from_pretrained(model_dir)
    tokenizer = AutoTokenizer.from_pretrained(model_dir)

    weights = extract_head_weights_from_state_dict(model.state_dict(), strategy_cls)
    npy_paths = save_head_npy(model_dir, weights, strategy_cls)

    encoder_dir = model_dir / encoder_dirname
    if encoder_dir.exists():
        shutil.rmtree(encoder_dir)
    save_encoder_only(model, tokenizer, encoder_dir, strategy_cls)

    return SplitModelResult(
        model_dir=model_dir,
        strategy_cls=strategy_cls,
        npy_paths=npy_paths,
        encoder_dir=encoder_dir,
        outtype=resolved_outtype,
    )
