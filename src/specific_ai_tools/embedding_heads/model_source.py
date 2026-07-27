# Copyright(C) 2026 Specific AI Inc. All rights reserved.
# SPDX-License-Identifier: MIT
"""Resolve a Hugging Face model card or local path to a directory of artifacts."""

from __future__ import annotations

from pathlib import Path

from huggingface_hub import snapshot_download


def resolve_model_dir(model: str | Path) -> Path:
    """Return a local directory for ``model`` (HF repo id or filesystem path).

    Parameters
    ----------
    model:
        Hugging Face repo id (e.g. ``"org/model"``) or a local path containing
        ``config.json`` and head/tokenizer artifacts.
    """
    path = Path(model).expanduser()
    if path.exists() and path.is_dir():
        return path.resolve()

    local_dir = snapshot_download(repo_id=str(model))
    return Path(local_dir).resolve()
