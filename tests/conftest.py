# Copyright(C) 2026 Specific AI Inc. All rights reserved.
# SPDX-License-Identifier: MIT
"""Pytest fixtures built from self-contained synthetic model artifacts."""

from __future__ import annotations

from pathlib import Path

import pytest
from fixture_data import (
    build_npy_model,
    build_safetensors_model,
    make_weights,
    write_config,
    write_npy_weights,
    write_safetensors_weights,
    write_tokenizer,
)


@pytest.fixture
def tiny_weights():
    return make_weights(seed=0)


@pytest.fixture
def npy_model_dir(tmp_path: Path, tiny_weights) -> Path:
    return build_npy_model(tmp_path / "npy_model", tiny_weights)


@pytest.fixture
def safetensors_model_dir(tmp_path: Path, tiny_weights) -> Path:
    return build_safetensors_model(tmp_path / "st_model", tiny_weights)


@pytest.fixture
def dual_format_dirs(tmp_path: Path, tiny_weights) -> tuple[Path, Path]:
    npy_dir = tmp_path / "dual_npy"
    st_dir = tmp_path / "dual_st"
    npy_dir.mkdir()
    st_dir.mkdir()
    write_config(npy_dir)
    write_config(st_dir)
    write_tokenizer(npy_dir)
    write_tokenizer(st_dir)
    write_npy_weights(npy_dir, tiny_weights)
    write_safetensors_weights(st_dir, tiny_weights)
    return npy_dir, st_dir
