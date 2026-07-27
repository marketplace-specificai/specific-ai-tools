# Copyright(C) 2026 Specific AI Inc. All rights reserved.
# SPDX-License-Identifier: MIT
"""LemonadeEmbeddingClassifier HTTP client tests (mocked requests.request)."""

from __future__ import annotations

from pathlib import Path
from typing import Any
from urllib.parse import urlparse

import numpy as np
import pytest
from fixture_data import HIDDEN_SIZE
from specific_ai_tools.embedding_heads.integrations import lemonade as lemonade_mod
from specific_ai_tools.embedding_heads.integrations.lemonade import (
    REQUIRED_LLAMACPP_ARGS,
    REQUIRED_RECIPE,
    LemonadeEmbeddingClassifier,
    LemonadeModelError,
    ensure_user_model_prefix,
    normalize_lemonade_base_url,
)


class FakeResponse:
    def __init__(self, payload: Any, status_code: int = 200):
        self._payload = payload
        self.status_code = status_code
        self.ok = 200 <= status_code < 300
        self.text = str(payload)

    def json(self):
        if isinstance(self._payload, Exception):
            raise self._payload
        return self._payload


class FakeRequests:
    def __init__(self, *, models_payload: dict | None = None):
        self.models_payload = models_payload if models_payload is not None else {"data": []}
        self.calls: list[dict] = []

    def request(self, method: str, url: str, json=None, timeout=None):
        self.calls.append({"method": method, "url": url, "json": json, "timeout": timeout})
        path = urlparse(url).path

        if method == "GET" and path == "/v1/models":
            return FakeResponse(self.models_payload)

        if method == "POST" and path == "/api/v1/pull":
            return FakeResponse({"status": "ok"})

        if method == "POST" and path == "/api/v1/load":
            model_name = (json or {}).get("model_name")
            self.models_payload = {
                "data": [
                    {
                        "id": model_name,
                        "recipe": REQUIRED_RECIPE,
                        "recipe_options": {"llamacpp_args": REQUIRED_LLAMACPP_ARGS},
                    }
                ]
            }
            return FakeResponse({"status": "ok"})

        if method == "POST" and path == "/v1/embeddings":
            inputs = (json or {}).get("input") or []
            return FakeResponse(
                {"data": [{"embedding": (np.arange(HIDDEN_SIZE, dtype=np.float64) * 0.001).tolist()} for _ in inputs]}
            )

        return FakeResponse({"error": f"unexpected {method} {path}"}, status_code=404)


def _configured_models_payload(model_name: str = "user.test-model") -> dict:
    return {
        "data": [
            {
                "id": model_name,
                "recipe": REQUIRED_RECIPE,
                "recipe_options": {"llamacpp_args": REQUIRED_LLAMACPP_ARGS},
            }
        ]
    }


def test_normalize_lemonade_base_url_strips_v1():
    assert normalize_lemonade_base_url("http://localhost:13305/v1/") == "http://localhost:13305"
    assert ensure_user_model_prefix("foo") == "user.foo"


def test_lemonade_embed_uses_token_ids(npy_model_dir: Path, monkeypatch):
    fake = FakeRequests(models_payload=_configured_models_payload())
    monkeypatch.setattr(lemonade_mod.requests, "request", fake.request)
    clf = LemonadeEmbeddingClassifier(
        lemonade_model_name="user.test-model",
        checkpoint="org/repo:model.gguf",
        lemonade_base_url="http://localhost:13305",
        model=npy_model_dir,
    )
    preds = clf.predict(["hello", "world"])
    assert len(preds) == 2
    embed_calls = [c for c in fake.calls if urlparse(c["url"]).path == "/v1/embeddings"]
    assert len(embed_calls) == 1
    assert len(embed_calls[0]["json"]["input"]) == 2
    assert embed_calls[0]["json"]["model"] == "user.test-model"
    assert not any(urlparse(c["url"]).path == "/api/v1/pull" for c in fake.calls)


def test_lemonade_pulls_and_loads_when_model_missing(npy_model_dir: Path, monkeypatch):
    fake = FakeRequests(models_payload={"data": []})
    monkeypatch.setattr(lemonade_mod.requests, "request", fake.request)
    LemonadeEmbeddingClassifier(
        lemonade_model_name="user.test-model",
        checkpoint="org/repo:bert-base-only.gguf",
        lemonade_base_url="http://localhost:13305/v1",
        model=npy_model_dir,
    )
    pull_calls = [c for c in fake.calls if urlparse(c["url"]).path == "/api/v1/pull"]
    load_calls = [c for c in fake.calls if urlparse(c["url"]).path == "/api/v1/load"]
    assert len(pull_calls) == 1
    assert len(load_calls) == 1
    assert pull_calls[0]["json"]["checkpoint"] == "org/repo:bert-base-only.gguf"
    assert pull_calls[0]["json"]["embedding"] is True
    assert load_calls[0]["json"]["llamacpp_args"] == REQUIRED_LLAMACPP_ARGS
    assert load_calls[0]["json"]["save_options"] is True


def test_lemonade_reloads_when_options_wrong(npy_model_dir: Path, monkeypatch):
    fake = FakeRequests(
        models_payload={
            "data": [
                {
                    "id": "user.test-model",
                    "recipe": REQUIRED_RECIPE,
                    "recipe_options": {"llamacpp_args": "--pooling mean"},
                }
            ]
        }
    )
    monkeypatch.setattr(lemonade_mod.requests, "request", fake.request)
    LemonadeEmbeddingClassifier(
        lemonade_model_name="user.test-model",
        checkpoint="org/repo:model.gguf",
        lemonade_base_url="http://localhost:13305",
        model=npy_model_dir,
    )
    assert any(urlparse(c["url"]).path == "/api/v1/load" for c in fake.calls)


def test_lemonade_list_models_error(npy_model_dir: Path, monkeypatch):
    fake = FakeRequests(models_payload={"error": "boom", "data": []})
    monkeypatch.setattr(lemonade_mod.requests, "request", fake.request)
    with pytest.raises(LemonadeModelError, match="listing"):
        LemonadeEmbeddingClassifier(
            lemonade_model_name="user.test-model",
            checkpoint="org/repo:model.gguf",
            lemonade_base_url="http://localhost:13305",
            model=npy_model_dir,
        )


def test_lemonade_adds_user_prefix(npy_model_dir: Path, monkeypatch):
    fake = FakeRequests(models_payload={"data": []})
    monkeypatch.setattr(lemonade_mod.requests, "request", fake.request)
    clf = LemonadeEmbeddingClassifier(
        lemonade_model_name="test-model",
        checkpoint="org/repo:model.gguf",
        lemonade_base_url="http://localhost:13305",
        model=npy_model_dir,
    )
    assert clf.lemonade_model_name == "user.test-model"
    load_calls = [c for c in fake.calls if urlparse(c["url"]).path == "/api/v1/load"]
    assert load_calls[0]["json"]["model_name"] == "user.test-model"
