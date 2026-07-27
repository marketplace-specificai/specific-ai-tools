# Copyright(C) 2026 Specific AI Inc. All rights reserved.
# SPDX-License-Identifier: MIT
"""Lemonade Server HTTP integration for CLS embeddings."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Optional
from urllib.parse import urlparse

import numpy as np
import requests

from specific_ai_tools.embedding_heads.classifier import EmbeddingClassifier

REQUIRED_LLAMACPP_ARGS = '--embd-normalize "-1" --pooling cls'
REQUIRED_RECIPE = "llamacpp"
DEFAULT_LEMONADE_BASE_URL = "http://localhost:13305"


class LemonadeModelError(RuntimeError):
    """Raised when Lemonade model discovery or load fails."""


def model_card_from_checkpoint(checkpoint: str) -> str:
    """Derive the Hugging Face model card from a Lemonade checkpoint.

    Expected format: ``org/repo:filename.gguf``.
    """
    if ":" not in checkpoint:
        raise ValueError(f"checkpoint must be of the form 'org/repo:filename.gguf', got {checkpoint!r}")
    return checkpoint.split(":", 1)[0]


def _normalize_model_id(model_id: str) -> str:
    return model_id.removeprefix("user.")


def ensure_user_model_prefix(model_id: str) -> str:
    """Return a Lemonade model id with the ``user.`` prefix."""
    if model_id.startswith("user."):
        return model_id
    return f"user.{model_id}"


def normalize_lemonade_base_url(base_url: str) -> str:
    """Normalize to Lemonade server root (no trailing slash, no ``/v1`` suffix)."""
    url = base_url.strip().rstrip("/")
    if url.endswith("/v1"):
        url = url[:-3].rstrip("/")
    parsed = urlparse(url)
    if not parsed.scheme or not parsed.netloc:
        raise ValueError(f"Invalid lemonade_base_url: {base_url!r}")
    return url


class LemonadeEmbeddingClassifier(EmbeddingClassifier):
    """:class:`EmbeddingClassifier` backed by Lemonade Server's HTTP API.

    On construction, ensures the GGUF encoder is available in Lemonade with
    CLS pooling and raw (unnormalized) embeddings
    (``--embd-normalize "-1" --pooling cls``), pulling and loading it when
    missing or misconfigured. Head / tokenizer artifacts are resolved from the
    Hugging Face model card embedded in ``checkpoint`` (the part before ``:``).

    Parameters
    ----------
    lemonade_model_name:
        Lemonade model id (e.g. ``"user.email-agent-triage"`` or
        ``"email-agent-triage"``). The ``user.`` prefix is added when
        missing before load/embed.
    checkpoint:
        Lemonade checkpoint ``org/repo:file.gguf``. The ``org/repo`` segment is
        also used as the Hugging Face model card for head weights / tokenizer.
    lemonade_base_url:
        Lemonade Server root URL (e.g. ``"http://localhost:13305"``).
        A trailing ``/v1`` is stripped if present.
    model:
        Optional override for the head/tokenizer source (local path or HF id).
        Defaults to the model card derived from ``checkpoint``.
    timeout:
        Optional HTTP timeout in seconds for Lemonade requests. When ``None``
        (default), requests wait indefinitely.
    """

    def __init__(
        self,
        lemonade_model_name: str,
        checkpoint: str,
        lemonade_base_url: str = DEFAULT_LEMONADE_BASE_URL,
        *,
        model: str | Path | None = None,
        timeout: float | None = None,
        max_length: Optional[int] = None,
        calibration_temperature: Optional[float] = None,
        selected_thresholds: Optional[dict[str, float]] = None,
        rejection_label_name: Optional[str] = None,
        confidence_rejection_enabled: Optional[bool] = None,
        **kwargs: Any,
    ):
        if not lemonade_model_name:
            raise ValueError("lemonade_model_name is required")
        if not checkpoint:
            raise ValueError("checkpoint is required")

        self.lemonade_model_name = ensure_user_model_prefix(lemonade_model_name)
        self.checkpoint = checkpoint
        self.lemonade_base_url = normalize_lemonade_base_url(lemonade_base_url)
        self.timeout = timeout

        self._ensure_model_loaded()

        head_model = model if model is not None else model_card_from_checkpoint(checkpoint)
        super().__init__(
            head_model,
            max_length=max_length,
            calibration_temperature=calibration_temperature,
            selected_thresholds=selected_thresholds,
            rejection_label_name=rejection_label_name,
            confidence_rejection_enabled=confidence_rejection_enabled,
            **kwargs,
        )

    def _url(self, path: str) -> str:
        if not path.startswith("/"):
            path = f"/{path}"
        return f"{self.lemonade_base_url}{path}"

    def _request_json(
        self,
        method: str,
        path: str,
        *,
        json_body: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        try:
            request_kwargs: dict[str, Any] = {"json": json_body}
            if self.timeout is not None:
                request_kwargs["timeout"] = self.timeout
            response = requests.request(
                method,
                self._url(path),
                **request_kwargs,
            )
        except requests.RequestException as exc:
            raise LemonadeModelError(f"Lemonade request failed ({method} {path}): {exc}") from exc

        try:
            payload = response.json()
        except ValueError as exc:
            raise LemonadeModelError(
                f"Lemonade returned non-JSON response ({method} {path}): "
                f"status={response.status_code} body={response.text[:500]!r}"
            ) from exc

        if not response.ok:
            detail = payload.get("error", payload) if isinstance(payload, dict) else payload
            raise LemonadeModelError(
                f"Lemonade error ({method} {path}): status={response.status_code} detail={detail!r}"
            )
        if not isinstance(payload, dict):
            raise LemonadeModelError(f"Unexpected Lemonade response type ({method} {path}): {type(payload)!r}")
        return payload

    def _find_registered_model(self, models_payload: dict[str, Any]) -> dict[str, Any] | None:
        target = _normalize_model_id(self.lemonade_model_name)
        for entry in models_payload.get("data") or []:
            if not isinstance(entry, dict):
                continue
            entry_id = entry.get("id")
            if isinstance(entry_id, str) and _normalize_model_id(entry_id) == target:
                return entry
        return None

    def _has_required_options(self, model_entry: dict[str, Any]) -> bool:
        if model_entry.get("recipe") != REQUIRED_RECIPE:
            return False
        recipe_options = model_entry.get("recipe_options")
        if not isinstance(recipe_options, dict):
            return False
        return recipe_options.get("llamacpp_args") == REQUIRED_LLAMACPP_ARGS

    def _ensure_model_loaded(self) -> None:
        """Pull/load the Lemonade GGUF model when missing or misconfigured."""
        models_payload = self._request_json("GET", "/v1/models")
        if "error" in models_payload:
            raise LemonadeModelError(f"Error listing Lemonade models: {models_payload.get('error')}")
        if "data" not in models_payload:
            raise LemonadeModelError(f"Lemonade /v1/models response missing 'data': {models_payload!r}")

        registered = self._find_registered_model(models_payload)
        if registered is not None and self._has_required_options(registered):
            return

        # Register / download the embedding checkpoint, then load with CLS pooling.
        self._request_json(
            "POST",
            "/api/v1/pull",
            json_body={
                "model_name": self.lemonade_model_name,
                "checkpoint": self.checkpoint,
                "recipe": REQUIRED_RECIPE,
                "embedding": True,
            },
        )
        self._request_json(
            "POST",
            "/api/v1/load",
            json_body={
                "model_name": self.lemonade_model_name,
                "llamacpp_args": REQUIRED_LLAMACPP_ARGS,
                "save_options": True,
            },
        )

    def get_embeddings(
        self,
        input_ids_batch: list[list[int]],
        texts: list[str],
    ) -> np.ndarray:
        """Fetch CLS embeddings via Lemonade ``POST /v1/embeddings``.

        Uses ``input_ids_batch`` (``texts`` is unused).
        """
        payload = {
            "model": self.lemonade_model_name,
            # OpenAI-compatible: array of token-id arrays (HF tokenizer output).
            "input": input_ids_batch,
        }
        response = self._request_json("POST", "/v1/embeddings", json_body=payload)
        data = response.get("data")
        if not isinstance(data, list) or len(data) != len(input_ids_batch):
            raise LemonadeModelError(
                f"Unexpected embeddings response: expected {len(input_ids_batch)} vectors, got {data!r}"
            )
        try:
            vectors = [item["embedding"] for item in data]
        except (KeyError, TypeError) as exc:
            raise LemonadeModelError(f"Malformed embeddings payload: {data!r}") from exc
        return np.asarray(vectors, dtype=np.float64)
